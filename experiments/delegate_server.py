"""Runtime Integration Phase 2A -- Delegate server (Process B side).

    python experiments/delegate_server.py --model gpt-4o-mini-2024-07-18 --port 8765

Wraps a real `dualflow.delegate_agent.DelegateAgent` behind an HTTP server so
`AgentDelegationRuntime`/`ClarifyingDelegate` (Process A side, unmodified)
can drive it via `dualflow.remote_delegate.RemoteDelegateAgent` — same
machine first (127.0.0.1), then a second physical machine (swap the host).

Neither `AgentDelegationRuntime` nor any B7 module
(`delegate_agent.py`/`clarification.py`/`experience_decision.py`/etc.) is
modified by this file — it only wraps `DelegateAgent`'s three existing
public methods (`propose`/`sample_candidates`/`ask_clarification`) behind
three POST endpoints, using the exact same wire-format (de)serialization
helpers `remote_delegate.py`'s `RemoteDelegateAgent` client uses (imported,
not duplicated).

Endpoints: POST /propose, POST /sample-candidates, POST /ask-clarification.
Standard library only (`http.server`) — no new dependency.

이 스크립트가 하지 않는 것:
  - Principal/Semantic/Authority 중 무엇도 실행하지 않는다 -- 순수하게
    Agent B(Delegate) 하나만 원격으로 노출한다. B가 자기 자신의 delegation
    을 검증하게 만드는 것은 이 프로젝트가 지금까지 지켜온 독립 검증
    구조를 무너뜨리므로, Verifier는 항상 Process A(호출자) 쪽에 남는다.
  - 인증/TLS/재시도 -- 로컬 smoke test(Phase 2A)와 두 프로세스 간 real-API
    단발 실행(Phase 2B)이 목표이고, production hardening은 범위 밖이다.
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXPERIMENTS_DIR))
from agent_smoke import build_llm_client  # noqa: E402

from dualflow.delegate_agent import CandidateDistribution, DelegateAgent  # noqa: E402
from dualflow.remote_delegate import (  # noqa: E402
    belief_from_list, distribution_to_dict, llm_response_to_dict, proposal_to_dict,
)
from dualflow.semantic import entropy as compute_entropy  # noqa: E402


def make_handler(delegate: DelegateAgent) -> type[BaseHTTPRequestHandler]:
    """`delegate`(실제 `DelegateAgent`, real 또는 fake `LLMClient` 무엇이든)
    를 감싸는 handler 클래스를 만든다 -- 매 요청마다 새로 만들지 않고
    같은 `delegate` 인스턴스를 재사용한다(매번 새 `DelegateAgent`를 만들
    이유가 없다, 상태 없는 wrapper일 뿐이다)."""

    class DelegateRequestHandler(BaseHTTPRequestHandler):
        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            return json.loads(raw)

        def _write_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802 -- BaseHTTPRequestHandler's own naming convention
            try:
                payload = self._read_json()
                if self.path == "/propose":
                    proposal = delegate.propose(
                        delegation=payload["delegation"], context=payload.get("context", ""))
                    self._write_json(200, proposal_to_dict(proposal))
                elif self.path == "/sample-candidates":
                    distribution = delegate.sample_candidates(
                        delegation=payload["delegation"], context=payload.get("context", ""),
                        n=payload.get("n", 10))
                    self._write_json(200, distribution_to_dict(distribution))
                elif self.path == "/ask-clarification":
                    belief = belief_from_list(payload["belief"])
                    # ask_clarification()은 distribution.belief만 읽는다
                    # (delegate_agent.py 참고) -- entropy/top/n_samples는
                    # 이 요청 안에서 다시 만들 필요가 있는 placeholder일
                    # 뿐이다(응답에는 실리지 않는다 -- 클라이언트가 자기
                    # 쪽 원본 distribution에서 그대로 채운다, remote_
                    # delegate.py의 ask_clarification() 참고).
                    top = max(belief, key=lambda i: belief[i])
                    placeholder_distribution = CandidateDistribution(
                        belief=belief, entropy=compute_entropy(belief), top=top,
                        top_probability=belief[top], n_unique=len(belief),
                        n_samples=len(belief), responses=[])
                    question = delegate.ask_clarification(
                        delegation=payload["delegation"], distribution=placeholder_distribution,
                        context=payload.get("context", ""),
                        target_facets=frozenset(payload.get("target_facets", ())))
                    self._write_json(200, {
                        "question": question.question, "raw_text": question.raw_text,
                        "response": llm_response_to_dict(question.response),
                    })
                else:
                    self._write_json(404, {"error": f"unknown path: {self.path}"})
            except Exception as e:  # noqa: BLE001 -- surface failures as JSON 500s, never a bare crash
                self._write_json(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, format_: str, *args) -> None:  # noqa: A002
            sys.stderr.write(f"[delegate_server] {self.address_string()} - {format_ % args}\n")

    return DelegateRequestHandler


def main(argv: list[str] | None = None) -> int:
    try:  # Windows 기본 콘솔(cp949 등)의 UnicodeEncodeError 방지
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    p.add_argument("--host", default="127.0.0.1",
                   help="0.0.0.0 to accept connections from another machine (Phase 2B/2C)")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args(argv)

    llm_client = build_llm_client(args.model)
    delegate = DelegateAgent(llm=llm_client)

    server = ThreadingHTTPServer((args.host, args.port), make_handler(delegate))
    print(f"Delegate server listening on http://{args.host}:{args.port} (model={args.model})")
    print("Endpoints: POST /propose, POST /sample-candidates, POST /ask-clarification")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
