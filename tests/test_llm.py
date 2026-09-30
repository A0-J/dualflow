"""LLMResponse/LLMClient/OpenAILLMClient — Phase B의 공통 model adapter 레이어.

여기서 검증하는 건 딱 이 레이어의 책임(prompt 보내기 -> text + 계측 metadata
받기)까지다. openai 패키지는 설치돼 있지 않아도 된다 — OpenAILLMClient는
client 객체를 duck-typing으로만 쓰므로, 이 테스트는 `openai.OpenAI()`가
아니라 같은 모양(`.responses.create(...)`)을 흉내 낸 순수 Python mock을
주입한다. API 호출은 전혀 일어나지 않는다.
"""

from __future__ import annotations

from types import SimpleNamespace

from dualflow.llm import LLMResponse, OpenAILLMClient


class _FakeInputTokensDetails:
    def __init__(self, cached_tokens: int):
        self.cached_tokens = cached_tokens


class _FakeUsage:
    def __init__(self, input_tokens: int, output_tokens: int, cached_tokens: int | None = None):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        if cached_tokens is not None:
            self.input_tokens_details = _FakeInputTokensDetails(cached_tokens)


class _FakeResponses:
    """`client.responses.create(...)`를 흉내 낸다."""

    def __init__(self, output_text: str, usage: _FakeUsage | None, model: str | None = None):
        self._output_text = output_text
        self._usage = usage
        self._model = model
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._model is not None:
            return SimpleNamespace(output_text=self._output_text, usage=self._usage,
                                   model=self._model)
        return SimpleNamespace(output_text=self._output_text, usage=self._usage)


class _FakeOpenAIClient:
    def __init__(self, output_text: str, usage: _FakeUsage | None = None, model: str | None = None):
        self.responses = _FakeResponses(output_text, usage, model)


class TestLLMResponse:
    def test_defaults_are_none(self):
        r = LLMResponse(text="hi")
        assert r.text == "hi"
        assert r.input_tokens is None
        assert r.output_tokens is None
        assert r.latency_ms is None

    def test_is_frozen(self):
        r = LLMResponse(text="hi")
        try:
            r.text = "changed"  # type: ignore[misc]
        except Exception:
            pass
        else:
            raise AssertionError("LLMResponse should be immutable")


class TestOpenAILLMClient:
    def test_generate_returns_text_and_usage(self):
        fake = _FakeOpenAIClient("the answer", _FakeUsage(input_tokens=12, output_tokens=3))
        llm = OpenAILLMClient(client=fake, model="gpt-4o-mini")

        resp = llm.generate(instructions="be terse", input_text="what is 2+2?")

        assert resp.text == "the answer"
        assert resp.input_tokens == 12
        assert resp.output_tokens == 3
        assert resp.latency_ms is not None and resp.latency_ms >= 0.0

    def test_generate_passes_model_instructions_and_input(self):
        fake = _FakeOpenAIClient("ok")
        llm = OpenAILLMClient(client=fake, model="gpt-4o-mini")

        llm.generate(instructions="system prompt", input_text="user text")

        assert len(fake.responses.calls) == 1
        call = fake.responses.calls[0]
        assert call["model"] == "gpt-4o-mini"
        assert call["instructions"] == "system prompt"
        assert call["input"] == "user text"

    def test_generate_tolerates_missing_usage(self):
        fake = _FakeOpenAIClient("ok", usage=None)
        llm = OpenAILLMClient(client=fake, model="gpt-4o-mini")

        resp = llm.generate(instructions="x", input_text="y")

        assert resp.text == "ok"
        assert resp.input_tokens is None
        assert resp.output_tokens is None

    def test_default_temperature_and_top_p_are_explicit_1_0(self):
        """2026-09-30 추가 — 이전엔 전혀 지정 안 하고 API 기본값에 맡겼는데,
        cross-model entropy 비교에서는 이게 통제해야 할 독립변수라는 지적을
        받아 명시적으로 노출시켰다. 기본값은 기존 API 기본값과 같은 1.0/1.0
        이라 여기서 바꾸지 않으면 이전까지의 모든 real-API 실행과
        byte-identical한 조건이 유지된다."""
        fake = _FakeOpenAIClient("ok")
        llm = OpenAILLMClient(client=fake, model="gpt-4o-mini")

        llm.generate(instructions="x", input_text="y")

        call = fake.responses.calls[0]
        assert call["temperature"] == 1.0
        assert call["top_p"] == 1.0

    def test_custom_temperature_and_top_p_are_passed_through_and_recorded(self):
        fake = _FakeOpenAIClient("ok")
        llm = OpenAILLMClient(client=fake, model="gpt-4o-mini", temperature=0.7, top_p=0.9)

        resp = llm.generate(instructions="x", input_text="y")

        call = fake.responses.calls[0]
        assert call["temperature"] == 0.7
        assert call["top_p"] == 0.9
        assert resp.temperature == 0.7
        assert resp.top_p == 0.9

    def test_served_model_snapshot_is_recorded_when_available(self):
        """요청한 이름(`gpt-4.1`)과 실제 서빙된 snapshot(`gpt-4.1-2025-04-14`)
        이 다를 수 있다 — 그 실제 값을 기록해서 사후 감사 가능하게 한다."""
        fake = _FakeOpenAIClient("ok", model="gpt-4.1-2025-04-14")
        llm = OpenAILLMClient(client=fake, model="gpt-4.1")

        resp = llm.generate(instructions="x", input_text="y")

        assert resp.model == "gpt-4.1-2025-04-14"

    def test_cached_input_tokens_extracted_when_present(self):
        """2026-09-30 추가 -- 실제 caching 적용 여부를 글자 수 추정이 아니라
        API가 보고한 값 그대로 기록해야 한다는 지적에 대응."""
        fake = _FakeOpenAIClient("ok", usage=_FakeUsage(100, 5, cached_tokens=80))
        llm = OpenAILLMClient(client=fake, model="gpt-4o-mini")

        resp = llm.generate(instructions="x", input_text="y")

        assert resp.input_tokens == 100
        assert resp.cached_input_tokens == 80

    def test_cached_input_tokens_defaults_to_none_when_absent(self):
        fake = _FakeOpenAIClient("ok", usage=_FakeUsage(100, 5))  # no cached_tokens
        llm = OpenAILLMClient(client=fake, model="gpt-4o-mini")

        resp = llm.generate(instructions="x", input_text="y")

        assert resp.cached_input_tokens is None

    def test_served_model_defaults_to_none_when_fake_lacks_the_field(self):
        fake = _FakeOpenAIClient("ok")  # model=None -- mimics a minimal fake with no .model
        llm = OpenAILLMClient(client=fake, model="gpt-4o-mini")

        resp = llm.generate(instructions="x", input_text="y")

        assert resp.model is None

    def test_no_openai_import_required(self):
        """OpenAILLMClient는 openai 패키지를 import하지 않는다 — 순수 mock으로도
        동작해야 한다는 걸 위 테스트들이 이미 증명하지만, 여기서는 모듈 자체가
        `openai`를 top-level에서 import하지 않는다는 것도 명시적으로 확인한다."""
        import dualflow.llm as llm_module

        assert "openai" not in vars(llm_module)
