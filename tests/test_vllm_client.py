from llm_serving_lab.client.vllm_client import (
    GenerationResult,
    TokenEvent,
)


def test_token_event():
    event = TokenEvent(
        token="hello",
        timestamp=1.0,
    )

    assert event.token == "hello"
    assert event.timestamp == 1.0


def test_generation_result():
    result = GenerationResult(
        text="hello world",
        input_tokens=10,
        output_tokens=2,
        total_tokens=12,
        ttft_ms=50.0,
        latency_ms=100.0,
        inter_token_latency_ms=25.0,
        tokens_per_second=20.0,
    )

    assert result.output_tokens == 2
    assert result.total_tokens == 12
    assert result.ttft_ms == 50.0