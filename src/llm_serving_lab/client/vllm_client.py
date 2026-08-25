from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import httpx


@dataclass
class TokenEvent:
    token: str
    timestamp: float


@dataclass
class GenerationResult:
    text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int

    ttft_ms: float
    latency_ms: float

    inter_token_latency_ms: float
    tokens_per_second: float

    token_events: list[TokenEvent] = field(default_factory=list)


class VLLMClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")

        self.client = httpx.AsyncClient(
            timeout=timeout
        )

    async def generate(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> GenerationResult:

        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        start = time.perf_counter()

        token_events: list[TokenEvent] = []
        generated_text: list[str] = []

        input_tokens = 0

        async with self.client.stream(
            "POST",
            f"{self.base_url}/v1/completions",
            json=payload,
        ) as response:

            response.raise_for_status()

            async for line in response.aiter_lines():

                if not line:
                    continue

                if not line.startswith("data:"):
                    continue

                data = line[len("data:"):].strip()

                if data == "[DONE]":
                    break

                chunk = json.loads(data)

                timestamp = time.perf_counter()

                choice = chunk["choices"][0]

                token = choice.get("text", "")

                if token:
                    generated_text.append(token)

                    token_events.append(
                        TokenEvent(
                            token=token,
                            timestamp=timestamp,
                        )
                    )

                usage = chunk.get("usage")

                if usage:
                    input_tokens = usage.get(
                        "prompt_tokens",
                        input_tokens,
                    )

        end = time.perf_counter()

        output_tokens = len(token_events)

        if token_events:

            ttft_ms = (
                token_events[0].timestamp - start
            ) * 1000.0

            if len(token_events) > 1:

                intervals = [
                    token_events[i].timestamp
                    - token_events[i - 1].timestamp
                    for i in range(1, len(token_events))
                ]

                inter_token_latency_ms = (
                    sum(intervals) / len(intervals)
                ) * 1000.0

            else:
                inter_token_latency_ms = 0.0

        else:

            ttft_ms = 0.0
            inter_token_latency_ms = 0.0

        latency_ms = (
            end - start
        ) * 1000.0

        total_tokens = (
            input_tokens + output_tokens
        )

        elapsed_seconds = end - start

        tokens_per_second = (
            output_tokens / elapsed_seconds
            if elapsed_seconds > 0
            else 0.0
        )

        return GenerationResult(
            text="".join(generated_text),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            ttft_ms=ttft_ms,
            latency_ms=latency_ms,
            inter_token_latency_ms=inter_token_latency_ms,
            tokens_per_second=tokens_per_second,
            token_events=token_events,
        )

    async def close(self) -> None:
        await self.client.aclose()