"""Token-tracking LLM client wrapper."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_anthropic import ChatAnthropic

from pipeline.models import LLMRequest, LLMResponse


class LLMClient:
    """Thin wrapper around ChatAnthropic that tracks token usage."""

    def __init__(
        self, model: str = "claude-3-5-haiku-20241022", max_workers: int = 5
    ) -> None:
        self.model = model
        self.max_workers = max_workers
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.request_count: int = 0
        self._lock = threading.Lock()

    def _send_one(self, index: int, request: LLMRequest) -> tuple[int, LLMResponse]:
        """Send a single LLM request. Returns (index, response) for ordering.

        Args:
            index: The position of this request in the input list.
            request: The LLM request containing prompt and response model.

        Returns:
            A (index, LLMResponse) tuple.
        """
        try:
            llm = ChatAnthropic(model=self.model)
            structured_llm = llm.with_structured_output(
                request.response_model, include_raw=True
            )
            result = structured_llm.invoke(request.prompt)

            usage = result["raw"].usage_metadata or {}
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)

            with self._lock:
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
                self.request_count += 1

            response = LLMResponse(
                request=request,
                parsed=result["parsed"],
                raw_text=str(result["raw"].content) if hasattr(result["raw"], "content") else "",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        except Exception as exc:
            with self._lock:
                self.request_count += 1
            response = LLMResponse(
                request=request,
                error=str(exc),
            )
        return (index, response)

    def send(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        """Send multiple LLM requests concurrently, preserving input order.

        Args:
            requests: The LLM requests to send.

        Returns:
            A list of LLMResponses in the same order as the requests.
        """
        results: dict[int, LLMResponse] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._send_one, i, req): i
                for i, req in enumerate(requests)
            }
            for future in as_completed(futures):
                idx, response = future.result()
                results[idx] = response

        assert len(results) == len(requests), (
            f"Expected {len(requests)} responses, got {len(results)}"
        )
        return [results[i] for i in range(len(requests))]

    def __repr__(self) -> str:
        total_tokens = self.total_input_tokens + self.total_output_tokens
        return (
            "LLMClient(\n"
            f"\tmodel={self.model!r},\n"
            f"\tmax_workers={self.max_workers},\n"
            f"\trequests={self.request_count},\n"
            f"\tinput_tokens={self.total_input_tokens},\n"
            f"\toutput_tokens={self.total_output_tokens},\n"
            f"\ttotal_tokens={total_tokens}\n"
            ")"
        )