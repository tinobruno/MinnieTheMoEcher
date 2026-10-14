"""
MinnieTheMoEcher - Zero Decode Bypass (ZDB) Python Client & SDK

Provides sub-100ms non-autoregressive System 1 classification,
decision making, and agent tool routing with zero decode passes.
"""

from __future__ import annotations
import time
import json
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union


@dataclass
class DecisionResult:
    """Represents a Zero Decode Bypass decision response."""
    decision: str
    confidence: float
    probabilities: Dict[str, float]
    decode_passes: int
    winning_index: int
    prefill_ms: float
    total_ms: float
    global_top1_token: Optional[str] = None
    global_top1_logit: Optional[float] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"<DecisionResult: winner='{self.decision}' "
                f"confidence={self.confidence*100:.2f}% "
                f"decode_passes={self.decode_passes} "
                f"latency={self.total_ms:.1f}ms>")


class MoecherClient:
    """Python Client for MinnieTheMoEcher high-speed inference engine."""

    def __init__(self, base_url: str = "http://127.0.0.1:8001", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def decide(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        choices: List[str],
        temperature: float = 1.0,
        bypass_prefix_cache: bool = False
    ) -> DecisionResult:
        """
        Executes a Zero Decode Bypass (ZDB) non-autoregressive decision.
        Intercepts logits at step-zero prefill boundary with 0 decode passes.

        Args:
            prompt: Text prompt string or list of OpenAI-style chat messages.
            choices: List of candidate options (e.g. ["APPROVED", "REJECTED"]).
            temperature: Softmax calibration temperature (default: 1.0).
            bypass_prefix_cache: If True, forces fresh prefill without reusing KV cache.

        Returns:
            DecisionResult containing the chosen label, confidence, and timings.
        """
        url = f"{self.base_url}/v1/decide"
        payload: Dict[str, Any] = {
            "choices": choices,
            "temperature": temperature,
            "format": "json",
            "bypass_prefix_cache": bypass_prefix_cache
        }

        if isinstance(prompt, list):
            payload["messages"] = prompt
        else:
            payload["prompt"] = prompt

        req_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=req_bytes,
            headers={"Content-Type": "application/json"}
        )

        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"ZDB Error ({e.code}): {err_body}") from e
        except urllib.error.URLError as e:
            raise ConnectionError(f"Could not connect to MinnieTheMoEcher at {self.base_url}: {e.reason}") from e

        wall_latency_ms = (time.perf_counter() - t0) * 1000.0

        timing = data.get("timing", {})
        prefill_ms = timing.get("prefill_ms", wall_latency_ms)
        total_ms = timing.get("total_ms", wall_latency_ms)

        global_top1 = data.get("global_top1", {})

        return DecisionResult(
            decision=data.get("decision", ""),
            confidence=float(data.get("confidence", 0.0)),
            probabilities={k: float(v) for k, v in data.get("probabilities", {}).items()},
            decode_passes=int(data.get("decode_passes", 0)),
            winning_index=int(data.get("winning_index", -1)),
            prefill_ms=float(prefill_ms),
            total_ms=float(total_ms),
            global_top1_token=global_top1.get("token"),
            global_top1_logit=global_top1.get("logit"),
            raw_response=data
        )

    def classify(
        self,
        text: str,
        labels: List[str],
        instruction: str = "Classify the following text into one of the candidate labels:"
    ) -> DecisionResult:
        """
        Convenience wrapper for zero-shot text classification.
        """
        prompt = f"{instruction}\nText: \"{text}\"\nLabel:"
        return self.decide(prompt=prompt, choices=labels)

    def route_tool(
        self,
        user_query: str,
        tools_dict: Dict[str, str]
    ) -> DecisionResult:
        """
        Instant sub-100ms agentic tool router with 0 decode passes.

        Args:
            user_query: The incoming user prompt to route.
            tools_dict: Map of tool_name -> tool_description.
        """
        tool_desc_lines = [f"- {name}: {desc}" for name, desc in tools_dict.items()]
        tools_block = "\n".join(tool_desc_lines)
        choices = list(tools_dict.keys())

        prompt = (
            f"You are an intelligent agent dispatcher. Available tools:\n"
            f"{tools_block}\n\n"
            f"User Query: \"{user_query}\"\n"
            f"Selected Tool:"
        )
        return self.decide(prompt=prompt, choices=choices)


if __name__ == "__main__":
    print("=" * 65)
    print(" MinnieTheMoEcher Zero Decode Bypass (ZDB) Python SDK Demo")
    print("=" * 65)

    client = MoecherClient()

    # 1. Underwriting Decision Demo
    print("\n1. Running Underwriting Decision...")
    loan_prompt = (
        "Evaluate applicant Jane Doe: Credit Score 790, Income $145,000, "
        "Debt-to-Income 18%, Late Payments 0, Requested $15,000 for Home Improvement.\n"
        "Decision:"
    )
    res = client.decide(loan_prompt, ["APPROVED", "REJECTED"])
    print(f"   Result: {res.decision} (Confidence: {res.confidence*100:.2f}%)")
    print(f"   Decode Passes: {res.decode_passes} | Total Hardware Latency: {res.total_ms:.2f} ms")

    # 2. Text Sentiment Classification Demo
    print("\n2. Running Sentiment Classification...")
    review = "The inference speed with this engine is unbelievable, sub-30ms latency on a single GPU!"
    res_sent = client.classify(review, ["POSITIVE", "NEGATIVE", "NEUTRAL"])
    print(f"   Text: '{review}'")
    print(f"   Sentiment: {res_sent.decision} ({res_sent.confidence*100:.2f}%) in {res_sent.total_ms:.2f} ms")

    # 3. Agent Tool Dispatcher Demo
    print("\n3. Running Agent Tool Dispatcher...")
    tools = {
        "web_search": "search the live internet for stock quotes, current events, news",
        "read_file": "read local filesystem files, documents, logs",
        "fetch_url": "download and scrape content from a web URL",
        "none": "no tool needed, answer using general internal knowledge"
    }
    query = "What is the current stock price of NVDA today?"
    res_tool = client.route_tool(query, tools)
    print(f"   Query: '{query}'")
    print(f"   Dispatched Tool: {res_tool.decision} ({res_tool.confidence*100:.2f}%) in {res_tool.total_ms:.2f} ms")
