"""Client for the language models, with an on-disk cache and append-only JSONL records.

Two providers:
  vllm       open-weight models served by vLLM (OpenAI-compatible API) on a GPU cluster
  anthropic  Claude models through the official Anthropic SDK (key in .env as ANTHROPIC_API_KEY)

The cache key includes the model, its revision and every decoding setting, so a reply is
reused only for an identical request. Each call returns a record dict; experiment scripts add
task fields and the oracle score before writing it.

Claude calls are paid. Every new call appends its cost to data/probes/cache/<model>/_spend.jsonl, and
the client refuses new calls once the model's spend cap is reached (registry `spend_cap_usd`, or the
SPEND_CAP_USD environment variable). Cached replies cost nothing. Run one Claude job per model at a time:
the running total is read when the client starts. No refusal fallback is used: a fallback would answer
with a different model, so a refusal is recorded as an invalid reply instead.
"""
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data/probes/cache"
REGISTRY = ROOT / "configs/model_registry.json"
SCHEMA_VERSION = 1


class SpendCapReached(RuntimeError):
    pass


def load_registry():
    return json.loads(REGISTRY.read_text())


class Client:
    def __init__(self, model_key, base_url=None, condition="T0", max_tokens=None):
        reg = load_registry()[model_key]
        self.key = model_key
        self.provider = reg.get("provider", "vllm")
        self.model = reg["hf_id"]
        self.revision = reg["revision"]
        self.decoding = dict(reg["decoding"])
        if max_tokens:  # a different reply cap is a different setting: part of the cache key and every record
            self.decoding["max_tokens"] = int(max_tokens)
        thinking = condition == "T4"
        self.extra = dict(reg.get("thinking_on" if thinking else "thinking_off", {}))
        self.base_url = base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        self._client = None
        self.price = reg.get("price_per_mtok")
        self.cap = float(os.environ.get("SPEND_CAP_USD") or reg.get("spend_cap_usd") or 0)
        self._lock = threading.Lock()
        self.ledger = CACHE_DIR / self.key / "_spend.jsonl"
        self.spent = self._ledger_total() if self.provider == "anthropic" else 0.0

    def _ledger_total(self):
        if not self.ledger.exists():
            return 0.0
        total = 0.0
        for line in self.ledger.read_text().splitlines():
            try:
                total += json.loads(line)["usd"]
            except (json.JSONDecodeError, KeyError):
                pass
        return total

    def _settings(self):
        return {"model": self.model, "revision": self.revision, **self.decoding, "extra": self.extra}

    def _cache_path(self, prompt, sample):
        key = hashlib.sha256(json.dumps([self._settings(), prompt, sample], sort_keys=True).encode()).hexdigest()
        return CACHE_DIR / self.key / f"{key}.json"

    def complete(self, prompt, sample=0):
        path = self._cache_path(prompt, sample)
        if path.exists():
            try:
                rec = json.loads(path.read_text())
                rec["cached"] = True
                return rec
            except json.JSONDecodeError:
                pass  # partial file from an interrupted job: recompute
        t0 = time.time()
        if self.provider == "anthropic":
            reply, reasoning, usage = self._call_anthropic(prompt)
        else:
            reply, reasoning, usage = self._call(prompt)
        rec = {
            "model_key": self.key, "model": self.model, "revision": self.revision,
            "decoding": self.decoding, "extra": self.extra, "sample": sample,
            "prompt_sha": hashlib.sha256(prompt.encode()).hexdigest()[:16],
            "reply": reply, "reasoning": reasoning, "usage": usage,
            "duration_s": round(time.time() - t0, 2), "hardware": os.environ.get("GPU_NAME"),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "schema_version": SCHEMA_VERSION,
        }
        if self.provider == "anthropic":
            rec["provider"] = "anthropic"
            rec["cost_usd"] = usage["cost_usd"]
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps({**rec, "prompt": prompt}))
        os.replace(tmp, path)  # atomic: concurrent jobs never read a half-written reply
        rec["cached"] = False
        return rec

    def batch(self, jobs, workers=32):
        """jobs: list of (prompt, sample). Returns records in the same order."""
        with ThreadPoolExecutor(workers) as ex:
            return list(ex.map(lambda j: self.complete(*j), jobs))

    def _call_anthropic(self, prompt):
        with self._lock:
            if self.cap and self.spent >= self.cap:
                raise SpendCapReached(f"{self.key}: spent ${self.spent:.2f} of the ${self.cap:.2f} cap")
        if self._client is None:
            import anthropic
            from dotenv import load_dotenv

            load_dotenv(ROOT / ".env")  # ANTHROPIC_API_KEY; never logged or written to records
            # The SDK retries rate limits, overload and connection errors with backoff.
            self._client = anthropic.Anthropic(max_retries=8, timeout=600)
        resp = self._client.messages.create(
            model=self.model, messages=[{"role": "user", "content": prompt}], **self.decoding, **self.extra,
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        reasoning = "".join(b.thinking for b in resp.content if b.type == "thinking") or None
        n_in, n_out = resp.usage.input_tokens, resp.usage.output_tokens  # output includes thinking tokens
        cost = (n_in * self.price["input"] + n_out * self.price["output"]) / 1e6
        with self._lock:
            self.spent += cost
            self.ledger.parent.mkdir(parents=True, exist_ok=True)
            with open(self.ledger, "a") as f:
                f.write(json.dumps({"usd": round(cost, 6), "in": n_in, "out": n_out,
                                    "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}) + "\n")
        usage = {"prompt": n_in, "completion": n_out, "stop_reason": resp.stop_reason,
                 "served_model": resp.model, "cost_usd": round(cost, 6)}
        return text, reasoning, usage

    def _call(self, prompt, retries=5):
        if self._client is None:
            from openai import OpenAI

            # A client timeout aborts the request on the server, so a long reply on a busy server must not time out:
            # the timeout is generous and this loop is the only retry.
            self._client = OpenAI(base_url=self.base_url, api_key="EMPTY", timeout=4 * 3600, max_retries=0)
        for attempt in range(retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model, messages=[{"role": "user", "content": prompt}],
                    extra_body=self.extra or None, **self.decoding,
                )
                msg = resp.choices[0].message
                reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
                usage = {"prompt": resp.usage.prompt_tokens, "completion": resp.usage.completion_tokens}
                return msg.content or "", reasoning, usage
            except Exception as e:  # server busy, timeouts
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
                print(f"retry {attempt + 1} for {self.key}: {e}", flush=True)
