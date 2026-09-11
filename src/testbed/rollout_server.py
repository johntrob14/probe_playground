"""Minimal HTTP generation server around a vLLM engine, one per GPU.

Used for (a) policy rollouts with a hot-swapped LoRA adapter and (b) the LLM judge in arm B.
POST /generate  {"prompt_token_ids": [[...], ...] | "prompts": [...], "n", "max_tokens", "temperature",
                 "top_p", "seed", "lora_path", "lora_id"}
  -> {"results": [{"prompt_token_ids": [...], "outputs": [{"token_ids": [...], "text": str, "finish_reason": str}]}]}
GET /health -> "ok"
"""
from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--tp", type=int, default=1); ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=8192); ap.add_argument("--enable-lora", action="store_true")
    ap.add_argument("--max-lora-rank", type=int, default=64)
    a = ap.parse_args()
    from testbed.detectors.llm_monitor import _compat_shims
    _compat_shims()
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem,
              max_model_len=a.max_model_len, enable_lora=a.enable_lora, max_lora_rank=a.max_lora_rank,
              max_loras=2, enable_prefix_caching=True)
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok")

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0)); req = json.loads(self.rfile.read(n))
            sp = SamplingParams(n=req.get("n", 1), temperature=req.get("temperature", 1.0), top_p=req.get("top_p", 1.0),
                                max_tokens=req.get("max_tokens", 4096), seed=req.get("seed"))
            lora = LoRARequest("policy", int(req["lora_id"]), req["lora_path"]) if req.get("lora_path") else None
            if "prompt_token_ids" in req:
                prompts = [TokensPrompt(prompt_token_ids=p) for p in req["prompt_token_ids"]]
            else:
                prompts = req["prompts"]
            with lock:
                outs = llm.generate(prompts, sp, lora_request=lora, use_tqdm=False)
            res = [{"prompt_token_ids": list(o.prompt_token_ids),
                    "outputs": [{"token_ids": list(c.token_ids), "text": c.text, "finish_reason": c.finish_reason} for c in o.outputs]}
                   for o in outs]
            body = json.dumps({"results": res}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)

    print(f"rollout server ready on :{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
