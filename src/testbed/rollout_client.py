from __future__ import annotations
import json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor


class RolloutClient:
    """Fans generation requests out across several rollout servers (one per GPU)."""

    def __init__(self, ports: list[int], timeout: float = 7200):
        self.urls = [f"http://127.0.0.1:{p}" for p in ports]; self.timeout = timeout

    def wait_ready(self, max_wait: float = 1800):
        t0 = time.time()
        for u in self.urls:
            while True:
                try:
                    urllib.request.urlopen(u + "/health", timeout=5).read(); break
                except Exception:
                    if time.time() - t0 > max_wait:
                        raise RuntimeError(f"rollout server {u} not ready")
                    time.sleep(5)

    def _post(self, url, payload):
        req = urllib.request.Request(url + "/generate", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=self.timeout).read())["results"]

    def generate(self, prompt_token_ids: list[list[int]] | None = None, prompts: list[str] | None = None, **kw) -> list[dict]:
        items = prompt_token_ids if prompt_token_ids is not None else prompts
        key = "prompt_token_ids" if prompt_token_ids is not None else "prompts"
        shards = [items[i::len(self.urls)] for i in range(len(self.urls))]
        with ThreadPoolExecutor(len(self.urls)) as ex:
            outs = list(ex.map(lambda us: self._post(us[0], {key: us[1], **kw}) if us[1] else [], zip(self.urls, shards)))
        # re-interleave
        res = [None] * len(items)
        for si, o in enumerate(outs):
            for j, r in enumerate(o):
                res[si + j * len(self.urls)] = r
        return res
