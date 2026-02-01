#!/root/.openclaw/seva/venv/bin/python
"""SEVA local service (keeps embedding model warm).

- Listens on 127.0.0.1 only.
- Provides small JSON HTTP API used by OpenClaw (agent) to avoid cold-start cost.

Endpoints (POST unless noted):
- GET  /status
- GET  /config
- POST /config-set    {"set": ["key=value", ...]}
- POST /recall        {"query": "...", "k": 5}
- POST /remember      {"text":"...","response":"...","channel":"...","sender":"..."}
- POST /score         {"text": "..."}
- POST /verify        {"claim": "..."}

This service does NOT replace the LLM. It's memory + verification + scoring.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from aiohttp import web
import aiohttp
from urllib.parse import quote_plus

SEVA_SRC = Path("/root/seva_openclaw")
if str(SEVA_SRC) not in sys.path:
    sys.path.insert(0, str(SEVA_SRC))

DATA_DIR = Path(os.environ.get("OPENCLAW_SEVA_DATA_DIR", "/root/.openclaw/seva/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = Path(os.environ.get("OPENCLAW_SEVA_CONFIG", "/root/.openclaw/seva/config.json"))

from memory.episodic import EpisodicMemory  # type: ignore
from memory.semantic import SemanticMemory, SentenceTransformersEmbedder  # type: ignore

try:
    from reasoning.truth_engine import TitanMindCore  # type: ignore
except Exception:
    TitanMindCore = None

# Load wikipedia_checker directly to bypass verification/__init__.py importing missing agent modules
try:
    _wc_path = SEVA_SRC / "verification" / "wikipedia_checker.py"
    _spec = importlib.util.spec_from_file_location("seva_wikipedia_checker", _wc_path)
    if _spec is None or _spec.loader is None:
        raise ImportError("could not load wikipedia_checker spec")
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)  # type: ignore
    WikipediaFactChecker = getattr(_mod, "WikipediaFactChecker")
except Exception:
    WikipediaFactChecker = None

DEFAULT_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "intercept": False,
    "memory": {
        "episodic": True,
        "semantic": True,
        "semantic_collection": "openclaw_seva",
        "embed_model": "all-MiniLM-L6-v2",
        "store_all": True,
    },
    "verification": {
        "enabled": True,
        "max_sources": 1,
        "providers": ["wikipedia", "sympy", "wolfram"],
        "wikipedia": True,
        "sympy": {"enabled": False},
        "wolfram": {"enabled": False, "appid": ""},
        "false_confidence_threshold": 0.85
    },
    "scoring": {"enabled": True},
}


def load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(DEFAULT_CONFIG, indent=2, sort_keys=True))
        return dict(DEFAULT_CONFIG)
    try:
        return json.loads(CONFIG_PATH.read_text())
    except Exception:
        CONFIG_PATH.write_text(json.dumps(DEFAULT_CONFIG, indent=2, sort_keys=True))
        return dict(DEFAULT_CONFIG)


def save_config(cfg: Dict[str, Any]) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, sort_keys=True))


def dotted_set(cfg: Dict[str, Any], pair: str) -> None:
    if "=" not in pair:
        raise ValueError(f"Invalid set entry (expected key=value): {pair}")
    key, val = pair.split("=", 1)

    # naive type casting
    if isinstance(val, str) and val.lower() in ("true", "false"):
        cast: Any = val.lower() == "true"
    else:
        try:
            cast = int(val)
        except ValueError:
            try:
                cast = float(val)
            except ValueError:
                cast = val

    cur: Any = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = cast


PRESETS_PATH = Path(os.environ.get("OPENCLAW_SEVA_PRESETS", str(Path(__file__).resolve().parent / "presets.json")))

def load_presets() -> Dict[str, Any]:
    try:
        return json.loads(PRESETS_PATH.read_text())
    except Exception:
        return {"modes": {}}


class SevaService:
    def __init__(self) -> None:
        self.cfg = load_config()
        self._embed_model = self.cfg.get("memory", {}).get("embed_model", "all-MiniLM-L6-v2")
        self.episodic = EpisodicMemory(DATA_DIR / "episodic.db")
        self.semantic = SemanticMemory(
            DATA_DIR / "semantic",
            collection_name=self.cfg.get("memory", {}).get("semantic_collection", "openclaw_seva"),
            embedder=SentenceTransformersEmbedder(model_name=self._embed_model),
        )

    def refresh_cfg(self) -> None:
        new_cfg = load_config()
        new_model = new_cfg.get("memory", {}).get("embed_model", "all-MiniLM-L6-v2")
        if getattr(self, '_embed_model', None) != new_model:
            self._embed_model = new_model
            # rebuild semantic memory with new embedder
            self.semantic = SemanticMemory(
                DATA_DIR / "semantic",
                collection_name=new_cfg.get("memory", {}).get("semantic_collection", "openclaw_seva"),
                embedder=SentenceTransformersEmbedder(model_name=self._embed_model),
            )
        self.cfg = new_cfg

    def status(self) -> Dict[str, Any]:
        self.refresh_cfg()
        return {
            "enabled": self.cfg.get("enabled", True),
            "intercept": self.cfg.get("intercept", False),
            "paths": {"data_dir": str(DATA_DIR), "config": str(CONFIG_PATH)},
            "memory": {
                "episodic": {"enabled": self.cfg.get("memory", {}).get("episodic", True), "episodes": self.episodic.size()},
                "semantic": {"enabled": self.cfg.get("memory", {}).get("semantic", True), "items": self.semantic.size()},
            },
            "verification": self.cfg.get("verification", {}),
            "scoring": {"enabled": bool(self.cfg.get("scoring", {}).get("enabled", True) and TitanMindCore is not None)},
            "deps": {"titanmind": TitanMindCore is not None, "wikipedia_checker": WikipediaFactChecker is not None},
        }

    def recall(self, query: str, k: int) -> Dict[str, Any]:
        self.refresh_cfg()
        items = []

        if self.cfg.get("enabled", True) and self.cfg.get("memory", {}).get("episodic", True):
            for ep in self.episodic.search(query, limit=k):
                items.append({
                    "source": "episodic",
                    "text": json.dumps(ep.get("content"), ensure_ascii=False),
                    "metadata": {"timestamp": ep.get("timestamp"), "event_type": ep.get("event_type")},
                    "score": None,
                })

        if self.cfg.get("enabled", True) and self.cfg.get("memory", {}).get("semantic", True):
            for r in self.semantic.search(query, k=k):
                score = None
                if r.get("distance") is not None:
                    score = 1.0 - float(r.get("distance"))
                items.append({
                    "source": "semantic",
                    "text": r.get("text", ""),
                    "metadata": r.get("metadata", {}),
                    "score": score,
                })

        items.sort(key=lambda it: (0 if it["source"] == "semantic" else 1, -(it["score"] or 0.0)))
        return {"query": query, "k": k, "results": items[:k]}

    def remember(self, text: str, response: str, channel: str, sender: str) -> Dict[str, Any]:
        self.refresh_cfg()
        if not self.cfg.get("enabled", True):
            return {"success": False, "error": "SEVA disabled"}

        if self.cfg.get("memory", {}).get("episodic", True):
            self.episodic.record_event(
                event_type="chat",
                content={"text": text, "response": response, "channel": channel, "sender": sender},
                outcome="ok",
                importance=0.5,
            )

        if self.cfg.get("memory", {}).get("semantic", True) and self.cfg.get("memory", {}).get("store_all", False):
            self.semantic.store(
                text=f"Q: {text}\nA: {response}",
                metadata={"channel": channel, "sender": sender, "timestamp": time.time(), "text_length": len(text) + len(response) + 6},
            )

        return {"success": True}

    def score(self, text: str) -> Dict[str, Any]:
        if TitanMindCore is None:
            return {"success": False, "error": "TitanMindCore unavailable"}
        tm = TitanMindCore()
        drift = tm.calculate_drift(text)
        entropy = tm.calculate_entropy(text)
        score = tm.truth_score(text, drift, entropy)
        return {"success": True, "drift": drift, "entropy": entropy, "score": score}

    async def _get_http(self) -> aiohttp.ClientSession:
        if self._http is None or self._http.closed:
            self._http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15))
        return self._http

    async def _wolfram_verify(self, claim: str) -> Optional[Dict[str, Any]]:
        vcfg = self.cfg.get("verification", {}).get("wolfram", {})
        enabled = bool(vcfg.get("enabled", False))
        appid = (vcfg.get("appid", "") or os.environ.get("WOLFRAM_APPID", "")).strip()
        if not enabled or not appid:
            return None

        # Wolfram|Alpha v2 query API (JSON)
        url = f"https://api.wolframalpha.com/v2/query?appid={quote_plus(appid)}&input={quote_plus(claim)}&output=json"
        session = await self._get_http()
        async with session.get(url) as resp:
            data = await resp.json(content_type=None)

        qr = data.get("queryresult") if isinstance(data, dict) else None
        if not isinstance(qr, dict):
            return {"type": "wolfram", "ok": False, "error": "bad_response"}

        success = bool(qr.get("success"))
        pods = qr.get("pods")
        evidence: List[str] = []
        if isinstance(pods, list):
            for pod in pods[:6]:
                if not isinstance(pod, dict):
                    continue
                title = str(pod.get("title", "")).strip()
                subpods = pod.get("subpods")
                texts: List[str] = []
                if isinstance(subpods, list):
                    for sp in subpods:
                        if isinstance(sp, dict) and sp.get("plaintext"):
                            texts.append(str(sp.get("plaintext")).strip())
                elif isinstance(subpods, dict) and subpods.get("plaintext"):
                    texts.append(str(subpods.get("plaintext")).strip())
                texts = [t for t in texts if t]
                if texts:
                    chunk = texts[0]
                    if title:
                        evidence.append(f"{title}: {chunk}")
                    else:
                        evidence.append(chunk)

        return {
            "type": "wolfram",
            "ok": True,
            "success": success,
            "evidence": evidence[:5],
            "raw": {"success": success, "numpods": qr.get("numpods"), "timing": qr.get("timing")},
        }

    async def _get_http(self) -> aiohttp.ClientSession:
        if self._http is None or self._http.closed:
            self._http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15))
        return self._http

    async def _verify_sympy(self, claim: str) -> Optional[Dict[str, Any]]:
        vcfg = self.cfg.get("verification", {}).get("sympy", {})
        if not bool(vcfg.get("enabled", False)):
            return None

        try:
            import sympy as sp
        except Exception as e:
            return {"type": "sympy", "ok": False, "error": f"sympy_unavailable: {e}"}

        raw = claim.strip()
        # normalize
        expr = raw.replace("^", "**")
        # handle simple equality forms
        if "==" in expr:
            left, right = expr.split("==", 1)
            left = left.strip(); right = right.strip()
            try:
                ok = sp.simplify(sp.sympify(left) - sp.sympify(right)) == 0
                return {"type": "sympy", "ok": True, "asserted": bool(ok), "confidence": 0.95, "evidence": [f"simplify({left} - ({right})) == 0 -> {ok}"]}
            except Exception as e:
                return {"type": "sympy", "ok": False, "error": f"parse_error: {e}"}
        if "=" in expr and expr.count("=") == 1:
            left, right = expr.split("=", 1)
            left = left.strip(); right = right.strip()
            # avoid treating assignments like x=2 as verification
            if any(ch.isalpha() for ch in left) and not any(ch.isalpha() for ch in right):
                # still can be equation, allow
                pass
            try:
                ok = sp.simplify(sp.sympify(left) - sp.sympify(right)) == 0
                return {"type": "sympy", "ok": True, "asserted": bool(ok), "confidence": 0.95, "evidence": [f"simplify({left} - ({right})) == 0 -> {ok}"]}
            except Exception as e:
                return {"type": "sympy", "ok": False, "error": f"parse_error: {e}"}

        # fallback: try evaluating numeric expression
        try:
            val = sp.N(sp.sympify(expr))
            return {"type": "sympy", "ok": True, "asserted": None, "confidence": 0.4, "evidence": [f"eval -> {val}"]}
        except Exception as e:
            return {"type": "sympy", "ok": False, "error": f"parse_error: {e}"}

    async def _verify_wolfram(self, claim: str) -> Optional[Dict[str, Any]]:
        vcfg = self.cfg.get("verification", {}).get("wolfram", {})
        enabled = bool(vcfg.get("enabled", False))
        appid = (vcfg.get("appid", "") or os.environ.get("WOLFRAM_APPID", "")).strip()
        if not enabled or not appid:
            return None

        url = f"https://api.wolframalpha.com/v2/query?appid={quote_plus(appid)}&input={quote_plus(claim)}&output=json"
        session = await self._get_http()
        async with session.get(url) as resp:
            data = await resp.json(content_type=None)

        qr = data.get("queryresult") if isinstance(data, dict) else None
        if not isinstance(qr, dict):
            return {"type": "wolfram", "ok": False, "error": "bad_response"}

        success = bool(qr.get("success"))
        pods = qr.get("pods")
        evidence = []
        if isinstance(pods, list):
            for pod in pods[:6]:
                if not isinstance(pod, dict):
                    continue
                title = str(pod.get("title", "")).strip()
                subpods = pod.get("subpods")
                texts = []
                if isinstance(subpods, list):
                    for sp in subpods:
                        if isinstance(sp, dict) and sp.get("plaintext"):
                            texts.append(str(sp.get("plaintext")).strip())
                elif isinstance(subpods, dict) and subpods.get("plaintext"):
                    texts.append(str(subpods.get("plaintext")).strip())
                texts = [t for t in texts if t]
                if texts:
                    chunk = texts[0]
                    evidence.append(f"{title}: {chunk}" if title else chunk)

        # Wolfram is evidence, not truth; asserted stays null
        return {"type": "wolfram", "ok": True, "asserted": None, "confidence": 0.6 if success else 0.2, "evidence": evidence[:5], "raw": {"success": success, "numpods": qr.get("numpods"), "timing": qr.get("timing")}}

    async def verify(self, claim: str, providers: Optional[list[str]] = None, run_all: bool = False) -> Dict[str, Any]:
        self.refresh_cfg()
        if not self.cfg.get("enabled", True) or not self.cfg.get("verification", {}).get("enabled", True):
            return {"success": False, "error": "verification disabled"}

        vcfg = self.cfg.get("verification", {})
        configured = vcfg.get("providers") if isinstance(vcfg.get("providers"), list) else ["wikipedia"]
        ordered = providers or configured

        out: Dict[str, Any] = {"claim": claim, "verified": None, "issues": [], "sources": []}
        if self.cfg.get("scoring", {}).get("enabled", True) and TitanMindCore is not None:
            out["score"] = self.score(claim)

        false_thresh = float(vcfg.get("false_confidence_threshold", 0.85))

        asserted_true = False
        asserted_false = False

        for name in ordered:
            n = str(name).lower().strip()
            if not n:
                continue
            try:
                if n in ("wiki", "wikipedia", "wikidata"):
                    if self.cfg.get("verification", {}).get("wikipedia", False) and WikipediaFactChecker is not None:
                        checker = WikipediaFactChecker()
                        try:
                            results = await checker.extract_and_verify_claims(claim)
                            src = {"type": "wikipedia", "ok": True, "results": [asdict(r) for r in results]}
                            out["sources"].append(src)
                            for r in src["results"]:
                                if isinstance(r, dict) and r.get("verified") is True:
                                    asserted_true = True
                        finally:
                            try:
                                await checker.close()
                            except Exception:
                                pass
                elif n == "sympy":
                    src = await self._verify_sympy(claim)
                    if src is not None:
                        out["sources"].append(src)
                        if src.get("asserted") is True:
                            asserted_true = True
                        if src.get("asserted") is False and float(src.get("confidence", 0.0)) >= false_thresh:
                            asserted_false = True
                elif n == "wolfram":
                    src = await self._verify_wolfram(claim)
                    if src is not None:
                        out["sources"].append(src)
                else:
                    out["issues"].append(f"unknown_provider: {n}")
            except Exception as e:
                out["issues"].append(f"provider_error:{n}:{e}")

            if not run_all and asserted_true:
                # if any provider strongly verifies true, we can stop early
                break
            if not run_all and asserted_false:
                out["verified"] = False
                return out

        out["verified"] = True if asserted_true else (False if asserted_false else None)
        return out



async def json_response(handler):
    async def _wrapped(request: web.Request) -> web.Response:
        try:
            payload = await handler(request)
            return web.json_response(payload)
        except web.HTTPException:
            raise
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    return _wrapped


def main() -> None:
    host = os.environ.get("OPENCLAW_SEVA_HOST", "127.0.0.1")
    port = int(os.environ.get("OPENCLAW_SEVA_PORT", "18790"))

    svc = SevaService()

    routes = web.RouteTableDef()

    @routes.get("/status")
    async def _status(request: web.Request):
        return web.json_response(svc.status())

    @routes.get("/config")
    async def _config(request: web.Request):
        return web.json_response(load_config())

    @routes.post("/config-set")
    async def _config_set(request: web.Request):
        body = await request.json()
        cfg = load_config()
        for pair in body.get("set", []) or []:
            dotted_set(cfg, pair)
        save_config(cfg)
        return web.json_response(cfg)

    @routes.post("/recall")
    async def _recall(request: web.Request):
        body = await request.json()
        return web.json_response(svc.recall(body.get("query", ""), int(body.get("k", 5))))

    @routes.post("/remember")
    async def _remember(request: web.Request):
        body = await request.json()
        return web.json_response(
            svc.remember(
                text=body.get("text", ""),
                response=body.get("response", ""),
                channel=body.get("channel", "unknown"),
                sender=body.get("sender", "unknown"),
            )
        )

    @routes.post("/score")
    async def _score(request: web.Request):
        body = await request.json()
        return web.json_response(svc.score(body.get("text", "")))

    
    @routes.post("/mode-set")
    async def _mode_set(request: web.Request):
        body = await request.json()
        mode = (body.get("mode") or "").strip()
        presets = load_presets().get("modes", {})
        if mode not in presets:
            return web.json_response({"success": False, "error": f"unknown_mode: {mode}", "modes": sorted(presets.keys())}, status=400)
        cfg = load_config()
        # apply preset dotted keys
        for k, v in presets[mode].items():
            # write into cfg via dotted path
            cur = cfg
            parts = k.split(".")
            for p2 in parts[:-1]:
                if p2 not in cur or not isinstance(cur[p2], dict):
                    cur[p2] = {}
                cur = cur[p2]
            cur[parts[-1]] = v
        cfg.setdefault("runtime", {})["mode"] = mode
        save_config(cfg)
        return web.json_response({"success": True, "mode": mode, "config": cfg})


    @routes.post("/verify")
    async def _verify(request: web.Request):
        body = await request.json()
        return web.json_response(await svc.verify(body.get("claim", ""), providers=body.get("providers"), run_all=bool(body.get("all", False))))

    app = web.Application()
    app.add_routes([web.get("/status", _status), web.get("/config", _config)])
    app.add_routes([
        web.post("/config-set", _config_set),
        web.post("/recall", _recall),
        web.post("/remember", _remember),
        web.post("/score", _score),
        web.post("/verify", _verify),
    ])

    web.run_app(app, host=host, port=port, print=None)


if __name__ == "__main__":
    main()
