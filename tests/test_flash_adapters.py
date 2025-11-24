import os
import sys
import types
import json
import pytest

# Ensure project root on sys.path to import scripts.flash_analysis
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import flash_analysis as fa


class FakeResponse:
    def __init__(self, data=b"{}"):
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self._data


def test_healthy_worker_ports_selects_fast(monkeypatch):
    """
    Ports with quick 1-token generate (<= 5s) should be selected; slow ones should be ignored.
    """
    # Simulated clock
    clock = {"t": 1000.0}

    def fake_time():
        return clock["t"]

    def fake_urlopen(req, timeout=5):
        url = getattr(req, "full_url", str(req))
        # Fast port 11434: +0.2s
        if "127.0.0.1:11434" in url:
            clock["t"] += 0.2
            return FakeResponse(b'{"response": "ok"}')
        # Slow port 11435: +6s (should be ignored)
        if "127.0.0.1:11435" in url:
            clock["t"] += 6.0
            return FakeResponse(b'{"response": "ok"}')
        raise Exception("Connection refused")

    # Monkeypatch time.time used inside healthy_worker_ports
    monkeypatch.setattr("time.time", fake_time)
    # Monkeypatch urllib.request.urlopen used inside healthy_worker_ports
    import urllib.request as _ur
    monkeypatch.setattr(_ur, "urlopen", fake_urlopen)

    # No need to set OLLAMA_BASE_URL for this test
    ports = fa.healthy_worker_ports([11434, 11435], per_port_timeout=5)
    assert ports == [11434], f"Expected only fast port; got {ports}"


def test_healthy_worker_ports_fallback_to_base_url(monkeypatch):
    """
    When none of the candidate ports pass, fallback to OLLAMA_BASE_URL port if it responds fast.
    """
    clock = {"t": 2000.0}

    def fake_time():
        return clock["t"]

    def fake_urlopen(req, timeout=5):
        url = getattr(req, "full_url", str(req))
        # Candidate ports fail:
        if any(p in url for p in ["127.0.0.1:11436", "127.0.0.1:11437"]):
            raise Exception("Connection refused")
        # Fallback base url 11434 responds fast
        if "127.0.0.1:11434" in url:
            clock["t"] += 0.1
            return FakeResponse(b'{"response": "ok"}')
        raise Exception("Unknown host")

    monkeypatch.setattr("time.time", fake_time)
    import urllib.request as _ur
    monkeypatch.setattr(_ur, "urlopen", fake_urlopen)

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    ports = fa.healthy_worker_ports([11436, 11437], per_port_timeout=5)
    assert ports == [11434], f"Expected fallback to base URL port; got {ports}"


def test_warmup_model_on_ports_backoff(monkeypatch, tmp_path):
    """
    warmup_model_on_ports should retry on transient errors and ultimately succeed without raising.
    """
    attempts = {"count": 0}

    def fake_generate(prompt, model, port, options=None, timeout=10):
        attempts["count"] += 1
        # First attempt fails, second succeeds
        if attempts["count"] == 1:
            raise RuntimeError("transient: model loading")
        return "ok"

    monkeypatch.setattr(fa, "ollama_generate", fake_generate)

    # Ensure OUTPUT_ROOT exists for potential logging
    out_dir = tmp_path / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(fa, "OUTPUT_ROOT", str(out_dir))

    fa.warmup_model_on_ports([11434], model="llama3.1:8b")
    assert attempts["count"] >= 2, "Expected at least one retry"


def test_build_deterministic_fallback_report(tmp_path, monkeypatch):
    """
    Deterministic fallback should synthesize a minimal but useful report from evidence.csv.
    """
    out_dir = tmp_path / "job-xyz"
    out_dir.mkdir(parents=True, exist_ok=True)

    ev_path = out_dir / "evidence.csv"
    ev_path.write_text(
        "document,type,value\n"
        "doc1.pdf,amount,€ 1.234.567\n"
        "doc1.pdf,percentage,12%\n"
        "doc1.pdf,rate,5.2%\n"
        "doc1.pdf,entity,De Brem Holding B.V.\n"
        "doc1.pdf,date,2025-06-30\n",
        encoding="utf-8",
    )

    # OUTPUT_ROOT for potential fallback scanning
    monkeypatch.setattr(fa, "OUTPUT_ROOT", str(tmp_path))

    report = fa.build_deterministic_fallback_report("Project De Brem", str(out_dir))
    assert "Deterministische fallback" in report
    assert "Top bedragen" in report
    assert "€ 1.234.567" in report
    assert "De Brem Holding" in report
    # Accept "onvoldoende data" if timeline parsing yields none
    assert "onvoldoende data" in report or "Tijdlijn" in report
