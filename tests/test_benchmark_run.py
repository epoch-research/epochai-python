import sys
import types
from types import SimpleNamespace

import pytest

from epochai.airtable.models import BenchmarkRun

LOGS_URL = "s3://some-bucket/run.eval"


def make_run():
    run = BenchmarkRun.from_id("recABC", fetch=False)
    run.logs = LOGS_URL
    return run


def make_sample(output_tokens):
    usage = None if output_tokens is None else SimpleNamespace(output_tokens=output_tokens)
    return SimpleNamespace(output=SimpleNamespace(usage=usage))


@pytest.fixture
def fake_read_eval_log(monkeypatch):
    """Inject a fake `inspect_ai.log` module so no real eval logs are needed."""
    calls = {}

    def read_eval_log(path, **kwargs):
        calls["path"] = path
        calls["kwargs"] = kwargs
        return SimpleNamespace(samples=[make_sample(100), make_sample(250), make_sample(None)])

    log_module = types.ModuleType("inspect_ai.log")
    log_module.read_eval_log = read_eval_log
    package = types.ModuleType("inspect_ai")
    package.log = log_module
    monkeypatch.setitem(sys.modules, "inspect_ai", package)
    monkeypatch.setitem(sys.modules, "inspect_ai.log", log_module)
    return calls


def test_read_eval_log_uses_logs_url(fake_read_eval_log):
    log = make_run().read_eval_log(resolve_attachments=True)
    assert fake_read_eval_log["path"] == LOGS_URL
    assert fake_read_eval_log["kwargs"] == {"resolve_attachments": True}
    assert len(log.samples) == 3


def test_response_lengths(fake_read_eval_log):
    assert make_run().response_lengths() == [100, 250, 0]


def test_response_lengths_without_samples(fake_read_eval_log, monkeypatch):
    def read_eval_log(path, **kwargs):
        return SimpleNamespace(samples=None)

    monkeypatch.setattr(sys.modules["inspect_ai.log"], "read_eval_log", read_eval_log)
    assert make_run().response_lengths() == []


def test_read_eval_log_requires_inspect_ai():
    # inspect-ai is not installed in the test environment, so the lazy import fails.
    with pytest.raises(ImportError, match="inspect-ai"):
        make_run().read_eval_log()
