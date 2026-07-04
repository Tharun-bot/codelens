"""Smoke test for scripts/benchmark.py — proves the benchmark runs end-to-end
without error on a tiny fixture repo. The real, full-scale benchmark run
(against a real OSS repo) is done manually, not in CI — see README.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from scripts.benchmark import run_benchmark, write_report  # noqa: E402


@pytest.fixture
def fixture_repo(tmp_path: Path) -> Path:
    (tmp_path / "sample.py").write_text(
        "def add(a, b):\n"
        "    \"\"\"Add two numbers.\"\"\"\n"
        "    return a + b\n"
        "\n"
        "def subtract(a, b):\n"
        "    return a - b\n"
    )
    return tmp_path


@pytest.mark.slow
def test_benchmark_runs_without_error(fixture_repo, tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'bench_test.db'}")

    stats = run_benchmark(
        repo=str(fixture_repo),
        num_queries=5,
        index_dir=tmp_path / "indexes",
    )

    assert stats["chunks_indexed"] == 2
    assert stats["num_queries"] == 5
    assert stats["index_time_s"] > 0
    assert stats["p50_ms"] >= 0


@pytest.mark.slow
def test_benchmark_report_is_written(fixture_repo, tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'bench_test2.db'}")

    stats = run_benchmark(
        repo=str(fixture_repo),
        num_queries=3,
        index_dir=tmp_path / "indexes",
    )

    report_path = tmp_path / "results.md"
    write_report(stats, report_path)

    assert report_path.exists()
    content = report_path.read_text()
    assert "CodeLens Benchmark Results" in content
    assert "p50" in content