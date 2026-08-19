import json
from pathlib import Path

import pytest

import hoptrace.cli as cli

FIXTURES = Path(__file__).parent / "fixtures"

pytest.importorskip("numpy")


@pytest.fixture(autouse=True)
def fixture_datasets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the CLI at committed fixtures instead of real downloads."""
    monkeypatch.setenv("HOPTRACE_DATA_DIR", str(tmp_path / "data"))

    def fake_ensure(name: str) -> Path:
        return {
            "beir-hotpotqa": FIXTURES / "beir",
            "musique": FIXTURES / "musique_dev.jsonl",
            "2wiki": FIXTURES / "2wiki_dev.json",
        }[name]

    monkeypatch.setattr(cli, "ensure_dataset", fake_ensure)


def test_eval_musique_pooled(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["eval", "--dataset", "musique"]) == 0
    out = capsys.readouterr().out
    assert "musique (pooled-dev)" in out
    assert "NOT comparable to published open-domain numbers" in out
    assert "recall@" in out
    assert "hit rule" in out


def test_eval_2wiki_distractor(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["eval", "--dataset", "2wiki", "--setting", "distractor"]) == 0
    out = capsys.readouterr().out
    assert "sanity check" in out


def test_eval_beir_builds_and_runs(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["eval", "--dataset", "beir-hotpotqa"]) == 0
    out = capsys.readouterr().out
    assert "beir-hotpotqa (corpus-scale)" in out
    assert "nDCG@10" in out
    # second run reuses the index
    assert cli.main(["eval", "--dataset", "beir-hotpotqa"]) == 0


def test_eval_json_output(tmp_path: Path) -> None:
    json_path = tmp_path / "out" / "report.json"
    assert cli.main(["eval", "--dataset", "musique", "--json", str(json_path)]) == 0
    assert json_path.is_file()


def test_eval_hops_two(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["eval", "--dataset", "musique", "--hops", "2"]) == 0
    out = capsys.readouterr().out
    assert "hoptrace@2hop" in out
    assert "retrieval config: hops=2" in out


def test_eval_diagnostics(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["eval", "--dataset", "musique", "--hops", "2", "--diagnostics"]) == 0
    out = capsys.readouterr().out
    assert "miss breakdown" in out
    assert "calibration [musique]" in out
    assert "annotated multi-hop" in out


def test_eval_pool_ablation(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["eval", "--dataset", "musique", "--pool-ablation", "2"]) == 0
    out = capsys.readouterr().out
    assert "pool precision" in out
    assert "specificity filter OFF" in out


def test_distractor_rejects_hops(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["eval", "--dataset", "musique", "--setting", "distractor", "--hops", "2"])
    assert code == 2
    assert "floor-only" in capsys.readouterr().err


def test_gate_refuses_partial_run(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["eval", "--dataset", "beir-hotpotqa", "--limit", "1", "--gate"])
    assert code == 2
    assert "refusing" in capsys.readouterr().err


def test_gate_on_fixture_fails_honestly(capsys: pytest.CaptureFixture[str]) -> None:
    # the 6-passage fixture cannot reach the published number; the gate still compares to it
    code = cli.main(["eval", "--dataset", "beir-hotpotqa", "--gate"])
    out = capsys.readouterr().out
    assert code in (0, 1)
    assert "gate [" in out
    assert "0.633" in out


def test_limit_flag(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["eval", "--dataset", "musique", "--limit", "1"]) == 0
    assert "NOT the reportable number" in capsys.readouterr().out


# --- product subcommands ---


@pytest.fixture
def office(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import hoptrace.server as server

    monkeypatch.setattr(server, "_registry", server.CorpusRegistry())
    src = tmp_path / "docs"
    src.mkdir()
    (src / "people.md").write_text(
        "# People\n\nAnna Nowak reports to Kowalski. We asked Anna about the atrium."
    )
    (src / "rooms.md").write_text("# Rooms\n\nKowalski occupies Room 4B near the atrium fountain.")
    (src / "cafe.md").write_text("# Cafe\n\nThe atrium fountain cafe closes at five daily.")
    return src


def test_ingest_and_retrieve_cli(office: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["ingest", str(office), "--corpus", "office"]) == 0
    out = capsys.readouterr().out
    assert "3 documents" in out

    assert (
        cli.main(["retrieve", "Where does the manager of Anna Nowak sit?", "--corpus", "office"])
        == 0
    )
    out = capsys.readouterr().out
    assert "path:" in out
    assert "kowalski" in out
    assert "(rooms.md)" in out  # the answer chunk's file on its result line
    assert "terms: " in out  # matched query words beside the score


def test_retrieve_json_cli(office: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["ingest", str(office), "--corpus", "office"]) == 0
    capsys.readouterr()
    assert cli.main(["retrieve", "Anna Nowak", "--corpus", "office", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["evidence"]


def test_explain_cli(office: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["ingest", str(office), "--corpus", "office"]) == 0
    capsys.readouterr()
    code = cli.main(
        ["explain", "2", "--corpus", "office", "--query", "Where does Anna Nowak's manager sit?"]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "chunk#2" in out
    assert "bm25 term" in out


def test_bracket_cli(office: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["ingest", str(office), "--corpus", "office"]) == 0
    capsys.readouterr()
    assert cli.main(["bracket", "--corpus", "office", "-n", "6"]) == 0
    out = capsys.readouterr().out
    assert "bm25-floor" in out
    assert "CAVEAT" in out


def test_unknown_corpus_cli(office: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["retrieve", "q", "--corpus", "nope"]) == 1
    assert "available" in capsys.readouterr().err


def test_serve_registered() -> None:
    # smoke: the subcommand exists and parses (running it would block on stdio)
    with pytest.raises(SystemExit):
        cli.main(["serve", "--help"])


def test_retrieve_rerank_without_extra_exits_with_install_hint(
    office: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import hoptrace.rerank as rerank

    def missing() -> tuple[object, object]:
        raise RuntimeError("reranking requires the rerank extra: install with hoptrace[rerank]")

    monkeypatch.setattr(rerank, "_require_runtime", missing)
    assert cli.main(["ingest", str(office), "--corpus", "office"]) == 0
    capsys.readouterr()
    code = cli.main(["retrieve", "Anna Nowak", "--corpus", "office", "--rerank"])
    assert code == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "hoptrace[rerank]" in err
    assert "Traceback" not in err


def test_rerank_flags_require_their_switch() -> None:
    with pytest.raises(SystemExit):
        cli.main(["retrieve", "q", "--corpus", "office", "--rerank-precision", "int8"])
    with pytest.raises(SystemExit):
        cli.main(["retrieve", "q", "--corpus", "office", "--rerank-model", "x"])
    with pytest.raises(SystemExit):
        cli.main(["eval", "--dataset", "musique", "--rerank-precision", "fp32"])
    with pytest.raises(SystemExit):
        cli.main(["eval", "--dataset", "musique", "--allow-dirty-manifest"])
    with pytest.raises(SystemExit):
        cli.main(["eval", "--dataset", "musique", "--rerank-top-n", "0"])


def test_eval_rerank_needs_hops_and_hipporag_rejects_pool_flags() -> None:
    with pytest.raises(SystemExit):
        cli.main(["eval", "--dataset", "musique", "--selection", "rerank"])  # hops 0
    with pytest.raises(SystemExit):
        cli.main(["eval", "--dataset", "hipporag-musique", "--corpus-pool", "all"])
    with pytest.raises(SystemExit):
        cli.main(["eval", "--dataset", "hipporag-musique", "--setting", "distractor"])
