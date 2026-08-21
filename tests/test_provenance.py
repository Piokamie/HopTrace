from hoppath.provenance import HopEdge, HopPath, render_path


def path_two_hop() -> HopPath:
    return HopPath(
        query_mention="Anna",
        edges=(
            HopEdge("anna", 12, 0.9, 1),
            HopEdge("kowalski", 31, 0.8, 2),
        ),
    )


def test_hop_count() -> None:
    assert path_two_hop().hop == 1
    assert HopPath("Anna", (HopEdge("anna", 12, 0.9, 1),)).hop == 0


def test_seed_source() -> None:
    assert path_two_hop().seed_source == "mention"
    assert HopPath(None, (HopEdge("", 5, 0.0, 0),)).seed_source == "bm25_only"


def test_render_matches_design_format() -> None:
    snippets = {12: "Anna reports to Kowalski", 31: "Kowalski sits in 4B"}
    rendered = render_path(path_two_hop(), snippets)
    assert rendered == (
        'query:"Anna" → chunk#12 ("Anna reports to Kowalski")'
        ' → entity:"kowalski" → chunk#31 ("Kowalski sits in 4B")'
    )


def test_render_names_source_files_when_known() -> None:
    snippets = {12: "Anna reports to Kowalski", 31: "Kowalski sits in 4B"}
    docs = {12: "people/anna.md", 31: "rooms/4b.md"}
    assert render_path(path_two_hop(), snippets, docs) == (
        'query:"Anna" → chunk#12 (people/anna.md: "Anna reports to Kowalski")'
        ' → entity:"kowalski" → chunk#31 (rooms/4b.md: "Kowalski sits in 4B")'
    )
    # file known, no snippet; and a partial docs map leaves other chunks bare
    assert render_path(path_two_hop(), {}, {12: "people/anna.md"}) == (
        'query:"Anna" → chunk#12 (people/anna.md) → entity:"kowalski" → chunk#31'
    )


def test_render_bm25_seed_and_missing_snippet() -> None:
    path = HopPath(None, (HopEdge("", 5, 0.0, 0),))
    assert render_path(path, {}) == "query:bm25 → chunk#5"


def test_snippet_truncation() -> None:
    long_text = "word " * 40
    rendered = render_path(HopPath("X", (HopEdge("x", 1, 0.5, 1),)), {1: long_text})
    assert "…" in rendered
    assert len(rendered) < 120
