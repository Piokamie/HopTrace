"""Dataset registry: pinned URLs, sizes, checksums, published baselines.

Datasets are downloaded to ``$HOPTRACE_DATA_DIR/datasets/`` on first use and
never committed. URLs, sizes and the gate baseline were verified live on
2026-08-13; SHA256 values are pinned after the first verified download —
``sha256=None`` means "record on first download", a mismatch against a
pinned value is a hard error.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from hoptrace.config import data_dir


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    url: str
    filename: str
    #: Approximate size in bytes as verified live (informational).
    size: int
    license: str
    sha256: str | None = None
    #: For zip archives: directory created under datasets/ after extraction.
    extract_dir: str | None = None


DATASETS: dict[str, DatasetSpec] = {
    "beir-hotpotqa": DatasetSpec(
        name="beir-hotpotqa",
        url="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/hotpotqa.zip",
        filename="hotpotqa.zip",
        size=654_025_350,
        license="CC BY-SA 4.0 (HotpotQA), BEIR packaging Apache-2.0",
        sha256=None,
        extract_dir="hotpotqa",
    ),
    "musique": DatasetSpec(
        name="musique",
        url=(
            "https://huggingface.co/datasets/dgslibisey/MuSiQue/resolve/main/"
            "musique_ans_v1.0_dev.jsonl"
        ),
        filename="musique_ans_v1.0_dev.jsonl",
        size=30_439_728,
        license="CC BY 4.0",
        sha256=None,
    ),
    "2wiki": DatasetSpec(
        name="2wiki",
        url="https://huggingface.co/datasets/voidful/2WikiMultihopQA/resolve/main/dev.json",
        filename="2wiki_dev.json",
        size=55_934_464,
        license="Apache-2.0",
        sha256=None,
    ),
    # The official host (curtis.ml.cmu.edu) was unreachable at verification
    # time; the distractor sanity check runs on MuSiQue/2Wiki per-question
    # paragraphs instead, and HotpotQA corpus-scale is covered via BEIR.
    "hotpotqa-distractor": DatasetSpec(
        name="hotpotqa-distractor",
        url="http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
        filename="hotpot_dev_distractor_v1.json",
        size=46_320_117,
        license="CC BY-SA 4.0",
        sha256=None,
    ),
}

#: BM25 parameters every eval run uses — the published-baseline
#: configuration (Anserini defaults). Single source for all harness
#: signatures and CLI defaults.
EVAL_K1 = 0.9
EVAL_B = 0.4

#: Gate target for the flat index HopTrace builds (title concatenated into
#: text). Sources: Pyserini BEIR reproduction matrix
#: (castorini.github.io/pyserini/2cr/beir.html, BM25 "flat"), and the BEIR
#: paper (Thakur et al., NeurIPS 2021, Table 2; Anserini multifield,
#: k1=0.9, b=0.4) as secondary reference.
BEIR_HOTPOTQA_BASELINES = {
    "bm25_flat_ndcg10": 0.633,
    "bm25_flat_recall100": 0.796,
    "bm25_multifield_ndcg10": 0.603,
    "bm25_multifield_recall100": 0.740,
    "k1": EVAL_K1,
    "b": EVAL_B,
}
GATE_TOLERANCE = 0.02


def datasets_dir() -> Path:
    return data_dir() / "datasets"


def ensure_dataset(name: str) -> Path:
    """Download (once) and return the local path of a dataset.

    For zip specs, returns the extracted directory.
    """
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; known: {sorted(DATASETS)}")
    spec = DATASETS[name]
    root = datasets_dir()
    root.mkdir(parents=True, exist_ok=True)
    archive = root / spec.filename

    if spec.extract_dir is not None:
        extracted = root / spec.extract_dir
        if extracted.is_dir():
            return extracted

    if not archive.is_file():
        _download(spec, archive)

    digest = _sha256(archive)
    recorded = _recorded_sha(root, spec.filename)
    expected = spec.sha256 or recorded
    if expected is not None and digest != expected:
        archive.rename(archive.with_suffix(archive.suffix + ".corrupt"))
        raise RuntimeError(
            f"checksum mismatch for {spec.filename}: expected {expected}, got {digest}."
            " The corrupt file was set aside; re-run to download again, or fetch"
            f" manually from {spec.url}"
        )
    (root / f"{spec.filename}.sha256").write_text(f"{digest}  {spec.filename}\n")

    if spec.extract_dir is not None:
        print(f"[hoptrace] extracting {spec.filename} ...", file=sys.stderr)
        staging = root / f".extract-{spec.extract_dir}"
        if staging.is_dir():
            shutil.rmtree(staging)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(staging)
        inner = staging / spec.extract_dir
        if not inner.is_dir():
            shutil.rmtree(staging, ignore_errors=True)
            raise RuntimeError(f"{spec.filename} did not contain {spec.extract_dir}/")
        extracted = root / spec.extract_dir
        inner.rename(extracted)
        shutil.rmtree(staging, ignore_errors=True)
        return extracted
    return archive


def _recorded_sha(root: Path, filename: str) -> str | None:
    """Digest recorded on the first verified download — read back so a
    re-download (or tampering) cannot silently pass when the spec ships
    ``sha256=None``."""
    sidecar = root / f"{filename}.sha256"
    if not sidecar.is_file():
        return None
    content = sidecar.read_text().split()
    return content[0] if content else None


def _download(spec: DatasetSpec, target: Path) -> None:
    print(
        f"[hoptrace] downloading {spec.name} (~{spec.size / 1e6:.0f} MB) from {spec.url}",
        file=sys.stderr,
    )
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(spec.url) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out, length=1 << 20)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"download failed for {spec.name} from {spec.url}: {exc}."
            f" Fetch it manually and place it at {target}"
        ) from exc
    tmp.rename(target)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
