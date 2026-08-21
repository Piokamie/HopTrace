"""Dataset registry: pinned URLs, sizes, checksums, published baselines.

Datasets are downloaded to ``$HOPPATH_DATA_DIR/datasets/`` on first use and
never committed. ``sha256=None`` means "record on first download"; a mismatch
against a pinned or recorded value is a hard error.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from hoppath.config import data_dir


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    url: str
    filename: str
    #: Approximate size in bytes (informational).
    size: int
    license: str
    sha256: str | None = None
    #: For zip archives: directory created under datasets/ after extraction.
    extract_dir: str | None = None


#: HippoRAG `legacy` branch, pinned to a commit so the published sample cannot drift.
_HIPPORAG_COMMIT = "b144c46df14cabe5f5822d8caded4bec5f709461"
_HIPPORAG_RAW = f"https://raw.githubusercontent.com/OSU-NLP-Group/HippoRAG/{_HIPPORAG_COMMIT}/data"

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
    # Train split — for the training-data builder, not evaluation.
    "musique-train": DatasetSpec(
        name="musique-train",
        url=(
            "https://huggingface.co/datasets/dgslibisey/MuSiQue/resolve/main/"
            "musique_ans_v1.0_train.jsonl"
        ),
        filename="musique_ans_v1.0_train.jsonl",
        size=241_046_755,
        license="CC BY 4.0",
        sha256=None,
    ),
    # HippoRAG's published validation samples and corpora (Gutiérrez et al.,
    # NeurIPS 2024); digest-pinned because they define the comparable protocol.
    **{
        f"hipporag-{short}{suffix_name}": DatasetSpec(
            name=f"hipporag-{short}{suffix_name}",
            url=f"{_HIPPORAG_RAW}/{stem}{suffix_file}.json",
            filename=f"hipporag_{stem}{suffix_file}.json",
            size=size,
            license="MIT (HippoRAG packaging); CC BY 4.0 MuSiQue / Apache-2.0 2Wiki",
            sha256=digest,
        )
        for short, stem, suffix_name, suffix_file, size, digest in (
            (
                "musique",
                "musique",
                "",
                "",
                12_543_629,
                "98ed4e21d3076532f6388d42320fb809599c63a0d8dffca8ece5e41922be6b46",
            ),
            (
                "musique",
                "musique",
                "-corpus",
                "_corpus",
                6_239_261,
                "73157a03ce3f0b1a5673dd5dc12bb970c24976dbffc688af9eecdd758c97ffcb",
            ),
            (
                "2wiki",
                "2wikimultihopqa",
                "",
                "",
                6_505_789,
                "895cba294064df0c3302c76847b1fc08d99b5619f7663dfaa3b65cd780f1cac4",
            ),
            (
                "2wiki",
                "2wikimultihopqa",
                "-corpus",
                "_corpus",
                3_083_943,
                "9d6e352952aafb18dab22bf8195039461321a44a949df902ae83bce134ad238a",
            ),
        )
    },
    # Train split — corpus mass for the all-splits pooled protocol only;
    # 2Wiki is the transfer holdout (ADR 0011), not a training source.
    "2wiki-train": DatasetSpec(
        name="2wiki-train",
        url="https://huggingface.co/datasets/voidful/2WikiMultihopQA/resolve/main/train.json",
        filename="2wiki_train.json",
        size=681_705_246,
        license="Apache-2.0",
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
    # Host unreliable; the distractor sanity check runs on MuSiQue/2Wiki instead.
    "hotpotqa-distractor": DatasetSpec(
        name="hotpotqa-distractor",
        url="http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
        filename="hotpot_dev_distractor_v1.json",
        size=46_320_117,
        license="CC BY-SA 4.0",
        sha256=None,
    ),
}

#: BM25 parameters for every eval run (Anserini defaults, the published-baseline configuration).
EVAL_K1 = 0.9
EVAL_B = 0.4

#: Gate targets for the flat index (title concatenated into text). Sources:
#: Pyserini BEIR reproduction matrix (castorini.github.io/pyserini/2cr/beir.html)
#: and the BEIR paper (Thakur et al., NeurIPS 2021, Table 2).
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
        print(f"[hoppath] extracting {spec.filename} ...", file=sys.stderr)
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
    """Digest recorded on first download, so a later re-download is still
    checked when the spec has ``sha256=None``."""
    sidecar = root / f"{filename}.sha256"
    if not sidecar.is_file():
        return None
    content = sidecar.read_text().split()
    return content[0] if content else None


def _download(spec: DatasetSpec, target: Path) -> None:
    print(
        f"[hoppath] downloading {spec.name} (~{spec.size / 1e6:.0f} MB) from {spec.url}",
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
