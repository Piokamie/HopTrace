"""Stage a trained reranker for distribution.

Copies ``model_int8.onnx``, ``tokenizer.json``, ``manifest.json`` into
``<repo>/models/<name>/`` and prints the ``ModelSpec`` registry entry with
every file's sha256. ``model.onnx`` (fp32) is hashed but not copied: it
ships on the GitHub Release the registry URLs point at.

    uv run python training/bundle_artifact.py $HOPTRACE_DATA_DIR/models/hoptrace-rerank-minilm-l6

Then paste the printed entry into ``src/hoptrace/rerank.py`` ``MODELS``,
attach ``model.onnx`` (and, for wheel installs, the three bundled files)
to the release named in ``_RELEASE``, and commit ``models/<name>/``.
Refuses artifacts whose manifest records a dirty trainer tree.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from hoptrace.rerank import sha256_of

BUNDLED = ("model_int8.onnx", "tokenizer.json", "manifest.json")
RELEASED = ("model.onnx",)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--allow-dirty", action="store_true", help="bundle a dirty-tree artifact anyway"
    )
    args = parser.parse_args(argv)

    manifest_path = args.model_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("dirty") or str(manifest.get("trainer_commit", "")).endswith("-dirty")
    ) and not args.allow_dirty:
        print(
            f"error: {manifest_path} records a dirty trainer tree"
            f" ({manifest.get('trainer_commit')});"
            " retrain from a clean commit before bundling (or --allow-dirty)",
            file=sys.stderr,
        )
        return 1
    missing = [n for n in BUNDLED + RELEASED if not (args.model_dir / n).is_file()]
    if missing:
        print(f"error: {args.model_dir} lacks {', '.join(missing)}", file=sys.stderr)
        return 1
    for name in BUNDLED + RELEASED:
        recorded = manifest.get("files", {}).get(name)
        actual = sha256_of(args.model_dir / name)
        if recorded is not None and recorded != actual:
            print(f"error: {name}: manifest says {recorded}, file is {actual}", file=sys.stderr)
            return 1

    name = manifest.get("model", args.model_dir.name)
    target = args.repo_root / "models" / name
    target.mkdir(parents=True, exist_ok=True)
    for filename in BUNDLED:
        shutil.copyfile(args.model_dir / filename, target / filename)
    print(f"bundled {', '.join(BUNDLED)} -> {target}", file=sys.stderr)

    print(f'    "{name}": ModelSpec(')
    print(f'        name="{name}",')
    print('        default_precision="int8",')
    print("        files=(")
    for filename in ("model_int8.onnx", "model.onnx", "tokenizer.json", "manifest.json"):
        digest = sha256_of(args.model_dir / filename)
        print(f'            ModelFile("{filename}", f"{{_RELEASE}}/{filename}", "{digest}"),')
    print("        ),")
    print("    ),")
    print(
        f"\nattach to the release: {', '.join(BUNDLED + RELEASED)} from {args.model_dir}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
