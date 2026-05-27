"""
Append dendro tree features to an existing Tyler features directory.

Run roofer2tyler.py on the buildings (roofer) output first so that metadata.city.json
and features/ exist and metadata (including geographicalExtent) comes from buildings.
Then run this script on trees.city.jsonl; it reads both the dendro and roofer
transforms, re-quantizes every tree vertex so the integer coordinates are correct
under roofer's transform, and writes each feature into the same features/ directory.
Metadata is left unchanged.

Example (WSL / bash):
    python3 /mnt/d/github/tyler/roofer2tyler.py /mnt/d/lidar/dendro/output.city.jsonl
    python3 /mnt/d/github/tyler/dendro2tyler.py /mnt/d/lidar/dendro/trees.city.jsonl /mnt/d/lidar/dendro
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from roofer2tyler import validate_feature


def read_header(jsonl_path: Path) -> Dict[str, Any]:
    """Read the first (metadata/header) line of a CityJSONL file."""
    with jsonl_path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            return json.loads(stripped)
    raise ValueError(f"{jsonl_path} is empty")


def iter_feature_lines(jsonl_path: Path):
    """Yield (parsed, raw_line) for each line after the first (metadata) line."""
    with jsonl_path.open("r", encoding="utf-8-sig") as f:
        first = True
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if first:
                first = False
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in {jsonl_path}: {e}. Line starts with: {stripped[:80]!r}"
                ) from e
            yield payload, stripped


def get_transform(header: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    """Extract (scale, translate) from a CityJSON header."""
    t = header.get("transform", {})
    scale = t.get("scale", [0.001, 0.001, 0.001])
    translate = t.get("translate", [0.0, 0.0, 0.0])
    return scale, translate


def requantize_vertices(
    vertices: list,
    src_scale: List[float],
    src_translate: List[float],
    dst_scale: List[float],
    dst_translate: List[float],
) -> list:
    """Convert integer vertices from one CityJSON transform to another.

    actual = int_v * src_scale + src_translate
    new_int = round((actual - dst_translate) / dst_scale)
    """
    out = []
    for v in vertices:
        actual = [v[i] * src_scale[i] + src_translate[i] for i in range(3)]
        new_v = [round((actual[i] - dst_translate[i]) / dst_scale[i]) for i in range(3)]
        out.append(new_v)
    return out


def append_tree_features(trees_jsonl: Path, output_dir: Path) -> int:
    """
    Append tree features from trees.city.jsonl to output_dir/features/,
    re-quantizing vertices from the dendro transform to the roofer transform.
    """
    if not trees_jsonl.exists():
        raise FileNotFoundError(f"Input file {trees_jsonl} does not exist.")

    features_dir = output_dir / "features"
    metadata_path = output_dir / "metadata.city.json"
    if not features_dir.is_dir():
        raise FileNotFoundError(
            f"Features directory {features_dir} not found. Run roofer2tyler on buildings output first."
        )
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file {metadata_path} not found. Run roofer2tyler on buildings output first."
        )

    dendro_header = read_header(trees_jsonl)
    src_scale, src_translate = get_transform(dendro_header)

    roofer_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    dst_scale, dst_translate = get_transform(roofer_metadata)

    transforms_match = src_scale == dst_scale and src_translate == dst_translate
    if not transforms_match:
        print(
            f"Re-quantizing: dendro translate={src_translate} -> roofer translate={dst_translate}"
        )

    count = 0
    for payload, _raw_line in iter_feature_lines(trees_jsonl):
        feature = validate_feature(payload)

        if not transforms_match:
            payload["vertices"] = requantize_vertices(
                payload["vertices"], src_scale, src_translate, dst_scale, dst_translate
            )

        feature_id = feature["id"]
        feature_path = features_dir / f"{feature_id}.jsonl"
        feature_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        count += 1

    return count


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append dendro tree features to an existing Tyler features directory."
    )
    parser.add_argument(
        "trees_jsonl",
        type=Path,
        help="Path to dendro trees.city.jsonl (first line = metadata, skipped).",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory containing metadata.city.json and features/ (from roofer2tyler).",
    )
    return parser.parse_args(args=args)


def main() -> None:
    namespace = parse_args()
    try:
        count = append_tree_features(namespace.trees_jsonl, namespace.output_dir)
    except (json.JSONDecodeError, ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc))
    print(f"Appended {count} tree features to {namespace.output_dir / 'features'}")


if __name__ == "__main__":
    main()
