#!/usr/bin/env python3
"""
Convert an OBJ tree template + a tree-crowns GeoPackage into a CityJSONSeq
file of SolitaryVegetationObject features, ready to feed Tyler 0.4.1
alongside a buildings CityJSONSeq.

REQUIRED PYTHON PACKAGES
    Always:                    geopandas, pyogrio (or fiona), shapely
    --simplify hull only:      numpy, scipy
    --simplify ratio:<f> only: numpy, trimesh, fast-simplification

    Recommended: use the bundled pixi env at /mnt/d/github/tyler/pixi.toml:
        pixi run python /mnt/d/github/tyler/trees2cityjsonseq.py ...

Usage:
    python3 trees2cityjsonseq.py \
        --template /path/to/tree_template.obj \
        --crowns   /path/to/tree_crowns.gpkg \
        [--out     /path/to/output.jsonl] \
        [--simplify {none,pyramid,hull,ratio:<f>}]

Default --out is <crowns-stem>-trees.city.jsonl next to the crowns file.
The .city.jsonl suffix is required so cjindex picks the file up under tyler-041's
NDJSON layout.

Required crowns columns:
    treeID, centroid_x, centroid_y, ground_z, tree_height, crown_area
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import geopandas as gpd

Vec3 = Tuple[float, float, float]


def parse_obj(path: Path) -> Tuple[List[Vec3], List[List[int]]]:
    verts: List[Vec3] = []
    faces: List[List[int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                idx = []
                for tok in line.split()[1:]:
                    idx.append(int(tok.split("/")[0]) - 1)
                faces.append(idx)
    if not verts or not faces:
        raise ValueError(f"OBJ {path} has no vertices or faces")
    return verts, faces


def normalize_template(verts: List[Vec3]) -> Tuple[List[Vec3], float, float]:
    cx = sum(v[0] for v in verts) / len(verts)
    cy = sum(v[1] for v in verts) / len(verts)
    minz = min(v[2] for v in verts)
    centred = [(v[0] - cx, v[1] - cy, v[2] - minz) for v in verts]
    xy_radius = max(math.hypot(v[0], v[1]) for v in centred)
    height = max(v[2] for v in centred)
    if xy_radius == 0 or height == 0:
        raise ValueError("Template has zero XY radius or zero height after normalization")
    return centred, xy_radius, height


def simplify_template(
    verts: List[Vec3], faces: List[List[int]], mode: str
) -> Tuple[List[Vec3], List[List[int]]]:
    if mode == "none":
        return verts, faces
    if mode == "pyramid":
        v = [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        f = [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]
        return v, f
    if mode == "hull":
        try:
            import numpy as np
            from scipy.spatial import ConvexHull
        except ImportError as e:
            raise SystemExit(f"--simplify hull requires scipy: {e}")
        pts = np.asarray(verts)
        hull = ConvexHull(pts)
        used = sorted({int(i) for tri in hull.simplices for i in tri})
        remap = {old: new for new, old in enumerate(used)}
        new_verts = [tuple(pts[i].tolist()) for i in used]
        new_faces = [[remap[int(i)] for i in tri] for tri in hull.simplices]
        return new_verts, new_faces
    if mode.startswith("ratio:"):
        try:
            ratio = float(mode.split(":", 1)[1])
        except ValueError:
            raise SystemExit(f"Invalid simplify ratio: {mode!r}")
        if not 0.0 < ratio < 1.0:
            raise SystemExit("--simplify ratio must be in (0, 1)")
        try:
            import numpy as np
            import trimesh
        except ImportError as e:
            raise SystemExit(f"--simplify ratio:* requires trimesh: {e}")
        tri_faces: List[List[int]] = []
        for face in faces:
            for i in range(1, len(face) - 1):
                tri_faces.append([face[0], face[i], face[i + 1]])
        mesh = trimesh.Trimesh(vertices=np.asarray(verts), faces=np.asarray(tri_faces), process=False)
        target = max(4, math.ceil(ratio * len(tri_faces)))
        simplified = mesh.simplify_quadric_decimation(face_count=target)
        return [tuple(v.tolist()) for v in simplified.vertices], [list(map(int, f)) for f in simplified.faces]
    raise SystemExit(f"Unknown --simplify mode: {mode!r}")


REQUIRED_COLS = ("treeID", "centroid_x", "centroid_y", "ground_z", "tree_height", "crown_area")


def read_crowns(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    missing = [c for c in REQUIRED_COLS if c not in gdf.columns]
    if missing:
        raise SystemExit(f"Crowns file is missing required columns: {missing}")
    return gdf


def epsg_uri(gdf: gpd.GeoDataFrame) -> str:
    code = gdf.crs.to_epsg() if gdf.crs is not None else None
    if code is None:
        raise SystemExit("Crowns file has no EPSG code in its CRS metadata")
    return f"https://www.opengis.net/def/crs/EPSG/0/{code}"


def build_header(gdf: gpd.GeoDataFrame, scale: List[float]) -> Tuple[dict, List[float]]:
    minx = float(gdf["centroid_x"].min())
    miny = float(gdf["centroid_y"].min())
    maxx = float(gdf["centroid_x"].max())
    maxy = float(gdf["centroid_y"].max())
    minz = float(gdf["ground_z"].min())
    maxz = float((gdf["ground_z"] + gdf["tree_height"]).max())
    translate = [minx, miny, minz]
    header = {
        "type": "CityJSON",
        "version": "2.0",
        "transform": {"scale": scale, "translate": translate},
        "metadata": {
            "referenceSystem": epsg_uri(gdf),
            "geographicalExtent": [minx, miny, minz, maxx, maxy, maxz],
        },
        "CityObjects": {},
        "vertices": [],
    }
    return header, translate


def quantize(actual: Vec3, scale: List[float], translate: List[float]) -> List[int]:
    return [
        round((actual[0] - translate[0]) / scale[0]),
        round((actual[1] - translate[1]) / scale[1]),
        round((actual[2] - translate[2]) / scale[2]),
    ]


def build_feature(
    row,
    template_verts: List[Vec3],
    template_faces: List[List[int]],
    xy_radius: float,
    height: float,
    scale: List[float],
    translate: List[float],
) -> dict:
    cx = float(row["centroid_x"])
    cy = float(row["centroid_y"])
    gz = float(row["ground_z"])
    th = float(row["tree_height"])
    ca = float(row["crown_area"])
    radius = math.sqrt(ca / math.pi)
    sx = sy = radius / xy_radius
    sz = th / height

    quant_verts = [
        quantize((cx + sx * v[0], cy + sy * v[1], gz + sz * v[2]), scale, translate)
        for v in template_verts
    ]
    boundaries = [[face] for face in template_faces]
    obj_id = f"tree-{row['treeID']}"
    return {
        "type": "CityJSONFeature",
        "id": obj_id,
        "CityObjects": {
            obj_id: {
                "type": "SolitaryVegetationObject",
                "attributes": {
                    "tree_height": th,
                    "crown_area": ca,
                    "ground_z": gz,
                },
                "geometry": [
                    {
                        "type": "MultiSurface",
                        "lod": "2",
                        "boundaries": boundaries,
                    }
                ],
            }
        },
        "vertices": quant_verts,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--template", type=Path, required=True, help="Path to OBJ tree template.")
    p.add_argument("--crowns", type=Path, required=True, help="Path to tree-crowns GeoPackage.")
    p.add_argument("--out", type=Path, default=None, help="Output CityJSONSeq path (default: <crowns-stem>-trees.city.jsonl).")
    p.add_argument(
        "--simplify",
        default="none",
        help="Template simplification: none | pyramid | hull | ratio:<0..1>. Default none.",
    )
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    ns = parse_args(argv)
    if not ns.template.exists():
        raise SystemExit(f"Template not found: {ns.template}")
    if not ns.crowns.exists():
        raise SystemExit(f"Crowns file not found: {ns.crowns}")
    out_path: Path = ns.out or ns.crowns.with_name(ns.crowns.stem + "-trees.city.jsonl")

    raw_verts, raw_faces = parse_obj(ns.template)
    n_v0, n_f0 = len(raw_verts), len(raw_faces)
    s_verts, s_faces = simplify_template(raw_verts, raw_faces, ns.simplify)
    print(f"Template: {n_v0} verts, {n_f0} faces -> simplify={ns.simplify} -> {len(s_verts)} verts, {len(s_faces)} faces")

    template_verts, xy_radius, height = normalize_template(s_verts)

    gdf = read_crowns(ns.crowns)
    print(f"Crowns: {len(gdf)} rows, CRS={gdf.crs}")

    scale = [0.001, 0.001, 0.001]
    header, translate = build_header(gdf, scale)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(header) + "\n")
        for _, row in gdf.iterrows():
            feature = build_feature(row, template_verts, s_faces, xy_radius, height, scale, translate)
            f.write(json.dumps(feature) + "\n")
            count += 1

    print(f"Wrote {count} features to {out_path}")


if __name__ == "__main__":
    main()
