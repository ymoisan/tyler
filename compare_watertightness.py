#!/usr/bin/env python3
"""Compare watertightness of two CityJSONL files (buildex vs roofer-c).

For each building's Solid shell, counts boundary edges (unpaired directed
edges) and non-manifold edges (>2 faces sharing an edge).  A watertight
solid has 0 boundary edges and 0 non-manifold edges.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def parse_buildings(path: str) -> dict:
    """Return {building_id: [list of Solid boundary arrays]}."""
    buildings: dict = {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("type") != "CityJSONFeature":
                continue
            fid = obj["id"]
            for co_id, co in obj["CityObjects"].items():
                if co["type"] not in ("Building", "BuildingPart"):
                    continue
                for geom in co.get("geometry", []):
                    if geom["type"] == "Solid":
                        # boundaries: list of shells, each shell is list of faces
                        # each face is list of rings (outer + holes)
                        bid = fid
                        if bid not in buildings:
                            buildings[bid] = []
                        buildings[bid].append(geom["boundaries"])
    return buildings


def check_watertightness(solid_boundaries) -> dict:
    """Check a single Solid's watertightness.

    Returns dict with boundary_edges, non_manifold_edges, total_faces.
    """
    edge_count: Counter = Counter()
    total_faces = 0

    for shell in solid_boundaries:
        for face_rings in shell:
            if not face_rings:
                continue
            total_faces += 1
            # Build directed edges from ALL rings (outer + holes)
            for ring in face_rings:
                if len(ring) < 3:
                    continue
                for i in range(len(ring)):
                    v1 = ring[i]
                    v2 = ring[(i + 1) % len(ring)]
                    edge_count[(v1, v2)] += 1

    # Check each directed edge for its reverse
    boundary_edges = 0
    non_manifold_edges = 0
    checked = set()

    for (v1, v2), count in edge_count.items():
        if (v1, v2) in checked:
            continue
        fwd = count
        rev = edge_count.get((v2, v1), 0)
        checked.add((v1, v2))
        checked.add((v2, v1))

        if fwd == 1 and rev == 1:
            pass  # perfect manifold edge
        elif fwd == 0 or rev == 0:
            boundary_edges += max(fwd, rev)
        else:
            non_manifold_edges += 1

    return {
        "boundary_edges": boundary_edges,
        "non_manifold_edges": non_manifold_edges,
        "total_faces": total_faces,
        "watertight": boundary_edges == 0 and non_manifold_edges == 0,
    }


def main():
    buildex_path = "/mnt/d/lidar/IGN/buildex_urban2/output.city.jsonl/output.city.jsonl"
    roofer_path = "/mnt/d/lidar/IGN/roofer_c_urban2/891849_6246888.city.jsonl"

    print("Parsing buildex CityJSONL...")
    buildex = parse_buildings(buildex_path)
    print(f"  Found {len(buildex)} buildings")

    print("Parsing roofer-c CityJSONL...")
    roofer = parse_buildings(roofer_path)
    print(f"  Found {len(roofer)} buildings")

    # Get union of building IDs
    all_ids = sorted(set(buildex.keys()) | set(roofer.keys()), key=lambda x: int(x) if x.isdigit() else x)

    results = []
    for bid in all_ids:
        row = {"id": bid}
        for label, data in [("buildex", buildex), ("roofer_c", roofer)]:
            if bid in data:
                # Combine all solids for this building
                combined = {"boundary_edges": 0, "non_manifold_edges": 0, "total_faces": 0, "watertight": True}
                for solid_bounds in data[bid]:
                    r = check_watertightness(solid_bounds)
                    combined["boundary_edges"] += r["boundary_edges"]
                    combined["non_manifold_edges"] += r["non_manifold_edges"]
                    combined["total_faces"] += r["total_faces"]
                    if not r["watertight"]:
                        combined["watertight"] = False
                row[f"{label}_boundary"] = combined["boundary_edges"]
                row[f"{label}_nonmanifold"] = combined["non_manifold_edges"]
                row[f"{label}_faces"] = combined["total_faces"]
                row[f"{label}_watertight"] = combined["watertight"]
            else:
                row[f"{label}_boundary"] = None
                row[f"{label}_nonmanifold"] = None
                row[f"{label}_faces"] = None
                row[f"{label}_watertight"] = None
        results.append(row)

    # Print per-building table (only non-watertight ones)
    print("\n" + "=" * 120)
    print("BUILDINGS WITH WATERTIGHTNESS ISSUES")
    print("=" * 120)
    header = f"{'ID':>6}  {'BX Bndry':>8} {'BX NM':>6} {'BX Faces':>8} {'BX WT':>6}  {'RC Bndry':>8} {'RC NM':>6} {'RC Faces':>8} {'RC WT':>6}"
    print(header)
    print("-" * 120)

    for row in results:
        bx_wt = row.get("buildex_watertight")
        rc_wt = row.get("roofer_c_watertight")
        if bx_wt is False or rc_wt is False:
            bx_b = row.get("buildex_boundary", "-")
            bx_n = row.get("buildex_nonmanifold", "-")
            bx_f = row.get("buildex_faces", "-")
            bx_w = "YES" if bx_wt else ("NO" if bx_wt is False else "-")
            rc_b = row.get("roofer_c_boundary", "-")
            rc_n = row.get("roofer_c_nonmanifold", "-")
            rc_f = row.get("roofer_c_faces", "-")
            rc_w = "YES" if rc_wt else ("NO" if rc_wt is False else "-")
            print(f"{row['id']:>6}  {bx_b!s:>8} {bx_n!s:>6} {bx_f!s:>8} {bx_w:>6}  {rc_b!s:>8} {rc_n!s:>6} {rc_f!s:>8} {rc_w:>6}")

    # Summary
    bx_total = sum(1 for r in results if r.get("buildex_watertight") is not None)
    bx_wt = sum(1 for r in results if r.get("buildex_watertight") is True)
    rc_total = sum(1 for r in results if r.get("roofer_c_watertight") is not None)
    rc_wt = sum(1 for r in results if r.get("roofer_c_watertight") is True)

    rc_ok_bx_not = sum(
        1 for r in results
        if r.get("roofer_c_watertight") is True and r.get("buildex_watertight") is False
    )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Buildex:  {bx_wt}/{bx_total} watertight ({100*bx_wt/bx_total:.1f}%)" if bx_total else "Buildex: no data")
    print(f"Roofer-c: {rc_wt}/{rc_total} watertight ({100*rc_wt/rc_total:.1f}%)" if rc_total else "Roofer-c: no data")
    print(f"Watertight in roofer-c but NOT buildex: {rc_ok_bx_not}")

    # Building 97 specifically
    print("\n" + "=" * 80)
    print("BUILDING 97 (Cathedral)")
    print("=" * 80)
    for row in results:
        if row["id"] == "97":
            for label in ["buildex", "roofer_c"]:
                b = row.get(f"{label}_boundary", "N/A")
                n = row.get(f"{label}_nonmanifold", "N/A")
                f = row.get(f"{label}_faces", "N/A")
                w = row.get(f"{label}_watertight", "N/A")
                print(f"  {label:>10}: boundary_edges={b}, non_manifold={n}, faces={f}, watertight={w}")
            break
    else:
        print("  Building 97 not found in either dataset")

    # Top 10 worst buildex buildings by boundary edges
    print("\n" + "=" * 80)
    print("TOP 20 WORST BUILDEX BUILDINGS (by boundary edges)")
    print("=" * 80)
    bx_rows = [r for r in results if r.get("buildex_boundary") is not None and r.get("buildex_boundary", 0) > 0]
    bx_rows.sort(key=lambda r: r.get("buildex_boundary", 0), reverse=True)
    for row in bx_rows[:20]:
        print(f"  Building {row['id']:>6}: {row['buildex_boundary']} boundary edges, {row['buildex_faces']} faces")


if __name__ == "__main__":
    main()
