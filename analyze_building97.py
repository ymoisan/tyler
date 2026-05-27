#!/usr/bin/env python3
"""Analyze building 97 in two CityJSONL files to understand missing domes."""

import json
import math
import sys

BUILDEX = "/mnt/d/lidar/IGN/buildex_urban2/output.city.jsonl"
ROOFER_C = "/mnt/d/lidar/IGN/roofer_c_urban2/891849_6246888.city.jsonl"


def load_building(filepath, target_ids):
    """Load a specific building line from CityJSONL. Returns (header, city_objects_line)."""
    header = None
    with open(filepath) as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            if i == 0:
                header = d
                continue
            keys = set(d.get("CityObjects", {}).keys())
            if keys & target_ids:
                return header, d
    return header, None


def transform_vertex(v, transform):
    """Apply CityJSON transform to a vertex."""
    s = transform["scale"]
    t = transform["translate"]
    return [v[0] * s[0] + t[0], v[1] * s[1] + t[1], v[2] * s[2] + t[2]]


def count_faces(geometry):
    """Count total faces in a geometry object."""
    count = 0
    gtype = geometry.get("type")
    boundaries = geometry.get("boundaries", [])
    if gtype == "Solid":
        for shell in boundaries:
            count += len(shell)
    elif gtype in ("MultiSurface", "CompositeSurface"):
        count += len(boundaries)
    return count


def get_face_semantic_indices(geometry):
    """Get the semantic index for each face."""
    gtype = geometry.get("type")
    values = geometry.get("semantics", {}).get("values", [])
    if gtype == "Solid" and values:
        # values is [[idx, idx, ...]] for Solid (one list per shell)
        return values[0] if values else []
    return values


def get_roof_vertex_indices(geometry, surfaces):
    """Get vertex indices that belong to RoofSurface faces."""
    roof_surface_indices = set()
    for i, s in enumerate(surfaces):
        if s.get("type") == "RoofSurface":
            roof_surface_indices.add(i)

    sem_indices = get_face_semantic_indices(geometry)
    gtype = geometry.get("type")
    boundaries = geometry.get("boundaries", [])

    vertex_indices = set()
    if gtype == "Solid":
        faces = boundaries[0] if boundaries else []
    else:
        faces = boundaries

    for fi, face in enumerate(faces):
        if fi < len(sem_indices) and sem_indices[fi] in roof_surface_indices:
            for ring in face:
                for vi in ring:
                    vertex_indices.add(vi)
    return vertex_indices


def analyze_building(filepath, label, target_ids):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  File: {filepath}")
    print(f"{'='*70}")

    header, data = load_building(filepath, target_ids)
    if data is None:
        print("  Building NOT FOUND!")
        return

    transform = header["transform"]
    vertices_raw = data.get("vertices", [])
    vertices = [transform_vertex(v, transform) for v in vertices_raw]

    cos = data.get("CityObjects", {})

    # Find the Building (parent) and BuildingPart
    building_obj = None
    part_obj = None
    for k, v in cos.items():
        if v.get("type") == "Building":
            building_obj = (k, v)
        elif v.get("type") == "BuildingPart":
            part_obj = (k, v)

    # --- Building-level attributes ---
    if building_obj:
        bk, bv = building_obj
        attrs = bv.get("attributes", {})
        print(f"\n  Building ID: {bk}")
        print(f"  Type: {bv.get('type')}")
        important_attrs = [
            "n_planes", "method_used", "gc_status", "ar_status",
            "decision", "gc_score", "ar_score", "n_roof_pts",
            "roof_reason", "class", "BU_id"
        ]
        for a in important_attrs:
            if a in attrs:
                print(f"    {a}: {attrs[a]}")

    # --- BuildingPart geometry ---
    obj = part_obj if part_obj else building_obj
    if not obj:
        print("  No geometry found!")
        return

    ok, ov = obj
    print(f"\n  BuildingPart ID: {ok}")
    print(f"  Total vertices in feature: {len(vertices)}")

    for geom in ov.get("geometry", []):
        lod = geom.get("lod")
        gtype = geom.get("type")
        print(f"\n  Geometry: type={gtype}, LoD={lod}")

        nfaces = count_faces(geom)
        print(f"  Total faces: {nfaces}")

        surfaces = geom.get("semantics", {}).get("surfaces", [])
        if not surfaces:
            print("  No semantic surfaces.")
            continue

        # Count by type
        type_counts = {}
        for s in surfaces:
            st = s.get("type", "Unknown")
            type_counts[st] = type_counts.get(st, 0) + 1

        print(f"\n  Semantic surfaces by type:")
        for st, cnt in sorted(type_counts.items()):
            print(f"    {st}: {cnt}")

        # RoofSurface details
        roof_surfaces = [(i, s) for i, s in enumerate(surfaces) if s.get("type") == "RoofSurface"]
        print(f"\n  Distinct RoofSurface semantics: {len(roof_surfaces)}")
        if roof_surfaces:
            print(f"  {'#':>4}  {'Slope':>8}  {'Azimuth':>8}  Extra attrs")
            print(f"  {'---':>4}  {'-----':>8}  {'-------':>8}  -----------")
            for i, (si, s) in enumerate(roof_surfaces):
                slope = s.get("rf_slope", s.get("slope", "N/A"))
                azimuth = s.get("rf_azimuth", s.get("azimuth", "N/A"))
                extra = {k: v for k, v in s.items() if k not in ("type", "rf_slope", "rf_azimuth", "slope", "azimuth")}
                slope_str = f"{slope:8.2f}" if isinstance(slope, (int, float)) else f"{slope:>8}"
                az_str = f"{azimuth:8.2f}" if isinstance(azimuth, (int, float)) else f"{azimuth:>8}"
                print(f"  {i+1:4d}  {slope_str}  {az_str}  {extra if extra else ''}")

        # Z range of roof vertices
        roof_vis = get_roof_vertex_indices(geom, surfaces)
        if roof_vis:
            roof_zs = [vertices[vi][2] for vi in roof_vis if vi < len(vertices)]
            if roof_zs:
                z_min = min(roof_zs)
                z_max = max(roof_zs)
                print(f"\n  Roof vertex Z range: {z_min:.3f} - {z_max:.3f}  (delta = {z_max - z_min:.3f} m)")
                print(f"  Number of roof vertices: {len(roof_zs)}")

        # All vertex Z range
        all_zs = [v[2] for v in vertices]
        if all_zs:
            print(f"  All vertex Z range:  {min(all_zs):.3f} - {max(all_zs):.3f}  (delta = {max(all_zs) - min(all_zs):.3f} m)")

    return len(roof_surfaces) if 'roof_surfaces' in dir() else 0, \
           (z_max - z_min) if 'z_max' in dir() else 0


# ---- Main ----
print("=" * 70)
print("  BUILDING 97 DOME ANALYSIS")
print("=" * 70)

r1 = analyze_building(BUILDEX, "BUILDEX (buildex_urban2)", {"97", "97-0"})
r2 = analyze_building(ROOFER_C, "ROOFER-C (roofer_c_urban2)", {"97", "97-0"})

# Comparison
print(f"\n{'='*70}")
print(f"  COMPARISON")
print(f"{'='*70}")

if r1 and r2:
    n_roof_bx, dz_bx = r1
    n_roof_rc, dz_rc = r2
    print(f"\n  {'Metric':<35} {'Buildex':>10} {'Roofer-C':>10}")
    print(f"  {'-----':<35} {'------':>10} {'--------':>10}")
    print(f"  {'Distinct RoofSurface planes':<35} {n_roof_bx:>10} {n_roof_rc:>10}")
    print(f"  {'Roof Z range (m)':<35} {dz_bx:>10.3f} {dz_rc:>10.3f}")

    print(f"\n  Dome assessment:")
    print(f"  - A dome typically requires 8+ roof planes with varying slopes & azimuths.")
    if n_roof_bx < 8 and n_roof_rc >= 8:
        print(f"  - Roofer-C has {n_roof_rc} planes => dome geometry is preserved.")
        print(f"  - Buildex has only {n_roof_bx} planes => dome has been COLLAPSED/simplified.")
    elif n_roof_bx >= 8:
        print(f"  - Buildex has {n_roof_bx} planes => dome geometry appears preserved.")
    else:
        print(f"  - Neither source has 8+ planes; building may not have a dome,")
        print(f"    or both sources simplified it.")

    if dz_bx < 1.0 and dz_rc > 1.0:
        print(f"  - Buildex roof Z range ({dz_bx:.2f}m) is nearly flat => confirms dome loss.")
        print(f"  - Roofer-C roof Z range ({dz_rc:.2f}m) shows vertical variation => dome present.")
    elif dz_bx > 1.0:
        print(f"  - Buildex roof Z range ({dz_bx:.2f}m) shows some vertical variation.")
