#!/usr/bin/env python3
"""Compare two CityJSONL files to understand GLB size differences."""

import json
import sys
from collections import defaultdict, Counter

def count_faces_in_boundaries(boundaries, geom_type):
    """Count faces in a geometry's boundaries."""
    count = 0
    if geom_type == "Solid":
        # Solid: boundaries = [ shell [ surface [ ring [idx...] ] ] ]
        for shell in boundaries:
            for surface in shell:
                count += 1  # each surface is a face (outer ring)
    elif geom_type in ("MultiSurface", "CompositeSurface"):
        # MultiSurface: boundaries = [ surface [ ring [idx...] ] ]
        for surface in boundaries:
            count += 1
    return count

def count_vertices_in_boundaries(boundaries, geom_type):
    """Collect all vertex indices referenced in boundaries."""
    indices = set()
    if geom_type == "Solid":
        for shell in boundaries:
            for surface in shell:
                for ring in surface:
                    for idx in ring:
                        indices.add(idx)
    elif geom_type in ("MultiSurface", "CompositeSurface"):
        for surface in boundaries:
            for ring in surface:
                for idx in ring:
                    indices.add(idx)
    return indices

def analyze_file(filepath, label):
    stats = {
        "label": label,
        "filepath": filepath,
        "total_features": 0,
        "type_counts": Counter(),
        "total_vertices": 0,
        "total_faces": 0,
        "geom_types": Counter(),
        "lods": Counter(),
        "bp_vertex_counts": [],
        "bp_face_counts": [],
        "bp_names": [],
        "feature_vertex_lists": [],  # list of (num_vertices_in_feature_array,)
        "duplicate_vertex_info": [],
    }

    with open(filepath, "r") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            # Skip header
            if obj.get("type") == "CityJSON":
                continue

            stats["total_features"] += 1
            vertices = obj.get("vertices", [])
            num_verts_in_array = len(vertices)
            stats["feature_vertex_lists"].append(num_verts_in_array)

            # Check for duplicate vertices in this feature
            if num_verts_in_array > 0:
                vert_tuples = [tuple(v) for v in vertices]
                unique_verts = set(vert_tuples)
                dup_count = num_verts_in_array - len(unique_verts)
                if dup_count > 0:
                    stats["duplicate_vertex_info"].append((stats["total_features"], num_verts_in_array, len(unique_verts), dup_count))

            city_objects = obj.get("CityObjects", {})
            for co_name, co in city_objects.items():
                co_type = co.get("type", "Unknown")
                stats["type_counts"][co_type] += 1

                for geom in co.get("geometry", []):
                    geom_type = geom.get("type", "Unknown")
                    lod = geom.get("lod", "?")
                    stats["geom_types"][geom_type] += 1
                    stats["lods"][str(lod)] += 1

                    boundaries = geom.get("boundaries", [])
                    face_count = count_faces_in_boundaries(boundaries, geom_type)
                    vert_indices = count_vertices_in_boundaries(boundaries, geom_type)

                    stats["total_faces"] += face_count
                    stats["total_vertices"] += len(vert_indices)

                    if co_type == "BuildingPart":
                        stats["bp_vertex_counts"].append(len(vert_indices))
                        stats["bp_face_counts"].append(face_count)
                        stats["bp_names"].append(co_name)

    return stats

def bucket(values, thresholds=[10, 50, 100, 500]):
    """Bucket values into ranges."""
    buckets = defaultdict(int)
    labels = []
    prev = 0
    for t in thresholds:
        labels.append(f"{prev}-{t-1}")
        prev = t
    labels.append(f"{thresholds[-1]}+")

    for v in values:
        placed = False
        prev = 0
        for i, t in enumerate(thresholds):
            if v < t:
                buckets[labels[i]] += 1
                placed = True
                break
            prev = t
        if not placed:
            buckets[labels[-1]] += 1
    return [(l, buckets[l]) for l in labels]

def print_comparison(s1, s2):
    w = 30  # column width

    def row(label, v1, v2):
        print(f"  {label:<40} {str(v1):>{w}} {str(v2):>{w}}")

    def header():
        print(f"  {'':40} {s1['label']:>{w}} {s2['label']:>{w}}")
        print(f"  {'='*40} {'='*w} {'='*w}")

    print("\n" + "="*104)
    print("  CityJSONL COMPARISON")
    print("="*104)

    header()
    row("Total CityJSONFeatures", s1["total_features"], s2["total_features"])
    print()

    # Type breakdown
    all_types = sorted(set(list(s1["type_counts"].keys()) + list(s2["type_counts"].keys())))
    for t in all_types:
        row(f"  CityObjects: {t}", s1["type_counts"].get(t, 0), s2["type_counts"].get(t, 0))
    row("  CityObjects: TOTAL",
        sum(s1["type_counts"].values()), sum(s2["type_counts"].values()))
    print()

    # Geometry types
    row("Geometry types:", "", "")
    all_gt = sorted(set(list(s1["geom_types"].keys()) + list(s2["geom_types"].keys())))
    for gt in all_gt:
        row(f"  {gt}", s1["geom_types"].get(gt, 0), s2["geom_types"].get(gt, 0))
    print()

    # LoDs
    row("LoDs:", "", "")
    all_lods = sorted(set(list(s1["lods"].keys()) + list(s2["lods"].keys())))
    for lod in all_lods:
        row(f"  LoD {lod}", s1["lods"].get(lod, 0), s2["lods"].get(lod, 0))
    print()

    # Totals
    row("Total referenced vertices", s1["total_vertices"], s2["total_vertices"])
    row("Total faces", s1["total_faces"], s2["total_faces"])
    ratio_v = s1["total_vertices"] / max(s2["total_vertices"], 1)
    ratio_f = s1["total_faces"] / max(s2["total_faces"], 1)
    row("Vertex ratio (buildex/roofer-c)", f"{ratio_v:.2f}x", "1.00x")
    row("Face ratio (buildex/roofer-c)", f"{ratio_f:.2f}x", "1.00x")
    print()

    # Per-BuildingPart stats
    def avg(lst):
        return f"{sum(lst)/len(lst):.1f}" if lst else "N/A"
    def med(lst):
        if not lst: return "N/A"
        s = sorted(lst)
        n = len(s)
        return f"{s[n//2]:.0f}"
    def mn(lst):
        return f"{min(lst)}" if lst else "N/A"
    def mx(lst):
        return f"{max(lst)}" if lst else "N/A"

    row("BuildingPart count", len(s1["bp_vertex_counts"]), len(s2["bp_vertex_counts"]))
    row("Avg vertices/BuildingPart", avg(s1["bp_vertex_counts"]), avg(s2["bp_vertex_counts"]))
    row("Median vertices/BuildingPart", med(s1["bp_vertex_counts"]), med(s2["bp_vertex_counts"]))
    row("Min vertices/BuildingPart", mn(s1["bp_vertex_counts"]), mn(s2["bp_vertex_counts"]))
    row("Max vertices/BuildingPart", mx(s1["bp_vertex_counts"]), mx(s2["bp_vertex_counts"]))
    print()
    row("Avg faces/BuildingPart", avg(s1["bp_face_counts"]), avg(s2["bp_face_counts"]))
    row("Median faces/BuildingPart", med(s1["bp_face_counts"]), med(s2["bp_face_counts"]))
    row("Min faces/BuildingPart", mn(s1["bp_face_counts"]), mn(s2["bp_face_counts"]))
    row("Max faces/BuildingPart", mx(s1["bp_face_counts"]), mx(s2["bp_face_counts"]))
    print()

    # Distributions
    print(f"\n  VERTEX DISTRIBUTION (per BuildingPart)")
    print(f"  {'Range':<20} {s1['label']:>{w}} {s2['label']:>{w}}")
    print(f"  {'-'*20} {'-'*w} {'-'*w}")
    b1 = bucket(s1["bp_vertex_counts"])
    b2 = bucket(s2["bp_vertex_counts"])
    for (l1, c1), (l2, c2) in zip(b1, b2):
        pct1 = f"{c1} ({100*c1/max(len(s1['bp_vertex_counts']),1):.1f}%)"
        pct2 = f"{c2} ({100*c2/max(len(s2['bp_vertex_counts']),1):.1f}%)"
        print(f"  {l1:<20} {pct1:>{w}} {pct2:>{w}}")

    print(f"\n  FACE DISTRIBUTION (per BuildingPart)")
    print(f"  {'Range':<20} {s1['label']:>{w}} {s2['label']:>{w}}")
    print(f"  {'-'*20} {'-'*w} {'-'*w}")
    b1 = bucket(s1["bp_face_counts"])
    b2 = bucket(s2["bp_face_counts"])
    for (l1, c1), (l2, c2) in zip(b1, b2):
        pct1 = f"{c1} ({100*c1/max(len(s1['bp_face_counts']),1):.1f}%)"
        pct2 = f"{c2} ({100*c2/max(len(s2['bp_face_counts']),1):.1f}%)"
        print(f"  {l1:<20} {pct1:>{w}} {pct2:>{w}}")

    # Sample BuildingParts
    print(f"\n  SAMPLE BuildingParts (first 5, middle 5, last 5)")
    for s in [s1, s2]:
        print(f"\n  --- {s['label']} ---")
        n = len(s["bp_vertex_counts"])
        if n == 0:
            print("    No BuildingParts found")
            continue
        indices = list(range(min(5, n)))
        if n > 10:
            mid = n // 2
            indices += list(range(mid-2, mid+3))
        indices += list(range(max(n-5, 0), n))
        indices = sorted(set(i for i in indices if 0 <= i < n))
        print(f"    {'Index':<10} {'Name':<25} {'Vertices':>10} {'Faces':>10}")
        for i in indices:
            print(f"    {i:<10} {s['bp_names'][i]:<25} {s['bp_vertex_counts'][i]:>10} {s['bp_face_counts'][i]:>10}")

    # Duplicate vertex analysis
    print(f"\n  DUPLICATE VERTEX ANALYSIS")
    for s in [s1, s2]:
        dups = s["duplicate_vertex_info"]
        if not dups:
            print(f"  {s['label']}: No duplicate vertices found in any feature")
        else:
            total_dup = sum(d[3] for d in dups)
            total_all = sum(d[1] for d in dups)
            features_with_dups = len(dups)
            print(f"  {s['label']}:")
            print(f"    Features with duplicate vertices: {features_with_dups}/{s['total_features']}")
            print(f"    Total duplicate vertices: {total_dup} (out of {total_all} in those features)")
            # Show worst offenders
            worst = sorted(dups, key=lambda x: x[3], reverse=True)[:5]
            print(f"    Top 5 worst offenders (feature#, total, unique, duplicates):")
            for feat_no, total, unique, dup in worst:
                print(f"      Feature {feat_no}: {total} total, {unique} unique, {dup} duplicates ({100*dup/total:.0f}%)")

    # Total vertex array sizes (sum of all feature vertex arrays)
    total_va_1 = sum(s1["feature_vertex_lists"])
    total_va_2 = sum(s2["feature_vertex_lists"])
    print(f"\n  TOTAL VERTEX ARRAY SIZE (sum of all feature vertex arrays)")
    row("Total vertex array entries", total_va_1, total_va_2)
    row("Ratio", f"{total_va_1/max(total_va_2,1):.2f}x", "1.00x")

    print()


if __name__ == "__main__":
    file1 = "/mnt/d/lidar/IGN/buildex_urban2/output.city.jsonl"
    file2 = "/mnt/d/lidar/IGN/roofer_c_urban2/891849_6246888.city.jsonl"

    print(f"Analyzing {file1} ...")
    s1 = analyze_file(file1, "buildex")
    print(f"Analyzing {file2} ...")
    s2 = analyze_file(file2, "roofer-c")

    print_comparison(s1, s2)
