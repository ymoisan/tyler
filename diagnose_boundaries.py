#!/usr/bin/env python3
"""Diagnose remaining boundary edges in buildex CityJSONL.

For each non-watertight building, classify boundary edges by the semantic
surface type of the face that owns them.
"""
import json
from collections import Counter, defaultdict


def main():
    path = "/mnt/d/lidar/IGN/buildex_urban2/output.city.jsonl/output.city.jsonl"

    # Category counters
    total_buildings = 0
    watertight_buildings = 0
    edge_type_counter = Counter()  # (owner_type, partner_type) → count
    buildings_by_issue = Counter()  # issue type → count

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
                    if geom["type"] != "Solid":
                        continue
                    total_buildings += 1

                    # Build face→semantic-type map
                    sem_surfaces = geom.get("semantics", {}).get("surfaces", [])
                    sem_values = geom.get("semantics", {}).get("values", [[]])[0]

                    edge_to_face_type = {}  # (v1,v2) → surface_type
                    edge_count = Counter()

                    for shell in geom["boundaries"]:
                        for fi, face_rings in enumerate(shell):
                            # Get semantic type for this face
                            sem_idx = sem_values[fi] if fi < len(sem_values) and sem_values[fi] is not None else None
                            sem_type = sem_surfaces[sem_idx]["type"] if sem_idx is not None and sem_idx < len(sem_surfaces) else "Unknown"

                            if not face_rings:
                                continue
                            for ring in face_rings:
                                if len(ring) < 3:
                                    continue
                                for i in range(len(ring)):
                                    v1 = ring[i]
                                    v2 = ring[(i + 1) % len(ring)]
                                    edge_count[(v1, v2)] += 1
                                    edge_to_face_type[(v1, v2)] = sem_type

                    # Find unpaired edges
                    boundary_edges = 0
                    issues = set()
                    for (v1, v2), count in edge_count.items():
                        rev = edge_count.get((v2, v1), 0)
                        if count == 1 and rev == 1:
                            continue  # paired
                        if count == 1 and rev == 0:
                            boundary_edges += 1
                            owner = edge_to_face_type.get((v1, v2), "?")
                            edge_type_counter[owner] += 1
                            issues.add(owner)

                    if boundary_edges == 0:
                        watertight_buildings += 1
                    else:
                        for issue in issues:
                            buildings_by_issue[issue] += 1

    print(f"Total buildings with Solid geometry: {total_buildings}")
    print(f"Watertight: {watertight_buildings} ({100*watertight_buildings/total_buildings:.1f}%)")
    print(f"\nUnpaired boundary edges by owning surface type:")
    for stype, count in edge_type_counter.most_common():
        print(f"  {stype}: {count}")
    print(f"\nBuildings with issues by surface type:")
    for stype, count in buildings_by_issue.most_common():
        print(f"  {stype}: {count} buildings")


if __name__ == "__main__":
    main()
