#!/usr/bin/env python3
"""Diagnose non-manifold vs boundary edge issues."""
import json
from collections import Counter

def main():
    path = "/mnt/d/lidar/IGN/buildex_urban2/output.city.jsonl/output.city.jsonl"

    only_boundary = 0
    only_nonmanifold = 0
    both = 0
    watertight = 0
    total = 0
    nm_edge_types = Counter()

    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("type") != "CityJSONFeature":
                continue
            for co_id, co in obj["CityObjects"].items():
                if co["type"] not in ("Building", "BuildingPart"):
                    continue
                for geom in co.get("geometry", []):
                    if geom["type"] != "Solid":
                        continue
                    total += 1

                    sem_surfaces = geom.get("semantics", {}).get("surfaces", [])
                    sem_values = geom.get("semantics", {}).get("values", [[]])[0]

                    edge_count = Counter()
                    edge_to_type = {}

                    for shell in geom["boundaries"]:
                        for fi, face_rings in enumerate(shell):
                            sem_idx = sem_values[fi] if fi < len(sem_values) and sem_values[fi] is not None else None
                            sem_type = sem_surfaces[sem_idx]["type"] if sem_idx is not None and sem_idx < len(sem_surfaces) else "?"
                            if not face_rings:
                                continue
                            for ring in face_rings:
                                if len(ring) < 3:
                                    continue
                                for i in range(len(ring)):
                                    v1 = ring[i]
                                    v2 = ring[(i + 1) % len(ring)]
                                    edge_count[(v1, v2)] += 1
                                    edge_to_type[(v1, v2)] = sem_type

                    has_boundary = False
                    has_nm = False
                    checked = set()
                    for (v1, v2), count in edge_count.items():
                        if (v1, v2) in checked:
                            continue
                        fwd = count
                        rev = edge_count.get((v2, v1), 0)
                        checked.add((v1, v2))
                        checked.add((v2, v1))
                        if fwd == 1 and rev == 1:
                            pass
                        elif fwd == 0 or rev == 0:
                            has_boundary = True
                        else:
                            has_nm = True
                            nm_edge_types[edge_to_type.get((v1, v2), "?")] += 1

                    if not has_boundary and not has_nm:
                        watertight += 1
                    elif has_boundary and not has_nm:
                        only_boundary += 1
                    elif not has_boundary and has_nm:
                        only_nonmanifold += 1
                    else:
                        both += 1

    print(f"Total: {total}")
    print(f"Watertight: {watertight} ({100*watertight/total:.1f}%)")
    print(f"Only boundary edges: {only_boundary}")
    print(f"Only non-manifold: {only_nonmanifold}")
    print(f"Both: {both}")
    print(f"\nNon-manifold edges by surface type:")
    for t, c in nm_edge_types.most_common():
        print(f"  {t}: {c}")

if __name__ == "__main__":
    main()
