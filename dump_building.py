#!/usr/bin/env python3
"""Dump edge topology for a specific building."""
import json
import sys
from collections import Counter

def main():
    path = "/mnt/d/lidar/IGN/buildex_urban2/output.city.jsonl/output.city.jsonl"
    target = sys.argv[1] if len(sys.argv) > 1 else "1"

    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("type") != "CityJSONFeature" or obj["id"] != target:
                continue

            verts = obj["vertices"]
            print(f"Building {target}: {len(verts)} vertices")

            for co_id, co in obj["CityObjects"].items():
                for geom in co.get("geometry", []):
                    if geom["type"] != "Solid":
                        continue

                    sem_surfaces = geom.get("semantics", {}).get("surfaces", [])
                    sem_values = geom.get("semantics", {}).get("values", [[]])[0]

                    edge_count = Counter()
                    edge_to_face = {}

                    for shell in geom["boundaries"]:
                        for fi, face_rings in enumerate(shell):
                            sem_idx = sem_values[fi] if fi < len(sem_values) and sem_values[fi] is not None else None
                            sem_type = sem_surfaces[sem_idx]["type"] if sem_idx is not None else "?"

                            outer = face_rings[0] if face_rings else []
                            print(f"  Face {fi} ({sem_type}): {len(outer)} vertices, indices={outer[:10]}{'...' if len(outer) > 10 else ''}")

                            for i in range(len(outer)):
                                v1 = outer[i]
                                v2 = outer[(i + 1) % len(outer)]
                                edge_count[(v1, v2)] += 1
                                edge_to_face[(v1, v2)] = (fi, sem_type)

                    print(f"\nTotal directed edges: {len(edge_count)}")
                    unpaired = []
                    for (v1, v2), count in edge_count.items():
                        rev = edge_count.get((v2, v1), 0)
                        if count == 1 and rev == 1:
                            continue
                        if count == 1 and rev == 0:
                            fi, stype = edge_to_face[(v1, v2)]
                            z1 = verts[v1][2] * 0.001 if v1 < len(verts) else "?"
                            z2 = verts[v2][2] * 0.001 if v2 < len(verts) else "?"
                            unpaired.append((v1, v2, fi, stype, z1, z2))

                    print(f"Unpaired boundary edges: {len(unpaired)}")
                    for v1, v2, fi, stype, z1, z2 in sorted(unpaired, key=lambda x: x[3]):
                        print(f"  ({v1}→{v2}) face={fi} type={stype} z1={z1:.3f} z2={z2:.3f}")
            return
    print(f"Building {target} not found")

if __name__ == "__main__":
    main()
