# Copyright (C) 2026  EMBL/Maximilian Dreimann, EMBL/Svetlana Dodonova
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Fiber assembly (original pipeline functions, verbatim): position-only straightness-based branch resolution,
union-find fiber assembly, and chain ordering. Branch resolution never uses particle orientation."""
import numpy as np
import pandas as pd
from itertools import combinations
from collections import defaultdict



def vertex_angle(pos_center, pos_a, pos_b):
    va = pos_a - pos_center
    vb = pos_b - pos_center
    cos_a = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb))
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))


def resolve_by_straightness(graph_df, positions):
    neighbors = defaultdict(list)  # node -> list of (neighbor, dist)
    for _, row in graph_df.iterrows():
        i, j, d = int(row["i"]), int(row["j"]), row["end_distance"]
        neighbors[i].append((j, d))
        neighbors[j].append((i, d))

    chosen_edges = defaultdict(set)  # node -> set of chosen neighbor ids
    for node, nbrs in neighbors.items():
        if len(nbrs) <= 2:
            for nbr, d in nbrs:
                chosen_edges[node].add(nbr)
            continue
        # degree >=3: find pair with vertex angle closest to 180
        best_pair = None
        best_angle_dev = 1e9
        for (a, da), (b, db) in combinations(nbrs, 2):
            ang = vertex_angle(positions[node], positions[a], positions[b])
            dev = abs(180 - ang)
            if dev < best_angle_dev:
                best_angle_dev = dev
                best_pair = (a, b)
        chosen_edges[node].add(best_pair[0])
        chosen_edges[node].add(best_pair[1])

    # keep edge only if BOTH endpoints chose each other
    kept = []
    seen = set()
    for _, row in graph_df.iterrows():
        i, j, d = int(row["i"]), int(row["j"]), row["end_distance"]
        key = tuple(sorted((i, j)))
        if key in seen:
            continue
        seen.add(key)
        if j in chosen_edges[i] and i in chosen_edges[j]:
            kept.append((i, j, d))

    return pd.DataFrame(kept, columns=["i", "j", "end_distance"])


def get_fibers(kept_df):
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for _, row in kept_df.iterrows():
        union(int(row["i"]), int(row["j"]))

    fibers = defaultdict(set)
    for n in parent:
        fibers[find(n)].add(n)
    return fibers






def order_chain(fiber_members, edges_df):
    """Order a simple chain (each node degree<=2) from one end to the other."""
    adj = defaultdict(list)
    for _, r in edges_df.iterrows():
        i, j = int(r["i"]), int(r["j"])
        if i in fiber_members and j in fiber_members:
            adj[i].append(j)
            adj[j].append(i)
    if len(fiber_members) == 1:
        return list(fiber_members)
    ends = [n for n in fiber_members if len(adj[n]) <= 1]
    start = ends[0] if ends else next(iter(fiber_members))
    order = [start]
    visited = {start}
    cur = start
    while True:
        nxts = [n for n in adj[cur] if n not in visited]
        if not nxts:
            break
        nxt = nxts[0]
        order.append(nxt)
        visited.add(nxt)
        cur = nxt
    return order
