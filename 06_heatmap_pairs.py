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
"""Step 6. Neighbour pairs and relative orientation difference (Fig. 3e,f).

Pairs = duplicate-filtered candidate pairs with end-to-end distance <= 3 nm (30 A). To avoid computing angles at branch
points, every particle may keep at most two neighbours: edges are sorted by end-to-end distance (ties by index) and
accepted greedily if both particles still have fewer than two accepted edges. This rule uses distances only (it never
looks at orientation), so it cannot bias the orientation statistics. For each accepted pair:
  theta = arccos(z_i . z_j)  (0-180 deg; z = superhelical axis, step 1)   centre-to-centre distance in nm (and 4 A-rounded).

Outputs: heatmap_pairs_{wt,del}.csv
"""
import numpy as np, pandas as pd
from collections import Counter
from common import get_args
from lib.classes import CONTACT_CUTOFF, PIXEL_NM


def greedy_two_neighbour_edges(pairs):
    degree, kept = Counter(), []
    for d, i, j in sorted(zip(pairs["end_distance"], pairs["i"], pairs["j"])):
        i, j = int(i), int(j)
        if degree[i] < 2 and degree[j] < 2:
            kept.append((i, j, d)); degree[i] += 1; degree[j] += 1
    return kept


if __name__ == "__main__":
    a = get_args(description=__doc__)
    for tag, label in [("wt", "KU216"), ("del", "DeltaB")]:
        df = pd.read_pickle(f"{a.out_dir}/particles_{tag}.pkl"); z = np.load(f"{a.out_dir}/z_lab_{tag}.npy"); pos = df[["x", "y", "z"]].to_numpy()
        rm = set(np.load(f"{a.out_dir}/duplicates_{tag}.npy").tolist())
        c = pd.read_csv(f"{a.out_dir}/contacts_{tag}.csv"); c = c[~(c["i"].isin(rm) | c["j"].isin(rm))]
        pool = c[c["end_distance"] <= CONTACT_CUTOFF]
        edges = greedy_two_neighbour_edges(pool)
        rows = []
        for i, j, d in edges:
            theta = np.degrees(np.arccos(np.clip(np.dot(z[i], z[j]), -1, 1))); cd = np.linalg.norm(pos[i] - pos[j]) / 10.0
            rows.append((label, i, j, df["tomogram"].iloc[i], d / 10.0, theta, cd, round(np.round(cd / PIXEL_NM) * PIXEL_NM, 1)))
        out = pd.DataFrame(rows, columns=["dataset", "particle_i", "particle_j", "tomogram", "end_to_end_distance_nm", "theta_deg", "center_to_center_distance_nm", "rounded_4A_nm"])
        out.to_csv(f"{a.out_dir}/heatmap_pairs_{tag}.csv", index=False)
        print(f"{label}: {len(pool)} pairs within 3 nm (before branch handling) -> {len(out)} neighbour pairs (max 2 per particle)")
