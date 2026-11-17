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
"""Step 7. Fibers and fiber-level statistics (Fig. 3h,i; Supp. Fig. 4e,f).

* Connected pairs: candidate pairs (duplicates removed) with end-to-end distance <= 3 nm (30 A). For this step the
  distances are evaluated with the ORIGINAL geometry implementation (lib/geometry_original.py) that produced the
  published fiber table; to save time only candidates within 40 A according to lib/geometry.py (step 4) are
  re-evaluated (the two implementations differ by up to 1.3 A, see README).
* Branch resolution: a particle with >2 neighbours keeps the two whose directions are straightest (vertex angle
  closest to 180 deg, from particle POSITIONS only; orientation is never used). An edge survives only if both of its
  particles chose it (lib/fibers.py: resolve_by_straightness). Fibers = connected components (union-find), >= 2 particles.
* Per fiber: contour_nm = (sum of particle lengths along the chain + sum of end-to-end gaps)/10 [fiber end-to-end
  distance]; total_bp = 30 bp x sum of dimers per particle [DNA condensed]; density = total_bp / contour_nm.
  rounded_4A_nm = contour length rounded to the nearest 4 A.

Outputs: fiber_edges_{wt,del}.csv, fiber_stats_{wt,del}.csv (rows in assembly order; keep this order - the bootstrap in step 9 resamples by position)
"""
import numpy as np, pandas as pd
from collections import defaultdict
from common import get_args
from lib.classes import RADIUS, CONTACT_CUTOFF, BP_PER_DIMER, PIXEL_NM, WT_CLASS_TO_N, DEL_CLASS_TO_N
from lib.geometry_original import end_to_end_distance as e2e_original
from lib.fibers import resolve_by_straightness, get_fibers, order_chain

PRESCREEN = 40.0   # Angstrom (margin over the 30 A cutoff for the re-evaluation)


if __name__ == "__main__":
    a = get_args(description=__doc__)
    for tag, label, c2n in [("wt", "KU216", WT_CLASS_TO_N), ("del", "DeltaB", DEL_CLASS_TO_N)]:
        df = pd.read_pickle(f"{a.out_dir}/particles_{tag}.pkl"); z = np.load(f"{a.out_dir}/z_lab_{tag}.npy"); L = np.load(f"{a.out_dir}/lengths_{tag}.npy")
        pos, classes, tomo = df[["x", "y", "z"]].to_numpy(), df["class"].to_numpy(), df["tomogram"].to_numpy()
        rm = set(np.load(f"{a.out_dir}/duplicates_{tag}.npy").tolist())
        c = pd.read_csv(f"{a.out_dir}/contacts_{tag}.csv"); c = c[~(c["i"].isin(rm) | c["j"].isin(rm))]
        cand = c[c["end_distance"] <= PRESCREEN]
        dist = [e2e_original(pos[i] - (L[i] / 2) * z[i], z[i], L[i], RADIUS, pos[j] - (L[j] / 2) * z[j], z[j], L[j], RADIUS) for i, j in zip(cand["i"], cand["j"])]
        pairs = pd.DataFrame({"i": cand["i"].to_numpy(), "j": cand["j"].to_numpy(), "end_distance": dist})
        pairs = pairs[pairs["end_distance"] <= CONTACT_CUTOFF].reset_index(drop=True)
        edges = resolve_by_straightness(pairs, pos); edges.to_csv(f"{a.out_dir}/fiber_edges_{tag}.csv", index=False)
        deg = defaultdict(int)
        for i, j in zip(edges["i"], edges["j"]): deg[int(i)] += 1; deg[int(j)] += 1
        assert max(deg.values()) <= 2
        fibers = get_fibers(edges); edist = {frozenset((int(i), int(j))): d for i, j, d in zip(edges["i"], edges["j"], edges["end_distance"])}
        recs = []
        for members in fibers.values():
            if len(members) < 2: continue
            order = order_chain(members, edges)
            if len(order) != len(members): continue
            contour = (sum(L[p] for p in order) + sum(edist[frozenset((order[k], order[k + 1]))] for k in range(len(order) - 1))) / 10.0
            bp = sum(c2n.get(classes[p], 0) * BP_PER_DIMER for p in order)
            recs.append({"dataset": label, "tomogram": tomo[order[0]], "n_particles": len(order), "contour_nm": contour, "total_bp": bp, "density": bp / contour})
        fs = pd.DataFrame(recs); fs["rounded_4A_nm"] = np.round(np.round(fs["contour_nm"] / PIXEL_NM) * PIXEL_NM, 1)
        fs.to_csv(f"{a.out_dir}/fiber_stats_{tag}.csv", index=False)
        print(f"{label}: {len(pairs)} pairs <= 3 nm -> {len(edges)} fiber edges -> {len(fs)} fibers; median contour {fs.contour_nm.median():.2f} nm "
              f"(4 A-rounded {fs.rounded_4A_nm.median():.1f}), median DNA condensed {fs.total_bp.median():.0f} bp")
