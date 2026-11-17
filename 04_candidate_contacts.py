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
"""Step 4. Candidate contact graph.

Within each tomogram, every particle pair with centre-to-centre distance < CANDIDATE_CUTOFF (450 A, a generous
pre-filter) is a candidate; its end-to-end (cap-disk) distance is computed with lib/geometry.py (n_iter = 60).
Duplicates (step 3) are NOT removed here; later steps drop them. Each tomogram is saved separately as it finishes, so
an interrupted run resumes where it stopped (about 5 min per dataset on one CPU core).

Outputs: contacts_{wt,del}.csv  (columns i, j, tomogram, end_distance [Angstrom]; i/j = particle index from step 1)
"""
import os, time, numpy as np, pandas as pd
from scipy.spatial import cKDTree
from common import get_args
from lib.classes import RADIUS, CANDIDATE_CUTOFF, WT_TOMOGRAMS, DEL_TOMOGRAMS
from lib.geometry import end_to_end_distance


def contacts_one_tomogram(pos, z, L, idx):
    rows = []
    for a, b in cKDTree(pos[idx]).query_pairs(r=CANDIDATE_CUTOFF, output_type="ndarray"):
        i, j = int(idx[a]), int(idx[b])
        d = end_to_end_distance(pos[i] - (L[i] / 2) * z[i], z[i], L[i], RADIUS,
                                pos[j] - (L[j] / 2) * z[j], z[j], L[j], RADIUS)
        rows.append((i, j, d))
    return rows


if __name__ == "__main__":
    a = get_args(lambda ap: ap.add_argument("--dataset", choices=["wt", "del", "both"], default="both"), __doc__)
    for tag, tomos in [("wt", WT_TOMOGRAMS), ("del", DEL_TOMOGRAMS)]:
        if a.dataset not in (tag, "both"): continue
        df = pd.read_pickle(f"{a.out_dir}/particles_{tag}.pkl"); z = np.load(f"{a.out_dir}/z_lab_{tag}.npy"); L = np.load(f"{a.out_dir}/lengths_{tag}.npy")
        pos, tomo = df[["x", "y", "z"]].to_numpy(), df["tomogram"].to_numpy()
        parts = []
        for k, t in enumerate(tomos, 1):
            f = f"{a.out_dir}/contacts_{tag}_tomo{k}.csv"
            if not os.path.exists(f):
                t0 = time.time(); rows = contacts_one_tomogram(pos, z, L, np.where(tomo == t)[0])
                pd.DataFrame(rows, columns=["i", "j", "end_distance"]).assign(tomogram=t)[["i", "j", "tomogram", "end_distance"]].to_csv(f, index=False)
                print(f"  {tag} tomogram_{k}: {len(rows)} candidate pairs ({time.time() - t0:.0f} s)", flush=True)
            parts.append(pd.read_csv(f))
        pd.concat(parts, ignore_index=True).to_csv(f"{a.out_dir}/contacts_{tag}.csv", index=False)
        print(f"{tag}: {sum(len(p) for p in parts)} candidate pairs -> contacts_{tag}.csv")
