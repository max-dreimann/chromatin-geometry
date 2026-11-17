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
"""Step 8. Random (steric-exclusion) null model for the nearest-neighbour distribution (Supp. Fig. 4a,b).

Per tomogram, the same particles (same classes, hence same lengths; duplicates removed) are re-placed at uniformly random
positions inside the bounding box of the real particles, with uniformly random orientations. Particles are placed one
after the other; a placement that overlaps (end-to-end/cylinder distance <= 0) an already placed particle is rejected
and re-drawn (up to 50 attempts; the last draw is kept if all fail). After placement, each particle's nearest-neighbour
end-to-end distance is computed exactly as for the real data. This is repeated N_RESAMPLES = 10 times per tomogram and
the results are pooled (resample index is not kept). Random seed = 1 for every tomogram (numpy default_rng).
Distances use lib/geometry.py with n_iter = 30 (overlap test and neighbour distances).

Each tomogram is saved when finished, so an interrupted run resumes (about 1-2 min per tomogram).
Outputs: null_{wt,del}_tomo{k}.npy (Angstrom), null_{wt,del}.csv
"""
import os, time, numpy as np, pandas as pd
from scipy.spatial import cKDTree
from common import get_args
from lib.classes import RADIUS, SLOPE, INTERCEPT, WT_CLASS_TO_N, DEL_CLASS_TO_N, WT_TOMOGRAMS, DEL_TOMOGRAMS, PIXEL_NM
from lib.geometry import cylinder_cylinder_distance, end_to_end_distance

N_RESAMPLES, SEED, N_ITER, MAX_ATTEMPTS = 10, 1, 30, 50


def random_axis(rng, n):
    v = rng.normal(size=(n, 3)); return v / np.linalg.norm(v, axis=1, keepdims=True)


def null_one_tomogram(pos, lengths, radius=RADIUS, n_resamples=N_RESAMPLES, seed=SEED):
    rng = np.random.default_rng(seed); n = len(pos); mins, maxs = pos.min(axis=0), pos.max(axis=0); pooled = []
    for _ in range(n_resamples):
        placed_pos, placed_axis, placed_P = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3)); tree = None
        for i in range(n):
            L = lengths[i]; ok = False
            for _attempt in range(MAX_ATTEMPTS):
                cand_pos = rng.uniform(mins, maxs); cand_axis = random_axis(rng, 1)[0]; cand_P = cand_pos - (L / 2) * cand_axis
                if i == 0: ok = True
                else:
                    cutoff = L / 2 + lengths[:i].max() / 2 + 2 * radius
                    overlap = False
                    for j in (tree.query_ball_point(cand_pos, r=cutoff) if tree is not None else []):
                        if cylinder_cylinder_distance(cand_P, cand_axis, L, radius, placed_P[j], placed_axis[j], lengths[j], radius, n_iter=N_ITER) <= 0:
                            overlap = True; break
                    ok = not overlap
                if ok: break
            placed_pos[i], placed_axis[i], placed_P[i] = cand_pos, cand_axis, cand_P; tree = cKDTree(placed_pos[:i + 1])
        ctree = cKDTree(placed_pos); cutoff = lengths.max() + 2 * radius + 100.0
        for i in range(n):
            best = np.inf
            for j in ctree.query_ball_point(placed_pos[i], r=cutoff):
                if j != i:
                    d = end_to_end_distance(placed_P[i], placed_axis[i], lengths[i], radius, placed_P[j], placed_axis[j], lengths[j], radius, n_iter=N_ITER)
                    best = min(best, d)
            if np.isfinite(best): pooled.append(best)
    return np.array(pooled)


if __name__ == "__main__":
    a = get_args(lambda ap: ap.add_argument("--dataset", choices=["wt", "del", "both"], default="both"), __doc__)
    for tag, label, tomos, c2n in [("wt", "KU216", WT_TOMOGRAMS, WT_CLASS_TO_N), ("del", "DeltaB", DEL_TOMOGRAMS, DEL_CLASS_TO_N)]:
        if a.dataset not in (tag, "both"): continue
        df = pd.read_pickle(f"{a.out_dir}/particles_{tag}.pkl"); rm = set(np.load(f"{a.out_dir}/duplicates_{tag}.npy").tolist())
        rows = []
        for k, t in enumerate(tomos, 1):
            f = f"{a.out_dir}/null_{tag}_tomo{k}.npy"
            if not os.path.exists(f):
                idx = np.array([i for i in np.where(df["tomogram"].to_numpy() == t)[0] if i not in rm]); t0 = time.time()
                lengths = INTERCEPT + SLOPE * df["class"].iloc[idx].map(c2n).to_numpy().astype(float)
                np.save(f, null_one_tomogram(df[["x", "y", "z"]].to_numpy()[idx], lengths)); print(f"  {label} tomogram_{k}: {time.time() - t0:.0f} s", flush=True)
            d = np.load(f) / 10.0; rows.append(pd.DataFrame({"dataset": label, "tomogram": f"tomogram_{k}", "null_nn_edge_to_edge_distance_nm": d, "rounded_4A_nm": np.round(np.round(d / PIXEL_NM) * PIXEL_NM, 1)}))
        pd.concat(rows, ignore_index=True).to_csv(f"{a.out_dir}/null_{tag}.csv", index=False); print(f"{label}: {sum(len(r) for r in rows)} null distances")
