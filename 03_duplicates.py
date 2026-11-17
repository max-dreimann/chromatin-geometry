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
"""Step 3. Duplicate particle detection.

Template matching / subtomogram picking can place two picks on one physical particle. Within each tomogram, a pair
is flagged as duplicate when centre-to-centre distance / mean(modelled length of the two particles) < 1.0 (the two
particles would have to overlap substantially to both be real). Of each flagged pair the second-encountered index is
removed. Output: duplicates_{wt,del}.npy (sorted indices of removed particles).
"""
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from common import get_args

RATIO_CUTOFF, SEARCH_RADIUS = 1.0, 200.0


def find_duplicates(positions, lengths, tomograms):
    remove = set()
    for t in np.unique(tomograms):
        idx = np.where(tomograms == t)[0]
        for a, b in cKDTree(positions[idx]).query_pairs(r=SEARCH_RADIUS, output_type="ndarray"):
            i, j = int(idx[a]), int(idx[b])
            if np.linalg.norm(positions[i] - positions[j]) / ((lengths[i] + lengths[j]) / 2) < RATIO_CUTOFF:
                if i not in remove and j not in remove:
                    remove.add(j)
    return remove


if __name__ == "__main__":
    a = get_args(description=__doc__)
    for tag in ["wt", "del"]:
        df = pd.read_pickle(f"{a.out_dir}/particles_{tag}.pkl"); L = np.load(f"{a.out_dir}/lengths_{tag}.npy")
        rm = find_duplicates(df[["x", "y", "z"]].to_numpy(), L, df["tomogram"].to_numpy())
        np.save(f"{a.out_dir}/duplicates_{tag}.npy", np.array(sorted(rm)))
        print(f"{tag}: {len(rm)} duplicates removed, {len(df) - len(rm)} particles retained")
