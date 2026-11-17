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
"""Step 2. Particle length model L(n) = INTERCEPT + SLOPE * n_dimer (Angstrom).

Measured directly from the 16 class-average reference maps (10 KU216 classes with 3-12 dimers, 6 DeltaB classes with 2-7
dimers): the length of a class = extent along the map x axis (the superhelical axis, pure-X convention) of all voxels
above 50 % of the map's peak density, times the voxel size. A straight line is fitted to all 16 points. The pipeline
uses the fit rounded to two decimals (SLOPE = 9.39, INTERCEPT = 20.13 in lib/classes.py).

Outputs: lengths_{wt,del}.npy (one length per particle, Angstrom)
"""
import numpy as np, pandas as pd, mrcfile
from common import get_args
from lib.classes import WT_MAPS, DEL_MAPS, WT_CLASS_TO_N, DEL_CLASS_TO_N, SLOPE, INTERCEPT


def measure_length_nm(path, frac=0.5):
    with mrcfile.open(path, permissive=True) as m:
        data, vox = m.data, float(m.voxel_size.x)
    coords = np.argwhere(data > frac * data.max())
    return (coords[:, 2].max() - coords[:, 2].min()) * vox / 10.0       # array axes are (z, y, x)


if __name__ == "__main__":
    a = get_args(description=__doc__)
    pts = []
    for label, maps in [("KU216", WT_MAPS), ("DeltaB", DEL_MAPS)]:
        for n, f in maps.items():
            l = measure_length_nm(f"{a.star_dir}/{f}"); pts.append((label, n, l)); print(f"  {label} {n:2d} dimers: {l:.2f} nm")
    n_all = np.array([p[1] for p in pts], float); l_all = np.array([p[2] for p in pts]) * 10
    slope, intercept = np.polyfit(n_all, l_all, 1)
    r2 = 1 - ((l_all - (slope * n_all + intercept)) ** 2).sum() / ((l_all - l_all.mean()) ** 2).sum()
    print(f"fit: L = {intercept:.4f} + {slope:.4f}*n (Angstrom), R^2 = {r2:.3f}; pipeline constants: {INTERCEPT} + {SLOPE}*n")
    pd.DataFrame(pts, columns=["dataset", "n_dimer", "length_nm"]).to_csv(f"{a.out_dir}/length_measurements.csv", index=False)
    for tag, c2n in [("wt", WT_CLASS_TO_N), ("del", DEL_CLASS_TO_N)]:
        df = pd.read_pickle(f"{a.out_dir}/particles_{tag}.pkl")
        lengths = INTERCEPT + SLOPE * df["class"].map(c2n).to_numpy().astype(float)
        np.save(f"{a.out_dir}/lengths_{tag}.npy", lengths)
        print(f"{tag}: {len(lengths)} particles, length range {lengths.min()/10:.2f}-{lengths.max()/10:.2f} nm")
