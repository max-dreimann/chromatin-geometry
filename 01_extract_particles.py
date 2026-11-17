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
"""Step 1. Particle table from the RELION class STAR files.

For every class: read its full refinement STAR file, keep the particles of the analysed tomograms, convert
coordinates to Angstrom (pixel coordinate * pixel size of that STAR file's optics block minus the refined origin
shift), convert the Euler angles to a rotation matrix R, and define the superhelical axis of each particle as
z_lab = R @ [1, 0, 0] ("pure-X" convention: the class reference maps are aligned so the superhelical axis lies
along x). Particle index = row in the concatenated table (classes in the order of lib/classes.py, particles in
STAR-file order).

Outputs: particles_{wt,del}.pkl, z_lab_{wt,del}.npy
"""
import numpy as np, pandas as pd, starfile
from common import get_args
from lib.classes import WT_CLASS_FILES, DEL_CLASS_FILES, WT_TOMOGRAMS, DEL_TOMOGRAMS
from lib.relion_geom import relion_euler_to_matrix


def extract(star_dir, class_files, tomograms):
    parts = []
    for cls, fname in class_files.items():
        s = starfile.read(f"{star_dir}/{fname}")
        p, pix = s["particles"], float(s["optics"]["rlnImagePixelSize"].iloc[0])
        sub = p[p["rlnMicrographName"].isin(tomograms)].copy()
        for ax_, col in zip("xyz", ["X", "Y", "Z"]):
            origin = sub[f"rlnOrigin{col}Angst"] if f"rlnOrigin{col}Angst" in sub.columns else 0.0
            sub[ax_] = sub[f"rlnCoordinate{col}"] * pix - origin
        sub["class"], sub["tomogram"] = cls, sub["rlnMicrographName"]
        sub["R"] = list(relion_euler_to_matrix(sub["rlnAngleRot"].to_numpy(), sub["rlnAngleTilt"].to_numpy(), sub["rlnAnglePsi"].to_numpy()))
        parts.append(sub[["x", "y", "z", "R", "class", "tomogram", "rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]])
        print(f"  {cls}: {len(sub)} particles")
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    a = get_args(description=__doc__)
    for tag, cf, tomos in [("wt", WT_CLASS_FILES, WT_TOMOGRAMS), ("del", DEL_CLASS_FILES, DEL_TOMOGRAMS)]:
        print(tag.upper())
        df = extract(a.star_dir, cf, tomos)
        df.to_pickle(f"{a.out_dir}/particles_{tag}.pkl")
        np.save(f"{a.out_dir}/z_lab_{tag}.npy", np.stack(df["R"].to_numpy()) @ np.array([1.0, 0.0, 0.0]))
        print(f"  total {len(df)} particles; per tomogram:", df["tomogram"].value_counts().to_dict())
