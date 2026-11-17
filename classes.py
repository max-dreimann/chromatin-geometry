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
"""Dataset definitions: which STAR file holds which particle class, which tomograms are analysed,
and the number of histone dimers per class. Pixel/tomogram constants used across the pipeline."""

# class label ("<n>dimer") -> number of histone dimers (n_dimer). The 9-12 dimer KU216 classes correspond to 3.5, 4, 4.5 and 5.5 superhelical turns.
WT_CLASS_TO_N = {"3dimer": 3, "4dimer": 4, "5dimer": 5, "6dimer": 6, "7dimer": 7, "8dimer": 8,
                 "9dimer": 9, "10dimer": 10, "11dimer": 11, "12dimer": 12}
DEL_CLASS_TO_N = {"2dimer": 2, "3dimer": 3, "4dimer": 4, "5dimer": 5, "6dimer": 6, "7dimer": 7}

# class -> STAR file of that class (KU216: WT_<n>dimer.star, n = 3-12; DeltaB: deltaB_<n>dimer.star, n = 2-7; n = histone dimers; 9-12 dimers = 3.5, 4, 4.5, 5.5 turns)
WT_CLASS_FILES = {cls: f"WT_{n}dimer.star" for cls, n in WT_CLASS_TO_N.items()}
DEL_CLASS_FILES = {cls: f"deltaB_{n}dimer.star" for cls, n in DEL_CLASS_TO_N.items()}

# tomograms analysed (order = tomogram_1, tomogram_2, ... in all result tables)
WT_TOMOGRAMS = ["KU216_stationary_01_ts_003.tomostar", "KU216_stationary_01_ts_012.tomostar",
                "KU216_stationary_06_ts_012.tomostar", "KU216_stationary_10_ts_006.tomostar",
                "KU216_stationary_10_ts_010.tomostar"]
DEL_TOMOGRAMS = ["stationary_deltaB_9-A3_ts_003.tomostar", "stationary_deltaB_9-A3_ts_012.tomostar",
                 "stationary_deltaB_9-A_ts_011.tomostar"]

# reference class-average maps used for the length model (n_dimer -> MRC file)
WT_MAPS = {n: f"WT_{n}dimer.mrc" for n in range(3, 13)}
DEL_MAPS = {n: f"deltaB_{n}dimer.mrc" for n in range(2, 8)}

# model constants
RADIUS = 45.0            # Angstrom, particle (capsule/disk) radius
SLOPE, INTERCEPT = 9.39, 20.13     # Angstrom: L(n) = INTERCEPT + SLOPE*n (rounded fit values, see 02_length_model.py)
CANDIDATE_CUTOFF = 450.0 # Angstrom centre-to-centre pre-filter for candidate pairs
CONTACT_CUTOFF = 30.0    # Angstrom (3 nm) end-to-end distance defining a connected pair
PIXEL_A = 4.0            # Angstrom: resolution limit of the data (refinement box / reference maps)
PIXEL_NM = PIXEL_A / 10.0
TOMO_PIXEL_A = 10.0      # Angstrom per pixel of the reconstructed tomograms (used for STAR export)
BP_PER_DIMER = 30
