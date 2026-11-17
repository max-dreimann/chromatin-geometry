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
"""RELION Euler angles -> rotation matrices (matches Euler_angles2matrix in RELION's Euler.cpp;
identity at rot=tilt=psi=0, tilt=90 rotates Z to -X). Angles in degrees; vectorised."""
import numpy as np


def relion_euler_to_matrix(rot, tilt, psi):
    rot, tilt, psi = np.deg2rad(rot), np.deg2rad(tilt), np.deg2rad(psi)
    ca, sa, cb, sb, cg, sg = np.cos(rot), np.sin(rot), np.cos(tilt), np.sin(tilt), np.cos(psi), np.sin(psi)
    N = np.atleast_1d(rot).shape[0]
    R = np.zeros((N, 3, 3))
    R[:, 0, 0] = cg * cb * ca - sg * sa; R[:, 0, 1] = cg * cb * sa + sg * ca; R[:, 0, 2] = -cg * sb
    R[:, 1, 0] = -sg * cb * ca - cg * sa; R[:, 1, 1] = -sg * cb * sa + cg * ca; R[:, 1, 2] = sg * sb
    R[:, 2, 0] = sb * ca; R[:, 2, 1] = sb * sa; R[:, 2, 2] = cb
    return R
