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
"""End-to-end (cap-disk) distance between two particles.

Each particle is a solid flat-ended cylinder (axis base P, unit axis, length L, radius r). Distances are
measured between the flat circular end caps only (not the cylinder side), so side-by-side stacking does not
count as close. The minimum distance between two convex sets (here, two disks) is found by alternating
projection (Cheney-Goldstein); n_iter=60 is the value used for every contact/NN/heatmap number in the paper.
"""
import numpy as np


def project_onto_cylinder(x, P, axis_unit, L, r):
    """Closest point of the solid capped cylinder (base P, unit axis, length L, radius r) to x.
    For L=0 this is the projection onto a flat disk."""
    v = x - P
    t = np.dot(v, axis_unit)
    t_clamped = np.clip(t, 0, L)
    closest_axis = P + t_clamped * axis_unit
    v_perp = v - t * axis_unit                  # true perpendicular component (full axial part removed)
    radial_dist = np.linalg.norm(v_perp)
    if radial_dist > r:
        return closest_axis + r * (v_perp / radial_dist)
    return closest_axis + v_perp


def cylinder_cylinder_distance(P1, axis1, L1, r1, P2, axis2, L2, r2, n_iter=60):
    x = P1 + 0.5 * L1 * axis1
    for _ in range(n_iter):
        x2 = project_onto_cylinder(x, P2, axis2, L2, r2)
        x = project_onto_cylinder(x2, P1, axis1, L1, r1)
    return float(np.linalg.norm(project_onto_cylinder(x, P2, axis2, L2, r2) - x))


def disk_disk_distance(C1, axis1, r1, C2, axis2, r2, n_iter=60):
    return cylinder_cylinder_distance(C1, axis1, 0.0, r1, C2, axis2, 0.0, r2, n_iter=n_iter)


def end_to_end_distance(P1, axis1, L1, r1, P2, axis2, L2, r2, n_iter=60):
    """P1/P2 = base points (position - (L/2)*axis). Minimum over the 4 cap-pair combinations."""
    caps1 = [P1, P1 + L1 * axis1]
    caps2 = [P2, P2 + L2 * axis2]
    best = np.inf
    for c1 in caps1:
        for c2 in caps2:
            best = min(best, disk_disk_distance(c1, axis1, r1, c2, axis2, r2, n_iter=n_iter))
    return best


def _self_test():
    ax = np.array([0., 0., 1.]); r = 45.0
    d1 = disk_disk_distance(np.array([0, 0, 100.]), ax, r, np.array([0, 0, 150.]), ax, r)
    d2 = end_to_end_distance(np.zeros(3), ax, 200., r, np.array([0, 0, 215.]), ax, 200., r)
    d3 = cylinder_cylinder_distance(np.zeros(3), ax, 200., r, np.array([200., 0, 0]), ax, 200., r)
    d4 = cylinder_cylinder_distance(np.zeros(3), ax, 200., r, np.array([80., 0, 0]), ax, 200., r)
    assert abs(d1 - 50) < 1e-6 and abs(d2 - 15) < 1e-6 and abs(d3 - 110) < 1e-6 and d4 == 0, (d1, d2, d3, d4)
    return d1, d2, d3, d4


if __name__ == "__main__":
    print("analytic self-test (expect 50, 15, 110, 0):", _self_test())
