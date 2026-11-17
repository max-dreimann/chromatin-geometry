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
"""ORIGINAL geometry implementation (used for the fiber-level statistics, Fig. 3h,i and Supp. Fig. 4e,f).
Merged verbatim from the original pipeline modules cylinder_contacts.py and end_to_end_distance.py; only the
sandbox sys.path lines and the demo code were removed. Functionally equivalent to lib/geometry.py: on real data the two
differ by up to 1.3 A in about 0.4 % of pairs, agree on every 3 nm contact, and differ after 4 A rounding for 10 of 7,939 (KU216)
and 2 of 6,812 (DeltaB) nearest-neighbour values (see README).
"""
import numpy as np



def project_onto_cylinder(x, P, axis_unit, L, r):
    """Project point x onto the solid capped cylinder with base P, axis
    direction axis_unit (unit vector), length L, radius r."""
    t = np.dot(x - P, axis_unit)
    perp = (x - P) - t * axis_unit
    perp_norm = np.linalg.norm(perp)

    t_clamped = np.clip(t, 0.0, L)

    if perp_norm > r:
        if perp_norm < 1e-12:
            perp_capped = perp  # degenerate, shouldn't happen given perp_norm>r>0
        else:
            perp_capped = perp * (r / perp_norm)
    else:
        perp_capped = perp

    if 0.0 <= t <= L and perp_norm <= r:
        return x  # already inside the solid cylinder

    return P + t_clamped * axis_unit + perp_capped


def cylinder_cylinder_distance(P1, axis1, L1, r1, P2, axis2, L2, r2, n_iter=60):
    """
    Minimum distance between two solid, flat-ended cylinders via alternating
    projection. Returns the distance (0 if overlapping).
    """
    q = P2 + 0.5 * L2 * axis2  # start at midpoint of cylinder 2's axis
    p = P1
    for _ in range(n_iter):
        p = project_onto_cylinder(q, P1, axis1, L1, r1)
        q = project_onto_cylinder(p, P2, axis2, L2, r2)
    return float(np.linalg.norm(p - q))







def disk_disk_distance(C1, axis1, r1, C2, axis2, r2, n_iter=60):
    return cylinder_cylinder_distance(C1, axis1, 0.0, r1, C2, axis2, 0.0, r2, n_iter=n_iter)


def end_to_end_distance(P1, axis1, L1, r1, P2, axis2, L2, r2, n_iter=60):
    """P1/P2 are the base points (start of axis, as used elsewhere in this
    codebase: P = position - (L/2)*axis). Returns min distance over all
    4 cap-pair combinations."""
    caps1 = [P1, P1 + L1 * axis1]
    caps2 = [P2, P2 + L2 * axis2]
    best = np.inf
    for c1 in caps1:
        for c2 in caps2:
            d = disk_disk_distance(c1, axis1, r1, c2, axis2, r2, n_iter=n_iter)
            best = min(best, d)
    return best


