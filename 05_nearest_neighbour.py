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
"""Step 5. Nearest-neighbour edge-to-edge distance, touching fraction, isolated particles.

Duplicates (step 3) are dropped from the candidate graph. For every remaining particle with at least one candidate
neighbour, the nearest-neighbour distance is the minimum end-to-end distance over all its candidates (a strict
single-NN analysis; Fig. 3a,b and Supp. Fig. 4a,b,d). Distances are reported in nm and rounded to the nearest 4 A
(the resolution limit of the data); "touching" means rounded distance == 0. A particle is "isolated" when it has no
partner within 3 nm (30 A) end-to-end distance.

Outputs: nn_{wt,del}.csv, nn_summary.csv (touching per tomogram and pooled), isolated_particles.csv
"""
import numpy as np, pandas as pd
from common import get_args
from lib.classes import PIXEL_NM, CONTACT_CUTOFF, WT_TOMOGRAMS, DEL_TOMOGRAMS


def rnd(d_nm):
    return np.round(np.round(np.asarray(d_nm) / PIXEL_NM) * PIXEL_NM, 1)


if __name__ == "__main__":
    a = get_args(description=__doc__)
    nn_rows, summ, iso = [], [], []
    for tag, label, tomos in [("wt", "KU216", WT_TOMOGRAMS), ("del", "DeltaB", DEL_TOMOGRAMS)]:
        df = pd.read_pickle(f"{a.out_dir}/particles_{tag}.pkl"); rm = set(np.load(f"{a.out_dir}/duplicates_{tag}.npy").tolist())
        c = pd.read_csv(f"{a.out_dir}/contacts_{tag}.csv"); c = c[~(c["i"].isin(rm) | c["j"].isin(rm))]
        best = {}
        for i, j, d in zip(c["i"], c["j"], c["end_distance"]):
            for p in (int(i), int(j)):
                if p not in best or d < best[p]: best[p] = d
        tomo_label = {t: f"tomogram_{k}" for k, t in enumerate(tomos, 1)}
        nn = pd.DataFrame({"particle_index": list(best), "nn_edge_to_edge_distance_A": list(best.values())})
        nn["dataset"] = label; nn["tomogram"] = df["tomogram"].to_numpy()[nn["particle_index"]]
        nn["tomogram"] = nn["tomogram"].map(tomo_label)
        nn["nn_edge_to_edge_distance_nm"] = nn["nn_edge_to_edge_distance_A"] / 10.0; nn["rounded_4A_nm"] = rnd(nn["nn_edge_to_edge_distance_nm"])
        nn = nn.sort_values(["tomogram", "particle_index"]).reset_index(drop=True); nn.to_csv(f"{a.out_dir}/nn_{tag}.csv", index=False)
        print(f"{label}: n = {len(nn)} particles with a neighbour; touching = {(nn.rounded_4A_nm == 0).mean() * 100:.2f}% ({(nn.rounded_4A_nm == 0).sum()})")
        for t in list(tomo_label.values()) + ["all"]:
            s = nn if t == "all" else nn[nn.tomogram == t]
            summ.append((label, t, len(s), int((s.rounded_4A_nm == 0).sum()), 100 * (s.rounded_4A_nm == 0).mean()))
        # isolated particles (3 nm cutoff on the duplicate-filtered candidate graph)
        connected = set(c[c["end_distance"] <= CONTACT_CUTOFF]["i"]) | set(c[c["end_distance"] <= CONTACT_CUTOFF]["j"])
        keep = np.array([k not in rm for k in range(len(df))])
        for t, lab in tomo_label.items():
            m = keep & (df["tomogram"].to_numpy() == t); total = int(m.sum()); withc = int(sum(1 for k in np.where(m)[0] if k in connected))
            iso.append((label, lab, total, withc, total - withc, round(100 * (total - withc) / total, 1)))
    pd.DataFrame(summ, columns=["dataset", "tomogram", "n_particles", "n_touching", "touching_percent"]).to_csv(f"{a.out_dir}/nn_summary.csv", index=False)
    pd.DataFrame(iso, columns=["dataset", "tomogram", "total_particles", "particles_with_contact", "isolated_particles", "isolated_percent"]).to_csv(f"{a.out_dir}/isolated_particles.csv", index=False)
    print(pd.read_csv(f"{a.out_dir}/isolated_particles.csv").to_string(index=False))
