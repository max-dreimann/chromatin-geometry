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
"""Step 9. Statistics for the KU216 vs DeltaB fiber comparisons (Fig. 3h,i; Supp. Fig. 4e,f).

For each metric: two-sided Mann-Whitney U test (scipy, asymptotic p with tie handling), effect size r = |z| / sqrt(N)
with z from the normal approximation to U (no tie correction; the tie-corrected value is also listed), the median
difference (KU216 - DeltaB), and bootstrap 95 % CIs for the median difference and for r: each group is resampled with
replacement N_BOOT = 5000 times (numpy default_rng, seed 42) and the statistic recomputed; the CI is the 2.5-97.5
percentile range. Metrics are processed in the fixed order below from ONE random stream, so the intervals are
reproducible; only the first two metrics (fiber end-to-end distance on 4 A-rounded values, DNA condensed) are reported
in the paper, the other two are computed to keep the random sequence identical to the published run.

Output: statistics.csv
"""
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from common import get_args

N_BOOT, SEED = 5000, 42
METRICS = [("fiber_end_to_end_distance_nm (4 A-rounded)", "rounded_4A_nm"), ("DNA_condensed_bp", "total_bp"),
           ("DNA_density_bp_per_nm", "density"), ("particles_per_fiber", "n_particles")]


def z_from_u(u, n1, n2):
    return (u - n1 * n2 / 2) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)


def tie_corrected_r(a, b):
    n1, n2 = len(a), len(b); N = n1 + n2; u, _ = mannwhitneyu(a, b, alternative="two-sided")
    _, t = np.unique(np.concatenate([a, b]), return_counts=True)
    sigma = np.sqrt(n1 * n2 / 12 * ((N + 1) - (t ** 3 - t).sum() / (N * (N - 1))))
    return abs((u - n1 * n2 / 2) / sigma) / np.sqrt(N)


if __name__ == "__main__":
    a_ = get_args(description=__doc__)
    wt, dl = pd.read_csv(f"{a_.out_dir}/fiber_stats_wt.csv"), pd.read_csv(f"{a_.out_dir}/fiber_stats_del.csv")
    rng, rows = np.random.default_rng(SEED), []
    for name, col in METRICS:
        a, b = wt[col].to_numpy(float), dl[col].to_numpy(float); n1, n2 = len(a), len(b)
        u, p = mannwhitneyu(a, b, alternative="two-sided"); r = abs(z_from_u(u, n1, n2)) / np.sqrt(n1 + n2)
        md, rs = np.empty(N_BOOT), np.empty(N_BOOT)
        for k in range(N_BOOT):
            sa, sb = a[rng.integers(0, n1, n1)], b[rng.integers(0, n2, n2)]
            md[k] = np.median(sa) - np.median(sb); ub, _ = mannwhitneyu(sa, sb, alternative="two-sided"); rs[k] = abs(z_from_u(ub, n1, n2)) / np.sqrt(n1 + n2)
        mlo, mhi = np.percentile(md, [2.5, 97.5]); rlo, rhi = np.percentile(rs, [2.5, 97.5])
        rows.append(dict(metric=name, n_KU216=n1, n_DeltaB=n2, median_KU216=np.median(a), median_DeltaB=np.median(b), U=u, p_value=p,
                         effect_size_r=r, r_ci_low=rlo, r_ci_high=rhi, r_tie_corrected=tie_corrected_r(a, b),
                         median_difference=np.median(a) - np.median(b), median_diff_ci_low=mlo, median_diff_ci_high=mhi))
    out = pd.DataFrame(rows); out.to_csv(f"{a_.out_dir}/statistics.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    print(out.T.to_string(header=False))
