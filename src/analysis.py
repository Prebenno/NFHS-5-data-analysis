import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from balance.stats_and_plots.weighted_stats import (
    weighted_mean as bal_wmean,
    weighted_quantile as bal_wquantile,
    var_of_weighted_mean as bal_var_of_wmean,
)

from .config import (
    ALL_BP_COLS, READING_PAIRS, HTN_ORDER,
    SYSTOLIC_COLS, DIASTOLIC_COLS,
)


WEALTH_MAPPING = {
    "poorest": "Poor",
    "poorer":  "Poor",
    "middle":  "Middle",
    "richer":  "Rich",
    "richest": "Rich",
}


def split_into_brackets(data, col, n_brackets, wcol="weight"):
    df = data.copy()

    vals = pd.to_numeric(df[col], errors="coerce")
    wts  = pd.to_numeric(df[wcol], errors="coerce")
    valid = np.isfinite(vals) & np.isfinite(wts) & (wts > 0)

    probs = [i / n_brackets for i in range(1, n_brackets)]
    wq = bal_wquantile(vals[valid], probs, wts[valid])
    inner = wq.iloc[:, 0].values.astype(float)

    edges = np.concatenate([[vals[valid].min()], inner, [vals[valid].max()]])
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6

    step = 100 // n_brackets
    labels = []
    for i in range(n_brackets):
        lo = i * step
        hi = min((i + 1) * step, 100)
        labels.append(f"Bottom {hi}%" if i == 0 else f"{lo}-{hi}%")

    bracket_name = f"{col} Bracket"
    df[bracket_name] = pd.cut(vals, bins=edges, labels=labels, include_lowest=True)

    ranges = [(edges[i], edges[i + 1]) for i in range(n_brackets)]
    return df, ranges


def group_stats(df, group_col, ycol, wcol="weight"):
    rows = []
    for name, grp in df.groupby(group_col):
        x = pd.to_numeric(grp[ycol], errors="coerce").to_numpy()
        w = pd.to_numeric(grp[wcol], errors="coerce").to_numpy()
        valid = np.isfinite(x) & np.isfinite(w) & (w > 0)

        if not valid.any():
            rows.append({group_col: name, "mean": np.nan, "lo": np.nan, "hi": np.nan})
            continue

        xs, ws = pd.Series(x[valid]), pd.Series(w[valid])
        mu = float(bal_wmean(xs, ws).iloc[0])
        se = float(np.sqrt(bal_var_of_wmean(xs, ws).iloc[0]))
        rows.append({group_col: name, "mean": mu, "lo": mu - 1.96 * se, "hi": mu + 1.96 * se})

    return pd.DataFrame(rows).sort_values(group_col)


def weighted_corr_matrix(df, readings=None, wcol="weight"):
    if readings is None:
        readings = ALL_BP_COLS

    W = pd.to_numeric(df[wcol], errors="coerce").to_numpy()
    cols = {c: pd.to_numeric(df[c], errors="coerce").to_numpy() for c in readings}
    n = len(readings)
    mat = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):
            a, b = cols[readings[i]], cols[readings[j]]
            ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(W) & (W > 0)

            if not ok.any():
                mat[i, j] = mat[j, i] = np.nan
                continue

            ao, bo, wo = a[ok], b[ok], W[ok]
            ma = float(bal_wmean(pd.Series(ao), pd.Series(wo)).iloc[0])
            mb = float(bal_wmean(pd.Series(bo), pd.Series(wo)).iloc[0])
            wsum = wo.sum()
            cov_ab = np.sum(wo * (ao - ma) * (bo - mb)) / wsum
            var_a  = np.sum(wo * (ao - ma) ** 2) / wsum
            var_b  = np.sum(wo * (bo - mb) ** 2) / wsum
            denom  = np.sqrt(var_a * var_b)
            r = cov_ab / denom if denom > 0 else np.nan
            mat[i, j] = mat[j, i] = r

    return pd.DataFrame(mat, index=readings, columns=readings)


def compute_bp_differences(df):
    out = df.copy()
    for pressure_type in ["SYSTOLIC", "DIASTOLIC"]:
        for r1, r2 in READING_PAIRS:
            out[f"Diff {r1}-{r2} {pressure_type}"] = (
                pd.to_numeric(out[f"{r1} {pressure_type} reading"], errors="coerce")
                - pd.to_numeric(out[f"{r2} {pressure_type} reading"], errors="coerce")
            )
    return out


def weighted_pearson_r(x, y, w):
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return np.nan

    xo, yo, wo = x[ok], y[ok], w[ok]
    mx = float(bal_wmean(pd.Series(xo), pd.Series(wo)).iloc[0])
    my = float(bal_wmean(pd.Series(yo), pd.Series(wo)).iloc[0])
    wsum = wo.sum()
    cov = np.sum(wo * (xo - mx) * (yo - my)) / wsum
    vx  = np.sum(wo * (xo - mx) ** 2) / wsum
    vy  = np.sum(wo * (yo - my) ** 2) / wsum
    d = np.sqrt(vx * vy)
    return cov / d if d > 0 else np.nan


def classify_hypertension(sbp):
    sbp = pd.to_numeric(sbp, errors="coerce")
    if pd.isna(sbp):
        return np.nan
    if sbp < 120:
        return "Normal"
    if sbp <= 139:
        return "Pre-Hypertensive"
    if sbp <= 159:
        return "Hypertensive 1"
    if sbp <= 179:
        return "Hypertensive 2"
    return "Hypertensive Crisis"


def add_hypertension_staging(df):
    df["Hypertension Staging"] = df["First SYSTOLIC reading"].apply(classify_hypertension)


def compute_relative_ratios_by_stage(df, group_var, baseline, mapping=None):
    df = df.copy()
    df[group_var] = df[group_var].astype(str).str.strip().str.lower()

    if mapping is not None:
        agg_col = group_var + "_agg"
        lc_map = {str(k).strip().lower(): v for k, v in mapping.items()}
        df[agg_col] = df[group_var].map(lc_map)
        eff_group = agg_col
    else:
        eff_group = group_var

    grouped = (
        df.dropna(subset=[eff_group, "Hypertension Staging"])
          .groupby([eff_group, "Hypertension Staging"])["weight"]
          .sum()
          .unstack(fill_value=0)
    )
    grouped = grouped.reindex(columns=HTN_ORDER, fill_value=0)
    pcts = grouped.div(grouped.sum(axis=1), axis=0) * 100

    candidates = baseline if isinstance(baseline, list) else [baseline]
    idx_lower = {str(idx).strip().lower(): idx for idx in pcts.index}

    matched = None
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in idx_lower:
            matched = idx_lower[key]
            break

    if matched is None:
        print(f"Baseline '{baseline}' not found. Available: {list(pcts.index)}")
        return None, None

    base_pct = pcts.loc[matched]
    relative = pcts.divide(base_pct, axis="columns")
    return relative, pcts


def build_comparisons(df_male, df_female, *, verbose=True):
    comparisons = {}

    if "Highest educational level attained" in df_male.columns:
        comparisons["Highest educational level attained"] = {
            "baseline": "no education, preschool", "mapping": None,
        }
        if verbose:
            print("  → Education baseline: 'No education, preschool'")

    if "Wealth index combined" in df_male.columns:
        comparisons["Wealth index combined"] = {
            "baseline": "middle", "mapping": WEALTH_MAPPING,
        }
        if verbose:
            print(f"  → Wealth baseline: 'Middle' (mapped: {WEALTH_MAPPING})")

    if "Caste or tribe" in df_male.columns:
        comparisons["Caste or tribe"] = {
            "baseline": ["none of them", "none of above"], "mapping": None,
        }
        if verbose:
            print("  → Caste baseline: 'None of them' / 'None of above'")

    if "Smokes or uses tobacco" in df_male.columns:
        comparisons["Smokes or uses tobacco"] = {
            "baseline": "no", "mapping": None,
        }
        if verbose:
            print("  → Tobacco baseline: 'No'")

    if "30 minutes prior to BP measure: smoked any tobacco product" in df_male.columns:
        comparisons["30 minutes prior to BP measure: smoked any tobacco product"] = {
            "baseline": "no", "mapping": None,
        }
        if verbose:
            print("  → Smoking before BP baseline: 'No'")

    if verbose:
        print(f"\nActive comparisons: {len(comparisons)}")

    return comparisons


def build_ratio_plots_list(comparisons, data_male, data_female):
    plots = []

    for var, cfg in comparisons.items():
        bl = cfg["baseline"]
        mp = cfg["mapping"]

        m_rel, _ = compute_relative_ratios_by_stage(data_male, var, bl, mp)
        f_rel, _ = compute_relative_ratios_by_stage(data_female, var, bl, mp)

        if m_rel is None or f_rel is None:
            continue

        bl_candidates = bl if isinstance(bl, list) else [bl]
        bl_set = {str(b).strip().lower() for b in bl_candidates}

        groups = sorted(set(m_rel.index) | set(f_rel.index))
        bl_label = bl_candidates[0]
        for grp in groups:
            if str(grp).strip().lower() in bl_set:
                continue
            plots.append((var, grp, bl_label, m_rel, f_rel))

    return plots


def ecdf_ci_from_weights(ecdf_vals, w_arr, z=1.96):
    w = np.asarray(w_arr, dtype=float)
    n = len(w)
    W = w.sum()
    D = W - (w ** 2).sum() / W
    C = W / (D * n)

    p = np.asarray(ecdf_vals, dtype=float)
    se = np.sqrt(p * (1.0 - p) * C)
    lo = np.clip(p - z * se, 0, 1)
    hi = np.clip(p + z * se, 0, 1)
    return lo, hi


def disease_sbp_pvalue(df, disease_col,
                       sbp_col="First systolic reading", wcol="weight"):
    sick = df[df[disease_col] == 1]
    groups = {}
    for sex in ("female", "male"):
        sub = sick[sick["sex"] == sex]
        sbp = pd.to_numeric(sub[sbp_col], errors="coerce")
        w = pd.to_numeric(sub[wcol], errors="coerce")
        ok = np.isfinite(sbp) & np.isfinite(w) & (w > 0)
        groups[sex] = sbp[ok].values

    _, pval = ks_2samp(groups["female"], groups["male"])
    return pval


SBP_BINS = [
    ("SBP < 80",  0, 80),
    ("80-89",     80, 90),
    ("90-99",     90, 100),
    ("100-109",  100, 110),
    ("110-119",  110, 120),
    ("120-129",  120, 130),
    ("130-139",  130, 140),
    ("140-149",  140, 150),
    ("150-159",  150, 160),
    ("160-169",  160, 170),
    ("170-179",  170, 180),
    ("SBP > 180", 180, 9999),
]


def sbp_disease_distribution(df, disease_col, sbp_col="First systolic reading", wcol="weight"):
    sick = df[df[disease_col] == 1].copy()
    sbp = pd.to_numeric(sick[sbp_col], errors="coerce")
    w = pd.to_numeric(sick[wcol], errors="coerce")
    ok = np.isfinite(sbp) & np.isfinite(w) & (w > 0)
    sbp, w = sbp[ok], w[ok]

    total = w.sum()
    pcts, cumuls = [], []
    running = 0.0
    for _, lo, hi in SBP_BINS:
        in_bin = (sbp >= lo) & (sbp < hi)
        pct = round(w[in_bin].sum() / total * 100, 2) if total > 0 else 0.0
        running = round(running + pct, 2)
        pcts.append(pct)
        cumuls.append(running)

    return pcts, cumuls


def build_disease_sbp_table(df_n5, df_n4, disease_col,
                            sbp_col="First systolic reading"):
    labels = [lbl for lbl, _, _ in SBP_BINS]

    w5_pct, w5_cum = sbp_disease_distribution(
        df_n5[df_n5["sex"] == "female"], disease_col, sbp_col)
    m5_pct, m5_cum = sbp_disease_distribution(
        df_n5[df_n5["sex"] == "male"], disease_col, sbp_col)

    if df_n4 is not None:
        w4_pct, w4_cum = sbp_disease_distribution(
            df_n4[df_n4["sex"] == "female"], disease_col, sbp_col)
        m4_pct, m4_cum = sbp_disease_distribution(
            df_n4[df_n4["sex"] == "male"], disease_col, sbp_col)

        table = pd.DataFrame({
            "SBP": labels,
            "Women NFHS-4": w4_pct, "Women Cumul.(4)": w4_cum,
            "Women NFHS-5": w5_pct, "Women Cumul.(5)": w5_cum,
            "Men NFHS-4": m4_pct, "Men Cumul.(4)": m4_cum,
            "Men NFHS-5": m5_pct, "Men Cumul.(5)": m5_cum,
        })
    else:
        table = pd.DataFrame({
            "SBP": labels,
            "Women NFHS-5": w5_pct, "Women Cumul.": w5_cum,
            "Men NFHS-5": m5_pct, "Men Cumul.": m5_cum,
        })

    table.attrs["ks_pvalue"] = disease_sbp_pvalue(df_n5, disease_col, sbp_col)
    return table
