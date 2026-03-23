import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from balance.stats_and_plots.weighted_stats import (
    weighted_mean as bal_wmean,
    weighted_quantile as bal_wquantile,
)
from balance.stats_and_plots.weighted_comparisons_stats import (
    _weighted_ecdf as bal_ecdf,
)

from .config import (
    CI_ALPHA, DPI, FIG_DIR,
    SYSTOLIC_COLS, DIASTOLIC_COLS, ALL_BP_COLS,
    READING_PAIRS, HTN_ORDER, PANEL_LABELS,
    N_BRACKETS, N_HEIGHT_BRACKETS,
    SBP_COL, DBP_COL, BMI_RANGES, COARSE_BMI, COARSE_AGE, BMI_ORDER, AGE_ORDER,
)
from .analysis import (
    group_stats, split_into_brackets, weighted_pearson_r,
    ecdf_ci_from_weights, disease_sbp_pvalue,
    get_mean_bp_by_age_bmi, prepare_heatmap_data, get_coarse_stats, compute_coarse_did,
)


def _ensure_fig_dir():
    os.makedirs(FIG_DIR, exist_ok=True)


def _save(fig, fname):
    _ensure_fig_dir()
    path = os.path.join(FIG_DIR, fname)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"Saved {path}")


def _hist_col(ax, data, col, bins, color, ylabel=False):
    vals = data[col].dropna()
    ax.hist(vals, bins=bins, color=color, alpha=0.7, density=True)
    ax.set_title(col, fontsize=11, fontweight="bold")
    if ylabel:
        ax.set_ylabel("Density")


def plot_bp_distributions(diff_df, sex_label, color, fname):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    for i, col in enumerate(SYSTOLIC_COLS):
        _hist_col(axes[0, i], diff_df, col, 200, color, ylabel=(i == 0))

    for i, col in enumerate(DIASTOLIC_COLS):
        _hist_col(axes[1, i], diff_df, col, 200, color, ylabel=(i == 0))

    fig.suptitle(f"{sex_label} Blood Pressure Reading Distributions",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, fname)
    plt.show()


def plot_bp_differences(diff_df, sex_label, threshold, fname,
                        *, color_diff_sys, color_diff_dia):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    row_colors = [color_diff_sys, color_diff_dia]
    types = ["SYSTOLIC", "DIASTOLIC"]
    type_nice = ["Systolic", "Diastolic"]

    for row in range(2):
        for i, (r1, r2) in enumerate(READING_PAIRS):
            ax = axes[row, i]
            vals = diff_df[f"Diff {r1}-{r2} {types[row]}"].dropna()
            clipped = vals[vals.between(-threshold, threshold)]
            ax.hist(clipped, bins=threshold * 2, alpha=0.7,
                    color=row_colors[row], density=True)
            ax.axvline(0, color="black", linestyle="--", linewidth=1)
            ax.set_title(f"{r1}–{r2} {type_nice[row]} (±{threshold})",
                         fontsize=10, fontweight="bold")
            ax.set_xlabel("Difference (mmHg)")
            if i == 0:
                ax.set_ylabel("Density")

    fig.suptitle(f"Reading-to-Reading BP Differences ({sex_label})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, fname)
    plt.show()


def plot_correlation_heatmap(corr_df, sex_label, fname, *, cmap="RdBu_r"):
    fig, ax = plt.subplots(figsize=(8, 6))

    short = [c.replace(" reading", "").replace("First ", "1st ")
              .replace("Second ", "2nd ").replace("Third ", "3rd ")
             for c in corr_df.columns]

    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap=cmap,
                vmin=-1, vmax=1, center=0, ax=ax,
                xticklabels=short, yticklabels=short)

    ax.set_title(f"Weighted Correlation Matrix ({sex_label})",
                 fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    _save(fig, fname)
    plt.show()


def _extract_hours(subset, time_col):
    tv = pd.to_numeric(subset[time_col], errors="coerce").dropna()
    tv = tv[(tv >= 600) & (tv <= 2200)]
    if len(tv) == 0:
        return None
    return (tv // 100) + (tv % 100) / 60.0


def plot_time_of_day(female_5, male_5, female_4, male_4, df_n4, fname,
                     *, color_male, color_female):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    tcol = "Time of first BP reading"
    xlabels = [f"{h:02d}:00" for h in range(6, 22)]

    datasets = [
        (axes[0], "NFHS-5 (2019–21)", [(female_5, "Female", color_female),
                                         (male_5, "Male", color_male)]),
    ]
    if tcol in df_n4.columns:
        datasets.append(
            (axes[1], "NFHS-4 (2015–16)", [(female_4, "Female", color_female),
                                             (male_4, "Male", color_male)]))
    else:
        datasets.append((axes[1], "NFHS-4 (2015–16)", []))

    for ax, title, groups in datasets:
        for subset, lbl, clr in groups:
            hrs = _extract_hours(subset, tcol)
            if hrs is not None:
                ax.hist(hrs, bins=32, color=clr, alpha=0.5,
                        label=lbl, density=True)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Time of day")
        ax.set_xticks(range(6, 22))
        ax.set_xticklabels(xlabels, rotation=45, fontsize=8)
        ax.legend(fontsize=10)

    axes[0].set_ylabel("Density")
    fig.suptitle("Time of First BP Reading", fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save(fig, fname)
    plt.show()


def plot_bp_by_time_of_day(female_5, male_5, female_4, male_4, df_n4, fname,
                           *, color_male, color_female):
    tcol = "Time of first BP reading"
    bp_col = "First SYSTOLIC reading"

    def hour_stats(subset):
        tmp = subset[[tcol, bp_col, "weight"]].copy()
        for c in tmp.columns:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
        tmp = tmp.dropna()
        tmp = tmp[(tmp[tcol] >= 600) & (tmp[tcol] <= 2200)]
        tmp["hour"] = (tmp[tcol] // 100).astype(int)
        return group_stats(tmp, "hour", bp_col)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    xlabels = [f"{h:02d}:00" for h in range(6, 22)]

    panels = [
        (axes[0], "NFHS-5 (2019–21)", female_5, male_5, True),
    ]
    if tcol in df_n4.columns:
        panels.append((axes[1], "NFHS-4 (2015–16)", female_4, male_4, True))
    else:
        panels.append((axes[1], "NFHS-4 (2015–16)", None, None, False))

    for ax, title, fdata, mdata, has_data in panels:
        if has_data:
            for lbl, subset, clr in [("Female", fdata, color_female),
                                      ("Male", mdata, color_male)]:
                gs = hour_stats(subset)
                ax.plot(gs["hour"], gs["mean"], color=clr, label=lbl, linewidth=2)
                ax.fill_between(gs["hour"], gs["lo"], gs["hi"],
                                color=clr, alpha=CI_ALPHA, linewidth=0)

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Hour of day", fontsize=11)
        ax.set_xticks(range(6, 22))
        ax.set_xticklabels(xlabels, rotation=45, fontsize=8)
        ax.legend(fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

    axes[0].set_ylabel("Mean First Systolic BP (mmHg)", fontsize=11)
    fig.suptitle("First Systolic Blood Pressure by Time of Day (95% CI)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save(fig, fname)
    plt.show()


def plot_age_demographics(female_5, male_5, female_4, male_4, fname,
                          *, color_male, color_female):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
    age_col = "Age of household members"
    bins = np.arange(15, 86, 1)
    centers = (bins[:-1] + bins[1:]) / 2

    survey_data = [
        (axes[0], "NFHS-5 (2019\u201321)", female_5, male_5),
        (axes[1], "NFHS-4 (2015\u201316)", female_4, male_4),
    ]

    for ax, title, fdat, mdat in survey_data:
        m_ages = pd.to_numeric(mdat[age_col], errors="coerce").dropna()
        m_ages = m_ages[(m_ages >= 15) & (m_ages <= 85)]
        mc, _ = np.histogram(m_ages, bins=bins, density=True)

        f_ages = pd.to_numeric(fdat[age_col], errors="coerce").dropna()
        f_ages = f_ages[(f_ages >= 15) & (f_ages <= 85)]
        fc, _ = np.histogram(f_ages, bins=bins, density=True)

        ax.barh(centers, -mc, height=0.9, color=color_male, alpha=0.75, label="Male")
        ax.barh(centers, fc, height=0.9, color=color_female, alpha=0.75, label="Female")

        peak = max(mc.max(), fc.max()) * 1.05
        ax.set_xlim(-peak, peak)
        ticks = ax.get_xticks()
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{abs(t):.3f}" for t in ticks])

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Density")
        ax.set_ylim(15, 85)
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(axis="x", linestyle="--", alpha=0.7)
        ax.axvline(0, color="black", linewidth=0.8)

    axes[0].set_ylabel("Age (years)")
    fig.suptitle("Population Pyramid: Male vs Female",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save(fig, fname)
    plt.show()


def _plot_mean_bp_by_x(ax, m_data, f_data, xcol, ycol,
                       color_male, color_female, xlims=None):
    ms = group_stats(m_data, xcol, ycol)
    fs = group_stats(f_data, xcol, ycol)

    ax.plot(ms[xcol], ms["mean"], color=color_male, label="Male")
    ax.fill_between(ms[xcol], ms["lo"], ms["hi"],
                    color=color_male, alpha=CI_ALPHA, linewidth=0)
    ax.plot(fs[xcol], fs["mean"], color=color_female, label="Female")
    ax.fill_between(fs[xcol], fs["lo"], fs["hi"],
                    color=color_female, alpha=CI_ALPHA, linewidth=0)

    if xlims:
        ax.set_xlim(*xlims)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.7)


def plot_sbp_by_age(female_5, male_5, female_4, male_4, fname,
                    *, color_male, color_female):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    age_col = "Age of household members"
    bp = "First SYSTOLIC reading"

    for ax, title, fd, md in [
        (axes[0], "NFHS-5 (2019\u201321)", female_5, male_5),
        (axes[1], "NFHS-4 (2015\u201316)", female_4, male_4),
    ]:
        _plot_mean_bp_by_x(ax, md, fd, age_col, bp,
                           color_male, color_female, xlims=(15, 85))
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Age (years)", fontsize=11)
        ax.set_ylabel("Mean Systolic Blood Pressure (mmHg)", fontsize=11)

    fig.suptitle("First Systolic Blood Pressure by Age: Male vs Female (95% CI)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save(fig, fname)
    plt.show()


def plot_bmi_demographics(female_5, male_5, female_4, male_4, fname,
                          *, color_male, color_female):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    for ax, title, fd, md in [
        (axes[0], "NFHS-5 (2019\u201321)", female_5, male_5),
        (axes[1], "NFHS-4 (2015\u201316)", female_4, male_4),
    ]:
        for lbl, sub, clr in [("Female", fd, color_female),
                               ("Male", md, color_male)]:
            bmi = sub["BMI"].dropna()
            bmi = bmi[(bmi >= 10) & (bmi <= 50)]
            ax.hist(bmi, bins=80, color=clr, alpha=0.5, label=lbl, density=True)

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Body Mass Index (BMI)")
        ax.legend(fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

    axes[0].set_ylabel("Density")
    fig.suptitle("BMI Distribution: Male vs Female",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save(fig, fname)
    plt.show()


def plot_bmi_vs_waist(female_2021, male_2021, fname,
                      *, color_male, color_female, cmap="RdBu_r"):
    from .analysis import weighted_corr_matrix as wcm

    cols = ["BMI", "Waist curcumference"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, lbl, data in [(axes[0], "Female", female_2021),
                           (axes[1], "Male", male_2021)]:
        valid = data.dropna(subset=cols)
        cm = wcm(valid, readings=cols)
        sns.heatmap(cm, annot=True, fmt=".2f", cmap=cmap,
                    vmin=-1, vmax=1, center=0, ax=ax,
                    xticklabels=["BMI", "Waist circ."],
                    yticklabels=["BMI", "Waist circ."],
                    square=True, cbar_kws={"shrink": 0.8})
        ax.set_title(f"Weighted Correlation ({lbl})",
                     fontsize=12, fontweight="bold")

    fig.suptitle("BMI & Waist Circumference — Correlation Matrix",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, fname)
    plt.show()


def plot_bp_individual_configs(male_2021, female_2021,
                               *, color_male, color_female):
    configs = [
        ("Age of household members", "First SYSTOLIC reading",
         "First Systolic Blood Pressure by Age (95% CI)",
         "Mean Systolic Blood Pressure (mmHg)",
         "Age of Household Members", None, None, None, None,
         "systolic_by_age.png"),
        ("Age of household members", "First DIASTOLIC reading",
         "First Diastolic Blood Pressure by Age (95% CI)",
         "Mean Diastolic Blood Pressure (mmHg)",
         "Age of Household Members", None, None, None, None,
         "diastolic_by_age.png"),
        ("Height_bin", "First SYSTOLIC reading",
         "First Systolic Blood Pressure by Height (95% CI)",
         "Mean Systolic Blood Pressure (mmHg)",
         "Height (cm)", None, None, 140, 190,
         "systolic_by_height.png"),
        ("Height_bin", "First DIASTOLIC reading",
         "First Diastolic Blood Pressure by Height (95% CI)",
         "Mean Diastolic Blood Pressure (mmHg)",
         "Height (cm)", None, None, 140, 190,
         "diastolic_by_height.png"),
    ]

    for xcol, ycol, title, ylabel, xlabel, ylim_lo, ylim_hi, ht_lo, ht_hi, fname in configs:
        fig, ax = plt.subplots(figsize=(8, 6))

        if "Height" in xcol:
            dm = male_2021[male_2021["Height"].between(ht_lo, ht_hi)].copy()
            df = female_2021[female_2021["Height"].between(ht_lo, ht_hi)].copy()
            dm["Height_bin"] = np.round(dm["Height"]).astype(int)
            df["Height_bin"] = np.round(df["Height"]).astype(int)
            ms = group_stats(dm, "Height_bin", ycol)
            fs = group_stats(df, "Height_bin", ycol)
            xkey = "Height_bin"
        else:
            ms = group_stats(male_2021, xcol, ycol)
            fs = group_stats(female_2021, xcol, ycol)
            xkey = xcol

        for stats, clr, lbl in [(ms, color_male, "Male"),
                                 (fs, color_female, "Female")]:
            ax.plot(stats[xkey], stats["mean"], color=clr, label=lbl)
            ax.fill_between(stats[xkey], stats["lo"], stats["hi"],
                            color=clr, alpha=CI_ALPHA, linewidth=0)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        if "Height" not in xcol:
            ax.set_xlim(15, 85)
        if ylim_lo is not None:
            ax.set_ylim(ylim_lo, ylim_hi)
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        fig.tight_layout()
        _save(fig, fname)
        plt.show()


def plot_systolic_by_bmi(male_2021, female_2021, fname="systolic_by_bmi.png",
                         *, color_male, color_female):
    fig, ax = plt.subplots(figsize=(8, 6))

    dm = male_2021[male_2021["BMI"].between(12, 30)].copy()
    dfem = female_2021[female_2021["BMI"].between(12, 30)].copy()
    dm["BMI_bin"] = np.floor(dm["BMI"]).astype(int)
    dfem["BMI_bin"] = np.floor(dfem["BMI"]).astype(int)

    bp = "First SYSTOLIC reading"
    for data, clr, lbl in [(dm, color_male, "Male"),
                            (dfem, color_female, "Female")]:
        s = group_stats(data, "BMI_bin", bp)
        ax.plot(s["BMI_bin"], s["mean"], color=clr, label=lbl)
        ax.fill_between(s["BMI_bin"], s["lo"], s["hi"],
                        color=clr, alpha=CI_ALPHA, linewidth=0)

    ax.set_title("Mean First Systolic Reading by BMI (95% CI)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Body Mass Index (BMI)", fontsize=11)
    ax.set_ylabel("Mean Systolic Blood Pressure (mmHg)", fontsize=11)
    ax.set_ylim(100, 145)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    fig.tight_layout()
    _save(fig, fname)
    plt.show()


def plot_systolic_by_waist(male_2021, female_2021,
                           fname="systolic_by_waist.png",
                           *, color_male, color_female):
    fig, ax = plt.subplots(figsize=(8, 6))
    wc = "Waist curcumference"

    dm = male_2021[male_2021[wc].between(40, 130)].copy()
    dfem = female_2021[female_2021[wc].between(40, 130)].copy()
    dm["WC_bin"] = (dm[wc] // 5) * 5
    dfem["WC_bin"] = (dfem[wc] // 5) * 5

    bp = "First SYSTOLIC reading"
    for data, clr, lbl in [(dm, color_male, "Male"),
                            (dfem, color_female, "Female")]:
        s = group_stats(data, "WC_bin", bp)
        ax.plot(s["WC_bin"], s["mean"], color=clr,
                label=lbl, marker="o", markersize=3)
        ax.fill_between(s["WC_bin"], s["lo"], s["hi"],
                        color=clr, alpha=CI_ALPHA, linewidth=0)

    ax.set_title("Mean First Systolic Reading by Waist Circumference "
                 "(95% CI, 5cm bins)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Waist Circumference (cm)", fontsize=11)
    ax.set_ylabel("Mean Systolic Blood Pressure (mmHg)", fontsize=11)
    ax.set_xlim(40, 130)
    ax.set_ylim(100, 150)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    fig.tight_layout()
    _save(fig, fname)
    plt.show()


def plot_systolic_by_age_bracket(male_2021, female_2021, bracket_col,
                                  n_brackets, title, ylabel, savename,
                                  stratify_sex="female",
                                  ylim_lo=None, ylim_hi=None,
                                  *, color_male, color_female):
    bp_col = "First SYSTOLIC reading"

    if stratify_sex == "female":
        strat_data, ranges = split_into_brackets(
            female_2021[female_2021[bracket_col].notna()],
            bracket_col, n_brackets)
        ref_data = male_2021
        ref_label, ref_color, pal = "Male (all)", color_male, "Reds"
    else:
        strat_data, ranges = split_into_brackets(
            male_2021[male_2021[bracket_col].notna()],
            bracket_col, n_brackets)
        ref_data = female_2021
        ref_label, ref_color, pal = "Female (all)", color_female, "Blues"

    bcn = f"{bracket_col} Bracket"
    strat = strat_data[strat_data[bcn].notnull()].copy()
    ref = ref_data[ref_data[bracket_col].notnull()].copy()

    if hasattr(strat[bcn], "cat"):
        ordered = list(strat[bcn].cat.categories)
    else:
        ordered = sorted(strat[bcn].dropna().unique(), key=str)

    range_labels = {str(b): f"{b} ({r[0]:.1f}–{r[1]:.1f})"
                    for b, r in zip(ordered, ranges)}

    fig, ax = plt.subplots(figsize=(12, 7))

    rs = group_stats(ref, "Age of household members", bp_col)
    ref_line, = ax.plot(rs["Age of household members"], rs["mean"],
                        color=ref_color, linestyle="-", linewidth=2,
                        label=ref_label)
    ax.fill_between(rs["Age of household members"], rs["lo"], rs["hi"],
                    color=ref_color, alpha=CI_ALPHA, linewidth=0)

    palette = sns.color_palette(pal, n_brackets + 2)[2:]
    handles = [ref_line]
    legend_text = [ref_label]

    for bracket, clr in zip(ordered, palette):
        chunk = strat[strat[bcn] == bracket]
        ss = group_stats(chunk, "Age of household members", bp_col)
        h, = ax.plot(ss["Age of household members"], ss["mean"],
                     color=clr, linestyle="--", linewidth=1.5)
        ax.fill_between(ss["Age of household members"], ss["lo"], ss["hi"],
                        color=clr, alpha=CI_ALPHA, linewidth=0)
        handles.append(h)
        legend_text.append(range_labels[str(bracket)])

    q_word = "Quartile" if n_brackets == 4 else "Quintile"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Age (years)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlim(15, 49)
    if ylim_lo is not None:
        ax.set_ylim(ylim_lo, ylim_hi)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend(handles=handles, labels=legend_text,
              title=f"{bracket_col} {q_word}", loc="upper left", fontsize=9)
    fig.tight_layout()
    _save(fig, savename)
    plt.show()


def _annotate_bar(ax, x_pos, value, y_range):
    pct = (value - 1.0) * 100.0
    offset = 0.02 * y_range
    if value >= 1:
        y = value + offset
        va = "bottom"
    else:
        y = value - offset
        va = "top"
    ax.text(x_pos, y, f"{pct:+.1f}%", ha="center", va=va, fontsize=8)


def _plot_single_ratio(ax, male_vals, female_vals, grp, baseline, var,
                       title_prefix="",
                       *, color_male, color_female):
    x = np.arange(len(HTN_ORDER))
    w = 0.3
    all_vals = list(male_vals.values) + list(female_vals.values)
    lo = min(0.0, min(all_vals) * 0.9)
    hi = max(2.0, max(all_vals) * 1.1)
    span = hi - lo

    ax.bar(x - 0.15, male_vals, width=w, label="Male", color=color_male)
    ax.bar(x + 0.15, female_vals, width=w, label="Female", color=color_female)
    ax.axhline(1, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(lo, hi)

    for i, stage in enumerate(HTN_ORDER):
        _annotate_bar(ax, x[i] - 0.15, male_vals[stage], span)
        _annotate_bar(ax, x[i] + 0.15, female_vals[stage], span)

    ax.set_xlabel("Hypertension Stage")
    ax.set_ylabel(f"Relative Ratio:\n{grp} % / {baseline} %", fontsize=9)
    ax.set_title(f"{title_prefix}{var} Category: {grp}\n"
                 f"Baseline: {baseline}", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(HTN_ORDER, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=8)


def plot_ratio_charts(ratio_plots, *, prefix="",
                      color_male, color_female):
    for idx, (var, grp, baseline, m_rel, f_rel) in enumerate(ratio_plots):
        fig, ax = plt.subplots(figsize=(8, 5))

        empty = pd.Series([0] * len(HTN_ORDER), index=HTN_ORDER)
        mv = m_rel.loc[grp] if grp in m_rel.index else empty
        fv = f_rel.loc[grp] if grp in f_rel.index else empty

        tp = f"{prefix} — " if prefix else ""
        _plot_single_ratio(ax, mv, fv, grp, baseline, var, tp,
                           color_male=color_male, color_female=color_female)

        safe_var = str(var).lower().replace(" ", "_").replace(":", "").replace(",", "")
        safe_grp = str(grp).lower().replace(" ", "_").replace("'", "").replace(",", "")
        tag = "nfhs4_" if "NFHS-4" in prefix else ""
        fname = f"ratio_{tag}{safe_var}_{safe_grp}.png"

        fig.tight_layout()
        _save(fig, fname)
        print(f"[{PANEL_LABELS[idx]}] Saved figs/{fname}")
        plt.show()


def plot_ratio_comparison(shared_vars, comparisons_5, comparisons_4,
                          data_male_5, data_female_5,
                          data_male_4, data_female_4,
                          *, color_male, color_female):
    from .analysis import compute_relative_ratios_by_stage

    for var in shared_vars:
        bl5, mp5 = comparisons_5[var]["baseline"], comparisons_5[var]["mapping"]
        bl4, mp4 = comparisons_4[var]["baseline"], comparisons_4[var]["mapping"]

        results = [
            compute_relative_ratios_by_stage(d, var, bl, mp)
            for d, bl, mp in [
                (data_male_5, bl5, mp5), (data_female_5, bl5, mp5),
                (data_male_4, bl4, mp4), (data_female_4, bl4, mp4),
            ]
        ]
        m5_rel, f5_rel, m4_rel, f4_rel = [r[0] for r in results]

        if any(r is None for r in [m5_rel, f5_rel, m4_rel, f4_rel]):
            continue

        bl_set = set()
        for bl in [bl5, bl4]:
            if isinstance(bl, list):
                bl_set.update(str(b).strip().lower() for b in bl)
            else:
                bl_set.add(str(bl).strip().lower())

        shared_groups = sorted(
            (set(m5_rel.index) | set(f5_rel.index))
            & (set(m4_rel.index) | set(f4_rel.index))
            - bl_set
        )

        bl5_label = bl5[0] if isinstance(bl5, list) else bl5
        bl4_label = bl4[0] if isinstance(bl4, list) else bl4

        for grp in shared_groups:
            fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

            panel_info = [
                (axes[0], "NFHS-5 (2019–21)", m5_rel, f5_rel, bl5_label),
                (axes[1], "NFHS-4 (2015–16)", m4_rel, f4_rel, bl4_label),
            ]

            for ax, label, m_rel, f_rel, bl in panel_info:
                empty = pd.Series([0] * len(HTN_ORDER), index=HTN_ORDER)
                mv = m_rel.loc[grp] if grp in m_rel.index else empty
                fv = f_rel.loc[grp] if grp in f_rel.index else empty

                x = np.arange(len(HTN_ORDER))
                ax.bar(x - 0.15, mv, width=0.3, label="Male", color=color_male)
                ax.bar(x + 0.15, fv, width=0.3, label="Female", color=color_female)
                ax.axhline(1, color="black", linestyle="--", linewidth=1)

                all_v = list(mv.values) + list(fv.values)
                lo = min(0.0, min(all_v) * 0.9)
                hi = max(2.0, max(all_v) * 1.1)
                ax.set_ylim(lo, hi)

                for si, stage in enumerate(HTN_ORDER):
                    for xoff, val in [(-0.15, mv[stage]), (0.15, fv[stage])]:
                        pct = (val - 1.0) * 100
                        sign = 1 if val >= 1 else -1
                        y = val + 0.02 * (hi - lo) * sign
                        va = "bottom" if val >= 1 else "top"
                        ax.text(x[si] + xoff, y, f"{pct:+.1f}%",
                                ha="center", va=va, fontsize=7)

                ax.set_title(f"{label}\nBaseline: {bl}",
                             fontsize=10, fontweight="bold")
                ax.set_xlabel("Hypertension Stage")
                ax.set_xticks(x)
                ax.set_xticklabels(HTN_ORDER, rotation=45, ha="right", fontsize=8)
                ax.legend(fontsize=8)

            axes[0].set_ylabel(f"Relative Ratio:\n{grp} % / baseline %", fontsize=9)
            fig.suptitle(f"{var}: {grp}", fontsize=12, fontweight="bold")
            fig.tight_layout()

            sv = var.lower().replace(" ", "_").replace(":", "").replace(",", "")
            sg = grp.lower().replace(" ", "_").replace("'", "").replace(",", "")
            _save(fig, f"ratio_compare_{sv}_{sg}.png")
            plt.show()


def plot_ecdf(df_both, conditions=None, xmax=210.0,
              *, color_male, color_female):
    if conditions is None:
        conditions = [
            ("has hypertension", "ecdf_hypertension.png"),
            ("has heart disease", "ecdf_heart_disease.png"),
            ("has diabetes", "ecdf_diabetes.png"),
            ("has chronic kidney disease", "ecdf_chronic_kidney_disease.png"),
            ("has cancer", "ecdf_cancer.png"),
            ("has chronic respiratory disease", "ecdf_chronic_respiratory_disease.png"),
            ("has thyroid disorder", "ecdf_thyroid_disorder.png"),
        ]

    sbp_col = "First systolic reading"

    for cond, fname in conditions:
        fig, ax = plt.subplots(figsize=(9, 5))

        ks_pval = disease_sbp_pvalue(df_both, cond, sbp_col)

        curves = []
        for sex, clr, lbl in [("female", color_female, "Women"),
                                ("male", color_male, "Men")]:
            mask = (df_both["sex"] == sex) & (df_both[cond] == 1)
            xv = pd.to_numeric(df_both.loc[mask, sbp_col], errors="coerce")
            wv = pd.to_numeric(df_both.loc[mask, "weight"], errors="coerce")

            ok = np.isfinite(xv) & np.isfinite(wv) & (wv > 0)
            xc, wc = xv[ok].values, wv[ok].values

            X, F = bal_ecdf(xc, wc)
            keep = X <= xmax
            Xp, Fp = X[keep], F[keep]

            lo_f, hi_f = ecdf_ci_from_weights(Fp, wc)
            lo_pct, hi_pct = lo_f * 100, hi_f * 100

            med = float(bal_wquantile(pd.Series(xc), [0.5],
                                      pd.Series(wc)).iloc[0, 0])

            curves.append((Xp, Fp, lo_pct, hi_pct, med, clr, lbl))

        for Xp, Fp, lo, hi, med, clr, lbl in curves:
            if Xp.size == 0:
                continue
            ax.plot(Xp, Fp * 100, color=clr, label=lbl)
            ax.fill_between(Xp, lo, hi, color=clr, alpha=0.18, linewidth=0)

            if np.isfinite(med):
                ax.axvline(med, linestyle="--", linewidth=1, color=clr, alpha=0.7)
                side = -15 if lbl == "Women" else 15
                ax.annotate(f"{med:.1f}", xy=(med, 0),
                            xycoords=ax.get_xaxis_transform(),
                            xytext=(side, -30), textcoords="offset points",
                            ha="center", va="top", color=clr,
                            fontsize=12, fontweight="bold")

        for pct in range(0, 101, 10):
            ax.axhline(pct, linestyle="--", linewidth=0.5, color="#cccccc")

        nice_name = cond.replace("has ", "").title()
        pstr = f"{ks_pval:.2e}" if ks_pval >= 1e-99 else f"{ks_pval:.1e}"
        ax.set_title(f"Cumulative SBP \u2014 {nice_name} (95% CI)\n"
                     f"KS test p-value (Women vs Men): {pstr}",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("SBP (mmHg)")
        ax.set_ylabel("Cumulative % ≤ SBP")
        ax.set_xlim(80, 180)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=10)
        ax.set_xticks(list(range(80, 181, 10)))
        ax.set_xticklabels([str(v) for v in range(80, 181, 10)])
        ax.set_yticks(list(range(0, 101, 10)))
        ax.set_yticklabels([f"{v}%" for v in range(0, 101, 10)])

        fig.tight_layout()
        _save(fig, fname)
        plt.show()


def plot_disease_sbp_table(table, disease_name, fname, pvalue=None):
    n_cols = len(table.columns)
    fig_width = max(12, n_cols * 1.6)
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    ax.axis("off")

    tbl = ax.table(cellText=table.values.tolist(),
                   colLabels=list(table.columns),
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.5)

    for j in range(n_cols):
        tbl[0, j].set_facecolor("#d4e6f1")
        tbl[0, j].set_text_props(fontweight="bold")

    if pvalue is None:
        pvalue = table.attrs.get("ks_pvalue", None) if hasattr(table, "attrs") else None
    title = f"SBP Distribution among Self-reported {disease_name}"
    if pvalue is not None:
        pstr = f"{pvalue:.2e}" if pvalue >= 1e-99 else f"{pvalue:.1e}"
        title += f"\nKS test p-value (Women vs Men): {pstr}"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)
    fig.tight_layout()
    _save(fig, fname)
    plt.show()




def plot_composite_heatmap(comparisons, suptitle, filename, vmin=-8, vmax=8):
    nrows = len(comparisons)
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 4.5 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    for i, (base, comp, label) in enumerate(comparisons):
        res = get_mean_bp_by_age_bmi(base, comp, bmi_dict=BMI_RANGES)
        df = prepare_heatmap_data(res)
        for j, (col, bp_lbl) in enumerate([
            ('Systolic Difference', 'Systolic'),
            ('Diastolic Difference', 'Diastolic'),
        ]):
            ax = axes[i, j]
            pivot = df.pivot(index='BMI', columns='Age', values=col)
            sns.heatmap(pivot, annot=False, cmap='coolwarm', center=0,
                        vmin=vmin, vmax=vmax, linewidths=.5, ax=ax)
            ax.set_title(f'{bp_lbl} BP Difference ({label})')
   
    plt.tight_layout()
    _save(fig, f'{filename}.png')
    plt.show()


def _coarse_panel(ax, stats, title, cmap='coolwarm', vmin=110, vmax=128, show_n=True, center=None):
    values = np.full((3, 3), np.nan)
    annot_text = np.empty((3, 3), dtype=object)
    total_n = 0
    for i, bmi in enumerate(BMI_ORDER):
        for j, age in enumerate(AGE_ORDER):
            if (bmi, age) in stats:
                v = stats[(bmi, age)]['value']
                n = stats[(bmi, age)]['n']
                values[i, j] = v
                total_n += n
                annot_text[i, j] = f"{v:.1f}\n(n={n})" if show_n else f"{v:.1f}"
            else:
                annot_text[i, j] = ""
    df_grid = pd.DataFrame(values, index=BMI_ORDER, columns=AGE_ORDER)
    sns.heatmap(df_grid, annot=annot_text, fmt='', cmap=cmap, center=center,
                linewidths=1, ax=ax, vmin=vmin, vmax=vmax,
                annot_kws={'fontsize': 10, 'fontweight': 'bold'})
    ax.set_title(title, fontsize=9)
    ax.set_ylabel('BMI Group')
    ax.set_xlabel('Age Group')
    return total_n


def _coarse_diff_panel(ax, stats_base, stats_comp, title, vmin=-5, vmax=5):
    values = np.full((3, 3), np.nan)
    annot_text = np.empty((3, 3), dtype=object)
    for i, bmi in enumerate(BMI_ORDER):
        for j, age in enumerate(AGE_ORDER):
            if (bmi, age) in stats_base and (bmi, age) in stats_comp:
                d = stats_comp[(bmi, age)]['value'] - stats_base[(bmi, age)]['value']
                values[i, j] = d
                annot_text[i, j] = f"{d:.1f}"
            else:
                annot_text[i, j] = ""
    df_grid = pd.DataFrame(values, index=BMI_ORDER, columns=AGE_ORDER)
    sns.heatmap(df_grid, annot=annot_text, fmt='', cmap='coolwarm', center=0,
                linewidths=1, ax=ax, vmin=vmin, vmax=vmax,
                annot_kws={'fontsize': 11, 'fontweight': 'bold'})
    ax.set_title(title, fontsize=9)
    ax.set_ylabel('BMI Group')
    ax.set_xlabel('Age Group')


def plot_coarse_comparison_row(ax_row, n5_data, n4_data, bp_col, label_n5, label_n4, gender, period,
                                bp_label='Systolic', abs_vmin=110, abs_vmax=128, diff_vmin=-5, diff_vmax=5):
    stats_n5 = get_coarse_stats(n5_data, bp_col)
    stats_n4 = get_coarse_stats(n4_data, bp_col)
    n_n5 = _coarse_panel(ax_row[0], stats_n5,
                          f'{bp_label} BP Comparison ({period})\n{label_n5} ({gender})',
                          cmap='coolwarm', vmin=abs_vmin, vmax=abs_vmax, show_n=True, center=(abs_vmin+abs_vmax)/2)
    n_n4 = _coarse_panel(ax_row[1], stats_n4,
                          f'{bp_label} BP Comparison ({period})\n{label_n4} ({gender})',
                          cmap='coolwarm', vmin=abs_vmin, vmax=abs_vmax, show_n=True, center=(abs_vmin+abs_vmax)/2)
    _coarse_diff_panel(ax_row[2], stats_n4, stats_n5,
                        f'{bp_label} BP Comparison ({period})\nDifference ({label_n5} - {label_n4}) ({gender})',
                        vmin=diff_vmin, vmax=diff_vmax)
    ax_row[0].set_title(ax_row[0].get_title() + f'\nTotal N = {n_n5}', fontsize=9)
    ax_row[1].set_title(ax_row[1].get_title() + f'\nTotal N = {n_n4}', fontsize=9)
    return n_n5, n_n4


def plot_coarse_did(datasets, filename, vmin=-2, vmax=2):

    fig, axes = plt.subplots(2, len(datasets), figsize=(7 * len(datasets), 10))
    for j, (gender, pre_n5, post_n5, pre_n4, post_n4) in enumerate(datasets):
        for i, (bp_col, bp_label) in enumerate([(SBP_COL, 'Mean_Systolic'), (DBP_COL, 'Mean_Diastolic')]):
            did_stats = compute_coarse_did(pre_n5, post_n5, pre_n4, post_n4, bp_col)

            ax = axes[i, j]
            values = np.full((3, 3), np.nan)
            annot_text = np.empty((3, 3), dtype=object)
            for ii, bmi in enumerate(BMI_ORDER):
                for jj, age in enumerate(AGE_ORDER):
                    if (bmi, age) in did_stats:
                        v = did_stats[(bmi, age)]['value']
                        values[ii, jj] = v
                        annot_text[ii, jj] = f"{v:.1f}"
                    else:
                        annot_text[ii, jj] = ""
            df_grid = pd.DataFrame(values, index=BMI_ORDER, columns=AGE_ORDER)
            sns.heatmap(df_grid, annot=annot_text, fmt='', cmap='coolwarm', center=0,
                        linewidths=1, ax=ax, vmin=vmin, vmax=vmax,
                        annot_kws={'fontsize': 12, 'fontweight': 'bold'})
            ax.set_title(f'BP Diff Change ({bp_label}) Change ({gender})', fontsize=11)
            ax.set_ylabel('BMI Group')
            ax.set_xlabel('Age Group')

    fig.text(0.01, 0.97, 'B', fontsize=20, fontweight='bold', va='top')
    plt.tight_layout()
    _save(fig, f'{filename}.png')
    plt.show()


def plot_coarse_absolute_grid(rows_config, bp_col, bp_label, filename,
                               abs_vmin=110, abs_vmax=130, diff_vmin=-6, diff_vmax=6,
                               suptitle=None):
    nrows = len(rows_config)
    fig, axes = plt.subplots(nrows, 3, figsize=(20, 5 * nrows))
    for i, (n5, n4, lbl_n5, lbl_n4, gender, period) in enumerate(rows_config):
        plot_coarse_comparison_row(axes[i], n5, n4, bp_col, lbl_n5, lbl_n4, gender, period,
                                   bp_label=bp_label, abs_vmin=abs_vmin, abs_vmax=abs_vmax,
                                   diff_vmin=diff_vmin, diff_vmax=diff_vmax)
    if suptitle:
        plt.suptitle(suptitle, fontsize=14, y=1.01)
    plt.tight_layout()
    _save(fig, f'{filename}.png')
    plt.show()
