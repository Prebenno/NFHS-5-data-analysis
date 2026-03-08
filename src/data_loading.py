import gc
import os

import numpy as np
import pandas as pd
import pyreadstat

from balance.stats_and_plots.weighted_stats import weighted_mean as bal_wmean


_NFHS5_COLS = [
    "hhid", "hvidx",
    "hv001", "hv002",
    "hv005", "hv021", "hv022", "hv025", "hv006", "hv007", "hv270",
    "hv104", "hv105", "hv106",
    "shb18s", "shb18d", "shb25s", "shb25d", "shb29s", "shb29d",
    "shb14a", "shb14b", "shb14c", "shb14d",
    "shb17",
    "ha40", "hb40", "ha3", "hb3",
    "sh305", "sh49", "sh25",
]

_NFHS5_RENAME = {
    "shb18s": "First SYSTOLIC reading",
    "shb18d": "First DIASTOLIC reading",
    "shb25s": "Second SYSTOLIC reading",
    "shb25d": "Second DIASTOLIC reading",
    "shb29s": "Third SYSTOLIC reading",
    "shb29d": "Third DIASTOLIC reading",
    "shb14b": "30 minutes prior to BP measure: had coffee, tea",
    "shb14c": "30 minutes prior to BP measure: smoked any tobacco product",
    "shb14d": "30 minutes prior to BP measure: use any other type of tobacco",
    "shb14a": "30 minutes prior to BP measure: eaten",
    "hv105":  "Age of household members",
    "ha40":   "Body Mass Index",
    "hb40":   "Body Mass Index.1",
    "ha3":    "Height_female",
    "hb3":    "Height_male",
    "hv104":  "Sex of household member",
    "sh305":  "Waist curcumference",
    "hv106":  "Highest educational level attained",
    "sh25":   "Smokes or uses tobacco",
    "shb17":  "Time of first BP reading",
    "hv005":  "Household sample weight (6 decimals)",
    "hv021":  "Primary sampling unit",
    "hv022":  "Sample strata for sampling errors",
    "hv025":  "Type of place of residence",
    "hv006":  "Month of interview",
    "hv007":  "Year of interview",
    "hv270":  "Wealth index combined",
    "sh49":   "Caste or tribe",
}

_NFHS5_LABEL_COLS = [
    "hv104", "hv106", "sh25",
    "shb14a", "shb14b", "shb14c", "shb14d",
    "hv270", "sh49", "hv025",
]

_ERROR_CODES_N5 = {
    "shb18s": [994, 995, 996, 999],
    "shb18d": [994, 995, 996, 999],
    "shb25s": [994, 995, 996, 999],
    "shb25d": [994, 995, 996, 999],
    "shb29s": [994, 995, 996, 999],
    "shb29d": [994, 995, 996, 999],
    "shb14a": [9],
    "shb14b": [9],
    "shb14c": [9],
    "shb14d": [9],
    "ha40": [9998, 9999],
    "hb40": [9998, 9999],
    "ha3":  [9998, 9999],
    "hb3":  [9998, 9999],
    "sh305": [999.5, 999.6, 999.9],
    "hv104": [9],
    "hv105": [98, 99],
}

BP_COLS = [
    "First SYSTOLIC reading",
    "First DIASTOLIC reading",
    "Second SYSTOLIC reading",
    "Second DIASTOLIC reading",
    "Third SYSTOLIC reading",
    "Third DIASTOLIC reading",
]

IMPLAUSIBLE_THRESHOLDS = {
    "Height": 250.0,
    "Waist curcumference": 200.0,
    "BMI": 60.0,
}


def _split_by_sex(df, sex_col="Sex of household member"):
    sex = df[sex_col].astype(str).str.strip().str.lower()
    male = df[sex.eq("male")].copy()
    female = df[sex.eq("female")].copy()
    return male, female


def load_nfhs5(dta_path="Data/household_2021_recode/IAPR7EFL.DTA", *, verbose=True):
    gc.collect()

    hdr, meta = pyreadstat.read_dta(dta_path, row_limit=1, apply_value_formats=False)
    all_cols = [c.lower() for c in hdr.columns]
    del hdr
    gc.collect()

    use_cols = [c for c in _NFHS5_COLS if c in all_cols]

    if verbose:
        print(f"NFHS-5: {len(use_cols)}/{len(_NFHS5_COLS)} columns found")
        missing = set(_NFHS5_COLS) - set(use_cols)
        if missing:
            print(f"  Missing: {missing}")

    df, meta = pyreadstat.read_dta(dta_path, usecols=use_cols, apply_value_formats=False)
    df.columns = df.columns.str.lower()

    if verbose:
        print(f"Loaded {len(df):,} person rows")

    for col, bad in _ERROR_CODES_N5.items():
        if col in df.columns:
            nums = pd.to_numeric(df[col], errors="coerce")
            df.loc[nums.isin(bad), col] = np.nan

    for c in _NFHS5_LABEL_COLS:
        if c in df.columns and c in meta.variable_value_labels:
            df[c] = df[c].map(meta.variable_value_labels[c]).fillna(df[c])

    rename_map = {k: v for k, v in _NFHS5_RENAME.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    df["weight_raw"] = pd.to_numeric(
        df.get("Household sample weight (6 decimals)"), errors="coerce")
    df["weight"] = df["weight_raw"] / 1_000_000.0
    df["psu"] = df.get("Primary sampling unit")
    df["strata"] = df.get("Sample strata for sampling errors")

    df["Age of household members"] = pd.to_numeric(df["Age of household members"], errors="coerce")
    df = df.dropna(subset=["Age of household members"])

    is_male = df["Sex of household member"].astype(str).str.strip().str.lower().eq("male")

    bmi_f = pd.to_numeric(df.get("Body Mass Index"), errors="coerce")
    bmi_m = pd.to_numeric(df.get("Body Mass Index.1"), errors="coerce")
    df["BMI"] = np.where(is_male, bmi_m, bmi_f) / 100.0

    ht_f = pd.to_numeric(df.get("Height_female"), errors="coerce")
    ht_m = pd.to_numeric(df.get("Height_male"), errors="coerce")
    df["Height"] = np.where(is_male, ht_m, ht_f) / 10.0

    for c in BP_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "Waist curcumference" in df.columns:
        df["Waist curcumference"] = pd.to_numeric(df["Waist curcumference"], errors="coerce")

    df = df.dropna(subset=BP_COLS, how="all").copy()

    for col_name, thresh in IMPLAUSIBLE_THRESHOLDS.items():
        if col_name in df.columns:
            bad = df[col_name] > thresh
            if bad.sum():
                df.loc[bad, col_name] = np.nan
                if verbose:
                    print(f"Cleaned {bad.sum():,} implausible '{col_name}' (>{thresh})")

    male_2021, female_2021 = _split_by_sex(df)

    if verbose:
        print(f"\nTotal persons with BP: {len(df):,}")
        print(f"Male: {len(male_2021):,}  |  Female: {len(female_2021):,}")
        print(f"Height — Male: {male_2021['Height'].notna().sum():,}  "
              f"Female: {female_2021['Height'].notna().sum():,}")
        print(f"BMI ({df['BMI'].notna().sum():,})")
        if "Waist curcumference" in df.columns:
            print(f"Waist ({df['Waist curcumference'].notna().sum():,})")

    return df, male_2021, female_2021


_NFHS4_COLS = [
    "hv001", "hv002", "hvidx",
    "hv005", "hv021", "hv022", "hv025", "hv006", "hv007", "hv270",
    "hv104", "hv105", "hv106",
    "shb16s", "shb16d", "shb23s", "shb23d", "shb27s", "shb27d",
    "shb12a", "shb12b", "shb12c", "shb12d",
    "shb15h", "shb15m",
    "ha40", "hb40", "ha3", "hb3", "ha35", "hb35",
    "sh36",
]

_NFHS4_RENAME = {
    "shb16s": "First SYSTOLIC reading",
    "shb16d": "First DIASTOLIC reading",
    "shb23s": "Second SYSTOLIC reading",
    "shb23d": "Second DIASTOLIC reading",
    "shb27s": "Third SYSTOLIC reading",
    "shb27d": "Third DIASTOLIC reading",
    "shb12a": "30 minutes prior to BP measure: eaten",
    "shb12b": "30 minutes prior to BP measure: had coffee, tea",
    "shb12c": "30 minutes prior to BP measure: smoked any tobacco product",
    "shb12d": "30 minutes prior to BP measure: use any other type of tobacco",
    "hv105":  "Age of household members",
    "ha40":   "Body Mass Index",
    "hb40":   "Body Mass Index.1",
    "ha3":    "Height_female",
    "hb3":    "Height_male",
    "hv104":  "Sex of household member",
    "hv106":  "Highest educational level attained",
    "hv005":  "Household sample weight (6 decimals)",
    "hv021":  "Primary sampling unit",
    "hv022":  "Sample strata for sampling errors",
    "hv025":  "Type of place of residence",
    "hv270":  "Wealth index combined",
    "sh36":   "Caste or tribe",
}

_ERROR_CODES_N4 = {
    "First SYSTOLIC reading":  [994, 995, 996, 999],
    "First DIASTOLIC reading": [994, 995, 996, 999],
    "Second SYSTOLIC reading":  [994, 995, 996, 999],
    "Second DIASTOLIC reading": [994, 995, 996, 999],
    "Third SYSTOLIC reading":  [994, 995, 996, 999],
    "Third DIASTOLIC reading": [994, 995, 996, 999],
    "Body Mass Index":   [9998, 9999],
    "Body Mass Index.1": [9998, 9999],
    "Height_female": [9994, 9995, 9996, 9999],
    "Height_male":   [9994, 9995, 9996, 9999],
}


def load_nfhs4(dta_path="Data/Household_2016/IAPR74FL.DTA", *, verbose=True):
    gc.collect()

    hdr, meta = pyreadstat.read_dta(dta_path, row_limit=1, apply_value_formats=False)
    all_cols = [c.lower() for c in hdr.columns]
    use = [c for c in _NFHS4_COLS if c in all_cols]
    del hdr
    gc.collect()

    if verbose:
        print(f"NFHS-4: {len(use)}/{len(_NFHS4_COLS)} columns found")
        missing = set(_NFHS4_COLS) - set(use)
        if missing:
            print(f"  Missing: {missing}")

    df, meta = pyreadstat.read_dta(dta_path, usecols=use, apply_value_formats=False)
    df.columns = df.columns.str.lower()

    if verbose:
        print(f"Loaded NFHS-4: {len(df):,} person-rows")

    df = df.rename(columns={k: v for k, v in _NFHS4_RENAME.items() if k in df.columns})

    for c in ["hv270", "sh36", "hv025", "hv104", "hv106",
              "shb12a", "shb12b", "shb12c", "shb12d"]:
        pretty = _NFHS4_RENAME.get(c, c)
        if pretty in df.columns and c in meta.variable_value_labels:
            df[pretty] = df[pretty].map(meta.variable_value_labels[c]).fillna(df[pretty])

    for bp30 in [
        "30 minutes prior to BP measure: smoked any tobacco product",
        "30 minutes prior to BP measure: eaten",
        "30 minutes prior to BP measure: had coffee, tea",
        "30 minutes prior to BP measure: use any other type of tobacco",
    ]:
        if bp30 in df.columns:
            v = df[bp30].astype(str).str.strip()
            if v.dropna().isin(['0', '0.0', '1', '1.0', 'nan']).all():
                df[bp30] = v.map({'0': 'no', '0.0': 'no', '1': 'yes', '1.0': 'yes'})

    if "shb15h" in df.columns and "shb15m" in df.columns:
        h = pd.to_numeric(df["shb15h"], errors="coerce")
        m = pd.to_numeric(df["shb15m"], errors="coerce")
        df["Time of first BP reading"] = h * 100 + m
        df.drop(columns=["shb15h", "shb15m"], inplace=True, errors="ignore")

    if "ha35" in df.columns and "hb35" in df.columns:
        ha35 = pd.to_numeric(df["ha35"], errors="coerce")
        hb35 = pd.to_numeric(df["hb35"], errors="coerce")
        is_male_tmp = df["Sex of household member"].astype(str).str.strip().str.lower().eq("male")
        smokes_f = (ha35 > 0) & (ha35 < 95)
        smokes_m = (hb35 > 0) & (hb35 < 95)
        df["Smokes or uses tobacco"] = np.where(is_male_tmp, smokes_m, smokes_f)
        df["Smokes or uses tobacco"] = df["Smokes or uses tobacco"].map(
            {True: "yes", False: "no", 1: "yes", 0: "no"})
        df.drop(columns=["ha35", "hb35"], inplace=True, errors="ignore")

    for col, bad in _ERROR_CODES_N4.items():
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce")
            df.loc[v.isin(bad), col] = np.nan

    df["weight_raw"] = pd.to_numeric(
        df.get("Household sample weight (6 decimals)"), errors="coerce")
    df["weight"] = df["weight_raw"] / 1_000_000.0
    df["psu"] = df.get("Primary sampling unit")
    df["strata"] = df.get("Sample strata for sampling errors")

    df["Age of household members"] = pd.to_numeric(df["Age of household members"], errors="coerce")

    is_male = df["Sex of household member"].astype(str).str.strip().str.lower().eq("male")

    bmi_f = pd.to_numeric(df.get("Body Mass Index"), errors="coerce")
    bmi_m = pd.to_numeric(df.get("Body Mass Index.1"), errors="coerce")
    df["BMI"] = np.where(is_male, bmi_m, bmi_f) / 100.0

    ht_f = pd.to_numeric(df.get("Height_female"), errors="coerce")
    ht_m = pd.to_numeric(df.get("Height_male"), errors="coerce")
    df["Height"] = np.where(is_male, ht_m, ht_f) / 10.0

    for c in BP_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=BP_COLS, how="all").copy()

    for col_name, thresh in {"Height": 250.0, "BMI": 60.0}.items():
        if col_name in df.columns:
            bad = df[col_name] > thresh
            if bad.sum():
                df.loc[bad, col_name] = np.nan
                if verbose:
                    print(f"NFHS-4: Cleaned {bad.sum():,} implausible '{col_name}'")

    male_2016, female_2016 = _split_by_sex(df)

    if verbose:
        print(f"\nNFHS-4 persons with BP: {len(df):,}")
        print(f"Male: {len(male_2016):,}  |  Female: {len(female_2016):,}")
        print(f"BMI — Male: {male_2016['BMI'].notna().sum():,}  "
              f"Female: {female_2016['BMI'].notna().sum():,}")
        print(f"Height — Male: {male_2016['Height'].notna().sum():,}  "
              f"Female: {female_2016['Height'].notna().sum():,}")

    return df, male_2016, female_2016


_INDIVIDUAL_COLS_M = [
    "sm627a", "sm627b", "sm627c", "sm627d", "sm627e", "sm627f", "sm627g",
    "mv005", "mv021", "mv022", "mv024",
    "smb18s", "smb25s", "smb29s", "mv012",
]

_INDIVIDUAL_COLS_F = [
    "s728a", "s728b", "s728c", "s728d", "s728e", "s728f", "s728g",
    "v005", "v021", "v022", "v024",
    "sb18s", "sb25s", "sb29s", "v012",
]

_DISEASE_RENAME_M = {
    "sm627a": "has diabetes",
    "sm627b": "has hypertension",
    "sm627c": "has chronic respiratory disease",
    "sm627d": "has thyroid disorder",
    "sm627e": "has heart disease",
    "sm627f": "has cancer",
    "sm627g": "has chronic kidney disease",
}

_DISEASE_RENAME_F = {
    "s728a": "has diabetes",
    "s728b": "has hypertension",
    "s728c": "has chronic respiratory disease",
    "s728d": "has thyroid disorder",
    "s728e": "has heart disease",
    "s728f": "has cancer",
    "s728g": "has chronic kidney disease",
}

_BP_ERROR_CODES_INDIVIDUAL = [994, 995, 996, 999]

DISEASE_COLS = list(_DISEASE_RENAME_M.values())


def load_individual_recode(
    dta_m="Data/Mens_2021/IAMR7EFL.DTA",
    dta_f="Data/Individual_2021/IAIR7EFL.DTA",
    *,
    verbose=True,
):
    df_men, _ = pyreadstat.read_dta(dta_m, usecols=_INDIVIDUAL_COLS_M, apply_value_formats=False)
    df_women, _ = pyreadstat.read_dta(dta_f, usecols=_INDIVIDUAL_COLS_F, apply_value_formats=False)

    df_men["weight"]  = df_men["mv005"] / 1_000_000.0
    df_women["weight"] = df_women["v005"] / 1_000_000.0
    df_men["psu"]     = df_men["mv021"]
    df_women["psu"]   = df_women["v021"]
    df_men["strata"]  = df_men["mv022"]
    df_women["strata"] = df_women["v022"]

    df_men   = df_men.rename(columns=_DISEASE_RENAME_M).assign(sex="male")
    df_women = df_women.rename(columns=_DISEASE_RENAME_F).assign(sex="female")

    df_both = pd.concat([df_men, df_women], ignore_index=True)

    df_both["First systolic reading"] = df_both[["sb18s", "smb18s"]].apply(
        pd.to_numeric, errors="coerce").bfill(axis=1).iloc[:, 0]
    df_both["Second systolic reading"] = df_both[["sb25s", "smb25s"]].apply(
        pd.to_numeric, errors="coerce").bfill(axis=1).iloc[:, 0]
    df_both["Third systolic reading"] = df_both[["sb29s", "smb29s"]].apply(
        pd.to_numeric, errors="coerce").bfill(axis=1).iloc[:, 0]

    for c in DISEASE_COLS:
        x = pd.to_numeric(df_both[c], errors="coerce")
        df_both[c] = np.where(x == 1, 1.0, np.where(x == 0, 0.0, np.nan))

    for bp_c in ["First systolic reading", "Second systolic reading", "Third systolic reading"]:
        v = pd.to_numeric(df_both[bp_c], errors="coerce")
        df_both.loc[v.isin(_BP_ERROR_CODES_INDIVIDUAL), bp_c] = np.nan
        df_both[bp_c] = pd.to_numeric(df_both[bp_c], errors="coerce")

    sbp_cols = ["First systolic reading", "Second systolic reading", "Third systolic reading"]
    df_both["Mean systolic reading"] = np.round(df_both[sbp_cols].mean(axis=1))

    if verbose:
        print(f"Individual recode: Men {len(df_men):,} + Women {len(df_women):,} "
              f"= {len(df_both):,}")
        print("Disease prevalence (weighted %):")
        for c in DISEASE_COLS:
            w = df_both["weight"].to_numpy()
            v = df_both[c].to_numpy()
            m = np.isfinite(v) & np.isfinite(w) & (w > 0)
            pct = np.sum(w[m] * v[m]) / np.sum(w[m]) * 100 if m.any() else np.nan
            print(f"  {c}: {pct:.2f}%")

    return df_both


_INDIVIDUAL_COLS_N4_M = [
    "sm622a", "sm622d", "sm622e",
    "mv005", "mv021", "mv022", "mv024",
    "smb16s", "smb23s", "smb27s", "mv012",
]

_INDIVIDUAL_COLS_N4_F = [
    "s723a", "s723d", "s723e",
    "v005", "v021", "v022", "v024",
    "sb16s", "sb23s", "sb27s", "v012",
]

_DISEASE_RENAME_N4_M = {
    "sm622a": "has diabetes",
    "sm622d": "has heart disease",
    "sm622e": "has cancer",
}

_DISEASE_RENAME_N4_F = {
    "s723a": "has diabetes",
    "s723d": "has heart disease",
    "s723e": "has cancer",
}

DISEASE_COLS_N4 = ["has diabetes", "has heart disease", "has cancer"]


def load_individual_recode_nfhs4(
    dta_m="Data/Mens_2016/IAMR74FL.DTA",
    dta_f="Data/Individual_2016/IAIR74FL.DTA",
    *,
    verbose=True,
):
    df_men, _ = pyreadstat.read_dta(dta_m, usecols=_INDIVIDUAL_COLS_N4_M, apply_value_formats=False)
    df_women, _ = pyreadstat.read_dta(dta_f, usecols=_INDIVIDUAL_COLS_N4_F, apply_value_formats=False)

    df_men["weight"]  = df_men["mv005"] / 1_000_000.0
    df_women["weight"] = df_women["v005"] / 1_000_000.0
    df_men["psu"]     = df_men["mv021"]
    df_women["psu"]   = df_women["v021"]
    df_men["strata"]  = df_men["mv022"]
    df_women["strata"] = df_women["v022"]

    df_men   = df_men.rename(columns=_DISEASE_RENAME_N4_M).assign(sex="male")
    df_women = df_women.rename(columns=_DISEASE_RENAME_N4_F).assign(sex="female")

    df_both = pd.concat([df_men, df_women], ignore_index=True)

    df_both["First systolic reading"] = df_both[["sb16s", "smb16s"]].apply(
        pd.to_numeric, errors="coerce").bfill(axis=1).iloc[:, 0]
    df_both["Second systolic reading"] = df_both[["sb23s", "smb23s"]].apply(
        pd.to_numeric, errors="coerce").bfill(axis=1).iloc[:, 0]
    df_both["Third systolic reading"] = df_both[["sb27s", "smb27s"]].apply(
        pd.to_numeric, errors="coerce").bfill(axis=1).iloc[:, 0]

    for c in DISEASE_COLS_N4:
        x = pd.to_numeric(df_both[c], errors="coerce")
        df_both[c] = np.where(x == 1, 1.0, np.where(x == 0, 0.0, np.nan))

    for bp_c in ["First systolic reading", "Second systolic reading", "Third systolic reading"]:
        v = pd.to_numeric(df_both[bp_c], errors="coerce")
        df_both.loc[v.isin(_BP_ERROR_CODES_INDIVIDUAL), bp_c] = np.nan
        df_both[bp_c] = pd.to_numeric(df_both[bp_c], errors="coerce")

    sbp_cols = ["First systolic reading", "Second systolic reading", "Third systolic reading"]
    df_both["Mean systolic reading"] = np.round(df_both[sbp_cols].mean(axis=1))

    if verbose:
        print(f"NFHS-4 individual recode: Men {len(df_men):,} + Women {len(df_women):,} "
              f"= {len(df_both):,}")
        print("Disease prevalence (weighted %):")
        for c in DISEASE_COLS_N4:
            w = df_both["weight"].to_numpy()
            v = df_both[c].to_numpy()
            m = np.isfinite(v) & np.isfinite(w) & (w > 0)
            pct = np.sum(w[m] * v[m]) / np.sum(w[m]) * 100 if m.any() else np.nan
            print(f"  {c}: {pct:.2f}%")
        print("Note: CKD not available in NFHS-4")

    return df_both
