"""
Init_analysis.py: calibrate worker characteristics from IPUMS ACS 2024 microdata.

Run once from anywhere:  python3 Model/Init_analysis.py
Reads   Data/IPUMS.csv and the SECTORS list in Model/ABM2.jl
Writes  Data/Calibration/
    wage_regression.csv     all coefficients of the wage regression
    qual_productivity.csv       mean productivity by qualification: wage (USD) at the sample-average
                                industry and experience mix; and relative productivity (high school = 1)
    exp_wage_premium.csv        relative wage by experience band (0-3 years = 1)
    wage_residuals.csv          spread of wages around the regression prediction, by qualification
    wage_by_cell.csv            predicted wage (USD) for each (model sector, qualification, experience band):
                                const + qualification effect + experience effect + sector industry effect
    qual_exp_mix_by_sector.csv  share of workers in each (qualification, experience band) cell,
                                by model sector; long format, shares sum to 1 within a sector
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parent.parent
IPUMS_FILE = ROOT / "Data" / "IPUMS.csv"
QCEW_FILE  = ROOT / "Data" / "QCEW annual wage and employment.csv"   # sanity check only
MODEL_FILE = ROOT / "Model" / "ABM2.jl"
OUT_DIR    = ROOT / "Data" / "Calibration"

USE_LOG_WAGE = True     # True: regress log wage (effects are multiplicative); False: wage in USD
MIN_SHARE    = 0.005    # within a sector, cell shares below this are set to 0 and the rest renormalised

QUALIFICATIONS = {
    1: "High school",
    2: "Associate degree",
    3: "Bachelor's, STEM",
    4: "Bachelor's, other",
    5: "Master's / professional",
    6: "Doctorate",
}
START_AGE = {1: 18, 2: 20, 3: 22, 4: 22, 5: 24, 6: 28}   # age at which work starts: experience = age - start age
STEM_FIELDS = {13, 20, 21, 24, 25, 36, 37, 50, 51}   # DEGFIELD codes: env. sci, comm. tech, computing,
                                                     # engineering (+tech), biology, maths, physical sci, nuclear/bio tech
EXP_BANDS = {1: "0-3", 2: "3-6", 3: "6-10", 4: "10-15", 5: "15-20", 6: "20-25", 7: "25-30", 8: "30+"}
EXP_EDGES = [0, 3, 6, 10, 15, 20, 25, 30, np.inf]

# ------------------------------------------------------------------
# 1. Load and classify
# ------------------------------------------------------------------
COLUMNS = ["PERWT", "AGE", "EDUCD", "DEGFIELD", "EMPSTAT", "CLASSWKRD",
           "INDNAICS", "WKSWORK1", "UHRSWORK", "INCWAGE"]

def load_workers():
    """Employed private-sector wage earners aged 18-64 (for-profit and non-profit)."""
    d = pd.read_csv(IPUMS_FILE, usecols=COLUMNS, dtype={"INDNAICS": str})
    d["industry"] = d["INDNAICS"].str.strip()
    keep = (d.EMPSTAT == 1) & d.CLASSWKRD.isin([22, 23]) & d.AGE.between(18, 64)
    return d[keep].copy()

def classify_qualification(educd, degfield):
    """Map IPUMS EDUCD (+ DEGFIELD for bachelor's) to the model's qualifications; 0 = dropped."""
    stem = degfield.isin(STEM_FIELDS)
    conditions = [
        (educd >= 62)  & (educd <= 64),      # 1 high school diploma / GED
        (educd >= 81)  & (educd <= 83),      # 2 associate's degree
        (educd == 101) & stem,               # 3 bachelor's, STEM field
        (educd == 101),                      # 4 bachelor's, any other field
        (educd >= 114) & (educd <= 115),     # 5 master's, professional degree
        (educd == 116),                      # 6 doctorate
    ]
    choices = [1, 2, 3, 4, 5, 6]             # np.select uses the first condition that is true
    return pd.Series(np.select(conditions, choices, default=0), index=educd.index)

def add_classifications(d):
    d["qual"] = classify_qualification(d.EDUCD, d.DEGFIELD)
    d = d[d.qual > 0].copy()                                                  # drop all other EDUCD values
    experience = (d.AGE - d.qual.map(START_AGE)).clip(lower=0)               # potential experience
    d["exp_band"] = pd.cut(experience, bins=EXP_EDGES, labels=list(EXP_BANDS), right=False).astype(int)
    d["industry3"] = d.industry.str[:3]                                       # regression industry dummy
    return d

# ------------------------------------------------------------------
# 2. Wage regression: wage ~ qualification + experience band + industry (dummies)
# ------------------------------------------------------------------
def full_time_full_year(d):
    """Restrict to full-time (35+ hours), full-year (50+ weeks) workers with valid wages."""
    return d[(d.UHRSWORK >= 35) & (d.WKSWORK1 >= 50) & (d.INCWAGE > 0) & (d.INCWAGE < 999998)]

def weighted_ols(X, y, w, chunk=200_000):
    """Weighted least squares via normal equations, accumulated in chunks to save memory."""
    k = X.shape[1]
    XtWX, XtWy = np.zeros((k, k)), np.zeros(k)
    for i in range(0, len(y), chunk):
        Xc, wc, yc = X[i:i + chunk].astype(float), w[i:i + chunk], y[i:i + chunk]
        XtWX += Xc.T @ (Xc * wc[:, None])
        XtWy += Xc.T @ (wc * yc)
    return np.linalg.solve(XtWX, XtWy)

def wage_regression(d):
    """Returns coefficients (pd.Series) and the regression sample. Reference groups:
    qualification 1, experience band 1 (0-3 years), and the first industry code."""
    s = full_time_full_year(d)
    dummies = [
        pd.get_dummies(s.qual,      prefix="qual", drop_first=True, dtype=np.uint8),
        pd.get_dummies(s.exp_band,  prefix="exp",  drop_first=True, dtype=np.uint8),
        pd.get_dummies(s.industry3, prefix="ind",  drop_first=True, dtype=np.uint8),
    ]
    X = pd.concat(dummies, axis=1)
    X.insert(0, "const", np.uint8(1))
    y = np.log(s.INCWAGE.to_numpy(float)) if USE_LOG_WAGE else s.INCWAGE.to_numpy(float)
    beta = weighted_ols(X.to_numpy(), y, s.PERWT.to_numpy(float))
    return pd.Series(beta, index=X.columns), s, X

def smearing_factor(coef, s, X):
    """Converts exp(predicted log wage) into a mean wage in USD: E[exp(residual)] (Duan 1983)."""
    resid = np.log(s.INCWAGE.to_numpy(float)) - X.to_numpy(float) @ coef.to_numpy()
    return np.average(np.exp(resid), weights=s.PERWT)

def relative_effects(coef, s, X, prefix, groups):
    """Wage of each group with the other regressors held at the sample mix.
    base  = average predicted wage if every worker were in the reference group
            (keeping their own values of the other regressors)
    level = base + coef            (log wage: exp(base + coef) * smearing factor, in USD)
    ratio = level / reference level (log wage: exp(coef))
    Returns (ratio, level, coef)."""
    b = np.array([0.0] + [coef.get(f"{prefix}_{g}", 0.0) for g in list(groups)[1:]])
    own_cols = [c for c in X.columns if c.startswith(prefix + "_")]
    w = s.PERWT.to_numpy(float)
    fitted = X.to_numpy(float) @ coef.to_numpy()
    own    = X[own_cols].to_numpy(float) @ coef[own_cols].to_numpy()
    base   = np.average(fitted - own, weights=w)          # reference group, other mix held fixed
    if USE_LOG_WAGE:
        return np.exp(b), np.exp(base + b) * smearing_factor(coef, s, X), b
    return (base + b) / base, base + b, b

def residual_spread(coef, s, X, groups):
    """Spread of wages around the regression prediction, by qualification.
    Log wage:  log_sd = weighted SD of log-wage residuals.
    USD wage:  cv = SD of residuals / mean wage; log_sd = sqrt(ln(1 + cv^2)) (same CV under a log-normal)."""
    w      = s.PERWT.to_numpy(float)
    wage   = s.INCWAGE.to_numpy(float)
    y      = np.log(wage) if USE_LOG_WAGE else wage
    resid  = y - X.to_numpy(float) @ coef.to_numpy()
    rows = []
    for g in [None] + list(groups):                                   # None = all workers
        m  = np.ones(len(s), bool) if g is None else (s.qual == g).to_numpy()
        sd = np.sqrt(np.average(resid[m] ** 2, weights=w[m]))
        log_sd = sd if USE_LOG_WAGE else np.sqrt(np.log1p((sd / np.average(wage[m], weights=w[m])) ** 2))
        rows.append({"qualification": "all" if g is None else g, "log_sd": log_sd})
    return pd.DataFrame(rows)

# ------------------------------------------------------------------
# 3. Qualification and experience mix by model sector
# ------------------------------------------------------------------
def model_sector_list():
    """Read SECTORS from ABM2.jl so the sector order matches the model exactly."""
    text  = MODEL_FILE.read_text()
    block = re.search(r"const SECTORS = \[(.*?)\]", text, re.S).group(1)
    return re.findall(r'"([^"]+)"', block)

# IPUMS industry codes that need a manual mapping (everything else uses its first 3 digits)
SPECIAL = {
    "23":      ["236", "237", "238"],   # construction is not split in the ACS
    "336M":    ["3361MV"],              # motor vehicles and parts
    "52M2":    ["523"],                 # securities, commodity contracts, investments
    "52M3":    ["522"], "522M": ["522"], "5221M": ["522"],   # banking and credit
    "53M":     ["532", "533"],          # commercial rental and leasing, intangible assets
}

def to_model_sectors(code, sectors):
    if code in SPECIAL:
        return SPECIAL[code]
    if code.startswith("336"):
        return ["3364OT"]                                    # 3364-3369 (33641M1, 3365, ...)
    if code.startswith("541"):
        if code.startswith("5411"):
            return []                                        # legal services: not a model sector
        return ["5415"] if code.startswith("5415") else ["5412OP"]
    return [code[:3]] if code[:3] in sectors else []         # unmatched codes are dropped

def qual_exp_mix(d, sectors):
    """Weighted share of workers in each (qualification, experience band) cell within each
    model sector. Returns long format (sector, qualification, exp_band, share) with every cell
    listed, plus the share of workers whose industry has no model sector."""
    mapping = {c: to_model_sectors(c, sectors) for c in d.industry.unique()}
    rows = d[["industry", "qual", "exp_band", "PERWT"]].copy()
    rows["sector"] = rows.industry.map(mapping)
    rows = rows.explode("sector").dropna(subset=["sector"])

    cells = pd.MultiIndex.from_product([sectors, list(QUALIFICATIONS), list(EXP_BANDS)],
                                       names=["sector", "qualification", "exp_band"])
    mix = rows.groupby(["sector", "qual", "exp_band"]).PERWT.sum()
    mix.index.names = cells.names
    mix = mix.reindex(cells, fill_value=0.0)
    mix = mix / mix.groupby(level="sector").transform("sum")
    mix = mix.where(mix >= MIN_SHARE, 0.0)                   # drop insignificant cells
    mix = mix / mix.groupby(level="sector").transform("sum")

    dropped = d.PERWT[d.industry.map(mapping).str.len() == 0].sum() / d.PERWT.sum()
    return mix.rename("share").reset_index(), dropped

# ------------------------------------------------------------------
# 4. Predicted wage by (sector, qualification, experience band)
# ------------------------------------------------------------------
def wage_by_cell(coef, s, X, sectors):
    """Predicted wage for every cell. The industry effect of a model sector is the weighted
    average of the regression industry effects of the IPUMS codes mapped to it."""
    rows = s[["industry", "industry3", "PERWT"]].copy()
    rows["effect"] = rows.industry3.map(lambda c: coef.get(f"ind_{c}", 0.0))   # reference industry = 0
    rows["sector"] = rows.industry.map(lambda c: to_model_sectors(c, sectors))
    rows = rows.explode("sector").dropna(subset=["sector"])
    weighted      = rows.effect * rows.PERWT
    sector_effect = weighted.groupby(rows.sector).sum() / rows.PERWT.groupby(rows.sector).sum()

    cells = pd.MultiIndex.from_product([sectors, list(QUALIFICATIONS), list(EXP_BANDS)],
                                       names=["sector", "qualification", "exp_band"]).to_frame(index=False)
    cells["wage"] = (coef["const"]
                     + cells.qualification.map(lambda q: coef.get(f"qual_{q}", 0.0))
                     + cells.exp_band.map(lambda e: coef.get(f"exp_{e}", 0.0))
                     + cells.sector.map(sector_effect))
    if USE_LOG_WAGE:                                         # back to USD levels
        cells["wage"] = np.exp(cells.wage) * smearing_factor(coef, s, X)
    return cells

def qcew_check(cells, mix):
    """Compare each sector's mix-weighted predicted wage with QCEW average annual pay."""
    q = pd.read_csv(QCEW_FILE, dtype={"industry_code": str})
    q = q[(q.own_code == 5) & (q.area_fips == "US000")].set_index("industry_code").avg_annual_pay
    m = cells.merge(mix, on=["sector", "qualification", "exp_band"])
    predicted = (m.wage * m.share).groupby(m.sector).sum()
    ratio = (predicted / q.reindex(predicted.index)).dropna()     # sectors with a single QCEW code
    return predicted, ratio

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d = add_classifications(load_workers())
    sectors = model_sector_list()

    coef, s, X = wage_regression(d)
    coef.rename("coefficient").to_csv(OUT_DIR / "wage_regression.csv", index_label="term")

    qual_rel, qual_level, _ = relative_effects(coef, s, X, "qual", QUALIFICATIONS)
    pd.DataFrame({"qualification": list(QUALIFICATIONS), "label": list(QUALIFICATIONS.values()),
                  "mean_productivity": qual_level,        # USD: wage at the sample-average industry and experience
                  "relative_productivity": qual_rel}
                 ).to_csv(OUT_DIR / "qual_productivity.csv", index=False)

    spread = residual_spread(coef, s, X, QUALIFICATIONS)
    spread.to_csv(OUT_DIR / "wage_residuals.csv", index=False, float_format="%.4f")

    exp_rel, _, exp_b = relative_effects(coef, s, X, "exp", EXP_BANDS)
    pd.DataFrame({"exp_band": list(EXP_BANDS), "years": list(EXP_BANDS.values()),
                  "coefficient": exp_b, "relative_wage": exp_rel}
                 ).to_csv(OUT_DIR / "exp_wage_premium.csv", index=False)

    mix, dropped = qual_exp_mix(d, sectors)
    mix.to_csv(OUT_DIR / "qual_exp_mix_by_sector.csv", index=False, float_format="%.5f")

    cells = wage_by_cell(coef, s, X, sectors)
    cells.to_csv(OUT_DIR / "wage_by_cell.csv", index=False, float_format="%.2f")
    predicted, ratio = qcew_check(cells, mix)

    # --- summary ---
    print(f"Workers: {len(d):,}   regression sample (full-time, full-year): {len(s):,}")
    print(f"Share of workers in IPUMS industries not mapped to a model sector: {dropped:.1%}")
    print("\nRelative productivity by qualification:\n", pd.Series(qual_rel, index=QUALIFICATIONS.values()).round(2))
    print("\nWage residual spread by qualification:\n", spread.round(3).to_string(index=False))
    print("\nRelative wage by experience band:\n", pd.Series(exp_rel, index=EXP_BANDS.values()).round(2))
    used = cells.merge(mix[mix.share > 0], on=["sector", "qualification", "exp_band"])
    print(f"\nPredicted cell wages (cells with workers): min {used.wage.min():,.0f}, "
          f"median {used.wage.median():,.0f}, max {used.wage.max():,.0f}; negative: {(used.wage < 0).sum()}")
    print(f"Sector average predicted wage / QCEW average pay ({len(ratio)} sectors): "
          f"median {ratio.median():.2f}, range {ratio.min():.2f}-{ratio.max():.2f}")
    n_cells = (mix.share > 0).groupby(mix.sector).sum()
    print(f"\nNon-zero cells per sector (of {len(QUALIFICATIONS) * len(EXP_BANDS)}, share >= {MIN_SHARE:.1%}): "
          f"min {n_cells.min()}, median {n_cells.median():.0f}, max {n_cells.max()}")

if __name__ == "__main__":
    main()
