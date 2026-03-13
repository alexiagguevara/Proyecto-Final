import numpy as np
from scipy.ndimage import convolve, distance_transform_edt
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize


def get_endpoints_and_junctions(skel):
    """
    Detecta endpoints y junctions en un skeleton binario.
    """
    skel = skel.astype(bool)
    kernel = np.ones((3, 3), dtype=np.uint8)
    neigh = convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)

    # neigh incluye el pixel central
    endpoints = skel & (neigh == 2)      # 1 vecino + sí mismo
    junctions = skel & (neigh >= 4)      # >=3 vecinos + sí mismo

    return endpoints, junctions


def extract_mask_features(mask):
    """
    Features globales de la máscara.
    """
    mask = mask.astype(bool)
    lbl = label(mask)
    props = regionprops(lbl)

    areas = np.array([p.area for p in props], dtype=float) if len(props) > 0 else np.array([])

    features = {}
    features["mask_area"] = int(mask.sum())
    features["fill_ratio"] = float(mask.mean())
    features["n_components"] = int(lbl.max())

    if len(areas) > 0:
        features["largest_component_area"] = float(np.max(areas))
        features["largest_component_ratio"] = float(np.max(areas) / mask.sum()) if mask.sum() > 0 else 0.0
        features["mean_component_area"] = float(np.mean(areas))
        features["std_component_area"] = float(np.std(areas))
        features["median_component_area"] = float(np.median(areas))
    else:
        features["largest_component_area"] = 0.0
        features["largest_component_ratio"] = 0.0
        features["mean_component_area"] = 0.0
        features["std_component_area"] = 0.0
        features["median_component_area"] = 0.0

    return features


def extract_skeleton_segment_features(skel):
    """
    Extrae segmentos del skeleton removiendo junctions.
    Cada componente restante se considera un segmento.
    """
    skel = skel.astype(bool)
    endpoints, junctions = get_endpoints_and_junctions(skel)

    # quitar junctions para separar segmentos
    seg_mask = skel & ~junctions
    seg_lbl = label(seg_mask)
    seg_props = regionprops(seg_lbl)

    seg_lengths = []
    tortuosities = []
    angles_deg = []

    for p in seg_props:
        coords = p.coords
        length = len(coords)

        if length < 2:
            continue

        seg_lengths.append(length)

        # endpoints dentro del segmento
        seg_endpoints = []
        for r, c in coords:
            if endpoints[r, c]:
                seg_endpoints.append((r, c))

        # tortuosidad y orientación si el segmento tiene dos endpoints
        if len(seg_endpoints) == 2:
            (r1, c1), (r2, c2) = seg_endpoints
            euclid = np.sqrt((r2 - r1)**2 + (c2 - c1)**2)

            if euclid > 0:
                tortuosities.append(length / euclid)

            angle = np.degrees(np.arctan2((r2 - r1), (c2 - c1)))
            angles_deg.append(angle)

    seg_lengths = np.array(seg_lengths, dtype=float)
    tortuosities = np.array(tortuosities, dtype=float)
    angles_deg = np.array(angles_deg, dtype=float)

    features = {}
    features["n_segments"] = int(len(seg_lengths))

    if len(seg_lengths) > 0:
        features["mean_segment_length"] = float(np.mean(seg_lengths))
        features["median_segment_length"] = float(np.median(seg_lengths))
        features["max_segment_length"] = float(np.max(seg_lengths))
        features["std_segment_length"] = float(np.std(seg_lengths))
    else:
        features["mean_segment_length"] = 0.0
        features["median_segment_length"] = 0.0
        features["max_segment_length"] = 0.0
        features["std_segment_length"] = 0.0

    if len(tortuosities) > 0:
        features["mean_tortuosity"] = float(np.mean(tortuosities))
        features["median_tortuosity"] = float(np.median(tortuosities))
        features["max_tortuosity"] = float(np.max(tortuosities))
    else:
        features["mean_tortuosity"] = 0.0
        features["median_tortuosity"] = 0.0
        features["max_tortuosity"] = 0.0

    if len(angles_deg) > 0:
        features["orientation_mean_deg"] = float(np.mean(angles_deg))
        features["orientation_std_deg"] = float(np.std(angles_deg))
    else:
        features["orientation_mean_deg"] = 0.0
        features["orientation_std_deg"] = 0.0

    return features


def extract_thickness_features(mask, skel):
    """
    Estima grosor a partir de distance transform sobre la máscara,
    muestreado en el skeleton.
    """
    mask = mask.astype(bool)
    skel = skel.astype(bool)

    dist = distance_transform_edt(mask)
    thickness = 2.0 * dist[skel]   # diámetro aproximado local

    features = {}

    if len(thickness) > 0:
        features["mean_thickness"] = float(np.mean(thickness))
        features["median_thickness"] = float(np.median(thickness))
        features["max_thickness"] = float(np.max(thickness))
        features["std_thickness"] = float(np.std(thickness))
    else:
        features["mean_thickness"] = 0.0
        features["median_thickness"] = 0.0
        features["max_thickness"] = 0.0
        features["std_thickness"] = 0.0

    return features


def extract_skeleton_global_features(skel):
    """
    Features globales del skeleton.
    """
    skel = skel.astype(bool)
    endpoints, junctions = get_endpoints_and_junctions(skel)

    total_pixels = skel.size

    features = {}
    features["skeleton_length"] = int(skel.sum())
    features["n_endpoints"] = int(endpoints.sum())
    features["n_junctions"] = int(junctions.sum())
    features["endpoint_density"] = float(endpoints.sum() / total_pixels)
    features["junction_density"] = float(junctions.sum() / total_pixels)
    features["branch_density"] = float(skel.sum() / total_pixels)

    return features


def extract_all_features(mask, skel=None):
    """
    Extrae todas las features relevantes de una máscara y su skeleton.
    """
    mask = mask.astype(bool)

    if skel is None:
        skel = skeletonize(mask)
    else:
        skel = skel.astype(bool)

    features = {}
    features.update(extract_mask_features(mask))
    features.update(extract_skeleton_global_features(skel))
    features.update(extract_skeleton_segment_features(skel))
    features.update(extract_thickness_features(mask, skel))

    return features

# Rank features
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu, shapiro
from statsmodels.stats.multitest import multipletests


# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────

def cohens_d(x, y):
    """Pooled Cohen's d effect size between two groups."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)

    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)

    pooled_std = np.sqrt(
        ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    )

    return 0.0 if pooled_std == 0 else (np.mean(y) - np.mean(x)) / pooled_std


def interpret_cohens_d(d):
    """Human-readable magnitude label for Cohen's d."""
    d = abs(d)
    if d < 0.20:
        return "negligible"
    elif d < 0.50:
        return "small"
    elif d < 0.80:
        return "medium"
    else:
        return "large"


def check_normality(x, y, alpha=0.05):
    """
    Shapiro-Wilk on both groups.
    Returns True only if BOTH pass normality.
    Skipped (returns None) if n < 3 or n > 5000.
    """
    if len(x) < 3 or len(y) < 3 or len(x) > 5000 or len(y) > 5000:
        return None
    return (shapiro(x).pvalue > alpha) and (shapiro(y).pvalue > alpha)


def minmax(series):
    """Min-max normalisation to [0,1]."""
    s = pd.Series(series, dtype=float)
    s_min, s_max = s.min(), s.max()

    if pd.isna(s_min) or pd.isna(s_max):
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)

    if s_max > s_min:
        return (s - s_min) / (s_max - s_min)

    # if all values equal, keep neutral score
    return pd.Series(np.full(len(s), 0.5), index=s.index, dtype=float)


# ─────────────────────────────────────────────
#  CORE PIPELINE
# ─────────────────────────────────────────────

def rank_features(
    df,
    condition_col="condition",
    ctrl_label="CTRL",
    hpmc_label="HPMC",
    alpha=0.05,
    fdr_method="fdr_bh",   # use "bonferroni" if stricter
    w_d=0.50,
    w_ttest=0.25,
    w_mwu=0.25,
):
    """
    Full feature-ranking pipeline for binary classification.

    Metrics
    -------
    1. p-value         : Welch t-test + Mann-Whitney U (+ FDR correction)
    2. Effect size     : Cohen's d  (with magnitude label)
    3. Separation score: Relative mean difference  (+ absolute difference)

    Returns
    -------
    pd.DataFrame ranked by composite_score (descending)
    """

    if abs((w_d + w_ttest + w_mwu) - 1.0) >= 1e-6:
        raise ValueError(
            f"Weights must sum to 1.0, got {w_d + w_ttest + w_mwu:.6f}"
        )

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    ctrl_df = df[df[condition_col] == ctrl_label]
    hpmc_df = df[df[condition_col] == hpmc_label]

    rows = []

    for feat in numeric_cols:
        x = ctrl_df[feat].dropna().values
        y = hpmc_df[feat].dropna().values

        if len(x) < 2 or len(y) < 2:
            continue

        # ── 1. Means & separation ─────────────────────────────────────────
        mean_ctrl = np.mean(x)
        mean_hpmc = np.mean(y)
        abs_diff = mean_hpmc - mean_ctrl
        rel_diff = abs_diff / (abs(mean_ctrl) + 1e-12)

        # ── 2. Statistical tests ──────────────────────────────────────────
        _, p_ttest = ttest_ind(x, y, equal_var=False)   # Welch

        try:
            _, p_mwu = mannwhitneyu(x, y, alternative="two-sided")
        except ValueError:
            p_mwu = np.nan

        # ── 3. Effect size ────────────────────────────────────────────────
        d = cohens_d(x, y)

        # ── 4. Normality flag ─────────────────────────────────────────────
        is_normal = check_normality(x, y, alpha=alpha)
        preferred_test = (
            "t-test" if is_normal is True else
            "mann-whitney" if is_normal is False else
            "unknown"
        )

        rows.append({
            "feature": feat,
            "mean_CTRL": mean_ctrl,
            "mean_HPMC": mean_hpmc,
            "abs_diff": abs_diff,
            "relative_diff": rel_diff,
            "cohens_d": d,
            "abs_d": abs(d),
            "effect_size": interpret_cohens_d(d),
            "p_ttest": p_ttest,
            "p_mannwhitney": p_mwu,
            "both_normal": is_normal,
            "preferred_test": preferred_test,
        })

    results_df = pd.DataFrame(rows)

    if results_df.empty:
        print("⚠️ No features could be evaluated.")
        return results_df

    # ── 5. FDR correction (multiple comparisons) ──────────────────────────
    valid_ttest = results_df["p_ttest"].fillna(1.0)
    valid_mwu = results_df["p_mannwhitney"].fillna(1.0)

    _, p_ttest_corr, _, _ = multipletests(valid_ttest, method=fdr_method)
    _, p_mwu_corr, _, _ = multipletests(valid_mwu, method=fdr_method)

    results_df["p_ttest_corrected"] = p_ttest_corr
    results_df["p_mannwhitney_corrected"] = p_mwu_corr

    results_df["sig_ttest"] = results_df["p_ttest_corrected"] < alpha
    results_df["sig_mwu"] = results_df["p_mannwhitney_corrected"] < alpha

    # ── 6. Composite score (using corrected p-values + proper normalisation) ──
    eps = 1e-12

    results_df["neglog_p_ttest_corr"] = -np.log10(
        np.clip(results_df["p_ttest_corrected"].fillna(1.0), eps, 1.0)
    )
    results_df["neglog_p_mwu_corr"] = -np.log10(
        np.clip(results_df["p_mannwhitney_corrected"].fillna(1.0), eps, 1.0)
    )

    results_df["abs_d_norm"] = minmax(results_df["abs_d"])
    results_df["p_ttest_normscore"] = minmax(results_df["neglog_p_ttest_corr"])
    results_df["p_mwu_normscore"] = minmax(results_df["neglog_p_mwu_corr"])

    results_df["composite_score"] = (
        w_d * results_df["abs_d_norm"]
        + w_ttest * results_df["p_ttest_normscore"]
        + w_mwu * results_df["p_mwu_normscore"]
    )

    results_df["composite_score_norm"] = results_df["composite_score"] * 100.0

    # ── 7. Final sort & importance tier ──────────────────────────────────
    results_df = results_df.sort_values(
        "composite_score_norm", ascending=False
    ).reset_index(drop=True)

    results_df["rank"] = results_df.index + 1
    results_df["importance_tier"] = results_df.apply(
        lambda row: "🔴 HIGH"   if (row["sig_ttest"] and row["sig_mwu"] and row["abs_d"] >= 0.8) else
                    "🟡 MEDIUM" if ((row["sig_ttest"] or row["sig_mwu"]) and row["abs_d"] >= 0.5) else
                    "🟢 LOW",
        axis=1
    )

    # ── 8. Column order ───────────────────────────────────────────────────
    col_order = [
        "rank", "feature", "importance_tier", "composite_score_norm",
        "effect_size", "cohens_d", "abs_d",
        "relative_diff", "abs_diff",
        "mean_CTRL", "mean_HPMC",
        "p_ttest", "p_ttest_corrected", "sig_ttest",
        "p_mannwhitney", "p_mannwhitney_corrected", "sig_mwu",
        "both_normal", "preferred_test",
        "neglog_p_ttest_corr", "neglog_p_mwu_corr",
        "abs_d_norm", "p_ttest_normscore", "p_mwu_normscore",
    ]
    results_df = results_df[[c for c in col_order if c in results_df.columns]]

    return results_df


# ─────────────────────────────────────────────
#  REPORT PRINTER
# ─────────────────────────────────────────────

def print_report(results_df, top_n=10, alpha=0.05):
    """Pretty-prints a summary of the top-N features."""
    print("=" * 70)
    print(f"  FEATURE IMPORTANCE REPORT  (top {top_n})")
    print("=" * 70)

    display_cols = [
        "rank", "feature", "importance_tier",
        "composite_score_norm", "effect_size",
        "sig_ttest", "sig_mwu",
    ]
    available = [c for c in display_cols if c in results_df.columns]
    print(results_df[available].head(top_n).to_string(index=False))

    print("\n── Significance summary ──────────────────────────────────────")
    sig_both = results_df["sig_ttest"] & results_df["sig_mwu"]
    sig_one = results_df["sig_ttest"] | results_df["sig_mwu"]
    high_eff = results_df["effect_size"].isin(["large", "medium"])

    print(f"  Significant in BOTH tests (FDR < {alpha})  : {sig_both.sum()}")
    print(f"  Significant in AT LEAST ONE test           : {sig_one.sum()}")
    print(f"  Medium or large effect size (|d| ≥ 0.50)  : {high_eff.sum()}")
    print(f"  Strong candidates (sig + medium/large d)   : {(sig_both & high_eff).sum()}")
    print("=" * 70)