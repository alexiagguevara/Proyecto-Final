import numpy as np
import pandas as pd

from scipy.stats import spearmanr, kruskal
from statsmodels.stats.multitest import multipletests


def minmax(series):
    s = pd.Series(series, dtype=float)
    s_min, s_max = s.min(), s.max()

    if pd.isna(s_min) or pd.isna(s_max):
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)

    if s_max > s_min:
        return (s - s_min) / (s_max - s_min)

    return pd.Series(np.full(len(s), 0.5), index=s.index, dtype=float)


def rank_temporal_features(df,
                           time_col="time_h",
                           time_label_col="time_label",
                           alpha=0.05,
                           fdr_method="fdr_bh",
                           w_rho=0.5,
                           w_spearman=0.25,
                           w_kruskal=0.25):
    """
    Ranking temporal de features:
    - Spearman con time_h
    - Kruskal-Wallis entre tiempos
    - score combinado
    """

    if abs((w_rho + w_spearman + w_kruskal) - 1.0) >= 1e-6:
        raise ValueError("Los pesos deben sumar 1.0")

    exclude_cols = {"time_h", "image_name"}
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]

    time_order = sorted(df[time_label_col].unique(), key=lambda x: int(x.replace("HS", "")))

    rows = []

    for feat in numeric_cols:
        x = df[feat].dropna().values
        t = df.loc[df[feat].notna(), time_col].values

        if len(x) < 3:
            continue

        # Spearman
        rho, p_spearman = spearmanr(t, x)

        # Kruskal
        groups = [df[df[time_label_col] == tl][feat].dropna().values for tl in time_order]
        if all(len(g) > 0 for g in groups):
            _, p_kruskal = kruskal(*groups)
        else:
            p_kruskal = np.nan

        # Dirección
        if rho > 0:
            trend = "increasing"
        elif rho < 0:
            trend = "decreasing"
        else:
            trend = "flat"

        # Medias por tiempo
        means_by_time = {tl: df[df[time_label_col] == tl][feat].mean() for tl in time_order}

        rows.append({
            "feature": feat,
            "spearman_rho": rho,
            "abs_rho": abs(rho),
            "p_spearman": p_spearman,
            "p_kruskal": p_kruskal,
            "trend": trend,
            **{f"mean_{tl}": means_by_time[tl] for tl in time_order}
        })

    results_df = pd.DataFrame(rows)

    if results_df.empty:
        print("⚠️ No se pudieron evaluar features.")
        return results_df

    # Corrección por múltiples comparaciones
    valid_spearman = results_df["p_spearman"].fillna(1.0)
    valid_kruskal = results_df["p_kruskal"].fillna(1.0)

    _, p_spearman_corr, _, _ = multipletests(valid_spearman, method=fdr_method)
    _, p_kruskal_corr, _, _ = multipletests(valid_kruskal, method=fdr_method)

    results_df["p_spearman_corrected"] = p_spearman_corr
    results_df["p_kruskal_corrected"] = p_kruskal_corr

    results_df["sig_spearman"] = results_df["p_spearman_corrected"] < alpha
    results_df["sig_kruskal"] = results_df["p_kruskal_corrected"] < alpha

    # Score compuesto
    eps = 1e-12
    results_df["neglog_p_spearman_corr"] = -np.log10(np.clip(results_df["p_spearman_corrected"], eps, 1.0))
    results_df["neglog_p_kruskal_corr"] = -np.log10(np.clip(results_df["p_kruskal_corrected"], eps, 1.0))

    results_df["abs_rho_norm"] = minmax(results_df["abs_rho"])
    results_df["p_spearman_normscore"] = minmax(results_df["neglog_p_spearman_corr"])
    results_df["p_kruskal_normscore"] = minmax(results_df["neglog_p_kruskal_corr"])

    results_df["composite_score"] = (
        w_rho * results_df["abs_rho_norm"]
        + w_spearman * results_df["p_spearman_normscore"]
        + w_kruskal * results_df["p_kruskal_normscore"]
    )

    results_df["composite_score_norm"] = results_df["composite_score"] * 100.0

    results_df = results_df.sort_values("composite_score_norm", ascending=False).reset_index(drop=True)
    results_df["rank"] = results_df.index + 1

    results_df["importance_tier"] = results_df.apply(
        lambda row: "🔴 HIGH" if (row["sig_spearman"] and row["sig_kruskal"] and row["abs_rho"] >= 0.7) else
                    "🟡 MEDIUM" if ((row["sig_spearman"] or row["sig_kruskal"]) and row["abs_rho"] >= 0.4) else
                    "🟢 LOW",
        axis=1
    )

    return results_df


def print_temporal_report(results_df, top_n=10, alpha=0.05):
    print("=" * 70)
    print(f"  TEMPORAL FEATURE RANKING REPORT (top {top_n})")
    print("=" * 70)

    cols = [
        "rank", "feature", "importance_tier",
        "composite_score_norm", "spearman_rho",
        "sig_spearman", "sig_kruskal", "trend"
    ]
    print(results_df[cols].head(top_n).to_string(index=False))

    print("\n── Summary ─────────────────────────────────────────────")
    sig_both = results_df["sig_spearman"] & results_df["sig_kruskal"]
    print(f"Significant in BOTH tests (FDR < {alpha}): {sig_both.sum()}")
    print(f"HIGH tier features: {(results_df['importance_tier'] == '🔴 HIGH').sum()}")
    print("=" * 70)


import matplotlib.pyplot as plt
import seaborn as sns

def plot_temporal_feature_space(df, save_path="temporal_feature_space.png"):
    time_order = ["0HS", "24HS", "48HS", "72HS"]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="median_segment_length",
        y="median_thickness",
        hue="time_label",
        hue_order=time_order,
        style="group",
        s=80
    )

    plt.title("Temporal feature space", fontsize=14, fontweight="bold")
    plt.xlabel("Median segment length")
    plt.ylabel("Median thickness")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Saved → {save_path}")