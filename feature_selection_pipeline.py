import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.inspection import permutation_importance

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[INFO] SHAP not installed. Run: pip install shap")


# =============================================================================
# CONFIG
# =============================================================================

CONDITION_COL = "condition"
GROUP_COL = "group"          # N1, N2, N3, N4
CTRL_LABEL = "CTRL"
HPMC_LABEL = "HPMC"
CORR_THRESHOLD = 0.85       # drop one feature from pairs above this |r|
DEFAULT_N_SPLITS = 4                 # 4 cultivos → leave-one-culture-out style
RANDOM_STATE = 42


# =============================================================================
# STEP 1 — KEEP ONLY HIGH-IMPORTANCE FEATURES FROM STAGE 1
# =============================================================================

def get_strong_candidates(results_df, df=None, tier_col="importance_tier", high_label="🔴 HIGH"):
    """
    Toma las features HIGH del ranking Stage 1.
    Si se pasa df, filtra además las que realmente existan en el dataframe.
    """
    strong = results_df[results_df[tier_col] == high_label]["feature"].tolist()

    if df is not None:
        strong = [f for f in strong if f in df.columns]

    print(f"[Step 1] Strong candidates from Stage 1: {len(strong)} features")
    print(f"         {strong}\n")
    return strong


# =============================================================================
# STEP 2 — CORRELATION PRUNING (Spearman)
# =============================================================================

def plot_correlation_matrix(corr_matrix, title="Spearman Correlation Matrix"):
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.4,
        ax=ax
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig("correlation_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[Step 2] Correlation matrix saved → correlation_matrix.png\n")


def prune_correlated_features(df, features, results_df, threshold=CORR_THRESHOLD):
    """
    Elimina una feature de cada par altamente correlacionado.
    Se queda con la que tenga mayor composite_score_norm del Stage 1.
    """
    corr_matrix = df[features].corr(method="spearman")
    plot_correlation_matrix(corr_matrix)

    score_map = dict(zip(results_df["feature"], results_df["composite_score_norm"]))

    flagged_pairs = []
    to_drop = set()

    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > threshold:
                fa, fb = cols[i], cols[j]

                keep = fa if score_map.get(fa, 0) >= score_map.get(fb, 0) else fb
                drop = fb if keep == fa else fa
                to_drop.add(drop)

                flagged_pairs.append({
                    "feature_a": fa,
                    "feature_b": fb,
                    "spearman_r": round(r, 4),
                    "score_a": round(score_map.get(fa, 0), 2),
                    "score_b": round(score_map.get(fb, 0), 2),
                    "kept": keep,
                    "dropped": drop
                })

    corr_pairs_df = pd.DataFrame(flagged_pairs)
    if not corr_pairs_df.empty:
        corr_pairs_df = corr_pairs_df.sort_values(
            "spearman_r", key=lambda x: np.abs(x), ascending=False
        )

    selected = [f for f in features if f not in to_drop]

    print(f"[Step 2] Pairs with |r| > {threshold}:")
    if corr_pairs_df.empty:
        print("         None found — no features dropped.")
    else:
        print(corr_pairs_df.to_string(index=False))

    print(f"\n         Dropped : {sorted(to_drop)}")
    print(f"         Kept    : {len(selected)} features → {selected}\n")

    return selected, corr_pairs_df


# =============================================================================
# STEP 3 — RFECV: RECURSIVE FEATURE ELLIMINATION WITH GROUPKFold
# =============================================================================

def run_rfecv(df,
              features,
              condition_col=CONDITION_COL,
              group_col=GROUP_COL,
              ctrl_label=CTRL_LABEL,
              hpmc_label=HPMC_LABEL,
              n_splits=DEFAULT_N_SPLITS):
    """
    Selección recursiva de features usando:
    - GroupKFold por cultivo
    - Logistic Regression con L1
    - escalado dentro del pipeline (sin leakage)
    """
    mask = df[condition_col].isin([ctrl_label, hpmc_label])

    X = df.loc[mask, features].values
    y_raw = df.loc[mask, condition_col].values
    y = (y_raw == hpmc_label).astype(int) #HPMC=1, CTRL=0
    groups = df.loc[mask, group_col].values

    cv = GroupKFold(n_splits=n_splits)

    estimator = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l1",
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE
        ))
    ])

    rfecv = RFECV(
        estimator=estimator,
        step=1,
        cv=cv,
        scoring="roc_auc",
        min_features_to_select=3,
        n_jobs=-1,
        importance_getter="named_steps.clf.coef_"
    )

    rfecv.fit(X, y, groups=groups)

    optimal_features = [f for f, s in zip(features, rfecv.support_) if s]

    #Plot CV score vs n_features
    n_scores = len(rfecv.cv_results_["mean_test_score"])
    x_range = range(rfecv.min_features_to_select,
                    rfecv.min_features_to_select + n_scores)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_range,
            rfecv.cv_results_["mean_test_score"],
            marker="o", linewidth=2, color="#2563EB", label="Mean ROC-AUC")
    ax.fill_between(
        x_range,
        rfecv.cv_results_["mean_test_score"] - rfecv.cv_results_["std_test_score"],
        rfecv.cv_results_["mean_test_score"] + rfecv.cv_results_["std_test_score"],
        alpha=0.2, color="#2563EB"
    )
    ax.axvline(rfecv.n_features_, color="#DC2626", linestyle="--",
               label=f"Optimal: {rfecv.n_features_} features")
    ax.set_xlabel("Number of Features Selected", fontsize=12)
    ax.set_ylabel("Group-CV ROC-AUC", fontsize=12)
    ax.set_title("RFECV — Optimal Feature Count", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("rfecv_curve.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"[Step 3] RFECV optimal features ({rfecv.n_features_}): {optimal_features}\n")
    return optimal_features, rfecv


# =============================================================================
# STEP 4 — MODEL COMPARISON: full vs high-tier vs pruned vs rfecv feature sets
# =============================================================================

def compare_models(df,
                   feature_sets,
                   condition_col=CONDITION_COL,
                   group_col=GROUP_COL,
                   ctrl_label=CTRL_LABEL,
                   hpmc_label=HPMC_LABEL,
                   n_splits=None):
    """
    Compara modelos con validación por cultivo (GroupKFold).
    """
    mask = df[condition_col].isin([ctrl_label, hpmc_label])

    y_raw = df.loc[mask, condition_col].values
    y = (y_raw == hpmc_label).astype(int)
    groups = df.loc[mask, group_col].values

    if n_splits is None:
        n_splits=len(np.unique(groups))

    cv = GroupKFold(n_splits=n_splits)

    models = {
        "LogReg (L2)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE
            ))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }

    scoring = ["roc_auc", "f1", "accuracy"]
    rows = []

    for feat_label, feats in feature_sets.items():
        X = df.loc[mask, feats].values

        for model_name, model in models.items():
            cv_res = cross_validate(
                model,
                X,
                y,
                cv=cv,
                groups=groups,
                scoring=scoring,
                return_train_score=False
            )

            rows.append({
                "Feature Set": feat_label,
                "Model": model_name,
                "n_features": len(feats),
                "features": feats.copy(),
                "ROC-AUC": f"{cv_res['test_roc_auc'].mean():.3f} ± {cv_res['test_roc_auc'].std():.3f}",
                "F1": f"{cv_res['test_f1'].mean():.3f} ± {cv_res['test_f1'].std():.3f}",
                "Accuracy": f"{cv_res['test_accuracy'].mean():.3f} ± {cv_res['test_accuracy'].std():.3f}",
                "_auc_mean": cv_res["test_roc_auc"].mean(),
                "_f1_mean": cv_res['test_f1'].mean(),
                "_acc_mean": cv_res['test_accuracy'].mean()
            })

    comparison_df = (
        pd.DataFrame(rows)
        .sort_values(["_auc_mean", "_f1_mean", "_acc_mean"], ascending=False)
        .reset_index(drop=True)
    )

    print("Model comparison results:")
    print(comparison_df.drop(columns=["_auc_mean", "_f1_mean", "_acc_mean"]).to_string(index=False))
    print()

    return comparison_df


# =============================================================================
# STEP 5 — INNER LOOP
# =============================================================================
from src.features import rank_features, print_report

def run_inner_loop_no_rfecv(df_outer_train,
                            condition_col=CONDITION_COL,
                            group_col=GROUP_COL,
                            ctrl_label=CTRL_LABEL,
                            hpmc_label=HPMC_LABEL,
                            corr_threshold=CORR_THRESHOLD):
    """
    Inner loop:
    1) rank_features sobre outer-train
    2) strong candidates
    3) correlation pruning
    4) compare_models sobre el set prunned
    5) elegir mejor modelo
    """

    # Stage 1 ranking SOLO con outer-train
    results_df = rank_features(
        df_outer_train,
        condition_col=condition_col,
        ctrl_label=ctrl_label,
        hpmc_label=hpmc_label
    )
    print_report(results_df, top_n=15)
    print()

    # Strong candidates
    strong = get_strong_candidates(results_df, df=df_outer_train)
    if len(strong) == 0:
        raise ValueError("No HIGH-tier features found in this outer fold.")

    # Correlation pruning
    pruned, corr_pairs = prune_correlated_features(
        df_outer_train,
        strong,
        results_df,
        threshold=corr_threshold
    )
    if len(pruned) == 0:
        raise ValueError("Correlation pruning removed all features in this outer fold.")

    # Feature set para comparar modelos
    feature_sets = {
        f"{len(pruned)} after corr-pruning": pruned
    }

    comparison = compare_models(
        df_outer_train,
        feature_sets,
        condition_col=condition_col,
        group_col=group_col,
        ctrl_label=ctrl_label,
        hpmc_label=hpmc_label,
        n_splits=len(df_outer_train[group_col].unique())
    )

    # Elegir mejor combinación por ROC-AUC, luego F1, luego Accuracy
    best_row = comparison.iloc[0]
    selected_model_name = best_row["Model"]
    selected_features = best_row["features"].copy()

    return {
        "results_df": results_df,
        "strong_candidates": strong,
        "after_corr_pruning": pruned,
        "corr_pairs": corr_pairs,
        "comparison_df": comparison,
        "selected_model_name": selected_model_name,
        "selected_features": selected_features,
    }


# =============================================================================
# MAIN ORCHESTRATOR OLD
# =============================================================================
"""
def run_full_pipeline(df,
                      results_df,
                      condition_col=CONDITION_COL,
                      group_col=GROUP_COL,
                      ctrl_label=CTRL_LABEL,
                      hpmc_label=HPMC_LABEL):
    """
"""
    Stage 2 completo:
    1) strong candidates
    2) correlation pruning
    3) RFECV con GroupKFold
    4) comparación de modelos
    5) interpretación exploratoria del modelo final
    """
"""
    print("=" * 70)
    print("  STAGE 2 — FEATURE SELECTION PIPELINE")
    print("=" * 70 + "\n")

    # Step 1
    strong = get_strong_candidates(results_df, df=df)

    # Step 2
    pruned, corr_pairs = prune_correlated_features(
        df, strong, results_df, threshold=CORR_THRESHOLD
    )

    # Step 3
    optimal, rfecv_obj = run_rfecv(
        df, pruned,
        condition_col=condition_col,
        group_col=group_col,
        ctrl_label=ctrl_label,
        hpmc_label=hpmc_label
    )

    # Step 4
    all_ranked_features = [f for f in results_df["feature"].tolist() if f in df.columns]

    feature_sets = {
        f"All {len(all_ranked_features)} ranked features": all_ranked_features,
        f"{len(strong)} HIGH-tier features": strong,
        f"{len(pruned)} after corr-pruning": pruned,
        f"{len(optimal)} RFECV optimal": optimal,
    }

    comparison = compare_models(
        df, feature_sets,
        condition_col=condition_col,
        group_col=group_col,
        ctrl_label=ctrl_label,
        hpmc_label=hpmc_label
    )

    # Step 5 (exploratory)
    evaluate_final_model(
        df, optimal,
        condition_col=condition_col,
        ctrl_label=ctrl_label,
        hpmc_label=hpmc_label
    )

    print("=" * 70)
    print("  PIPELINE COMPLETE")
    print(f"  Final feature set ({len(optimal)}): {optimal}")
    print("=" * 70)

    return {
        "strong_candidates": strong,
        "after_corr_pruning": pruned,
        "corr_pairs": corr_pairs,
        "rfecv_optimal": optimal,
        "rfecv_object": rfecv_obj,
        "model_comparison": comparison,
    }
"""