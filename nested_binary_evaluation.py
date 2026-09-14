import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

from feature_selection_pipeline import (
    run_inner_loop_no_rfecv,
    GROUP_COL,
    CONDITION_COL,
    CTRL_LABEL,
    HPMC_LABEL
)
from final_binary_model import train_selected_model

def evaluate_outer_fold(model, df_test, selected_features,
                        condition_col=CONDITION_COL,
                        ctrl_label=CTRL_LABEL,
                        hpmc_label=HPMC_LABEL):
    mask = df_test[condition_col].isin([ctrl_label, hpmc_label])

    X_test = df_test.loc[mask, selected_features].copy()
    y_true = (df_test.loc[mask, condition_col] == hpmc_label).astype(int).values

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }

    pred_df = df_test.loc[mask, ["image_name", "group", "condition"]].copy()
    pred_df["y_true"] = y_true
    pred_df["y_pred"] = y_pred
    pred_df["y_score"] = y_score

    return metrics, pred_df

def run_nested_leave_one_culture_out(df,
                                     group_col=GROUP_COL,
                                     condition_col=CONDITION_COL,
                                     ctrl_label=CTRL_LABEL,
                                     hpmc_label=HPMC_LABEL):
    """
    Outer loop:
    deja 1 cultivo afuera como test.
    Inner loop:
    ranking -> strong -> corr pruning -> compare_models.
    """

    groups = sorted(df[group_col].unique())
    outer_rows = []
    outer_pred_dfs = []
    outer_selected = []

    for outer_test_group in groups:
        print("=" * 80)
        print(f"[OUTER TEST GROUP] {outer_test_group}")

        df_outer_train = df[df[group_col] != outer_test_group].copy()
        df_outer_test  = df[df[group_col] == outer_test_group].copy()

        # INNER LOOP
        inner_out = run_inner_loop_no_rfecv(
            df_outer_train,
            condition_col=condition_col,
            group_col=group_col,
            ctrl_label=ctrl_label,
            hpmc_label=hpmc_label
        )

        selected_model_name = inner_out["selected_model_name"]
        selected_features = inner_out["selected_features"]

        print(f"Selected model: {selected_model_name}")
        print(f"Selected features: {selected_features}")

        # reentrenar en TODO el outer-train
        final_model = train_selected_model(
            df_outer_train,
            selected_features=selected_features,
            model_name=selected_model_name,
            condition_col=condition_col,
            ctrl_label=ctrl_label,
            hpmc_label=hpmc_label
        )

        # evaluar en outer-test
        metrics, pred_df = evaluate_outer_fold(
            final_model,
            df_outer_test,
            selected_features=selected_features,
            condition_col=condition_col,
            ctrl_label=ctrl_label,
            hpmc_label=hpmc_label
        )

        outer_rows.append({
            "outer_test_group": outer_test_group,
            "selected_model": selected_model_name,
            "selected_features": ", ".join(selected_features),
            **metrics
        })

        pred_df["outer_test_group"] = outer_test_group
        pred_df["selected_model"] = selected_model_name
        pred_df["selected_features"] = ", ".join(selected_features)
        outer_pred_dfs.append(pred_df)

        outer_selected.append({
            "outer_test_group": outer_test_group,
            "selected_model": selected_model_name,
            "selected_features": selected_features,
            "comparison_df": inner_out["comparison_df"],
            "results_df": inner_out["results_df"],
            "corr_pairs": inner_out["corr_pairs"],
        })

    summary_df = pd.DataFrame(outer_rows)
    pred_all_df = pd.concat(outer_pred_dfs, ignore_index=True)

    global_metrics, global_cm = evaluate_global_from_oof(pred_all_df)

    return {
        "summary_df": summary_df,
        "pred_all_df": pred_all_df,
        "outer_details": outer_selected,
        "global_metrics": global_metrics,
        "global_confusion_matrix": global_cm
    }

def evaluate_global_from_oof(pred_all_df):
    """
    Calcula métricas globales a partir de todas las predicciones out-of-fold
    concatenadas de los outer folds.

    Espera columnas:
    - y_true
    - y_pred
    - y_score
    """
    y_true = pred_all_df["y_true"].values
    y_pred = pred_all_df["y_pred"].values
    y_score = pred_all_df["y_score"].values

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

    return metrics, cm

def plot_global_roc_from_oof(pred_all_df,
    save_path="nested_global_roc.png",
    title="Global ROC curve (out-of-fold)"):
    """
    Grafica curva ROC global usando todas las predicciones out-of-fold.
    """
    y_true = pred_all_df["y_true"].values
    y_score = pred_all_df["y_score"].values

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Saved ROC curve → {save_path}")

def plot_global_confusion_matrix_from_oof(pred_all_df,
    class_names=("CTRL", "HPMC"),
    normalize=False,
    save_path="nested_global_confusion_matrix.png",
    title="Global confusion matrix (out-of-fold)"):
    """
    Grafica matriz de confusión global usando todas las predicciones out-of-fold.
    """
    y_true = pred_all_df["y_true"].values
    y_pred = pred_all_df["y_pred"].values

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm_plot = cm.astype(float)
        row_sums = cm_plot.sum(axis=1, keepdims=True)
        cm_plot = np.divide(cm_plot, row_sums, where=row_sums != 0)
        fmt = ".2f"
    else:
        cm_plot = cm
        fmt = "d"

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Saved confusion matrix → {save_path}")