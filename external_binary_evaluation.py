import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

from src.preprocessing import extract_green_channel, normalize_dataset
from final_binary_model import (
    FINAL_FEATURES,
    build_feature_dataset,
    load_final_binary_model
)
from load_data_evaluation import load_evaluation_images


CTRL_LABEL = "CTRL"
HPMC_LABEL = "HPMC"

DISPLAY_LABELS = {
    "CTRL": "Control",
    "HPMC": "Pro-inflammatory (LPS)"
}


def train_final_random_forest(df,
                              feature_list=FINAL_FEATURES,
                              condition_col="condition",
                              ctrl_label=CTRL_LABEL,
                              hpmc_label=HPMC_LABEL):
    """
    RF solo como comparación secundaria.
    Se entrena sobre TODO el dataset original.
    """
    mask = df[condition_col].isin([ctrl_label, hpmc_label])

    X = df.loc[mask, feature_list].copy()
    y = (df.loc[mask, condition_col] == hpmc_label).astype(int)

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    return model


def evaluate_predictions(y_true, y_pred, y_score):
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
        "tp": tp
    }
    return metrics, cm


def evaluate_model_on_external_df(model, eval_df, feature_list, model_name="Model"):
    X = eval_df[feature_list].copy()
    y_true = (eval_df["condition"] == HPMC_LABEL).astype(int).values

    y_pred = model.predict(X)
    y_score = model.predict_proba(X)[:, 1]

    metrics, cm = evaluate_predictions(y_true, y_pred, y_score)

    return {
        "model_name": model_name,
        "metrics": metrics,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_score
    }


def evaluate_by_group(model, eval_df, feature_list, model_name="Model"):
    rows = []

    for group in sorted(eval_df["group"].unique()):
        sub = eval_df[eval_df["group"] == group].copy()
        result = evaluate_model_on_external_df(model, sub, feature_list, model_name)

        m = result["metrics"]
        rows.append({
            "model": model_name,
            "group": group,
            "n_images": len(sub),
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "specificity": m["specificity"],
            "f1": m["f1"],
            "roc_auc": m["roc_auc"],
            "tn": m["tn"],
            "fp": m["fp"],
            "fn": m["fn"],
            "tp": m["tp"]
        })

    return pd.DataFrame(rows)


def plot_roc_curves(results_dict, save_path="external_roc_curve.png"):
    plt.figure(figsize=(7, 6))

    for model_name, result in results_dict.items():
        fpr, tpr, _ = roc_curve(result["y_true"], result["y_score"])
        auc = roc_auc_score(result["y_true"], result["y_score"])
        plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("External ROC curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Saved ROC curve → {save_path}")


def run_external_binary_evaluation(train_df=None,
                                   eval_base_path="data/Evaluacion",
                                   q=80,
                                   variant="closing_r3",
                                   prune_iters=5,
                                   include_random_forest=False):
    """
    train_df:
        Solo hace falta si include_random_forest=True.
        Para LR usa el modelo ya guardado.
    """

    # 1) cargar dataset externo
    eval_dataset = load_evaluation_images(eval_base_path)

    eval_dataset = extract_green_channel(eval_dataset)
    eval_dataset = normalize_dataset(eval_dataset, robust=True)

    # 2) extraer features externas con el MISMO pipeline
    eval_df = build_feature_dataset(
        eval_dataset,
        q=q,
        variant=variant,
        prune_iters=prune_iters
    )

    print("\n[External set composition]")
    print(eval_df.groupby(["group", "condition"]).size())

    # 3) cargar modelo final guardado de LR
    lr_model, lr_metadata = load_final_binary_model()

    # chequeo de consistencia
    expected_features = lr_metadata["features"]
    if expected_features != FINAL_FEATURES:
        print("[WARNING] FINAL_FEATURES y metadata['features'] no coinciden.")
        print("Using metadata features from saved model.")
    feature_list = expected_features

    # 4) evaluar LR
    lr_result = evaluate_model_on_external_df(
        model=lr_model,
        eval_df=eval_df,
        feature_list=feature_list,
        model_name="Logistic Regression"
    )

    lr_by_group = evaluate_by_group(
        model=lr_model,
        eval_df=eval_df,
        feature_list=feature_list,
        model_name="Logistic Regression"
    )

    summary_rows = [{
        "model": "Logistic Regression",
        **lr_result["metrics"]
    }]

    results_for_roc = {
        "Logistic Regression": lr_result
    }

    by_group_df = lr_by_group.copy()

    #CHEQUEOS!!
    lr_prob_df = inspect_external_probabilities(
        model=lr_model,
        eval_df=eval_df,
        feature_list=feature_list
    )
    feature_shift_df = None
    if train_df is not None:
        feature_shift_df = compare_train_vs_external_by_condition(
            train_df=train_df,
            eval_df=eval_df,
            feature_list=feature_list
        )

    # 5) opcional: RF como comparación secundaria
    if include_random_forest:
        if train_df is None:
            raise ValueError("train_df is required when include_random_forest=True")

        rf_model = train_final_random_forest(
            train_df,
            feature_list=feature_list
        )

        rf_result = evaluate_model_on_external_df(
            model=rf_model,
            eval_df=eval_df,
            feature_list=feature_list,
            model_name="Random Forest"
        )

        rf_by_group = evaluate_by_group(
            model=rf_model,
            eval_df=eval_df,
            feature_list=feature_list,
            model_name="Random Forest"
        )

        summary_rows.append({
            "model": "Random Forest",
            **rf_result["metrics"]
        })

        by_group_df = pd.concat([by_group_df, rf_by_group], ignore_index=True)
        results_for_roc["Random Forest"] = rf_result

    summary_df = pd.DataFrame(summary_rows)

    print("\n[External evaluation - GLOBAL]")
    print(summary_df.to_string(index=False))

    print("\n[External evaluation - BY GROUP]")
    print(by_group_df.to_string(index=False))

    plot_roc_curves(results_for_roc)

    return {
        "eval_df": eval_df,
        "summary_df": summary_df,
        "by_group_df": by_group_df,
        "results_for_roc": results_for_roc,
        "lr_prob_df": lr_prob_df,
        "feature_shift_df": feature_shift_df
    }

def plot_confusion_matrix_from_result(result,
                                      class_names=("Control", "Pro-inflammatory (LPS)"),
                                      normalize=False,
                                      save_path="confusion_matrix.png"):
    """
    Grafica la matriz de confusión a partir del dict result
    devuelto por evaluate_model_on_external_df().
    """
    cm = result["confusion_matrix"].copy()

    if normalize:
        cm = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, where=row_sums != 0)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Confusion Matrix — {result['model_name']}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Saved confusion matrix → {save_path}")

def inspect_external_probabilities(model, eval_df, feature_list):
    X = eval_df[feature_list].copy()
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)

    out = eval_df[["image_name", "group", "condition"]].copy()
    out["prob_HPMC"] = probs
    out["pred_numeric"] = preds
    out["pred_label"] = out["pred_numeric"].map({0: "CTRL", 1: "HPMC"})
    return out.sort_values("prob_HPMC", ascending=False).reset_index(drop=True)


def compare_train_vs_external_by_condition(train_df, eval_df, feature_list,
                                           condition_col="condition",
                                           ctrl_label="CTRL",
                                           hpmc_label="HPMC"):
    rows = []

    for feat in feature_list:
        for cond in [ctrl_label, hpmc_label]:
            train_vals = train_df[train_df[condition_col] == cond][feat]
            eval_vals = eval_df[eval_df[condition_col] == cond][feat]

            rows.append({
                "feature": feat,
                "condition": cond,
                "train_mean": train_vals.mean(),
                "train_std": train_vals.std(),
                "eval_mean": eval_vals.mean(),
                "eval_std": eval_vals.std(),
            })

    return pd.DataFrame(rows)