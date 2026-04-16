import pandas as pd
from pipeline import process_single_image

def build_temporal_feature_dataset(dataset,
                                   q=80,
                                   variant="closing_r3",
                                   prune_iters=5):
    """
    Recorre el dataset temporal y arma un DataFrame de features.
    dataset[time_label][group] = [
        {"name": ..., "image": ..., "group": ..., "time_label": ..., "time_h": ...},
        ...
    ]
    """
    rows = []

    for time_label, groups in dataset.items():
        for group, imgs in groups.items():
            for item in imgs:
                img_name = item["name"]
                img = item["image"]

                result = process_single_image(
                    img,
                    q=q,
                    variant=variant,
                    prune_iters=prune_iters
                )

                feats = result["features"].copy()
                feats["time_label"] = item["time_label"]
                feats["time_h"] = item["time_h"]
                feats["group"] = item["group"]
                feats["image_name"] = img_name

                rows.append(feats)

    return pd.DataFrame(rows)

# Plot features finales de binario vs tiempo
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal

def plot_temporal_feature_boxplots(df, save_path="temporal_feature_boxplots.png"):
    features = [
        ("median_thickness", "Median thickness"),
        ("median_tortuosity", "Median tortuosity"),
        ("median_segment_length", "Median segment length"),
    ]

    time_order = ["0HS", "24HS", "48HS", "72HS"]

    fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=True)

    for ax, (feat, label) in zip(axes, features):
        sns.boxplot(
            data=df,
            x="time_label",
            y=feat,
            order=time_order,
            ax=ax,
            color="#A7C7E7",
            fliersize=0,
            linewidth=1.4
        )

        sns.stripplot(
            data=df,
            x="time_label",
            y=feat,
            order=time_order,
            ax=ax,
            color="black",
            alpha=0.7,
            size=4,
            jitter=0.12
        )

        groups = [df[df["time_label"] == t][feat].dropna().values for t in time_order]
        _, pval = kruskal(*groups)

        p_text = "Kruskal–Wallis p < 0.001" if pval < 0.001 else f"Kruskal–Wallis p = {pval:.3f}"

        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.text(0.5, 0.95, p_text, transform=ax.transAxes,
                ha="center", va="top", fontsize=11)

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(True, axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Time after treatment", fontsize=11)

    fig.suptitle("Temporal evolution of selected morphological features",
                 fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Saved → {save_path}")

# Score continuo 
import numpy as np
import pandas as pd

def minmax_norm(series):
    s = pd.Series(series, dtype=float)
    s_min, s_max = s.min(), s.max()
    if s_max > s_min:
        return (s - s_min) / (s_max - s_min)
    return pd.Series(np.full(len(s), 0.5), index=s.index)

def add_proinflammatory_score(df,
                              thickness_col="median_thickness",
                              segment_col="median_segment_length"):
    df = df.copy()

    # normalización global
    df["thickness_norm"] = minmax_norm(df[thickness_col])
    df["segment_length_norm"] = minmax_norm(df[segment_col])

    # invertir thickness: menos grosor = más score proinflamatorio
    df["thickness_inverted"] = 1.0 - df["thickness_norm"]

    # score base [0,1]
    df["proinflammatory_score_0_1"] = (
        0.5 * df["thickness_inverted"] +
        0.5 * df["segment_length_norm"]
    )

    # score [0,100]
    df["proinflammatory_score"] = df["proinflammatory_score_0_1"] * 100.0

    return df

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal

def plot_proinflammatory_score(df, save_path="proinflammatory_score_boxplot.png"):
    time_order = ["0HS", "24HS", "48HS", "72HS"]

    plt.figure(figsize=(8, 6))

    sns.boxplot(
        data=df,
        x="time_label",
        y="proinflammatory_score",
        order=time_order,
        color="#B7D3E9",
        fliersize=0,
        linewidth=1.4
    )

    sns.stripplot(
        data=df,
        x="time_label",
        y="proinflammatory_score",
        order=time_order,
        color="black",
        alpha=0.7,
        size=4,
        jitter=0.12
    )

    groups = [df[df["time_label"] == t]["proinflammatory_score"].dropna().values for t in time_order]
    _, pval = kruskal(*groups)

    if pval < 0.001:
        p_text = "Kruskal–Wallis p < 0.001"
    else:
        p_text = f"Kruskal–Wallis p = {pval:.3f}"

    plt.title("Pro-inflammatory progression score", fontsize=14, fontweight="bold")
    plt.xlabel("Time after treatment")
    plt.ylabel("Score (0–100)")
    plt.text(0.5, 0.95, p_text, transform=plt.gca().transAxes,
             ha="center", va="top", fontsize=11)

    plt.grid(True, axis="y", alpha=0.25)
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Saved → {save_path}")

#PIPELINE FINAL PARA EL EL CASO TEMPORAL
import joblib
import pandas as pd
import numpy as np
from final_binary_model import prepare_single_image

TEMPORAL_FEATURES = [
    "median_thickness",
    "median_segment_length"
]


def minmax_norm_single(value, min_val, max_val):
    """
    Normaliza un valor individual a [0,1] usando min-max global.
    """
    if max_val > min_val:
        return (value - min_val) / (max_val - min_val)
    return 0.5


def get_progression_category(score):
    """
    Categoría interpretativa del score temporal.
    """
    if score < 25:
        return "low inflammatory morphology"
    elif score < 50:
        return "mild inflammatory morphology"
    elif score < 75:
        return "moderate inflammatory morphology"
    else:
        return "high inflammatory morphology"


def build_temporal_score_metadata(df):
    """
    Construye y devuelve la metadata necesaria para calcular
    el score temporal de forma consistente en imágenes nuevas.
    """
    metadata = {
        "features": TEMPORAL_FEATURES,
        "thickness_min": float(df["median_thickness"].min()),
        "thickness_max": float(df["median_thickness"].max()),
        "segment_min": float(df["median_segment_length"].min()),
        "segment_max": float(df["median_segment_length"].max()),
        "score_formula": "0.5 * (1 - thickness_norm) + 0.5 * segment_length_norm",
        "score_range": [0, 100],
        "categories": {
            "low": [0, 25],
            "mild": [25, 50],
            "moderate": [50, 75],
            "high": [75, 100]
        },
        "task": "temporal_proinflammatory_progression"
    }
    return metadata


def save_temporal_score_model(df,
                              metadata_path="temporal_score_metadata.joblib"):
    """
    Guarda la metadata del score temporal.
    No entrena un modelo ML porque el score es explícito e interpretable.
    """
    metadata = build_temporal_score_metadata(df)
    joblib.dump(metadata, metadata_path)

    print(f"Temporal score metadata guardada en: {metadata_path}")
    return metadata


def load_temporal_score_model(metadata_path="temporal_score_metadata.joblib"):
    """
    Carga la metadata del score temporal.
    """
    metadata = joblib.load(metadata_path)
    return metadata


def predict_temporal_progression_score(img,
                                       metadata_path="temporal_score_metadata.joblib",
                                       robust=True):
    """
    Calcula el score temporal/proinflamatorio para una imagen nueva.

    Devuelve:
    - score 0–100
    - categoría interpretativa
    - features usadas
    - todas las features extraídas
    - outputs intermedios: wavelet, mask, skeleton
    """
    metadata = load_temporal_score_model(metadata_path)

    # 1) preparar imagen
    prepared_img = prepare_single_image(img, robust=robust)

    # 2) procesar imagen completa con el mismo pipeline
    result = process_single_image(
        prepared_img,
        q=80,
        variant="closing_r3",
        prune_iters=5
    )

    all_features = result["features"]

    # 3) features del score
    thickness = float(all_features["median_thickness"])
    segment_length = float(all_features["median_segment_length"])

    thickness_norm = minmax_norm_single(
        thickness,
        metadata["thickness_min"],
        metadata["thickness_max"]
    )

    segment_norm = minmax_norm_single(
        segment_length,
        metadata["segment_min"],
        metadata["segment_max"]
    )

    thickness_inverted = 1.0 - thickness_norm

    score_0_1 = 0.5 * thickness_inverted + 0.5 * segment_norm
    score = float(score_0_1 * 100.0)

    category = get_progression_category(score)

    temporal_features = {
        "median_thickness": thickness,
        "median_segment_length": segment_length
    }

    return {
        "score": score,
        "category": category,
        "score_features": temporal_features,
        "all_features": all_features,
        "wavelet": result["wavelet"],
        "mask": result["mask"],
        "skeleton": result["skeleton"]
    }

# Chequeo con un ejemplo de cada categoria temporal
def test_temporal_progression_on_four_images(temporal_dataset,
                                             group="N1",
                                             image_idx=0,
                                             metadata_path="temporal_score_metadata.joblib"):
    """
    Toma una imagen del mismo índice para 0HS, 24HS, 48HS y 72HS
    dentro de un mismo grupo/cultivo, calcula el score temporal
    y muestra si la progresión es creciente.

    Parámetros
    ----------
    temporal_dataset : dict
        Dataset temporal cargado con estructura:
        temporal_dataset[time_label][group] = [ {...}, {...}, ... ]
    group : str
        Grupo/cultivo a usar (ej. "N1")
    image_idx : int
        Índice de la imagen dentro de cada grupo (0 a 3)
    metadata_path : str
        Ruta al archivo .joblib con metadata del score temporal
    """

    time_order = ["0HS", "24HS", "48HS", "72HS"]
    rows = []

    for time_label in time_order:
        item = temporal_dataset[time_label][group][image_idx]
        img = item["image"]
        img_name = item["name"]

        result = predict_temporal_progression_score(
            img,
            metadata_path=metadata_path
        )

        rows.append({
            "time_label": time_label,
            "group": group,
            "image_idx": image_idx,
            "image_name": img_name,
            "score": result["score"],
            "category": result["category"],
            "median_thickness": result["score_features"]["median_thickness"],
            "median_segment_length": result["score_features"]["median_segment_length"]
        })

    df_test = pd.DataFrame(rows)

    print(df_test)

    scores = df_test["score"].values
    is_monotonic_increasing = all(scores[i] <= scores[i + 1] for i in range(len(scores) - 1))

    print("\n¿El score sube progresivamente?")
    print(is_monotonic_increasing)

    return df_test

def test_temporal_progression_all_groups(temporal_dataset,
                                         image_idx=0,
                                         metadata_path="temporal_score_metadata.joblib"):
    """
    Prueba una imagen del mismo índice en todos los grupos (N1-N4)
    para verificar progresión temporal del score.
    """
    all_results = []

    for group in sorted(temporal_dataset["0HS"].keys()):
        df_group = test_temporal_progression_on_four_images(
            temporal_dataset,
            group=group,
            image_idx=image_idx,
            metadata_path=metadata_path
        )
        all_results.append(df_group)

    return pd.concat(all_results, ignore_index=True)