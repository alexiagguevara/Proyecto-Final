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
"""
Dos modos de análisis:

  Modo 2A  —  Morphological Recovery Score anclado a referencias  (RECOMENDADO)
              Requiere: imagen nueva + imagen control + imagen inflamada
              del mismo experimento.
              0   = igual a la referencia inflamada
              100 = igual a la referencia control
              >100 = más basal que el control
              <0   = más inflamada que la referencia inflamada

  Modo 2B  —  Morphological Recovery Score absoluto  (fallback)
              Requiere: solo la imagen nueva.
              Usa referencias poblacionales del training set (0HS y 72HS).
              Menos robusto ante variabilidad entre réplicas.
              Incluye advertencia explícita.

Decisiones metodológicas
-------------------------
  - Features: median_thickness (rho=-0.918) y median_segment_length (rho=+0.747)
  - Pesos derivados de Spearman:
        w_thickness = 0.918 / (0.918 + 0.747) ≈ 0.551
        w_segment   = 0.747 / (0.918 + 0.747) ≈ 0.449
  - Cortes de categorías para recovery:
        0–34   → low recovery
        34–52  → mild recovery
        52–71  → moderate recovery
        71+    → high recovery
  - El score no se clampea a [0, 100]; valores fuera del rango son informativos.

"""

import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.stats import kruskal

from pipeline import process_single_image
from final_binary_model import prepare_single_image


TEMPORAL_FEATURES = [
    "median_thickness",
    "median_segment_length"
]

TIME_ORDER = ["0HS", "24HS", "48HS", "72HS"]

INFLAMAMATORY_LABELS = [
    "low inflammatory morphology",
    "mild inflammatory morphology",
    "moderate inflammatory morphology",
    "high inflammatory morphology",
]


def get_temporal_feature_weights(results_temporal,
                                 features=TEMPORAL_FEATURES):
    """
    Calcula pesos a partir de |spearman_rho| guardado en results_temporal.
    """
    subset = results_temporal[results_temporal["feature"].isin(features)].copy()

    if len(subset) != len(features):
        missing = set(features) - set(subset["feature"])
        raise ValueError(f"Faltan features en results_temporal: {missing}")

    subset["abs_rho"] = subset["spearman_rho"].abs()
    total = subset["abs_rho"].sum()

    if total == 0:
        raise ValueError("La suma de |rho| es 0, no se pueden calcular pesos.")

    return {
        row["feature"]: float(row["abs_rho"] / total)
        for _, row in subset.iterrows()
    }


def get_population_anchors(df,
                           thickness_col="median_thickness",
                           segment_col="median_segment_length"):
    """
    Usa medias poblacionales de 0HS y 72HS como anchors del score absoluto.
    """
    ctrl = df[df["time_label"] == "0HS"]
    inflam = df[df["time_label"] == "72HS"]

    return {
        "ctrl_ref": {
            "median_thickness": float(ctrl[thickness_col].mean()),
            "median_segment_length": float(ctrl[segment_col].mean()),
        },
        "inflam_ref": {
            "median_thickness": float(inflam[thickness_col].mean()),
            "median_segment_length": float(inflam[segment_col].mean()),
        }
    }

def build_group_time_summary(df,
                             thickness_col="median_thickness",
                             segment_col="median_segment_length"):
    """
    Resume el dataframe temporal a nivel grupo × tiempo.
    Cada fila representa el valor promedio de un cultivo en un tiempo dado.
    """
    df_group = (
        df.groupby(["group", "time_label", "time_h"], as_index=False)
        .agg({
            thickness_col: "mean",
            segment_col: "mean"
        })
        .sort_values(["group", "time_h"])
        .reset_index(drop=True)
    )
    return df_group

def get_group_level_weights(df_group,
                            thickness_col="median_thickness",
                            segment_col="median_segment_length"):
    """
    Calcula pesos usando Spearman rho a nivel grupo × tiempo.
    """
    rho_thickness, p_thickness = spearmanr(df_group["time_h"], df_group[thickness_col])
    rho_segment, p_segment = spearmanr(df_group["time_h"], df_group[segment_col])

    abs_rho_thickness = abs(rho_thickness)
    abs_rho_segment = abs(rho_segment)

    total = abs_rho_thickness + abs_rho_segment
    if total == 0:
        raise ValueError("La suma de |rho| a nivel grupo es 0.")

    weights = {
        "median_thickness": float(abs_rho_thickness / total),
        "median_segment_length": float(abs_rho_segment / total),
    }

    rho_info = {
        "median_thickness": {
            "rho": float(rho_thickness),
            "p_value": float(p_thickness),
        },
        "median_segment_length": {
            "rho": float(rho_segment),
            "p_value": float(p_segment),
        }
    }

    return weights, rho_info

def _feature_score(value, ctrl_ref, inflam_ref, direction):
    """
    Score inflamatorio interno:
      0 = control
      1 = inflamada
    """
    span = abs(inflam_ref - ctrl_ref)
    if span < 1e-12:
        return 0.5

    if direction == "decrease":
        return (ctrl_ref - value) / (ctrl_ref - inflam_ref)
    elif direction == "increase":
        return (value - ctrl_ref) / (inflam_ref - ctrl_ref)
    else:
        raise ValueError(f"direction inválida: {direction}")


def _compute_absolute_inflammatory_score(thickness,
                                         segment_length,
                                         ctrl_thickness,
                                         inflam_thickness,
                                         ctrl_segment,
                                         inflam_segment,
                                         w_thickness,
                                         w_segment):
    """
    Score absoluto corregido:
      0   = morfología tipo control
      100 = morfología tipo 72HS
    """
    s_t = _feature_score(thickness, ctrl_thickness, inflam_thickness, "decrease")
    s_s = _feature_score(segment_length, ctrl_segment, inflam_segment, "increase")

    score_01 = w_thickness * s_t + w_segment * s_s
    return float(score_01 * 100.0)


def get_category_thresholds_from_data(df_with_score,
                                      score_col="absolute_inflammatory_score",
                                      time_order=TIME_ORDER):
    """
    Calcula umbrales como puntos medios entre medias adyacentes.
    """
    means = (
        df_with_score.groupby("time_label")[score_col]
        .mean()
        .reindex(time_order)
    )

    thresholds = []
    for i in range(len(means) - 1):
        thresholds.append(float((means.iloc[i] + means.iloc[i + 1]) / 2.0))

    return {
        "means_by_time": means.to_dict(),
        "thresholds": thresholds
    }


def get_progression_category(score, thresholds):
    """
    Categorías basadas en thresholds derivados de los datos.
    """
    for i, t in enumerate(thresholds):
        if score < t:
            return INFLAMAMATORY_LABELS[i]
    return INFLAMAMATORY_LABELS[-1]

def get_recovery_thresholds_from_inflammatory(thresholds):
    """
    Convierte thresholds del score inflamatorio a thresholds del recovery score.
    """
    return [100 - t for t in reversed(thresholds)]


def get_recovery_category(score, recovery_thresholds):
    """
    Categoría interpretativa del recovery score.
    """
    if score < recovery_thresholds[0]:
        return "minimal recovery"
    elif score < recovery_thresholds[1]:
        return "partial recovery"
    elif score < recovery_thresholds[2]:
        return "substantial recovery"
    else:
        return "near-complete recovery"

def build_temporal_score_metadata(df):
    """
    Construye metadata del score absoluto corregido usando calibración
    a nivel grupo × tiempo.

    Esto permite:
    - respetar la estructura biológica del experimento,
    - reducir pseudorreplicación,
    - estimar pesos y thresholds de forma más robusta.
    """
    # 1) resumen grupo × tiempo
    df_group = build_group_time_summary(df)

    # 2) anchors poblacionales
    anchors = get_population_anchors(
        df_group,
        thickness_col="median_thickness",
        segment_col="median_segment_length"
    )

    # 3) pesos a nivel grupo
    weights, rho_info = get_group_level_weights(
        df_group,
        thickness_col="median_thickness",
        segment_col="median_segment_length"
    )

    # 4) construir score sobre tabla resumida para derivar thresholds
    df_group_tmp = df_group.copy()

    ctrl_t = anchors["ctrl_ref"]["median_thickness"]
    ctrl_s = anchors["ctrl_ref"]["median_segment_length"]
    inflam_t = anchors["inflam_ref"]["median_thickness"]
    inflam_s = anchors["inflam_ref"]["median_segment_length"]

    w_t = weights["median_thickness"]
    w_s = weights["median_segment_length"]

    df_group_tmp["absolute_inflammatory_score"] = df_group_tmp.apply(
        lambda row: _compute_absolute_inflammatory_score(
            thickness=row["median_thickness"],
            segment_length=row["median_segment_length"],
            ctrl_thickness=ctrl_t,
            inflam_thickness=inflam_t,
            ctrl_segment=ctrl_s,
            inflam_segment=inflam_s,
            w_thickness=w_t,
            w_segment=w_s
        ),
        axis=1
    )

    threshold_info = get_category_thresholds_from_data(
        df_group_tmp,
        score_col="absolute_inflammatory_score"
    )

    recovery_thresholds = get_recovery_thresholds_from_inflammatory(
        threshold_info["thresholds"]
    )

    metadata = {
        "features": TEMPORAL_FEATURES,
        "weights": weights,
        "rho_group_level": rho_info,
        "population_anchors": anchors,
        "thresholds": threshold_info["thresholds"],
        "recovery_thresholds": recovery_thresholds,
        "means_by_time": threshold_info["means_by_time"],
        "task": "temporal_absolute_inflammatory_score",
        "calibration_level": "group_time",
        "warning": (
            "Score calculado sin referencias del experimento actual. "
            "La interpretación puede verse afectada por variabilidad biológica entre réplicas."
        )
    }

    return metadata


def save_temporal_score_model(df,
                              metadata_path="temporal_score_metadata.joblib"):
    """
    Guarda la metadata del score temporal absoluto,
    calibrado a nivel grupo × tiempo.
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
    Calcula el score temporal absoluto para una imagen nueva.

    Devuelve:
    - score absoluto inflamatorio
    - categoría interpretativa
    - features usadas
    - todas las features extraídas
    - outputs intermedios: wavelet, mask, skeleton
    """
    metadata = load_temporal_score_model(metadata_path)

    anchors = metadata["population_anchors"]
    weights = metadata["weights"]
    thresholds = metadata["thresholds"]

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

    score = _compute_absolute_inflammatory_score(
        thickness=thickness,
        segment_length=segment_length,
        ctrl_thickness=anchors["ctrl_ref"]["median_thickness"],
        inflam_thickness=anchors["inflam_ref"]["median_thickness"],
        ctrl_segment=anchors["ctrl_ref"]["median_segment_length"],
        inflam_segment=anchors["inflam_ref"]["median_segment_length"],
        w_thickness=weights["median_thickness"],
        w_segment=weights["median_segment_length"]
    )

    category = get_progression_category(score, thresholds)

    temporal_features = {
        "median_thickness": thickness,
        "median_segment_length": segment_length
    }

    return {
        "infalmmatory_score": score,
        "category": category,
        "warning": metadata["warning"],
        "score_features": temporal_features,
        "all_features": all_features,
        "wavelet": result["wavelet"],
        "mask": result["mask"],
        "skeleton": result["skeleton"],
        "mode": "absolute_fallback"
    }

# score relativo
def compute_reference_anchor_from_images(image_list, robust=True):
    """
    Calcula un anchor promedio a partir de una lista de imágenes de referencia.

    Extrae las features necesarias de cada imagen y devuelve el promedio de:
    - median_thickness
    - median_segment_length
    """
    if image_list is None or len(image_list) == 0:
        raise ValueError("La lista de imágenes de referencia está vacía.")

    thickness_values = []
    segment_values = []

    for img in image_list:
        prepared_img = prepare_single_image(img, robust=robust)

        result = process_single_image(
            prepared_img,
            q=80,
            variant="closing_r3",
            prune_iters=5
        )

        feats = result["features"]
        thickness_values.append(float(feats["median_thickness"]))
        segment_values.append(float(feats["median_segment_length"]))

    return {
        "median_thickness": float(np.mean(thickness_values)),
        "median_segment_length": float(np.mean(segment_values)),
        "n_images": len(image_list),
        "all_thickness_values": thickness_values,
        "all_segment_values": segment_values,
    }

def predict_temporal_progression_score_anchored(img_new,
                                                ctrl_images,
                                                inflam_images,
                                                metadata_path="temporal_score_metadata.joblib",
                                                robust=True):
    """
    Calcula el score temporal relativo/anclado usando:
    - una imagen nueva
    - una lista de imágenes control del mismo experimento
    - una lista de imágenes inflamadas del mismo experimento

    El score sigue orientado como score proinflamatorio:
      0   = igual al control promedio
      100 = igual a la inflamada promedio

    También devuelve el recovery score complementario:
      recovery_score = 100 - inflammatory_score

    Devuelve:
    - inflammatory_score
    - recovery_score
    - categoría interpretativa basada en inflammatory_score
    - features usadas
    - features de referencia
    - todas las features extraídas de la imagen nueva
    - outputs intermedios: wavelet, mask, skeleton
    """
    metadata = load_temporal_score_model(metadata_path)

    weights = metadata["weights"]
    recovery_thresholds = metadata["recovery_thresholds"]

    # 1) preparar imagen nueva
    prepared_new = prepare_single_image(img_new, robust=robust)

    # 2) procesar la imagen nueva con el mismo pipeline
    result_new = process_single_image(
        prepared_new,
        q=80,
        variant="closing_r3",
        prune_iters=5
    )

    # 3) extraer features necesarias
    feats_new = result_new["features"]

    thickness_new = float(feats_new["median_thickness"])
    segment_new = float(feats_new["median_segment_length"])

    # 4) calcular anchors promedio del experimento
    ctrl_anchor = compute_reference_anchor_from_images(ctrl_images, robust=robust)
    inflam_anchor = compute_reference_anchor_from_images(inflam_images, robust=robust)

    thickness_ctrl = ctrl_anchor["median_thickness"]
    segment_ctrl = ctrl_anchor["median_segment_length"]

    thickness_inflam = inflam_anchor["median_thickness"]
    segment_inflam = inflam_anchor["median_segment_length"]

    # 5) score relativo/anclado
    inflammatory_score = _compute_absolute_inflammatory_score(
        thickness=thickness_new,
        segment_length=segment_new,
        ctrl_thickness=thickness_ctrl,
        inflam_thickness=thickness_inflam,
        ctrl_segment=segment_ctrl,
        inflam_segment=segment_inflam,
        w_thickness=weights["median_thickness"],
        w_segment=weights["median_segment_length"]
    )

    recovery_score = 100.0 - inflammatory_score
    recovery_category = get_recovery_category(recovery_score, recovery_thresholds)

    note = (
        f"Score calculado usando {ctrl_anchor['n_images']} imágenes control y "
        f"{inflam_anchor['n_images']} imágenes inflamadas del mismo experimento."
    )

    return {
        "inflammatory_score": inflammatory_score,
        "recovery_score": recovery_score,
        "recovery_category": recovery_category,
        "score_features": {
            "median_thickness": thickness_new,
            "median_segment_length": segment_new
        },
        "reference_features": {
            "control": {
                "median_thickness": thickness_ctrl,
                "median_segment_length": segment_ctrl,
                "n_images": ctrl_anchor["n_images"]
            },
            "inflamed": {
                "median_thickness": thickness_inflam,
                "median_segment_length": segment_inflam,
                "n_images": inflam_anchor["n_images"]
            }
        },
        "all_features": feats_new,
        "wavelet": result_new["wavelet"],
        "mask": result_new["mask"],
        "skeleton": result_new["skeleton"],
        "mode": "anchored_relative_multi_reference",
        "note": note
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

import itertools
import pandas as pd

def temporal_pairwise_order_accuracy(df,
                                     score_col="absolute_inflammatory_score",
                                     time_col="time_h"):
    """
    Calcula el porcentaje de pares de imágenes correctamente ordenados
    según el tiempo esperado.

    Para cada par (i, j) con distinto time_h:
      si time_i < time_j, esperamos score_i < score_j
    """
    rows = df.reset_index(drop=True)
    total_pairs = 0
    correct_pairs = 0
    ties = 0

    for i, j in itertools.combinations(range(len(rows)), 2):
        t_i = rows.loc[i, time_col]
        t_j = rows.loc[j, time_col]
        s_i = rows.loc[i, score_col]
        s_j = rows.loc[j, score_col]

        if t_i == t_j:
            continue

        total_pairs += 1

        if t_i < t_j:
            if s_i < s_j:
                correct_pairs += 1
            elif s_i == s_j:
                ties += 1
        else:
            if s_j < s_i:
                correct_pairs += 1
            elif s_i == s_j:
                ties += 1

    accuracy = correct_pairs / total_pairs if total_pairs > 0 else float("nan")
    tie_rate = ties / total_pairs if total_pairs > 0 else float("nan")

    return {
        "total_pairs": total_pairs,
        "correct_pairs": correct_pairs,
        "ties": ties,
        "pairwise_order_accuracy": accuracy,
        "tie_rate": tie_rate
    }

def temporal_pairwise_order_by_timepair(df,
                                        score_col="absolute_inflammatory_score",
                                        time_label_col="time_label",
                                        time_col="time_h"):
    """
    Calcula accuracy de orden para cada par de tiempos.
    """
    time_pairs = [
        ("0HS", "24HS"),
        ("0HS", "48HS"),
        ("0HS", "72HS"),
        ("24HS", "48HS"),
        ("24HS", "72HS"),
        ("48HS", "72HS"),
    ]

    results = []

    for t1, t2 in time_pairs:
        df1 = df[df[time_label_col] == t1]
        df2 = df[df[time_label_col] == t2]

        total = 0
        correct = 0
        ties = 0

        for _, row1 in df1.iterrows():
            for _, row2 in df2.iterrows():
                total += 1
                if row1[score_col] < row2[score_col]:
                    correct += 1
                elif row1[score_col] == row2[score_col]:
                    ties += 1

        acc = correct / total if total > 0 else float("nan")
        tie_rate = ties / total if total > 0 else float("nan")

        results.append({
            "time_pair": f"{t1} vs {t2}",
            "total_pairs": total,
            "correct_pairs": correct,
            "ties": ties,
            "pairwise_order_accuracy": acc,
            "tie_rate": tie_rate
        })

    return pd.DataFrame(results)

# Chequeo de score relativo en 4 tiempos de un mismo grupo
def test_temporal_progression_anchored_on_four_images_multi_ref(temporal_dataset,
                                                                group="N1",
                                                                image_idx=0):
    """
    Usa como referencias:
    - todas las imágenes 0HS del grupo
    - todas las imágenes 72HS del grupo

    y calcula el score relativo para:
    0HS, 24HS, 48HS, 72HS
    """
    time_order = ["0HS", "24HS", "48HS", "72HS"]

    ctrl_images = [item["image"] for item in temporal_dataset["0HS"][group]]
    inflam_images = [item["image"] for item in temporal_dataset["72HS"][group]]

    rows = []

    for time_label in time_order:
        item = temporal_dataset[time_label][group][image_idx]

        result = predict_temporal_progression_score_anchored(
            img_new=item["image"],
            ctrl_images=ctrl_images,
            inflam_images=inflam_images
        )

        rows.append({
            "time_label": time_label,
            "group": group,
            "image_idx": image_idx,
            "image_name": item["name"],
            "inflammatory_score": result["inflammatory_score"],
            "recovery_score": result["recovery_score"],
            "recovery_category": result["recovery_category"]
        })

    df_test = pd.DataFrame(rows)
    print(df_test)

    inflammatory = df_test["inflammatory_score"].values
    monotonic = all(inflammatory[i] <= inflammatory[i + 1] for i in range(len(inflammatory) - 1))

    print("\n¿El inflammatory score sube progresivamente?")
    print(monotonic)

    return df_test

# Validacion score relativo
import pandas as pd

def validate_anchored_monotonicity_all(temporal_dataset):
    time_order = ["0HS", "24HS", "48HS", "72HS"]
    rows = []
    traj_ok = 0
    total_traj = 0

    for group in sorted(temporal_dataset["0HS"].keys()):
        n_imgs = len(temporal_dataset["0HS"][group])

        for image_idx in range(n_imgs):
            img_ctrl = temporal_dataset["0HS"][group][image_idx]["image"]
            img_inflam = temporal_dataset["72HS"][group][image_idx]["image"]

            scores = []
            row = {
                "group": group,
                "image_idx": image_idx,
            }

            for time_label in time_order:
                item = temporal_dataset[time_label][group][image_idx]

                result = predict_temporal_progression_score_anchored(
                    img_new=item["image"],
                    img_ctrl=img_ctrl,
                    img_inflam=img_inflam
                )

                s = result["inflammatory_score"]
                scores.append(s)
                row[f"{time_label}_score"] = s

            monotonic = all(scores[i] <= scores[i + 1] for i in range(len(scores) - 1))
            row["monotonic"] = monotonic

            rows.append(row)
            total_traj += 1
            traj_ok += int(monotonic)

    df_monotonic = pd.DataFrame(rows)
    summary = {
        "total_trajectories": total_traj,
        "monotonic_trajectories": traj_ok,
        "monotonicity_rate": traj_ok / total_traj if total_traj > 0 else float("nan")
    }

    return df_monotonic, summary

def build_anchored_relative_scores_dataframe(temporal_dataset):
    time_order = ["0HS", "24HS", "48HS", "72HS"]
    rows = []

    for group in sorted(temporal_dataset["0HS"].keys()):
        n_imgs = len(temporal_dataset["0HS"][group])

        for image_idx in range(n_imgs):
            img_ctrl = temporal_dataset["0HS"][group][image_idx]["image"]
            img_inflam = temporal_dataset["72HS"][group][image_idx]["image"]

            for time_label in time_order:
                item = temporal_dataset[time_label][group][image_idx]

                result = predict_temporal_progression_score_anchored(
                    img_new=item["image"],
                    img_ctrl=img_ctrl,
                    img_inflam=img_inflam
                )

                rows.append({
                    "group": group,
                    "image_idx": image_idx,
                    "time_label": time_label,
                    "time_h": item["time_h"],
                    "image_name": item["name"],
                    "relative_inflammatory_score": result["inflammatory_score"],
                    "relative_recovery_score": result["recovery_score"],
                    "recovery_category": result["recovery_category"]
                })

    return pd.DataFrame(rows)

import itertools

def temporal_pairwise_order_accuracy(df,
                                     score_col,
                                     time_col="time_h"):
    rows = df.reset_index(drop=True)
    total_pairs = 0
    correct_pairs = 0
    ties = 0

    for i, j in itertools.combinations(range(len(rows)), 2):
        t_i = rows.loc[i, time_col]
        t_j = rows.loc[j, time_col]
        s_i = rows.loc[i, score_col]
        s_j = rows.loc[j, score_col]

        if t_i == t_j:
            continue

        total_pairs += 1

        if t_i < t_j:
            if s_i < s_j:
                correct_pairs += 1
            elif s_i == s_j:
                ties += 1
        else:
            if s_j < s_i:
                correct_pairs += 1
            elif s_i == s_j:
                ties += 1

    return {
        "total_pairs": total_pairs,
        "correct_pairs": correct_pairs,
        "ties": ties,
        "pairwise_order_accuracy": correct_pairs / total_pairs if total_pairs > 0 else float("nan"),
        "tie_rate": ties / total_pairs if total_pairs > 0 else float("nan")
    }

def temporal_pairwise_order_by_timepair(df,
                                        score_col,
                                        time_label_col="time_label"):
    time_pairs = [
        ("0HS", "24HS"),
        ("0HS", "48HS"),
        ("0HS", "72HS"),
        ("24HS", "48HS"),
        ("24HS", "72HS"),
        ("48HS", "72HS"),
    ]

    results = []

    for t1, t2 in time_pairs:
        df1 = df[df[time_label_col] == t1]
        df2 = df[df[time_label_col] == t2]

        total = 0
        correct = 0
        ties = 0

        for _, row1 in df1.iterrows():
            for _, row2 in df2.iterrows():
                total += 1
                if row1[score_col] < row2[score_col]:
                    correct += 1
                elif row1[score_col] == row2[score_col]:
                    ties += 1

        results.append({
            "time_pair": f"{t1} vs {t2}",
            "total_pairs": total,
            "correct_pairs": correct,
            "ties": ties,
            "pairwise_order_accuracy": correct / total if total > 0 else float("nan"),
            "tie_rate": ties / total if total > 0 else float("nan")
        })

    return pd.DataFrame(results)

import numpy as np

def anchored_reference_sensitivity_one_image(temporal_dataset,
                                             group,
                                             target_time,
                                             target_idx):
    target_img = temporal_dataset[target_time][group][target_idx]["image"]

    rows = []

    n_ctrl = len(temporal_dataset["0HS"][group])
    n_inflam = len(temporal_dataset["72HS"][group])

    for ctrl_idx in range(n_ctrl):
        for inflam_idx in range(n_inflam):
            img_ctrl = temporal_dataset["0HS"][group][ctrl_idx]["image"]
            img_inflam = temporal_dataset["72HS"][group][inflam_idx]["image"]

            result = predict_temporal_progression_score_anchored(
                img_new=target_img,
                img_ctrl=img_ctrl,
                img_inflam=img_inflam
            )

            rows.append({
                "group": group,
                "target_time": target_time,
                "target_idx": target_idx,
                "ctrl_idx": ctrl_idx,
                "inflam_idx": inflam_idx,
                "inflammatory_score": result["inflammatory_score"],
                "recovery_score": result["recovery_score"],
                "recovery_category": result["recovery_category"]
            })

    df_sens = pd.DataFrame(rows)

    summary = {
        "group": group,
        "target_time": target_time,
        "target_idx": target_idx,
        "n_reference_combinations": len(df_sens),
        "inflam_mean": float(df_sens["inflammatory_score"].mean()),
        "inflam_std": float(df_sens["inflammatory_score"].std()),
        "inflam_min": float(df_sens["inflammatory_score"].min()),
        "inflam_max": float(df_sens["inflammatory_score"].max()),
        "recovery_mean": float(df_sens["recovery_score"].mean()),
        "recovery_std": float(df_sens["recovery_score"].std()),
        "recovery_min": float(df_sens["recovery_score"].min()),
        "recovery_max": float(df_sens["recovery_score"].max()),
    }

    return df_sens, summary

def anchored_reference_sensitivity_all(temporal_dataset,
                                       target_times=("24HS", "48HS")):
    rows = []

    for group in sorted(temporal_dataset["0HS"].keys()):
        n_imgs = len(temporal_dataset["0HS"][group])

        for target_time in target_times:
            for target_idx in range(n_imgs):
                _, summary = anchored_reference_sensitivity_one_image(
                    temporal_dataset,
                    group=group,
                    target_time=target_time,
                    target_idx=target_idx
                )
                rows.append(summary)

    return pd.DataFrame(rows)

import matplotlib.pyplot as plt
import seaborn as sns

def plot_relative_inflammatory_score(df):
    plt.figure(figsize=(8, 6))

    sns.boxplot(
        data=df,
        x="time_label",
        y="relative_inflammatory_score",
        order=["0HS", "24HS", "48HS", "72HS"],
        color="#B7D3E9",
        fliersize=0,
        linewidth=1.4
    )

    sns.stripplot(
        data=df,
        x="time_label",
        y="relative_inflammatory_score",
        order=["0HS", "24HS", "48HS", "72HS"],
        color="black",
        alpha=0.7,
        size=4,
        jitter=0.12
    )

    plt.title("Anchored relative inflammatory score", fontsize=14, fontweight="bold")
    plt.xlabel("Time after treatment")
    plt.ylabel("Inflammatory score")
    plt.grid(True, axis="y", alpha=0.25)
    sns.despine()
    plt.tight_layout()
    plt.show()

def plot_relative_score_by_group(df):
    df_group_rel = (
        df.groupby(["group", "time_label", "time_h"], as_index=False)
        .agg({"relative_inflammatory_score": "mean"})
        .sort_values(["group", "time_h"])
    )

    plt.figure(figsize=(8, 6))

    sns.lineplot(
        data=df_group_rel,
        x="time_h",
        y="relative_inflammatory_score",
        hue="group",
        marker="o",
        linewidth=2
    )

    plt.title("Anchored relative inflammatory score progression by group",
              fontsize=14, fontweight="bold")
    plt.xlabel("Time (h)")
    plt.ylabel("Mean inflammatory score")
    plt.grid(True, alpha=0.25)
    sns.despine()
    plt.tight_layout()
    plt.show()

    return df_group_rel

#multireference

def validate_anchored_monotonicity_all_multi_ref(temporal_dataset):
    time_order = ["0HS", "24HS", "48HS", "72HS"]
    rows = []
    traj_ok = 0
    total_traj = 0

    for group in sorted(temporal_dataset["0HS"].keys()):
        ctrl_images = [item["image"] for item in temporal_dataset["0HS"][group]]
        inflam_images = [item["image"] for item in temporal_dataset["72HS"][group]]

        n_imgs = len(temporal_dataset["0HS"][group])

        for image_idx in range(n_imgs):
            scores = []
            row = {
                "group": group,
                "image_idx": image_idx,
            }

            for time_label in time_order:
                item = temporal_dataset[time_label][group][image_idx]

                result = predict_temporal_progression_score_anchored(
                    img_new=item["image"],
                    ctrl_images=ctrl_images,
                    inflam_images=inflam_images
                )

                s = result["inflammatory_score"]
                scores.append(s)
                row[f"{time_label}_score"] = s

            monotonic = all(scores[i] <= scores[i + 1] for i in range(len(scores) - 1))
            row["monotonic"] = monotonic

            rows.append(row)
            total_traj += 1
            traj_ok += int(monotonic)

    df_monotonic = pd.DataFrame(rows)
    summary = {
        "total_trajectories": total_traj,
        "monotonic_trajectories": traj_ok,
        "monotonicity_rate": traj_ok / total_traj if total_traj > 0 else float("nan")
    }

    return df_monotonic, summary

def build_anchored_relative_scores_dataframe_multi_ref(temporal_dataset):
    time_order = ["0HS", "24HS", "48HS", "72HS"]
    rows = []

    for group in sorted(temporal_dataset["0HS"].keys()):
        ctrl_images = [item["image"] for item in temporal_dataset["0HS"][group]]
        inflam_images = [item["image"] for item in temporal_dataset["72HS"][group]]

        n_imgs = len(temporal_dataset["0HS"][group])

        for image_idx in range(n_imgs):
            for time_label in time_order:
                item = temporal_dataset[time_label][group][image_idx]

                result = predict_temporal_progression_score_anchored(
                    img_new=item["image"],
                    ctrl_images=ctrl_images,
                    inflam_images=inflam_images
                )

                rows.append({
                    "group": group,
                    "image_idx": image_idx,
                    "time_label": time_label,
                    "time_h": item["time_h"],
                    "image_name": item["name"],
                    "relative_inflammatory_score": result["inflammatory_score"],
                    "relative_recovery_score": result["recovery_score"],
                    "recovery_category": result["recovery_category"]
                })

    return pd.DataFrame(rows)

def summarize_relative_score_by_group(df,
                                      score_col="relative_inflammatory_score"):
    """
    Resume el score relativo por grupo y tiempo.
    """
    summary = (
        df.groupby(["group", "time_label", "time_h"], as_index=False)[score_col]
        .agg(["mean", "std", "median", "min", "max"])
        .reset_index()
        .sort_values(["group", "time_h"])
    )
    return summary

def validate_groupwise_monotonicity(summary_group_df,
                                    mean_col="mean"):
    """
    Evalúa monotonicidad de las medias del score por grupo.
    """
    rows = []

    for group in sorted(summary_group_df["group"].unique()):
        sub = (
            summary_group_df[summary_group_df["group"] == group]
            .sort_values("time_h")
        )

        means = sub[mean_col].values
        monotonic = all(means[i] <= means[i + 1] for i in range(len(means) - 1))

        rows.append({
            "group": group,
            "0HS_mean": means[0],
            "24HS_mean": means[1],
            "48HS_mean": means[2],
            "72HS_mean": means[3],
            "monotonic": monotonic
        })

    df_monotonic_group = pd.DataFrame(rows)

    summary = {
        "total_groups": len(df_monotonic_group),
        "monotonic_groups": int(df_monotonic_group["monotonic"].sum()),
        "group_monotonicity_rate": float(df_monotonic_group["monotonic"].mean())
    }

    return df_monotonic_group, summary

import matplotlib.pyplot as plt
import seaborn as sns

def plot_groupwise_relative_summary(summary_group_df):
    plt.figure(figsize=(8, 6))

    for group in sorted(summary_group_df["group"].unique()):
        sub = (
            summary_group_df[summary_group_df["group"] == group]
            .sort_values("time_h")
        )

        plt.errorbar(
            sub["time_h"],
            sub["mean"],
            yerr=sub["std"],
            marker="o",
            linewidth=2,
            capsize=4,
            label=group
        )

    plt.title("Group-wise temporal progression of anchored relative score",
              fontsize=14, fontweight="bold")
    plt.xlabel("Time (h)")
    plt.ylabel("Mean relative inflammatory score")
    plt.grid(True, alpha=0.25)
    plt.legend(title="Group")
    sns.despine()
    plt.tight_layout()
    plt.show()

# analisis complementario
def build_temporal_score_metadata_leave_one_group_out(df, held_out_group):
    """
    Construye metadata temporal dejando un grupo afuera.
    Recalcula la calibración group-level usando solo los otros grupos.
    """
    df_train = df[df["group"] != held_out_group].copy()

    metadata = build_temporal_score_metadata(df_train)
    metadata["held_out_group"] = held_out_group

    return metadata

def predict_temporal_progression_score_anchored_with_metadata(img_new,
                                                              ctrl_images,
                                                              inflam_images,
                                                              metadata,
                                                              robust=True):
    """
    Igual que el score relativo multi-ref final, pero usando una metadata
    pasada explícitamente (por ejemplo, leave-one-group-out).
    """
    weights = metadata["weights"]
    recovery_thresholds = metadata["recovery_thresholds"]

    prepared_new = prepare_single_image(img_new, robust=robust)

    result_new = process_single_image(
        prepared_new,
        q=80,
        variant="closing_r3",
        prune_iters=5
    )

    feats_new = result_new["features"]

    thickness_new = float(feats_new["median_thickness"])
    segment_new = float(feats_new["median_segment_length"])

    ctrl_anchor = compute_reference_anchor_from_images(
        ctrl_images,
        robust=robust
    )
    inflam_anchor = compute_reference_anchor_from_images(
        inflam_images,
        robust=robust
    )

    thickness_ctrl = ctrl_anchor["median_thickness"]
    segment_ctrl = ctrl_anchor["median_segment_length"]

    thickness_inflam = inflam_anchor["median_thickness"]
    segment_inflam = inflam_anchor["median_segment_length"]

    inflammatory_score = _compute_absolute_inflammatory_score(
        thickness=thickness_new,
        segment_length=segment_new,
        ctrl_thickness=thickness_ctrl,
        inflam_thickness=thickness_inflam,
        ctrl_segment=segment_ctrl,
        inflam_segment=segment_inflam,
        w_thickness=weights["median_thickness"],
        w_segment=weights["median_segment_length"]
    )

    recovery_score = 100.0 - inflammatory_score
    recovery_category = get_recovery_category(
        recovery_score,
        recovery_thresholds
    )

    return {
        "inflammatory_score": inflammatory_score,
        "recovery_score": recovery_score,
        "recovery_category": recovery_category,
        "score_features": {
            "median_thickness": thickness_new,
            "median_segment_length": segment_new
        },
        "reference_features": {
            "control": {
                "median_thickness": thickness_ctrl,
                "median_segment_length": segment_ctrl,
                "n_images": ctrl_anchor["n_images"]
            },
            "inflamed": {
                "median_thickness": thickness_inflam,
                "median_segment_length": segment_inflam,
                "n_images": inflam_anchor["n_images"]
            }
        },
        "all_features": feats_new,
        "wavelet": result_new["wavelet"],
        "mask": result_new["mask"],
        "skeleton": result_new["skeleton"],
        "mode": "anchored_relative_multi_reference_LOO"
    }

def build_anchored_relative_scores_dataframe_multi_ref_loo(temporal_dataset, df_temporal):
    """
    Evalúa el score relativo multi-ref en esquema leave-one-group-out.

    Para cada grupo:
    - recalcula metadata usando los otros grupos
    - evalúa el grupo dejado afuera usando sus propias refs control/inflamadas
    """
    time_order = ["0HS", "24HS", "48HS", "72HS"]
    rows = []
    metadata_by_group = {}

    all_groups = sorted(temporal_dataset["0HS"].keys())

    for held_out_group in all_groups:
        metadata = build_temporal_score_metadata_leave_one_group_out(
            df_temporal,
            held_out_group
        )
        metadata_by_group[held_out_group] = metadata

        ctrl_images = [item["image"] for item in temporal_dataset["0HS"][held_out_group]]
        inflam_images = [item["image"] for item in temporal_dataset["72HS"][held_out_group]]

        n_imgs = len(temporal_dataset["0HS"][held_out_group])

        for image_idx in range(n_imgs):
            for time_label in time_order:
                item = temporal_dataset[time_label][held_out_group][image_idx]

                result = predict_temporal_progression_score_anchored_with_metadata(
                    img_new=item["image"],
                    ctrl_images=ctrl_images,
                    inflam_images=inflam_images,
                    metadata=metadata
                )

                rows.append({
                    "held_out_group": held_out_group,
                    "group": held_out_group,
                    "image_idx": image_idx,
                    "time_label": time_label,
                    "time_h": item["time_h"],
                    "image_name": item["name"],
                    "relative_inflammatory_score": result["inflammatory_score"],
                    "relative_recovery_score": result["recovery_score"],
                    "recovery_category": result["recovery_category"]
                })

    return pd.DataFrame(rows), metadata_by_group

def validate_anchored_monotonicity_all_multi_ref_loo(df_scores_loo):
    """
    Evalúa monotonicidad de trayectorias usando el dataframe LOO ya calculado.
    """
    rows = []
    traj_ok = 0
    total_traj = 0

    for group in sorted(df_scores_loo["group"].unique()):
        sub_group = df_scores_loo[df_scores_loo["group"] == group]

        for image_idx in sorted(sub_group["image_idx"].unique()):
            sub = (
                sub_group[sub_group["image_idx"] == image_idx]
                .sort_values("time_h")
            )

            scores = sub["relative_inflammatory_score"].values
            monotonic = all(scores[i] <= scores[i + 1] for i in range(len(scores) - 1))

            rows.append({
                "group": group,
                "image_idx": image_idx,
                "0HS_score": scores[0],
                "24HS_score": scores[1],
                "48HS_score": scores[2],
                "72HS_score": scores[3],
                "monotonic": monotonic
            })

            total_traj += 1
            traj_ok += int(monotonic)

    df_monotonic = pd.DataFrame(rows)
    summary = {
        "total_trajectories": total_traj,
        "monotonic_trajectories": traj_ok,
        "monotonicity_rate": traj_ok / total_traj if total_traj > 0 else float("nan")
    }

    return df_monotonic, summary