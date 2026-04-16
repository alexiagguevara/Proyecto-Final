from multiprocessing import process
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.preprocessing import normalize_dataset, apply_filters, build_wavelet_mask_candidate, get_skeleton, prune_skeleton
from pipeline import extract_all_features

# Features finales elegidas por RFECV
FINAL_FEATURES = [
    "median_thickness",
    "median_tortuosity",
    "median_segment_length"
]


def prepare_single_image(img, robust=True):
    """
    Prepara una imagen individual para el pipeline:
    - si viene RGB, extrae el canal verde
    - la normaliza a uint8 [0, 255] reutilizando normalize_dataset()
    """
    # 1) Extraer canal verde si la imagen viene RGB
    if img.ndim == 3:
        green_img = img[:, :, 1]
    else:
        green_img = img

    # 2) Reutilizar normalize_dataset sobre un mini-dataset
    temp_dataset = {
        "tmp_group": {
            "tmp_cond": [{"name": "tmp_img", "image": green_img}]
        }
    }

    norm_dataset = normalize_dataset(temp_dataset, robust=robust)
    norm_img = norm_dataset["tmp_group"]["tmp_cond"][0]["image"]

    return norm_img


def process_single_image(img,
                         q=80,
                         variant="closing_r3",
                         prune_iters=5):
    """
    Procesa una imagen completa y devuelve:
    - wavelet
    - mask
    - skeleton
    - features
    """

    # 1. Wavelet
    _, _, wavelet_img = apply_filters(img)

    # 2. Máscara final
    mask = build_wavelet_mask_candidate(
        wavelet_img,
        q=q,
        variant=variant
    )

    # 3. Skeleton
    skel = get_skeleton(mask)

    # 4. Pruning
    skel = prune_skeleton(skel, prune_iters=prune_iters)

    # 5. Features
    feats = extract_all_features(mask, skel)

    return {
        "wavelet": wavelet_img,
        "mask": mask,
        "skeleton": skel,
        "features": feats
    }


def build_feature_dataset(dataset,
                          q=80,
                          variant="closing_r3",
                          prune_iters=5):
    """
    Recorre todo el dataset y arma un DataFrame de features.

    dataset esperado:
    dataset[folder_group][condition] = [
        {"name": ..., "image": ..., "true_group": ...},
        ...
    ]
    """

    rows = []

    for group, conds in dataset.items():
        for condition, imgs in conds.items():
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
                feats["folder_group"] = group
                feats["group"] = item["true_group"]
                feats["condition"] = condition
                feats["image_name"] = img_name

                rows.append(feats)

    df = pd.DataFrame(rows)
    return df

def train_final_binary_model(df,
                             condition_col="condition",
                             ctrl_label="CTRL",
                             hpmc_label="HPMC"):
    """
    Entrena el modelo final binario CTRL vs 72hs_LPS usando las 3 features finales.
    """
    mask = df[condition_col].isin([ctrl_label, hpmc_label])

    X = df.loc[mask, FINAL_FEATURES].copy()
    y = (df.loc[mask, condition_col] == hpmc_label).astype(int)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ])

    model.fit(X, y)

    metadata = {
        "features": FINAL_FEATURES,
        "target_mapping": {
            0: ctrl_label,
            1: hpmc_label
        },
        "model_type": "LogisticRegression + StandardScaler",
        "task": "CTRL vs 72hs_LPS",
        "q": 80,
        "variant": "closing_r3",
        "prune_iters": 5
    }

    return model, metadata


def save_final_binary_model(df,
                            model_path="final_binary_model.joblib",
                            metadata_path="final_binary_model_metadata.joblib"):
    """
    Entrena y guarda el modelo final binario.
    """
    model, metadata = train_final_binary_model(df)

    joblib.dump(model, model_path)
    joblib.dump(metadata, metadata_path)

    print(f"Modelo guardado en: {model_path}")
    print(f"Metadata guardada en: {metadata_path}")

    return model, metadata


def load_final_binary_model(model_path="final_binary_model.joblib",
                            metadata_path="final_binary_model_metadata.joblib"):
    """
    Carga modelo y metadata.
    """
    model = joblib.load(model_path)
    metadata = joblib.load(metadata_path)
    return model, metadata


def predict_inflammatory_state(img,
                               model_path="final_binary_model.joblib",
                               metadata_path="final_binary_model_metadata.joblib",
                               q=80,
                               variant="closing_r3",
                               prune_iters=5):
    """
    Predice CTRL vs 72hs_LPS a partir de una imagen nueva.
    
    Devuelve:
    - predicción de clase
    - probabilidades
    - features usadas por el modelo
    - todas las features extraídas
    - outputs intermedios (mask, skeleton, etc.)
    """
    model, metadata = load_final_binary_model(model_path, metadata_path)

    # 1) prepare image
    prepared_img = prepare_single_image(img, robust=True)

    # 2) process complete image
    result = process_single_image(
        prepared_img,
        q=metadata["q"],
        variant=metadata["variant"],
        prune_iters=metadata["prune_iters"]
    )

    all_features = result["features"]

    # 3) final features of model
    model_features = {k: all_features[k] for k in metadata["features"]}

    X_new = pd.DataFrame([model_features])[metadata["features"]]

    pred_num = model.predict(X_new)[0]
    proba = model.predict_proba(X_new)[0]

    result = {
        "pred_label": metadata["target_mapping"][pred_num],
        "pred_numeric": int(pred_num),
        "prob_CTRL": float(proba[0]),
        "prob_72hs_LPS": float(proba[1]),
        "model_features": model_features,
        "all_features": all_features,
        "wavelet": result["wavelet"],
        "mask": result["mask"],
        "skeleton": result["skeleton"]
    }

    return result

#USE OF MODEL
"""
import joblib
import pandas as pd

model = joblib.load("final_binary_model.joblib")
metadata = joblib.load("final_binary_model_metadata.joblib")
features = metadata["features"]
"""

#Recordar que antes de pasarlo al modelo, en el software la imagen que se sube debe pasar por:
#canal verde, normalización, wavelet, máscara final, skeleton, pruning, extracción de las 3 features
