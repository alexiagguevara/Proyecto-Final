from multiprocessing import process
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.preprocessing import normalize_dataset, apply_filters, build_wavelet_mask_candidate, get_skeleton, prune_skeleton
from pipeline import extract_all_features

from src.features import rank_features, print_report
from feature_selection_pipeline import (
    get_strong_candidates,
    prune_correlated_features,
    CONDITION_COL,
    CTRL_LABEL,
    HPMC_LABEL,
    CORR_THRESHOLD,
)


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

def select_final_features_from_full_dataset(df,
                                            condition_col=CONDITION_COL,
                                            ctrl_label=CTRL_LABEL,
                                            hpmc_label=HPMC_LABEL,
                                            corr_threshold=CORR_THRESHOLD):
    """
    Selecciona el feature set final sobre TODO el dataset original usando:
    rank_features -> HIGH-tier -> correlation pruning
    """

    # 1) ranking sobre full dataset
    results_df = rank_features(
        df,
        condition_col=condition_col,
        ctrl_label=ctrl_label,
        hpmc_label=hpmc_label
    )
    print_report(results_df, top_n=15)
    print()

    # 2) HIGH-tier
    strong = get_strong_candidates(results_df, df=df)
    if len(strong) == 0:
        raise ValueError("No HIGH-tier features found on full dataset.")

    # 3) correlation pruning
    pruned, corr_pairs = prune_correlated_features(
        df,
        strong,
        results_df,
        threshold=corr_threshold
    )
    if len(pruned) == 0:
        raise ValueError("Correlation pruning removed all features on full dataset.")

    return pruned, results_df, corr_pairs

def train_final_binary_model(df,
                             condition_col=CONDITION_COL,
                             ctrl_label=CTRL_LABEL,
                             hpmc_label=HPMC_LABEL,
                             selected_features=None):
    """
    Entrena el modelo final binario usando:
    - selected_features, si se pasan explícitamente
    - o la nueva estrategia final:
      rank_features -> HIGH-tier -> correlation pruning
    """

    # si no se pasan features, usar la nueva selección final
    if selected_features is None:
        selected_features, results_df, corr_pairs = select_final_features_from_full_dataset(
            df,
            condition_col=condition_col,
            ctrl_label=ctrl_label,
            hpmc_label=hpmc_label
        )
    else:
        results_df = None
        corr_pairs = None

    mask = df[condition_col].isin([ctrl_label, hpmc_label])

    X = df.loc[mask, selected_features].copy()
    y = (df.loc[mask, condition_col] == hpmc_label).astype(int)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE
        ))
    ])

    model.fit(X, y)

    metadata = {
        "features": selected_features,
        "target_mapping": {
            0: ctrl_label,
            1: hpmc_label
        },
        "model_type": "LogisticRegression + StandardScaler",
        "task": "CTRL vs Pro-inflammatory",
        "selection_pipeline": "rank_features -> HIGH-tier -> correlation pruning",
        "corr_threshold": CORR_THRESHOLD,
        "q": 80,
        "variant": "closing_r3",
        "prune_iters": 5
    }

    return model, metadata, results_df, corr_pairs


def save_final_binary_model(df,
                            model_path="final_binary_model.joblib",
                            metadata_path="final_binary_model_metadata.joblib",
                            selected_features=None):
    """
    Entrena y guarda el modelo final binario con la nueva estrategia.
    Mantiene los mismos nombres de archivo para no cambiar la app.
    """
    model, metadata, results_df, corr_pairs = train_final_binary_model(
        df,
        selected_features=selected_features
    )

    joblib.dump(model, model_path)
    joblib.dump(metadata, metadata_path)

    print(f"Modelo guardado en: {model_path}")
    print(f"Metadata guardada en: {metadata_path}")
    print(f"Features finales ({len(metadata['features'])}): {metadata['features']}")

    return model, metadata, results_df, corr_pairs


from app_ui.paths import resource_path



def load_final_binary_model(model_path=None, metadata_path=None):
    """
    Carga modelo y metadata.
    """
    if model_path is None:
        model_path = resource_path("final_binary_model.joblib")
    if metadata_path is None:
        metadata_path = resource_path("final_binary_model_metadata.joblib")

    model = joblib.load(model_path)
    metadata = joblib.load(metadata_path)
    return model, metadata

def predict_inflammatory_state(img,
                               model_path=None,
                               metadata_path=None,
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


#Recordar que antes de pasarlo al modelo, en el software la imagen que se sube debe pasar por:
#canal verde, normalización, wavelet, máscara final, skeleton, pruning, extracción de las 3 features

#OUTER LOOP 

from sklearn.ensemble import RandomForestClassifier

CONDITION_COL = "condition"
CTRL_LABEL = "CTRL"
HPMC_LABEL = "HPMC"
DEFAULT_N_SPLITS = 4                 # 4 cultivos → leave-one-culture-out style
RANDOM_STATE = 42

def train_selected_model(df_train,
                         selected_features,
                         model_name,
                         condition_col=CONDITION_COL,
                         ctrl_label=CTRL_LABEL,
                         hpmc_label=HPMC_LABEL):
    """
    Entrena el modelo elegido sobre TODO el outer-train.
    """
    mask = df_train[condition_col].isin([ctrl_label, hpmc_label])

    X = df_train.loc[mask, selected_features].copy()
    y = (df_train.loc[mask, condition_col] == hpmc_label).astype(int)

    if model_name == "LogReg (L2)":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE
            ))
        ])
    elif model_name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    model.fit(X, y)
    return model