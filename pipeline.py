
from src.preprocessing import apply_filters, build_wavelet_mask_candidate, get_skeleton, prune_skeleton
from src.features import extract_all_features

def process_single_image(img,
                         q=80,
                         variant="closing_r3",
                         prune_iters=5):
    """
    Procesa una imagen completa y devuelve:
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

import pandas as pd

def build_feature_dataset(dataset,
                          q=80,
                          variant="closing_r3",
                          prune_iters=5):
    """
    Recorre todo el dataset y arma un DataFrame de features.
    
    dataset esperado:
    dataset[group][condition] = [
        {"name": ..., "image": ...},
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
                feats["group"] = group
                feats["condition"] = condition
                feats["image_name"] = img_name

                rows.append(feats)

    df = pd.DataFrame(rows)
    return df

import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_boxplots(df, feature_list):
    n = len(feature_list)
    fig, axes = plt.subplots(n, 1, figsize=(8, 4*n))

    if n == 1:
        axes = [axes]

    for ax, feat in zip(axes, feature_list):
        sns.boxplot(data=df, x="condition", y=feat, ax=ax)
        sns.stripplot(data=df, x="condition", y=feat, ax=ax, color="black", alpha=0.6)
        ax.set_title(feat)

    plt.tight_layout()
    plt.show()