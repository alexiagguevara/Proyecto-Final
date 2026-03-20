
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


import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

def plot_final_feature_boxplots(df, save_path="final_3_feature_boxplots.png"):
    """
    Boxplots finales de las 3 features seleccionadas,
    con p-values ajustados por FDR (Benjamini-Hochberg).
    """

    final_features = [
        ("median_thickness", "Median thickness"),
        ("median_tortuosity", "Median tortuosity"),
        ("median_segment_length", "Median segment length"),
    ]

    # calcular p-values crudos
    raw_pvals = []
    for feat, _ in final_features:
        ctrl_vals = df[df["condition"] == "CTRL"][feat].dropna().values
        hpmc_vals = df[df["condition"] == "HPMC"][feat].dropna().values
        _, pval = mannwhitneyu(ctrl_vals, hpmc_vals, alternative="two-sided")
        raw_pvals.append(pval)

    # corrección FDR
    _, pvals_corr, _, _ = multipletests(raw_pvals, method="fdr_bh")

    fig, axes = plt.subplots(3, 1, figsize=(8, 11), sharex=True)

    palette = {
        "CTRL": "#8FB3D9",
        "HPMC": "#E6A57E"
    }

    for ax, (feat, label), pval_corr in zip(axes, final_features, pvals_corr):
        sns.boxplot(
            data=df,
            x="condition",
            y=feat,
            ax=ax,
            palette=palette,
            width=0.5,
            fliersize=0,
            linewidth=1.4
        )

        sns.stripplot(
            data=df,
            x="condition",
            y=feat,
            ax=ax,
            color="black",
            alpha=0.7,
            size=4,
            jitter=0.12
        )

        if pval_corr < 0.001:
            p_text = "FDR-adjusted p < 0.001"
        else:
            p_text = f"FDR-adjusted p = {pval_corr:.3f}"

        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.text(
            0.5, 0.95, p_text,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=11
        )

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(True, axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Condition", fontsize=11)

    fig.suptitle(
        "Final selected morphological features",
        fontsize=15,
        fontweight="bold",
        y=0.995
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Saved → {save_path}")