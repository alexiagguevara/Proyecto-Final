import numpy as np

def extract_green_channel(dataset):
    """
    Toma un dataset original (RGB) y devuelve un nuevo diccionario con el canal verde de cada imagen.
    """
    green_dataset = {}
    for grupo, condiciones in dataset.items():
        green_dataset[grupo] = {}
        for cond, imgs in condiciones.items():
            green_dataset[grupo][cond] = []
            for img_dict in imgs:
                # Extraemos el canal verde y guardamos junto al nombre
                green_img = img_dict["image"][:, :, 1]
                green_dataset[grupo][cond].append({"name": img_dict["name"], "image": green_img})
    return green_dataset

def check_image_ranges(dataset):
    """
    Imprime el rango (min, max) de cada imagen en el dataset.
    dataset: diccionario con estructura dataset[grupo][cond] = lista de dicts {"name", "image"}
    """
    for grupo, condiciones in dataset.items():
        for cond, imgs in condiciones.items():
            print(f"\n{grupo} - {cond}:")
            for img_dict in imgs:
                img = img_dict["image"]
                name = img_dict["name"]
                print(f"  {name}: min={img.min()}, max={img.max()}")

def normalize_dataset(dataset, robust=True):
    """
    Normaliza cada imagen del dataset a rango [0, 255].

    Parámetros
    ----------
    dataset : dict
        Diccionario con estructura dataset[grupo][cond] = [{'name': str, 'image': np.ndarray}, ...]
    robust : bool, opcional
        Si True, usa normalización robusta por percentiles (p1–p99).
        Si False, usa min–max tradicional.

    Retorna
    -------
    dict : dataset normalizado (uint8)
    """
    norm_dataset = {}

    for group, conds in dataset.items():
        norm_dataset[group] = {}
        for cond, imgs in conds.items():
            norm_dataset[group][cond] = []
            for entry in imgs:
                img = entry["image"].astype(np.float32)
                
                if robust:
                    # Normalización robusta: ignora extremos del 1% y 99%
                    p1, p99 = np.percentile(img, (1, 99))
                    if p99 > p1:
                        img = np.clip((img - p1) / (p99 - p1), 0, 1)
                    else:
                        # fallback por si hay imagen casi uniforme
                        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                else:
                    # Normalización min–max clásica
                    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

                norm = (img * 255).astype(np.uint8)

                norm_dataset[group][cond].append({
                    "name": entry["name"],
                    "image": norm
                })
    return norm_dataset


import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_nl_means, estimate_sigma, denoise_wavelet
from skimage.measure import shannon_entropy

def apply_filters(image):
    """Aplica los 3 filtros de ruido a una imagen en escala de grises"""
    # Filtro bilateral
    # Parámetros para el filtro bilateral
    d = 9  # Diámetro del vecindario de cada píxel
    sigma_color = 75  # Sigma en el espacio de color
    sigma_space = 75  # Sigma en el espacio de coordenadas, si reduzco este se suaviza mas
    
    bilateral = cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    # Non-Local Means
    sigma_est = np.mean(estimate_sigma(image, channel_axis=None))
    nlm = denoise_nl_means(image, h=1.15 * sigma_est, fast_mode=True, patch_size=5, patch_distance=6, channel_axis=None)
    nlm = (nlm * 255).astype(np.uint8)  # volver a uint8

    # Wavelet denoising
    wavelet = denoise_wavelet(image, method='BayesShrink', mode='soft', wavelet='sym4', wavelet_levels=2, rescale_sigma=True, channel_axis=None)
    wavelet = (wavelet * 255).astype(np.uint8) # volver a uint8

    return bilateral, nlm, wavelet

def tenengrad_sharpness(img):
    """Calcula la nitidez basada en el gradiente de Sobel (Tenengrad)."""
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    g = np.hypot(gx, gy)
    return float(np.var(g))

def residual_signal_ratio(original, denoised):
    """Mide cuánto cambió la imagen (energía del residuo / energía de la señal)."""
    resid = original.astype(np.float32) - denoised.astype(np.float32)
    e_res = np.mean(resid**2)
    e_sig = np.mean(original.astype(np.float32)**2) + 1e-12
    return float(e_res / e_sig)

def compute_metrics(image, original=None):
    """Calcula rango, media, std, entropía, Tenengrad y opcionalmente RSR."""
    image = image.astype(np.float32)
    min_val, max_val = image.min(), image.max()
    rango = max_val - min_val
    media = image.mean()
    std = image.std()
    entropia = shannon_entropy(image)
    ten = tenengrad_sharpness(image)
    rsr = np.nan
    if original is not None:
        rsr = residual_signal_ratio(original, image)
    return rango, media, std, entropia, ten, rsr

def show_filters_with_metrics(ctrl_img, hpmc_img):
    """Muestra CTRL y HPMC con los 3 filtros + métricas"""
    filtros = ["Original", "Bilateral", "NLM", "Wavelet"]

    # Aplicar filtros
    ctrl_bilateral, ctrl_nlm, ctrl_wavelet = apply_filters(ctrl_img)
    hpmc_bilateral, hpmc_nlm, hpmc_wavelet = apply_filters(hpmc_img)

    ctrl_images = [ctrl_img, ctrl_bilateral, ctrl_nlm, ctrl_wavelet]
    hpmc_images = [hpmc_img, hpmc_bilateral, hpmc_nlm, hpmc_wavelet]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for i in range(4):
        # CTRL
        axes[0, i].imshow(ctrl_images[i], cmap='gray', vmin=0, vmax=255)
        axes[0, i].axis('off')
        rango, media, std, entropia, ten, rsr = compute_metrics(
            ctrl_images[i],
            original=ctrl_img if filtros[i] != "Original" else None)
        axes[0, i].set_title(f"{filtros[i]}\nR:{rango:.0f} M:{media:.1f}\nSTD:{std:.1f} Ent:{entropia:.2f}\nTng:{ten:.1f}\nRSR:{rsr:.3f}")

        # HPMC
        axes[1, i].imshow(hpmc_images[i], cmap='gray', vmin=0, vmax=255)
        axes[1, i].axis('off')
        rango, media, std, entropia,  ten, rsr  = compute_metrics(
            hpmc_images[i],
            original=hpmc_img if filtros[i] != "Original" else None)
        axes[1, i].set_title(f"{filtros[i]}\nR:{rango:.0f} M:{media:.1f}\nSTD:{std:.1f} Ent:{entropia:.2f}\nTng:{ten:.1f}\nRSR:{rsr:.3f}")

    axes[0, 0].set_ylabel("CTRL", fontsize=14)
    axes[1, 0].set_ylabel("HPMC", fontsize=14)
    plt.tight_layout()
    plt.show()

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8,8)):
    """
    Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization)
    para mejorar contraste local sin amplificar ruido.
    """
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(image)
    return enhanced

import matplotlib.pyplot as plt

def compare_clahe_mosaic(ctrl_wavelet, hpmc_wavelet, clip_values=[1.0, 2.0, 4.0]):
    """
    Muestra un mosaico comparativo entre Wavelet y Wavelet+CLAHE
    para distintas intensidades de contraste (clipLimit).
    """
    # Aplicar CLAHE a cada clipLimit
    ctrl_clahe_list = [apply_clahe(ctrl_wavelet, clip_limit=c) for c in clip_values]
    hpmc_clahe_list = [apply_clahe(hpmc_wavelet, clip_limit=c) for c in clip_values]

    # Crear mosaico 2x(N+1)
    n_cols = len(clip_values) + 1
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8))

    # Fila 1: CTRL
    axes[0, 0].imshow(ctrl_wavelet, cmap='gray')
    axes[0, 0].set_title("CTRL - Wavelet (filtrada)")
    for i, c in enumerate(clip_values):
        axes[0, i + 1].imshow(ctrl_clahe_list[i], cmap='gray')
        axes[0, i + 1].set_title(f"CTRL + CLAHE\nclip={c}")

    # Fila 2: HPMC
    axes[1, 0].imshow(hpmc_wavelet, cmap='gray')
    axes[1, 0].set_title("HPMC - Wavelet (filtrada)")
    for i, c in enumerate(clip_values):
        axes[1, i + 1].imshow(hpmc_clahe_list[i], cmap='gray')
        axes[1, i + 1].set_title(f"HPMC + CLAHE\nclip={c}")

    # Formato
    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def evaluate_clahe(ctrl_wavelet, hpmc_wavelet, clip_values=[1.0, 2.0, 4.0]):
    """
    Calcula Entropía, Tenengrad y RSR para distintas intensidades de CLAHE.
    Imprime los resultados para que el usuario elija manualmente el mejor.
    """
    print("Evaluando efecto de CLAHE sobre Wavelet filtrado...\n")
    print(f"{'clipLimit':<10} {'Entropía':>10} {'Tenengrad':>12} {'RSR':>10}")
    print("-" * 45)

    for c in clip_values:
        ctrl_c = apply_clahe(ctrl_wavelet, clip_limit=c)
        hpmc_c = apply_clahe(hpmc_wavelet, clip_limit=c)

        # Métricas promedio entre CTRL y HPMC
        ent_ctrl = shannon_entropy(ctrl_c)
        tng_ctrl = tenengrad_sharpness(ctrl_c)
        rsr_ctrl = residual_signal_ratio(ctrl_wavelet, ctrl_c)

        ent_hpmc = shannon_entropy(hpmc_c)
        tng_hpmc = tenengrad_sharpness(hpmc_c)
        rsr_hpmc = residual_signal_ratio(hpmc_wavelet, hpmc_c)

        ent = (ent_ctrl + ent_hpmc) / 2
        tng = (tng_ctrl + tng_hpmc) / 2
        rsr = (rsr_ctrl + rsr_hpmc) / 2

        print(f"{c:<10.1f} {ent:>10.3f} {tng:>12.1f} {rsr:>10.4f}")

from skimage.filters import frangi
import numpy as np

def apply_frangi(image, sigmas=range(1, 10), alpha=0.5, beta=0.5, gamma=25, black_ridges=True):
    """
    Filtro de Frangi ajustado para imágenes de fluorescencia (GFAP):
    - Escalas más amplias (1–9)
    - Gamma mayor (mejor contraste estructural)
    - Realce logarítmico posterior para evitar pérdida de rango dinámico.
    """
    img_norm = image.astype(np.float32) / 255.0

    enhanced = frangi(
        img_norm,
        sigmas=sigmas,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        black_ridges=black_ridges
    )

    # Evitar valores negativos y realzar rango bajo
    enhanced = np.clip(enhanced, 0, None)
    if enhanced.max() > 0:
        enhanced = enhanced / enhanced.max()
    enhanced = np.power(enhanced, 0.2)  # realce agresivo de contraste
    enhanced = (enhanced * 255).astype(np.uint8)
    return enhanced


def compare_frangi_mosaic(ctrl_original, hpmc_original, ctrl_wavelet, hpmc_wavelet, clip_limit=2.0):
    """
    Muestra un mosaico comparativo de:
    Wavelet → Wavelet+CLAHE → Frangi
    para CTRL y HPMC.
    """
    # Aplicar CLAHE
    #ctrl_clahe = apply_clahe(ctrl_wavelet, clip_limit=clip_limit)
    #hpmc_clahe = apply_clahe(hpmc_wavelet, clip_limit=clip_limit)

    # Aplicar Frangi
    ctrl_frangi = apply_frangi(ctrl_wavelet)
    hpmc_frangi = apply_frangi(hpmc_wavelet)

    # Crear mosaico 2x4
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    # Fila CTRL
    axes[0, 0].imshow(ctrl_original, cmap='gray')
    axes[0, 0].set_title("CTRL - Original")
    axes[0, 1].imshow(ctrl_wavelet, cmap='gray')
    axes[0, 1].set_title("CTRL - Wavelet")
    #axes[0, 2].imshow(ctrl_clahe, cmap='gray')
    #axes[0, 2].set_title(f"CTRL - CLAHE (clip={clip_limit})")
    axes[0, 3].imshow(ctrl_frangi, cmap='gray')
    axes[0, 3].set_title("CTRL - Frangi")

    # Fila HPMC
    axes[1, 0].imshow(hpmc_original, cmap='gray')
    axes[1, 0].set_title("HPMC - Original")
    axes[1, 1].imshow(hpmc_wavelet, cmap='gray')
    axes[1, 1].set_title("HPMC - Wavelet")
    #axes[1, 2].imshow(hpmc_clahe, cmap='gray')
    #axes[1, 2].set_title(f"HPMC - CLAHE (clip={clip_limit})")
    axes[1, 3].imshow(hpmc_frangi, cmap='gray')
    axes[1, 3].set_title("HPMC - Frangi")

    # Formato
    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def compare_frangi_mosaic_v2(ctrl_original, hpmc_original,
                          ctrl_wavelet, hpmc_wavelet,
                          clip_limit=2.0):
    """
    Muestra un mosaico comparativo de:
    Wavelet → Frangi → Frangi+CLAHE (visual)
    para CTRL y HPMC.
    """
    # Aplicar Frangi directamente sobre las imágenes Wavelet
    ctrl_frangi = apply_frangi(ctrl_wavelet)
    hpmc_frangi = apply_frangi(hpmc_wavelet)

    # Aplicar CLAHE solo para visualización (opcional)
    ctrl_frangi_clahe = apply_clahe(ctrl_frangi, clip_limit=clip_limit)
    hpmc_frangi_clahe = apply_clahe(hpmc_frangi, clip_limit=clip_limit)

    # Crear mosaico 2x4
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    # Fila CTRL
    axes[0, 0].imshow(ctrl_original, cmap='gray')
    axes[0, 0].set_title("CTRL - Original")
    axes[0, 1].imshow(ctrl_wavelet, cmap='gray')
    axes[0, 1].set_title("CTRL - Wavelet")
    axes[0, 2].imshow(ctrl_frangi, cmap='gray')
    axes[0, 2].set_title("CTRL - Frangi (Wavelet)")
    axes[0, 3].imshow(ctrl_frangi_clahe, cmap='gray')
    axes[0, 3].set_title(f"CTRL - Frangi + CLAHE (clip={clip_limit})")

    # Fila HPMC
    axes[1, 0].imshow(hpmc_original, cmap='gray')
    axes[1, 0].set_title("HPMC - Original")
    axes[1, 1].imshow(hpmc_wavelet, cmap='gray')
    axes[1, 1].set_title("HPMC - Wavelet")
    axes[1, 2].imshow(hpmc_frangi, cmap='gray')
    axes[1, 2].set_title("HPMC - Frangi (Wavelet)")
    axes[1, 3].imshow(hpmc_frangi_clahe, cmap='gray')
    axes[1, 3].set_title(f"HPMC - Frangi + CLAHE (clip={clip_limit})")

    # Formato
    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def compare_frangi_metrics(ctrl_wavelet, hpmc_wavelet, clip_limit=1.0):
    """
    Calcula Entropía, Tenengrad y RSR para:
    - Wavelet
    - Frangi (sobre Wavelet)
    - Frangi + CLAHE (visual)
    """
    # Aplicar filtros
    ctrl_frangi = apply_frangi(ctrl_wavelet)
    hpmc_frangi = apply_frangi(hpmc_wavelet)

    ctrl_frangi_clahe = apply_clahe(ctrl_frangi, clip_limit=clip_limit)
    hpmc_frangi_clahe = apply_clahe(hpmc_frangi, clip_limit=clip_limit)

    # Definir versiones a comparar
    labels = ["Wavelet", "Frangi", f"Frangi + CLAHE (clip={clip_limit})"]
    ctrl_versions = [ctrl_wavelet, ctrl_frangi, ctrl_frangi_clahe]
    hpmc_versions = [hpmc_wavelet, hpmc_frangi, hpmc_frangi_clahe]

    print("CTRL:")
    print(f"{'Versión':<28} {'Entropía':>10} {'Tenengrad':>12} {'RSR':>10}")
    print("-" * 60)
    for lbl, img in zip(labels, ctrl_versions):
        ent = shannon_entropy(img)
        tng = tenengrad_sharpness(img)
        rsr = residual_signal_ratio(ctrl_wavelet, img)
        print(f"{lbl:<28} {ent:10.3f} {tng:12.1f} {rsr:10.4f}")

    print("\nHPMC:")
    print(f"{'Versión':<28} {'Entropía':>10} {'Tenengrad':>12} {'RSR':>10}")
    print("-" * 60)
    for lbl, img in zip(labels, hpmc_versions):
        ent = shannon_entropy(img)
        tng = tenengrad_sharpness(img)
        rsr = residual_signal_ratio(hpmc_wavelet, img)
        print(f"{lbl:<28} {ent:10.3f} {tng:12.1f} {rsr:10.4f}")


from skimage.filters import sato

def apply_sato(image,
               sigmas=range(1, 6),
               black_ridges=True,
               gamma_exp=0.35):
    """
    Aplica el filtro de Sato (ridge / tubeness) sobre una imagen.

    Parámetros:
    - sigmas: iterable de escalas (en píxeles) → grosor de filamentos
    - black_ridges: True si los filamentos son claros sobre fondo oscuro
    - gamma_exp: exponente para realzar respuestas débiles (post-procesado)

    Devuelve imagen uint8 [0,255]
    """
    img_norm = image.astype(np.float32) / 255.0

    ridge = sato(
        img_norm,
        sigmas=sigmas,
        black_ridges=black_ridges
    )

    # Normalización robusta + realce de respuestas bajas
    ridge = np.clip(ridge, 0, None)
    if ridge.max() > 0:
        ridge = ridge / ridge.max()

    ridge = np.power(ridge, gamma_exp)

    ridge = (ridge * 255).astype(np.uint8)
    return ridge


def compare_sato_mosaic(ctrl_wavelet,
                         hpmc_wavelet,
                         sigmas_list=[range(1,4), range(1,6), range(1,8)],
                         show_clahe=False,
                         clip_limit=2.0):
    """
    Muestra mosaico comparando Sato con distintos rangos de sigmas.
    Opcionalmente muestra Sato + CLAHE.
    """
    n = len(sigmas_list)

        # Cálculo de columnas
    cols_per_sigma = 2 if show_clahe else 1
    total_cols = 1 + n * cols_per_sigma

    fig, axes = plt.subplots(2, total_cols, figsize=(5*total_cols, 8))

    # Columna base: Wavelet
    axes[0,0].imshow(ctrl_wavelet, cmap='gray')
    axes[0,0].set_title("CTRL - Wavelet")
    axes[1,0].imshow(hpmc_wavelet, cmap='gray')
    axes[1,0].set_title("HPMC - Wavelet")

    col = 1
    for sigmas in sigmas_list:
        # Sato
        ctrl_sato = apply_sato(ctrl_wavelet, sigmas=sigmas)
        hpmc_sato = apply_sato(hpmc_wavelet, sigmas=sigmas)

        axes[0,col].imshow(ctrl_sato, cmap='gray')
        axes[0,col].set_title(f"CTRL - Sato\nsigmas={sigmas.start}-{sigmas.stop-1}")

        axes[1,col].imshow(hpmc_sato, cmap='gray')
        axes[1,col].set_title(f"HPMC - Sato\nsigmas={sigmas.start}-{sigmas.stop-1}")

        col += 1

        # ---- Opcional: Sato + CLAHE ----
        if show_clahe:
            ctrl_sato_clahe = apply_clahe(ctrl_sato, clip_limit=clip_limit)
            hpmc_sato_clahe = apply_clahe(hpmc_sato, clip_limit=clip_limit)

            axes[0,col].imshow(ctrl_sato_clahe, cmap='gray')
            axes[0,col].set_title(
                f"CTRL - Sato + CLAHE\nsigmas={sigmas.start}-{sigmas.stop-1}, clip={clip_limit}"
            )

            axes[1,col].imshow(hpmc_sato_clahe, cmap='gray')
            axes[1,col].set_title(
                f"HPMC - Sato + CLAHE\nsigmas={sigmas.start}-{sigmas.stop-1}, clip={clip_limit}"
            )

            col += 1

    # ---- Formato ----
    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def compare_sato_metrics(ctrl_wavelet,
                         hpmc_wavelet,
                         sigmas_list=[range(1,4), range(1,6), range(1,8)],
                         show_clahe=False,
                         clip_limit=2.0):
    """
    Calcula Entropía, Tenengrad y RSR para:
    - Wavelet (baseline)
    - Sato (distintos sigmas)
    - (opcional) Sato + CLAHE (solo visualización)
    """

    # ---- CTRL ----
    ctrl_labels = ["Wavelet"]
    ctrl_versions = [ctrl_wavelet]

    for sigmas in sigmas_list:
        ctrl_sato = apply_sato(ctrl_wavelet, sigmas=sigmas)
        ctrl_labels.append(f"Sato {sigmas.start}-{sigmas.stop-1}")
        ctrl_versions.append(ctrl_sato)

        if show_clahe:
            ctrl_sato_clahe = apply_clahe(ctrl_sato, clip_limit=clip_limit)
            ctrl_labels.append(f"Sato {sigmas.start}-{sigmas.stop-1} + CLAHE({clip_limit})")
            ctrl_versions.append(ctrl_sato_clahe)

    print("CTRL:")
    print(f"{'Versión':<28} {'Entropía':>10} {'Tenengrad':>12} {'RSR':>10}")
    print("-" * 60)
    for lbl, img in zip(ctrl_labels, ctrl_versions):
        ent = shannon_entropy(img)
        tng = tenengrad_sharpness(img)
        rsr = residual_signal_ratio(ctrl_wavelet, img)  # baseline Wavelet
        print(f"{lbl:<28} {ent:10.3f} {tng:12.1f} {rsr:10.4f}")

    # ---- HPMC ----
    hpmc_labels = ["Wavelet"]
    hpmc_versions = [hpmc_wavelet]

    for sigmas in sigmas_list:
        hpmc_sato = apply_sato(hpmc_wavelet, sigmas=sigmas)
        hpmc_labels.append(f"Sato {sigmas.start}-{sigmas.stop-1}")
        hpmc_versions.append(hpmc_sato)

        if show_clahe:
            hpmc_sato_clahe = apply_clahe(hpmc_sato, clip_limit=clip_limit)
            hpmc_labels.append(f"Sato {sigmas.start}-{sigmas.stop-1} + CLAHE({clip_limit})")
            hpmc_versions.append(hpmc_sato_clahe)

    print("\nHPMC:")
    print(f"{'Versión':<28} {'Entropía':>10} {'Tenengrad':>12} {'RSR':>10}")
    print("-" * 60)
    for lbl, img in zip(hpmc_labels, hpmc_versions):
        ent = shannon_entropy(img)
        tng = tenengrad_sharpness(img)
        rsr = residual_signal_ratio(hpmc_wavelet, img)  # baseline Wavelet
        print(f"{lbl:<28} {ent:10.3f} {tng:12.1f} {rsr:10.4f}")


def compare_sato_clahe_order_metrics(ctrl_wavelet,
                                     hpmc_wavelet,
                                     sigmas_list=[range(1,4), range(1,6), range(1,8)],
                                     clip_limit=2.0):
    """
    Compara el orden CLAHE/Sato para distintos rangos de sigmas.
    Baseline siempre: Wavelet.
    """

    def _print_block(title, wavelet_img):
        print(title)
        print(f"{'Versión':<45} {'Entropía':>10} {'Tenengrad':>12} {'RSR':>10}")
        print("-" * 80)

        # Baseline
        ent = shannon_entropy(wavelet_img)
        tng = tenengrad_sharpness(wavelet_img)
        print(f"{'Wavelet':<45} {ent:10.3f} {tng:12.1f} {0.0:10.4f}")

        for sigmas in sigmas_list:
            tag = f"{sigmas.start}-{sigmas.stop-1}"

            # Wavelet -> Sato
            sato = apply_sato(wavelet_img, sigmas=sigmas)

            ent = shannon_entropy(sato)
            tng = tenengrad_sharpness(sato)
            rsr = residual_signal_ratio(wavelet_img, sato)
            print(f"{f'Sato {tag}':<45} {ent:10.3f} {tng:12.1f} {rsr:10.4f}")

            # Wavelet -> CLAHE -> Sato
            clahe_pre = apply_clahe(wavelet_img, clip_limit=clip_limit)
            sato_from_clahe = apply_sato(clahe_pre, sigmas=sigmas)

            ent = shannon_entropy(sato_from_clahe)
            tng = tenengrad_sharpness(sato_from_clahe)
            rsr = residual_signal_ratio(wavelet_img, sato_from_clahe)
            print(f"{f'CLAHE({clip_limit}) → Sato {tag}':<45} {ent:10.3f} {tng:12.1f} {rsr:10.4f}")

            # Wavelet -> Sato -> CLAHE (visual)
            sato_clahe_post = apply_clahe(sato, clip_limit=clip_limit)

            ent = shannon_entropy(sato_clahe_post)
            tng = tenengrad_sharpness(sato_clahe_post)
            rsr = residual_signal_ratio(wavelet_img, sato_clahe_post)
            print(f"{f'Sato {tag} → CLAHE({clip_limit})':<45} {ent:10.3f} {tng:12.1f} {rsr:10.4f}")

        print()

    _print_block("CTRL:", ctrl_wavelet)
    _print_block("HPMC:", hpmc_wavelet)

def compare_sato_clahe_order_mosaic(ctrl_wavelet,
                                    hpmc_wavelet,
                                    sigmas_list=[range(1,4), range(1,6), range(1,8)],
                                    clip_limit=2.0):
    """
    Mosaico:
    Wavelet | Sato | CLAHE→Sato | Sato→CLAHE
    para cada rango de sigmas.
    """
    for sigmas in sigmas_list:
        tag = f"{sigmas.start}-{sigmas.stop-1}"

        # CTRL
        ctrl_sato = apply_sato(ctrl_wavelet, sigmas=sigmas)
        ctrl_clahe_pre = apply_clahe(ctrl_wavelet, clip_limit=clip_limit)
        ctrl_sato_from_clahe = apply_sato(ctrl_clahe_pre, sigmas=sigmas)
        ctrl_sato_clahe_post = apply_clahe(ctrl_sato, clip_limit=clip_limit)

        # HPMC
        hpmc_sato = apply_sato(hpmc_wavelet, sigmas=sigmas)
        hpmc_clahe_pre = apply_clahe(hpmc_wavelet, clip_limit=clip_limit)
        hpmc_sato_from_clahe = apply_sato(hpmc_clahe_pre, sigmas=sigmas)
        hpmc_sato_clahe_post = apply_clahe(hpmc_sato, clip_limit=clip_limit)

        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        fig.suptitle(f"Sato sigmas={tag}", fontsize=16)

        axes[0,0].imshow(ctrl_wavelet, cmap='gray'); axes[0,0].set_title("CTRL - Wavelet")
        axes[0,1].imshow(ctrl_sato, cmap='gray'); axes[0,1].set_title("CTRL - Sato")
        axes[0,2].imshow(ctrl_sato_from_clahe, cmap='gray'); axes[0,2].set_title("CTRL - CLAHE → Sato")
        axes[0,3].imshow(ctrl_sato_clahe_post, cmap='gray'); axes[0,3].set_title("CTRL - Sato → CLAHE")

        axes[1,0].imshow(hpmc_wavelet, cmap='gray'); axes[1,0].set_title("HPMC - Wavelet")
        axes[1,1].imshow(hpmc_sato, cmap='gray'); axes[1,1].set_title("HPMC - Sato")
        axes[1,2].imshow(hpmc_sato_from_clahe, cmap='gray'); axes[1,2].set_title("HPMC - CLAHE → Sato")
        axes[1,3].imshow(hpmc_sato_clahe_post, cmap='gray'); axes[1,3].set_title("HPMC - Sato → CLAHE")

        for ax in axes.ravel():
            ax.axis("off")

        plt.tight_layout()
        plt.show()


# SEGMENTACIÓN
import numpy as np
import cv2 as cv
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, remove_small_holes, binary_opening, binary_closing, disk
from skimage.measure import label
from skimage.morphology import skeletonize
from scipy.ndimage import convolve
from skimage.filters import threshold_triangle

def binarize_sato(sato_img,
                  method="percentile",
                  q=88,
                  block_size=51,
                  C=2,
                  k=2,
                  blur_ksize=3):
    """
    Devuelve mask binaria (bool) a partir del mapa Sato (uint8 o float).
    """
    x = sato_img.copy()

    # Asegurar uint8 0-255 para métodos de OpenCV
    if x.dtype != np.uint8:
        x = x.astype(np.float32)
        x = x - x.min()
        if x.max() > 0:
            x = x / x.max()
        x = (255 * x).astype(np.uint8)

    if blur_ksize and blur_ksize > 1:
        x_blur = cv.GaussianBlur(x, (blur_ksize, blur_ksize), 0)
    else:
        x_blur = x

    if method == "percentile":
        T = np.percentile(x_blur, q)
        mask = x_blur >= T
    
    elif method == "triangle":
        T = threshold_triangle(x_blur)
        mask = x_blur > T

    elif method == "otsu":
        T = threshold_otsu(x_blur)
        mask = x_blur >= T

    elif method == "adaptive":
        # block_size debe ser impar
        if block_size % 2 == 0:
            block_size += 1
        th = cv.adaptiveThreshold(
            x_blur, 255,
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY,
            block_size, C
        )
        mask = th.astype(bool)

    elif method == "kmeans":
        data = x_blur.reshape(-1, 1).astype(np.float32)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 0.2)
        _, labels, centers = cv.kmeans(data, k, None, criteria, 5, cv.KMEANS_PP_CENTERS)
        centers = centers.flatten()
        # asumimos filamentos = cluster de mayor intensidad
        fg = np.argmax(centers)
        mask = (labels.flatten() == fg).reshape(x_blur.shape)

    elif method == "canny_fill":
        # Canny da bordes; los engrosamos y rellenamos para tener máscara aproximada
        edges = cv.Canny(x_blur, 50, 150)
        edges = cv.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
        # rellenar regiones cerradas (aprox)
        inv = cv.bitwise_not(edges)
        mask = inv.astype(bool)

    else:
        raise ValueError("method must be: percentile, triangle, otsu, adaptive, kmeans, canny_fill")

    return mask


def clean_mask(mask,
               min_size=200,
               hole_size=200,
               open_r=1,
               close_r=2):
    """
    Limpieza morfológica suave para no romper ramificaciones.
    """
    mask = remove_small_objects(mask, min_size=min_size)
    mask = binary_opening(mask, footprint=disk(open_r))
    mask = binary_closing(mask, footprint=disk(close_r))
    mask = remove_small_holes(mask, area_threshold=hole_size)
    return mask


def skeleton_and_graph_stats(mask):
    """
    Métricas rápidas para comparar métodos.
    """
    skel = skeletonize(mask)

    # longitud (px)
    skel_len = int(skel.sum())

    # componentes conectadas
    lbl = label(mask)
    n_comp = int(lbl.max())

    # fill ratio
    fill = float(mask.mean())

    # junctions aproximados: pixeles del skeleton con >=3 vecinos
    K = np.ones((3,3), dtype=np.uint8)
    nb = convolve(skel.astype(np.uint8), K, mode="constant")
    junctions = int(((skel == 1) & (nb >= 4)).sum())  # >=3 vecinos + el propio

    return {
        "fill_ratio": fill,
        "n_components": n_comp,
        "skeleton_length": skel_len,
        "junctions": junctions
    }, skel

    #VISUALIZACIÓN SEGMENTACIÓN
import matplotlib.pyplot as plt

def compare_binarization_methods_mosaic(ctrl_sato, hpmc_sato,
                                       methods=("percentile","otsu","adaptive","kmeans"),
                                       percentile_q=88,
                                       adaptive_block=51,
                                       adaptive_C=2,
                                       kmeans_k=2):
    """
    Mosaico 2 x (len(methods)+1):
    Sato | máscaras por método
    + imprime stats por método
    """
    cols = 1 + len(methods)
    fig, axes = plt.subplots(2, cols, figsize=(5*cols, 8))

    axes[0,0].imshow(ctrl_sato, cmap="gray"); axes[0,0].set_title("CTRL - Sato")
    axes[1,0].imshow(hpmc_sato, cmap="gray"); axes[1,0].set_title("HPMC - Sato")

    print("== MASK STATS (after clean_mask) ==")

    for j, m in enumerate(methods, start=1):
        ctrl_mask = binarize_sato(ctrl_sato, method=m, q=percentile_q,
                                  block_size=adaptive_block, C=adaptive_C, k=kmeans_k)
        hpmc_mask = binarize_sato(hpmc_sato, method=m, q=percentile_q,
                                  block_size=adaptive_block, C=adaptive_C, k=kmeans_k)

        ctrl_mask = clean_mask(ctrl_mask)
        hpmc_mask = clean_mask(hpmc_mask)

        ctrl_stats, ctrl_skel = skeleton_and_graph_stats(ctrl_mask)
        hpmc_stats, hpmc_skel = skeleton_and_graph_stats(hpmc_mask)

        axes[0,j].imshow(ctrl_mask, cmap="gray")
        axes[0,j].set_title(f"CTRL - {m}")
        axes[1,j].imshow(hpmc_mask, cmap="gray")
        axes[1,j].set_title(f"HPMC - {m}")

        print(f"\nMethod: {m}")
        print(" CTRL:", ctrl_stats)
        print(" HPMC:", hpmc_stats)

    for ax in axes.ravel(): ax.axis("off")
    plt.tight_layout()
    plt.show()

    # gana percentil

# comparamos distinots q para percentil
import matplotlib.pyplot as plt

def compare_percentile_q_mosaic(ctrl_sato,
                                hpmc_sato,
                                q_list=(82, 85, 88, 90),
                                min_size=200,
                                hole_size=200,
                                open_r=1,
                                close_r=2):
    """
    Compara máscaras por percentile q sobre el mapa Sato.
    Muestra mosaico y reporta stats.
    """
    cols = 1 + len(q_list)
    fig, axes = plt.subplots(2, cols, figsize=(5*cols, 8))

    axes[0,0].imshow(ctrl_sato, cmap="gray"); axes[0,0].set_title("CTRL - Sato")
    axes[1,0].imshow(hpmc_sato, cmap="gray"); axes[1,0].set_title("HPMC - Sato")

    print("== PERCENTILE-Q STATS (after clean_mask) ==")

    for j, q in enumerate(q_list, start=1):
        ctrl_mask = binarize_sato(ctrl_sato, method="percentile", q=q)
        hpmc_mask = binarize_sato(hpmc_sato, method="percentile", q=q)

        ctrl_mask = clean_mask(ctrl_mask,
                               min_size=min_size,
                               hole_size=hole_size,
                               open_r=open_r,
                               close_r=close_r)
        hpmc_mask = clean_mask(hpmc_mask,
                               min_size=min_size,
                               hole_size=hole_size,
                               open_r=open_r,
                               close_r=close_r)

        ctrl_stats, ctrl_skel = skeleton_and_graph_stats(ctrl_mask)
        hpmc_stats, hpmc_skel = skeleton_and_graph_stats(hpmc_mask)

        axes[0,j].imshow(ctrl_mask, cmap="gray")
        axes[0,j].set_title(f"CTRL - q={q}")

        axes[1,j].imshow(hpmc_mask, cmap="gray")
        axes[1,j].set_title(f"HPMC - q={q}")

        print(f"\nq = {q}")
        print(" CTRL:", ctrl_stats)
        print(" HPMC:", hpmc_stats)

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    # gana el 85

# mejorar la mask
from skimage.morphology import binary_closing, binary_dilation, disk

def refine_mask_variants(mask):
    """
    Genera variantes suaves de postprocesado morfológico
    a partir de una máscara binaria base.
    """
    variants = {
        "base": mask,
        "closing_r2": binary_closing(mask, footprint=disk(2)),
        "dilate_r1": binary_dilation(mask, footprint=disk(1)),
        "closing_r2_dilate_r1": binary_dilation(
            binary_closing(mask, footprint=disk(2)),
            footprint=disk(1)
        )
    }
    return variants

import matplotlib.pyplot as plt

def compare_mask_refinement_mosaic(ctrl_sato, hpmc_sato, q=85):
    """
    Compara distintas variantes de refinamiento morfológico
    sobre la máscara percentile(q).
    """

    # máscara base
    ctrl_mask = binarize_sato(ctrl_sato, method="percentile", q=q)
    hpmc_mask = binarize_sato(hpmc_sato, method="percentile", q=q)

    ctrl_mask = clean_mask(ctrl_mask, min_size=200, hole_size=200, open_r=1, close_r=2)
    hpmc_mask = clean_mask(hpmc_mask, min_size=200, hole_size=200, open_r=1, close_r=2)

    # variantes
    ctrl_variants = refine_mask_variants(ctrl_mask)
    hpmc_variants = refine_mask_variants(hpmc_mask)

    labels = list(ctrl_variants.keys())

    fig, axes = plt.subplots(2, len(labels) + 1, figsize=(5 * (len(labels) + 1), 8))

    axes[0, 0].imshow(ctrl_sato, cmap="gray")
    axes[0, 0].set_title("CTRL - Sato")

    axes[1, 0].imshow(hpmc_sato, cmap="gray")
    axes[1, 0].set_title("HPMC - Sato")

    print("== MASK REFINEMENT STATS ==")

    for j, lbl in enumerate(labels, start=1):
        ctrl_v = ctrl_variants[lbl]
        hpmc_v = hpmc_variants[lbl]

        ctrl_stats, ctrl_skel = skeleton_and_graph_stats(ctrl_v)
        hpmc_stats, hpmc_skel = skeleton_and_graph_stats(hpmc_v)

        axes[0, j].imshow(ctrl_v, cmap="gray")
        axes[0, j].set_title(f"CTRL - {lbl}")

        axes[1, j].imshow(hpmc_v, cmap="gray")
        axes[1, j].set_title(f"HPMC - {lbl}")

        print(f"\nVariant: {lbl}")
        print(" CTRL:", ctrl_stats)
        print(" HPMC:", hpmc_stats)

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    # el closing_r2 es al pedo (es lo mismo que el base), dilate mejora, closing_r2_dilate_r1 es igual a dilate
    # --> quedamos con solo dilate

#skeleton
from skimage.morphology import binary_dilation, disk

def build_final_mask_from_sato(sato_img, q=85,
                               min_size=200, hole_size=200,
                               open_r=1, close_r=2,
                               dilate_r=1):
    """
    Pipeline final de máscara:
    Sato -> percentile(q) -> clean_mask -> dilatación suave
    """
    base_mask = binarize_sato(sato_img, method="percentile", q=q)

    base_mask = clean_mask(
        base_mask,
        min_size=min_size,
        hole_size=hole_size,
        open_r=open_r,
        close_r=close_r
    )

    final_mask = binary_dilation(base_mask, footprint=disk(dilate_r))

    return base_mask, final_mask

import matplotlib.pyplot as plt

def show_final_mask_and_skeleton(ctrl_sato, hpmc_sato, q=85,
                                 min_size=200, hole_size=200,
                                 open_r=1, close_r=2,
                                 dilate_r=1):
    """
    Muestra:
    Sato -> mask base -> mask final -> skeleton
    para CTRL y HPMC.
    También imprime stats de la máscara final.
    """

    # CTRL
    ctrl_base, ctrl_final = build_final_mask_from_sato(
        ctrl_sato, q=q,
        min_size=min_size, hole_size=hole_size,
        open_r=open_r, close_r=close_r,
        dilate_r=dilate_r
    )
    ctrl_stats, ctrl_skel = skeleton_and_graph_stats(ctrl_final)

    # HPMC
    hpmc_base, hpmc_final = build_final_mask_from_sato(
        hpmc_sato, q=q,
        min_size=min_size, hole_size=hole_size,
        open_r=open_r, close_r=close_r,
        dilate_r=dilate_r
    )
    hpmc_stats, hpmc_skel = skeleton_and_graph_stats(hpmc_final)

    # Mosaico
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    axes[0,0].imshow(ctrl_sato, cmap="gray")
    axes[0,0].set_title("CTRL - Sato")

    axes[0,1].imshow(ctrl_base, cmap="gray")
    axes[0,1].set_title(f"CTRL - Base mask (q={q})")

    axes[0,2].imshow(ctrl_final, cmap="gray")
    axes[0,2].set_title(f"CTRL - Final mask (+ dilate r={dilate_r})")

    axes[0,3].imshow(ctrl_skel, cmap="gray")
    axes[0,3].set_title("CTRL - Skeleton")

    axes[1,0].imshow(hpmc_sato, cmap="gray")
    axes[1,0].set_title("HPMC - Sato")

    axes[1,1].imshow(hpmc_base, cmap="gray")
    axes[1,1].set_title(f"HPMC - Base mask (q={q})")

    axes[1,2].imshow(hpmc_final, cmap="gray")
    axes[1,2].set_title(f"HPMC - Final mask (+ dilate r={dilate_r})")

    axes[1,3].imshow(hpmc_skel, cmap="gray")
    axes[1,3].set_title("HPMC - Skeleton")

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    print("== FINAL MASK STATS ==")
    print("CTRL:", ctrl_stats)
    print("HPMC:", hpmc_stats)

    return {
        "ctrl": {
            "base_mask": ctrl_base,
            "final_mask": ctrl_final,
            "skeleton": ctrl_skel,
            "stats": ctrl_stats
        },
        "hpmc": {
            "base_mask": hpmc_base,
            "final_mask": hpmc_final,
            "skeleton": hpmc_skel,
            "stats": hpmc_stats
        }
    }

#pruning del skeleton
import numpy as np
from scipy.ndimage import convolve

def prune_skeleton(skel, prune_iters=5):
    """
    Poda endpoints de un skeleton de forma iterativa.

    Parámetros
    ----------
    skel : np.ndarray (bool)
        Skeleton binario.
    prune_iters : int
        Número de iteraciones de poda.

    Retorna
    -------
    np.ndarray (bool)
        Skeleton podado.
    """
    pruned = skel.copy().astype(bool)

    kernel = np.ones((3, 3), dtype=np.uint8)

    for _ in range(prune_iters):
        # contar vecinos en la vecindad 3x3
        neigh = convolve(pruned.astype(np.uint8), kernel, mode="constant", cval=0)

        # endpoint: pixel del skeleton con exactamente 1 vecino + él mismo = 2
        endpoints = pruned & (neigh == 2)

        # eliminar endpoints
        pruned = pruned & ~endpoints

    return pruned

import matplotlib.pyplot as plt

def compare_skeleton_pruning(ctrl_final_mask, hpmc_final_mask,
                             prune_list=(0, 3, 5, 8)):
    """
    Compara skeleton sin pruning y con distintos niveles de pruning.
    """

    fig, axes = plt.subplots(2, len(prune_list) + 1, figsize=(5*(len(prune_list)+1), 8))

    # skeleton base
    ctrl_stats, ctrl_skel = skeleton_and_graph_stats(ctrl_final_mask)
    hpmc_stats, hpmc_skel = skeleton_and_graph_stats(hpmc_final_mask)

    axes[0, 0].imshow(ctrl_final_mask, cmap="gray")
    axes[0, 0].set_title("CTRL - Final mask")

    axes[1, 0].imshow(hpmc_final_mask, cmap="gray")
    axes[1, 0].set_title("HPMC - Final mask")

    print("== SKELETON PRUNING STATS ==")

    for j, p in enumerate(prune_list, start=1):
        ctrl_pruned = prune_skeleton(ctrl_skel, prune_iters=p)
        hpmc_pruned = prune_skeleton(hpmc_skel, prune_iters=p)

        ctrl_pruned_stats, _ = skeleton_and_graph_stats(ctrl_pruned)
        hpmc_pruned_stats, _ = skeleton_and_graph_stats(hpmc_pruned)

        axes[0, j].imshow(ctrl_pruned, cmap="gray", vmin=0, vmax=1)
        axes[0, j].set_title(f"CTRL - prune={p}")

        axes[1, j].imshow(hpmc_pruned, cmap="gray", vmin=0, vmax=1)
        axes[1, j].set_title(f"HPMC - prune={p}")

        print(f"\nprune_iters = {p}")
        print(" CTRL:", ctrl_pruned_stats)
        print(" HPMC:", hpmc_pruned_stats)

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.show()

import numpy as np
from skimage.morphology import binary_dilation, disk

def build_hybrid_mask_from_wavelet_and_sato(
        wavelet_img,
        sato_img,
        q_sato=85,
        q_wavelet=70,
        seed_dilate=2
):
    """
    Construye una máscara híbrida usando:
    Sato como semilla + Wavelet como soporte.
    """

    # semillas (filamentos detectados por Sato)
    t_sato = np.percentile(sato_img, q_sato)
    seed = sato_img > t_sato

    # región donde puede existir filamento
    t_wave = np.percentile(wavelet_img, q_wavelet)
    support = wavelet_img > t_wave

    # expandir semillas
    seed = binary_dilation(seed, disk(seed_dilate))

    # intersección con soporte
    hybrid_mask = seed & support

    return seed, support, hybrid_mask

def compare_hybrid_mask(ctrl_wavelet, hpmc_wavelet,
                        sigmas=range(1,6),
                        q_sato=85,
                        q_wavelet=70):

    ctrl_sato = apply_sato(ctrl_wavelet, sigmas=sigmas)
    hpmc_sato = apply_sato(hpmc_wavelet, sigmas=sigmas)

    ctrl_seed, ctrl_support, ctrl_mask = build_hybrid_mask_from_wavelet_and_sato(
        ctrl_wavelet, ctrl_sato, q_sato, q_wavelet
    )

    hpmc_seed, hpmc_support, hpmc_mask = build_hybrid_mask_from_wavelet_and_sato(
        hpmc_wavelet, hpmc_sato, q_sato, q_wavelet
    )

    fig, axes = plt.subplots(2,4, figsize=(20,8))

    axes[0,0].imshow(ctrl_sato, cmap="gray")
    axes[0,0].set_title("CTRL Sato")

    axes[0,1].imshow(ctrl_seed, cmap="gray")
    axes[0,1].set_title("CTRL seeds")

    axes[0,2].imshow(ctrl_support, cmap="gray")
    axes[0,2].set_title("CTRL support")

    axes[0,3].imshow(ctrl_mask, cmap="gray")
    axes[0,3].set_title("CTRL hybrid mask")

    axes[1,0].imshow(hpmc_sato, cmap="gray")
    axes[1,0].set_title("HPMC Sato")

    axes[1,1].imshow(hpmc_seed, cmap="gray")
    axes[1,1].set_title("HPMC seeds")

    axes[1,2].imshow(hpmc_support, cmap="gray")
    axes[1,2].set_title("HPMC support")

    axes[1,3].imshow(hpmc_mask, cmap="gray")
    axes[1,3].set_title("HPMC hybrid mask")

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    return ctrl_mask, hpmc_mask

#percentile sobre wavelet en vez de sato
from skimage.morphology import binary_closing, binary_dilation, disk

def refine_wavelet_mask_variants(mask):
    """
    Variantes de refinamiento morfológico para máscaras segmentadas desde Wavelet.
    """
    variants = {
        "base": mask,
        "closing_r2": binary_closing(mask, footprint=disk(2)),
        "closing_r3": binary_closing(mask, footprint=disk(3)),
        "closing_r3_dilate_r1": binary_dilation(
            binary_closing(mask, footprint=disk(3)),
            footprint=disk(1)
        )
    }
    return variants

import matplotlib.pyplot as plt

def compare_wavelet_masks(ctrl_wavelet, hpmc_wavelet,
                          q_list=(70, 75, 80, 85),
                          min_size=200,
                          hole_size=200,
                          open_r=1,
                          close_r=2):
    """
    Compara máscaras segmentadas directamente sobre Wavelet
    usando distintos percentiles y refinamientos morfológicos.
    """

    for q in q_list:
        # máscaras base
        ctrl_mask = binarize_sato(ctrl_wavelet, method="percentile", q=q)
        hpmc_mask = binarize_sato(hpmc_wavelet, method="percentile", q=q)

        ctrl_mask = clean_mask(ctrl_mask,
                               min_size=min_size,
                               hole_size=hole_size,
                               open_r=open_r,
                               close_r=close_r)
        hpmc_mask = clean_mask(hpmc_mask,
                               min_size=min_size,
                               hole_size=hole_size,
                               open_r=open_r,
                               close_r=close_r)

        ctrl_variants = refine_wavelet_mask_variants(ctrl_mask)
        hpmc_variants = refine_wavelet_mask_variants(hpmc_mask)

        labels = list(ctrl_variants.keys())

        fig, axes = plt.subplots(2, len(labels)+1, figsize=(5*(len(labels)+1), 8))
        fig.suptitle(f"Wavelet segmentation - q={q}", fontsize=16)

        axes[0,0].imshow(ctrl_wavelet, cmap="gray")
        axes[0,0].set_title("CTRL - Wavelet")

        axes[1,0].imshow(hpmc_wavelet, cmap="gray")
        axes[1,0].set_title("HPMC - Wavelet")

        print(f"\n== WAVELET MASK STATS | q = {q} ==")

        for j, lbl in enumerate(labels, start=1):
            ctrl_v = ctrl_variants[lbl]
            hpmc_v = hpmc_variants[lbl]

            ctrl_stats, _ = skeleton_and_graph_stats(ctrl_v)
            hpmc_stats, _ = skeleton_and_graph_stats(hpmc_v)

            axes[0,j].imshow(ctrl_v, cmap="gray")
            axes[0,j].set_title(f"CTRL - {lbl}")

            axes[1,j].imshow(hpmc_v, cmap="gray")
            axes[1,j].set_title(f"HPMC - {lbl}")

            print(f"\nVariant: {lbl}")
            print(" CTRL:", ctrl_stats)
            print(" HPMC:", hpmc_stats)

        for ax in axes.ravel():
            ax.axis("off")

        plt.tight_layout()
        plt.show()
#q80 (o q75)

#metemos skeleton a esta opción
from skimage.morphology import binary_closing, binary_dilation, disk

def build_wavelet_mask_candidate(wavelet_img, q=80, variant="closing_r3",
                                 min_size=200, hole_size=200,
                                 open_r=1, close_r=2):
    """
    Construye dos candidatos de máscara final sobre Wavelet:
    - closing_r3
    - closing_r3_dilate_r1
    """

    # máscara base
    mask = binarize_sato(wavelet_img, method="percentile", q=q)

    # limpieza base
    mask = clean_mask(
        mask,
        min_size=min_size,
        hole_size=hole_size,
        open_r=open_r,
        close_r=close_r
    )

    if variant == "closing_r3":
        final_mask = binary_closing(mask, footprint=disk(3))

    elif variant == "closing_r3_dilate_r1":
        final_mask = binary_dilation(
            binary_closing(mask, footprint=disk(3)),
            footprint=disk(1)
        )

    else:
        raise ValueError("variant must be 'closing_r3' or 'closing_r3_dilate_r1'")

    return final_mask

import matplotlib.pyplot as plt

def compare_wavelet_finalists(ctrl_wavelet, hpmc_wavelet,
                              q=80,
                              prune_iters=5):
    """
    Compara los dos candidatos finalistas:
    - q=80 + closing_r3
    - q=80 + closing_r3_dilate_r1

    mostrando:
    Wavelet -> máscara -> skeleton -> skeleton pruned
    """

    variants = ["closing_r3", "closing_r3_dilate_r1"]

    fig, axes = plt.subplots(2, 1 + len(variants)*3, figsize=(24, 8))

    # mostrar wavelet
    axes[0,0].imshow(ctrl_wavelet, cmap="gray")
    axes[0,0].set_title("CTRL - Wavelet")

    axes[1,0].imshow(hpmc_wavelet, cmap="gray")
    axes[1,0].set_title("HPMC - Wavelet")

    print("== FINALIST PIPELINE STATS ==")

    col = 1
    for variant in variants:
        # máscaras
        ctrl_mask = build_wavelet_mask_candidate(ctrl_wavelet, q=q, variant=variant)
        hpmc_mask = build_wavelet_mask_candidate(hpmc_wavelet, q=q, variant=variant)

        # skeletons
        ctrl_stats, ctrl_skel = skeleton_and_graph_stats(ctrl_mask)
        hpmc_stats, hpmc_skel = skeleton_and_graph_stats(hpmc_mask)

        # pruning
        ctrl_pruned = prune_skeleton(ctrl_skel, prune_iters=prune_iters)
        hpmc_pruned = prune_skeleton(hpmc_skel, prune_iters=prune_iters)

        ctrl_pruned_stats, _ = skeleton_and_graph_stats(ctrl_pruned)
        hpmc_pruned_stats, _ = skeleton_and_graph_stats(hpmc_pruned)

        # CTRL
        axes[0,col].imshow(ctrl_mask, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[0,col].set_title(f"CTRL - {variant}\nmask")

        axes[0,col+1].imshow(ctrl_skel, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[0,col+1].set_title(f"CTRL - {variant}\nskeleton")

        axes[0,col+2].imshow(ctrl_pruned, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[0,col+2].set_title(f"CTRL - {variant}\npruned ({prune_iters})")

        # HPMC
        axes[1,col].imshow(hpmc_mask, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[1,col].set_title(f"HPMC - {variant}\nmask")

        axes[1,col+1].imshow(hpmc_skel, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[1,col+1].set_title(f"HPMC - {variant}\nskeleton")

        axes[1,col+2].imshow(hpmc_pruned, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[1,col+2].set_title(f"HPMC - {variant}\npruned ({prune_iters})")

        print(f"\nVariant: {variant}")
        print(" CTRL mask  :", ctrl_stats)
        print(" CTRL pruned:", ctrl_pruned_stats)
        print(" HPMC mask  :", hpmc_stats)
        print(" HPMC pruned:", hpmc_pruned_stats)

        col += 3

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.show()

from skimage.morphology import skeletonize

def get_skeleton(mask):
    """
    Genera el skeleton binario de una máscara.

    Parameters
    ----------
    mask : np.ndarray
        Imagen binaria (0/1 o bool)

    Returns
    -------
    skel : np.ndarray
        Skeleton binario
    """

    mask = mask.astype(bool)

    skel = skeletonize(mask)

    return skel