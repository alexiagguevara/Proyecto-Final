from src.load_data import load_images
from src.visualize import show_examples
from src.visualize import show_all_channels_mosaic
from src.preprocessing import extract_green_channel
from src.visualize import show_rgb_vs_green_verif  
from src.preprocessing import check_image_ranges
from src.preprocessing import normalize_dataset
from src.preprocessing import show_filters_with_metrics, apply_clahe, apply_filters, compare_clahe_mosaic, evaluate_clahe, compare_frangi_mosaic
from src.preprocessing import compare_frangi_mosaic_v2, compare_frangi_metrics
from src.preprocessing import compare_sato_mosaic, compare_sato_metrics, compare_sato_clahe_order_metrics, compare_sato_clahe_order_mosaic
from src.preprocessing import apply_sato, compare_binarization_methods_mosaic, compare_percentile_q_mosaic
from src.preprocessing import compare_mask_refinement_mosaic
from src.preprocessing import show_final_mask_and_skeleton
from src.preprocessing import compare_skeleton_pruning, build_final_mask_from_sato
from src.preprocessing import compare_hybrid_mask
from src.preprocessing import compare_wavelet_masks
from src.preprocessing import compare_wavelet_finalists
from src.preprocessing import build_wavelet_mask_candidate


if __name__ == "__main__":
    base_path = "data/Fotos"  
    dataset = load_images(base_path)

    ## Verificación del contenido
    for grupo, condiciones in dataset.items():
        for cond, imgs in condiciones.items():
            if imgs:
                shape = imgs[0]['image'].shape
                print(f"{grupo} - {cond}: {len(imgs)} imágenes | Dimensiones: {shape}")
            else:
                print(f"{grupo} - {cond}: 0 imágenes")

    # Mostrar ejemplos
    # show_examples(dataset, n=1)

    # Mosaico de todas las imágenes CTRL
    # show_all_channels_mosaic(dataset, cond_filter="CTRL")

    # Mosaico de todas las imágenes HPMC
    # show_all_channels_mosaic(dataset, cond_filter="HPMC")

     # Preprocesamiento: extraer canal verde y crear su dataset
    green_dataset = extract_green_channel(dataset)

    # Verificación: mostramos un ejemplo
    #show_rgb_vs_green_verif(dataset, green_dataset, "N3 y N4")

    # Revisar rangos de todas las imágenes
    #check_image_ranges(green_dataset)

    # Normalizar todas las imágenes del dataset del canal verde
    green_norm_dataset = normalize_dataset(green_dataset, robust=True)
    # Revisar rangos de todas las imágenes
    #check_image_ranges(green_norm_dataset)

    # Elegimos un ejemplo CTRL y HPMC del green_dataset
    ctrl_example = green_norm_dataset["N1 y N2"]["CTRL"][0]["image"]
    hpmc_example = green_norm_dataset["N1 y N2"]["HPMC"][0]["image"]

    # Comparar filtros con métricas
    #show_filters_with_metrics(ctrl_example, hpmc_example) #resultado: wavelet es el mejor

    # Aplicar filtro Wavelet a la imagen normalizada
    _, _, ctrl_wavelet = apply_filters(ctrl_example)
    _, _, hpmc_wavelet = apply_filters(hpmc_example)

    # Aplicar CLAHE sobre las versiones filtradas (clip_values=[1.0, 2.0, 4.0])
    #compare_clahe_mosaic(ctrl_wavelet, hpmc_wavelet) #visualmente conviene 2.0
    
    #Métricas para elegir el mejor clip_value
    #evaluate_clahe(ctrl_wavelet, hpmc_wavelet) #me quedo con el 2.0

    #compare_frangi_mosaic(ctrl_example, hpmc_example,
    #                  ctrl_wavelet, hpmc_wavelet,
    #                  clip_limit=1.0)

    #compare_frangi_mosaic_v2(ctrl_example, hpmc_example,
    #                  ctrl_wavelet, hpmc_wavelet,
    #                  clip_limit=1.0)

    #compare_frangi_metrics(ctrl_wavelet, hpmc_wavelet, clip_limit=1.0)

# Ya tenés ctrl_wavelet y hpmc_wavelet, no vamos a ejecutar el clahe
sigmas_list = [range(1,4), range(1,6), range(1,8)]

#compare_sato_mosaic(ctrl_wavelet, hpmc_wavelet, sigmas_list, show_clahe=False, clip_limit=2.0) 
#compare_sato_metrics(ctrl_wavelet, hpmc_wavelet, sigmas_list, show_clahe=False, clip_limit=2.0)

#compare_sato_clahe_order_mosaic(ctrl_wavelet, hpmc_wavelet, sigmas_list=sigmas_list, clip_limit=1.0)
#compare_sato_clahe_order_metrics(ctrl_wavelet, hpmc_wavelet, sigmas_list=sigmas_list, clip_limit=1.0)

# ya tenés ctrl_wavelet, hpmc_wavelet
ctrl_sato = apply_sato(ctrl_wavelet, sigmas=range(1,6))
hpmc_sato = apply_sato(hpmc_wavelet, sigmas=range(1,6))

#compare_binarization_methods_mosaic(
#    ctrl_sato, hpmc_sato,
#    methods=("percentile","otsu","adaptive","kmeans"),
    
#    percentile_q=88,
#    adaptive_block=51,
#    adaptive_C=2,
#    kmeans_k=2
#)
#gana percentile, ahora me quedo con el 88

#compare_percentile_q_mosaic(
#    ctrl_sato,
#    hpmc_sato,
#    q_list=(82, 85, 88, 90),
#    min_size=200,
#    hole_size=200,
#    open_r=1,
#    close_r=2
#)
#gana el 85

#compare_mask_refinement_mosaic(ctrl_sato, hpmc_sato, q=85)
# el closing_r2 es al pedo (es lo mismo que el base), dilate mejora, closing_r2_dilate_r1 es igual a dilate
# --> quedamos con solo dilate

#results = show_final_mask_and_skeleton(
#    ctrl_sato,
#   hpmc_sato,
 #   q=85,
 #   min_size=200,
 #   hole_size=200,
 #   open_r=1,
 #   close_r=2,
 #   dilate_r=1
#)

#probamos triangle contra percentile
#compare_binarization_methods_mosaic(
#    ctrl_sato, hpmc_sato,
#    methods=("percentile","triangle"),
  
#    percentile_q=85,
#    adaptive_block=51,
#    adaptive_C=2,
#    kmeans_k=2
#)
#triangle detecta muy poco

#pruning del skeleton
#ctrl_base, ctrl_final = build_final_mask_from_sato(ctrl_sato, q=85, dilate_r=1)
#hpmc_base, hpmc_final = build_final_mask_from_sato(hpmc_sato, q=85, dilate_r=1)

#compare_skeleton_pruning(
 #   ctrl_final,
 #   hpmc_final,
 #   prune_list=(0, 3, 5, 8)
#)

#ctrl_mask, hpmc_mask = compare_hybrid_mask(
#    ctrl_wavelet,
#    hpmc_wavelet,
#    sigmas=range(1,6),
#    q_sato=85,
#    q_wavelet=70
#)

#compare_wavelet_masks(
#    ctrl_wavelet,
#    hpmc_wavelet,
#    q_list=(70, 75, 80, 85),
#    min_size=200,
#    hole_size=200,
#    open_r=1,
#    close_r=2
#)

#compare_wavelet_finalists(
#    ctrl_wavelet,
#    hpmc_wavelet,
#    q=80,
#    prune_iters=5
#)
#quedamos solo con closing_r3

## EXTRAEMOS FEATURES DE CADA IMAGEN ( 1 CTRL Y 1 HPMC)
# from src.features import extract_all_features
# from src.preprocessing import get_skeleton, prune_skeleton

# # máscara final
# ctrl_mask = build_wavelet_mask_candidate(ctrl_wavelet, q=80, variant="closing_r3")
# hpmc_mask = build_wavelet_mask_candidate(hpmc_wavelet, q=80, variant="closing_r3")

# # skeleton
# ctrl_skel = get_skeleton(ctrl_mask)
# hpmc_skel = get_skeleton(hpmc_mask)

# # pruning
# ctrl_skel_pruned = prune_skeleton(ctrl_skel, prune_iters=5)
# hpmc_skel_pruned = prune_skeleton(hpmc_skel, prune_iters=5)

# # features
# ctrl_features = extract_all_features(ctrl_mask, ctrl_skel_pruned)
# hpmc_features = extract_all_features(hpmc_mask, hpmc_skel_pruned)

# print("CTRL FEATURES")
# for k, v in ctrl_features.items():
#     print(f"{k}: {v}")

# print("\nHPMC FEATURES")
# for k, v in hpmc_features.items():
#     print(f"{k}: {v}")

#EXTRAEMOS FEATURES DE TODAS
from pipeline import  build_feature_dataset

df_features = build_feature_dataset(
    green_norm_dataset,
    q=80,
    variant="closing_r3",
    prune_iters=5
)

print(df_features.head())
print(df_features.shape)

df_features.to_csv("features_dataset.csv", index=False)

import pandas as pd

df_features = pd.read_csv("features_dataset.csv")

summary = df_features.groupby("condition").mean(numeric_only=True)
print(summary.T)

# VISUALIZAMOS FEATURES
from pipeline import plot_feature_boxplots

plot_feature_boxplots(df_features, [
    "skeleton_length",
    "branch_density",
    "n_junctions",
    "n_segments",
    "largest_component_ratio",
    "mean_segment_length"
])