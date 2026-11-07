"""
Sensibilidad de los resultados al número de iteraciones de poda (prune_iters).

Uso (desde el notebook o desde un script en la raíz del proyecto):

    from sensibilidad_poda import run_sensitivity
    out = run_sensitivity(green_norm_dataset, csv_ref="features_dataset.csv")
    print(out["features_finales"].round(3).to_string(index=False))

green_norm_dataset es el mismo diccionario que le pasaste a build_feature_dataset.

Para no repetir trabajo, el filtrado wavelet, la máscara y el esqueleto se calculan UNA vez
por imagen (no dependen de prune_iters). Para cada valor de prune_iters solo se repiten la
poda y la extracción de características. Después, para cada valor:

  1) se resumen las 9 características del modelo final (mediana por condición, cambio
     porcentual respecto de no podar y d de Cohen) más skeleton_length como referencia;
  2) se repite la selección de características final sobre las 32 imágenes
     (select_final_features_from_full_dataset) y se guarda qué conjunto resulta;
  3) opcionalmente (run_nested=True) se corre la validación anidada leave-one-culture-out
     y se guardan las métricas globales.

Devuelve un diccionario con tres DataFrames y guarda tres CSV:
  - "resumen"            -> sensibilidad_poda.csv                       (una fila por prune_iters)
  - "features_finales"   -> sensibilidad_poda_features_finales.csv     (formato largo)
  - "conjuntos_finales"  -> sensibilidad_poda_conjuntos_finales.csv    (conjunto elegido por prune_iters)

Si pasás csv_ref (tu features_dataset.csv, generado con prune_iters=5), el script compara
su propio resultado con prune_iters=5 contra ese CSV para confirmar que reproduce tus números.
"""
import contextlib
import io

import numpy as np
import pandas as pd

from src.preprocessing import apply_filters, build_wavelet_mask_candidate, get_skeleton, prune_skeleton
from src.features import extract_all_features

PRUNE_LIST = (0, 3, 5)
CTRL, HPMC = "CTRL", "HPMC"

# Las 9 características del modelo final (sección 5.1.1 de la tesis).
FEATURES_FINALES = [
    "median_thickness", "median_tortuosity", "median_segment_length",
    "median_component_area", "mean_tortuosity", "largest_component_ratio",
    "max_segment_length", "max_thickness", "n_components",
]
# Se agrega como referencia directa del efecto de la poda (no es una de las 9).
FEATURES_REFERENCIA = ["skeleton_length"]

# Características calculadas solo sobre la máscara: no dependen de la poda por construcción.
MASK_FEATURES = {
    "mask_area", "fill_ratio", "n_components", "largest_component_area",
    "largest_component_ratio", "mean_component_area", "std_component_area",
    "median_component_area",
}

D_UMBRAL = 0.8  # umbral de |d| de Cohen para "alta importancia" (sección 4.6.1)


def cohen_d(a, b):
    """d de Cohen con desvío combinado; positivo si b (HPMC) > a (CTRL)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (b.mean() - a.mean()) / sp if sp > 0 else np.nan


def _precompute(dataset, q=80, variant="closing_r3"):
    """Parte costosa (se hace una sola vez): wavelet -> máscara -> esqueleto sin podar."""
    cache = []
    for group, conds in dataset.items():
        for condition, imgs in conds.items():
            for item in imgs:
                _, _, wavelet = apply_filters(item["image"])
                mask = build_wavelet_mask_candidate(wavelet, q=q, variant=variant)
                cache.append(dict(folder_group=group, condition=condition,
                                  image_name=item["name"], group=item["true_group"],
                                  mask=mask, skel0=get_skeleton(mask)))
    return cache


def _features_df(cache, prune_iters):
    rows = []
    for c in cache:
        skel = prune_skeleton(c["skel0"], prune_iters=prune_iters)
        feats = dict(extract_all_features(c["mask"], skel))
        feats["folder_group"] = c["folder_group"]
        feats["group"] = c["group"]
        feats["condition"] = c["condition"]
        feats["image_name"] = c["image_name"]
        rows.append(feats)
    return pd.DataFrame(rows)


def _check_against_csv(df5, csv_ref):
    ref = pd.read_csv(csv_ref)
    num = [c for c in ref.columns if c in df5.columns and pd.api.types.is_numeric_dtype(ref[c])]
    a = df5.set_index("image_name")[num].sort_index()
    b = ref.set_index("image_name")[num].sort_index()
    diff = float(np.nanmax(np.abs(a.values - b.values)))
    print(f"[chequeo] diferencia máxima entre prune_iters=5 recalculado y {csv_ref}: {diff:.3e}")
    if diff > 1e-6:
        print("[chequeo] ATENCIÓN: no coincide con tu CSV. Revisá q, variant y el dataset de entrada.")


def _summarize_features(df, p, features):
    """Filas (formato largo) con mediana por condición, n válido y d de Cohen."""
    rows = []
    for f in features:
        if f not in df.columns:
            print(f"[aviso] la característica {f} no está en el DataFrame de features; se omite.")
            continue
        c = df.loc[df["condition"] == CTRL, f].dropna()
        h = df.loc[df["condition"] == HPMC, f].dropna()
        rows.append(dict(
            prune=p, feature=f,
            origen="máscara" if f in MASK_FEATURES else "esqueleto",
            med_CTRL=c.median(), med_HPMC=h.median(),
            d=cohen_d(c, h),
            n_valid_CTRL=len(c), n_valid_HPMC=len(h),
        ))
    return rows


def _select_final_set(df):
    """Selección final sobre todas las imágenes (la misma que usa el modelo guardado)."""
    from final_binary_model import select_final_features_from_full_dataset
    with contextlib.redirect_stdout(io.StringIO()):  # la función imprime un reporte largo
        selected, _, _ = select_final_features_from_full_dataset(df)
    return list(selected)


def run_sensitivity(norm_dataset, prune_list=PRUNE_LIST, q=80, variant="closing_r3",
                    run_nested=True, csv_ref=None, features_finales=None,
                    out_csv="sensibilidad_poda.csv",
                    out_csv_features="sensibilidad_poda_features_finales.csv",
                    out_csv_conjuntos="sensibilidad_poda_conjuntos_finales.csv"):
    features_finales = list(features_finales) if features_finales else FEATURES_FINALES
    features_tabla = features_finales + [f for f in FEATURES_REFERENCIA if f not in features_finales]

    if run_nested:
        from nested_binary_evaluation import run_nested_leave_one_culture_out

    print("Calculando wavelet, máscara y esqueleto (una sola vez por imagen)...")
    cache = _precompute(norm_dataset, q=q, variant=variant)

    resumen, largo, conjuntos = [], [], []
    for p in prune_list:
        print(f"\n=== prune_iters = {p} ===")
        df = _features_df(cache, p)
        if p == 5 and csv_ref:
            _check_against_csv(df, csv_ref)

        largo.extend(_summarize_features(df, p, features_tabla))

        # --- conjunto final que resultaría con este valor de poda ---
        row = {"prune_iters": p}
        try:
            sel = _select_final_set(df)
            row["n_features_finales"] = len(sel)
            row["features_finales"] = ", ".join(sel)
            row["faltan_vs_modelo_guardado"] = ", ".join(sorted(set(features_finales) - set(sel)))
            row["sobran_vs_modelo_guardado"] = ", ".join(sorted(set(sel) - set(features_finales)))
            print(f"Conjunto final ({len(sel)}): {sel}")
            conjuntos.append(dict(prune_iters=p, n=len(sel), features=", ".join(sel),
                                  faltan=row["faltan_vs_modelo_guardado"],
                                  sobran=row["sobran_vs_modelo_guardado"]))
        except Exception as e:  # p. ej. final_binary_model no importable desde acá
            print(f"[aviso] no se pudo repetir la selección final con prune_iters={p}: {e!r}")

        # --- validación anidada (lenta) ---
        if run_nested:
            out = run_nested_leave_one_culture_out(df)
            gm = out["global_metrics"]
            row["AUC_global"] = gm["roc_auc"]
            row["F1_global"] = gm["f1"]
            row["accuracy_global"] = gm["accuracy"]
            row["modelo_por_fold"] = ", ".join(out["summary_df"]["selected_model"])
            row["variables_por_fold"] = " | ".join(out["summary_df"]["selected_features"])
        resumen.append(row)

    # ---- tabla larga: cambio porcentual de cada mediana respecto de no podar ----
    largo = pd.DataFrame(largo)
    if not largo.empty and (largo["prune"] == 0).any():
        base = largo[largo["prune"] == 0].set_index("feature")

        def _pct(valor, ref):
            return np.nan if (pd.isna(ref) or ref == 0) else 100 * (valor / ref - 1)

        for cond in ("CTRL", "HPMC"):
            col = f"med_{cond}"
            largo[f"{col}_pct_vs_sin_poda"] = [
                _pct(r[col], base.loc[r["feature"], col]) for _, r in largo.iterrows()]
    if not largo.empty:
        largo["abs_d_ge_0.8"] = largo["d"].abs() >= D_UMBRAL

    resumen = pd.DataFrame(resumen)
    sl = largo[largo["feature"] == "skeleton_length"] if not largo.empty else largo
    if not sl.empty:
        for cond in ("CTRL", "HPMC"):
            m = sl.set_index("prune")[f"med_{cond}_pct_vs_sin_poda"]
            resumen[f"skeleton_length_{cond}_pct_vs_sin_poda"] = resumen["prune_iters"].map(m)

    conjuntos = pd.DataFrame(conjuntos)
    resumen.to_csv(out_csv, index=False)
    largo.to_csv(out_csv_features, index=False)
    conjuntos.to_csv(out_csv_conjuntos, index=False)
    print(f"\nGuardado en {out_csv}, {out_csv_features} y {out_csv_conjuntos}")
    return {"resumen": resumen, "features_finales": largo, "conjuntos_finales": conjuntos}


#EVALUADRO DE PROGRESIÓN
"""
Sensibilidad del evaluador de progresión al número de iteraciones de poda (prune_iters).

Este script NO reimplementa nada de tu evaluador: llama a tus funciones de temporal/pipeline_temp.py
exactamente como las usás en main_temp.ipynb. Lo único que cambia es qué valor de prune_iters usa el
procesamiento de imágenes y de dónde se lee la metadata, y eso se hace con un "parche temporal" que
se activa solo dentro de cada corrida y se deshace al terminar (ver _entorno_poda):

  * pipeline_temp.process_single_image  -> misma función de pipeline.py, pero con prune_iters=P
    (tus funciones anchored tienen prune_iters=5 escrito adentro) y con memoria para no procesar dos
    veces la misma imagen con el mismo P;
  * pipeline.apply_filters              -> memoria del filtrado wavelet (es lo más lento y no depende
    de la poda), calculado una sola vez por imagen;
  * pipeline_temp.load_temporal_score_model -> devuelve la metadata calibrada con ese P, así
    predict_temporal_progression_score_anchored usa los pesos y umbrales de ese P y NO el archivo
    temporal_score_metadata.joblib de la app (que no se toca ni se sobrescribe).

Dentro de cada corrida se ejecuta, en este orden, lo mismo que hiciste en el notebook:
  1) build_temporal_feature_dataset(temporal_green_norm, prune_iters=P)                  (celda 13)
  2) build_temporal_score_metadata(df)  (= save_temporal_score_model sin guardar)         (celda 42)
  3) Reference-Free: _compute_absolute_inflammatory_score por imagen y temporal_pairwise_*  (celdas 57-60)
     y sobre la tabla cultivo x tiempo (celdas 31-35 y 55)
  4) Experiment-Anchored: build_anchored_relative_scores_dataframe_multi_ref,
     validate_anchored_monotonicity_all_multi_ref, summarize_relative_score_by_group y
     validate_groupwise_monotonicity                                                    (celdas 101-110)
  5) Leave-one-culture-out: build_anchored_relative_scores_dataframe_multi_ref_loo y
     validate_anchored_monotonicity_all_multi_ref_loo                                   (celdas 116-119)

Uso (en el notebook, con los objetos que ya tenés):

    import sensibilidad_poda_progresion as spp
    out = spp.run_sensitivity_progression(
        temporal_green_norm,            # el mismo que le pasaste a build_temporal_feature_dataset
        temporal_dataset,               # el dataset original (sin normalizar) que usan las funciones anchored
        metadata_ref="temporal_score_metadata.joblib",
    )
    print(out["resumen"].T.to_string())

Con prune_iters=5 el script imprime sus números junto a los de la tesis y compara con tu metadata
guardada; además recalcula un score con la función ORIGINAL sin parches y lo compara con el del script.

Memoria: guarda el wavelet de cada imagen (~1 GB para 64 imágenes de 1360x1024 en float64).
Guarda tres CSV:
  - sensibilidad_poda_progresion.csv           (una fila por prune_iters)
  - sensibilidad_poda_progresion_pares.csv     (exactitud por par de tiempos, por modo)
  - sensibilidad_poda_progresion_scores.csv    (score de cada imagen en cada modo y valor de poda)
"""
import contextlib
import copy
import hashlib

import numpy as np
import pandas as pd

import pipeline as pipeline_root
from temporal import pipeline_temp as pt

PRUNE_LIST = (0, 3, 5, 7, 10)
TIME_ORDER = ["0HS", "24HS", "48HS", "72HS"]
FEATS = ("median_thickness", "median_segment_length")
SCORE_FREE = "absolute_inflammatory_score"
SCORE_REL = "relative_inflammatory_score"

# Valores publicados en la tesis para prune_iters=5 (secciones 5.2.1 a 5.2.4): (valor, tolerancia)
TESIS = {
    "peso_median_thickness": (0.4974, 0.0006),
    "peso_median_segment_length": (0.5026, 0.0006),
    "ref_ctrl_median_thickness": (14.64, 0.01),
    "ref_inflam_median_thickness": (10.25, 0.01),
    "ref_ctrl_median_segment_length": (13.69, 0.01),
    "ref_inflam_median_segment_length": (15.81, 0.01),
    "umbral_1": (16.81, 0.01), "umbral_2": (48.56, 0.01), "umbral_3": (81.75, 0.01),
    "free_pares_correctos": (1403, 0), "free_empates": (21, 0),
    "free_tabla_resumida_correctos": (96, 0),
    "anch_pares_correctos": (1413, 0), "anch_empates": (5, 0),
    "anch_series_monotonas": (11, 0),
    "loo_pares_correctos": (1412, 0), "loo_series_monotonas": (11, 0),
    "loo_misma_categoria": (60, 0),
}


# ----------------------------------------------------------------------------------------------
# Parche temporal (se activa solo dentro de cada corrida)
# ----------------------------------------------------------------------------------------------
def _key(img):
    a = np.ascontiguousarray(img)
    return (a.shape, a.dtype.str, hashlib.sha1(a.tobytes()).hexdigest())


@contextlib.contextmanager
def _entorno_poda(p_forzada, caches, state):
    """Fuerza prune_iters=p_forzada en tus funciones y usa state['meta'] como metadata temporal."""
    orig_proc = pipeline_root.process_single_image          # la función real de pipeline.py
    orig_filters = pipeline_root.apply_filters
    prev_proc, prev_load = pt.process_single_image, pt.load_temporal_score_model
    prev_filters = pipeline_root.apply_filters

    def filtros_con_memoria(img, *args, **kwargs):
        if args or kwargs:
            return orig_filters(img, *args, **kwargs)
        k = _key(img)
        if k not in caches["wavelet"]:
            caches["wavelet"][k] = orig_filters(img)[2]
        return None, None, caches["wavelet"][k]               # process_single_image solo usa el 3.º

    def proceso_con_prune_forzado(img, q=80, variant="closing_r3", prune_iters=5):
        k = (_key(img), q, variant, p_forzada)
        if k not in caches["features"]:
            res = orig_proc(img, q=q, variant=variant, prune_iters=p_forzada)
            caches["features"][k] = res["features"]
        return {"wavelet": None, "mask": None, "skeleton": None,
                "features": copy.copy(caches["features"][k])}

    pipeline_root.apply_filters = filtros_con_memoria
    pt.process_single_image = proceso_con_prune_forzado
    pt.load_temporal_score_model = lambda metadata_path=None: state["meta"]
    try:
        yield
    finally:
        pipeline_root.apply_filters = prev_filters
        pt.process_single_image = prev_proc
        pt.load_temporal_score_model = prev_load


# ----------------------------------------------------------------------------------------------
# Una corrida completa para un valor de poda (llama a tus funciones, sin reimplementarlas)
# ----------------------------------------------------------------------------------------------
def _score_fila(row, meta):
    a, w = meta["population_anchors"], meta["weights"]
    return pt._compute_absolute_inflammatory_score(
        thickness=row["median_thickness"], segment_length=row["median_segment_length"],
        ctrl_thickness=a["ctrl_ref"]["median_thickness"], inflam_thickness=a["inflam_ref"]["median_thickness"],
        ctrl_segment=a["ctrl_ref"]["median_segment_length"], inflam_segment=a["inflam_ref"]["median_segment_length"],
        w_thickness=w["median_thickness"], w_segment=w["median_segment_length"])


def _correr_un_valor(norm_dataset, raw_dataset, p, q, variant, run_loo, caches):
    state = {"meta": None}
    with _entorno_poda(p, caches, state):
        # 1-2) features y calibración (celdas 13 y 42; la metadata NO se guarda)
        df_t = pt.build_temporal_feature_dataset(norm_dataset, q=q, variant=variant, prune_iters=p)
        meta = pt.build_temporal_score_metadata(df_t)
        state["meta"] = meta

        # 3) Reference-Free (celdas 57-60 y 31-35, 55)
        df_abs = df_t.copy()
        df_abs[SCORE_FREE] = df_abs.apply(lambda r: _score_fila(r, meta), axis=1)
        free_ind = pt.temporal_pairwise_order_accuracy(df_abs, score_col=SCORE_FREE, time_col="time_h")
        free_por_par = pt.temporal_pairwise_order_by_timepair(df_abs, score_col=SCORE_FREE)
        df_grp = pt.build_group_time_summary(df_t)
        df_grp[SCORE_FREE] = df_grp.apply(lambda r: _score_fila(r, meta), axis=1)
        free_grp = pt.temporal_pairwise_order_accuracy(df_grp, score_col=SCORE_FREE, time_col="time_h")

        # 4) Experiment-Anchored (celdas 101-110)
        df_rel = pt.build_anchored_relative_scores_dataframe_multi_ref(raw_dataset)
        anch_ind = pt.temporal_pairwise_order_accuracy(df_rel, score_col=SCORE_REL)
        anch_por_par = pt.temporal_pairwise_order_by_timepair(df_rel, score_col=SCORE_REL)
        _, anch_mono = pt.validate_anchored_monotonicity_all_multi_ref(raw_dataset)
        resumen_grp = pt.summarize_relative_score_by_group(df_rel)
        _, anch_grp_mono = pt.validate_groupwise_monotonicity(resumen_grp)

        # 5) Leave-one-culture-out (celdas 116-119)
        loo = None
        if run_loo:
            df_loo, meta_loo = pt.build_anchored_relative_scores_dataframe_multi_ref_loo(raw_dataset, df_t)
            _, loo_mono = pt.validate_anchored_monotonicity_all_multi_ref_loo(df_loo)
            loo = dict(df=df_loo, meta=meta_loo, mono=loo_mono,
                       ind=pt.temporal_pairwise_order_accuracy(df_loo, score_col=SCORE_REL),
                       por_par=pt.temporal_pairwise_order_by_timepair(df_loo, score_col=SCORE_REL))

    return dict(df_t=df_t, meta=meta, df_abs=df_abs, free_ind=free_ind, free_por_par=free_por_par,
                free_grp=free_grp, df_rel=df_rel, anch_ind=anch_ind, anch_por_par=anch_por_par,
                anch_mono=anch_mono, anch_grp_mono=anch_grp_mono, loo=loo)


# ----------------------------------------------------------------------------------------------
# Resumen, chequeos y concordancia
# ----------------------------------------------------------------------------------------------
def _fila_resumen(p, r):
    meta, df_t = r["meta"], r["df_t"]
    row = {"prune_iters": p}
    for f in FEATS:
        row[f"{f}_rho_cultivo_tiempo"] = meta["rho_group_level"][f]["rho"]
        row[f"{f}_peso"] = meta["weights"][f]
        row[f"{f}_ref_0h"] = meta["population_anchors"]["ctrl_ref"][f]
        row[f"{f}_ref_72h"] = meta["population_anchors"]["inflam_ref"][f]
        for tl in TIME_ORDER:
            row[f"{f}_media_{tl}"] = float(df_t.loc[df_t["time_label"] == tl, f].mean())
    th = meta["thresholds"]
    rec = meta["recovery_thresholds"]
    row.update(umbral_1=th[0], umbral_2=th[1], umbral_3=th[2],
               umbral_rec_1=rec[0], umbral_rec_2=rec[1], umbral_rec_3=rec[2])
    for nombre, d in (("free", r["free_ind"]), ("anch", r["anch_ind"])):
        row[f"{nombre}_pares_total"] = d["total_pairs"]
        row[f"{nombre}_pares_correctos"] = d["correct_pairs"]
        row[f"{nombre}_empates"] = d["ties"]
        row[f"{nombre}_exactitud"] = d["pairwise_order_accuracy"]
    row["free_tabla_resumida_correctos"] = r["free_grp"]["correct_pairs"]
    row["free_tabla_resumida_total"] = r["free_grp"]["total_pairs"]
    row["anch_series_monotonas"] = r["anch_mono"]["monotonic_trajectories"]
    row["series_total"] = r["anch_mono"]["total_trajectories"]
    row["anch_cultivos_monotonos"] = r["anch_grp_mono"]["monotonic_groups"]
    if r["loo"] is not None:
        loo = r["loo"]
        row["loo_pares_correctos"] = loo["ind"]["correct_pairs"]
        row["loo_empates"] = loo["ind"]["ties"]
        row["loo_series_monotonas"] = loo["mono"]["monotonic_trajectories"]
        pesos_t = [m["weights"]["median_thickness"] for m in loo["meta"].values()]
        row["loo_peso_thickness_min"], row["loo_peso_thickness_max"] = min(pesos_t), max(pesos_t)
        for i in range(3):
            v = [m["recovery_thresholds"][i] for m in loo["meta"].values()]
            row[f"loo_umbral_rec_{i+1}_min"], row[f"loo_umbral_rec_{i+1}_max"] = min(v), max(v)
    return row


def _tabla_scores(p, r):
    llaves = ["group", "time_label", "image_name"]
    a = r["df_abs"][llaves + [SCORE_FREE]].rename(columns={SCORE_FREE: "score_free"})
    a["categoria_free"] = [pt.get_progression_category(s, r["meta"]["thresholds"]) for s in a["score_free"]]
    b = r["df_rel"][llaves + ["image_idx", SCORE_REL, "recovery_category"]].rename(
        columns={SCORE_REL: "score_anchored", "recovery_category": "categoria_anchored"})
    out = a.merge(b, on=llaves, how="outer")
    if r["loo"] is not None:
        c = r["loo"]["df"][llaves + [SCORE_REL, "recovery_category"]].rename(
            columns={SCORE_REL: "score_loo", "recovery_category": "categoria_loo"})
        out = out.merge(c, on=llaves, how="left")
    out.insert(0, "prune_iters", p)
    return out


def _chequeo_tesis(row5):
    print("\n[chequeo] prune_iters=5 frente a los valores de la tesis (secciones 5.2.1 a 5.2.4):")
    obs = {
        "peso_median_thickness": row5["median_thickness_peso"],
        "peso_median_segment_length": row5["median_segment_length_peso"],
        "ref_ctrl_median_thickness": row5["median_thickness_ref_0h"],
        "ref_inflam_median_thickness": row5["median_thickness_ref_72h"],
        "ref_ctrl_median_segment_length": row5["median_segment_length_ref_0h"],
        "ref_inflam_median_segment_length": row5["median_segment_length_ref_72h"],
        "umbral_1": row5["umbral_1"], "umbral_2": row5["umbral_2"], "umbral_3": row5["umbral_3"],
        "free_pares_correctos": row5["free_pares_correctos"], "free_empates": row5["free_empates"],
        "free_tabla_resumida_correctos": row5["free_tabla_resumida_correctos"],
        "anch_pares_correctos": row5["anch_pares_correctos"], "anch_empates": row5["anch_empates"],
        "anch_series_monotonas": row5["anch_series_monotonas"],
        "loo_pares_correctos": row5.get("loo_pares_correctos"),
        "loo_series_monotonas": row5.get("loo_series_monotonas"),
        "loo_misma_categoria": row5.get("loo_misma_categoria"),
    }
    mal = 0
    for k, (esperado, tol) in TESIS.items():
        v = obs.get(k)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            print(f"   -- {k:34s} (no calculado)")
            continue
        ok = abs(float(v) - esperado) <= tol + 1e-12
        mal += (not ok)
        print(f"   {'OK ' if ok else 'DIF'} {k:34s} script={float(v):12.4f}   tesis={esperado}")
    print("[chequeo] " + ("todo coincide con la tesis." if mal == 0 else
          f"{mal} valor(es) difieren de la tesis: NO uses las otras filas hasta entender por qué."))


def _chequeo_metadata(meta5, metadata_ref):
    import joblib
    ref = joblib.load(metadata_ref)
    dif = []
    for f in FEATS:
        dif.append(abs(meta5["weights"][f] - ref["weights"][f]))
        dif.append(abs(meta5["population_anchors"]["ctrl_ref"][f] - ref["population_anchors"]["ctrl_ref"][f]))
        dif.append(abs(meta5["population_anchors"]["inflam_ref"][f] - ref["population_anchors"]["inflam_ref"][f]))
    dif += [abs(a - b) for a, b in zip(meta5["thresholds"], ref["thresholds"])]
    dif += [abs(a - b) for a, b in zip(meta5["recovery_thresholds"], ref["recovery_thresholds"])]
    d = max(dif)
    print(f"[chequeo] metadata recalculada (prune_iters=5) vs {metadata_ref}: diferencia máxima {d:.2e}"
          + ("  -> idéntica." if d < 1e-9 else "  -> ATENCIÓN: difiere de la metadata guardada."))


def _chequeo_funcion_original(r5, raw_dataset, metadata_ref):
    """Un score anchored con tu función original, SIN parches, contra el del script."""
    g = sorted(raw_dataset["0HS"].keys())[0]
    ctrl = [it["image"] for it in raw_dataset["0HS"][g]]
    infl = [it["image"] for it in raw_dataset["72HS"][g]]
    res = pt.predict_temporal_progression_score_anchored(
        img_new=raw_dataset["24HS"][g][0]["image"], ctrl_images=ctrl, inflam_images=infl,
        metadata_path=metadata_ref)
    df = r5["df_rel"]
    esperado = float(df[(df["group"] == g) & (df["image_idx"] == 0) & (df["time_label"] == "24HS")]
                     [SCORE_REL].iloc[0])
    d = abs(res["inflammatory_score"] - esperado)
    print(f"[chequeo] score anchored ({g}, imagen 0, 24 h): función original {res['inflammatory_score']:.6f} "
          f"vs script {esperado:.6f}  (dif {d:.1e})" + ("  -> idéntico." if d < 1e-9 else "  -> ATENCIÓN."))


def run_sensitivity_progression(norm_dataset, raw_dataset, prune_list=PRUNE_LIST, q=80, variant="closing_r3",
                                run_loo=True, metadata_ref=None,
                                out_csv="sensibilidad_poda_progresion.csv",
                                out_csv_pares="sensibilidad_poda_progresion_pares.csv",
                                out_csv_scores="sensibilidad_poda_progresion_scores.csv"):
    caches = {"wavelet": {}, "features": {}}
    corridas, filas, pares = {}, [], []

    for p in prune_list:
        print(f"\n=== prune_iters = {p} ===")
        r = _correr_un_valor(norm_dataset, raw_dataset, p, q, variant, run_loo, caches)
        corridas[p] = r
        filas.append(_fila_resumen(p, r))
        for modo, tabla in (("Reference-Free", r["free_por_par"]), ("Experiment-Anchored", r["anch_por_par"]),
                            ("Leave-one-culture-out", r["loo"]["por_par"] if r["loo"] else None)):
            if tabla is None:
                continue
            t = tabla.copy()
            t.insert(0, "modo", modo)
            t.insert(0, "prune_iters", p)
            pares.append(t)

    resumen = pd.DataFrame(filas)
    scores = pd.concat([_tabla_scores(p, corridas[p]) for p in prune_list], ignore_index=True)

    # --- concordancia de cada prune_iters con prune_iters=5
    if 5 in corridas:
        ref = scores[scores["prune_iters"] == 5].set_index(["group", "time_label", "image_name"])
        for idx, p in enumerate(prune_list):
            cur = scores[scores["prune_iters"] == p].set_index(["group", "time_label", "image_name"]).loc[ref.index]
            for nombre, col in (("free", "score_free"), ("anch", "score_anchored"), ("loo", "score_loo")):
                if col in cur and cur[col].notna().all() and ref[col].notna().all():
                    resumen.loc[idx, f"{nombre}_rho_score_vs_p5"] = float(
                        cur[col].rank().corr(ref[col].rank()))                    # Spearman
                    cat = {"free": "categoria_free", "anch": "categoria_anchored", "loo": "categoria_loo"}[nombre]
                    resumen.loc[idx, f"{nombre}_misma_categoria_vs_p5_pct"] = 100 * float(
                        (cur[cat].to_numpy() == ref[cat].to_numpy()).mean())
        # LOO frente a la calibración completa (60/64 en la tesis), con el mismo prune_iters
        for idx, p in enumerate(prune_list):
            s = scores[scores["prune_iters"] == p]
            if "categoria_loo" in s and s["categoria_loo"].notna().all():
                resumen.loc[idx, "loo_misma_categoria"] = int((s["categoria_anchored"] == s["categoria_loo"]).sum())

        row5 = resumen.loc[resumen["prune_iters"] == 5].iloc[0].to_dict()
        _chequeo_tesis(row5)
        if metadata_ref:
            _chequeo_metadata(corridas[5]["meta"], metadata_ref)
            _chequeo_funcion_original(corridas[5], raw_dataset, metadata_ref)

    resumen.to_csv(out_csv, index=False)
    pares = pd.concat(pares, ignore_index=True)
    pares.to_csv(out_csv_pares, index=False)
    scores.to_csv(out_csv_scores, index=False)
    print(f"\nGuardado en {out_csv}, {out_csv_pares} y {out_csv_scores}")
    return {"resumen": resumen, "pares_por_tiempo": pares, "scores": scores}