"""
Alternativa a la poda iterativa: eliminar SOLO ramas terminales (extremo -> unión) más cortas que L
y dejar intactas todas las demás (la propuesta de tu tutor), comparada con la poda actual.

Qué hace este archivo
---------------------
1) DIAGNÓSTICO (run_diagnostico). Sobre el esqueleto SIN podar de cada imagen mide:
     * la distribución de longitudes de las ramas terminales (extremo -> unión);
     * la distribución de longitudes de los segmentos aislados (sin uniones);
     * qué le pasa a cada píxel que borra prune_iters=5: si era una rama espuria corta,
       un fragmento aislado pequeño, o la punta (hasta 5 px) de una rama larga / un segmento
       aislado largo. Esto cuantifica exactamente las dos preocupaciones de tu tutor.
   Guarda: diag_ramas_hist.csv, diag_poda_descomposicion.csv, diag_poda_resumen.csv, diag_ramas_poda.png

2) ALTERNATIVA PARA EL CLASIFICADOR (run_alternativa_clasificador). Reemplaza la poda por
   prune_spurs_by_length(skel, L) y repite lo de sensibilidad_poda.py: mediana/d de Cohen de las 9
   características, cambio de skeleton_length y conjunto final de características (y opcionalmente la
   validación anidada). Guarda: alt_longitud_clasificador.csv, alt_longitud_clasificador_features.csv,
   alt_longitud_clasificador_conjuntos.csv

3) ALTERNATIVA PARA EL EVALUADOR DE PROGRESIÓN (run_alternativa_progresion). Llama a TUS funciones de
   pipeline_temp.py (igual que sensibilidad_poda.py) y solo cambia la función de poda.
   Guarda: alt_longitud_progresion.csv, alt_longitud_progresion_pares.csv, alt_longitud_progresion_scores.csv

run_todo(...) corre los tres pasos compartiendo el filtrado wavelet.

Definiciones (las MISMAS que usa tu features.py)
------------------------------------------------
  * extremo = píxel del esqueleto con 1 vecino (neigh == 2 en la convolución 3x3);
  * unión   = píxel con >= 3 vecinos (neigh >= 4); se usa get_endpoints_and_junctions de src/features.py;
  * longitud de una rama = cantidad de píxeles entre el extremo y la unión, SIN contar la unión. Es
    exactamente el "largo de segmento" que usa extract_skeleton_segment_features, así que L está en las
    mismas unidades que median_segment_length;
  * rama terminal = camino extremo -> unión; segmento aislado = camino extremo -> extremo sin uniones.

Equivalencias útiles para interpretar
-------------------------------------
  * La poda iterativa con p pasos borra las ramas terminales de longitud <= p y los segmentos aislados de
    longitud <= 2p, y además acorta p píxeles la punta de TODAS las demás ramas.
  * L = p + 1 borra las mismas ramas espurias (longitud <= p) pero NO acorta las ramas largas.
  * "L6+aisl11" borra las ramas terminales <= 5 px y los segmentos aislados <= 10 px (lo mismo que
    prune_iters=5 filtra) pero sin acortar nada más. Comparar iter5 vs L6+aisl11 aísla el efecto de
    "acortar las puntas"; comparar L6 vs L6+aisl11 aísla el efecto de borrar aislados pequeños.

Uso en el notebook:

    import sensibilidad_poda_longitud as spl
    d = spl.run_diagnostico(green_norm_dataset, temporal_green_norm)                 # rápido
    c = spl.run_alternativa_clasificador(green_norm_dataset, csv_ref="features_dataset.csv")
    p = spl.run_alternativa_progresion(temporal_green_norm, temporal_dataset,
                                       metadata_ref="temporal_score_metadata.joblib")
    # o los tres juntos:  spl.run_todo(green_norm_dataset, temporal_green_norm, temporal_dataset,
    #                                  csv_ref="features_dataset.csv",
    #                                  metadata_ref="temporal_score_metadata.joblib")

Requiere en la misma carpeta: sensibilidad_poda.py (los que ya usaste).
"""
import contextlib
import copy

import numpy as np
import pandas as pd

from src.features import get_endpoints_and_junctions
from src.preprocessing import prune_skeleton  # la poda iterativa ORIGINAL (para descomponer lo que borra)

# Variantes que se comparan. "iter" = tu poda iterativa; "longitud" = ramas terminales < L (y, si
# aislados_min > 0, también segmentos aislados con menos de aislados_min píxeles).
SPECS = [
    dict(nombre="iter0", modo="iter", p=0),
    dict(nombre="iter5", modo="iter", p=5),
    dict(nombre="L6", modo="longitud", L=6, aislados_min=0),
    dict(nombre="L11", modo="longitud", L=11, aislados_min=0),
    dict(nombre="L16", modo="longitud", L=16, aislados_min=0),
    dict(nombre="L6+aisl11", modo="longitud", L=6, aislados_min=11),
]
P_DIAG = 5                     # poda cuyo efecto se descompone en el diagnóstico
TOPE_HIST = 60                 # longitudes >= TOPE_HIST se agrupan en la última barra
CTRL, HPMC = "CTRL", "HPMC"
_VEC = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


# ----------------------------------------------------------------------------------------------
# Núcleo: recorrido de ramas
# ----------------------------------------------------------------------------------------------
def _recorrer(S, U, r0, c0, limite):
    """Camina desde un extremo (r0, c0) por el esqueleto S (con borde de 1 px); U marca las uniones.

    Devuelve (tipo, camino):
      "terminal" -> llegó a una unión (camino = píxeles de la rama, sin la unión);
      "aislada"  -> terminó en otro extremo sin pasar por ninguna unión;
      "larga"    -> alcanzó `limite` píxeles sin definirse (solo si limite no es None).
    Un píxel que no es unión tiene como máximo 2 vecinos (el anterior y el siguiente), por eso el
    recorrido es único.
    """
    camino = [(r0, c0)]
    vistos = {(r0, c0)}
    r, c = r0, c0
    while True:
        sig = None
        toca_union = False
        for dr, dc in _VEC:
            rr, cc = r + dr, c + dc
            if not S[rr, cc] or (rr, cc) in vistos:
                continue
            if U[rr, cc]:
                toca_union = True
            else:
                sig = (rr, cc)
        if toca_union:
            return "terminal", camino
        if sig is None:
            return "aislada", camino
        if limite is not None and len(camino) >= limite:
            return "larga", camino
        camino.append(sig)
        vistos.add(sig)
        r, c = sig


def _preparar(skel):
    skel = np.asarray(skel).astype(bool)
    ext, uni = get_endpoints_and_junctions(skel)
    S = np.pad(skel, 1)
    U = np.pad(uni, 1)
    extremos = np.argwhere(np.pad(ext, 1)).tolist()
    return skel, S, U, extremos


def prune_spurs_by_length(skel, L, min_isolated=0):
    """Elimina las ramas terminales (extremo -> unión) con MENOS de L píxeles.

    Todo lo demás queda intacto: las ramas terminales de longitud >= L (sin acortarlas), los lazos, las
    ramas entre uniones y los segmentos aislados (salvo que min_isolated > 0: entonces se borran los
    segmentos aislados con menos de min_isolated píxeles). Las ramas se detectan una sola vez sobre el
    esqueleto de entrada y se borran a la vez.
    """
    skel = np.asarray(skel).astype(bool)
    if L <= 1 and min_isolated <= 0:
        return skel.copy()
    skel, S, U, extremos = _preparar(skel)
    out = S.copy()
    limite = max(int(L), int(min_isolated))
    consumidos = set()
    for r, c in extremos:
        if (r, c) in consumidos:
            continue
        tipo, cam = _recorrer(S, U, r, c, limite)
        n = len(cam)
        borrar = False
        if tipo == "terminal":
            borrar = n < L
        elif tipo == "aislada":
            consumidos.add(cam[-1])          # el otro extremo del mismo segmento
            borrar = n < min_isolated
        if borrar:
            for rr, cc in cam:
                out[rr, cc] = False
    return out[1:-1, 1:-1]


def ramas(skel):
    """Todas las ramas terminales y segmentos aislados del esqueleto (sin límite de longitud).

    Devuelve dict con "terminales" y "aisladas": listas de caminos [(r, c), ...] en coordenadas de skel.
    """
    skel, S, U, extremos = _preparar(skel)
    term, aisl, consumidos = [], [], set()
    for r, c in extremos:
        if (r, c) in consumidos:
            continue
        tipo, cam = _recorrer(S, U, r, c, None)
        cam = [(rr - 1, cc - 1) for rr, cc in cam]
        if tipo == "terminal":
            term.append(cam)
        else:
            consumidos.add((cam[-1][0] + 1, cam[-1][1] + 1))
            aisl.append(cam)
    return {"terminales": term, "aisladas": aisl}


def descomponer_poda(skel, p=P_DIAG):
    """Descompone los píxeles que borra la poda iterativa de p pasos según a qué pertenecían.

    Categorías (cantidad de píxeles borrados):
      espuria_terminal   rama terminal de longitud <= p (desaparece entera)
      punta_terminal     rama terminal de longitud  > p (se le acortan hasta p px: ACORTAMIENTO DE RAMAS REALES)
      aislado_pequeno    segmento aislado de longitud <= 2p (desaparece entero)
      punta_aislado      segmento aislado de longitud  > 2p (se le acortan los dos extremos)
      otros              píxeles que no pertenecían a ninguna de esas ramas (p. ej. una unión que
                         queda como extremo cuando se borran todas sus ramas)
    """
    skel = np.asarray(skel).astype(bool)
    borrados = skel & ~prune_skeleton(skel, prune_iters=p)
    rm = ramas(skel)
    cuenta = dict(espuria_terminal=0, punta_terminal=0, aislado_pequeno=0, punta_aislado=0)
    asignados = 0
    n_term_corta = n_term_larga = n_aisl_peq = n_aisl_larg = 0
    for cam in rm["terminales"]:
        k = int(sum(borrados[r, c] for r, c in cam))
        if len(cam) <= p:
            cuenta["espuria_terminal"] += k
            n_term_corta += 1
        else:
            cuenta["punta_terminal"] += k
            n_term_larga += 1
        asignados += k
    for cam in rm["aisladas"]:
        k = int(sum(borrados[r, c] for r, c in cam))
        if len(cam) <= 2 * p:
            cuenta["aislado_pequeno"] += k
            n_aisl_peq += 1
        else:
            cuenta["punta_aislado"] += k
            n_aisl_larg += 1
        asignados += k
    cuenta["otros"] = int(borrados.sum()) - asignados
    cuenta.update(skeleton_px=int(skel.sum()), borrados_px=int(borrados.sum()),
                  n_terminales=len(rm["terminales"]), n_terminales_cortas=n_term_corta,
                  n_terminales_largas=n_term_larga, n_aislados=len(rm["aisladas"]),
                  n_aislados_pequenos=n_aisl_peq, n_aislados_largos=n_aisl_larg)
    return cuenta, rm


# ----------------------------------------------------------------------------------------------
# Parche temporal de la función de poda
# ----------------------------------------------------------------------------------------------
def _crear_podador(spec):
    """Devuelve la función de poda de la variante, o None si es la poda iterativa original."""
    if spec["modo"] == "iter":
        return None
    L, m = int(spec["L"]), int(spec.get("aislados_min", 0))

    def podador(skel, prune_iters=None):          # prune_iters se ignora a propósito
        return prune_spurs_by_length(skel, L, m)
    return podador


@contextlib.contextmanager
def _parchar(modulo, nombre, nueva):
    if nueva is None:
        yield
        return
    original = getattr(modulo, nombre)
    setattr(modulo, nombre, nueva)
    try:
        yield
    finally:
        setattr(modulo, nombre, original)


# ----------------------------------------------------------------------------------------------
# 1) Diagnóstico
# ----------------------------------------------------------------------------------------------
def _precompute_temporal(norm_dataset, q, variant, wavelet_cache):
    """wavelet -> máscara -> esqueleto sin podar de cada imagen del dataset temporal
    (los mismos pasos 1-3 de process_single_image). El wavelet se guarda en wavelet_cache."""
    import hashlib
    from src.preprocessing import apply_filters, build_wavelet_mask_candidate, get_skeleton
    out = []
    for time_label, groups in norm_dataset.items():
        for group, imgs in groups.items():
            for item in imgs:
                a = np.ascontiguousarray(item["image"])
                k = (a.shape, a.dtype.str, hashlib.sha1(a.tobytes()).hexdigest())
                if k not in wavelet_cache:
                    wavelet_cache[k] = apply_filters(item["image"])[2]
                mask = build_wavelet_mask_candidate(wavelet_cache[k], q=q, variant=variant)
                out.append(dict(dataset="temporal", grupo=time_label, cultivo=group,
                                image_name=item["name"], skel0=get_skeleton(mask)))
    return out


def _filas_diagnostico(items, p):
    desc, hist = [], []
    for it in items:
        cuenta, rm = descomponer_poda(it["skel0"], p=p)
        base = dict(dataset=it["dataset"], grupo=it["grupo"], cultivo=it["cultivo"], image_name=it["image_name"])
        desc.append({**base, **cuenta})
        for tipo, lista in (("terminal", rm["terminales"]), ("aislada", rm["aisladas"])):
            largos = np.minimum([len(c) for c in lista], TOPE_HIST) if lista else np.array([], int)
            vals, cnt = np.unique(largos, return_counts=True)
            for v, n in zip(vals, cnt):
                hist.append({**base, "tipo": tipo, "longitud": int(v), "n": int(n)})
    return desc, hist


def _resumen_diagnostico(desc, p):
    d = desc.copy()
    cats = ["espuria_terminal", "punta_terminal", "aislado_pequeno", "punta_aislado", "otros"]
    g = d.groupby(["dataset", "grupo"])
    s = g[["skeleton_px", "borrados_px"] + cats + ["n_terminales", "n_terminales_cortas", "n_terminales_largas",
                                                   "n_aislados", "n_aislados_pequenos", "n_aislados_largos"]].sum()
    s["n_imagenes"] = g.size()
    s["pct_esqueleto_borrado"] = 100 * s["borrados_px"] / s["skeleton_px"]
    for c in cats:
        s[f"pct_de_lo_borrado_{c}"] = 100 * s[c] / s["borrados_px"].replace(0, np.nan)
    s["pct_ramas_terminales_cortas"] = 100 * s["n_terminales_cortas"] / s["n_terminales"].replace(0, np.nan)
    s["pct_aislados_pequenos"] = 100 * s["n_aislados_pequenos"] / s["n_aislados"].replace(0, np.nan)
    s.insert(0, "p_diagnostico", p)
    return s.reset_index()


def _figura_diagnostico(hist, resumen, p, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cats = ["espuria_terminal", "punta_terminal", "aislado_pequeno", "punta_aislado", "otros"]
    nombres = {"espuria_terminal": f"ramas terminales ≤{p} px (espurias)",
               "punta_terminal": f"punta (hasta {p} px) de ramas terminales largas",
               "aislado_pequeno": f"segmentos aislados ≤{2*p} px",
               "punta_aislado": "puntas de segmentos aislados largos", "otros": "otros"}
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
    t = hist[hist["tipo"] == "terminal"]
    for (ds, gr), h in t.groupby(["dataset", "grupo"]):
        v = h.groupby("longitud")["n"].sum().reindex(range(1, TOPE_HIST + 1), fill_value=0)
        ax[0].plot(v.index, v.cumsum() / v.sum(), label=f"{ds}: {gr}")
    for x in (p, 2 * p):
        ax[0].axvline(x, ls="--", lw=0.8, color="gray")
    ax[0].set(xlabel="longitud de la rama terminal (px)", ylabel="fracción acumulada de ramas terminales",
              title="Longitud de ramas terminales (esqueleto sin podar)", xlim=(1, 40))
    ax[0].legend(fontsize=7)
    etiquetas = [f"{r.dataset}\n{r.grupo}" for r in resumen.itertuples()]
    base = np.zeros(len(resumen))
    for c in cats:
        v = resumen[f"pct_de_lo_borrado_{c}"].fillna(0).to_numpy()
        ax[1].bar(range(len(resumen)), v, bottom=base, label=nombres[c])
        base += v
    ax[1].set_xticks(range(len(resumen)))
    ax[1].set_xticklabels(etiquetas, fontsize=7)
    ax[1].set(ylabel=f"% de los píxeles borrados por prune_iters={p}", title=f"¿Qué borra prune_iters={p}?")
    ax[1].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def run_diagnostico(norm_dataset_clasif=None, norm_dataset_temporal=None, p=P_DIAG, q=80, variant="closing_r3",
                    wavelet_cache=None, out_hist="diag_ramas_hist.csv",
                    out_desc="diag_poda_descomposicion.csv", out_resumen="diag_poda_resumen.csv",
                    out_png="diag_ramas_poda.png"):
    """Mide ramas terminales, segmentos aislados y qué borra la poda iterativa de p pasos."""
    wavelet_cache = {} if wavelet_cache is None else wavelet_cache
    items = []
    if norm_dataset_clasif is not None:
        import sensibilidad_poda as sp
        for c in sp._precompute(norm_dataset_clasif, q=q, variant=variant):
            items.append(dict(dataset="clasificador", grupo=c["condition"], cultivo=c["folder_group"],
                              image_name=c["image_name"], skel0=c["skel0"]))
    if norm_dataset_temporal is not None:
        items += _precompute_temporal(norm_dataset_temporal, q, variant, wavelet_cache)
    desc, hist = _filas_diagnostico(items, p)
    desc, hist = pd.DataFrame(desc), pd.DataFrame(hist)
    resumen = _resumen_diagnostico(desc, p)
    hist.to_csv(out_hist, index=False)
    desc.to_csv(out_desc, index=False)
    resumen.to_csv(out_resumen, index=False)
    _figura_diagnostico(hist, resumen, p, out_png)

    pd.set_option("display.width", 200)
    cols = ["dataset", "grupo", "n_imagenes", "pct_esqueleto_borrado", "pct_ramas_terminales_cortas",
            "pct_aislados_pequenos"] + [c for c in resumen.columns if c.startswith("pct_de_lo_borrado_")]
    print(f"\n[diagnóstico] qué borra prune_iters={p} (porcentajes sobre el total de cada grupo):")
    print(resumen[cols].round(1).to_string(index=False))
    print(f"\nGuardado en {out_hist}, {out_desc}, {out_resumen} y {out_png}")
    return {"hist": hist, "descomposicion": desc, "resumen": resumen}


# ----------------------------------------------------------------------------------------------
# 2) Alternativa para el clasificador
# ----------------------------------------------------------------------------------------------
# Valores de la tesis / de los CSV ya obtenidos con prune_iters=5 (d de Cohen y medianas de skeleton_length)
REF_ITER5 = {"median_thickness": -5.511560, "median_tortuosity": -3.415598, "median_segment_length": 2.921209,
             "mean_tortuosity": -1.902865, "max_segment_length": 1.587764, "max_thickness": -0.995326,
             "skeleton_length": 5.421671}


def _chequeo_iter5_clasif(largo):
    sub = largo[largo["variante"] == "iter5"].set_index("feature")
    if sub.empty:
        return
    print("\n[chequeo] spec iter5 frente a los resultados que ya obtuviste con prune_iters=5 (d de Cohen):")
    mal = 0
    for f, esperado in REF_ITER5.items():
        if f not in sub.index:
            continue
        v = float(sub.loc[f, "d"])
        ok = abs(v - esperado) < 1e-3
        mal += (not ok)
        print(f"   {'OK ' if ok else 'DIF'} {f:24s} script={v:9.4f}   antes={esperado:9.4f}")
    print("[chequeo] " + ("coincide." if mal == 0 else f"{mal} valor(es) difieren: revisá q, variant y el dataset."))


def run_alternativa_clasificador(norm_dataset, specs=None, q=80, variant="closing_r3", run_nested=False,
                                 csv_ref=None, features_finales=None,
                                 out_csv="alt_longitud_clasificador.csv",
                                 out_csv_features="alt_longitud_clasificador_features.csv",
                                 out_csv_conjuntos="alt_longitud_clasificador_conjuntos.csv"):
    import sensibilidad_poda as sp
    specs = specs or SPECS
    features_finales = list(features_finales) if features_finales else sp.FEATURES_FINALES
    features_tabla = features_finales + [f for f in sp.FEATURES_REFERENCIA if f not in features_finales]
    if run_nested:
        from nested_binary_evaluation import run_nested_leave_one_culture_out

    print("Calculando wavelet, máscara y esqueleto (una sola vez por imagen)...")
    cache = sp._precompute(norm_dataset, q=q, variant=variant)

    resumen, largo, conjuntos = [], [], []
    for spec in specs:
        nombre = spec["nombre"]
        print(f"\n=== {nombre} ===")
        with _parchar(sp, "prune_skeleton", _crear_podador(spec)):
            df = sp._features_df(cache, spec.get("p", P_DIAG))
        if nombre == "iter5" and csv_ref:
            sp._check_against_csv(df, csv_ref)
        largo.extend(sp._summarize_features(df, nombre, features_tabla))

        row = {"variante": nombre}
        try:
            sel = sp._select_final_set(df)
            row.update(n_features_finales=len(sel), features_finales=", ".join(sel),
                       faltan_vs_modelo_guardado=", ".join(sorted(set(features_finales) - set(sel))),
                       sobran_vs_modelo_guardado=", ".join(sorted(set(sel) - set(features_finales))))
            print(f"Conjunto final ({len(sel)}): {sel}")
            conjuntos.append(dict(variante=nombre, n=len(sel), features=", ".join(sel),
                                  faltan=row["faltan_vs_modelo_guardado"], sobran=row["sobran_vs_modelo_guardado"]))
        except Exception as e:
            print(f"[aviso] no se pudo repetir la selección final con {nombre}: {e!r}")
        if run_nested:
            out = run_nested_leave_one_culture_out(df)
            gm = out["global_metrics"]
            row.update(AUC_global=gm["roc_auc"], F1_global=gm["f1"], accuracy_global=gm["accuracy"],
                       modelo_por_fold=", ".join(out["summary_df"]["selected_model"]),
                       variables_por_fold=" | ".join(out["summary_df"]["selected_features"]))
        resumen.append(row)

    largo = pd.DataFrame(largo).rename(columns={"prune": "variante"})
    if not largo.empty and (largo["variante"] == "iter0").any():
        base = largo[largo["variante"] == "iter0"].set_index("feature")

        def _pct(v, ref):
            return np.nan if (pd.isna(ref) or ref == 0) else 100 * (v / ref - 1)
        for cond in ("CTRL", "HPMC"):
            col = f"med_{cond}"
            largo[f"{col}_pct_vs_sin_poda"] = [_pct(r[col], base.loc[r["feature"], col])
                                               for _, r in largo.iterrows()]
    if not largo.empty:
        largo["abs_d_ge_0.8"] = largo["d"].abs() >= sp.D_UMBRAL
    resumen = pd.DataFrame(resumen)
    sl = largo[largo["feature"] == "skeleton_length"] if not largo.empty else largo
    if not sl.empty and "med_CTRL_pct_vs_sin_poda" in sl:
        for cond in ("CTRL", "HPMC"):
            resumen[f"skeleton_length_{cond}_pct_vs_sin_poda"] = resumen["variante"].map(
                sl.set_index("variante")[f"med_{cond}_pct_vs_sin_poda"])
    conjuntos = pd.DataFrame(conjuntos)
    _chequeo_iter5_clasif(largo)
    resumen.to_csv(out_csv, index=False)
    largo.to_csv(out_csv_features, index=False)
    conjuntos.to_csv(out_csv_conjuntos, index=False)
    print(f"\nGuardado en {out_csv}, {out_csv_features} y {out_csv_conjuntos}")
    return {"resumen": resumen, "features": largo, "conjuntos": conjuntos}


# ----------------------------------------------------------------------------------------------
# 3) Alternativa para el evaluador de progresión (usa tus funciones vía sensibilidad_poda)
# ----------------------------------------------------------------------------------------------
def run_alternativa_progresion(norm_dataset, raw_dataset, specs=None, q=80, variant="closing_r3", run_loo=True,
                               metadata_ref=None, wavelet_cache=None,
                               out_csv="alt_longitud_progresion.csv",
                               out_csv_pares="alt_longitud_progresion_pares.csv",
                               out_csv_scores="alt_longitud_progresion_scores.csv"):
    import sensibilidad_poda as spp
    pipeline_root = spp.pipeline_root
    specs = specs or SPECS
    wavelet_cache = {} if wavelet_cache is None else wavelet_cache
    corridas, filas, pares = {}, [], []

    for spec in specs:
        nombre = spec["nombre"]
        print(f"\n=== {nombre} ===")
        caches = {"wavelet": wavelet_cache, "features": {}}     # features separadas por variante
        p = spec.get("p", P_DIAG)
        with _parchar(pipeline_root, "prune_skeleton", _crear_podador(spec)):
            r = spp._correr_un_valor(norm_dataset, raw_dataset, p, q, variant, run_loo, caches)
        corridas[nombre] = r
        fila = spp._fila_resumen(nombre, r)
        fila["variante"] = fila.pop("prune_iters")
        filas.append(fila)
        for modo, tabla in (("Reference-Free", r["free_por_par"]), ("Experiment-Anchored", r["anch_por_par"]),
                            ("Leave-one-culture-out", r["loo"]["por_par"] if r["loo"] else None)):
            if tabla is None:
                continue
            t = tabla.copy()
            t.insert(0, "modo", modo)
            t.insert(0, "variante", nombre)
            pares.append(t)

    resumen = pd.DataFrame(filas)
    resumen = resumen[["variante"] + [c for c in resumen.columns if c != "variante"]]
    scores = []
    for nombre, r in corridas.items():
        s = spp._tabla_scores(nombre, r).rename(columns={"prune_iters": "variante"})
        scores.append(s)
    scores = pd.concat(scores, ignore_index=True)

    llave = ["group", "time_label", "image_name"]
    if "iter5" in corridas:
        ref = scores[scores["variante"] == "iter5"].set_index(llave)
        for idx, nombre in enumerate(resumen["variante"]):
            cur = scores[scores["variante"] == nombre].set_index(llave).loc[ref.index]
            for k, col, cat in (("free", "score_free", "categoria_free"),
                                ("anch", "score_anchored", "categoria_anchored"),
                                ("loo", "score_loo", "categoria_loo")):
                if col in cur and cur[col].notna().all() and ref[col].notna().all():
                    resumen.loc[idx, f"{k}_rho_score_vs_iter5"] = float(cur[col].rank().corr(ref[col].rank()))
                    resumen.loc[idx, f"{k}_misma_categoria_vs_iter5_pct"] = 100 * float(
                        (cur[cat].to_numpy() == ref[cat].to_numpy()).mean())
    for idx, nombre in enumerate(resumen["variante"]):
        s = scores[scores["variante"] == nombre]
        if "categoria_loo" in s and s["categoria_loo"].notna().all():
            resumen.loc[idx, "loo_misma_categoria"] = int((s["categoria_anchored"] == s["categoria_loo"]).sum())

    if "iter5" in corridas:
        row5 = resumen.loc[resumen["variante"] == "iter5"].iloc[0].to_dict()
        spp._chequeo_tesis(row5)
        if metadata_ref:
            spp._chequeo_metadata(corridas["iter5"]["meta"], metadata_ref)
            spp._chequeo_funcion_original(corridas["iter5"], raw_dataset, metadata_ref)

    pares = pd.concat(pares, ignore_index=True)
    resumen.to_csv(out_csv, index=False)
    pares.to_csv(out_csv_pares, index=False)
    scores.to_csv(out_csv_scores, index=False)
    print(f"\nGuardado en {out_csv}, {out_csv_pares} y {out_csv_scores}")
    return {"resumen": resumen, "pares_por_tiempo": pares, "scores": scores}


# ----------------------------------------------------------------------------------------------
def run_todo(norm_clasif, norm_temporal, raw_temporal, csv_ref=None, metadata_ref=None, run_nested=False,
             specs=None, q=80, variant="closing_r3"):
    """Diagnóstico + alternativa para el clasificador + alternativa para el evaluador."""
    wl = {}
    d = run_diagnostico(norm_clasif, norm_temporal, q=q, variant=variant, wavelet_cache=wl)
    c = run_alternativa_clasificador(norm_clasif, specs=specs, q=q, variant=variant, run_nested=run_nested,
                                     csv_ref=csv_ref)
    p = run_alternativa_progresion(norm_temporal, raw_temporal, specs=specs, q=q, variant=variant,
                                   metadata_ref=metadata_ref, wavelet_cache=wl)
    return {"diagnostico": d, "clasificador": c, "progresion": p}