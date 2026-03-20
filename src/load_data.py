import os
import re
import tifffile as tiff

def extract_image_number(filename):
    """
    Extrae el número de nombres tipo 'Image107.tif'
    """
    m = re.search(r"Image(\d+)", filename)
    if m is None:
        raise ValueError(f"No pude extraer número de: {filename}")
    return int(m.group(1))


def assign_true_group_labels(folder_group, files_sorted):
    """
    Asigna N1/N2 o N3/N4 según la carpeta:
    - primeras 4 imágenes -> primer cultivo
    - últimas 4 imágenes  -> segundo cultivo
    """
    if len(files_sorted) != 8:
        raise ValueError(
            f"Esperaba 8 imágenes en {folder_group}, pero encontré {len(files_sorted)}"
        )

    if folder_group == "N1 y N2":
        first_label, second_label = "N1", "N2"
    elif folder_group == "N3 y N4":
        first_label, second_label = "N3", "N4"
    else:
        raise ValueError(f"Grupo desconocido: {folder_group}")

    labels = {}
    for i, file in enumerate(files_sorted):
        labels[file] = first_label if i < 4 else second_label

    return labels


def load_images(base_path):
    dataset = {}
    for grupo in os.listdir(base_path):
        grupo_path = os.path.join(base_path, grupo)
        if os.path.isdir(grupo_path):
            dataset[grupo] = {}

            for cond in os.listdir(grupo_path):
                cond_path = os.path.join(grupo_path, cond)

                if os.path.isdir(cond_path):
                    dataset[grupo][cond] = []

                    tif_files = [
                        f for f in os.listdir(cond_path)
                        if f.endswith(".tif")
                    ]

                    # Orden estable por número de imagen
                    tif_files = sorted(tif_files, key=extract_image_number)

                    # Asignar true_group
                    group_labels = assign_true_group_labels(grupo, tif_files)

                    for file in tif_files:
                        img_path = os.path.join(cond_path, file)
                        img = tiff.imread(img_path)

                        dataset[grupo][cond].append({
                            "name": file,
                            "image": img,
                            "true_group": group_labels[file]
                        })

    return dataset