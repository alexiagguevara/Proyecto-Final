import os
import re
import tifffile as tiff

def extract_image_number(filename):
    m = re.search(r"Image(\d+)", filename)
    if m is None:
        raise ValueError(f"No pude extraer número de: {filename}")
    return int(m.group(1))

def parse_time_label(time_label):
    """
    Convierte etiquetas tipo '0HS', '24HS', '48HS', '72HS' a entero en horas.
    """
    m = re.match(r"(\d+)\s*HS", time_label.upper())
    if m is None:
        raise ValueError(f"No pude interpretar el tiempo: {time_label}")
    return int(m.group(1))

def load_temporal_images(base_path):
    """
    Carga dataset temporal con estructura:
    base_path/
        0HS/N1/*.tif
        0HS/N2/*.tif
        ...
        72HS/N4/*.tif
    """
    dataset = {}

    time_dirs = [
        d for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d)) and not d.startswith(".")
    ]

    for time_label in sorted(time_dirs, key=parse_time_label):
        time_path = os.path.join(base_path, time_label)
        dataset[time_label] = {}
        time_h = parse_time_label(time_label)

        group_dirs = [
            g for g in os.listdir(time_path)
            if os.path.isdir(os.path.join(time_path, g)) and not g.startswith(".")
        ]

        for group in sorted(group_dirs):
            group_path = os.path.join(time_path, group)

            tif_files = [
                f for f in os.listdir(group_path) 
                if f.endswith(".tif") and not f.startswith(".")
            ]
            tif_files = sorted(tif_files, key=extract_image_number)

            dataset[time_label][group] = []

            for file in tif_files:
                img_path = os.path.join(group_path, file)
                img = tiff.imread(img_path)

                dataset[time_label][group].append({
                    "name": file,
                    "image": img,
                    "group": group,
                    "time_label": time_label,
                    "time_h": time_h
                })

    return dataset