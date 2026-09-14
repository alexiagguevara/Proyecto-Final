import os
import re
import tifffile as tiff

def extract_image_number(filename):
    m = re.search(r"Image(\d+)", filename)
    if m is None:
        return filename
    return int(m.group(1))

def load_evaluation_images(base_path):
    """
    Carga dataset externo con estructura:
    base_path/
        Control/N3/*.tif
        Control/N4/*.tif
        Proinflammatory/N3/*.tif
        Proinflammatory/N4/*.tif

    Devuelve:
    dataset[group][condition] = [
        {"name": ..., "image": ..., "true_group": ...},
        ...
    ]
    """
    condition_map = {
        "Control": "CTRL",
        "Proinflammatory": "HPMC"
    }

    dataset = {}

    for raw_condition, mapped_condition in condition_map.items():
        cond_path = os.path.join(base_path, raw_condition)
        if not os.path.isdir(cond_path):
            continue

        group_dirs = [
            g for g in os.listdir(cond_path)
            if os.path.isdir(os.path.join(cond_path, g)) and not g.startswith(".")
        ]

        for group in sorted(group_dirs):
            group_path = os.path.join(cond_path, group)

            tif_files = [
                f for f in os.listdir(group_path)
                if f.endswith(".tif") and not f.startswith(".")
            ]
            tif_files = sorted(tif_files, key=extract_image_number)

            if group not in dataset:
                dataset[group] = {}
            if mapped_condition not in dataset[group]:
                dataset[group][mapped_condition] = []

            for file in tif_files:
                img_path = os.path.join(group_path, file)
                img = tiff.imread(img_path)

                dataset[group][mapped_condition].append({
                    "name": file,
                    "image": img,
                    "true_group": group
                })

    return dataset