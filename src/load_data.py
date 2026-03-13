import os
import tifffile as tiff 

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
                    for file in os.listdir(cond_path):
                        if file.endswith(".tif"):
                            img_path = os.path.join(cond_path, file)
                            img = tiff.imread(img_path)
                            dataset[grupo][cond].append({"name": file, "image": img})
    return dataset