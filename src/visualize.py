import matplotlib.pyplot as plt

def show_examples(dataset, n):
    """
    Muestra algunas imágenes de cada grupo y condición.
    dataset: diccionario cargado con load_images
    n: cantidad de imágenes a mostrar por condición
    """
    for grupo, condiciones in dataset.items():
        for cond, imgs in condiciones.items():
            plt.figure(figsize=(10, 5))
            for i in range(min(n, len(imgs))):
                plt.subplot(1, n, i + 1)
                if imgs[i]['image'].ndim == 3 and imgs[i]['image'].shape[2] == 3:
                    plt.imshow(imgs[i]['image'])
                else:
                    plt.imshow(imgs[i]['image'], cmap="gray")
                plt.title(f"{grupo} - {cond} - {imgs[i]['name']}")
                plt.axis("off")
            plt.show()


def show_all_channels_mosaic(dataset, cond_filter="CTRL"):
    """
    Muestra todas las imágenes de una condición (CTRL o HPMC) en un mosaico:
    cada fila = imagen original + canales R, G, B por separado.
    """
    for grupo, condiciones in dataset.items():
        if cond_filter not in condiciones:
            continue
        imgs = condiciones[cond_filter]
        n_imgs = len(imgs)
        fig, axes = plt.subplots(n_imgs, 4, figsize=(18, 3*n_imgs))
        fig.suptitle(f"Imágenes {cond_filter}", fontsize=20)
        f = 12  # font size

        for i, img_dict in enumerate(imgs):
            img = img_dict["image"]
            name = img_dict["name"]

            ax_row = axes[i] if n_imgs > 1 else axes  # por si hay solo una fila

            # Imagen original
            ax_row[0].imshow(img)
            ax_row[0].set_title(f"{name} - Original", fontsize=f)
            ax_row[0].axis(False)

            # Canal rojo
            ax_row[1].imshow(img[:, :, 0], cmap="gray", vmin=0, vmax=255)
            ax_row[1].set_title("Rojo", fontsize=f)
            ax_row[1].axis(False)

            # Canal verde
            ax_row[2].imshow(img[:, :, 1], cmap="gray", vmin=0, vmax=255)
            ax_row[2].set_title("Verde", fontsize=f)
            ax_row[2].axis(False)

            # Canal azul
            ax_row[3].imshow(img[:, :, 2], cmap="gray", vmin=0, vmax=255)
            ax_row[3].set_title("Azul", fontsize=f)
            ax_row[3].axis(False)

        plt.tight_layout()
        plt.show()

def show_rgb_vs_green_verif(rgb_dataset, green_dataset, grupo):
    """
    Muestra un ejemplo de comparación entre imagen RGB y su correspondiente canal verde extraído.
    La primera fila es RGB, la segunda fila es el canal verde.
    """
    condiciones = list(rgb_dataset[grupo].keys())
    fig, axes = plt.subplots(2, len(condiciones), figsize=(5*len(condiciones), 10))
    f = 12
    
    for i, cond in enumerate(condiciones):
        # Tomamos la primera imagen de cada condición
        rgb_img = rgb_dataset[grupo][cond][0]["image"]
        green_img = green_dataset[grupo][cond][0]["image"]
        
        # Fila 1: RGB
        axes[0, i].imshow(rgb_img)
        axes[0, i].set_title(f"{cond} - RGB", fontsize=f)
        axes[0, i].axis("off")
        
        # Fila 2: verde
        axes[1, i].imshow(green_img, cmap="gray")
        axes[1, i].set_title(f"{cond} - Canal Verde", fontsize=f)
        axes[1, i].axis("off")
    
    plt.show()
    
