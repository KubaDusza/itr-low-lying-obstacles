import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Carpetas de los dos experimentos.
base = Path("/data/cs_courses/jgomezel/esanet_pilot")

rgb_path = base / "runs_50_rgb/esanet_r34nbt1d_rgb_s0/history.csv"
rgbd_path = base / "runs_50_rgbd/esanet_r34nbt1d_rgbd_s0/history.csv"

# Leemos las métricas de las 50 épocas.
rgb = pd.read_csv(rgb_path)
rgbd = pd.read_csv(rgbd_path)

# Mostramos las columnas disponibles.
print("Columnas:", list(rgb.columns))

# Creamos una carpeta para las gráficas.
output = base / "figures_50"
output.mkdir(exist_ok=True)

# Métricas que queremos comparar.
metrics = [
    ("train_loss", "Training loss", "training_loss.png"),
    ("val_loss", "Validation loss", "validation_loss.png"),
    ("val_IoU", "Validation obstacle IoU", "validation_iou.png"),
]

for column, title, filename in metrics:

    plt.figure(figsize=(9, 5))

    # Eje horizontal: número de época.
    plt.plot(rgb["epoch"], rgb[column], label="ESANet RGB")
    plt.plot(rgbd["epoch"], rgbd[column], label="ESANet RGB-D")

    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.title(title + " — ISOD v0, 50 epochs")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(output / filename, dpi=200, bbox_inches="tight")
    plt.close()

    print("Guardado:", output / filename)

# Identificamos el mejor resultado de validación.
for name, data in [("RGB", rgb), ("RGB-D", rgbd)]:
    best = data.loc[data["val_IoU"].idxmax()]
    print(
        f"{name}: mejor IoU = {best['val_IoU']:.4f}, "
        f"época = {int(best['epoch'])}"
    )

print("Gráficas terminadas.")
