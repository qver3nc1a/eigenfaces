from sklearn.datasets import fetch_olivetti_faces
from pathlib import Path
from PIL import Image

faces = fetch_olivetti_faces()

for i, image in enumerate(faces.images):
    person = faces.target[i] + 1
    folder = Path("data") / f"s{person:02d}"
    folder.mkdir(parents=True, exist_ok=True)

    image = Image.fromarray((image * 255).astype("uint8"))
    image.save(folder / f"{i:02d}.png")
