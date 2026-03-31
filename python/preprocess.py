# python/preprocess.py
import os
import numpy as np
from PIL import Image

IMG_SIZE = (32, 32)
CLASSES = {
    "great_wall":      0,
    "taj_mahal":       1,
    "christ_redeemer": 2,
}

def load_dataset(data_dir, split=0.8):
    X, y = [], []
    for class_name, label in CLASSES.items():
        folder = os.path.join(data_dir, class_name)
        if not os.path.exists(folder):
            print(f"Dossier manquant : {folder}")
            continue
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"  {class_name} : {len(files)} images trouvées")
        for fname in files:
            try:
                img = Image.open(os.path.join(folder, fname))
                img = img.resize(IMG_SIZE).convert("L")  # Grayscale
                vec = np.array(img).flatten() / 255.0    # Normalise [0,1]
                X.append(vec)
                y.append(label)
            except Exception as e:
                print(f"  ⚠️  Erreur sur {fname} : {e}")

    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.int32)

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    # Split train/test
    split_i = int(len(X) * split)
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    print("Chargement du dataset...")
    X_train, y_train, X_test, y_test = load_dataset("../dataset")

    np.save("../test_cases/X_train.npy", X_train)
    np.save("../test_cases/y_train.npy", y_train)
    np.save("../test_cases/X_test.npy",  X_test)
    np.save("../test_cases/y_test.npy",  y_test)

    print(f"Train : {X_train.shape} | Test : {X_test.shape}")
    print(f"   Labels train : {np.bincount(y_train)}")
    print(f"   Labels test  : {np.bincount(y_test)}")
    print("Sauvegardé dans test_cases/")