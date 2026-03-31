# python/generate_curves.py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ctypes
import os

os.makedirs("../test_cases/plots", exist_ok=True)

# --- Charge la lib C ---
lib = ctypes.CDLL("../lib/linear_model.so")
lib.lm_create.restype    = ctypes.c_void_p
lib.lm_destroy.argtypes  = [ctypes.c_void_p]
lib.lm_predict.restype   = ctypes.c_int
lib.lm_predict.argtypes  = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
lib.lm_train.argtypes    = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double),
                             ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                             ctypes.c_double, ctypes.c_int]
lib.lm_evaluate.restype  = ctypes.c_double
lib.lm_evaluate.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double),
                             ctypes.POINTER(ctypes.c_int), ctypes.c_int]

def to_c_double(arr):
    return arr.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
def to_c_int(arr):
    return arr.astype(np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))

def train_tracked(X, y, lr=0.01, epochs=100):
    """Entraîne et retourne la liste d'erreurs par epoch"""
    model = lib.lm_create()
    errors_list = []
    n = len(X)
    for e in range(epochs):
        # On réentraîne epoch par epoch pour tracker
        pass
    lib.lm_destroy(model)

    # Version simple : on relance l'entraînement en capturant stdout
    # On utilise plutôt une implémentation Python pour le tracking
    errors_list = []
    W = np.zeros((3, X.shape[1]))
    b = np.zeros(3)

    for e in range(epochs):
        errors = 0
        idx = np.arange(n)
        for i in idx:
            scores = W @ X[i] + b
            pred = np.argmax(scores)
            truth = y[i]
            if pred != truth:
                errors += 1
                W[truth] += lr * X[i]
                W[pred]  -= lr * X[i]
                b[truth] += lr
                b[pred]  -= lr
        errors_list.append(errors)
        if errors == 0:
            break

    return errors_list, W, b

# ============================================================
# COURBE 1 — Erreurs par epoch sur les 4 cas
# ============================================================
print(" Génération courbe 1 : erreurs par epoch...")

# Cas 1
X_lin = np.load("../test_cases/X_linear.npy")
y_lin = np.where(np.load("../test_cases/y_linear.npy") == 1, 0, 1).astype(np.int32)
X_lin_pad = np.pad(X_lin, ((0,0),(0,1022)))
errors_lin, _, _ = train_tracked(X_lin_pad, y_lin)

# Cas 2
X_xor = np.load("../test_cases/X_xor.npy")
y_xor = np.where(np.load("../test_cases/y_xor.npy") == 1, 0, 1).astype(np.int32)
X_xor_pad = np.pad(X_xor, ((0,0),(0,1022)))
errors_xor, _, _ = train_tracked(X_xor_pad, y_xor, epochs=100)

# Cas 3
X_cir = np.load("../test_cases/X_circles.npy")
y_cir = np.where(np.load("../test_cases/y_circles.npy") == 1, 0, 1).astype(np.int32)
X_cir_pad = np.pad(X_cir, ((0,0),(0,1022)))
errors_cir, _, _ = train_tracked(X_cir_pad, y_cir, epochs=100)

plt.figure(figsize=(10, 5))
plt.plot(errors_lin, label="Cas 1 - Linéairement séparable", color="green")
plt.plot(errors_xor, label="Cas 2 - XOR", color="red")
plt.plot(errors_cir, label="Cas 3 - Cercles", color="orange")
plt.title("Évolution des erreurs par epoch — Modèle Linéaire")
plt.xlabel("Epoch")
plt.ylabel("Nombre d'erreurs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../test_cases/plots/courbe1_erreurs_par_epoch.png")
plt.close()
print(" courbe1_erreurs_par_epoch.png")

# ============================================================
# COURBE 2 — Frontière de décision Cas 1 vs Cas 2
# ============================================================
print(" Génération courbe 2 : frontières de décision...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (X_pad, y, X_orig, title) in zip(axes, [
    (X_lin_pad, y_lin, X_lin, "Cas 1 — Linéairement séparable"),
    (X_xor_pad, y_xor, X_xor, "Cas 2 — XOR"),
]):
    _, W, b = train_tracked(X_pad, y, epochs=200)

    # Grille de décision sur les 2 premières features
    x_min, x_max = X_orig[:,0].min()-1, X_orig[:,0].max()+1
    y_min, y_max = X_orig[:,1].min()-1, X_orig[:,1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_pad = np.pad(grid, ((0,0),(0,1022)))
    scores = grid_pad @ W.T + b
    Z = np.argmax(scores, axis=1).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlGn')
    colors = ['green' if c == 0 else 'red' for c in y]
    ax.scatter(X_orig[:,0], X_orig[:,1], c=colors, edgecolors='k', s=40)
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

plt.tight_layout()
plt.savefig("../test_cases/plots/courbe2_frontieres_decision.png")
plt.close()
print(" courbe2_frontieres_decision.png")

# ============================================================
# COURBE 3 — Surapprentissage : Train vs Test par epochs
# ============================================================
print(" Génération courbe 3 : surapprentissage train vs test...")

X_train = np.load("../test_cases/X_train.npy")
y_train = np.load("../test_cases/y_train.npy").astype(np.int32)
X_test  = np.load("../test_cases/X_test.npy")
y_test  = np.load("../test_cases/y_test.npy").astype(np.int32)

acc_train_list = []
acc_test_list  = []
W = np.zeros((3, 1024))
b = np.zeros(3)

for e in range(100):
    # 1 epoch
    for i in range(len(X_train)):
        scores = W @ X_train[i] + b
        pred  = np.argmax(scores)
        truth = y_train[i]
        if pred != truth:
            W[truth] += 0.01 * X_train[i]
            W[pred]  -= 0.01 * X_train[i]
            b[truth] += 0.01
            b[pred]  -= 0.01

    # Accuracy train
    preds_train = np.argmax(X_train @ W.T + b, axis=1)
    acc_train_list.append(np.mean(preds_train == y_train) * 100)

    # Accuracy test
    preds_test = np.argmax(X_test @ W.T + b, axis=1)
    acc_test_list.append(np.mean(preds_test == y_test) * 100)

plt.figure(figsize=(10, 5))
plt.plot(acc_train_list, label="Accuracy Train", color="blue")
plt.plot(acc_test_list,  label="Accuracy Test",  color="red", linestyle="--")
plt.title("Surapprentissage — Train vs Test (Dataset Monuments)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True)
plt.axhline(y=100, color='blue', alpha=0.2)
plt.tight_layout()
plt.savefig("../test_cases/plots/courbe3_surapprentissage.png")
plt.close()
print("courbe3_surapprentissage.png")

# ============================================================
# COURBE 4 — Distribution du dataset (déséquilibre)
# ============================================================
print(" Génération courbe 4 : distribution du dataset...")

classes = ["Great Wall", "Taj Mahal", "Christ Rédempteur"]
train_counts = np.bincount(y_train)
test_counts  = np.bincount(y_test)

x = np.arange(len(classes))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width/2, train_counts, width, label='Train', color='steelblue')
ax.bar(x + width/2, test_counts,  width, label='Test',  color='coral')
ax.set_title("Distribution des classes — Dataset Monuments")
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_ylabel("Nombre d'images")
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("../test_cases/plots/courbe4_distribution_dataset.png")
plt.close()
print("courbe4_distribution_dataset.png")

print("\nToutes les courbes générées dans test_cases/plots/")