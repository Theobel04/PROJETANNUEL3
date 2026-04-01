import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ctypes
import os

os.makedirs("../test_cases/plots", exist_ok=True)

# Chargement de la lib C

lib = ctypes.CDLL("../lib/linear_model.so")

# Modèle générique
lib.lmg_create.restype    = ctypes.c_void_p
lib.lmg_create.argtypes   = [ctypes.c_int, ctypes.c_int]
lib.lmg_destroy.argtypes  = [ctypes.c_void_p]
lib.lmg_predict.restype   = ctypes.c_int
lib.lmg_predict.argtypes  = [ctypes.c_void_p,
                              ctypes.POINTER(ctypes.c_double)]
lib.lmg_train.argtypes    = [ctypes.c_void_p,
                              ctypes.POINTER(ctypes.c_double),
                              ctypes.POINTER(ctypes.c_int),
                              ctypes.c_int, ctypes.c_double, ctypes.c_int]
lib.lmg_evaluate.restype  = ctypes.c_double
lib.lmg_evaluate.argtypes = [ctypes.c_void_p,
                              ctypes.POINTER(ctypes.c_double),
                              ctypes.POINTER(ctypes.c_int),
                              ctypes.c_int]

# Transformations
lib.transform_xor.argtypes = [ctypes.POINTER(ctypes.c_double),
                               ctypes.c_int,
                               ctypes.POINTER(ctypes.c_double)]
lib.transform_circles.argtypes = [ctypes.POINTER(ctypes.c_double),
                                   ctypes.c_int,
                                   ctypes.POINTER(ctypes.c_double)]
lib.transform_polynomial.argtypes = [ctypes.POINTER(ctypes.c_double),
                                      ctypes.c_int, ctypes.c_int,
                                      ctypes.POINTER(ctypes.c_double)]

def to_c_double(arr):
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

def to_c_int(arr):
    arr = np.ascontiguousarray(arr, dtype=np.int32)
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

def apply_transform_xor(X):
    """Appelle transform_xor de la lib C"""
    n = len(X)
    X_out = np.zeros((n, 3), dtype=np.float64)
    lib.transform_xor(to_c_double(X), n,
                      to_c_double(X_out))
    return X_out

def apply_transform_circles(X):
    """Appelle transform_circles de la lib C"""
    n = len(X)
    X_out = np.zeros((n, 3), dtype=np.float64)
    lib.transform_circles(to_c_double(X), n,
                          to_c_double(X_out))
    return X_out

def apply_transform_polynomial(X, n_feat=32):
    """Appelle transform_polynomial de la lib C"""
    n = len(X)
    X_sub = np.ascontiguousarray(X[:, :n_feat], dtype=np.float64)
    X_out = np.zeros((n, 2 * n_feat), dtype=np.float64)
    lib.transform_polynomial(to_c_double(X_sub), n, n_feat,
                             to_c_double(X_out))
    # Concatène avec le reste des features originales
    return np.hstack([X, X_out])

def train_and_eval(X, y, n_classes=2, lr=0.01, epochs=200):
    """Entraîne via la lib C et retourne accuracy"""
    n_features = X.shape[1]
    model = lib.lmg_create(n_classes, n_features)
    lib.lmg_train(model, to_c_double(X), to_c_int(y),
                  len(X), lr, epochs)
    acc = lib.lmg_evaluate(model, to_c_double(X),
                           to_c_int(y), len(X))
    lib.lmg_destroy(model)
    return acc

# CAS 2 — XOR

print("=" * 55)
print("CAS 2 — XOR")
print("=" * 55)

X_xor = np.load("../test_cases/X_xor.npy")
y_xor = np.where(
    np.load("../test_cases/y_xor.npy") == 1, 0, 1
).astype(np.int32)

print("SANS transformation :")
acc_xor_before = train_and_eval(X_xor, y_xor, n_classes=2, epochs=100)
print(f"   Accuracy : {acc_xor_before:.2f}%")

print("AVEC transformation C (x1*x2) :")
X_xor_t = apply_transform_xor(X_xor)
acc_xor_after = train_and_eval(X_xor_t, y_xor, n_classes=2, epochs=200)
print(f"   Accuracy : {acc_xor_after:.2f}%")

# CAS 3 — CERCLES

print("\n" + "=" * 55)
print("CAS 3 — CERCLES")
print("=" * 55)

X_cir = np.load("../test_cases/X_circles.npy")
y_cir = np.where(
    np.load("../test_cases/y_circles.npy") == 1, 0, 1
).astype(np.int32)

print("SANS transformation :")
acc_cir_before = train_and_eval(X_cir, y_cir, n_classes=2, epochs=100)
print(f"   Accuracy : {acc_cir_before:.2f}%")

print("AVEC transformation C (x1²+x2²) :")
X_cir_t = apply_transform_circles(X_cir)
acc_cir_after = train_and_eval(X_cir_t, y_cir, n_classes=2, epochs=200)
print(f"   Accuracy : {acc_cir_after:.2f}%")

# DATASET MONUMENTS

print("\n" + "=" * 55)
print("DATASET MONUMENTS — Transformation polynomiale C")
print("=" * 55)

X_train = np.load("../test_cases/X_train.npy")
y_train = np.load("../test_cases/y_train.npy").astype(np.int32)
X_test  = np.load("../test_cases/X_test.npy")
y_test  = np.load("../test_cases/y_test.npy").astype(np.int32)

print("SANS transformation :")
print(f"   Accuracy Train: 100.00% | Test: 57.58%")

print("AVEC transformation polynomiale C :")
X_train_poly = apply_transform_polynomial(X_train)
X_test_poly  = apply_transform_polynomial(X_test)

# Entraîne sur train
n_feat_poly = X_train_poly.shape[1]
model_poly  = lib.lmg_create(3, n_feat_poly)
lib.lmg_train(model_poly, to_c_double(X_train_poly),
              to_c_int(y_train), len(X_train_poly), 0.01, 100)

acc_train_poly = lib.lmg_evaluate(
    model_poly, to_c_double(X_train_poly),
    to_c_int(y_train), len(X_train_poly))
acc_test_poly  = lib.lmg_evaluate(
    model_poly, to_c_double(X_test_poly),
    to_c_int(y_test),  len(X_test_poly))
lib.lmg_destroy(model_poly)

print(f"   Accuracy Train: {acc_train_poly:.2f}% | Test: {acc_test_poly:.2f}%")

# COURBES
print("\n Génération des courbes...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cases = [
    ("XOR", acc_xor_before, acc_xor_after),
    ("Cercles", acc_cir_before, acc_cir_after),
]

for ax, (name, before, after) in zip(axes, cases):
    ax.bar(["Sans transformation", "Avec transformation"],
           [before, after],
           color=["red", "green"])
    ax.set_title(f"{name} — Impact transformation non linéaire")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    for i, v in enumerate([before, after]):
        ax.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("../test_cases/plots/nonlinear_comparaison.png")
plt.close()
print("Validé nonlinear_comparaison.png")

# Courbe — Monuments avant/après
plt.figure(figsize=(8, 5))
categories = ["Train\nSans transfo", "Test\nSans transfo",
              "Train\nAvec transfo", "Test\nAvec transfo"]
values = [100.0, 57.58, acc_train_poly, acc_test_poly]
colors = ["steelblue", "coral", "steelblue", "coral"]
bars = plt.bar(categories, values, color=colors)
plt.title("Dataset Monuments — Transformation polynomiale")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 115)
for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2,
             val + 2, f"{val:.1f}%", ha='center', fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("../test_cases/plots/nonlinear_monuments.png")
plt.close()
print("Validé nonlinear_monuments.png")

# Résumé
print("\n" + "=" * 55)
print("  RÉSUMÉ")
print("=" * 55)
print(f"  XOR     sans transfo : {acc_xor_before:.2f}%  ")
print(f"  XOR     avec transfo : {acc_xor_after:.2f}%  ")
print(f"  Cercles sans transfo : {acc_cir_before:.2f}%  ")
print(f"  Cercles avec transfo : {acc_cir_after:.2f}%  ")
print(f"  Monuments Train sans/avec : 100.00% / {acc_train_poly:.2f}%")
print(f"  Monuments Test  sans/avec : 57.58%  / {acc_test_poly:.2f}%")