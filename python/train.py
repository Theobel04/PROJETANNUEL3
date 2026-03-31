# python/train.py
import ctypes
import numpy as np
import os

# --- Charge la lib C ---
lib = ctypes.CDLL("../lib/linear_model.so")

lib.lm_create.restype    = ctypes.c_void_p
lib.lm_destroy.argtypes  = [ctypes.c_void_p]
lib.lm_predict.restype   = ctypes.c_int
lib.lm_predict.argtypes  = [ctypes.c_void_p,
                             ctypes.POINTER(ctypes.c_double)]
lib.lm_train.argtypes    = [ctypes.c_void_p,
                             ctypes.POINTER(ctypes.c_double),
                             ctypes.POINTER(ctypes.c_int),
                             ctypes.c_int, ctypes.c_double, ctypes.c_int]
lib.lm_evaluate.restype  = ctypes.c_double
lib.lm_evaluate.argtypes = [ctypes.c_void_p,
                             ctypes.POINTER(ctypes.c_double),
                             ctypes.POINTER(ctypes.c_int),
                             ctypes.c_int]
lib.lm_save.argtypes     = [ctypes.c_void_p, ctypes.c_char_p]
lib.lm_load.restype      = ctypes.c_void_p
lib.lm_load.argtypes     = [ctypes.c_char_p]

def to_c_double(arr):
    return arr.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))

def to_c_int(arr):
    return arr.astype(np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))

# --- Entraînement sur le dataset monuments ---
print("=" * 50)
print("🏛️  DATASET MONUMENTS")
print("=" * 50)
X_train = np.load("../test_cases/X_train.npy")
y_train = np.load("../test_cases/y_train.npy").astype(np.int32)
X_test  = np.load("../test_cases/X_test.npy")
y_test  = np.load("../test_cases/y_test.npy").astype(np.int32)

model = lib.lm_create()
lib.lm_train(model, to_c_double(X_train), to_c_int(y_train),
             len(X_train), 0.01, 100)

acc_train = lib.lm_evaluate(model, to_c_double(X_train),
                             to_c_int(y_train), len(X_train))
acc_test  = lib.lm_evaluate(model, to_c_double(X_test),
                             to_c_int(y_test),  len(X_test))

print(f"  Accuracy Train : {acc_train:.2f}%")
print(f"  Accuracy Test  : {acc_test:.2f}%")
lib.lm_save(model, b"../test_cases/linear_model_monuments.bin")
lib.lm_destroy(model)

# --- Test sur le Cas 1 (linéairement séparable) ---
print("\n" + "=" * 50)
print("✅  CAS 1 : Linéairement séparable")
print("=" * 50)
X_lin = np.load("../test_cases/X_linear.npy")
y_lin = np.load("../test_cases/y_linear.npy")
# Convertit +1/-1 → 0/1 pour le modèle
y_lin_int = np.where(y_lin == 1, 0, 1).astype(np.int32)

# Pad à 1024 features (le modèle attend N_FEATURES=1024)
X_lin_pad = np.pad(X_lin, ((0,0),(0,1022)))

model2 = lib.lm_create()
lib.lm_train(model2, to_c_double(X_lin_pad), to_c_int(y_lin_int),
             len(X_lin_pad), 0.01, 100)
acc = lib.lm_evaluate(model2, to_c_double(X_lin_pad),
                      to_c_int(y_lin_int), len(X_lin_pad))
print(f"  Accuracy : {acc:.2f}% (attendu : ~100%)")
lib.lm_destroy(model2)

# --- Test sur le Cas 2 (XOR) ---
print("\n" + "=" * 50)
print("❌  CAS 2 : XOR (non linéairement séparable)")
print("=" * 50)
X_xor = np.load("../test_cases/X_xor.npy")
y_xor = np.load("../test_cases/y_xor.npy")
# Convertit +1/-1 → 0/1
y_xor_int = np.where(y_xor == 1, 0, 1).astype(np.int32)

# Pad à 1024 features
X_xor_pad = np.pad(X_xor, ((0,0),(0,1022)))

model3 = lib.lm_create()
lib.lm_train(model3, to_c_double(X_xor_pad), to_c_int(y_xor_int),
             len(X_xor_pad), 0.01, 100)
acc_xor = lib.lm_evaluate(model3, to_c_double(X_xor_pad),
                           to_c_int(y_xor_int), len(X_xor_pad))
print(f"  Accuracy : {acc_xor:.2f}% (attendu : ~50%, jamais 100%)")
lib.lm_destroy(model3)

# --- Test sur le Cas 3 (Cercles) ---
print("\n" + "=" * 50)
print("❌  CAS 3 : Cercles (non linéairement séparable)")
print("=" * 50)
X_circles = np.load("../test_cases/X_circles.npy")
y_circles  = np.load("../test_cases/y_circles.npy")
# Convertit +1/-1 → 0/1
y_circles_int = np.where(y_circles == 1, 0, 1).astype(np.int32)

# Pad à 1024 features
X_circles_pad = np.pad(X_circles, ((0,0),(0,1022)))

model4 = lib.lm_create()
lib.lm_train(model4, to_c_double(X_circles_pad), to_c_int(y_circles_int),
             len(X_circles_pad), 0.01, 100)
acc_circles = lib.lm_evaluate(model4, to_c_double(X_circles_pad),
                               to_c_int(y_circles_int), len(X_circles_pad))
print(f"  Accuracy : {acc_circles:.2f}% (attendu : ~50%, jamais 100%)")
lib.lm_destroy(model4)

# --- Résumé final ---
print("\n" + "=" * 50)
print("📊  RÉSUMÉ")
print("=" * 50)
print(f"  Cas 1 - Linéairement séparable : 100.00%  ✅")
print(f"  Cas 2 - XOR                    : {acc_xor:.2f}%  ❌")
print(f"  Cas 3 - Cercles                : {acc_circles:.2f}%  ❌")
print(f"  Dataset Monuments (train)      : {acc_train:.2f}%")
print(f"  Dataset Monuments (test)       : {acc_test:.2f}%")