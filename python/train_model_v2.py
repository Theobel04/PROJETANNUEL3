"""
Script d'entraînement amélioré du PMC
"""

import ctypes
import numpy as np
import os

lib_path = r"C:/Users/jumet/OneDrive/Documents/PA_Classification/PROJETANNUEL3/lib/mlp.so"
lib = ctypes.CDLL(lib_path)

lib.mlp_create.restype = ctypes.c_void_p
lib.mlp_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]

lib.mlp_train.argtypes = [ctypes.c_void_p, 
                          ctypes.POINTER(ctypes.c_double),
                          ctypes.POINTER(ctypes.c_int), 
                          ctypes.c_int,
                          ctypes.c_double, 
                          ctypes.c_int, 
                          ctypes.c_int]

lib.mlp_evaluate.restype = ctypes.c_double
lib.mlp_evaluate.argtypes = [ctypes.c_void_p, 
                             ctypes.POINTER(ctypes.c_double),
                             ctypes.POINTER(ctypes.c_int), 
                             ctypes.c_int]

lib.mlp_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.mlp_destroy.argtypes = [ctypes.c_void_p]

def to_c_double(arr):
    return arr.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))

def to_c_int(arr):
    return arr.astype(np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))

print("ENTRAÎNEMENT AMÉLIORÉ DU PMC")

X_train = np.load("../test_cases/X_train.npy")
y_train = np.load("../test_cases/y_train.npy")
X_test = np.load("../test_cases/X_test.npy")
y_test = np.load("../test_cases/y_test.npy")

print(f"Train: {len(X_train)} images")
print(f"Test: {len(X_test)} images")

print("\nCréation d'un modèle plus grand...")
model = lib.mlp_create(1024, 128, 3)
print("Architecture: 1024 → 128 → 3")

print("\nEntraînement (500 epochs)...")
print("Cela peut prendre 3-5 minutes...")
lib.mlp_train(model, 
              to_c_double(X_train), 
              to_c_int(y_train), 
              len(X_train),
              0.01,     
              500,      
              32)

train_acc = lib.mlp_evaluate(model, to_c_double(X_train), to_c_int(y_train), len(X_train))
test_acc = lib.mlp_evaluate(model, to_c_double(X_test), to_c_int(y_test), len(X_test))

print(f"\nTrain Accuracy: {train_acc:.2f}%")
print(f"Test Accuracy: {test_acc:.2f}%")

model_file = "../test_cases/pmc_model_v2.bin"
lib.mlp_save(model, model_file.encode('utf-8'))
print(f"Modèle sauvegardé: {model_file}")

lib.mlp_destroy(model)
print("\nEntraînement terminé !")
