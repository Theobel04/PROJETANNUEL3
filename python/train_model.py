"""
Script d'entraînement du PMC
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

print("ENTRAÎNEMENT DU PMC (Perceptron Multi-Couches)")

print("\n1. Chargement des données...")
X_train = np.load("../test_cases/X_train.npy")
y_train = np.load("../test_cases/y_train.npy")
X_test = np.load("../test_cases/X_test.npy")
y_test = np.load("../test_cases/y_test.npy")

print(f"   Images d'entraînement: {len(X_train)}")
print(f"   Images de test: {len(X_test)}")
print(f"   Taille des images: {X_train.shape[1]} pixels")

print("\n2. Création du modèle...")
model = lib.mlp_create(1024, 64, 3)
print("   Architecture: 1024 entrées → 64 neurones cachés → 3 sorties")

print("\n3. Entraînement en cours (200 epochs)...")
print("   Cela peut prendre 1-2 minutes...")
print("-" * 40)

lib.mlp_train(model, 
              to_c_double(X_train), 
              to_c_int(y_train), 
              len(X_train),
              0.01,     
              200,      
              32)       

print("\n" + "-" * 40)
print("4. Évaluation du modèle...")
train_acc = lib.mlp_evaluate(model, to_c_double(X_train), to_c_int(y_train), len(X_train))
test_acc = lib.mlp_evaluate(model, to_c_double(X_test), to_c_int(y_test), len(X_test))

print(f"   Accuracy sur l'entraînement: {train_acc:.2f}%")
print(f"   Accuracy sur le test: {test_acc:.2f}%")

print("\n5. Sauvegarde du modèle...")
model_file = "../test_cases/pmc_model.bin"
lib.mlp_save(model, model_file.encode('utf-8'))
print(f"   Modèle sauvegardé: {model_file}")

lib.mlp_destroy(model)

print("ENTRAÎNEMENT TERMINÉ !")
