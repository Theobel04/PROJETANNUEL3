import ctypes
import numpy as np
import os

lib_path = r"C:/Users/jumet/OneDrive/Documents/PA_Classification/PROJETANNUEL3/lib/mlp.so"
print(f"Chargement: {lib_path}")
print(f"Existe: {os.path.exists(lib_path)}")

lib = ctypes.CDLL(lib_path)
print("Librairie chargée")

lib.mlp_create.restype = ctypes.c_void_p
lib.mlp_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]

lib.mlp_train.argtypes = [ctypes.c_void_p, 
                          ctypes.POINTER(ctypes.c_double),
                          ctypes.POINTER(ctypes.c_int), 
                          ctypes.c_int,
                          ctypes.c_double, 
                          ctypes.c_int, 
                          ctypes.c_int]

lib.mlp_predict.restype = ctypes.c_int
lib.mlp_predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]

lib.mlp_evaluate.restype = ctypes.c_double
lib.mlp_evaluate.argtypes = [ctypes.c_void_p, 
                             ctypes.POINTER(ctypes.c_double),
                             ctypes.POINTER(ctypes.c_int), 
                             ctypes.c_int]

lib.mlp_destroy.argtypes = [ctypes.c_void_p]

def to_c_double(arr):
    return arr.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))

def to_c_int(arr):
    return arr.astype(np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))

print("\n1. Test XOR")
X_xor = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float64)
y_xor = np.array([0, 1, 1, 0], dtype=np.int32)

model = lib.mlp_create(2, 4, 2)
print("Modèle créé")

lib.mlp_train(model, to_c_double(X_xor), to_c_int(y_xor), 4, 0.5, 500, 4)
acc = lib.mlp_evaluate(model, to_c_double(X_xor), to_c_int(y_xor), 4)
print(f"Accuracy XOR: {acc:.1f}%")

lib.mlp_destroy(model)
print("\nTest terminé")
