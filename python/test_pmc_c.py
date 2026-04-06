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

print("PMC (PERCEPTRON MULTI-COUCHES) EN C")

print("\n1. TEST XOR")
X_xor = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float64)
y_xor = np.array([0, 1, 1, 0], dtype=np.int32)

model = lib.mlp_create(2, 4, 2)
print("Entraînement sur XOR...")
lib.mlp_train(model, to_c_double(X_xor), to_c_int(y_xor), 4, 0.8, 1000, 4)
acc = lib.mlp_evaluate(model, to_c_double(X_xor), to_c_int(y_xor), 4)
print(f"\nAccuracy XOR: {acc:.1f}%")

print("\nPrédictions:")
for i in range(4):
    x_ptr = to_c_double(X_xor[i:i+1])
    pred = lib.mlp_predict(model, x_ptr)
    print(f"  {X_xor[i]} -> {pred} (attendu: {y_xor[i]})")

lib.mlp_destroy(model)

print("\n2. TEST DATASET MONUMENTS")
try:
    X_train = np.load("../test_cases/X_train.npy")
    y_train = np.load("../test_cases/y_train.npy")
    X_test = np.load("../test_cases/X_test.npy")
    y_test = np.load("../test_cases/y_test.npy")
    
    print(f"Images: {len(X_train)} train, {len(X_test)} test")
    print("Entraînement du PMC sur les monuments...")
    
    model2 = lib.mlp_create(1024, 64, 3)
    lib.mlp_train(model2, to_c_double(X_train), to_c_int(y_train), 
                  len(X_train), 0.01, 200, 32)
    
    train_acc = lib.mlp_evaluate(model2, to_c_double(X_train), to_c_int(y_train), len(X_train))
    test_acc = lib.mlp_evaluate(model2, to_c_double(X_test), to_c_int(y_test), len(X_test))
    
    print(f"\nTrain Accuracy: {train_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    lib.mlp_destroy(model2)
    
except FileNotFoundError:
    print("Dataset non trouvé. Nécéssite le fichier: python preprocess.py")

print("TESTS TERMINÉS")