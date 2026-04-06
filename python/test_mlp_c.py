import ctypes
import numpy as np
import matplotlib.pyplot as plt

lib = ctypes.CDLL("../lib/mlp.so")

lib.mlp_create.restype = ctypes.c_void_p
lib.mlp_train.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double),
                          ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                          ctypes.c_double, ctypes.c_int, ctypes.c_int]
lib.mlp_predict.restype = ctypes.c_int
lib.mlp_evaluate.restype = ctypes.c_double

def to_c_double(arr):
    return arr.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))

def to_c_int(arr):
    return arr.astype(np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))

X_xor = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float64)
y_xor = np.array([0, 1, 1, 0], dtype=np.int32)

model = lib.mlp_create(2, 4, 2)
lib.mlp_train(model, to_c_double(X_xor), to_c_int(y_xor), 4, 0.5, 500, 4)
acc = lib.mlp_evaluate(model, to_c_double(X_xor), to_c_int(y_xor), 4)
print(f"XOR Accuracy: {acc:.1f}%")