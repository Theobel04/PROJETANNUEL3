import ctypes
import numpy as np

lib_path = r"C:/Users/jumet/OneDrive/Documents/PA_Classification/PROJETANNUEL3/lib/mlp.so"
lib = ctypes.CDLL(lib_path)

lib.mlp_predict.restype = ctypes.c_int
lib.mlp_predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
lib.mlp_load.restype = ctypes.c_void_p
lib.mlp_load.argtypes = [ctypes.c_char_p]

def to_c_double(arr):
    return arr.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))

model_file = "../test_cases/pmc_model.bin"
model = lib.mlp_load(model_file.encode('utf-8'))
print(f"Modèle chargé")

X_test = np.load("../test_cases/X_test.npy")
y_test = np.load("../test_cases/y_test.npy")

print(f"\nTest sur {len(X_test)} images")
print("="*50)

confusion = np.zeros((3, 3), dtype=int)

for i in range(len(X_test)):
    x = X_test[i]
    x_ptr = to_c_double(x.reshape(1, -1))
    pred = lib.mlp_predict(model, x_ptr)
    true = y_test[i]
    confusion[true][pred] += 1

print("\nMatrice de confusion (lignes=vérité, colonnes=prédiction):")
print("            Pred:0     Pred:1     Pred:2")
for i in range(3):
    print(f"Vérité {i}:   {confusion[i][0]:^9} {confusion[i][1]:^9} {confusion[i][2]:^9}")

print("\nInterprétation:")
print("  - Colonne 0 = Christ the Redeemer")
print("  - Colonne 1 = Great Wall of China")  
print("  - Colonne 2 = Taj Mahal")

print("\nAccuracy par classe:")
for i in range(3):
    total = np.sum(confusion[i])
    correct = confusion[i][i]
    acc = correct/total*100 if total > 0 else 0
    print(f"  Classe {i}: {acc:.1f}% ({correct}/{total})")

lib.mlp_destroy(model)
