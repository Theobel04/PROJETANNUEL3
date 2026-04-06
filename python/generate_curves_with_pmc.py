import ctypes
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("../test_cases/plots", exist_ok=True)

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

X_train = np.load("../test_cases/X_train.npy")
y_train = np.load("../test_cases/y_train.npy")
X_test = np.load("../test_cases/X_test.npy")
y_test = np.load("../test_cases/y_test.npy")

print(f"Dataset: Train {X_train.shape}, Test {X_test.shape}")
print("Entraînement du PMC...")

n_train = ctypes.c_int(len(X_train))
n_test = ctypes.c_int(len(X_test))

model = lib.mlp_create(1024, 64, 3)
lib.mlp_train(model, 
              to_c_double(X_train), 
              to_c_int(y_train), 
              n_train,  
              0.01,     
              200,      
              32)       

train_acc = lib.mlp_evaluate(model, to_c_double(X_train), to_c_int(y_train), n_train)
test_acc = lib.mlp_evaluate(model, to_c_double(X_test), to_c_int(y_test), n_test)

print(f"\nTrain Accuracy: {train_acc:.2f}%")
print(f"Test Accuracy: {test_acc:.2f}%")

labels = ['Modèle Linéaire', 'PMC (C)']
train_scores = [100.0, train_acc]
test_scores = [57.58, test_acc]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, train_scores, width, label='Train', color='steelblue')
bars2 = ax.bar(x + width/2, test_scores, width, label='Test', color='coral')

ax.set_ylabel('Accuracy (%)')
ax.set_title('Comparaison Modèle Linéaire vs PMC')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.set_ylim(0, 110)

for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

plt.tight_layout()
plt.savefig("../test_cases/plots/comparison_linear_vs_pmc.png", dpi=150)
plt.close()
print("Graphique sauvegardé: comparison_linear_vs_pmc.png")

lib.mlp_destroy(model)
print("\nCourbe OK!")
