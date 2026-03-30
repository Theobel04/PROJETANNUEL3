import numpy as np
import matplotlib
matplotlib.use('Agg') # pour tout récupérer en fichier
import matplotlib.pyplot as plt
import os # pour créer des répertoires si besoin

os.makedirs("../test_cases", exist_ok=True)

# Linéairement séparable (2 gaussiennes)
np.random.seed(42)
class1 = np.random.randn(100, 2) + [2, 2]
class2 = np.random.randn(100, 2) + [-2, -2]
labels1 = np.ones(100)
labels2 = -np.ones(100)

X_linear = np.vstack([class1, class2])
y_linear = np.hstack([labels1, labels2])
np.save("../test_cases/X_linear.npy", X_linear)
np.save("../test_cases/y_linear.npy", y_linear)

plt.figure()
plt.scatter(class1[:,0], class1[:,1], c='blue', label='Classe +1')
plt.scatter(class2[:,0], class2[:,1], c='red', label='Classe -1')
plt.title("Cas 1 : Linéairement séparable")
plt.legend()
plt.savefig("../test_cases/cas1_lineaire.png")
plt.close()

# XOR (non linéairement séparable)
X_xor = np.array([[1,1],[1,-1],[-1,1],[-1,-1],
                   [0.9,0.9],[0.9,-0.9],[-0.9,0.9],[-0.9,-0.9]])
y_xor = np.array([-1, 1, 1, -1, -1, 1, 1, -1])
np.save("../test_cases/X_xor.npy", X_xor)
np.save("../test_cases/y_xor.npy", y_xor)

plt.figure()
colors = ['blue' if y == 1 else 'red' for y in y_xor]
plt.scatter(X_xor[:,0], X_xor[:,1], c=colors)
plt.title("Cas 2 : XOR (non linéairement séparable)")
plt.savefig("../test_cases/cas2_xor.png")
plt.close()

# Cercles concentriques
from sklearn.datasets import make_circles
X_circles, y_circles = make_circles(n_samples=200, noise=0.1, factor=0.4)
y_circles = np.where(y_circles == 0, -1, 1)
np.save("../test_cases/X_circles.npy", X_circles)
np.save("../test_cases/y_circles.npy", y_circles)

plt.figure()
plt.scatter(X_circles[:,0], X_circles[:,1], c=y_circles, cmap='bwr')
plt.title("Cas 3 : Cercles (non linéairement séparable)")
plt.savefig("../test_cases/cas3_circles.png")
plt.close()
