import numpy as np
from sklearn.linear_model import SGDClassifier

print("="*50)
print("TEST XOR - Modèle Linéaire vs MLP")
print("="*50)

X_xor = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float64)
y_xor = np.array([0, 1, 1, 0])

print("\nDonnées XOR:")
for i in range(4):
    print(f"  {X_xor[i]} -> {y_xor[i]}")

print("\n--- Modèle Linéaire ---")
linear = SGDClassifier(loss='hinge', max_iter=1000, tol=None, eta0=0.1)
linear.fit(X_xor, y_xor)
linear_acc = linear.score(X_xor, y_xor) * 100
print(f"Accuracy: {linear_acc:.1f}%")
print("Le modèle linéaire ne peut pas résoudre XOR (car non linéaire)")


print("\n--- Après transformation (x1*x2) ---")
X_xor_transformed = np.column_stack([X_xor, X_xor[:,0] * X_xor[:,1]])
print(f"Nouvelles features: (x1, x2, x1*x2)")
linear2 = SGDClassifier(loss='hinge', max_iter=100, tol=None, eta0=0.1)
linear2.fit(X_xor_transformed, y_xor)
linear2_acc = linear2.score(X_xor_transformed, y_xor) * 100
print(f"Accuracy après transformation: {linear2_acc:.1f}%")

print("\nConclusion: XOR nécessite une transformation non linéaire")
