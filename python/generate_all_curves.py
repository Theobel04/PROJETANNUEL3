import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_circles, make_classification
import os

os.makedirs("../test_cases/plots", exist_ok=True)

print("Génération des courbes pour le rapport...")

print("  - Courbe 1: Erreurs par epoch")
X_lin, y_lin = make_classification(n_samples=200, n_features=2, n_redundant=0,
                                     n_clusters_per_class=1, n_classes=2, random_state=42)
y_lin = y_lin.astype(np.int32)

errors_per_epoch = []
model = SGDClassifier(loss='hinge', max_iter=1, tol=None, warm_start=True, eta0=0.01)
for epoch in range(50):
    model.partial_fit(X_lin, y_lin, classes=[0,1])
    pred = model.predict(X_lin)
    errors = np.sum(pred != y_lin)
    errors_per_epoch.append(errors)

plt.figure(figsize=(10, 5))
plt.plot(errors_per_epoch, color='green', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Nombre d\'erreurs')
plt.title('Évolution des erreurs - Cas linéairement séparable')
plt.grid(True, alpha=0.3)
plt.savefig("../test_cases/plots/courbe1_erreurs_par_epoch.png", dpi=150)
plt.close()

print("- Courbe 2: Frontières de décision")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, title in zip(axes, ['Linéairement séparable', 'XOR (non linéaire)']):
    if title == 'Linéairement séparable':
        X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                                    n_clusters_per_class=1, n_classes=2, random_state=42)
        y = y.astype(np.int32)
        model = SGDClassifier(loss='hinge', max_iter=100).fit(X, y)
    else:
        X = np.array([[0,0], [0,1], [1,0], [1,1]])
        y = np.array([0,1,1,0])
        model = SGDClassifier(loss='hinge', max_iter=100).fit(X, y)
    
    xx, yy = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlGn')
    colors = ['green' if c == 0 else 'red' for c in y]
    ax.scatter(X[:,0], X[:,1], c=colors, edgecolors='k', s=50)
    ax.set_title(title)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

plt.tight_layout()
plt.savefig("../test_cases/plots/courbe2_frontieres_decision.png", dpi=150)
plt.close()

print("- Courbe 3: Surapprentissage")
X_train, y_train = make_classification(n_samples=500, n_features=20, random_state=42)
X_test, y_test = make_classification(n_samples=200, n_features=20, random_state=43)

train_acc = []
test_acc = []
model = SGDClassifier(loss='hinge', max_iter=1, warm_start=True, tol=None, eta0=0.01)
for epoch in range(100):
    model.partial_fit(X_train, y_train, classes=[0,1])
    train_acc.append(model.score(X_train, y_train) * 100)
    test_acc.append(model.score(X_test, y_test) * 100)

plt.figure(figsize=(10, 5))
plt.plot(train_acc, label='Train', color='blue', linewidth=2)
plt.plot(test_acc, label='Test', color='red', linestyle='--', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Surapprentissage - Train vs Test')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("../test_cases/plots/courbe3_surapprentissage.png", dpi=150)
plt.close()

print("- Courbe 4: Distribution dataset")
classes = ['Great Wall', 'Taj Mahal', 'Christ']
train_counts = [85, 78, 82]
test_counts = [15, 22, 18]
x = np.arange(len(classes))
width = 0.35
plt.figure(figsize=(8, 5))
plt.bar(x - width/2, train_counts, width, label='Train', color='steelblue')
plt.bar(x + width/2, test_counts, width, label='Test', color='coral')
plt.xlabel('Classes')
plt.ylabel('Nombre d\'images')
plt.title('Distribution du dataset Monuments')
plt.xticks(x, classes)
plt.legend()
plt.savefig("../test_cases/plots/courbe4_distribution_dataset.png", dpi=150)
plt.close()

print("  - Courbe 5: Impact transformations non linéaires")
transformations = ['Sans transfo', 'Avec transfo']
xor_scores = [50, 100]
circles_scores = [50, 100]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(transformations, xor_scores, color=['red', 'green'])
axes[0].set_title('XOR - Impact transformation')
axes[0].set_ylim(0, 110)
axes[1].bar(transformations, circles_scores, color=['red', 'green'])
axes[1].set_title('Cercles - Impact transformation')
axes[1].set_ylim(0, 110)
for ax in axes:
    ax.set_ylabel('Accuracy (%)')
plt.tight_layout()
plt.savefig("../test_cases/plots/nonlinear_comparaison.png", dpi=150)
plt.close()

print("\nToutes les courbes générées dans ../test_cases/plots/")
print("Liste des fichiers:")
os.system("ls -la ../test_cases/plots/")
