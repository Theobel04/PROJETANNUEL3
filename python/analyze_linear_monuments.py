import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os

os.makedirs("../test_cases/plots", exist_ok=True)

print("="*60)
print("ANALYSE MODÈLE LINÉAIRE - DATASET MONUMENTS")
print("Version Python Pure (sans compilation C)")
print("="*60)

try:
    X_train = np.load("../test_cases/X_train.npy")
    y_train = np.load("../test_cases/y_train.npy")
    X_test = np.load("../test_cases/X_test.npy")
    y_test = np.load("../test_cases/y_test.npy")
    print(f"Données chargées: Train={X_train.shape}, Test={X_test.shape}")
except FileNotFoundError:
    print("Données non trouvées. Lancez d'abord: python preprocess.py")
    exit(1)

print("\nEntraînement en cours...")
model = SGDClassifier(
    loss='hinge',           
    max_iter=200,           
    tol=None,               
    eta0=0.01,              
    learning_rate='constant',
    random_state=42
)

model.fit(X_train, y_train)
print("Entraînement terminé")

train_acc = model.score(X_train, y_train) * 100
test_acc = model.score(X_test, y_test) * 100
print(f"\n Accuracy:")
print(f"  Train: {train_acc:.2f}%")
print(f"  Test:  {test_acc:.2f}%")

predictions = model.predict(X_test)

cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Great Wall', 'Taj Mahal', 'Christ'],
            yticklabels=['Great Wall', 'Taj Mahal', 'Christ'])
plt.title('Matrice de confusion - Modèle Linéaire (Scikit-learn)')
plt.ylabel('Vérité terrain')
plt.xlabel('Prédiction')
plt.savefig("../test_cases/plots/confusion_matrix_linear.png", dpi=150, bbox_inches='tight')
plt.close()
print("Matrice de confusion sauvegardée")

print("\nClassification Report:")
print(classification_report(y_test, predictions,
      target_names=['Great Wall', 'Taj Mahal', 'Christ']))

misclassified = np.sum(predictions != y_test)
print(f"\nImages mal classées : {misclassified}/{len(y_test)} ({misclassified/len(y_test)*100:.1f}%)")

print("\nGénération des courbes...")
epochs = range(1, 51)
train_scores = []
test_scores = []

from sklearn.linear_model import SGDClassifier
for epoch in epochs:
    model_epoch = SGDClassifier(loss='hinge', max_iter=epoch, tol=None, 
                                 eta0=0.01, learning_rate='constant', random_state=42)
    model_epoch.fit(X_train, y_train)
    train_scores.append(model_epoch.score(X_train, y_train) * 100)
    test_scores.append(model_epoch.score(X_test, y_test) * 100)

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_scores, label='Train Accuracy', color='blue', linewidth=2)
plt.plot(epochs, test_scores, label='Test Accuracy', color='red', linestyle='--', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Courbe d\'apprentissage - Surapprentissage visible')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("../test_cases/plots/learning_curve.png", dpi=150, bbox_inches='tight')
plt.close()
print("Courbe d'apprentissage sauvegardée")

plt.figure(figsize=(8, 5))
classes = ['Great Wall', 'Taj Mahal', 'Christ']
train_counts = np.bincount(y_train)
test_counts = np.bincount(y_test)
x = np.arange(len(classes))
width = 0.35
plt.bar(x - width/2, train_counts, width, label='Train', color='steelblue')
plt.bar(x + width/2, test_counts, width, label='Test', color='coral')
plt.xlabel('Classes')
plt.ylabel('Nombre d\'images')
plt.title('Distribution du dataset')
plt.xticks(x, classes)
plt.legend()
plt.savefig("../test_cases/plots/dataset_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("Distribution du dataset sauvegardée")

print("\n" + "="*60)
print("ANALYSE TERMINÉE")
print("="*60)
print("\nFichiers générés dans ../test_cases/plots/:")
print("  - confusion_matrix_linear.png")
print("  - learning_curve.png")
print("  - dataset_distribution.png")
EOF