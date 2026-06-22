# 🧠 30 Days of Deep Learning from Scratch

> **Auteur** : Programme conçu par un expert en Deep Learning & professeur universitaire senior  
> **Niveau** : Débutant → Intermédiaire → Avancé  
> **Prérequis** : Python de base, notions de mathématiques (lycée)  
> **Outils** : Python 3.10+, NumPy, Matplotlib, puis PyTorch

---

## 🎯 Objectif Global

À la fin de ces 30 jours, vous serez capable de :

1. **Comprendre** les fondements mathématiques du Deep Learning (dérivées, gradient, optimisation)
2. **Implémenter from scratch** un réseau de neurones multicouche complet avec NumPy
3. **Construire** un CNN (réseau convolutif) from scratch
4. **Maîtriser** les techniques d'optimisation modernes (SGD, Momentum, Adam)
5. **Utiliser PyTorch** pour reconstruire et déployer vos modèles
6. **Réaliser** un projet complet de classification d'images (MNIST)

---

## 📈 Progression Pédagogique

```
Semaine 1 (J1-J5)   : 🧱 Fondations — Le neurone, les activations, la descente de gradient
Semaine 2 (J6-J10)  : ⚙️ Mécanique — Régression logistique, backpropagation, fonctions de perte
Semaine 3 (J11-J15) : 🏗️ Construction — Réseaux de neurones multicouches (MLP) from scratch
Semaine 4 (J16-J20) : 🔧 Optimisation — SGD, Momentum, Adam, régularisation, dropout
Semaine 5 (J21-J25) : 👁️ Vision — CNN from scratch (convolution, pooling, architectures)
Semaine 6 (J26-J30) : 🚀 Intégration — PyTorch + Projet final MNIST
```

---

## 💡 Conseils pour Pratiquer Efficacement

1. **Tapez chaque ligne de code vous-même** — ne copiez-collez jamais
2. **Expérimentez** — changez les hyperparamètres, cassez le code, observez
3. **Dessinez** — schématisez les réseaux et les flux de données sur papier
4. **Expliquez à voix haute** — si vous ne pouvez pas l'expliquer simplement, vous ne le comprenez pas
5. **Revoyez le jour précédent** avant de commencer le nouveau jour
6. **Codez au moins 2h par jour** — la régularité bat l'intensité
7. **Tenez un journal** — notez vos questions, vos erreurs, vos "aha moments"

---

---

# 📅 SEMAINE 1 : FONDATIONS (Jours 1–5)

---

## 📅 Jour 1 — Le Neurone Artificiel : La Brique Fondamentale

### 📚 Concept Théorique

Un **neurone artificiel** est la plus petite unité de calcul d'un réseau de neurones. Il est inspiré du neurone biologique : il reçoit des signaux en entrée, les traite, et produit une sortie.

Un neurone fait deux choses :
1. **Combinaison linéaire** : il multiplie chaque entrée par un poids, puis ajoute un biais
2. **Activation** : il passe le résultat dans une fonction qui décide de la sortie

### 🧠 Intuition

Imaginez un neurone comme un **juge** qui doit prendre une décision (oui/non).

- Les **entrées** ($x_1, x_2, ...$) sont les preuves présentées
- Les **poids** ($w_1, w_2, ...$) représentent l'importance de chaque preuve
- Le **biais** ($b$) est le préjugé initial du juge
- La **fonction d'activation** est le seuil de décision

### 🧮 Formules Essentielles

**Combinaison linéaire (somme pondérée) :**

$$z = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b = \sum_{i=1}^{n} w_i x_i + b$$

**Notation vectorielle :**

$$z = \mathbf{w}^T \mathbf{x} + b$$

**Sortie du neurone :**

$$\hat{y} = f(z)$$

où $f$ est la fonction d'activation.

### 💻 Implémentation From Scratch

```python
import numpy as np

# ============================================================
# JOUR 1 : Le Neurone Artificiel
# ============================================================

# --- Étape 1 : Définir les entrées ---
# Imaginons un neurone qui décide si on va faire du sport
# x1 = météo (1 = beau, 0 = mauvais)
# x2 = énergie (1 = haute, 0 = basse)
# x3 = temps libre (1 = oui, 0 = non)

x = np.array([1, 0, 1])  # beau temps, pas d'énergie, temps libre
print(f"Entrées : {x}")

# --- Étape 2 : Initialiser les poids et le biais ---
# Les poids représentent l'importance de chaque facteur
w = np.array([0.6, 0.3, 0.2])  # la météo compte le plus
b = -0.5                        # biais (seuil de base)

print(f"Poids   : {w}")
print(f"Biais   : {b}")

# --- Étape 3 : Calculer la somme pondérée ---
# z = w1*x1 + w2*x2 + w3*x3 + b
z = np.dot(w, x) + b  # np.dot fait le produit scalaire
print(f"\nSomme pondérée z = {z}")

# --- Étape 4 : Appliquer une fonction d'activation (Step Function) ---
# La plus simple : si z > 0, on sort 1 (oui), sinon 0 (non)
def step_function(z):
    """Fonction marche : retourne 1 si z > 0, sinon 0."""
    return 1 if z > 0 else 0

output = step_function(z)
print(f"Sortie du neurone : {output}")
print(f"Décision : {'Oui, on fait du sport !' if output == 1 else 'Non, on reste à la maison.'}")

# --- Étape 5 : Encapsuler dans une classe ---
class Neuron:
    """Un neurone artificiel simple."""

    def __init__(self, n_inputs):
        """
        Initialise le neurone avec des poids aléatoires.

        Args:
            n_inputs: nombre d'entrées du neurone
        """
        # Poids aléatoires petits (distribution normale)
        self.weights = np.random.randn(n_inputs) * 0.1
        # Biais initialisé à zéro
        self.bias = 0.0

    def forward(self, x):
        """
        Calcul de la sortie du neurone (propagation avant).

        Args:
            x: vecteur d'entrées (numpy array)

        Returns:
            Sortie du neurone après activation
        """
        # Étape 1 : somme pondérée
        z = np.dot(self.weights, x) + self.bias
        # Étape 2 : activation (step function)
        return step_function(z)

# --- Test de la classe ---
print("\n--- Test avec la classe Neuron ---")
neuron = Neuron(n_inputs=3)
print(f"Poids initiaux : {neuron.weights}")
print(f"Biais initial  : {neuron.bias}")

# Tester avec plusieurs entrées
test_inputs = [
    np.array([1, 1, 1]),  # tout est favorable
    np.array([0, 0, 0]),  # rien n'est favorable
    np.array([1, 0, 1]),  # partiellement favorable
]

for inp in test_inputs:
    result = neuron.forward(inp)
    print(f"  Entrée {inp} → Sortie {result}")
```


---

## 📅 Jour 2 — Les Fonctions d'Activation

### 📚 Concept Théorique

La **fonction d'activation** est ce qui rend un réseau de neurones puissant. Sans elle, un empilement de neurones ne serait qu'une simple transformation linéaire (une droite). L'activation introduit de la **non-linéarité**, ce qui permet au réseau d'apprendre des relations complexes.

### 🧠 Intuition

Pensez aux fonctions d'activation comme des **filtres** :
- **Step** : interrupteur ON/OFF (trop brutal, pas de nuance)
- **Sigmoid** : un variateur de lumière (doux, entre 0 et 1)
- **Tanh** : comme sigmoid mais centré sur zéro (-1 à 1)
- **ReLU** : un clapet anti-retour (laisse passer le positif, bloque le négatif)

### 🧮 Formules Essentielles

**Sigmoid :**
$$\sigma(z) = \frac{1}{1 + e^{-z}} \qquad \text{sortie} \in (0, 1)$$

**Dérivée de Sigmoid :**
$$\sigma'(z) = \sigma(z) \cdot (1 - \sigma(z))$$

**Tanh :**
$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} \qquad \text{sortie} \in (-1, 1)$$

**Dérivée de Tanh :**
$$\tanh'(z) = 1 - \tanh^2(z)$$

**ReLU (Rectified Linear Unit) :**
$$\text{ReLU}(z) = \max(0, z) \qquad \text{sortie} \in [0, +\infty)$$

**Dérivée de ReLU :**
$$\text{ReLU}'(z) = \begin{cases} 1 & \text{si } z > 0 \\ 0 & \text{si } z \leq 0 \end{cases}$$

### 💻 Implémentation From Scratch

```python
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# JOUR 2 : Les Fonctions d'Activation
# ============================================================

# --- 1. Sigmoid ---
def sigmoid(z):
    """
    Fonction sigmoid : écrase les valeurs entre 0 et 1.
    Utile pour les probabilités (classification binaire).
    """
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    """Dérivée de sigmoid : utilisée pour la backpropagation."""
    s = sigmoid(z)
    return s * (1 - s)  # formule élégante !

# --- 2. Tanh ---
def tanh(z):
    """
    Tangente hyperbolique : écrase entre -1 et 1.
    Mieux que sigmoid car centrée sur zéro.
    """
    return np.tanh(z)  # NumPy le fait nativement

def tanh_derivative(z):
    """Dérivée de tanh."""
    return 1 - np.tanh(z) ** 2

# --- 3. ReLU ---
def relu(z):
    """
    ReLU : la plus populaire aujourd'hui.
    Simple, efficace, rapide à calculer.
    Retourne z si z > 0, sinon 0.
    """
    return np.maximum(0, z)

def relu_derivative(z):
    """Dérivée de ReLU : 1 si z > 0, sinon 0."""
    return (z > 0).astype(float)

# --- 4. Leaky ReLU ---
def leaky_relu(z, alpha=0.01):
    """
    Leaky ReLU : variante de ReLU qui laisse passer
    un petit gradient pour les valeurs négatives.
    Résout le problème du "dying ReLU".
    """
    return np.where(z > 0, z, alpha * z)

def leaky_relu_derivative(z, alpha=0.01):
    """Dérivée de Leaky ReLU."""
    return np.where(z > 0, 1, alpha)

# --- Visualisation ---
z = np.linspace(-5, 5, 200)  # 200 points entre -5 et 5

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Fonctions d'Activation et leurs Dérivées", fontsize=16, fontweight='bold')

# Liste des fonctions et leurs noms
activations = [
    ("Sigmoid", sigmoid, sigmoid_derivative),
    ("Tanh", tanh, tanh_derivative),
    ("ReLU", relu, relu_derivative),
    ("Leaky ReLU", leaky_relu, leaky_relu_derivative),
]

for i, (name, func, deriv) in enumerate(activations):
    # Fonction
    axes[0, i].plot(z, func(z), 'b-', linewidth=2)
    axes[0, i].set_title(f"{name}", fontweight='bold')
    axes[0, i].axhline(y=0, color='k', linewidth=0.5)
    axes[0, i].axvline(x=0, color='k', linewidth=0.5)
    axes[0, i].grid(True, alpha=0.3)

    # Dérivée
    axes[1, i].plot(z, deriv(z), 'r-', linewidth=2)
    axes[1, i].set_title(f"Dérivée de {name}", fontweight='bold')
    axes[1, i].axhline(y=0, color='k', linewidth=0.5)
    axes[1, i].axvline(x=0, color='k', linewidth=0.5)
    axes[1, i].grid(True, alpha=0.3)

axes[0, 0].set_ylabel("f(z)", fontsize=12)
axes[1, 0].set_ylabel("f'(z)", fontsize=12)

plt.tight_layout()
plt.savefig("activation_functions.png", dpi=150)
plt.show()

# --- Comparaison numérique ---
print("=== Comparaison des activations pour z = [-2, -1, 0, 1, 2] ===\n")
test_values = np.array([-2, -1, 0, 1, 2])

print(f"{'z':>5} | {'Sigmoid':>8} | {'Tanh':>8} | {'ReLU':>8} | {'Leaky ReLU':>11}")
print("-" * 55)
for val in test_values:
    print(f"{val:>5} | {sigmoid(val):>8.4f} | {tanh(val):>8.4f} | "
          f"{relu(val):>8.4f} | {leaky_relu(val):>11.4f}")
```

### 📌 Quand utiliser quelle activation ?

| Activation | Quand l'utiliser | Avantages | Inconvénients |
|---|---|---|---|
| Sigmoid | Couche de sortie (classif. binaire) | Probabilité [0,1] | Vanishing gradient |
| Tanh | Couches cachées (ancien) | Centrée sur 0 | Vanishing gradient |
| ReLU | Couches cachées (défaut) | Rapide, pas de vanishing | Dying neurons |
| Leaky ReLU | Couches cachées | Résout dying ReLU | Hyperparamètre α |

---

## 📅 Jour 3 — La Fonction de Coût (Loss Function)

### 📚 Concept Théorique

La **fonction de coût** (ou *loss function*) mesure **à quel point notre modèle se trompe**. C'est le "bulletin de notes" du réseau. L'objectif de l'apprentissage est de **minimiser** cette fonction.

### 🧠 Intuition

Imaginez que vous jouez aux fléchettes :
- La **cible** est la vraie réponse ($y$)
- Votre **lancer** est la prédiction du modèle ($\hat{y}$)
- La **distance au centre** est la loss
- **Apprendre** = ajuster votre technique pour vous rapprocher du centre

### 🧮 Formules Essentielles

**MSE (Mean Squared Error) — pour la régression :**
$$\mathcal{L}_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**Binary Cross-Entropy — pour la classification binaire :**
$$\mathcal{L}_{\text{BCE}} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

### 💻 Implémentation From Scratch

```python
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# JOUR 3 : Les Fonctions de Coût (Loss Functions)
# ============================================================

# --- 1. Mean Squared Error (MSE) ---
def mse_loss(y_true, y_pred):
    """
    Erreur quadratique moyenne.
    Pénalise fortement les grandes erreurs (à cause du carré).

    Args:
        y_true: valeurs réelles (numpy array)
        y_pred: prédictions du modèle (numpy array)

    Returns:
        Scalaire : la loss moyenne
    """
    return np.mean((y_true - y_pred) ** 2)

def mse_gradient(y_true, y_pred):
    """
    Gradient de MSE par rapport à y_pred.
    C'est cette dérivée qu'on utilise pour la backpropagation.
    """
    n = len(y_true)
    return -2 / n * (y_true - y_pred)  # = 2/n * (y_pred - y_true)

# --- 2. Binary Cross-Entropy (BCE) ---
def binary_cross_entropy(y_true, y_pred):
    """
    Entropie croisée binaire.
    La loss standard pour la classification binaire.
    
    Note : on ajoute epsilon pour éviter log(0) = -inf
    """
    epsilon = 1e-15  # tout petit nombre pour la stabilité numérique
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # clamp entre ε et 1-ε
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

def bce_gradient(y_true, y_pred):
    """
    Gradient de BCE par rapport à y_pred.
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return (-y_true / y_pred + (1 - y_true) / (1 - y_pred)) / len(y_true)

# --- Démonstration : voir comment la loss change ---
print("=== Démonstration MSE ===")
y_true = np.array([1.0, 0.0, 1.0, 1.0])

# Simuler des prédictions de plus en plus précises
predictions = [
    np.array([0.0, 1.0, 0.0, 0.0]),  # tout faux
    np.array([0.5, 0.5, 0.5, 0.5]),  # au milieu
    np.array([0.8, 0.2, 0.8, 0.8]),  # presque bon
    np.array([1.0, 0.0, 1.0, 1.0]),  # parfait
]

for pred in predictions:
    loss = mse_loss(y_true, pred)
    print(f"  Prédiction: {pred} → MSE = {loss:.4f}")

print("\n=== Démonstration Cross-Entropy ===")
for pred in predictions:
    pred_clipped = np.clip(pred, 0.01, 0.99)  # éviter 0 et 1 exacts
    loss = binary_cross_entropy(y_true, pred_clipped)
    print(f"  Prédiction: {pred_clipped} → BCE = {loss:.4f}")

# --- Visualisation : Loss en fonction de la prédiction ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

y_pred_range = np.linspace(0.01, 0.99, 100)

# Pour y_true = 1 : on veut que y_pred → 1
mse_when_true = (1 - y_pred_range) ** 2
bce_when_true = -np.log(y_pred_range)

axes[0].plot(y_pred_range, mse_when_true, 'b-', label='MSE', linewidth=2)
axes[0].plot(y_pred_range, bce_when_true, 'r-', label='BCE', linewidth=2)
axes[0].set_title("Quand y_true = 1", fontweight='bold')
axes[0].set_xlabel("ŷ (prédiction)")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Pour y_true = 0 : on veut que y_pred → 0
mse_when_false = y_pred_range ** 2
bce_when_false = -np.log(1 - y_pred_range)

axes[1].plot(y_pred_range, mse_when_false, 'b-', label='MSE', linewidth=2)
axes[1].plot(y_pred_range, bce_when_false, 'r-', label='BCE', linewidth=2)
axes[1].set_title("Quand y_true = 0", fontweight='bold')
axes[1].set_xlabel("ŷ (prédiction)")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("Comparaison MSE vs Cross-Entropy", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("loss_functions.png", dpi=150)
plt.show()
```

### 📌 Résumé : Quelle loss pour quel problème ?

| Problème | Loss recommandée | Pourquoi |
|---|---|---|
| Régression (prédire un nombre) | MSE | Mesure la distance au carré |
| Classification binaire (oui/non) | Binary Cross-Entropy | Pénalise fortement les erreurs confiantes |
| Classification multi-classes | Categorical Cross-Entropy | Extension naturelle de BCE |

---

## 📅 Jour 4 — La Descente de Gradient

### 📚 Concept Théorique

La **descente de gradient** est l'algorithme qui permet au réseau d'**apprendre**. C'est le mécanisme qui ajuste les poids pour minimiser la loss.

Le principe : on calcule la direction dans laquelle la loss diminue le plus vite (le gradient), puis on fait un petit pas dans cette direction.

### 🧠 Intuition

Imaginez que vous êtes **les yeux bandés au sommet d'une montagne** et vous voulez descendre dans la vallée :
1. Vous tâtez le sol autour de vous pour sentir la **pente** (= calcul du gradient)
2. Vous faites un **pas** dans la direction la plus descendante (= mise à jour des poids)
3. Vous **répétez** jusqu'à atteindre le point le plus bas (= minimum de la loss)

La taille du pas = le **learning rate** ($\alpha$) :
- Trop grand → vous sautez par-dessus la vallée
- Trop petit → vous mettez une éternité à descendre
- Juste bien → vous convergez efficacement

### 🧮 Formules Essentielles

**Règle de mise à jour :**

$$w \leftarrow w - \alpha \cdot \frac{\partial \mathcal{L}}{\partial w}$$

Où :
- $w$ : le poids à mettre à jour
- $\alpha$ : le learning rate (typiquement 0.01, 0.001)
- $\frac{\partial \mathcal{L}}{\partial w}$ : le gradient de la loss par rapport à $w$

### 💻 Implémentation From Scratch

```python
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# JOUR 4 : La Descente de Gradient
# ============================================================

# --- Exemple 1 : Minimiser une fonction simple ---
# Trouvons le minimum de f(x) = (x - 3)² + 1
# Le minimum est évidemment x = 3, f(3) = 1
# Mais faisons-le trouver par la descente de gradient !

def f(x):
    """La fonction à minimiser."""
    return (x - 3) ** 2 + 1

def df(x):
    """La dérivée de f (le gradient)."""
    return 2 * (x - 3)  # df/dx = 2(x - 3)

# Paramètres
x = -5.0            # point de départ (loin du minimum)
learning_rate = 0.1  # taille du pas
n_iterations = 50    # nombre d'itérations

# Historique pour la visualisation
history_x = [x]
history_f = [f(x)]

print("=== Descente de Gradient sur f(x) = (x-3)² + 1 ===\n")

for i in range(n_iterations):
    gradient = df(x)              # 1. Calculer le gradient
    x = x - learning_rate * gradient  # 2. Mettre à jour x
    
    history_x.append(x)
    history_f.append(f(x))
    
    if i < 10 or i % 10 == 0:  # afficher les 10 premières + tous les 10
        print(f"  Itération {i+1:>3} : x = {x:>8.4f}, f(x) = {f(x):>8.4f}, gradient = {gradient:>8.4f}")

print(f"\n  Résultat final : x = {x:.6f} (attendu : 3.0)")
print(f"  f(x) finale    : {f(x):.6f} (attendu : 1.0)")

# --- Visualisation ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1 : Trajectoire sur la fonction
x_plot = np.linspace(-6, 8, 200)
axes[0].plot(x_plot, f(x_plot), 'b-', linewidth=2, label="f(x) = (x-3)² + 1")
axes[0].plot(history_x, history_f, 'ro-', markersize=4, label="Descente de gradient")
axes[0].plot(history_x[0], history_f[0], 'g*', markersize=15, label="Départ")
axes[0].plot(history_x[-1], history_f[-1], 'r*', markersize=15, label="Arrivée")
axes[0].set_title("Trajectoire de la descente", fontweight='bold')
axes[0].set_xlabel("x")
axes[0].set_ylabel("f(x)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2 : Loss au fil des itérations
axes[1].plot(history_f, 'g-', linewidth=2)
axes[1].set_title("Convergence de la loss", fontweight='bold')
axes[1].set_xlabel("Itération")
axes[1].set_ylabel("f(x)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("gradient_descent.png", dpi=150)
plt.show()

# --- Exemple 2 : Effet du learning rate ---
print("\n=== Effet du Learning Rate ===\n")

learning_rates = [0.001, 0.01, 0.1, 0.5, 0.9, 1.01]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, lr in enumerate(learning_rates):
    x = -5.0
    history = [x]
    
    for _ in range(30):
        x = x - lr * df(x)
        history.append(x)
        if abs(x) > 1000:  # divergence !
            break
    
    # Visualisation
    x_plot = np.linspace(-6, 10, 200)
    axes[idx].plot(x_plot, f(x_plot), 'b-', linewidth=2)
    history_f_plot = [f(h) for h in history if abs(h) < 100]
    history_clipped = [h for h in history if abs(h) < 100]
    if history_clipped:
        axes[idx].plot(history_clipped, history_f_plot, 'ro-', markersize=3)
    
    converged = abs(history[-1] - 3) < 0.1 if abs(history[-1]) < 1000 else False
    status = "✅ Converge" if converged else "❌ Diverge/Lent"
    axes[idx].set_title(f"lr = {lr} — {status}", fontweight='bold')
    axes[idx].set_ylim(-1, 70)
    axes[idx].grid(True, alpha=0.3)

plt.suptitle("Effet du Learning Rate", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("learning_rate_effect.png", dpi=150)
plt.show()
```

### 📌 Règles d'or du Learning Rate

| Learning Rate | Effet |
|---|---|
| Trop petit (0.0001) | Convergence très lente, risque de rester coincé |
| Bon (0.001 - 0.01) | Convergence stable et régulière |
| Trop grand (> 1.0) | Oscillation ou divergence ! |

---
