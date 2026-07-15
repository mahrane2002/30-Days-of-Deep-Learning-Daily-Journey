# 🧠 30 Days of Deep Learning from Scratch

---

##  Objectifs Globaux :

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

## 📅 Jour 5 — Le Perceptron : Premier Modèle d'Apprentissage

### 📚 Concept Théorique

Le **perceptron** est le premier algorithme d'apprentissage (Rosenblatt, 1958). C'est un neurone qui **apprend automatiquement ses poids** à partir des données.

L'idée : on montre des exemples au perceptron, et à chaque erreur, il ajuste ses poids pour corriger sa prédiction.

### 🧠 Intuition

Le perceptron est comme un **étudiant** :
1. Il fait une prédiction (examen)
2. On lui dit s'il a bon ou faux (correction)
3. S'il a faux, il ajuste sa compréhension (mise à jour des poids)
4. On recommence jusqu'à ce qu'il ait tout bon

### 🧮 Formules Essentielles

**Prédiction :**
$$\hat{y} = \text{step}(\mathbf{w}^T \mathbf{x} + b)$$

**Règle de mise à jour du Perceptron :**
$$w_i \leftarrow w_i + \alpha \cdot (y - \hat{y}) \cdot x_i$$
$$b \leftarrow b + \alpha \cdot (y - \hat{y})$$

Où $(y - \hat{y})$ est l'erreur.

### 💻 Implémentation From Scratch

```python
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# JOUR 5 : Le Perceptron
# ============================================================

class Perceptron:
    """
    Perceptron : le premier modèle d'apprentissage supervisé.
    Peut résoudre des problèmes linéairement séparables.
    """

    def __init__(self, n_features, learning_rate=0.1):
        """
        Args:
            n_features: nombre de caractéristiques en entrée
            learning_rate: vitesse d'apprentissage (alpha)
        """
        self.weights = np.zeros(n_features)  # poids initialisés à 0
        self.bias = 0.0
        self.lr = learning_rate

    def predict(self, x):
        """
        Fait une prédiction pour un seul exemple.
        
        Args:
            x: vecteur de caractéristiques
        
        Returns:
            0 ou 1
        """
        z = np.dot(self.weights, x) + self.bias
        return 1 if z > 0 else 0

    def train(self, X, y, n_epochs=100):
        """
        Entraîne le perceptron sur les données.
        
        Args:
            X: matrice de données (n_samples, n_features)
            y: vecteur de labels (n_samples,)
            n_epochs: nombre de passages sur les données
        
        Returns:
            Liste des erreurs par époque
        """
        errors_per_epoch = []

        for epoch in range(n_epochs):
            errors = 0
            for xi, yi in zip(X, y):
                # 1. Faire une prédiction
                prediction = self.predict(xi)
                
                # 2. Calculer l'erreur
                error = yi - prediction
                
                # 3. Mettre à jour les poids SI erreur != 0
                if error != 0:
                    self.weights += self.lr * error * xi
                    self.bias += self.lr * error
                    errors += 1

            errors_per_epoch.append(errors)

            # Afficher la progression
            if epoch < 5 or epoch % 10 == 0:
                print(f"  Époque {epoch+1:>3} : {errors} erreur(s)")

            # Arrêt anticipé si aucune erreur
            if errors == 0:
                print(f"\n  ✅ Convergence atteinte à l'époque {epoch + 1} !")
                break

        return errors_per_epoch

# --- Exemple 1 : Porte logique AND ---
print("=== Perceptron pour la porte AND ===\n")

X_and = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])
y_and = np.array([0, 0, 0, 1])  # AND : 1 seulement si les deux sont 1

perceptron_and = Perceptron(n_features=2, learning_rate=0.1)
errors_and = perceptron_and.train(X_and, y_and, n_epochs=100)

print(f"\nPoids finaux : {perceptron_and.weights}")
print(f"Biais final  : {perceptron_and.bias}")
print("\nTest :")
for xi, yi in zip(X_and, y_and):
    pred = perceptron_and.predict(xi)
    status = "✅" if pred == yi else "❌"
    print(f"  {xi} → prédit: {pred}, attendu: {yi} {status}")

# --- Exemple 2 : Porte logique OR ---
print("\n=== Perceptron pour la porte OR ===\n")

y_or = np.array([0, 1, 1, 1])

perceptron_or = Perceptron(n_features=2, learning_rate=0.1)
errors_or = perceptron_or.train(X_and, y_or, n_epochs=100)

print(f"\nTest :")
for xi, yi in zip(X_and, y_or):
    pred = perceptron_or.predict(xi)
    status = "✅" if pred == yi else "❌"
    print(f"  {xi} → prédit: {pred}, attendu: {yi} {status}")

# --- Exemple 3 : Le problème XOR (impossible !) ---
print("\n=== Perceptron pour XOR (ça ne marchera PAS) ===\n")

y_xor = np.array([0, 1, 1, 0])

perceptron_xor = Perceptron(n_features=2, learning_rate=0.1)
errors_xor = perceptron_xor.train(X_and, y_xor, n_epochs=100)

print(f"\nTest :")
for xi, yi in zip(X_and, y_xor):
    pred = perceptron_xor.predict(xi)
    status = "✅" if pred == yi else "❌"
    print(f"  {xi} → prédit: {pred}, attendu: {yi} {status}")

print("\n💡 Le XOR n'est PAS linéairement séparable.")
print("   → On a besoin de PLUSIEURS couches (réseau profond) !")

# --- Visualisation des frontières de décision ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
titles = ["AND", "OR", "XOR (impossible)"]
perceptrons = [perceptron_and, perceptron_or, perceptron_xor]
labels = [y_and, y_or, y_xor]

for ax, title, perc, y in zip(axes, titles, perceptrons, labels):
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
    Z = np.array([perc.predict(np.array([a, b]))
                  for a, b in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=['#FFCCCC', '#CCCCFF'], alpha=0.5)
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    
    for xi, yi in zip(X_and, y):
        color = 'blue' if yi == 1 else 'red'
        marker = 'o' if yi == 1 else 'x'
        ax.plot(xi[0], xi[1], marker, color=color, markersize=15, markeredgewidth=3)
    
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.grid(True, alpha=0.3)

plt.suptitle("Frontières de Décision du Perceptron", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("perceptron_decision_boundaries.png", dpi=150)
plt.show()
```

### 📌 Ce qu'il faut retenir du Jour 5

> **Le perceptron est puissant mais limité** : il ne peut résoudre que les problèmes **linéairement séparables** (séparables par une droite). Le XOR montre qu'un seul neurone ne suffit pas.
>
> **Solution** → Empiler plusieurs couches de neurones = **réseau de neurones** ! C'est ce qu'on fera à partir du Jour 11.

---

---

# 📅 SEMAINE 2 : MÉCANIQUE (Jours 6–10)

---

## 📅 Jour 6 — La Régression Logistique

### 📚 Concept Théorique

La **régression logistique** est le perceptron amélioré. Au lieu de sortir brutalement 0 ou 1, elle produit une **probabilité** entre 0 et 1 grâce à la fonction sigmoid.

C'est le modèle le plus fondamental pour la **classification binaire** et la base de tous les réseaux de neurones.

### 🧠 Intuition

Si le perceptron est un interrupteur (ON/OFF), la régression logistique est un **variateur** : elle nous dit "je suis sûr à 87% que c'est un chat" au lieu de juste "c'est un chat".

### 🧮 Formules Essentielles

**Modèle :**
$$\hat{y} = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

**Loss (Binary Cross-Entropy) :**
$$\mathcal{L} = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

**Gradients :**
$$\frac{\partial \mathcal{L}}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) \cdot x_j^{(i)}$$
$$\frac{\partial \mathcal{L}}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})$$

### 💻 Implémentation From Scratch

```python
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# JOUR 6 : Régression Logistique from Scratch
# ============================================================

class LogisticRegression:
    """
    Régression logistique : classification binaire avec probabilités.
    C'est un neurone avec sigmoid + cross-entropy loss.
    """

    def __init__(self, n_features, learning_rate=0.1):
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.lr = learning_rate
        self.losses = []

    def sigmoid(self, z):
        """Fonction sigmoid : écrase z entre 0 et 1."""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def forward(self, X):
        """
        Propagation avant : calcule les probabilités.
        
        Args:
            X: matrice (m, n) — m exemples, n features
        
        Returns:
            Vecteur de probabilités (m,)
        """
        z = X @ self.weights + self.bias  # @ = produit matriciel
        return self.sigmoid(z)

    def compute_loss(self, y_true, y_pred):
        """Calcule la binary cross-entropy loss."""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def train(self, X, y, n_epochs=1000, verbose=True):
        """
        Entraîne le modèle par descente de gradient.
        """
        m = len(y)

        for epoch in range(n_epochs):
            # --- Forward pass ---
            y_pred = self.forward(X)

            # --- Calcul de la loss ---
            loss = self.compute_loss(y, y_pred)
            self.losses.append(loss)

            # --- Calcul des gradients ---
            error = y_pred - y
            dw = (1 / m) * (X.T @ error)
            db = (1 / m) * np.sum(error)

            # --- Mise à jour des paramètres ---
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # --- Affichage ---
            if verbose and (epoch < 5 or epoch % 200 == 0):
                accuracy = np.mean((y_pred >= 0.5) == y)
                print(f"  Époque {epoch+1:>4} : loss = {loss:.4f}, accuracy = {accuracy:.2%}")

    def predict(self, X):
        """Retourne les prédictions (0 ou 1)."""
        return (self.forward(X) >= 0.5).astype(int)

    def predict_proba(self, X):
        """Retourne les probabilités."""
        return self.forward(X)


# --- Créer un dataset synthétique ---
np.random.seed(42)

# Classe 0 : centrée autour de (2, 2)
X0 = np.random.randn(100, 2) + np.array([2, 2])
# Classe 1 : centrée autour de (6, 6)
X1 = np.random.randn(100, 2) + np.array([6, 6])

X = np.vstack([X0, X1])
y = np.hstack([np.zeros(100), np.ones(100)])

# Mélanger les données
shuffle_idx = np.random.permutation(len(y))
X, y = X[shuffle_idx], y[shuffle_idx]

# --- Entraîner ---
print("=== Régression Logistique ===\n")
model = LogisticRegression(n_features=2, learning_rate=0.1)
model.train(X, y, n_epochs=1000)

# --- Résultats ---
predictions = model.predict(X)
accuracy = np.mean(predictions == y)
print(f"\nAccuracy finale : {accuracy:.2%}")

# --- Visualisation ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1 : Frontière de décision
xx, yy = np.meshgrid(np.linspace(-1, 9, 200), np.linspace(-1, 9, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
probs = model.predict_proba(grid).reshape(xx.shape)

axes[0].contourf(xx, yy, probs, levels=50, cmap='RdBu_r', alpha=0.6)
axes[0].contour(xx, yy, probs, levels=[0.5], colors='black', linewidths=2)
axes[0].scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='red', label='Classe 0', edgecolors='black')
axes[0].scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='blue', label='Classe 1', edgecolors='black')
axes[0].set_title("Frontière de Décision", fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2 : Courbe de loss
axes[1].plot(model.losses, 'g-', linewidth=2)
axes[1].set_title("Convergence de la Loss", fontweight='bold')
axes[1].set_xlabel("Époque")
axes[1].set_ylabel("Binary Cross-Entropy")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("logistic_regression.png", dpi=150)
plt.show()
```

---

## 📅 Jour 7 — Backpropagation : Le Cœur de l'Apprentissage

### 📚 Concept Théorique

La **backpropagation** (rétropropagation) est l'algorithme qui permet de calculer **les gradients de la loss par rapport à chaque poids** du réseau. C'est une application récursive de la **règle de la chaîne** (chain rule).

### 🧠 Intuition

Pensez à une **chaîne de dominos** :

```
Entrée → Neurone 1 → Neurone 2 → ... → Sortie → Loss
```

La backpropagation remonte cette chaîne en sens inverse pour répondre à la question : "**Si je change un peu le poids $w$ dans le neurone 1, de combien change la loss ?**"

### 🧮 Formules Essentielles

**Règle de la chaîne :**

Si $\mathcal{L}$ dépend de $z$ qui dépend de $w$ :

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial z} \cdot \frac{\partial z}{\partial w}$$

**Pour un neurone avec sigmoid :**

$$z = wx + b$$
$$a = \sigma(z)$$
$$\mathcal{L} = -(y \log(a) + (1-y) \log(1-a))$$

**Les gradients (en remontant) :**

$$\frac{\partial \mathcal{L}}{\partial a} = -\frac{y}{a} + \frac{1-y}{1-a}$$

$$\frac{\partial \mathcal{L}}{\partial z} = a - y$$

$$\frac{\partial \mathcal{L}}{\partial w} = (a - y) \cdot x$$

$$\frac{\partial \mathcal{L}}{\partial b} = a - y$$

### 💻 Implémentation From Scratch

```python
import numpy as np

# ============================================================
# JOUR 7 : Backpropagation — pas à pas
# ============================================================

# Réseau à 2 couches : 2 entrées → 2 neurones cachés → 1 sortie

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(a):
    """Dérivée de sigmoid en fonction de la SORTIE a (pas de z)."""
    return a * (1 - a)

# Données XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Initialisation des poids
np.random.seed(42)
W1 = np.random.randn(2, 2) * 0.5
b1 = np.zeros((1, 2))
W2 = np.random.randn(2, 1) * 0.5
b2 = np.zeros((1, 1))

learning_rate = 1.0
losses = []

print("=== Backpropagation sur XOR ===\n")

for epoch in range(10000):
    # ========== FORWARD PASS ==========
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)
    
    # Loss (MSE)
    loss = np.mean((y - a2) ** 2)
    losses.append(loss)
    
    # ========== BACKWARD PASS ==========
    m = X.shape[0]
    
    # Gradient de la loss par rapport à a2
    dL_da2 = (2 / m) * (a2 - y)
    
    # Couche 2 : remonter à travers sigmoid
    da2_dz2 = sigmoid_derivative(a2)
    dL_dz2 = dL_da2 * da2_dz2
    
    # Gradients des poids couche 2
    dL_dW2 = a1.T @ dL_dz2
    dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)
    
    # Propager l'erreur vers la couche 1
    dL_da1 = dL_dz2 @ W2.T
    da1_dz1 = sigmoid_derivative(a1)
    dL_dz1 = dL_da1 * da1_dz1
    
    # Gradients des poids couche 1
    dL_dW1 = X.T @ dL_dz1
    dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)
    
    # ========== MISE À JOUR ==========
    W2 -= learning_rate * dL_dW2
    b2 -= learning_rate * dL_db2
    W1 -= learning_rate * dL_dW1
    b1 -= learning_rate * dL_db1
    
    if epoch < 5 or epoch % 2000 == 0:
        predictions = (a2 > 0.5).astype(int)
        accuracy = np.mean(predictions == y)
        print(f"  Époque {epoch+1:>5} : loss = {loss:.6f}, accuracy = {accuracy:.0%}")

# Test final
print("\n=== Résultat final ===")
z1 = X @ W1 + b1
a1 = sigmoid(z1)
z2 = a1 @ W2 + b2
a2 = sigmoid(z2)

for i in range(4):
    print(f"  {X[i]} → prob = {a2[i, 0]:.4f}, prédit = {int(a2[i, 0] > 0.5)}, attendu = {y[i, 0]}")

print("\n🎉 XOR résolu avec la backpropagation et 2 couches !")
```

### 📌 Résumé de la Backpropagation

```
Forward:  X → z1 → a1 → z2 → a2 → Loss
Backward: X ← dz1 ← da1 ← dz2 ← da2 ← dLoss
```

> La backpropagation n'est **rien d'autre** que la règle de la chaîne appliquée systématiquement en sens inverse.

---

## 📅 Jour 8 — Graphe de Calcul et Autograd Simplifié

### 📚 Concept Théorique

Aujourd'hui, nous allons construire un **graphe de calcul** (computation graph), la structure qu'utilisent PyTorch et TensorFlow en interne. Chaque opération est un noeud, et les gradients sont propagés automatiquement.

### 🧠 Intuition

Un graphe de calcul est comme une **recette de cuisine inversée**. Si le plat est trop salé (loss élevée), vous remontez chaque étape pour savoir **quel ingrédient ajuster et de combien**.

### 💻 Implémentation From Scratch

```python
import numpy as np

# ============================================================
# JOUR 8 : Graphe de Calcul et Autograd simplifié
# ============================================================

class Value:
    """
    Un noeud dans le graphe de calcul.
    Stocke une valeur et son gradient.
    Inspiré de micrograd d'Andrej Karpathy.
    """

    def __init__(self, data, children=(), operation=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._children = set(children)
        self._op = operation

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        """Addition : c = a + b → dc/da = 1, dc/db = 1"""
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += result.grad
            other.grad += result.grad
        result._backward = _backward
        
        return result

    def __mul__(self, other):
        """Multiplication : c = a * b → dc/da = b, dc/db = a"""
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * result.grad
            other.grad += self.data * result.grad
        result._backward = _backward
        
        return result

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __pow__(self, n):
        """Puissance : c = a^n → dc/da = n * a^(n-1)"""
        result = Value(self.data ** n, (self,), f'**{n}')
        
        def _backward():
            self.grad += n * (self.data ** (n - 1)) * result.grad
        result._backward = _backward
        
        return result

    def sigmoid(self):
        """Sigmoid : σ(a) = 1 / (1 + e^(-a))"""
        s = 1 / (1 + np.exp(-self.data))
        result = Value(s, (self,), 'sigmoid')
        
        def _backward():
            self.grad += s * (1 - s) * result.grad
        result._backward = _backward
        
        return result

    def backward(self):
        """Lance la backpropagation (tri topologique)."""
        topo_order = []
        visited = set()
        
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._children:
                    build_topo(child)
                topo_order.append(node)
        
        build_topo(self)
        
        self.grad = 1.0
        for node in reversed(topo_order):
            node._backward()


# --- Démonstration ---
print("=== Graphe de Calcul : Neurone avec Autograd ===\n")

x1, x2 = Value(2.0), Value(3.0)
w1, w2 = Value(-1.0), Value(0.5)
b = Value(0.1)

z = x1 * w1 + x2 * w2 + b
a = z.sigmoid()
y_true = Value(1.0)
loss = (a - y_true) ** 2

print(f"z    = {z.data:.4f}")
print(f"a    = {a.data:.4f}")
print(f"loss = {loss.data:.4f}")

loss.backward()

print(f"\ndL/dw1 = {w1.grad:.4f}")
print(f"dL/dw2 = {w2.grad:.4f}")
print(f"dL/db  = {b.grad:.4f}")
```

## 📅 Jour 9 — Architecture de Réseau Modulaire

### 📚 Concept Théorique

Restructurons notre code en une **architecture propre et réutilisable**. Chaque couche est un module avec `forward()` et `backward()`.

### 🧠 Intuition

Chaque couche est un **Lego** : elle a une interface standard (entrée → sortie) et peut être empilée avec d'autres. C'est exactement comme ça que fonctionnent PyTorch et TensorFlow.

### 💻 Implémentation From Scratch

```python
import numpy as np

# ============================================================
# JOUR 9 : Architecture de réseau modulaire
# ============================================================

class Layer:
    """Classe de base pour toutes les couches."""
    def forward(self, x):
        raise NotImplementedError
    def backward(self, grad_output):
        raise NotImplementedError

class Dense(Layer):
    """Couche dense (fully connected)."""

    def __init__(self, n_input, n_output):
        # Xavier initialization
        limit = np.sqrt(6 / (n_input + n_output))
        self.W = np.random.uniform(-limit, limit, (n_input, n_output))
        self.b = np.zeros((1, n_output))
        self.input = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.input = x
        return x @ self.W + self.b

    def backward(self, grad_output):
        self.dW = self.input.T @ grad_output
        self.db = np.sum(grad_output, axis=0, keepdims=True)
        return grad_output @ self.W.T

class Sigmoid(Layer):
    def forward(self, x):
        self.output = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return self.output
    def backward(self, grad_output):
        return grad_output * self.output * (1 - self.output)

class ReLU(Layer):
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)
    def backward(self, grad_output):
        return grad_output * (self.input > 0)

class TanhLayer(Layer):
    def forward(self, x):
        self.output = np.tanh(x)
        return self.output
    def backward(self, grad_output):
        return grad_output * (1 - self.output ** 2)

class NeuralNetwork:
    """Réseau de neurones séquentiel."""

    def __init__(self, layers):
        self.layers = layers
        self.losses = []

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self, learning_rate):
        for layer in self.layers:
            if isinstance(layer, Dense):
                layer.W -= learning_rate * layer.dW
                layer.b -= learning_rate * layer.db

    def train(self, X, y, n_epochs=1000, learning_rate=0.1):
        for epoch in range(n_epochs):
            y_pred = self.forward(X)
            loss = np.mean((y - y_pred) ** 2)
            self.losses.append(loss)
            
            grad = (2 / len(y)) * (y_pred - y)
            self.backward(grad)
            self.update(learning_rate)
            
            if epoch < 5 or epoch % 1000 == 0:
                accuracy = np.mean((y_pred > 0.5).astype(int) == y)
                print(f"  Époque {epoch+1:>5} : loss = {loss:.6f}, accuracy = {accuracy:.0%}")

# --- Test sur XOR ---
print("=== Réseau Modulaire sur XOR ===\n")
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

model = NeuralNetwork([
    Dense(2, 4), TanhLayer(),
    Dense(4, 1), Sigmoid(),
])
model.train(X, y, n_epochs=5000, learning_rate=1.0)

print("\nPrédictions finales :")
output = model.forward(X)
for i in range(4):
    print(f"  {X[i]} → {output[i, 0]:.4f} (attendu: {y[i, 0]})")
```

## 📅 Jour 10 — Mini-Batches et Métriques d'Évaluation

### 📚 Concept Théorique

En pratique, avec des millions d'exemples, on utilise des **mini-batches** : de petits sous-ensembles traités à chaque itération.

### 🧮 Types de Descente de Gradient

| Type | Taille du batch | Vitesse | Stabilité |
|---|---|---|---|
| Batch GD | m (toutes les données) | Lente | Très stable |
| Stochastic GD (SGD) | 1 (un exemple) | Très rapide | Très bruyant |
| Mini-batch GD | 32, 64, 128... | Bon compromis | Bon compromis |

### 💻 Implémentation From Scratch

```python
import numpy as np

# ============================================================
# JOUR 10 : Mini-Batch Training + Métriques
# ============================================================

def create_batches(X, y, batch_size=32, shuffle=True):
    """Découpe les données en mini-batches."""
    n_samples = len(y)
    if shuffle:
        indices = np.random.permutation(n_samples)
        X, y = X[indices], y[indices]
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        yield X[start:end], y[start:end]

# --- Métriques d'évaluation ---
def accuracy(y_true, y_pred):
    predictions = (y_pred >= 0.5).astype(int)
    return np.mean(predictions == y_true)

def precision(y_true, y_pred):
    predictions = (y_pred >= 0.5).astype(int)
    tp = np.sum((predictions == 1) & (y_true == 1))
    pp = np.sum(predictions == 1)
    return tp / max(pp, 1)

def recall(y_true, y_pred):
    predictions = (y_pred >= 0.5).astype(int)
    tp = np.sum((predictions == 1) & (y_true == 1))
    ap = np.sum(y_true == 1)
    return tp / max(ap, 1)

def f1_score(y_true, y_pred):
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return 2 * p * r / max(p + r, 1e-10)

# --- Démonstration des métriques ---
print("=== Métriques d'Évaluation ===\n")
y_ex = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0]).reshape(-1, 1)
y_pred_ex = np.array([0.9, 0.8, 0.2, 0.3, 0.6, 0.4, 0.1, 0.7, 0.8, 0.2]).reshape(-1, 1)

print(f"Accuracy  : {accuracy(y_ex, y_pred_ex):.2%}")
print(f"Precision : {precision(y_ex, y_pred_ex):.2%}")
print(f"Recall    : {recall(y_ex, y_pred_ex):.2%}")
print(f"F1 Score  : {f1_score(y_ex, y_pred_ex):.2%}")
```

### 📌 Résumé Semaine 2

> Vous savez maintenant :
> - ✅ Construire un classifieur binaire (régression logistique)
> - ✅ Implémenter la backpropagation pas à pas
> - ✅ Construire un graphe de calcul (mini autograd)
> - ✅ Architecturer un réseau modulaire avec des couches
> - ✅ Entraîner avec des mini-batches et évaluer avec les métriques

---

---

# 📅 SEMAINE 3 : CONSTRUCTION DU MLP (Jours 11–15)





## 📅 Jour 11 — Le Perceptron Multi-Couches (MLP) Complet

### 📚 Concept Théorique

Le **MLP** (Multi-Layer Perceptron) est le premier véritable **réseau de neurones profond**. Chaque couche apprend un **niveau d'abstraction** différent :
- Couche 1 : détecte des motifs simples
- Couche 2 : combine ces motifs en formes complexes
- Couche 3 : reconnaît des concepts de haut niveau

### 💻 Implémentation From Scratch — MLP Complet

```python
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# JOUR 11 : MLP Complet From Scratch
# ============================================================

class DenseLayer:
    """Couche dense avec choix d'activation."""
    
    def __init__(self, n_in, n_out, activation='relu'):
        if activation == 'relu':
            self.W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
        else:
            self.W = np.random.randn(n_in, n_out) * np.sqrt(1.0 / n_in)
        self.b = np.zeros((1, n_out))
        self.activation = activation
    
    def _activate(self, z):
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'none':
            return z
    
    def _activate_derivative(self):
        if self.activation == 'relu':
            return (self.z > 0).astype(float)
        elif self.activation == 'sigmoid':
            return self.a * (1 - self.a)
        elif self.activation == 'tanh':
            return 1 - self.a ** 2
        elif self.activation == 'none':
            return np.ones_like(self.z)
    
    def forward(self, x):
        self.input = x
        self.z = x @ self.W + self.b
        self.a = self._activate(self.z)
        return self.a
    
    def backward(self, grad_output):
        grad_z = grad_output * self._activate_derivative()
        self.dW = self.input.T @ grad_z / len(grad_output)
        self.db = np.mean(grad_z, axis=0, keepdims=True)
        return grad_z @ self.W.T


class MLP:
    """Multi-Layer Perceptron complet."""
    
    def __init__(self, layer_sizes, activations, loss='bce'):
        assert len(activations) == len(layer_sizes) - 1
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(DenseLayer(layer_sizes[i], layer_sizes[i + 1], activations[i]))
        self.loss_type = loss
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out
    
    def compute_loss(self, y_true, y_pred):
        eps = 1e-15
        if self.loss_type == 'bce':
            y_pred = np.clip(y_pred, eps, 1 - eps)
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        elif self.loss_type == 'mse':
            return np.mean((y_true - y_pred) ** 2)
    
    def compute_loss_gradient(self, y_true, y_pred):
        eps = 1e-15
        m = len(y_true)
        if self.loss_type == 'bce':
            y_pred = np.clip(y_pred, eps, 1 - eps)
            return (-y_true / y_pred + (1 - y_true) / (1 - y_pred)) / m
        elif self.loss_type == 'mse':
            return 2 * (y_pred - y_true) / m
    
    def fit(self, X_train, y_train, X_val=None, y_val=None,
            epochs=100, lr=0.01, batch_size=32, verbose=True):
        n = len(X_train)
        for epoch in range(epochs):
            perm = np.random.permutation(n)
            X_s, y_s = X_train[perm], y_train[perm]
            
            epoch_loss = 0
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                X_b, y_b = X_s[start:end], y_s[start:end]
                
                y_pred = self.forward(X_b)
                epoch_loss += self.compute_loss(y_b, y_pred)
                
                grad = self.compute_loss_gradient(y_b, y_pred)
                for layer in reversed(self.layers):
                    grad = layer.backward(grad)
                
                for layer in self.layers:
                    layer.W -= lr * layer.dW
                    layer.b -= lr * layer.db
            
            # Métriques
            train_pred = self.forward(X_train)
            train_acc = np.mean((train_pred >= 0.5) == y_train)
            self.history['train_loss'].append(epoch_loss / (n // batch_size + 1))
            self.history['train_acc'].append(train_acc)
            
            if X_val is not None:
                val_pred = self.forward(X_val)
                val_loss = self.compute_loss(y_val, val_pred)
                val_acc = np.mean((val_pred >= 0.5) == y_val)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
            
            if verbose and (epoch < 5 or (epoch + 1) % 50 == 0):
                msg = f"  Époque {epoch+1:>4} : train_acc={train_acc:.2%}"
                if X_val is not None:
                    msg += f", val_acc={val_acc:.2%}"
                print(msg)

# --- Test : cercles concentriques ---
def make_circles(n_samples=500, noise=0.1):
    n = n_samples // 2
    theta_in = np.random.uniform(0, 2*np.pi, n)
    r_in = np.random.normal(1, noise, n)
    X_in = np.column_stack([r_in*np.cos(theta_in), r_in*np.sin(theta_in)])
    theta_out = np.random.uniform(0, 2*np.pi, n)
    r_out = np.random.normal(3, noise, n)
    X_out = np.column_stack([r_out*np.cos(theta_out), r_out*np.sin(theta_out)])
    X = np.vstack([X_in, X_out])
    y = np.vstack([np.zeros((n,1)), np.ones((n,1))])
    perm = np.random.permutation(n_samples)
    return X[perm], y[perm]

np.random.seed(42)
X, y = make_circles(600, noise=0.15)
X_train, X_test = X[:480], X[480:]
y_train, y_test = y[:480], y[480:]

print("=== MLP sur Cercles Concentriques ===\n")
model = MLP([2, 32, 16, 1], ['relu', 'relu', 'sigmoid'], loss='bce')
model.fit(X_train, y_train, X_test, y_test, epochs=200, lr=0.05, batch_size=32)
```

## 📅 Jour 12 — Initialisation des Poids

### 📚 Concept Théorique

L'initialisation des poids est **cruciale** :
- **Trop petits** → vanishing gradient (les gradients disparaissent)
- **Trop grands** → exploding gradient (les gradients explosent)

### 🧮 Méthodes d'Initialisation

**Xavier/Glorot** (pour sigmoid, tanh) :
$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{\text{in}} + n_{\text{out}}}}\right)$$

**He** (pour ReLU) :
$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{\text{in}}}}\right)$$

### 💻 Implémentation

```python
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# JOUR 12 : Initialisation des Poids
# ============================================================

def visualize_activations(init_method, title):
    np.random.seed(0)
    n_layers, n_neurons = 6, 256
    x = np.random.randn(1000, n_neurons)
    activations = [x]
    
    for i in range(n_layers):
        if init_method == 'small_random':
            W = np.random.randn(n_neurons, n_neurons) * 0.01
        elif init_method == 'large_random':
            W = np.random.randn(n_neurons, n_neurons) * 1.0
        elif init_method == 'xavier':
            W = np.random.randn(n_neurons, n_neurons) * np.sqrt(1.0 / n_neurons)
        elif init_method == 'he':
            W = np.random.randn(n_neurons, n_neurons) * np.sqrt(2.0 / n_neurons)
        
        x = x @ W
        x = np.tanh(x) if init_method != 'he' else np.maximum(0, x)
        activations.append(x)
    
    fig, axes = plt.subplots(1, n_layers + 1, figsize=(20, 3))
    fig.suptitle(f"Initialisation : {title}", fontweight='bold')
    for i, act in enumerate(activations):
        axes[i].hist(act.ravel(), bins=50, density=True, color='steelblue')
        axes[i].set_title(f"Couche {i}")
    plt.tight_layout()
    plt.show()

print("=== Effet de l'Initialisation ===\n")
visualize_activations('small_random', 'Petits poids → activations → 0')
visualize_activations('large_random', 'Grands poids → activations saturent')
visualize_activations('xavier', 'Xavier ✅ (pour tanh/sigmoid)')
visualize_activations('he', 'He ✅ (pour ReLU)')
```

## 📅 Jour 13 — Softmax et Classification Multi-Classes

### 📚 Concept Théorique

Pour **N classes**, on utilise **softmax** qui transforme un vecteur de scores en une **distribution de probabilités**.

### 🧮 Formules

**Softmax :**
$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

**Categorical Cross-Entropy :**
$$\mathcal{L} = -\sum_{i=1}^{K} y_i \log(\hat{y}_i)$$

**Gradient combiné (softmax + CE) :**
$$\frac{\partial \mathcal{L}}{\partial z_i} = \hat{y}_i - y_i$$

### 💻 Implémentation From Scratch

```python
import numpy as np

# ============================================================
# JOUR 13 : Softmax et Classification Multi-Classes
# ============================================================

def softmax(z):
    """Softmax stable numériquement."""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(y_true_onehot, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true_onehot * np.log(y_pred), axis=1))

def one_hot_encode(y, n_classes):
    one_hot = np.zeros((len(y), n_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot

class MLPMultiClass:
    """MLP pour classification multi-classes avec softmax."""
    
    def __init__(self, layer_sizes, lr=0.01):
        self.layers = []
        self.lr = lr
        for i in range(len(layer_sizes) - 1):
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.layers.append({'W': W, 'b': b})
    
    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        out = X
        for i, layer in enumerate(self.layers):
            z = out @ layer['W'] + layer['b']
            self.z_values.append(z)
            out = np.maximum(0, z) if i < len(self.layers) - 1 else softmax(z)
            self.activations.append(out)
        return out
    
    def backward(self, y_onehot):
        m = len(y_onehot)
        delta = self.activations[-1] - y_onehot
        grads = []
        for i in range(len(self.layers) - 1, -1, -1):
            dW = self.activations[i].T @ delta / m
            db = np.mean(delta, axis=0, keepdims=True)
            grads.insert(0, {'dW': dW, 'db': db})
            if i > 0:
                delta = delta @ self.layers[i]['W'].T
                delta *= (self.z_values[i-1] > 0)
        for i, grad in enumerate(grads):
            self.layers[i]['W'] -= self.lr * grad['dW']
            self.layers[i]['b'] -= self.lr * grad['db']
    
    def fit(self, X, y, epochs=500):
        n_classes = len(np.unique(y))
        y_onehot = one_hot_encode(y, n_classes)
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = cross_entropy_loss(y_onehot, y_pred)
            self.backward(y_onehot)
            if epoch % 100 == 0:
                acc = np.mean(np.argmax(y_pred, axis=1) == y)
                print(f"  Époque {epoch:>4} : loss = {loss:.4f}, accuracy = {acc:.2%}")

# Test avec 3 classes
np.random.seed(42)
n_per_class = 100
X = np.vstack([
    np.random.randn(n_per_class, 2) + [0, 3],
    np.random.randn(n_per_class, 2) + [3, -1],
    np.random.randn(n_per_class, 2) + [-3, -1],
])
y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class), 2*np.ones(n_per_class)]).astype(int)

print("=== MLP Multi-Classes ===\n")
model = MLPMultiClass([2, 32, 16, 3], lr=0.1)
model.fit(X, y, epochs=500)
print(f"\nAccuracy : {np.mean(np.argmax(model.forward(X), axis=1) == y):.2%}")
```

## 📅 Jour 14 — Batch Normalization

### 📚 Concept Théorique

La **Batch Normalization** normalise les activations pour stabiliser et accélérer l'entraînement.

### 🧮 Formules

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \qquad y_i = \gamma \hat{x}_i + \beta$$

### 💻 Implémentation

```python
import numpy as np

# ============================================================
# JOUR 14 : Batch Normalization
# ============================================================

class BatchNorm:
    def __init__(self, n_features, momentum=0.9, epsilon=1e-5):
        self.gamma = np.ones((1, n_features))
        self.beta = np.zeros((1, n_features))
        self.epsilon = epsilon
        self.momentum = momentum
        self.running_mean = np.zeros((1, n_features))
        self.running_var = np.ones((1, n_features))
    
    def forward(self, x, training=True):
        if training:
            mean = np.mean(x, axis=0, keepdims=True)
            var = np.var(x, axis=0, keepdims=True)
            self.std = np.sqrt(var + self.epsilon)
            self.x_norm = (x - mean) / self.std
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            self.x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        return self.gamma * self.x_norm + self.beta

# Test
np.random.seed(42)
x = np.random.randn(32, 4) * 10 + 5
bn = BatchNorm(4)
x_norm = bn.forward(x)
print("Avant : moyenne =", x.mean(axis=0).round(2), "| std =", x.std(axis=0).round(2))
print("Après : moyenne =", x_norm.mean(axis=0).round(4), "| std =", x_norm.std(axis=0).round(4))
```
