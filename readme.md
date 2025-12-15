# 🎶 Classification de Genres Musicaux : Analyse et Deep Learning

## ⭐ Introduction

Ce projet a pour but de développer et d'évaluer un modèle de *Deep Learning* pour la classification automatique de genres musicaux (Rock, Jazz, Blues, etc.). Nous utilisons l'ensemble de données **GTZAN** et le framework **PyTorch** pour l'entraînement.

L'originalité de cette approche réside dans le *Feature Engineering* audio réalisé par **Librosa**, transformant le signal sonore brut en caractéristiques numériques exploitables par le réseau de neurones.

## 📊 Résultat Clé

Le modèle a obtenu une performance satisfaisante après entraînement :

| Métrique | Valeur |
| :--- | :--- |
| **Précision (Accuracy)** | **86.67%** |

---

## 🛠️ Explication Détaillée des Bibliothèques et Outils

Chaque outil Python utilisé dans ce projet a un rôle précis dans le pipeline de traitement audio et de Deep Learning.

### 1. PyTorch (`torch`, `torchvision`, `torchaudio`)

PyTorch est le **framework de Deep Learning** principal de ce projet.

* **Rôle :** Il est utilisé pour construire, entraîner, et évaluer le réseau de neurones.
* **Concepts clés :**
    * **Tenseurs (`torch.Tensor`)** : Structure de données fondamentale, similaire aux tableaux NumPy, mais optimisée pour le calcul sur GPU.
    * **Module (`nn.Module`)** : La classe de base pour toutes les architectures de réseaux de neurones (couches, fonctions d'activation).
    * **AutoGrad** : Le système de différentiation automatique qui permet de calculer les gradients nécessaires à la rétropropagation et à l'entraînement du modèle.
    * **GPU (CUDA)** : Permet d'accélérer massivement les calculs matriciels en utilisant le GPU, géré par l'appel `.to(device)`.



### 2. Librosa

Librosa est la bibliothèque incontournable pour l'**analyse audio et musicale**. C'est le cœur du *Feature Engineering* (extraction de caractéristiques).

* **Rôle :** Transformer le signal audio (forme d'onde) en représentations numériques compressées et significatives.
* **Fonctionnalités clés :**
    * `librosa.load()` : Charge le fichier audio en mémoire sous forme d'un tableau NumPy (forme d'onde) et de sa fréquence d'échantillonnage ($sr$).
    * **MFCC (Mel-Frequency Cepstral Coefficients)** : Les caractéristiques spectrales utilisées comme entrée dans notre modèle. Elles représentent la forme générale de l'enveloppe spectrale et sont très robustes pour la reconnaissance de genre/timbre.
    * **Spectrogramme** : Représentation visuelle de la fréquence du signal en fonction du temps.
* **En bref :** C'est le pont entre le fichier `.wav` et l'entrée du modèle PyTorch.

### 3. Scikit-learn (`sklearn`)

Scikit-learn est la bibliothèque standard pour le **Machine Learning classique** et les outils utilitaires.

* **Rôle :** Gestion des données et évaluation des performances.
* **Fonctionnalités clés :**
    * `train_test_split` : Sépare les données en ensembles d'entraînement, de validation et de test. Essentiel pour éviter l'**overfitting**.
    * `accuracy_score` : Calcule la précision globale du modèle.
    * `classification_report` : Fournit un résumé des métriques (Précision, Rappel, F1-Score) pour chaque classe de genre.
    * Normalisation / Mise à l'échelle des données.

### 4. NumPy

NumPy est le fondement du **calcul scientifique** en Python.

* **Rôle :** Manipulation et stockage des tableaux de données numériques de grande dimension.
* **Utilisation dans le projet :** Tous les signaux audio chargés par Librosa et toutes les caractéristiques MFCCs sont initialement des tableaux NumPy. NumPy facilite les opérations arithmétiques vectorielles rapides.

### 5. Pandas

Pandas est la bibliothèque dédiée à l'**analyse et à la manipulation de données**.

* **Rôle :** Organisation des données et des étiquettes (labels).
* **Utilisation dans le projet :** Peut être utilisé pour créer un **DataFrame** qui mappe chaque chemin de fichier audio à son genre correspondant, simplifiant l'étape de chargement et d'étiquetage des données.

### 6. Autres Utilitaires

| Bibliothèque | Rôle | Explication |
| :--- | :--- | :--- |
| **`glob` (module Python)** | Gestion des chemins de fichiers | Permet de rechercher et de lister des fichiers en utilisant des motifs (ex: `dataset/**/*.wav`), ce qui est crucial pour charger tous les fichiers audio d'une arborescence. |
| **`tqdm`** | Indicateur de progression | Ajoute une barre de progression à l'écran pour suivre l'avancement des boucles longues (extraction de features, entraînement des époques). |
| **Matplotlib / Seaborn** | Visualisation des données | Utilisées pour créer des graphiques de performance (ex: courbes de perte d'entraînement) et la **Matrice de Confusion** pour analyser les erreurs de classification. |

---

## 🔥 Approche Avancée : Le Transformer Audio (AST)

Bien que l'implémentation actuelle utilise probablement des MFCCs avec un réseau classique, il est crucial de connaître les architectures de pointe comme **AST (Audio Spectrogram Transformer)**.

AST adapte l'architecture **Transformer** (révolutionnaire en PNL et Vision) à l'audio, en traitant le **Mel-spectrogramme** comme une image à analyser via des mécanismes d'attention.

### Pré-entraînements Possibles pour AST

L'efficacité d'AST dépend fortement des données de pré-entraînement utilisées :

| ImageNet pretrain | AudioSet pretrain | Valide ? | Explication |
| :---: | :---: | :---: | :--- |
| ❌ False | ❌ False | ✔️ Oui | Entraînement *from scratch* (bas de ligne). |
| ✔️ True | ❌ False | ✔️ Oui | Utilise les poids du **Vision Transformer (ViT)** entraîné sur ImageNet pour l'initialisation, puis *finetune* sur l'audio. |
| ❌ False | ✔️ True | ❌ NON | **Interdit :** L'architecture AST/ViT pour l'audio repose sur l'initialisation ImageNet pour ses couches de base. |
| ✔️ True | ✔️ True | ✔️ Oui | L'approche la plus performante : Initialisation ImageNet suivie d'un pré-entraînement sur **AudioSet** (vaste ensemble de données audio). C'est le meilleur *transfer learning*. |



---


