

## 🎛️ Préprocessing

La fonction `glob.glob` en Python permet de **rechercher et lister des fichiers** en utilisant un motif (pattern), par exemple :

```python
files = glob.glob("dataset/**/*.wav", recursive=True)
```

Elle est très utile pour parcourir automatiquement les fichiers audio.

---

## 🎵 Librosa

**librosa** est une bibliothèque Python spécialisée dans le traitement audio.
Elle permet notamment de :

* charger et lire des fichiers audio
* appliquer des transformations (STFT, spectrogrammes, filtres…)
* extraire des features audio :

  * MFCC
  * Chroma
  * Mel-spectrogram
  * etc.
* visualiser les signaux audio

👉 **C’est un peu comme NumPy + OpenCV, mais pour l’audio.**

---



## 🔥 AST — Pré-entraînements possibles

Le modèle **AST (Audio Spectrogram Transformer)** peut théoriquement être utilisé avec 4 combinaisons de pré-entraînements :

| ImageNet pretrain | AudioSet pretrain | Valide ? | Explication                                                |
| ----------------- | ----------------- | -------- | ---------------------------------------------------------- |
| ❌ False           | ❌ False           | ✔️ Oui   | Modèle vierge (entraînement from scratch).                 |
| ✔️ True           | ❌ False           | ✔️ Oui   | ViT pré-entraîné sur ImageNet, puis adapté à l’audio.      |
| ❌ False           | ✔️ True           | ❌ NON    | Interdit : le modèle AudioSet dépend forcément d’ImageNet. |
| ✔️ True           | ✔️ True           | ✔️ Oui   | Modèle AST pré-entraîné AudioSet (meilleure performance).  |

---


<img width="467" height="411" alt="Image" src="https://github.com/user-attachments/assets/41e8752e-244e-41b8-a1fe-ca9e7401e039" />