# 🐱 Cat Breeds T-Zone Biometrics Dataset

> **HOG Feature Vectors extracted from the T-Zone (inter-ocular region) of 25 cat breeds — ready for machine learning classification, breed recognition, and biometric analysis.**

---

## 📖 Overview

This dataset provides **Histogram of Oriented Gradients (HOG)** feature descriptors extracted from the **T-zone** of cat faces — the region bounded by the eyes and the nose bridge. This anatomically consistent region is rich in discriminative texture and shape information, making it ideal for breed classification, facial recognition, and feline biometrics research.

| Property | Value |
|---|---|
| **Dataset Name** | Cat Breeds T-Zone Biometrics |
| **Total Samples** | 11,582 cats |
| **Breeds** | 25 |
| **Feature Dimensionality** | 1,764 per sample |
| **Feature Type** | HOG (Histogram of Oriented Gradients) |
| **Extraction Region** | T-Zone (Eyes → Nose bridge) |
| **Data Format** | JSON |
| **Feature Range** | [0.0, ~0.90] (L2-Hys normalized) |

---

## 🧬 HOG Feature Extraction Details

### What is the T-Zone?
The **T-Zone** is the facial region between the two eyes and the nose — a triangular/trapezoidal area that encodes highly breed-specific geometric ratios (inter-eye distance, eye-to-nose proportions) and local texture patterns (fur density, skin texture in hairless breeds).

### HOG Parameters

| Parameter | Value |
|---|---|
| **Orientations** | 9 gradient bins (0°–180°) |
| **Cell Grid** | 14 × 14 cells = 196 cells |
| **Feature Vector Length** | 196 cells × 9 orientations = **1,764** |
| **Normalization** | L2-Hys block normalization |
| **Values** | Float64, range [0.0, ~0.90] |

The resulting 1,764-dimensional vector captures:
- **Gradient orientation histograms** — encodes edge directions (fur patterns, outline shapes)
- **Spatial layout** — 14×14 spatial resolution over the T-zone patch
- **Local normalization** — invariant to illumination changes

---

## 📊 Dataset Statistics

### Per-Breed Sample Counts

| Breed | Cat Count | Min Feature | Max Feature | Mean | Std Dev |
|---|---:|---:|---:|---:|---:|
| American Bobtail | 445 | 0.0000 | 0.7002 | 0.1271 | 0.1079 |
| Tuxedo | 951 | 0.0000 | 0.8553 | 0.1219 | 0.1137 |
| Oriental Long Hair | 18 | 0.0000 | 0.6364 | 0.1268 | 0.1081 |
| Abyssinian | 132 | 0.0000 | 0.6046 | 0.1302 | 0.1040 |
| Canadian Hairless | 4 | 0.0000 | 0.6048 | 0.1238 | 0.1115 |
| Japanese Bobtail | 45 | 0.0000 | 0.6193 | 0.1252 | 0.1100 |
| Persian | 1,121 | 0.0000 | 0.7850 | 0.1301 | 0.1041 |
| Selkirk Rex | 20 | 0.0000 | 0.5929 | 0.1295 | 0.1049 |
| Tortoiseshell | 650 | 0.0000 | 0.9009 | 0.1316 | 0.1023 |
| Ragamuffin | 72 | 0.0000 | 0.6706 | 0.1266 | 0.1084 |
| Dilute Calico | 1,597 | 0.0000 | 0.7112 | 0.1280 | 0.1067 |
| Chinchilla | 0 | — | — | — | — |
| Chausie | 10 | 0.0000 | 0.5037 | 0.1272 | 0.1077 |
| Tabby | 1,536 | 0.0000 | 0.6727 | 0.1290 | 0.1055 |
| Cymric | 5 | 0.0000 | 0.4552 | 0.1328 | 0.1007 |
| Nebelung | 50 | 0.0000 | 0.5976 | 0.1336 | 0.0996 |
| American Wirehair | 9 | 0.0000 | 0.5199 | 0.1254 | 0.1098 |
| Sphynx - Hairless Cat | 41 | 0.0000 | 0.6306 | 0.1255 | 0.1096 |
| Extra-Toes Cat (Hemingway) | 488 | 0.0000 | 0.8794 | 0.1273 | 0.1076 |
| Tonkinese | 78 | 0.0000 | 0.5468 | 0.1317 | 0.1021 |
| Dilute Tortoiseshell | 1,202 | 0.0000 | 0.6508 | 0.1325 | 0.1011 |
| Burmese | 103 | 0.0000 | 0.6248 | 0.1314 | 0.1025 |
| Domestic Medium Hair | 2,523 | 0.0000 | 0.7270 | 0.1281 | 0.1066 |
| Ocicat | 60 | 0.0000 | 0.6670 | 0.1300 | 0.1043 |
| Himalayan | 422 | 0.0000 | 0.6145 | 0.1327 | 0.1009 |

> ⚠️ **Note:** The `Chinchilla` breed has **0 samples** in this release. It is listed in the schema but lacks extracted data.

### Dataset Distribution

```
Domestic Medium Hair   ████████████████████████  2523
Dilute Calico          ████████████████          1597
Tabby                  ███████████████           1536
Dilute Tortoiseshell   ████████████              1202
Persian                ███████████               1121
Tortoiseshell          ██████                     650
Extra-Toes Cat         █████                      488
American Bobtail       ████                       445
Himalayan              ████                       422
Tuxedo                 █████████                  951
Ragamuffin             ▉                           72
Tonkinese              ▉                           78
Burmese                █                          103
Abyssinian             █                          132
Nebelung               ▌                           50
Ocicat                 ▌                           60
Japanese Bobtail       ▌                           45
Sphynx - Hairless Cat  ▍                           41
Selkirk Rex            ▏                           20
Oriental Long Hair     ▏                           18
Chausie                ▏                           10
American Wirehair      ▏                            9
Canadian Hairless      ▏                            4
Cymric                 ▏                            5
Chinchilla             (no samples)                 0
```

---

## 📁 File Structure

```
cat-breeds-tzone-biometrics/
│
├── all_50_cat_breeds_master_biometrics.json   ← Main dataset file (~860 MB)
├── README.md                                  ← This file
└── analyse_dataset.py                         ← Python analysis/stats script
```

---

## 🗂️ JSON Schema

```json
{
  "dataset": "Cat Breeds T-Zone Biometrics",
  "total_cats_extracted": 11582,
  "breeds": [
    {
      "breed_name": "American Bobtail",
      "cats": [
        {
          "cat_id": "Cat 2",
          "original_file": "32216375_2222.jpg",
          "features": [0.0616, 0.1045, 0.1469, ...]   // 1764 float values
        }
      ]
    }
  ]
}
```

### Field Descriptions

| Field | Type | Description |
|---|---|---|
| `dataset` | `string` | Dataset identifier label |
| `total_cats_extracted` | `int` | Total number of cat records |
| `breeds[]` | `array` | List of breed objects |
| `breed_name` | `string` | Name of the cat breed |
| `cats[]` | `array` | List of individual cat records |
| `cat_id` | `string` | Unique identifier (e.g. `"Cat 2"`) |
| `original_file` | `string` | Source image filename |
| `features` | `float[]` | 1764-dimensional HOG descriptor |

---

## 🐍 Quick Start (Python)

### Load the Dataset

```python
import json
import numpy as np

with open("all_50_cat_breeds_master_biometrics.json", "r") as f:
    data = json.load(f)

# Flatten into feature matrix and label array
X, y = [], []
for breed in data["breeds"]:
    for cat in breed["cats"]:
        X.append(cat["features"])
        y.append(breed["breed_name"])

X = np.array(X)   # shape: (11582, 1764)
print(f"Feature matrix shape: {X.shape}")
print(f"Unique breeds: {len(set(y))}")
```

### Train a Classifier

```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

clf = SVC(kernel="rbf", C=10, gamma="scale")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))
```

### Visualise HOG Feature Map for a Cat

```python
import matplotlib.pyplot as plt
import numpy as np

# Reshape 1764-dim vector → (14, 14, 9) → mean over orientations → 14×14 heatmap
cat = data["breeds"][0]["cats"][0]
feat = np.array(cat["features"]).reshape(14, 14, 9)
heatmap = feat.mean(axis=2)

plt.figure(figsize=(6, 5))
plt.imshow(heatmap, cmap="hot", interpolation="bilinear")
plt.colorbar(label="Mean HOG magnitude")
plt.title(f"HOG Heatmap — {data['breeds'][0]['breed_name']}")
plt.axis("off")
plt.tight_layout()
plt.savefig("hog_heatmap.png", dpi=150)
plt.show()
```

---

## 🔬 Research Applications

This dataset is suitable for:

- 🐾 **Cat Breed Classification** — Train SVMs, CNNs, or GBM classifiers on the 1764-dim feature space
- 🧠 **Transfer Learning Baselines** — Use HOG features as fixed embeddings to benchmark classical vs. deep models
- 📏 **Biometric Distance Analysis** — Measure Euclidean / cosine distances across breeds for similarity studies
- 🔍 **Facial Landmark Consistency Studies** — Analyse how T-zone geometry varies across breeds
- 📉 **Dimensionality Reduction** — Apply PCA / t-SNE / UMAP to visualise breed clusters in 2D/3D
- 🐱 **Rare Breed Detection** — Identify data imbalance challenges (e.g., Cymric = 5 vs. Domestic Medium Hair = 2523)

---

## ⚠️ Known Limitations

| Limitation | Detail |
|---|---|
| **Class Imbalance** | Ranges from 4 (Canadian Hairless) to 2,523 (Domestic Medium Hair) |
| **Missing Breed** | Chinchilla has 0 samples — excluded from any classification task |
| **File Naming** | Breed name `"all_50_cat_breeds..."` suggests 50 breeds were planned; only 25 are present with data |
| **Region Dependency** | Features are T-zone only — whole-face or body features not included |
| **No Landmark Coords** | Exact pixel coordinates of the extracted patch are not stored in the JSON |

---

---

## 📄 License

This dataset is released for **academic and research use**. Please ensure compliance with the source image licenses before commercial use.

---

*Made with ❤️ for feline biometrics research.*
