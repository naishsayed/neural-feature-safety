<h1 align="center">🧠 Neural Feature Safety</h1>

<p align="center">
  <em>AI safety classification and two-sided content monitoring using Sparse Autoencoder neural features</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Transformers-HuggingFace-yellow.svg" alt="Transformers">
  <img src="https://img.shields.io/badge/scikit--learn-ML-orange.svg" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Sparse%20Features-2048-purple.svg" alt="SAE Features">
</p>

---

## 📖 About The Project

**Neural Feature Safety** is a research-oriented AI safety system that extracts sparse neural features from text and uses them to estimate whether content is harmful.

The current system combines **DistilBERT**, a **Sparse Autoencoder (SAE)**, and a **Logistic Regression classifier** to create a terminal-based safety controller.

✨ **Current capabilities:**
- 🧠 2,048-dimensional sparse neural feature extraction
- 🛡️ Harmful / unharmful classification
- 🚦 `ALLOW`, `REVIEW`, and `BLOCK` decisions
- 👤 User-input safety monitoring
- 🤖 AI-output safety monitoring
- 🔬 Feature ablation and intervention experiments
- 🎯 Targeted cybersecurity safety evaluation
- 📊 Threshold and challenge-set evaluation

---

## ⚙️ System Architecture

```text
                    ┌─────────────────┐
                    │   User Prompt   │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │    DistilBERT   │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │   Mean Pooling  │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Sparse Autoencoder│
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ 2048 SAE Features│
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Safety Classifier│
                    └────────┬────────┘
                             ↓
                 ┌───────────┼───────────┐
                 ↓           ↓           ↓
              ALLOW       REVIEW       BLOCK
```

The interaction pipeline monitors both sides of the conversation:

```text
User
 ↓
Input Safety Monitor
 ↓
AI Model / Simulated Response
 ↓
Output Safety Monitor
 ↓
User
```

---

## 🧩 Model Components

| Component | Configuration |
|---|---|
| Base Model | `distilbert-base-uncased` |
| Transformer Representation | 768 dimensions |
| Sparse Autoencoder | 2,048 latent features |
| Classifier | Logistic Regression |
| Scaling | StandardScaler |
| Training | Balanced classification |
| Block Threshold | `0.50` |
| Review Threshold | `0.30` |

### Decision Policy

```text
Probability < 0.30       → ALLOW
0.30 ≤ Probability < 0.50 → REVIEW
Probability ≥ 0.50       → BLOCK
```

---

## 📊 Training Data

The final classifier combines SAE features from multiple sources:

| Dataset | Samples |
|---|---:|
| WildGuardMix | 78,070 |
| OASST1 | 14,240 |
| Hard Negatives | 643 |
| Conversation Examples | 100 |
| Targeted Cybersecurity | 298 |
| **Total** | **93,351** |

**Training distribution**

```text
Harmful       41,793
Unharmful     51,558
```

---

## 📈 Evaluation

### Held-Out Test Set

| Metric | Result |
|---|---:|
| Accuracy | **80.69%** |
| Harmful Precision | **85.03%** |
| Harmful Recall | **68.57%** |
| Harmful F1 | **75.92%** |
| False Positive Rate | **9.63%** |
| False Negative Rate | **31.43%** |

### Safety Challenge Set

The final model was also evaluated on a targeted 115-example challenge set.

| Metric | Result |
|---|---:|
| Accuracy | **99.13%** |
| Harmful Precision | **100.00%** |
| Harmful Recall | **96.00%** |
| Harmful F1 | **97.96%** |
| False Positive Rate | **0.00%** |
| False Negative Rate | **4.00%** |

> The challenge set is a targeted evaluation and should not be interpreted as general-world model performance.

---

## 🔬 Experiments Performed

The project currently includes experiments covering:

- Sparse Autoencoder feature extraction
- SAE sparsity analysis
- Safety classifier training
- Hard-negative classification
- Conversation-enhanced classification
- Targeted cybersecurity data
- Feature specificity
- Feature contribution analysis
- Feature ablation
- Feature intervention
- False-positive analysis
- Threshold calibration and sweep
- Challenge-set evaluation
- Input safety testing
- Output safety testing

---

## 🚀 How To Run

### 1️⃣ Clone the repository

```bash
git clone https://github.com/naishsayed/neural-feature-safety.git
cd neural-feature-safety
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the safety controller

```bash
python src/safety_controller.py
```

### 4️⃣ Run the complete terminal pipeline

```bash
python src/safety_pipeline.py
```

The pipeline accepts a user prompt, evaluates it, and—when allowed—accepts a simulated AI response for a second safety evaluation.

---

## 📁 Project Structure

```text
neural-feature-safety/
│
├── data/
│   └── processed/
│
├── models/
│   ├── sparse_autoencoder_full.pt
│   └── final_conversation_weighted_classifier.pkl
│
├── src/
│   ├── autoencoder.py
│   ├── safety_controller.py
│   ├── safety_model.py
│   ├── safety_pipeline.py
│   ├── train_final_classifier.py
│   ├── evaluate_final_challenge.py
│   ├── evaluate_threshold_sweep.py
│   └── ...
│
├── experiments/
├── tests/
├── requirements.txt
└── README.md
```

---

## 🛠️ Current Status

**Functional Research Prototype**

The SAE feature extraction pipeline, classifier, safety controller, challenge evaluation, and terminal input/output monitoring are currently operational.

### Next Phase

The next stage is to replace the simulated AI response with an actual AI model so the system can operate as a complete safety gateway:

```text
User Prompt
     ↓
Input Safety
     ↓
AI Model
     ↓
Output Safety
     ↓
Safe Response
```

---

## 👨‍💻 Project

**Final Year CSE (AI & ML) Major Project**

Built as an experimental system for studying **neural representations, sparse features, interpretability, and AI safety control**.
