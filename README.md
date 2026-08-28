Neural Feature Extraction and AI Safety Control Using Sparse Autoencoders

Final Year CSE (AI & ML) Major Project

Overview

This project explores the use of Sparse Autoencoders (SAEs) for neural feature extraction and AI safety classification.

The system uses DistilBERT to obtain language representations, a Sparse Autoencoder to transform them into 2,048 sparse neural features, and a Logistic Regression classifier to estimate harmfulness.

The current prototype supports safety monitoring for both user input and AI output in a terminal-based environment.

Architecture

User Input
    ↓
DistilBERT
    ↓
Mean Pooling
    ↓
Sparse Autoencoder
    ↓
2,048 SAE Features
    ↓
Safety Classifier
    ↓
Harmful Probability
    ↓
ALLOW / REVIEW / BLOCK

For the current interaction pipeline:

User → Input Safety → AI Response → Output Safety → User

The AI response is currently simulated through terminal input.

Model Components

Base model: distilbert-base-uncased

Representation size: 768

SAE latent features: 2,048

Classifier: Logistic Regression

Decision thresholds:

< 0.30 → ALLOW

0.30–0.49 → REVIEW

≥ 0.50 → BLOCK

Training Data

The final classifier was trained using combined SAE features from:

Dataset

Samples

WildGuardMix

78,070

OASST1

14,240

Hard Negatives

643

Conversation Examples

100

Targeted Cybersecurity

298

Total

93,351

Training distribution:

Harmful: 41,793

Unharmful: 51,558

Evaluation

Held-Out Test Set

Metric

Result

Accuracy

80.69%

Harmful Precision

85.03%

Harmful Recall

68.57%

Harmful F1

75.92%

False Positive Rate

9.63%

False Negative Rate

31.43%

Confusion matrix:

[[854  91]
 [237 517]]

Safety Challenge Set

The 115-example challenge set produced:

Metric

Result

Accuracy

99.13%

Harmful Precision

100.00%

Harmful Recall

96.00%

Harmful F1

97.96%

False Positive Rate

0.00%

False Negative Rate

4.00%

Confusion matrix:

[[90  0]
 [ 1 24]]

The challenge-set result is a targeted evaluation and should not be treated as general-world performance.

Experiments Completed

The project has included:

SAE feature extraction and sparsity analysis

Safety classifier development

Hard-negative training

Conversation-focused training

Targeted cybersecurity dataset development

Feature specificity analysis

Feature ablation

Feature intervention

False-positive feature analysis

Threshold sweep

Final test evaluation

Safety challenge evaluation

Terminal input testing

Terminal output testing

Current Safety Pipeline

Run the main controller:

python src/safety_controller.py

Run the user-input and simulated-output pipeline:

python src/safety_pipeline.py

Example:

User Input
    ↓
Input Safety Check
    ↓
ALLOW
    ↓
AI Response
    ↓
Output Safety Check
    ↓
ALLOW / REVIEW / BLOCK

A harmful user request can be blocked before reaching the AI, while a harmful AI response can be blocked before being displayed.

Key Model Files

models/
├── sparse_autoencoder_full.pt
└── final_conversation_weighted_classifier.pkl

Project Structure

neural-feature-safety/
├── src/
├── models/
├── data/
│   └── processed/
├── experiments/
├── tests/
├── requirements.txt
└── README.md

Limitations

The current test-set performance is not production-ready.

Short conversational prompts can produce false positives.

Some ambiguous cybersecurity/privacy prompts can produce false negatives.

The challenge set is relatively small.

The current output pipeline uses a simulated AI response.

Next Phase

The next development stage is to integrate an actual AI model into the terminal pipeline:

User
 ↓
Input Safety Monitor
 ↓
AI Model
 ↓
Output Safety Monitor
 ↓
User

This will turn the current prototype into a complete two-sided AI safety gateway.

Status

Functional Research Prototype

The core SAE feature extraction, classifier, evaluation, feature analysis, and terminal safety-control components are operational.