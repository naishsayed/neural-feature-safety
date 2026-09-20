import os

import torch
import joblib

from transformers import AutoTokenizer, AutoModel

from .autoencoder import SparseAutoencoder


class SafetyModel:

    def __init__(
        self,
        threshold=0.50
    ):

        root_dir = os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.threshold = threshold

        self.tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased"
        )

        self.distilbert = AutoModel.from_pretrained(
            "distilbert-base-uncased"
        ).to(self.device)

        self.distilbert.eval()

        self.sae = SparseAutoencoder(
            input_dim=768,
            hidden_dim=2048
        ).to(self.device)

        sae_path = os.path.join(
            root_dir,
            "models",
            "sparse_autoencoder_full.pt"
        )

        self.sae.load_state_dict(
            torch.load(
                sae_path,
                map_location=self.device,
                weights_only=False
            )
        )

        self.sae.eval()

        classifier_path = os.path.join(
            root_dir,
            "models",
            "final_safety_classifier.pkl"
        )

        self.classifier = joblib.load(
            classifier_path
        )

    def analyze(self, text):

        inputs = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = self.distilbert(
                **inputs
            )

            hidden_states = outputs.last_hidden_state

            attention_mask = inputs["attention_mask"]

            mask = attention_mask.unsqueeze(
                -1
            ).expand(
                hidden_states.size()
            ).float()

            masked_hidden_states = (
                hidden_states * mask
            )

            summed = masked_hidden_states.sum(
                dim=1
            )

            counts = mask.sum(
                dim=1
            )

            pooled = summed / counts

            _, sparse_features = self.sae(
                pooled
            )

            feature_activation = sparse_features[
                0,
                850
            ].item()

        sparse_features_np = (
            sparse_features.cpu().numpy()
        )

        probabilities = (
            self.classifier.predict_proba(
                sparse_features_np
            )[0]
        )

        unharmful_probability = probabilities[0]
        harmful_probability = probabilities[1]

        if harmful_probability >= self.threshold:

            risk_level = "HIGH RISK"
            decision = "BLOCK"

        else:

            risk_level = "LOW RISK"
            decision = "ALLOW"

        return {
            "harmful_probability":
                float(harmful_probability),
            "unharmful_probability":
                float(unharmful_probability),
            "risk_level":
                risk_level,
            "decision":
                decision,
            "threshold":
                self.threshold,
            "feature_850_activation":
                float(feature_activation),
            "device":
                str(self.device)
        }