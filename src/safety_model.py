import torch
import joblib

from transformers import AutoTokenizer, AutoModel

from autoencoder import SparseAutoencoder


class SafetyModel:

    def __init__(self):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

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

        self.sae.load_state_dict(
            torch.load(
                "models/sparse_autoencoder_full.pt",
                map_location=self.device
            )
        )

        self.sae.eval()

        self.classifier = joblib.load(
            "models/final_safety_classifier.pkl"
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

            masked_hidden_states = hidden_states * mask

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

        sparse_features = sparse_features.cpu().numpy()

        probabilities = self.classifier.predict_proba(
            sparse_features
        )[0]

        unharmful_probability = probabilities[0]
        harmful_probability = probabilities[1]

        if harmful_probability >= 0.60:
            risk_level = "HIGH RISK"
            decision = "BLOCK"
        else:
            risk_level = "LOW RISK"
            decision = "ALLOW"

        return {
            "harmful_probability": harmful_probability,
            "unharmful_probability": unharmful_probability,
            "risk_level": risk_level,
            "decision": decision
        }