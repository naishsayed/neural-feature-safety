import pandas as pd
from pathlib import Path


# File locations
TRAIN_PATH = "data/raw/wildguard_train.parquet"
TEST_PATH = "data/raw/wildguard_test.parquet"

OUTPUT_DIR = Path("data/processed")


def prepare_prompt_data(data):
    """Prepare data for prompt safety classification."""

    prompt_data = data[
        [
            "prompt",
            "prompt_harm_label",
            "adversarial",
            "subcategory",
        ]
    ].copy()

    # Remove rows with missing values
    prompt_data = prompt_data.dropna(
        subset=["prompt", "prompt_harm_label"]
    )

    return prompt_data


def prepare_response_data(data):
    """Prepare data for response safety classification."""

    response_data = data[
        [
            "response",
            "response_harm_label",
            "response_refusal_label",
            "subcategory",
        ]
    ].copy()

    # Keep only responses that have a safety label
    response_data = response_data[
        response_data["response_harm_label"].isin(
            ["harmful", "unharmful"]
        )
    ]

    # Remove empty responses
    response_data = response_data.dropna(
        subset=["response"]
    )

    return response_data


def main():

    print("Loading WildGuardMix dataset...")

    train_data = pd.read_parquet(TRAIN_PATH)
    test_data = pd.read_parquet(TEST_PATH)

    print("Training rows:", len(train_data))
    print("Testing rows:", len(test_data))

    # Prepare prompt datasets
    prompt_train = prepare_prompt_data(train_data)
    prompt_test = prepare_prompt_data(test_data)

    # Prepare response datasets
    response_train = prepare_response_data(train_data)
    response_test = prepare_response_data(test_data)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save processed datasets
    prompt_train.to_csv(
        OUTPUT_DIR / "prompt_train.csv",
        index=False
    )

    prompt_test.to_csv(
        OUTPUT_DIR / "prompt_test.csv",
        index=False
    )

    response_train.to_csv(
        OUTPUT_DIR / "response_train.csv",
        index=False
    )

    response_test.to_csv(
        OUTPUT_DIR / "response_test.csv",
        index=False
    )

    print("\nDataset preparation completed.")

    print("\nPrompt dataset:")
    print("Train:", len(prompt_train))
    print("Test:", len(prompt_test))

    print("\nResponse dataset:")
    print("Train:", len(response_train))
    print("Test:", len(response_test))

    print("\nPrompt labels:")
    print(prompt_train["prompt_harm_label"].value_counts())

    print("\nResponse labels:")
    print(response_train["response_harm_label"].value_counts())


if __name__ == "__main__":
    main()