from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer


DEFAULT_DATASET_NAME = "cnn_dailymail"
DEFAULT_DATASET_CONFIG = "3.0.0"


def prepare_data(
    model_checkpoint="gpt2",
    dataset_name=DEFAULT_DATASET_NAME,
    dataset_config=DEFAULT_DATASET_CONFIG,
    max_input_length=1024,
    max_target_length=128,
    train_size=100,
    eval_size=10,
):
    print("Loading dataset...")
    dataset = load_dataset(dataset_name, dataset_config, streaming=False)

    print(f"Loading tokenizer for {model_checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_examples = min(train_size, len(dataset["train"]))
    validation_examples = min(eval_size, len(dataset["validation"]))
    dataset_subset = DatasetDict(
        {
            "train": dataset["train"].select(range(train_examples)),
            "validation": dataset["validation"].select(range(validation_examples)),
        }
    )

    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["article"],
            max_length=max_input_length,
            truncation=True,
            padding="max_length",
        )
        labels = tokenizer(
            examples["highlights"],
            max_length=max_target_length,
            truncation=True,
            padding="max_length",
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Preprocessing dataset subset...")
    tokenized_datasets = dataset_subset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset_subset["train"].column_names,
    )

    print("Dataset preparation complete.")
    return tokenized_datasets, tokenizer


if __name__ == "__main__":
    tokenized_datasets, tokenizer = prepare_data()
    print(tokenized_datasets)
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")
