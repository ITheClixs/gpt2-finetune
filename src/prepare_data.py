from datasets import load_dataset
from transformers import AutoTokenizer

def prepare_data(model_checkpoint="gpt2", max_input_length=1024, max_target_length=128, train_size=100, eval_size=10):
    # Load the dataset
    print("Loading dataset...")
    dataset = load_dataset("arxiv_daily")

    # Load tokenizer
    print(f"Loading tokenizer for {model_checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def preprocess_function(examples):
        # Tokenize the input (article) and target (abstract)
        model_inputs = tokenizer(
            examples["article"],
            max_length=max_input_length,
            truncation=True,
            padding="max_length"
        )
        labels = tokenizer(
            examples["abstract"],
            max_length=max_target_length,
            truncation=True,
            padding="max_length"
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Preprocessing dataset...")
    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["article", "abstract", "id"]
    )


    # Limit dataset size for faster training
    # Use slicing if select is not available
    train_dataset = list(tokenized_datasets["train"])[:train_size]
    validation_dataset = list(tokenized_datasets["validation"])[:eval_size]

    print("Dataset preparation complete.")
    return {"train": train_dataset, "validation": validation_dataset}, tokenizer

if __name__ == "__main__":
    tokenized_datasets, tokenizer = prepare_data()
    print(tokenized_datasets)
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")
