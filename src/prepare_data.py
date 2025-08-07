from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

def prepare_data(model_checkpoint="gpt2", max_input_length=1024, max_target_length=128, train_size=100, eval_size=10):
    # Load the dataset
    print("Loading dataset...")
    # Using 'cnn_dailymail' as a replacement for the deprecated 'scientific_papers' dataset.
    # This is a standard dataset for summarization tasks.
    dataset = load_dataset("cnn_dailymail", "3.0.0", streaming=False)

    # Load tokenizer
    print(f"Loading tokenizer for {model_checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    # Set padding token for gpt2 if it's not already set.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess_function(examples):
        # Tokenize the input (article) and target (highlights)
        model_inputs = tokenizer(
            examples["article"],
            max_length=max_input_length,
            truncation=True,
            padding="max_length"
        )
        # The target column in cnn_dailymail is 'highlights'
        labels = tokenizer(
            examples["highlights"],
            max_length=max_target_length,
            truncation=True,
            padding="max_length"
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Preprocessing dataset...")
    # The cnn_dailymail dataset contains 'article', 'highlights', and 'id'.
    # We remove the original columns after tokenization.
    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["article", "highlights", "id"]
    )


    # Limit dataset size for faster training using the .select() method for efficiency.
    # For streaming/IterableDataset, use .take() directly.
    if isinstance(tokenized_datasets, DatasetDict):
        if hasattr(tokenized_datasets["train"], "select"):
            train_dataset = tokenized_datasets["train"].select(range(train_size))
            validation_dataset = tokenized_datasets["validation"].select(range(eval_size))
        elif isinstance(tokenized_datasets["train"], list):
            train_dataset = tokenized_datasets["train"][:train_size]
            validation_dataset = tokenized_datasets["validation"][:eval_size]
        else:
            train_dataset = tokenized_datasets["train"].take(train_size)
            validation_dataset = tokenized_datasets["validation"].take(eval_size)
    else:
        # For IterableDataset, use .take() directly on the dataset object
        train_dataset = tokenized_datasets.take(train_size)
        validation_dataset = tokenized_datasets.take(eval_size)

    print("Dataset preparation complete.")
    return DatasetDict({"train": train_dataset, "validation": validation_dataset}), tokenizer

if __name__ == "__main__":
    tokenized_datasets, tokenizer = prepare_data()
    print(tokenized_datasets)
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")