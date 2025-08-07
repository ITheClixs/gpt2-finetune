from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

def prepare_data(model_checkpoint="gpt2", max_input_length=1024, max_target_length=128, train_size=100, eval_size=10):
    # Load the dataset
    print("Loading dataset...")
    # Using 'scientific_papers' with the 'arxiv' subset.
    # trust_remote_code=True is required to use the latest dataset script.
    dataset = load_dataset("scientific_papers", "arxiv", streaming=False, trust_remote_code=True)

    # Load tokenizer
    print(f"Loading tokenizer for {model_checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    # Set padding token for gpt2 if it's not already set.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    # The scientific_papers dataset contains 'article', 'abstract', and 'section_names'.
    # We remove the original columns after tokenization.
    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["article", "abstract", "section_names"]
    )


    # Limit dataset size for faster training using the .select() method for efficiency.
    train_dataset = tokenized_datasets["train"].select(range(train_size))
    validation_dataset = tokenized_datasets["validation"].select(range(eval_size))

    print("Dataset preparation complete.")
    return DatasetDict({"train": train_dataset, "validation": validation_dataset}), tokenizer

if __name__ == "__main__":
    tokenized_datasets, tokenizer = prepare_data()
    print(tokenized_datasets)
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")
