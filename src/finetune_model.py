from transformers import AutoModelForCausalLM
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from .prepare_data import prepare_data

def finetune_model():
    model_checkpoint = "gpt2"
    tokenized_datasets, tokenizer = prepare_data(model_checkpoint=model_checkpoint, train_size=100, eval_size=10)

    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1, # Reduced for faster training
        weight_decay=0.01,
        save_total_limit=1, # Only last model is saved.
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10, # Log less frequently
        save_steps=10, # Save less frequently
    )

    # Ensure train_dataset and eval_dataset are Dataset objects, not DatasetDict
    if hasattr(tokenized_datasets, "train") and hasattr(tokenized_datasets, "validation"):
        train_dataset = tokenized_datasets.train
        eval_dataset = tokenized_datasets.validation
    else:
        train_dataset = tokenized_datasets
        eval_dataset = None

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("Starting model fine-tuning...")
    trainer.train()

    print("Saving fine-tuned model...")
    trainer.save_model("./fine_tuned_gpt2_summarizer")
    tokenizer.save_pretrained("./fine_tuned_gpt2_summarizer")

if __name__ == "__main__":
    finetune_model()
