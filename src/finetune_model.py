from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from .prepare_data import prepare_data

def finetune_model():
    model_checkpoint = "gpt2"
    tokenized_datasets, tokenizer = prepare_data(model_checkpoint=model_checkpoint)

    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2, # Only last 2 models are saved.
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
    )

    print("Starting model fine-tuning...")
    trainer.train()

    print("Saving fine-tuned model...")
    trainer.save_model("./fine_tuned_gpt2_summarizer")
    tokenizer.save_pretrained("./fine_tuned_gpt2_summarizer")

if __name__ == "__main__":
    finetune_model()
