import os

import torch
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, default_data_collator

from .prepare_data import prepare_data


def finetune_model(
    model_checkpoint="gpt2",
    output_dir="./results",
    model_output_dir="./fine_tuned_gpt2_summarizer",
    train_size=100,
    eval_size=10,
    max_length=1024,
    max_target_length=128,
    num_train_epochs=1,
    local_files_only=None,
):
    if local_files_only is None:
        local_files_only = (
            os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1"
        )

    tokenized_datasets, tokenizer = prepare_data(
        model_checkpoint=model_checkpoint,
        train_size=train_size,
        eval_size=eval_size,
        max_length=max_length,
        max_target_length=max_target_length,
        local_files_only=local_files_only,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_checkpoint,
        local_files_only=local_files_only,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        report_to=[],
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=default_data_collator,
    )

    print("Starting model fine-tuning...")
    trainer.train()

    print(f"Saving fine-tuned model to {model_output_dir}...")
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)


if __name__ == "__main__":
    finetune_model()
