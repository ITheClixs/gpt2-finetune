# GPT-2 Fine-Tuning For Summarization

This project fine-tunes a GPT-2 causal language model on a very small subset of the Hugging Face `cnn_dailymail` dataset. It is a minimal CPU-friendly training pipeline intended to demonstrate fine-tuning mechanics, not to produce a strong summarization model.

## What It Does

Running `python main.py`:

1. Loads `cnn_dailymail` version `3.0.0`.
2. Selects 100 training articles and 10 validation articles.
3. Formats each example as a causal language modeling prompt:

   ```text
   Summarize the following article:

   <article>

   Summary:
   <highlights>
   ```

4. Masks the prompt portion in the labels so loss is computed only on the summary tokens.
5. Fine-tunes GPT-2 for 1 epoch.
6. Saves the trained model to `./fine_tuned_gpt2_summarizer`.

## Requirements

- Python 3.8+
- `accelerate`
- `transformers`
- `datasets`
- `torch`

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

```bash
cd /path/to/gpt2-finetune
python main.py
```

## Notes

- The code expects the dataset and model to be available through Hugging Face, unless they are already cached locally.
- Training is intentionally small and fast, so summary quality will be limited.
- The preprocessing is designed for GPT-2 style causal LM training, not encoder-decoder summarization.
