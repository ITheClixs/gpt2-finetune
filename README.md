# GPT-2 Fine-tuning for Scientific Paper Summarization

This project demonstrates how to fine-tune a GPT-2 model for scientific paper summarization using a minimal dataset. It's optimized for quick execution on resource-constrained environments like a MacBook Air M4 without a dedicated GPU.

## Overview

The primary goal is to provide a basic example of fine-tuning a pre-trained GPT-2 model to generate summaries of scientific papers. Due to computational limitations and time constraints (aiming for ~30 minutes of training), the model is trained on a very small subset of the `arxiv_daily` dataset. This setup is intended for demonstrating the fine-tuning workflow rather than achieving high-quality summarization.

## Features

*   **Minimal Dataset Fine-tuning:** Configured to train on a small subset of the `arxiv_daily` dataset for rapid iteration.
*   **Hugging Face Transformers:** Leverages the powerful `transformers` library for model loading, tokenization, and training.
*   **Modular Structure:** Organized into a `src` directory for better code management.

## Requirements

*   Python 3.8+
*   `transformers`
*   `datasets`
*   `torch`

### Python Dependencies

Install dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## Project Structure

```
gpt2-finetune/
├── src/
│   ├── __init__.py           # Makes `src` a Python package
│   ├── prepare_data.py       # Handles dataset loading and preprocessing
│   └── finetune_model.py     # Contains the fine-tuning logic for GPT-2
├── main.py                   # Main entry point to start fine-tuning
└── requirements.txt          # Lists Python dependencies
```

## How to Use

Follow these steps to set up the environment and run the fine-tuning process:

1.  **Navigate to the project directory:**

    ```bash
    cd /path/to/gpt2-finetune
    ```

2.  **Create and activate a virtual environment (recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate    # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the fine-tuning script:**

    ```bash
    python main.py
    ```

    This command will:

    *   Download the `arxiv_daily` dataset (if not already cached).
    *   Preprocess a small subset of the dataset (100 training examples, 10 validation examples).
    *   Load a pre-trained GPT-2 model.
    *   Fine-tune the model for 1 epoch.
    *   Save the fine-tuned model and tokenizer to `./fine_tuned_gpt2_summarizer`.

## Important Notes

*   **Training Time:** The fine-tuning process is configured to be very fast (aiming for under 30 minutes on a modern CPU) by using a drastically reduced dataset size and fewer training epochs. This is ideal for quick demonstrations and testing the pipeline.
*   **Summarization Quality:** Due to the minimal training data, the resulting fine-tuned model will have very limited summarization capabilities. It is not expected to produce high-quality, coherent summaries for real-world use cases.
*   **Computational Resources:** While optimized for CPU, fine-tuning large language models can still be resource-intensive. Monitor your system's resource usage during the process.

## Next Steps

After successful fine-tuning, you can proceed to build a simple application to load and use your fine-tuned model for generating summaries.

## License

This project is open-source and available for educational and personal use.

## Contributions

Contributions and improvements are welcome. Feel free to fork the repository and submit pull requests.