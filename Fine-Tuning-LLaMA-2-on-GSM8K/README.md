# Fine-Tuning LLaMA-2-7B on GSM8K for Mathematical Reasoning

This repository contains the implementation for fine-tuning the **LLaMA-2-7B** model on the **GSM8K dataset** to enhance mathematical reasoning capabilities. The project uses **Hugging Face Transformers**, **PyTorch**, and **PEFT (Parameter-Efficient Fine-Tuning)** techniques like LoRA to optimize the model for solving grade-school-level math problems.

## Features
- Fine-tunes the LLaMA-2-7B model using the GSM8K training set.
- Evaluates performance on the GSM8K test set by calculating accuracy based on exact matches of answers.
- Implements 4-bit quantization for efficient training and inference.
- Logs training metrics using **Weights & Biases (wandb)**.
- Saves and optionally pushes the fine-tuned model to the Hugging Face Hub.

---

## Dataset

The **GSM8K** dataset is a benchmark designed for evaluating mathematical reasoning in language models. It includes grade-school-level math problems with corresponding solutions.

### Dataset Splits:
1. **Training Set**: Used for fine-tuning.
2. **Validation Set**: Used for intermediate evaluation during training.
3. **Test Set**: Used for final evaluation of model performance.

---

## Setup Instructions

### Prerequisites
1. Python 3.8 or later
2. CUDA-compatible GPU (recommended)
3. Required Python libraries:
   - `transformers`
   - `datasets`
   - `torch`
   - `wandb`
   - `peft`

### Installation
1. Clone this repository:
2. Install dependencies:
3. Log in to Hugging Face using your Hugging Face key

---

## Training Process

The fine-tuning process involves:
1. Loading the LLaMA-2-7B model from Hugging Face.
2. Preprocessing the GSM8K dataset into prompts suitable for causal language modeling.
3. Applying PEFT with LoRA configurations to reduce computational costs.
4. Training the model over multiple epochs with logging enabled via Weights & Biases.

### Key Training Hyperparameters:
- Learning Rate: 2 × 10⁻⁴
- Batch Size: 1 (with gradient accumulation)
- Number of Epochs: 3
- Optimizer: AdamW

To start training, run:
python finetune_model.py
text

---

## Evaluation

After training, evaluate the model's performance on the GSM8K test set by calculating accuracy based on exact matches of answers:
python eval.py



## Files in Repository

| File Name          | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `finetune_model.py`| Script for fine-tuning LLaMA-2-7B on GSM8K using LoRA and PEFT techniques. |
| `eval.py`          | Script for evaluating the fine-tuned model's performance on the test set.  |
| `requirements.txt` | List of required Python dependencies.                                      |

---

## Insights and Challenges

1. **Insights**:
   - Fine-tuning large language models like LLaMA-2 can significantly enhance their reasoning abilities when provided with domain-specific datasets like GSM8K.
   - Parameter-efficient techniques such as LoRA enable effective training even with limited computational resources.

2. **Challenges**:
   - Training stability issues due to small batch sizes.
   - Balancing computational efficiency with model performance when using quantization techniques.

