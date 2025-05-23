import os
import warnings
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from datasets import load_dataset, Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline)
from sklearn.metrics import accuracy_score
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import wandb
import logging

# Set up Weights & Biases (wandb) for logging
os.environ["WANDB_PROJECT"] = #your environment name
os.environ["WANDB_LOG_MODEL"] = "true"
os.environ["WANDB_WATCH"] = "false"
wandb.login(key="your key")

# Disable warnings and set environment variables
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {device}")

# Step 1: Use Hugging Face Token for Model Access
hf_token = # your hugging face token
model_name = "meta-llama/Llama-2-7b-hf"

# Load Model with 4-bit Quantization Configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load the model and tokenizer from Hugging Face using the token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=bnb_config,  # Add BitsAndBytes config here
    use_auth_token=hf_token,
    torch_dtype=torch.float16
).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
tokenizer.pad_token = tokenizer.eos_token

# Step 2: Load the GSM8K dataset and split it into train and evaluation sets
dataset = load_dataset("openai/gsm8k", "main")
split_dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)
train_data = split_dataset['train']
eval_data = split_dataset['test']
test_data = dataset['test']

def create_prompt(example):
    return {'prompt': f"Q: {example['question']} A: {example['answer']}"}

def create_prompt_for_evaluation(example):
    return {'prompt': f"Q: {example['question']} A:"}

# Apply different preprocessing to evaluation dataset
eval_data = eval_data.map(create_prompt_for_evaluation)
train_data = train_data.map(create_prompt)

def preprocess_data(examples):
    prompts = ["Q: " + q + " \nA: " + a for q, a in zip(examples['question'], examples['answer'])]
    model_inputs = tokenizer(prompts, truncation=True, padding="max_length", max_length=512)
    return model_inputs

def preprocess_data_for_validation(examples):
    prompts = ["Q: " + q + " \nA: " for q in examples['question']]
    model_inputs = tokenizer(prompts, truncation=True, padding="max_length", max_length=512)
    return model_inputs

train_dataset = train_data.map(preprocess_data, batched=True)
eval_dataset = eval_data.map(preprocess_data_for_validation, batched=True)

# Step 3: Define PEFT LoRA Configuration
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# Step 4: Training Arguments
training_args = TrainingArguments(
    output_dir="./logs",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    optim="adamw_torch",
    logging_steps=50,
    save_strategy="epoch",
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    evaluation_strategy="epoch",
    save_total_limit=2,
    report_to="wandb",
)

# Initialize the Trainer and start fine-tuning
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    dataset_text_field="prompt",
    tokenizer=tokenizer,
    max_seq_length=1024,
    args=training_args,
    packing=False,
)

trainer.train()

# Save the model locally before pushing
trainer.save_model("./logs/final_model")

# Push the fine-tuned model to Hugging Face Hub
model.push_to_hub(
    "#Your repo",
    use_auth_token=hf_token,
    commit_message="fine-tuned on GSM-8k in A100",
    private=False
)