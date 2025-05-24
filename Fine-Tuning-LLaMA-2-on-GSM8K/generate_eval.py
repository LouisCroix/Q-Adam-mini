import os
import logging
import torch
from datasets import load_dataset,load_from_disk
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, PeftModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# dataset_name = "openai/gsm8k"
dataset = load_from_disk(f"./datasets/gsm8k/main/test")

tokenizer_path = "./.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6"
# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.add_special_tokens({"pad_token": "<<PAD>>"})

model_path = "./q-adam-mini-checkpoints/model_AdamW-llama_lora_gsm_2e-5_80_128_256"
# model_path = "./.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6"
if "lora" in model_path:
    lora_config = LoraConfig.from_pretrained(model_path)
    new_model = AutoModelForCausalLM.from_pretrained(lora_config.base_model_name_or_path).to(device)
    new_model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=4)
    new_model = PeftModel.from_pretrained(new_model, model_path)
else:
    new_model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

test_dataset = dataset

def create_prompt(rec):
    # inst = "<INST> <SYS> Read the Question below and provide a numerical answer.</SYS></INST>\n"
    # question = f"<INST> QUESTION:\n{rec['question']}\n"
    # end = "</INST>"

    # parts = [inst, question, end]
    # formatted_prompt = "\n".join(parts).replace('\\n', '\n')
    # rec["text"] = formatted_prompt
    # return rec
    rec["text"] = f'<question>: {rec["question"]}\n<answer>: '
    return rec

test_dataset = test_dataset.map(create_prompt)
# print(test_dataset[1111]["text"])

# Test text input
tst = """<INST> <SYS> Read the Question below and provide a numerical answer.</SYS></INST>
<INST> INSTRUCTION:
Cynthia eats one serving of ice cream every night. She buys cartons of ice cream with 15 servings of ice cream per carton at a cost of $4.00 per carton. After 60 days, how much will she spend on ice cream?
</INST>
####"""
batch = tokenizer(tst, return_tensors='pt').to(device)
new_model.eval()
with torch.cuda.amp.autocast():
    output_tokens = new_model.generate(**batch, max_new_tokens=90)

# print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))

# Function to extract numerical answer
import re
def extract_numerical_answer_from_dataset(text):
    match = re.search(r'#### (\d+)', text)
    return int(match.group(1)) if match else None

def extract_numerical_answer_from_true_answer(text):
    match = re.search(r'#### (\d+)', text)
    return int(match.group(1)) if match else None

# Function to add extracted numbers to dataset
def add_extracted_number(example):
    example['extracted_number'] = extract_numerical_answer_from_true_answer(example['answer'])
    return example

# Apply extraction to test dataset
test_dataset = test_dataset.map(add_extracted_number)

from tqdm import tqdm

# Initialize lists for predictions and truths
predicted_answers = []
true_answers = []

# Variables to track accuracy
correct = 0
total = 0

# Evaluate model on dataset
for example in tqdm(test_dataset, desc="Evaluating model"):
    input_text = tokenizer(example['text'], return_tensors='pt').to(device)
    with torch.no_grad():
        output_tokens = new_model.generate(**input_text, max_length=512)
    prediction_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    predicted_number = extract_numerical_answer_from_dataset(prediction_text)
    print(f"input: {example['text']}\nanswer: {example['extracted_number']}\noutput: {prediction_text}\npred: {predicted_number}", flush=True)
    true_number = example['extracted_number']

    # Append predictions and truths
    predicted_answers.append(predicted_number)
    true_answers.append(true_number)

    # Check correctness
    if predicted_number == true_number:
        correct += 1
    
    total += 1
    print(f"Current accuracy: {correct/total:.2%}", flush=True)

# Calculate final accuracy
accuracy = correct / total if total > 0 else 0
print(f"Accuracy: {accuracy:.2%}", flush=True)