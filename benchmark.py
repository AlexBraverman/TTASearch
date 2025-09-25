import os
import time
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen3-4B-Instruct-2507"  # can change to model needed

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def llm_generate_local(question, max_new_tokens=1024):
    # Apply chat template if supported
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        prompt = question

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[0][len(inputs['input_ids'][0]):]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response

print("Loading dataset...") 
gsm8k = load_dataset("gsm8k", split="test")  # can change to dataset needed

results_dir = "gsm8k_llm_results"
os.makedirs(results_dir, exist_ok=True)
start_time = time.time()

for idx, problem in enumerate(tqdm(gsm8k, desc="Running evaluation")):
    question = problem.get("question", "N/A")
    ground_truth = problem.get("answer", "N/A")

    try:
        llm_answer = llm_generate_local(question)
    except Exception as e:
        llm_answer = f"[ERROR] {e}"
        print(f"Error processing problem {idx}: {e}")

    file_path = os.path.join(results_dir, f"Problem{idx}.txt")
    with open(file_path, 'w') as f:
        f.write(f"Problem {idx}\n")
        f.write(f"Question:\n{question}\n")
        f.write("-" * 50 + "\n")
        f.write(f"LLM Answer:\n{llm_answer}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Ground Truth:\n{ground_truth}\n")
        f.write("*" * 50 + "\n")
import os
import time
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# -------------------------------
# LLM Setup
# -------------------------------
model_name = "Qwen/Qwen3-4B-Instruct-2507"  # change to your model

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -------------------------------
# Generation function
# -------------------------------
def llm_generate_local(question, max_new_tokens=1024):
    # Apply chat template if supported
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        prompt = question

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[0][len(inputs['input_ids'][0]):]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response

# -------------------------------
# Load GSM8K dataset
# -------------------------------
print("Loading GSM8K dataset...")
gsm8k = load_dataset("gsm8k", split="train")  # use 'train' split; can also use 'test'

# -------------------------------
# Setup output directory
# -------------------------------
results_dir = "gsm8k_llm_results"
os.makedirs(results_dir, exist_ok=True)
start_time = time.time()

# -------------------------------
# Run evaluation on all problems
# -------------------------------
for idx, problem in enumerate(tqdm(gsm8k, desc="Running GSM8K evaluation")):
    question = problem.get("question", "N/A")
    ground_truth = problem.get("answer", "N/A")

    try:
        llm_answer = llm_generate_local(question)
    except Exception as e:
        llm_answer = f"[ERROR] {e}"
        print(f"Error processing problem {idx}: {e}")

    file_path = os.path.join(results_dir, f"Problem{idx}.txt")
    with open(file_path, 'w') as f:
        f.write(f"Problem {idx}\n")
        f.write(f"Question:\n{question}\n")
        f.write("-" * 50 + "\n")
        f.write(f"LLM Answer:\n{llm_answer}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Ground Truth:\n{ground_truth}\n")
        f.write("*" * 50 + "\n")

    time.sleep(0.05)


elapsed = time.time() - start_time
print(f"Completed in {elapsed:.2f} seconds. Results saved in '{results_dir}/' directory.")
