import os
import time
import re
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen3-4B-Instruct-2507"
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"  # uses CUDA if available
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def llm_generate_local(prompt, max_new_tokens=1024):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.3,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

class Node:
    def __init__(self, question, answer, depth=0):
        self.question = question
        self.answer = answer
        self.depth = depth
        self.critique = self.generate_critic()
        self.reward = self.compute_reward()
        self.f = None  # f(n) = g(n) + h(n)

    def generate_critic(self):
        prompt = (
            f"Question:\n{self.question}\n\n"
            f"Answer:\n{self.answer}\n\n"
            "Provide constructive criticism and assign a grade out of 100 "
            "in the format 'Grade: xx'."
        )
        return llm_generate_local(prompt)

    def parse_score(self, critique_text):
        match = re.search(r'Grade:\s*(\d{1,3})', critique_text)
        return max(0, min(int(match.group(1)), 100)) if match else 50

    def compute_reward(self, num_evals=3):
        # Self-consistency: multiple independent evaluations
        total = 0
        for _ in range(num_evals):
            critique = self.generate_critic()
            total += self.parse_score(critique)
        return total / num_evals

    def compute_f(self, w=3.0):
        g = w * self.depth
        h = 100 - self.reward
        self.f = g + h
        return self.f


class AStarLLM:
    def __init__(self, question, max_iter=8, w=1.0):
        self.question = question
        self.max_iter = max_iter
        self.w = w
        self.nodes_to_visit = []

    def search(self):
        # Initialize root
        root_answer = llm_generate_local(f"Solve step by step:\n{self.question}")
        root = Node(self.question, root_answer, depth=0)
        root.compute_f(self.w)
        self.nodes_to_visit.append(root)
        best_node = root

        for _ in range(self.max_iter):
            if not self.nodes_to_visit:
                break

            # Select node with minimum f(n)
            current_node = min(self.nodes_to_visit, key=lambda n: n.f)
            self.nodes_to_visit.remove(current_node)

            if current_node.reward >= 95:
                return current_node, current_node.reward

            # Generate children
            for _ in range(2):
                child_prompt = (
                    f"Question:\n{self.question}\n\n"
                    f"Previous Answer:\n{current_node.answer}\n\n"
                    f"Critique:\n{current_node.critique}\n\n"
                    "Using the feedback above, attempt the problem again step by step."
                )
                child_answer = llm_generate_local(child_prompt, temperature=0.7)
                child_node = Node(self.question, child_answer, depth=current_node.depth + 1)
                child_node.compute_f(self.w)
                self.nodes_to_visit.append(child_node)

                if child_node.reward > best_node.reward:
                    best_node = child_node

        return best_node, best_node.reward

# edit based on dataset
dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
results_dir = "math500_astar_results"
os.makedirs(results_dir, exist_ok=True)

start_time = time.time()
for idx, problem in enumerate(tqdm(dataset, desc="Running A*")):
    question = problem["problem"] # edit based on dataset
    ground_truth = problem["solution"] # edit based on dataset

    astar = AStarLLM(question, max_iter=8)
    final_node, score = astar.search()

    print(f"\nProblem {idx + 1}")
    print("Answer:\n", final_node.get_answer())
    print("Critique Score:", score)
    print("-" * 50)

    # Save to file
    file_path = os.path.join(results_dir, f"Problem{idx + 1}.txt")
    with open(file_path, 'w') as f:
        f.write(f"Problem {idx + 1}\n")
        f.write(f"Question:\n{question}\n")
        f.write("-" * 50 + "\n")
        f.write(f"LLM Answer:\n{final_node.get_answer()}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Score:\n{score}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Ground Truth:\n{ground_truth}\n")
        f.write("*" * 50 + "\n")

elapsed = time.time() - start_time
print(f"\nCompleted in {elapsed:.2f} seconds. Results saved in '{results_dir}/'")
