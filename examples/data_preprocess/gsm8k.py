import os
import argparse
import random
from datasets import load_dataset
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# The Prompt Bank for Exploration
# These represent different 'reasoning modes' the RL algorithm will explore.
PROMPT_MODES = {
    "standard": (
        "A math problem is given below. Solve it step by step. "
        "Return the final answer inside \boxed{}."
    ),
    "decompose": (
        "A math problem is given below. First, decompose the problem into smaller "
        "sub-problems. Solve each sub-problem step by step. "
        "Return the final answer inside \boxed{}."
    ),
    "self_verify": (
        "A math problem is given below. Solve it step by step, and then explicitly "
        "double-check your arithmetic and logic before finalizing the result. "
        "Return the final answer inside \boxed{}."
    ),
    "alternate": (
        "A math problem is given below. Solve it using two completely different "
        "methods to verify your logic. If the methods yield different results, "
        "determine the error. Return the final answer inside \boxed{}."
    )
}

def extract_answer(answer_str: str) -> str:
    """Extracts the final answer from GSM8K's answer string (comes after ####)."""
    if "####" in answer_str:
        return answer_str.split("####")[1].strip()
    return answer_str.strip()

def process_dataset(split: str, template_type: str = "uniform", num_samples: int = None):
    """
    Processes the GSM8K dataset and assigns prompt families.
    
    Args:
        split: 'train' or 'test'
        template_type: 'standard' (only uses standard prompt) or 'uniform' (mixes all prompts)
        num_samples: Optional limit on the number of samples.
    """
    print(f"Loading GSM8K {split} split...")
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    
    if num_samples is not None and num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))
        
    data_rows = []
    
    for row in dataset:
        question = row["question"]
        raw_answer = row["answer"]
        ground_truth = extract_answer(raw_answer)
        
        # Decide which prompt family to assign
        if template_type == "uniform":
            prompt_family = random.choice(list(PROMPT_MODES.keys()))
        else:
            prompt_family = "standard"
            
        system_instruction = PROMPT_MODES[prompt_family]
        
        # Format the full prompt for the model
        # You may need to wrap this in a Chat template (e.g. <|im_start|>user...<|im_end|>)
        # depending on your base model (like Qwen2.5)
        # Here we provide a simple generic user/assistant structure.
        full_prompt = f"<|im_start|>system\nYou are a helpful math assistant.<|im_end|>\n<|im_start|>user\n{system_instruction}\n\nProblem:\n{question}<|im_end|>\n<|im_start|>assistant\n"
        
        # verl usually expects specific columns, common ones are:
        # 'prompt' (the text input), 'ground_truth' (for the reward function), 'prompt_family' (for tracking)
        data_rows.append({
            "prompt": full_prompt,
            "ground_truth": ground_truth,
            "prompt_family": prompt_family,
            "question": question,
        })
        
    df = pd.DataFrame(data_rows)
    return df

def main():
    parser = argparse.ArgumentParser(description="Preprocess GSM8K for AdaPrompt-GRPO.")
    parser.add_argument("--local_dir", type=str, default="data/gsm8k", help="Output directory")
    parser.add_argument("--template_type", type=str, choices=["standard", "uniform"], default="uniform", 
                        help="Use 'standard' for baseline, 'uniform' for prompt mixing.")
    parser.add_argument("--train_size", type=int, default=None, help="Limit training size")
    parser.add_argument("--test_size", type=int, default=None, help="Limit test size")
    
    args = parser.parse_args()
    
    os.makedirs(args.local_dir, exist_ok=True)
    
    # Process Train
    train_df = process_dataset("train", args.template_type, args.train_size)
    train_table = pa.Table.from_pandas(train_df)
    train_path = os.path.join(args.local_dir, "train.parquet")
    pq.write_table(train_table, train_path)
    print(f"Saved {len(train_df)} training samples to {train_path}")
    
    # Process Test
    test_df = process_dataset("test", args.template_type, args.test_size)
    test_table = pa.Table.from_pandas(test_df)
    test_path = os.path.join(args.local_dir, "test.parquet")
    pq.write_table(test_table, test_path)
    print(f"Saved {len(test_df)} testing samples to {test_path}")

if __name__ == "__main__":
    main()
