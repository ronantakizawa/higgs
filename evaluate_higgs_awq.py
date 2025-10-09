#!/usr/bin/env python3
"""
Higgs-Llama-3-70B AWQ Evaluation Script

Evaluates the quantized model on:
1. Generation quality tests (various tasks)
2. Perplexity measurement
3. Performance benchmarks (latency, throughput)
"""

import os
import gc
import time
import math
import json
import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset

# Configuration
MODEL_ID = "ronantakizawa/higgs-llama-3-70b-awq"
RESULTS_FILE = "evaluation_results.json"

print("="*70)
print("🧪 Higgs-Llama-3-70B AWQ Evaluation")
print("="*70)
print(f"\n📦 Model: {MODEL_ID}\n")

# Load Model
print("="*70)
print("⏳ Loading Quantized Model")
print("="*70)

start_time = time.time()

model = AutoAWQForCausalLM.from_quantized(
    MODEL_ID,
    fuse_layers=True,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

elapsed = time.time() - start_time
print(f"\n✅ Model loaded in {elapsed:.2f} seconds")

if torch.cuda.is_available():
    mem_allocated = torch.cuda.memory_allocated() / 1024**3
    print(f"   GPU Memory: {mem_allocated:.2f} GB")

device = next(model.parameters()).device

# Part 1: Generation Quality Tests
print("\n" + "="*70)
print("📋 PART 1: Generation Quality Tests")
print("="*70)

test_suite = {
    "general_knowledge": [
        {
            "prompt": "Explain the theory of relativity in simple terms.",
            "keywords": ["einstein", "relativity", "space", "time", "gravity"]
        },
        {
            "prompt": "What are the main differences between RNA and DNA?",
            "keywords": ["rna", "dna", "nucleotide", "uracil", "thymine"]
        },
        {
            "prompt": "Describe how photosynthesis works.",
            "keywords": ["photosynthesis", "light", "carbon", "oxygen", "chlorophyll"]
        }
    ],
    "reasoning": [
        {
            "prompt": "If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets?",
            "keywords": ["5", "minutes", "same"]
        },
        {
            "prompt": "Explain the trolley problem and its ethical implications.",
            "keywords": ["trolley", "ethical", "dilemma", "utilitarian", "choice"]
        },
        {
            "prompt": "A bat and a ball cost $1.10 in total. The bat costs $1 more than the ball. How much does the ball cost?",
            "keywords": ["5", "cents", "0.05", "nickel"]
        }
    ],
    "code_generation": [
        {
            "prompt": "Write a Python function to find the longest common subsequence of two strings.",
            "keywords": ["def", "subsequence", "return", "dynamic", "programming"]
        },
        {
            "prompt": "Implement a binary search algorithm in Python.",
            "keywords": ["def", "binary", "search", "while", "mid"]
        }
    ],
    "creative_writing": [
        {
            "prompt": "Write a haiku about artificial intelligence.",
            "keywords": ["silicon", "mind", "algorithm", "digital", "learn", "machine", "think"]
        },
        {
            "prompt": "Complete this story: Once upon a time in a distant galaxy...",
            "keywords": ["space", "planet", "star", "ship", "alien", "adventure"]
        }
    ],
    "mathematics": [
        {
            "prompt": "Solve: If x + 2 = 5, what is x?",
            "keywords": ["3", "three", "x = 3", "x=3"]
        },
        {
            "prompt": "What is the derivative of x^2?",
            "keywords": ["2x", "2 * x", "twice"]
        }
    ]
}

results = {}
total_correct = 0
total_tests = 0
total_time = 0

for category, tests in test_suite.items():
    print(f"\n{'='*70}")
    print(f"📂 Category: {category.upper().replace('_', ' ')}")
    print('='*70)

    category_results = []
    category_correct = 0

    for i, test in enumerate(tests, 1):
        prompt = test["prompt"]
        keywords = [kw.lower() for kw in test["keywords"]]

        print(f"\n{i}. 📝 Prompt: {prompt}")

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        start_time = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
        )
        generation_time = time.time() - start_time

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        result_lower = result.lower()

        # Check keywords
        found_keywords = [kw for kw in keywords if kw in result_lower]
        keyword_score = len(found_keywords) / len(keywords)
        is_correct = keyword_score >= 0.4  # 40% threshold

        print(f"   ✅ Output: {result[:200]}{'...' if len(result) > 200 else ''}")
        print(f"   🎯 Keywords found: {len(found_keywords)}/{len(keywords)} ({keyword_score*100:.0f}%)")
        print(f"   {'✓' if is_correct else '✗'} {'PASS' if is_correct else 'FAIL'}")
        print(f"   ⏱️  Time: {generation_time:.2f}s")

        category_results.append({
            "prompt": prompt,
            "output": result,
            "keywords_found": found_keywords,
            "keyword_score": keyword_score,
            "pass": is_correct,
            "time": generation_time
        })

        if is_correct:
            category_correct += 1
        total_correct += 1 if is_correct else 0
        total_tests += 1
        total_time += generation_time

    results[category] = {
        "tests": category_results,
        "accuracy": category_correct / len(tests),
        "avg_time": sum(t["time"] for t in category_results) / len(tests)
    }

    print(f"\n{'─'*70}")
    print(f"📊 {category.upper().replace('_', ' ')} Summary:")
    print(f"   Accuracy: {category_correct}/{len(tests)} ({results[category]['accuracy']*100:.0f}%)")
    print(f"   Avg Time: {results[category]['avg_time']:.2f}s")

# Part 2: Perplexity Measurement
print(f"\n{'='*70}")
print("📐 PART 2: Perplexity Measurement")
print('='*70 + "\n")

def calculate_perplexity(model, tokenizer, texts, max_samples=100):
    """Calculate perplexity on a set of texts"""
    device = next(model.parameters()).device
    model.eval()

    total_loss = 0
    total_tokens = 0
    samples_used = 0

    print(f"⏳ Calculating perplexity on {min(len(texts), max_samples)} samples...")

    with torch.no_grad():
        for i, text in enumerate(texts[:max_samples]):
            # Tokenize
            encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            input_ids = encodings.input_ids.to(device)

            # Skip very short sequences
            if input_ids.shape[1] < 2:
                continue

            # Calculate loss
            try:
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss

                # Accumulate
                total_loss += loss.item() * input_ids.shape[1]
                total_tokens += input_ids.shape[1]
                samples_used += 1

                if (i + 1) % 25 == 0:
                    print(f"   Processed {i+1}/{min(len(texts), max_samples)} samples...")
            except Exception as e:
                print(f"   Skipping sample {i+1} due to error: {e}")
                continue

    # Calculate perplexity
    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
    else:
        avg_loss = float('inf')
        perplexity = float('inf')

    return perplexity, avg_loss, samples_used, total_tokens

# Load WikiText-2 for perplexity
print("Loading WikiText-2 dataset...")
wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
wikitext_texts = [sample['text'] for sample in wikitext if len(sample['text'].strip()) > 100]

perplexity, avg_loss, samples_used, total_tokens = calculate_perplexity(
    model,
    tokenizer,
    wikitext_texts,
    max_samples=100
)

print(f"\n✅ Perplexity Calculation Complete:")
print(f"   • Perplexity: {perplexity:.4f}")
print(f"   • Average Loss: {avg_loss:.4f}")
print(f"   • Samples: {samples_used}")
print(f"   • Tokens: {total_tokens:,}")
print(f"\n   Interpretation:")
if perplexity < 10:
    print(f"   🌟 EXCELLENT - Very low perplexity (< 10)")
elif perplexity < 20:
    print(f"   ✅ GOOD - Low perplexity (10-20)")
elif perplexity < 40:
    print(f"   👍 ACCEPTABLE - Moderate perplexity (20-40)")
else:
    print(f"   ⚠️  HIGH - Consider re-quantization (> 40)")

# Part 3: Performance Benchmarks
print(f"\n{'='*70}")
print("⚡ PART 3: Performance Benchmarks")
print('='*70 + "\n")

# Throughput test
print("Testing throughput (tokens/second)...")
test_prompt = "Write a detailed explanation of quantum computing and its applications in modern technology."
inputs = tokenizer(test_prompt, return_tensors="pt").to(device)

# Warmup
model.generate(**inputs, max_new_tokens=50)

# Actual benchmark
num_tokens_list = [50, 100, 200]
throughput_results = []

for num_tokens in num_tokens_list:
    start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=num_tokens,
        do_sample=False
    )
    elapsed = time.time() - start

    tokens_generated = outputs.shape[1] - inputs.input_ids.shape[1]
    throughput = tokens_generated / elapsed

    throughput_results.append({
        "tokens": num_tokens,
        "time": elapsed,
        "throughput": throughput
    })

    print(f"   • {num_tokens} tokens: {throughput:.2f} tokens/sec ({elapsed:.2f}s total)")

avg_throughput = sum(r["throughput"] for r in throughput_results) / len(throughput_results)
print(f"\n   Average throughput: {avg_throughput:.2f} tokens/sec")

# Part 4: Final Summary
print(f"\n{'='*70}")
print("📊 EVALUATION SUMMARY")
print('='*70 + "\n")

overall_accuracy = total_correct / total_tests
avg_latency = total_time / total_tests

print(f"🎯 Generation Tests:")
print(f"   • Overall Accuracy: {total_correct}/{total_tests} ({overall_accuracy*100:.0f}%)")
print(f"   • Average Latency: {avg_latency:.2f}s")
print(f"\n📈 Per-Category Results:")
for category, result in results.items():
    print(f"   • {category.replace('_', ' ').title()}: {result['accuracy']*100:.0f}% accuracy, {result['avg_time']:.2f}s avg")

print(f"\n📐 Perplexity:")
print(f"   • Score: {perplexity:.4f}")
print(f"   • Quality: {'EXCELLENT' if perplexity < 10 else 'GOOD' if perplexity < 20 else 'ACCEPTABLE' if perplexity < 40 else 'HIGH'}")

print(f"\n⚡ Performance:")
print(f"   • Average Throughput: {avg_throughput:.2f} tokens/sec")
print(f"   • Average Latency: {avg_latency:.2f}s per generation")

if torch.cuda.is_available():
    mem_allocated = torch.cuda.memory_allocated() / 1024**3
    print(f"\n💾 GPU Memory:")
    print(f"   • Allocated: {mem_allocated:.2f} GB")
    print(f"   • Original Model: ~140 GB")
    print(f"   • Memory Saved: ~{140 - mem_allocated:.1f} GB ({((140 - mem_allocated)/140*100):.1f}%)")

# Save results to JSON
evaluation_results = {
    "model": MODEL_ID,
    "quantization": "AWQ 4-bit",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "generation_tests": {
        "overall_accuracy": overall_accuracy,
        "total_correct": total_correct,
        "total_tests": total_tests,
        "avg_latency": avg_latency,
        "by_category": {
            cat: {
                "accuracy": res["accuracy"],
                "avg_time": res["avg_time"],
                "tests": len(res["tests"])
            } for cat, res in results.items()
        }
    },
    "perplexity": {
        "score": perplexity,
        "avg_loss": avg_loss,
        "samples": samples_used,
        "tokens": total_tokens
    },
    "performance": {
        "avg_throughput_tokens_per_sec": avg_throughput,
        "avg_latency_sec": avg_latency,
        "throughput_by_length": throughput_results
    },
    "gpu_memory_gb": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else None
}

with open(RESULTS_FILE, "w") as f:
    json.dump(evaluation_results, f, indent=2)

print(f"\n💾 Results saved to: {RESULTS_FILE}")
print("\n" + "="*70)
print("✅ EVALUATION COMPLETE!")
print("="*70)
