#!/usr/bin/env python3
"""
Higgs-Llama-3-70B AWQ Evaluation Script with Baseline Comparison

Evaluates and compares:
- Baseline: bosonai/Higgs-Llama-3-70B (FP16, ~140GB)
- Quantized: ronantakizawa/higgs-llama-3-70b-awq (AWQ 4-bit, ~37GB)

Metrics:
1. Generation quality tests (various tasks)
2. Perplexity measurement
3. Performance benchmarks (latency, throughput)
4. Side-by-side comparison
"""

import os
import gc
import time
import math
import json
import torch
from awq import AutoAWQForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Configuration
BASELINE_MODEL_ID = "bosonai/Higgs-Llama-3-70B"
QUANTIZED_MODEL_ID = "ronantakizawa/higgs-llama-3-70b-awq"
RESULTS_FILE = "evaluation_comparison.json"
COMPARE_BASELINE = True  # Set to False to only evaluate quantized model

print("="*70)
print("🧪 Higgs-Llama-3-70B: Baseline vs AWQ Comparison")
print("="*70)
print(f"\n📦 Baseline: {BASELINE_MODEL_ID}")
print(f"📦 Quantized: {QUANTIZED_MODEL_ID}\n")

# Function to load model
def load_model(model_id, is_quantized=False):
    """Load model and tokenizer"""
    print(f"\n⏳ Loading {model_id}...")
    start_time = time.time()

    if is_quantized:
        model = AutoAWQForCausalLM.from_quantized(
            model_id,
            fuse_layers=True,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    elapsed = time.time() - start_time
    mem_allocated = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0

    print(f"✅ Loaded in {elapsed:.2f}s | GPU Memory: {mem_allocated:.2f} GB")

    return model, tokenizer

# Note: We'll load models one at a time to avoid memory issues
# Models will be loaded in the evaluation loop below

# Function to run generation tests
def run_generation_tests(model, tokenizer, model_name):
    """Run generation quality tests on a model"""
    print(f"\n{'='*70}")
    print(f"📋 Generation Quality Tests: {model_name}")
    print('='*70)

    device = next(model.parameters()).device

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

    overall_accuracy = total_correct / total_tests if total_tests > 0 else 0
    avg_latency = total_time / total_tests if total_tests > 0 else 0

    return {
        'results': results,
        'overall_accuracy': overall_accuracy,
        'total_correct': total_correct,
        'total_tests': total_tests,
        'avg_latency': avg_latency,
        'total_time': total_time
    }

# Function to calculate perplexity
def calculate_perplexity(model, tokenizer, texts, model_name, max_samples=100):
    """Calculate perplexity on a set of texts"""
    print(f"\n{'='*70}")
    print(f"📐 Perplexity Measurement: {model_name}")
    print('='*70 + "\n")

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

    print(f"\n✅ Perplexity: {perplexity:.4f}")
    print(f"   • Average Loss: {avg_loss:.4f}")
    print(f"   • Samples: {samples_used}")
    print(f"   • Tokens: {total_tokens:,}")

    return {
        'perplexity': perplexity,
        'avg_loss': avg_loss,
        'samples': samples_used,
        'tokens': total_tokens
    }

# Function to run performance benchmarks
def run_performance_benchmarks(model, tokenizer, model_name):
    """Run performance benchmarks on a model"""
    print(f"\n{'='*70}")
    print(f"⚡ Performance Benchmarks: {model_name}")
    print('='*70 + "\n")

    device = next(model.parameters()).device
    test_prompt = "Write a detailed explanation of quantum computing and its applications in modern technology."
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)

    # Warmup
    print("Warming up...")
    model.generate(**inputs, max_new_tokens=50)

    # Actual benchmark
    print("\nTesting throughput (tokens/second)...")
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

    return {
        'throughput_results': throughput_results,
        'avg_throughput': avg_throughput
    }

# Run evaluations on all models
print("\n" + "="*70)
print("🚀 RUNNING EVALUATIONS")
print("="*70)

all_results = {}

# Load WikiText-2 for perplexity (shared across models)
print("\nLoading WikiText-2 dataset for perplexity testing...")
wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
wikitext_texts = [sample['text'] for sample in wikitext if len(sample['text'].strip()) > 100]
print(f"✅ Loaded {len(wikitext_texts)} test samples")

# Evaluate models sequentially (one at a time to avoid memory issues)
models_to_eval = []
if COMPARE_BASELINE:
    models_to_eval.append(('baseline', BASELINE_MODEL_ID, False, 'Baseline (FP16)'))
models_to_eval.append(('quantized', QUANTIZED_MODEL_ID, True, 'Quantized (AWQ 4-bit)'))

for model_key, model_id, is_quantized, model_name in models_to_eval:
    print(f"\n{'='*70}")
    print(f"📥 LOADING & EVALUATING: {model_name}")
    print('='*70)

    # Load model
    model, tokenizer = load_model(model_id, is_quantized=is_quantized)

    # Capture memory AFTER loading (before running tests)
    memory_gb = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    print(f"📊 GPU Memory after loading: {memory_gb:.2f} GB")

    # Run tests
    gen_results = run_generation_tests(model, tokenizer, model_name)
    ppl_results = calculate_perplexity(model, tokenizer, wikitext_texts, model_name, max_samples=100)
    perf_results = run_performance_benchmarks(model, tokenizer, model_name)

    all_results[model_key] = {
        'name': model_name,
        'memory_gb': memory_gb,  # Use captured memory from after loading
        'generation': gen_results,
        'perplexity': ppl_results,
        'performance': perf_results
    }

    # Clear GPU cache before loading next model
    print(f"\n🧹 Clearing GPU memory...")
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    freed_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    print(f"   GPU Memory after cleanup: {freed_memory:.2f} GB")

# Comparison Summary
print(f"\n{'='*70}")
print("📊 COMPARISON SUMMARY")
print('='*70)

if COMPARE_BASELINE and 'baseline' in all_results and 'quantized' in all_results:
    baseline = all_results['baseline']
    quantized = all_results['quantized']

    print(f"\n🎯 Generation Quality:")
    print(f"   Baseline:   {baseline['generation']['overall_accuracy']*100:.1f}% accuracy, {baseline['generation']['avg_latency']:.2f}s avg latency")
    print(f"   Quantized:  {quantized['generation']['overall_accuracy']*100:.1f}% accuracy, {quantized['generation']['avg_latency']:.2f}s avg latency")
    acc_delta = (quantized['generation']['overall_accuracy'] - baseline['generation']['overall_accuracy']) * 100
    print(f"   → Accuracy change: {acc_delta:+.1f}%")

    print(f"\n📐 Perplexity:")
    print(f"   Baseline:   {baseline['perplexity']['perplexity']:.4f}")
    print(f"   Quantized:  {quantized['perplexity']['perplexity']:.4f}")
    ppl_increase = quantized['perplexity']['perplexity'] - baseline['perplexity']['perplexity']
    ppl_percent = (ppl_increase / baseline['perplexity']['perplexity']) * 100
    print(f"   → Perplexity increase: {ppl_increase:+.4f} ({ppl_percent:+.1f}%)")

    print(f"\n⚡ Performance:")
    print(f"   Baseline:   {baseline['performance']['avg_throughput']:.2f} tokens/sec")
    print(f"   Quantized:  {quantized['performance']['avg_throughput']:.2f} tokens/sec")
    speedup = quantized['performance']['avg_throughput'] / baseline['performance']['avg_throughput']
    print(f"   → Speedup: {speedup:.2f}x")

    print(f"\n💾 Memory:")
    print(f"   Baseline:   {baseline['memory_gb']:.2f} GB")
    print(f"   Quantized:  {quantized['memory_gb']:.2f} GB")
    mem_saved = baseline['memory_gb'] - quantized['memory_gb']
    mem_percent = (mem_saved / baseline['memory_gb']) * 100
    print(f"   → Memory saved: {mem_saved:.2f} GB ({mem_percent:.1f}%)")

    print(f"\n✨ Summary:")
    print(f"   • Quality degradation: {acc_delta:+.1f}% accuracy, {ppl_percent:+.1f}% perplexity")
    print(f"   • Performance gain: {speedup:.2f}x faster")
    print(f"   • Memory reduction: {mem_percent:.1f}%")
else:
    quantized = all_results['quantized']
    print(f"\n🎯 Generation Quality: {quantized['generation']['overall_accuracy']*100:.1f}%")
    print(f"📐 Perplexity: {quantized['perplexity']['perplexity']:.4f}")
    print(f"⚡ Throughput: {quantized['performance']['avg_throughput']:.2f} tokens/sec")
    print(f"💾 Memory: {quantized['memory_gb']:.2f} GB")

# Save results
results_data = {
    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
    'models': all_results,
    'comparison': {
        'baseline_model': BASELINE_MODEL_ID if COMPARE_BASELINE else None,
        'quantized_model': QUANTIZED_MODEL_ID
    }
}

with open(RESULTS_FILE, "w") as f:
    json.dump(results_data, f, indent=2)

print(f"\n💾 Results saved to: {RESULTS_FILE}")
print("\n" + "="*70)
print("✅ EVALUATION COMPLETE!")
print("="*70)
