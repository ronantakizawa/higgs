#!/usr/bin/env python3
"""
Higgs-Llama-3-70B AWQ 4-bit Quantization Script

Simple script to quantize, save, and upload the model.
Assumes model is already downloaded to HuggingFace cache.
"""

import os
import gc
import time
import torch

# Set memory optimization before importing anything else
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset
from huggingface_hub import HfApi, create_repo, login

# Configuration
MODEL_PATH = "bosonai/Higgs-Llama-3-70B"
QUANT_PATH = "/workspace/higgs-llama-3-70b-awq"
HF_MODEL_ID = "ronantakizawa/higgs-llama-3-70b-awq"

QUANT_CONFIG = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

print("="*70)
print("🚀 Higgs-Llama-3-70B AWQ Quantization")
print("="*70)
print(f"\n📦 Model: {MODEL_PATH}")
print(f"💾 Output: {QUANT_PATH}")
print(f"🚀 Upload to: {HF_MODEL_ID}\n")

# Step 1: Load Model
print("="*70)
print("⏳ STEP 1: Loading Model")
print("="*70)

start_time = time.time()

# Clear GPU memory before loading
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

model = AutoAWQForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map={"": 0},  # Force ALL layers on GPU 0 (no offloading)
    torch_dtype=torch.float16,  # Force FP16 to materialize tensors properly
    low_cpu_mem_usage=True,
    use_cache=False,
    cache_dir="/workspace/.cache/huggingface"
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    cache_dir="/workspace/.cache/huggingface"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

elapsed = time.time() - start_time
print(f"\n✅ Model loaded in {elapsed/60:.1f} minutes")
print(f"   GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# Step 2: Prepare Calibration Data
print("\n" + "="*70)
print("📚 STEP 2: Preparing Calibration Data")
print("="*70)

dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)

calibration_data = []
target_samples = 512
min_length = 200
max_length = 1000

print(f"Collecting {target_samples} samples from C4 dataset...")

for sample in dataset:
    text = sample.get('text', '').strip()
    if min_length <= len(text) <= max_length:
        calibration_data.append(text)
    if len(calibration_data) >= target_samples:
        break

print(f"✅ Prepared {len(calibration_data)} calibration samples")

# Step 3: Run Quantization
print("\n" + "="*70)
print("🔧 STEP 3: Running AWQ Quantization")
print("="*70)
print("⏳ This will take 1-2 hours for 70B model...\n")

start_time = time.time()

model.quantize(
    tokenizer,
    quant_config=QUANT_CONFIG,
    calib_data=calibration_data
)

elapsed = time.time() - start_time
print(f"\n✅ Quantization completed in {elapsed/60:.1f} minutes!")

# Step 4: Save Model
print("\n" + "="*70)
print("💾 STEP 4: Saving Quantized Model")
print("="*70)

model.save_quantized(QUANT_PATH)
tokenizer.save_pretrained(QUANT_PATH)

print(f"✅ Model saved to {QUANT_PATH}")

# Check size
def get_dir_size(path):
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / (1024**3)

quantized_size = get_dir_size(QUANT_PATH)
print(f"\n📊 Quantized model size: {quantized_size:.2f} GB")
print(f"   Original: ~140 GB")
print(f"   Reduction: ~{((140 - quantized_size) / 140 * 100):.1f}%")

# Clear GPU memory
del model
torch.cuda.empty_cache()
gc.collect()

# Step 5: Upload to HuggingFace
print("\n" + "="*70)
print("🚀 STEP 5: Uploading to HuggingFace")
print("="*70)

# Login (will use cached token or prompt)
login()
print("✅ Logged in to HuggingFace")

# Create model card
model_card = f"""---
language:
- en
license: llama3
tags:
- awq
- quantized
- 4-bit
- llama-3
- bosonai
base_model: bosonai/Higgs-Llama-3-70B
---

# Higgs-Llama-3-70B AWQ 4-bit Quantized

This is a 4-bit AWQ quantized version of [bosonai/Higgs-Llama-3-70B](https://huggingface.co/bosonai/Higgs-Llama-3-70B).

## Model Details

- **Base Model:** bosonai/Higgs-Llama-3-70B (70B parameters)
- **Quantization Method:** AWQ (Activation-aware Weight Quantization)
- **Quantization Precision:** 4-bit
- **Group Size:** 128
- **Original Size:** ~140 GB (FP16)
- **Quantized Size:** ~{quantized_size:.1f} GB
- **Memory Reduction:** ~{((140 - quantized_size) / 140 * 100):.1f}%

## Usage

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model = AutoAWQForCausalLM.from_quantized(
    "{HF_MODEL_ID}",
    fuse_layers=True,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("{HF_MODEL_ID}")

prompt = "Explain quantum computing in simple terms."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Requirements

- GPU Memory: ~40-45 GB VRAM
- CUDA required for AWQ
- Python 3.8+

```bash
pip install autoawq transformers accelerate
```

## License

Llama 3 License
"""

readme_path = os.path.join(QUANT_PATH, "README.md")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(model_card)

print(f"✅ Model card created")

# Create repository and upload
print(f"\n🚀 Uploading to {HF_MODEL_ID}...")

create_repo(HF_MODEL_ID, repo_type="model", exist_ok=True)
print(f"✅ Repository ready")

api = HfApi()
api.upload_folder(
    folder_path=QUANT_PATH,
    repo_id=HF_MODEL_ID,
    repo_type="model",
    commit_message="Upload AWQ 4-bit quantized Higgs-Llama-3-70B"
)

print(f"\n✅ Upload complete!")
print(f"   View at: https://huggingface.co/{HF_MODEL_ID}")

print("\n" + "="*70)
print("🎉 ALL DONE!")
print("="*70)
print(f"Model: {HF_MODEL_ID}")
print(f"Size: {quantized_size:.2f} GB")
print(f"Reduction: ~{((140 - quantized_size) / 140 * 100):.1f}%")
print("\nDon't forget to delete the GPU instance to stop billing!")
