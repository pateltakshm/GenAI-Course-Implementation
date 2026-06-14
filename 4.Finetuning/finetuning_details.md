# Fine-Tuning Reference: Unsloth & HuggingFace

---

## 1. Core Concepts Before You Start

### What is Fine-Tuning?
- Adapting a pre-trained model on a smaller, task-specific dataset
- Two main approaches: **Full Fine-Tuning** (all weights updated) vs **PEFT** (Parameter-Efficient Fine-Tuning)
- Full fine-tuning is expensive; PEFT methods like **LoRA** and **QLoRA** are standard practice

### LoRA (Low-Rank Adaptation)
- Instead of updating all weights, inserts small trainable matrices (adapters) into attention layers
- Parameters: `r` (rank), `lora_alpha` (scaling), `lora_dropout`, `target_modules`
- Rule of thumb: `lora_alpha = 2 * r` for stable training
- Lower `r` = fewer parameters, faster, less expressive. Higher `r` = more capacity, slower

### QLoRA (Quantized LoRA)
- Loads model in 4-bit or 8-bit quantization, then applies LoRA adapters on top
- Dramatically reduces GPU VRAM — enables fine-tuning 70B models on a single GPU
- Uses `bitsandbytes` library for quantization

### Instruction Tuning vs Chat Tuning
- **Instruction tuning**: supervised learning on (instruction, response) pairs — best for task-specific models
- **Chat tuning**: uses multi-turn conversation templates (system/user/assistant roles)
- Always match the **chat template** of the base model when fine-tuning a chat model

---

## 2. Dataset Preparation

### Format
- Common formats: JSONL, CSV, Parquet, HuggingFace Dataset
- Each sample needs an `instruction`, `input` (optional), and `output` field — or a `text` field with the full prompt already formatted

### Alpaca Format (most common)
```json
{
  "instruction": "Summarize the following text.",
  "input": "The quick brown fox...",
  "output": "A fox jumps over a dog."
}
```

### ShareGPT / Chat Format
```json
{
  "conversations": [
    {"from": "human", "value": "What is 2+2?"},
    {"from": "gpt",   "value": "4"}
  ]
}
```

### Key Checks
- Remove duplicates and near-duplicates
- Ensure consistent formatting — one wrong template ruins a batch
- Aim for **quality over quantity**: 1K high-quality samples often beats 100K noisy ones
- Split into train / eval (90/10 or 95/5)

---

## 3. HuggingFace Fine-Tuning

### Key Libraries
| Library | Purpose |
|---|---|
| `transformers` | Model loading, tokenization, training |
| `datasets` | Dataset loading and preprocessing |
| `peft` | LoRA / QLoRA adapters |
| `trl` | SFTTrainer, DPOTrainer, reward modeling |
| `bitsandbytes` | 4-bit / 8-bit quantization |
| `accelerate` | Multi-GPU / distributed training |
| `evaluate` | Metrics (ROUGE, BLEU, etc.) |

### Loading a Model with 4-bit Quantization
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4 — best for LLMs
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,      # nested quantization saves more VRAM
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tokenizer.pad_token = tokenizer.eos_token   # required for batched training
```

### Applying LoRA with PEFT
```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=16,                            # rank
    lora_alpha=32,                   # scaling factor
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()   # sanity check — should be ~1-5%
```

### SFTTrainer (Supervised Fine-Tuning)
```python
from trl import SFTTrainer
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,   # effective batch = 2*4 = 8
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    fp16=False,                      # use bf16 instead on modern GPUs
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    optim="paged_adamw_32bit",       # memory-efficient optimizer
    report_to="wandb",               # or "tensorboard"
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",       # field containing formatted prompt
    max_seq_length=2048,
    packing=True,                    # pack multiple samples into one sequence for efficiency
)

trainer.train()
```

### Saving & Merging Adapters
```python
# Save only the LoRA adapter (small — a few MB)
trainer.model.save_pretrained("lora_adapter")
tokenizer.save_pretrained("lora_adapter")

# Merge adapter into base model for inference (removes adapter overhead)
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
merged_model = PeftModel.from_pretrained(base_model, "lora_adapter")
merged_model = merged_model.merge_and_unload()
merged_model.save_pretrained("merged_model")
```

### Pushing to HuggingFace Hub
```python
from huggingface_hub import login
login(token="hf_...")

model.push_to_hub("your-username/model-name")
tokenizer.push_to_hub("your-username/model-name")
```

### Important TrainingArguments Parameters
| Parameter | What it controls |
|---|---|
| `per_device_train_batch_size` | Batch size per GPU — lower if OOM |
| `gradient_accumulation_steps` | Simulate larger batch without more VRAM |
| `gradient_checkpointing` | Trade compute for VRAM (slower but memory efficient) |
| `max_grad_norm` | Gradient clipping — 0.3 is common for QLoRA |
| `warmup_ratio` | % of steps for LR warmup |
| `weight_decay` | L2 regularization — 0.01 is typical |
| `optim` | Use `paged_adamw_32bit` or `adamw_bnb_8bit` for VRAM savings |

---

## 4. Unsloth Fine-Tuning

### Why Unsloth?
- 2–5x faster training than standard HuggingFace + PEFT
- 60–80% less VRAM usage via custom CUDA kernels
- Drop-in replacement — same API as HuggingFace Trainer / TRL
- Supports Llama 3, Mistral, Phi, Gemma, Qwen, and more

### Installation
```bash
pip install unsloth
# For specific CUDA versions:
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### Loading a Model with Unsloth
```python
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-Instruct-bnb-4bit",  # pre-quantized 4bit
    max_seq_length=2048,        # context window
    dtype=None,                 # auto-detect (bfloat16 on Ampere+)
    load_in_4bit=True,          # QLoRA
)
```

### Applying LoRA with Unsloth
```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],  # include MLP for better results
    lora_alpha=16,
    lora_dropout=0,             # 0 is optimized in Unsloth
    bias="none",
    use_gradient_checkpointing="unsloth",   # Unsloth's optimized version
    random_state=42,
    use_rslora=False,           # Rank-Stabilized LoRA — set True for higher r values
    loftq_config=None,
)
```

### Formatting with Chat Templates
```python
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3",    # or "chatml", "mistral", "phi-3", "gemma", etc.
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
             for convo in convos]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)
```

### SFTTrainer with Unsloth
```python
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir="outputs",
    ),
)

trainer_stats = trainer.train()
```

### Inference with Unsloth (after training)
```python
FastLanguageModel.for_inference(model)   # enables 2x faster inference

inputs = tokenizer(
    [tokenizer.apply_chat_template(
        [{"role": "user", "content": "Your question here"}],
        tokenize=False,
        add_generation_prompt=True,
    )],
    return_tensors="pt",
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True)
print(tokenizer.batch_decode(outputs))
```

### Saving with Unsloth

```python
# Save LoRA adapter only
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

# Save merged model in float16
model.save_pretrained_merged("merged_model", tokenizer, save_method="merged_16bit")

# Save in 4-bit quantized format (smallest size)
model.save_pretrained_merged("merged_4bit", tokenizer, save_method="merged_4bit_forced")

# Save as GGUF for llama.cpp / Ollama
model.save_pretrained_gguf("gguf_model", tokenizer, quantization_method="q4_k_m")

# Push to Hub
model.push_to_hub_merged("username/model-name", tokenizer, save_method="merged_16bit", token="hf_...")
model.push_to_hub_gguf("username/model-name-gguf", tokenizer, quantization_method="q4_k_m", token="hf_...")
```

### Unsloth-Supported Models (as of mid-2025)
- Llama 3 / 3.1 / 3.2 / 3.3
- Mistral / Mixtral
- Phi-3 / Phi-4
- Gemma / Gemma 2
- Qwen 2 / 2.5
- DeepSeek R1 / V3

---

## 5. Hyperparameter Cheat Sheet

| Hyperparameter | Recommended Range | Notes |
|---|---|---|
| `r` (LoRA rank) | 8–64 | 16 is a safe default; higher for complex tasks |
| `lora_alpha` | `r` to `2*r` | Set equal to `r` in Unsloth |
| `learning_rate` | 1e-4 to 3e-4 | Lower for larger models |
| `num_train_epochs` | 1–5 | Watch eval loss — stop early if it rises |
| `batch_size` (effective) | 16–64 | Use gradient accumulation to hit this |
| `max_seq_length` | 512–4096 | Match your data's actual length distribution |
| `warmup_ratio` | 0.03–0.1 | Always use warmup |
| `weight_decay` | 0.01–0.1 | Prevents overfitting |
| `lora_dropout` | 0.0–0.1 | 0 recommended in Unsloth |

---

## 6. Common Pitfalls & Fixes

| Problem | Cause | Fix |
|---|---|---|
| OOM (Out of Memory) | Batch too large or seq too long | Reduce batch, enable gradient checkpointing, use 4-bit |
| Loss doesn't decrease | LR too low or data formatting wrong | Check chat template, raise LR slightly |
| Loss goes to 0 immediately | Data leakage / model has seen the data | Use a different dataset or check for contamination |
| Gibberish output | Wrong chat template applied | Match tokenizer chat template to base model |
| Model forgets general knowledge | Too many epochs / LR too high | Reduce epochs, lower LR, add general data to mix |
| Slow training | Not using Flash Attention | Add `attn_implementation="flash_attention_2"` |

---

## 7. Advanced Techniques

### DPO (Direct Preference Optimization)
- Train on preferred vs rejected response pairs — improves alignment without a reward model
- Use `trl.DPOTrainer`; dataset needs `prompt`, `chosen`, `rejected` fields

### ORPO (Odds Ratio Preference Optimization)
- Combines SFT and preference learning in one pass — faster than DPO
- Supported in `trl.ORPOTrainer`

### Continued Pre-Training
- Fine-tune on raw text (no instruction format) to inject domain knowledge before SFT
- Use `DataCollatorForLanguageModeling` with `mlm=False`

### Flash Attention 2
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
)
```

### Gradient Checkpointing
```python
model.gradient_checkpointing_enable()   # trades speed for VRAM
```

---

## 8. Workflow Checklist

- [ ] Choose base model (size vs capability tradeoff)
- [ ] Prepare and validate dataset (format, quality, size)
- [ ] Select fine-tuning method (full / LoRA / QLoRA)
- [ ] Set LoRA config (`r`, `alpha`, `target_modules`)
- [ ] Configure TrainingArguments (LR, batch, epochs)
- [ ] Run a short smoke-test (1–2 steps) before full training
- [ ] Monitor train/eval loss — stop if eval loss increases
- [ ] Run inference tests on held-out examples
- [ ] Merge and quantize if deploying (GGUF for Ollama, GPTQ, AWQ)
- [ ] Push to HuggingFace Hub with a model card