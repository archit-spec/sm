from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = Path("./pretrained_qwen_30b/checkpoint-200").resolve()

print("Loading your trained model from:", model_dir)
model = AutoModelForCausalLM.from_pretrained(
    str(model_dir),
    torch_dtype=torch.bfloat16,     # T4-friendly
    device_map="auto",
    trust_remote_code=True,        # Qwen models typically need this
    local_files_only=True,         # force local, avoids Hub validation
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    str(model_dir),
    trust_remote_code=True,
    local_files_only=True,
)

# If pad token is missing, set it to eos to silence warnings.
if tokenizer.pad_token is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token

test_prompts = [
    "use crate::",
    "pub struct PaymentRequest",
    "async fn handle_payment",
    "#[derive(Clone, Debug)]",
    "impl From<",
    "pub enum PaymentStatus",
]

print("\nTesting repo-specific recall:")
print("=" * 50)
for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=150,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    print("\nInput:", prompt)
    print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("-" * 50)
