#!/usr/bin/env python3
"""
Script to upload your trained model to Hugging Face Hub
"""

import os
import argparse
from huggingface_hub import HfApi, login, create_repo
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def check_login():
    """Check if user is logged in to Hugging Face"""
    try:
        api = HfApi()
        user_info = api.whoami()
        logger.info(f"Logged in as: {user_info['name']} ({user_info.get('email', 'N/A')})")
        return user_info['name']
    except Exception as e:
        logger.error("Not logged in to Hugging Face Hub")
        logger.info("Please run: huggingface-cli login")
        return None

def upload_model(model_path, repo_name, username, private=False, description=None):
    """Upload model to Hugging Face Hub"""
    
    # Check if logged in
    current_user = check_login()
    if not current_user:
        return False
    
    # Verify model path exists
    if not os.path.exists(model_path):
        logger.error(f"Model path does not exist: {model_path}")
        return False
    
    # Check required files
    required_files = ['config.json', 'tokenizer.json', 'tokenizer_config.json']
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            missing_files.append(file)
    
    if missing_files:
        logger.warning(f"Missing files: {missing_files}")
        logger.info("Continuing anyway...")
    
    try:
        api = HfApi()
        repo_id = f"{username}/{repo_name}"
        
        logger.info(f"Creating repository: {repo_id}")
        
        # Create repository
        create_repo(
            repo_id=repo_id,
            private=private,
            exist_ok=True,
            repo_type="model"
        )
        
        # Create model card if description provided
        if description:
            model_card = f"""---
license: apache-2.0
base_model: Qwen/Qwen3-4B
tags:
- code
- hyperswitch
- pretrained
- qwen
language:
- en
pipeline_tag: text-generation
---

# {repo_name}

{description}

## Model Details

- **Base Model**: Qwen/Qwen3-4B
- **Training Data**: Hyperswitch repository code
- **Final Training Loss**: 0.4966
- **Training Progress**: 92% complete (excellent results)

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "{repo_id}",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

# Generate code
prompt = "use crate::"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=150, temperature=0.3)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

## Training Details

This model was fine-tuned on the Hyperswitch codebase to improve code completion and generation for Rust payment processing systems.
"""
            
            # Write model card
            model_card_path = os.path.join(model_path, "README.md")
            with open(model_card_path, "w", encoding="utf-8") as f:
                f.write(model_card)
            logger.info("Created model card")
        
        # Upload the model
        logger.info(f"Uploading model files from {model_path}...")
        logger.info("This may take several minutes...")
        
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload trained Qwen-4B model on Hyperswitch data"
        )
        
        logger.info(f"✅ Successfully uploaded model to: https://huggingface.co/{repo_id}")
        return True
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Upload trained model to Hugging Face Hub")
    parser.add_argument("--model_path", type=str, 
                       default="./pretrained_qwen_4b/emergency_checkpoint",
                       help="Path to the trained model directory")
    parser.add_argument("--repo_name", type=str,
                       default="qwen-4b-hyperswitch-pretrained",
                       help="Repository name on Hugging Face Hub")
    parser.add_argument("--username", type=str,
                       default="architt11",
                       help="Hugging Face username")
    parser.add_argument("--private", action="store_true",
                       help="Make repository private")
    parser.add_argument("--description", type=str,
                       default="Qwen-4B model fine-tuned on Hyperswitch codebase for improved Rust code completion",
                       help="Model description")
    
    args = parser.parse_args()
    
    print("🚀 Hugging Face Model Upload Tool")
    print("=" * 50)
    print(f"Model Path: {args.model_path}")
    print(f"Repository: {args.username}/{args.repo_name}")
    print(f"Private: {args.private}")
    print("=" * 50)
    
    # Confirm upload
    confirm = input("\nProceed with upload? [y/N]: ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("Upload cancelled.")
        return
    
    # Upload model
    success = upload_model(
        model_path=args.model_path,
        repo_name=args.repo_name,
        username=args.username,
        private=args.private,
        description=args.description
    )
    
    if success:
        print(f"\n🎉 Upload completed successfully!")
        print(f"📝 Your model is available at: https://huggingface.co/{args.username}/{args.repo_name}")
        print("\n📖 Usage example:")
        print(f'from transformers import AutoModelForCausalLM, AutoTokenizer')
        print(f'model = AutoModelForCausalLM.from_pretrained("{args.username}/{args.repo_name}")')
        print(f'tokenizer = AutoTokenizer.from_pretrained("{args.username}/{args.repo_name}")')
    else:
        print("\n❌ Upload failed. Check the logs above for details.")

if __name__ == "__main__":
    main()
