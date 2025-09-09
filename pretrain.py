import argparse
import logging
import os
import math
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import torch
import torch.nn as nn
from accelerate import Accelerator
from huggingface_hub import HfApi, login
import gc

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_accelerator():
    """Setup accelerator for multi-GPU training."""
    accelerator = Accelerator()
    logger.info(f"Using accelerator with {accelerator.num_processes} processes")
    logger.info(f"Device: {accelerator.device}")
    logger.info(f"Mixed precision: {accelerator.mixed_precision}")
    return accelerator

def get_gpu_memory_info():
    """Get GPU memory information."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            logger.info(f"GPU {i}: {props.name}, Memory: {memory_gb:.1f} GB")
            
            # Current memory usage
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            cached = torch.cuda.memory_reserved(i) / (1024**3)
            logger.info(f"GPU {i} Memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")

def calculate_optimal_batch_size(model_name, max_length, num_gpus):
    """Calculate optimal batch size based on model size and available memory."""
    # Model size estimates (in billions of parameters)
    model_sizes = {
        "Qwen/Qwen3-4B": 4,
        "Qwen/Qwen3-7B": 7,
        "Qwen/Qwen3-14B": 14,
        "Qwen/Qwen3-32B": 32,
        "Qwen/Qwen3-Coder-30B": 30,
    }
    
    # Get model size
    model_size_b = 4  # default
    for key, size in model_sizes.items():
        if key in model_name:
            model_size_b = size
            break
    
    # Memory calculations (rough estimates)
    # H200 has ~140GB memory
    memory_per_gpu = 140
    
    # Memory usage breakdown (in GB):
    # - Model weights (fp16): model_size_b * 2
    # - Gradients: model_size_b * 2  
    # - Optimizer states (AdamW): model_size_b * 8
    # - Activations: depends on batch_size and sequence_length
    
    model_memory = model_size_b * 2  # fp16 weights
    gradient_memory = model_size_b * 2
    optimizer_memory = model_size_b * 8
    
    base_memory = model_memory + gradient_memory + optimizer_memory
    available_memory = memory_per_gpu - base_memory - 10  # 10GB buffer
    
    # Activation memory per sample (rough estimate)
    # For transformer: ~2 * num_layers * hidden_size * sequence_length * batch_size
    activation_memory_per_token = model_size_b * 0.001  # GB per token
    activation_memory_per_sample = activation_memory_per_token * max_length
    
    max_batch_per_gpu = max(1, int(available_memory / activation_memory_per_sample))
    optimal_batch_size = min(max_batch_per_gpu, 4)  # Cap at 4 for stability
    
    logger.info(f"Model size: {model_size_b}B parameters")
    logger.info(f"Estimated base memory usage: {base_memory:.1f} GB")
    logger.info(f"Available memory for activations: {available_memory:.1f} GB")
    logger.info(f"Calculated optimal batch size per GPU: {optimal_batch_size}")
    
    return optimal_batch_size

def upload_to_huggingface(model_path, repo_name, username="architt11", private=False):
    """Upload trained model to Hugging Face Hub."""
    try:
        api = HfApi()
        repo_id = f"{username}/{repo_name}"
        logger.info(f"Uploading model to: {repo_id}")
        
        # Create the repository
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        
        # Upload the model files
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            repo_type="model",
        )
        
        logger.info(f"Successfully uploaded model to https://huggingface.co/{repo_id}")
        return f"https://huggingface.co/{repo_id}"
        
    except Exception as e:
        logger.error(f"Failed to upload model to Hugging Face Hub: {e}")
        logger.info("Make sure you're logged in with: huggingface-cli login")
        raise

def load_model_and_tokenizer(model_name, use_flash_attention=False):
    """Load the model and tokenizer with memory optimizations."""
    try:
        logger.info(f"Loading model: {model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Add padding token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Model loading configuration
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        
        # Add flash attention if available
        if use_flash_attention:
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Using Flash Attention 2")
            except:
                logger.info("Flash Attention 2 not available, using default attention")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        # Enable gradient checkpointing to save memory
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise

def load_and_process_dataset(dataset_name, content_field="content", streaming=False, max_samples=None):
    """Load and process the dataset."""
    try:
        logger.info(f"Loading dataset: {dataset_name}")
        
        # Load dataset
        if streaming:
            dataset = load_dataset(dataset_name, streaming=True)
            dataset = dataset["train"]
        else:
            dataset = load_dataset(dataset_name)
            if "train" not in dataset:
                raise ValueError("Dataset does not have a 'train' split")
            dataset = dataset["train"]
        
        # Limit samples if specified
        if max_samples and not streaming:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
            logger.info(f"Limited dataset to {len(dataset)} samples")
        
        def extract_content(example):
            if content_field not in example:
                available_fields = list(example.keys())
                raise ValueError(f"Content field '{content_field}' not found. Available fields: {available_fields}")
            
            text = example[content_field]
            if not isinstance(text, str):
                text = str(text)
            
            return {"text": text}
        
        # Process dataset
        if streaming:
            processed_dataset = dataset.map(extract_content)
        else:
            processed_dataset = dataset.map(
                extract_content, 
                remove_columns=dataset.column_names,
                desc="Processing dataset"
            )
        
        return processed_dataset
        
    except Exception as e:
        logger.error(f"Failed to load and process dataset {dataset_name}: {e}")
        raise

def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize the text data."""
    # Tokenize with truncation and padding
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_special_tokens_mask=True,
    )
    
    # For causal LM, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

def clear_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

class CustomTrainer(Trainer):
    """Custom trainer with memory optimizations."""
    
    def training_step(self, model, inputs):
        """Override training step to add memory management."""
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        
        if self.args.n_gpu > 1:
            loss = loss.mean()
        
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        self.accelerator.backward(loss)
        
        # Clear cache periodically
        if self.state.global_step % 10 == 0:
            clear_memory()
        
        return loss.detach()

def main(args):
    try:
        # Set environment variables for memory optimization
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Setup accelerator for multai-GPU training
        accelerator = None
        if args.multi_gpu:
            accelerator = setup_accelerator()
            
        # Get GPU memory info
        get_gpu_memory_info()
        
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(args.model_name, False)
        
        # Calculate optimal batch size if not specified
        if args.auto_batch_size or args.batch_size <= 0:
            num_gpus = torch.cuda.device_count() if args.multi_gpu else 1
            optimal_batch_size = calculate_optimal_batch_size(args.model_name, args.max_length, num_gpus)
            args.batch_size = optimal_batch_size
            logger.info(f"Using calculated batch size: {args.batch_size}")
        
        # Load and process dataset
        dataset = load_and_process_dataset(
            args.dataset_name, 
            args.content_field, 
            streaming=args.streaming,
            max_samples=args.max_samples
        )
        
        # Tokenize dataset
        logger.info("Tokenizing dataset...")
        if args.streaming:
            # For streaming datasets, we can't preprocess everything
            def tokenize_streaming(examples):
                return tokenize_function(examples, tokenizer, args.max_length)
            
            tokenized_dataset = dataset.map(
                tokenize_streaming,
                batched=True,
                remove_columns=["text"]
            )
        else:
            tokenized_dataset = dataset.map(
                lambda x: tokenize_function(x, tokenizer, args.max_length),
                batched=True,
                remove_columns=["text"],
                desc="Tokenizing",
                num_proc=4
            )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8,  # For tensor core optimization
        )
        
        # Calculate effective batch size
        effective_batch_size = args.batch_size * args.gradient_accumulation_steps
        if args.multi_gpu:
            effective_batch_size *= torch.cuda.device_count()
        
        logger.info(f"Effective batch size: {effective_batch_size}")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=0.01,
            warmup_steps=args.warmup_steps,
            save_steps=args.save_steps,
            save_total_limit=2,
            logging_steps=args.logging_steps,
            eval_strategy="no",
            fp16=torch.cuda.is_available() and not args.bf16,
            bf16=args.bf16,
            dataloader_num_workers=2,
            remove_unused_columns=False,
            gradient_checkpointing=True,
            ddp_find_unused_parameters=False,
            dataloader_pin_memory=False,
            max_grad_norm=1.0,
            lr_scheduler_type="cosine",
            report_to=["wandb"] if args.use_wandb else [],
            run_name=f"{args.model_name.split('/')[-1]}-{args.dataset_name.split('/')[-1]}",
            seed=42,
        )
        
        # Initialize trainer
        trainer_class = MemoryOptimizedTrainer if args.use_custom_trainer else Trainer
        trainer = trainer_class(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        
        # Clear memory before training
        clear_memory()
        
        # Train
        logger.info("Starting training...")
        logger.info(f"Number of training samples: {len(tokenized_dataset) if not args.streaming else 'streaming'}")
        
        try:
            trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        except Exception as e:
            logger.error(f"Training failed: {e}")
            # Try to save what we have
            if hasattr(trainer, 'state') and trainer.state.global_step > 0:
                logger.info("Attempting to save checkpoint...")
                trainer.save_model(f"{args.output_dir}/emergency_checkpoint")
            raise
        
        # Save the final model
        save_model = True
        if accelerator and not accelerator.is_main_process:
            save_model = False
        
        if save_model:
            logger.info(f"Saving model to {args.output_dir}")
            trainer.save_model(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            
            # Upload to Hugging Face Hub if requested
            if args.upload_to_hf:
                upload_to_huggingface(
                    args.output_dir,
                    args.hf_repo_name,
                    args.hf_username,
                    args.hf_private
                )
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        get_gpu_memory_info()  # Show final memory state
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory-optimized pretraining for large language models")
    
    # Model and data arguments
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B", 
                       help="Model name from Hugging Face Hub")
    parser.add_argument("--dataset_name", type=str, default="archit11/hyperswitch-code-only", 
                       help="Dataset name from Hugging Face Hub")
    parser.add_argument("--content_field", type=str, default="content", 
                       help="Field containing the text data")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./pretrained_model", 
                       help="Output directory for saving model")
    parser.add_argument("--batch_size", type=int, default=-1, 
                       help="Batch size per device (-1 for auto)")
    parser.add_argument("--auto_batch_size", action="store_true", 
                       help="Automatically calculate optimal batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, 
                       help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=1, 
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, 
                       help="Learning rate")
    parser.add_argument("--max_length", type=int, default=2048, 
                       help="Maximum sequence length")
    parser.add_argument("--warmup_steps", type=int, default=100, 
                       help="Number of warmup steps")
    
    # Memory and performance arguments
    parser.add_argument("--multi_gpu", action="store_true", 
                       help="Enable multi-GPU training")
    parser.add_argument("--bf16", action="store_true", 
                       help="Use bfloat16 instead of float16")
    parser.add_argument("--use_flash_attention", action="store_true", default=True, 
                       help="Use Flash Attention 2 if available")
    parser.add_argument("--use_custom_trainer", action="store_true", 
                       help="Use custom trainer with memory optimizations")
    
    # Data arguments
    parser.add_argument("--streaming", action="store_true", 
                       help="Use streaming dataset (for very large datasets)")
    parser.add_argument("--max_samples", type=int, default=None, 
                       help="Maximum number of samples to use (for testing)")
    
    # Logging and saving arguments
    parser.add_argument("--save_steps", type=int, default=500, 
                       help="Save checkpoint every N steps")
    parser.add_argument("--logging_steps", type=int, default=10, 
                       help="Log every N steps")
    parser.add_argument("--use_wandb", action="store_true", 
                       help="Use Weights & Biases for logging")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, 
                       help="Resume training from checkpoint")
    
    # Hugging Face Hub arguments
    parser.add_argument("--upload_to_hf", action="store_true", 
                       help="Upload model to Hugging Face Hub after training")
    parser.add_argument("--hf_repo_name", type=str, default="qwen-pretrained", 
                       help="Hugging Face repository name")
    parser.add_argument("--hf_username", type=str, default="architt11", 
                       help="Hugging Face username")
    parser.add_argument("--hf_private", action="store_true", 
                       help="Make Hugging Face repository private")
    
    args = parser.parse_args()
    
    # Set auto batch size if batch_size is -1
    if args.batch_size == -1:
        args.auto_batch_size = True
        args.batch_size = 1  # Fallback value
    
    main(args)