import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
import os
from pathlib import Path
import json
import random
import time
import logging
import argparse
from typing import Tuple, List, Dict, Any


def setup_logging():
    """Configure logging for the training process."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )
    return logging.getLogger(__name__)


def prepare_dataset(json_path: str, val_split: float = 0.1, sample_ratio: float = 1.0) -> Tuple[str, str]:
    """
    Prepare dataset with support for sampling and train/validation split.

    Args:
        json_path: Path to JSON data file
        val_split: Validation set ratio
        sample_ratio: Sampling ratio for quick experiments

    Returns:
        Tuple containing paths to train and validation jsonl files
    """
    logger = logging.getLogger(__name__)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    jsonl_data = []
    for sample in data.get('samples', []):
        dpo_sample = {
            "prompt": json.dumps(sample["prompt"]),
            "chosen": json.dumps(sample["chosen"]),
            "rejected": json.dumps(sample["rejected"])
        }
        jsonl_data.append(dpo_sample)

    # Random sampling for quick experiments
    if sample_ratio < 1.0:
        total_samples = len(jsonl_data)
        sample_size = max(1, int(total_samples * sample_ratio))
        jsonl_data = random.sample(jsonl_data, sample_size)
        logger.info(f"Sampled {sample_size}/{total_samples} samples for quick experimentation")

    # Split into train and validation sets
    num_val = max(1, int(len(jsonl_data) * val_split))
    train_data = jsonl_data[:-num_val]
    val_data = jsonl_data[-num_val:]

    # Save datasets
    train_path = 'train.jsonl'
    val_path = 'val.jsonl'
    
    for split, data, filename in [
        ('train', train_data, train_path),
        ('val', val_data, val_path)
    ]:
        with open(filename, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')

    logger.info(f"Dataset split completed: {len(train_data)} training samples, {len(val_data)} validation samples")
    return train_path, val_path


def calculate_batch_size(max_seq_length: int) -> int:
    """
    Calculate optimal batch size based on available GPU memory.
    
    Args:
        max_seq_length: Maximum sequence length for the model
        
    Returns:
        Calculated batch size
    """
    logger = logging.getLogger(__name__)
    
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, using default batch size of 4")
        return 4
        
    # Get GPU memory in bytes
    total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
    # Use 75% of available memory to be safe
    available_memory = total_gpu_memory * 0.75
    
    # Estimate memory per sample based on sequence length
    # This is a rough estimate and may need adjustment based on model size
    bytes_per_token = 16  # For 4-bit quantization
    estimated_sample_memory = 2 * max_seq_length * bytes_per_token * 4  # 2 sequences per sample (chosen & rejected)
    
    batch_size = max(1, int(available_memory / estimated_sample_memory))
    
    # Cap at reasonable values
    batch_size = min(batch_size, 32)
    
    logger.info(f"Automatically calculated batch size: {batch_size}")
    return batch_size


def prepare_training(args):
    """
    Prepare model, tokenizer, and datasets for DPO training.
    
    Args:
        args: Command line arguments
        
    Returns:
        Configured DPO trainer
    """
    logger = logging.getLogger(__name__)
    
    # Check CUDA availability
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"Current GPU device: {torch.cuda.get_device_name(0)}")
        logger.info(f"Available GPU count: {torch.cuda.device_count()}")
        
        # Print GPU memory info
        logger.info("GPU memory information:")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name}, {props.total_memory / 1024**3:.1f}GB")

    # Create output directory
    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    # Prepare dataset
    train_path, val_path = prepare_dataset(
        args.json_path, 
        val_split=args.val_split, 
        sample_ratio=args.sample_ratio
    )

    # Load datasets
    train_dataset = load_dataset('json', data_files=train_path, split='train')
    val_dataset = load_dataset('json', data_files=val_path, split='train')
    logger.info(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples")

    # Initialize model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        args.model_path,
        load_in_4bit=True,
        max_seq_length=args.max_seq_length
    )

    # Calculate batch size
    batch_size = calculate_batch_size(args.max_seq_length) if args.batch_size is None else args.batch_size

    # Prepare LoRA adapter
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=args.target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=args.seed,
        max_seq_length=args.max_seq_length
    )

    # Set padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    # Configure gradient accumulation steps
    gradient_accumulation_steps = max(1, args.target_batch_size // batch_size)

    # Configure training arguments
    training_args = DPOConfig(
        output_dir=args.output_path,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        num_train_epochs=args.num_epochs,
        logging_dir=f"{args.output_path}/logs",
        bf16=args.bf16,
        fp16=args.fp16,
        fp16_opt_level="O2",
        remove_unused_columns=False,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        gradient_checkpointing=True,
        loss_type=args.loss_type,
        report_to="tensorboard",
        seed=args.seed
    )

    logger.info("Initializing DPO trainer...")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        beta=args.beta
    )

    # Print training configuration
    logger.info("\nTraining configuration:")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    logger.info(f"Loss type: {args.loss_type}")
    logger.info(f"Beta value: {args.beta}")
    logger.info(f"Learning rate: {args.learning_rate}")
    
    return trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="DPO Fine-tuning for Language Models")
    
    # Data arguments
    parser.add_argument("--json_path", type=str, 
                        default="/path/to/your/dataset.json",
                        help="Path to the JSON dataset file")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation set ratio")
    parser.add_argument("--sample_ratio", type=float, default=1.0,
                        help="Ratio of data to sample for quick experiments")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, 
                        default="/path/to/base/model",
                        help="Path to base model")
    parser.add_argument("--output_path", type=str, 
                        default="/path/to/output",
                        help="Path to save the model")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="LoRA dropout")
    parser.add_argument("--target_modules", type=str, nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj", 
                                 "gate_proj", "down_proj", "up_proj"],
                        help="Target modules for LoRA")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Training batch size (auto-calculated if None)")
    parser.add_argument("--target_batch_size", type=int, default=32,
                        help="Target effective batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Logging steps")
    parser.add_argument("--eval_steps", type=int, default=50,
                        help="Evaluation steps")
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Save steps")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Maximum number of checkpoints to keep")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio")
    parser.add_argument("--max_grad_norm", type=float, default=0.3,
                        help="Maximum gradient norm")
    parser.add_argument("--bf16", action="store_true",
                        help="Use bfloat16 precision")
    parser.add_argument("--fp16", action="store_true",
                        help="Use fp16 precision")
    parser.add_argument("--loss_type", type=str, default="exo_pair",
                        choices=["exo_pair", "sigmoid", "hinge", "ipo"],
                        help="DPO loss type")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO beta parameter")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    return parser.parse_args()


def main():
    """Main function to run DPO training."""
    # Set up logging
    logger = setup_logging()
    
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Set seeds for reproducibility
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        
        logger.info("Preparing for training...")
        trainer = prepare_training(args)

        # Record start time
        start_time = time.time()

        logger.info("Starting DPO training...")
        trainer.train()

        # Calculate training time
        training_time = time.time() - start_time
        logger.info(f"Training completed! Total time: {training_time / 3600:.2f} hours")

        # Save best model
        best_ckpt = trainer.state.best_model_checkpoint
        logger.info(f"Best checkpoint: {best_ckpt}")
        logger.info(f"Best validation loss: {trainer.state.best_metric}")

        # Create a README.md in the best checkpoint directory
        if best_ckpt:
            with open(os.path.join(best_ckpt, "README.md"), "w") as f:
                f.write(f"# Best Checkpoint\n\n")
                f.write(f"- Validation Loss: {trainer.state.best_metric}\n")
                f.write(f"- Training completed on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        trainer.save_model(best_ckpt if best_ckpt else args.output_path)
        logger.info("Model saved successfully!")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
