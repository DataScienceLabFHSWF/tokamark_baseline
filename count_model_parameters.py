#!/usr/bin/env python3
"""
Collect model parameter counts for all PLUME and baseline models.
Useful for understanding model size differences and reporting in paper.
"""

import sys
import yaml
import argparse
from pathlib import Path

try:
    from globals import REPO_ROOT
except ImportError:
    from .globals import REPO_ROOT

from tokamark.data import initialize_TokaMark_dataset
from tokamark.tasks import get_task_metadata
from torch.utils.data import DataLoader

from src.model_factory import MODEL_CHOICES, create_model


def count_parameters(model):
    """Count total trainable and non-trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def count_model_parameters(model_name, dataloader, metadata, config, task_name="task_3-1"):
    """Instantiate a model and count its parameters."""
    try:
        model = create_model(
            model_name=model_name,
            dataloader=dataloader,
            metadata=metadata,
            config=config,
            task_name=task_name,
            verbose=False,
        )
        total, trainable = count_parameters(model)
        return {
            "model": model_name,
            "total_params": total,
            "trainable_params": trainable,
            "non_trainable_params": total - trainable,
            "status": "success",
            "error": None,
        }
    except Exception as e:
        return {
            "model": model_name,
            "total_params": None,
            "trainable_params": None,
            "non_trainable_params": None,
            "status": "failed",
            "error": str(e),
        }


def format_number(n):
    """Format number with commas for readability."""
    if n is None:
        return "N/A"
    return f"{n:,}"


def main():
    parser = argparse.ArgumentParser(
        description="Collect model parameter counts for all Task 3-1 models"
    )
    parser.add_argument("--task", default="task_3-1", help="Task name")
    parser.add_argument("--config", default="/src/config/config_model.yaml", help="Config path")
    parser.add_argument("--split", default="random", help="Data split")
    parser.add_argument("--seed", type=int, default=23, help="Random seed")
    parser.add_argument("--output", default=None, help="Output CSV file (default: stdout)")
    
    args = parser.parse_args()

    print("=" * 80)
    print(f"Collecting model parameter counts for {args.task}")
    print(f"Config: {args.config}, Split: {args.split}, Seed: {args.seed}")
    print("=" * 80)
    print()

    # Load config
    config_path = REPO_ROOT + args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize minimal dataloader and metadata (needed for model instantiation)
    print("Initializing dataset...")
    dataset = initialize_TokaMark_dataset(
        task_name=args.task,
        split=args.split,
        seed=args.seed,
    )
    metadata = get_task_metadata(args.task)
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 32),
        shuffle=False,
        num_workers=0,
    )
    
    # Count parameters for each model
    results = []
    print("\nCounting parameters for each model...\n")
    
    for model_name in MODEL_CHOICES:
        print(f"  {model_name:25} ...", end=" ", flush=True)
        result = count_model_parameters(
            model_name=model_name,
            dataloader=dataloader,
            metadata=metadata,
            config=config,
            task_name=args.task,
        )
        results.append(result)
        
        if result["status"] == "success":
            total_fmt = format_number(result["total_params"])
            train_fmt = format_number(result["trainable_params"])
            print(f"✓ {total_fmt:>15} (trainable: {train_fmt:>15})")
        else:
            print(f"✗ {result['error']}")
    
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    
    # Display table
    print(f"{'Model':<25} {'Total Params':>18} {'Trainable':>18} {'Non-trainable':>18}")
    print("-" * 80)
    
    total_params_list = []
    for result in results:
        if result["status"] == "success":
            total_fmt = format_number(result["total_params"])
            train_fmt = format_number(result["trainable_params"])
            non_train_fmt = format_number(result["non_trainable_params"])
            print(f"{result['model']:<25} {total_fmt:>18} {train_fmt:>18} {non_train_fmt:>18}")
            total_params_list.append(result["total_params"])
        else:
            print(f"{result['model']:<25} {'FAILED':<18} {result['error'][:30]}")
    
    print()
    if total_params_list:
        min_params = min(total_params_list)
        max_params = max(total_params_list)
        avg_params = sum(total_params_list) / len(total_params_list)
        print(f"Min parameters:  {format_number(int(min_params))}")
        print(f"Max parameters:  {format_number(int(max_params))}")
        print(f"Avg parameters:  {format_number(int(avg_params))}")
        print(f"Std dev:         {format_number(int(max_params - min_params))} (max - min)")
    
    print()
    
    # Optionally save to CSV
    if args.output:
        import csv
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["model", "total_params", "trainable_params", "non_trainable_params", "status", "error"]
            )
            writer.writeheader()
            writer.writerows(results)
        
        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
