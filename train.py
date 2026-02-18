"""
Main training script for MoCo Market Embeddings
This script orchestrates the entire training pipeline
"""

import torch
import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import MoCoConfig
from data_preprocessing import MarketDataPreprocessor, create_dataloaders
from augmentation import MarketDataAugmentation
from model import MoCo
from trainer import MoCoTrainer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train MoCo for Market Embeddings')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, 
                       default='/home/gaian/Desktop/contrastive_learning_model/tensor_values.parquet',
                       help='Path to parquet data file')
    parser.add_argument('--train_split', type=float, default=0.9,
                       help='Train/val split ratio (use 1.0 for no validation)')
    
    # Model arguments
    parser.add_argument('--embedding_dim', type=int, default=128,
                       help='Embedding dimension')
    parser.add_argument('--projection_dim', type=int, default=32,
                       help='Projection dimension')
    parser.add_argument('--queue_size', type=int, default=4096,
                       help='Queue size for negative samples')
    parser.add_argument('--momentum', type=float, default=0.999,
                       help='Momentum coefficient')
    parser.add_argument('--temperature', type=float, default=0.07,
                       help='Temperature for InfoNCE loss')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                       help='Number of warmup epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                       help='Early stopping patience')
    
    # Augmentation arguments
    parser.add_argument('--mask_prob', type=float, default=0.20,
                       help='Probability of masking non-zero elements')
    parser.add_argument('--noise_std', type=float, default=0.02,
                       help='Standard deviation of Gaussian noise')
    
    # System arguments
    parser.add_argument('--device', type=str, default=("cuda" if torch.cuda.is_available() else "cpu"),
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--use_mixed_precision', action='store_true', default=True,
                       help='Use mixed precision training')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, 
                       default='/home/claude/moco_market/outputs',
                       help='Output directory')
    parser.add_argument('--checkpoint_dir', type=str,
                       default='/home/claude/moco_market/checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str,
                       default='/home/claude/moco_market/logs',
                       help='Log directory')
    
    # Misc
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def main():
    """Main training function"""
    
    # Parse arguments
    args = parse_args()
    
    # Create config
    config = MoCoConfig()
    
    # Update config with command line arguments
    for key, value in vars(args).items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    print("\n" + "="*70)
    print(" "*20 + "MoCo Market Embeddings Training")
    print("="*70)
    print(config)
    print("="*70 + "\n")
    
    # Set random seeds
    torch.manual_seed(config.random_seed)
    
    # ==================== STEP 1: Load and Preprocess Data ====================
    print("\n" + "="*70)
    print("STEP 1: Loading and Preprocessing Data")
    print("="*70)
    
    preprocessor = MarketDataPreprocessor(config)
    df = preprocessor.load_data(config.data_path)
    data_tensor = preprocessor.preprocess(df)
    
    # Split data
    train_data, val_data = preprocessor.split_data(data_tensor, config.train_split)
    
    # Save preprocessor
    preprocessor_path = os.path.join(config.output_dir, 'preprocessor.pkl')
    preprocessor.save_preprocessor(preprocessor_path)
    
    # ==================== STEP 2: Create DataLoaders ====================
    print("\n" + "="*70)
    print("STEP 2: Creating DataLoaders")
    print("="*70)
    
    train_loader, val_loader = create_dataloaders(
        train_data, 
        val_data, 
        config,
        temporal_neighbors=preprocessor.temporal_neighbors
    )
    
    # ==================== STEP 3: Create Model ====================
    print("\n" + "="*70)
    print("STEP 3: Creating MoCo Model")
    print("="*70)
    
    model = MoCo(config)
    
    # ==================== STEP 4: Create Augmentation ====================
    print("\n" + "="*70)
    print("STEP 4: Setting Up Augmentation")
    print("="*70)
    
    augmentation = MarketDataAugmentation(
        config,
        temporal_neighbors=preprocessor.temporal_neighbors
    )
    print(f"Augmentation configured:")
    print(f"  - Mask probability: {config.mask_prob}")
    print(f"  - Noise std: {config.noise_std}")
    print(f"  - Temporal probability: {config.temporal_prob}")
    
    # ==================== STEP 5: Train Model ====================
    print("\n" + "="*70)
    print("STEP 5: Training MoCo Model")
    print("="*70)
    
    trainer = MoCoTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        augmentation=augmentation
    )
    
    trained_model = trainer.train()
    
    # ==================== STEP 6: Save Final Model ====================
    print("\n" + "="*70)
    print("STEP 6: Saving Final Model")
    print("="*70)
    
    final_model_path = os.path.join(config.output_dir, 'final_model.pt')
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': config.__dict__
    }, final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    # ==================== DONE ====================
    print("\n" + "="*70)
    print(" "*25 + "TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModel files saved in:")
    print(f"  - Best model: {config.checkpoint_dir}/best_model.pt")
    print(f"  - Final model: {final_model_path}")
    print(f"  - Preprocessor: {preprocessor_path}")
    print(f"  - Training logs: {config.log_dir}/training_stats.json")
    print("\nTo use the trained model:")
    print("  from inference import MarketEmbeddingInference")
    print("  from trainer import load_trained_model")
    print(f"  model, config = load_trained_model('{config.checkpoint_dir}/best_model.pt')")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()