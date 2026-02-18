"""
Configuration file for MoCo Market Embeddings
All hyperparameters and settings in one place
"""
import os
import torch
class MoCoConfig:
    """Configuration class for MoCo training"""
    
    def __init__(self):
        # ==================== DATA CONFIGURATION ====================
        self.data_path = "/home/gaian/Desktop/contrastive_learning_model/tensor_values.parquet"
        self.train_split = 0.9  # 90% train, 10% val (configurable)
        self.random_seed = 42
        
        # ==================== MODEL ARCHITECTURE ====================
        self.input_dim = 16197  # Number of indicators
        self.encoder_hidden_dims = [2048, 1024, 512, 256]
        self.embedding_dim = 128  # Final embedding size
        self.projection_dim = 32  # Projection head output
        self.dropout = 0.3
        
        # ==================== MOCO SPECIFIC ====================
        self.queue_size = 4096
        self.momentum = 0.999  # Momentum coefficient for key encoder
        self.temperature = 0.07  # Temperature for InfoNCE loss
        
        # ==================== AUGMENTATION ====================
        self.mask_prob = 0.20  # Mask 20% of non-zero elements
        self.noise_std = 0.02  # 2% Gaussian noise
        self.temporal_prob = 0.2  # 20% use temporal neighbors
        self.regional_prob = 0.1  # 10% use regional neighbors
        
        # ==================== TRAINING ====================
        self.batch_size = 64
        self.num_epochs = 200
        self.learning_rate = 3e-4
        self.weight_decay = 1e-4
        self.warmup_epochs = 1
        
        # Gradient clipping
        self.grad_clip_norm = 1.0
        
        # Early stopping
        self.early_stopping_patience = 20
        
        # Mixed precision training
        self.use_mixed_precision = True
        
        # ==================== DATA LOADING ====================
        self.num_workers = 4
        self.pin_memory = True
        self.prefetch_factor = 2
        
        # ==================== PATHS ====================
        # self.output_dir = "/home/claude/moco_market/outputs"
        # self.checkpoint_dir = "/home/claude/moco_market/checkpoints"
        # self.log_dir = "/home/claude/moco_market/logs"

        
        self.output_dir = os.path.join(os.getcwd(), "outputs")
        self.checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
        self.log_dir = os.path.join(os.getcwd(), "logs")
                
        # ==================== DEVICE ====================
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # ==================== LOGGING ====================
        self.log_interval = 10  # Log every 10 batches
        self.save_interval = 5  # Save checkpoint every 5 epochs
        
    def update(self, **kwargs):
        """Update configuration with custom values"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")
    
    def __repr__(self):
        """Pretty print configuration"""
        lines = ["MoCo Configuration:"]
        lines.append("=" * 60)
        for key, value in self.__dict__.items():
            lines.append(f"  {key:30s}: {value}")
        lines.append("=" * 60)
        return "\n".join(lines)


# Example usage:
if __name__ == "__main__":
    config = MoCoConfig()
    print(config)
    
    # Update for full training (no validation)
    config.update(train_split=1.0, num_epochs=300)
    print("\nUpdated config:")
    print(config)