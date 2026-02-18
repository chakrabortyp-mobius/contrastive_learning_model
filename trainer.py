"""
Training module for MoCo
Includes trainer class, loss computation, and full training loop
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import os
import time
from tqdm import tqdm
import json


class MoCoTrainer:
    """
    Trainer class for MoCo model
    Handles training loop, validation, checkpointing, and logging
    """
    
    def __init__(self, model, train_loader, val_loader, config, augmentation):
        """
        Args:
            model: MoCo model
            train_loader: Training dataloader
            val_loader: Validation dataloader (can be None)
            config: Configuration object
            augmentation: Augmentation object
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.augmentation = augmentation
        
        # Device
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        print(f"\nUsing device: {self.device}")
        
        # Optimizer (only for query encoder and projection head)
        self.optimizer = torch.optim.AdamW(
            list(model.encoder_q.parameters()) + list(model.projection_q.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Loss function (Cross Entropy for InfoNCE)
        self.criterion = nn.CrossEntropyLoss()
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config.use_mixed_precision else None
        
        # Tracking
        self.current_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_train_loss = float('inf')  # Track best train loss if no validation
        self.patience_counter = 0
        
        # Create output directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        # Training stats
        self.stats = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_time': []
        }
    
    def get_lr(self, epoch):
        """
        Learning rate schedule with warmup and cosine annealing
        
        Args:
            epoch: Current epoch number
        
        Returns:
            lr: Learning rate for this epoch
        """
        if epoch < self.config.warmup_epochs:
            # Linear warmup
            return (epoch / self.config.warmup_epochs) * self.config.learning_rate
        else:
            # Cosine annealing
            progress = (epoch - self.config.warmup_epochs) / \
                      (self.config.num_epochs - self.config.warmup_epochs)
            return self.config.learning_rate * 0.5 * (1 + np.cos(np.pi * progress))
    
    def train_epoch(self, epoch):
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
        
        Returns:
            avg_loss: Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        
        # Update learning rate
        lr = self.get_lr(epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.num_epochs}")
        
        for batch_idx, markets in enumerate(pbar):
            markets = markets.to(self.device)
            
            # Generate two augmented views
            view1, view2 = self.augmentation.get_two_views(markets)
            
            # Mixed precision forward pass
            if self.scaler is not None:
                with autocast():
                    logits, labels, _ = self.model(view1, view2)
                    loss = self.criterion(logits, labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.encoder_q.parameters()) + \
                    list(self.model.projection_q.parameters()),
                    max_norm=self.config.grad_clip_norm
                )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Without mixed precision
                logits, labels, _ = self.model(view1, view2)
                loss = self.criterion(logits, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.encoder_q.parameters()) + \
                    list(self.model.projection_q.parameters()),
                    max_norm=self.config.grad_clip_norm
                )
                
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{lr:.6f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    @torch.no_grad()
    def validate(self):
        """
        Validate the model
        
        Returns:
            avg_loss: Average validation loss
        """
        if self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0
        
        pbar = tqdm(self.val_loader, desc="Validation")
        
        for markets in pbar:
            markets = markets.to(self.device)
            
            # Generate two views
            view1, view2 = self.augmentation.get_two_views(markets)
            
            if self.scaler is not None:
                with autocast():
                    logits, labels, _ = self.model(view1, view2)
                    loss = self.criterion(logits, labels)
            else:
                logits, labels, _ = self.model(view1, view2)
                loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def save_checkpoint(self, epoch, is_best=False):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_losses[-1] if self.train_losses else None,
            'val_loss': self.val_losses[-1] if self.val_losses else None,
            'config': self.config.__dict__,
            'stats': self.stats
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir, 
            f'checkpoint_epoch_{epoch}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(
                self.config.checkpoint_dir,
                'best_model.pt'
            )
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best model (epoch {epoch})")
    
    def save_stats(self):
        """Save training statistics to JSON"""
        stats_path = os.path.join(self.config.log_dir, 'training_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def train(self):
        """
        Full training loop
        
        Returns:
            model: Trained model
        """
        print("\n" + "="*60)
        print("Starting MoCo Training")
        print("="*60)
        print(f"Total epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Train batches: {len(self.train_loader)}")
        if self.val_loader:
            print(f"Val batches: {len(self.val_loader)}")
        else:
            print(f"Val batches: 0 (no validation)")
        print(f"Early stopping patience: {self.config.early_stopping_patience}")
        print("="*60 + "\n")
        
        for epoch in range(self.config.num_epochs):
            epoch_start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            if self.val_loader is not None:
                val_loss = self.validate()
                self.val_losses.append(val_loss)
                
                # Check for best model
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                print(f"\nEpoch {epoch}/{self.config.num_epochs}:")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f} {'(BEST)' if is_best else ''}")
                print(f"  Learning Rate: {self.get_lr(epoch):.6f}")
                print(f"  Patience: {self.patience_counter}/{self.config.early_stopping_patience}")
            else:
                # No validation - track best train loss
                is_best = train_loss < self.best_train_loss
                if is_best:
                    self.best_train_loss = train_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                print(f"\nEpoch {epoch}/{self.config.num_epochs}:")
                print(f"  Train Loss: {train_loss:.4f} {'(BEST)' if is_best else ''}")
                print(f"  Learning Rate: {self.get_lr(epoch):.6f}")
                print(f"  Patience: {self.patience_counter}/{self.config.early_stopping_patience}")
            
            epoch_time = time.time() - epoch_start_time
            print(f"  Epoch Time: {epoch_time:.2f}s")
            
            # Update stats
            self.stats['train_loss'].append(train_loss)
            self.stats['val_loss'].append(val_loss if self.val_loader else None)
            self.stats['learning_rate'].append(self.get_lr(epoch))
            self.stats['epoch_time'].append(epoch_time)
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, is_best=is_best)
            
            # Save stats
            self.save_stats()
            
            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"\n{'='*60}")
                print(f"Early stopping triggered at epoch {epoch}")
                print(f"Best {'val' if self.val_loader else 'train'} loss: "
                      f"{self.best_val_loss if self.val_loader else self.best_train_loss:.4f}")
                print(f"{'='*60}\n")
                break
        
        print("\n" + "="*60)
        print("Training Completed!")
        print(f"Best {'val' if self.val_loader else 'train'} loss: "
              f"{self.best_val_loss if self.val_loader else self.best_train_loss:.4f}")
        print("="*60 + "\n")
        
        # Load best model
        best_checkpoint = torch.load(
            os.path.join(self.config.checkpoint_dir, 'best_model.pt')
        )
        self.model.load_state_dict(best_checkpoint['model_state_dict'])
        
        return self.model


def load_trained_model(checkpoint_path, device='cuda'):
    """
    Load a trained MoCo model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        model: Loaded model
        config: Configuration used for training
    """
    from model import MoCo
    from config import MoCoConfig
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct config
    config = MoCoConfig()
    for key, value in checkpoint['config'].items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Create model
    model = MoCo(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"\nModel loaded from {checkpoint_path}")
    print(f"Trained for {checkpoint['epoch']} epochs")
    if checkpoint['val_loss']:
        print(f"Val loss: {checkpoint['val_loss']:.4f}")
    else:
        print(f"Train loss: {checkpoint['train_loss']:.4f}")
    
    return model, config


# Example usage
if __name__ == "__main__":
    from config import MoCoConfig
    from model import MoCo
    from augmentation import MarketDataAugmentation
    import torch.utils.data as data
    
    # Config
    config = MoCoConfig()
    config.update(num_epochs=5, batch_size=4)  # Small for testing
    
    # Dummy data
    train_data = torch.randn(100, config.input_dim)
    val_data = torch.randn(20, config.input_dim)
    
    train_loader = data.DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = data.DataLoader(val_data, batch_size=config.batch_size)
    
    # Model
    model = MoCo(config)
    
    # Augmentation
    augmentation = MarketDataAugmentation(config)
    
    # Trainer
    trainer = MoCoTrainer(model, train_loader, val_loader, config, augmentation)
    
    # Train
    trained_model = trainer.train()