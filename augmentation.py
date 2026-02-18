"""
Data augmentation module for market indicator data
Implements masking, noise injection, and temporal/regional mixing
"""

import torch
import numpy as np


class MarketDataAugmentation:
    """
    Augmentation strategies for market indicator data
    Focus on non-zero elements only (preserve missing data structure)
    """
    
    def __init__(self, config, temporal_neighbors=None):
        """
        Args:
            config: Configuration object
            temporal_neighbors: Dict mapping index to list of temporal neighbor indices
        """
        self.mask_prob = config.mask_prob
        self.noise_std = config.noise_std
        self.temporal_prob = config.temporal_prob
        self.regional_prob = config.regional_prob
        self.temporal_neighbors = temporal_neighbors if temporal_neighbors else {}
    
    def mask_nonzero_elements(self, x):
        """
        Mask a random subset of non-zero elements
        
        Args:
            x: Tensor [batch_size, input_dim]
        
        Returns:
            x_masked: Tensor with some non-zero elements masked to 0
        """
        x_masked = x.clone()
        
        # Find non-zero elements
        non_zero_mask = (x != 0.0)
        
        # Create random mask for non-zero elements
        # Keep (1 - mask_prob) of non-zero elements
        keep_mask = torch.bernoulli(
            torch.ones_like(x) * (1 - self.mask_prob)
        ).bool()
        
        # Only mask non-zero elements
        final_mask = non_zero_mask & keep_mask
        
        # Apply mask
        x_masked = x * final_mask.float()
        
        return x_masked
    
    def add_gaussian_noise(self, x):
        """
        Add Gaussian noise to non-zero elements only
        
        Args:
            x: Tensor [batch_size, input_dim]
        
        Returns:
            x_noisy: Tensor with noise added to non-zero elements
        """
        x_noisy = x.clone()
        
        # Find non-zero elements
        non_zero_mask = (x != 0.0)
        
        # Generate noise
        noise = torch.randn_like(x) * self.noise_std
        
        # Apply noise only to non-zero elements
        x_noisy = x + noise * non_zero_mask.float()
        
        return x_noisy
    
    def temporal_mixup(self, x, indices):
        """
        Mix with temporal neighbors (same country, different year)
        
        Args:
            x: Tensor [batch_size, input_dim]
            indices: Original indices of samples in the dataset
        
        Returns:
            x_mixed: Tensor mixed with temporal neighbors
        """
        x_mixed = x.clone()
        batch_size = x.shape[0]
        
        for i in range(batch_size):
            # Check if this sample has temporal neighbors
            idx = indices[i].item() if torch.is_tensor(indices[i]) else indices[i]
            
            if idx in self.temporal_neighbors and len(self.temporal_neighbors[idx]) > 0:
                # Randomly select a temporal neighbor
                neighbor_idx = np.random.choice(self.temporal_neighbors[idx])
                
                # Mix with lambda from Beta distribution
                lambda_mix = np.random.beta(0.7, 0.3)  # Favor original (0.7)
                
                # Note: In practice, you'd need access to the full dataset here
                # For now, we'll skip this and implement it in the training loop
                # This is a placeholder
                pass
        
        return x_mixed
    
    def augment(self, x, apply_noise=True, apply_mask=True):
        """
        Apply augmentation pipeline
        
        Args:
            x: Tensor [batch_size, input_dim]
            apply_noise: Whether to apply Gaussian noise
            apply_mask: Whether to apply masking
        
        Returns:
            x_aug: Augmented tensor
        """
        x_aug = x.clone()
        
        # Apply masking
        if apply_mask:
            x_aug = self.mask_nonzero_elements(x_aug)
        
        # Apply noise
        if apply_noise:
            x_aug = self.add_gaussian_noise(x_aug)
        
        return x_aug
    
    def get_two_views(self, x):
        """
        Generate two different augmented views of the same data
        
        Args:
            x: Tensor [batch_size, input_dim]
        
        Returns:
            view1: First augmented view
            view2: Second augmented view
        """
        # View 1: Mask + Noise
        view1 = self.mask_nonzero_elements(x)
        view1 = self.add_gaussian_noise(view1)
        
        # View 2: Different mask + Different noise
        view2 = self.mask_nonzero_elements(x)
        view2 = self.add_gaussian_noise(view2)
        
        return view1, view2


class PositivePairGenerator:
    """
    Generate positive pairs for contrastive learning
    Handles self-augmentation, temporal, and regional positives
    """
    
    def __init__(self, config, temporal_neighbors=None):
        """
        Args:
            config: Configuration object
            temporal_neighbors: Dict mapping index to list of temporal neighbor indices
        """
        self.config = config
        self.temporal_neighbors = temporal_neighbors if temporal_neighbors else {}
        self.temporal_prob = config.temporal_prob
        self.augmentation = MarketDataAugmentation(config, temporal_neighbors)
    
    def generate_positive_pair(self, x, indices=None, full_dataset=None):
        """
        Generate a positive pair for the given batch
        
        Strategy:
        - With prob (1 - temporal_prob): Self-augmentation (same market, different augmentations)
        - With prob temporal_prob: Temporal neighbor (if available)
        
        Args:
            x: Tensor [batch_size, input_dim]
            indices: Original indices of samples in the dataset
            full_dataset: Full dataset tensor (for temporal neighbors)
        
        Returns:
            x_query: Query view
            x_key: Key view (positive)
        """
        batch_size = x.shape[0]
        
        # Decide positive pair strategy
        use_temporal = (
            np.random.rand() < self.temporal_prob and 
            indices is not None and 
            full_dataset is not None
        )
        
        if use_temporal:
            # Try to use temporal neighbors
            x_key = self._get_temporal_positives(x, indices, full_dataset)
        else:
            # Use self-augmentation
            x_key = x.clone()
        
        # Generate two augmented views
        x_query = self.augmentation.augment(x)
        x_key = self.augmentation.augment(x_key)
        
        return x_query, x_key
    
    def _get_temporal_positives(self, x, indices, full_dataset):
        """
        Get temporal neighbors as positive samples
        
        Args:
            x: Original batch [batch_size, input_dim]
            indices: Original indices
            full_dataset: Full dataset tensor
        
        Returns:
            x_temporal: Batch with temporal neighbors (where available)
        """
        x_temporal = x.clone()
        batch_size = x.shape[0]
        
        for i in range(batch_size):
            idx = indices[i].item() if torch.is_tensor(indices[i]) else indices[i]
            
            # Check if temporal neighbor exists
            if idx in self.temporal_neighbors and len(self.temporal_neighbors[idx]) > 0:
                # Randomly select one temporal neighbor
                neighbor_idx = np.random.choice(self.temporal_neighbors[idx])
                x_temporal[i] = full_dataset[neighbor_idx]
        
        return x_temporal


# Example usage
if __name__ == "__main__":
    from config import MoCoConfig
    
    config = MoCoConfig()
    
    # Create dummy data
    batch_size = 4
    input_dim = 100
    
    # Simulate data with some zeros
    x = torch.randn(batch_size, input_dim)
    x[x < 0] = 0  # Half elements are zero
    
    print(f"Original data shape: {x.shape}")
    print(f"Non-zero elements: {(x != 0).sum().item()}/{x.numel()}")
    
    # Test augmentation
    augmenter = MarketDataAugmentation(config)
    
    # Test masking
    x_masked = augmenter.mask_nonzero_elements(x)
    print(f"\nAfter masking:")
    print(f"Non-zero elements: {(x_masked != 0).sum().item()}/{x_masked.numel()}")
    
    # Test noise
    x_noisy = augmenter.add_gaussian_noise(x)
    print(f"\nAfter adding noise:")
    print(f"Non-zero elements: {(x_noisy != 0).sum().item()}/{x_noisy.numel()}")
    
    # Test two views
    view1, view2 = augmenter.get_two_views(x)
    print(f"\nTwo views generated:")
    print(f"View 1 non-zeros: {(view1 != 0).sum().item()}")
    print(f"View 2 non-zeros: {(view2 != 0).sum().item()}")
    print(f"Views are different: {not torch.equal(view1, view2)}")