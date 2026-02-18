"""
MoCo model architecture
Includes Encoder, Projection Head, and MoCo framework with queue
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class Encoder(nn.Module):
    """
    Main encoder network that compresses input to embedding space
    """
    
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.3):
        """
        Args:
            input_dim: Input dimension (16197 for your case)
            hidden_dims: List of hidden layer dimensions [2048, 1024, 512, 256]
            output_dim: Final embedding dimension (128)
            dropout: Dropout probability
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final embedding layer (no dropout, no activation)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Xavier initialization for linear layers"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, input_dim]
        
        Returns:
            embeddings: [batch_size, output_dim]
        """
        return self.network(x)


class ProjectionHead(nn.Module):
    """
    MLP projection head for mapping embeddings to contrastive space
    Used only during training
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Args:
            input_dim: Embedding dimension (128)
            hidden_dim: Hidden layer dimension (64)
            output_dim: Projection dimension (32)
        """
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: Embeddings [batch_size, input_dim]
        
        Returns:
            projections: [batch_size, output_dim]
        """
        return self.net(x)


class MoCo(nn.Module):
    """
    Momentum Contrast (MoCo) framework
    
    Key components:
    1. Query encoder (trainable)
    2. Key encoder (momentum updated)
    3. Queue of negative samples
    4. Projection heads for both encoders
    """
    
    def __init__(self, config):
        """
        Args:
            config: Configuration object with all hyperparameters
        """
        super().__init__()
        
        # Store config
        self.config = config
        
        # Hyperparameters
        self.queue_size = config.queue_size
        self.momentum = config.momentum
        self.temperature = config.temperature
        self.embedding_dim = config.embedding_dim
        self.projection_dim = config.projection_dim
        
        # ==================== Query Encoder (Trainable) ====================
        self.encoder_q = Encoder(
            input_dim=config.input_dim,
            hidden_dims=config.encoder_hidden_dims,
            output_dim=config.embedding_dim,
            dropout=config.dropout
        )
        
        self.projection_q = ProjectionHead(
            input_dim=config.embedding_dim,
            hidden_dim=64,
            output_dim=config.projection_dim
        )
        
        # ==================== Key Encoder (Momentum) ====================
        self.encoder_k = Encoder(
            input_dim=config.input_dim,
            hidden_dims=config.encoder_hidden_dims,
            output_dim=config.embedding_dim,
            dropout=config.dropout
        )
        
        self.projection_k = ProjectionHead(
            input_dim=config.embedding_dim,
            hidden_dim=64,
            output_dim=config.projection_dim
        )
        
        # Initialize key encoder with query encoder weights
        self._copy_params_from_q_to_k()
        
        # Key encoder should not compute gradients
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        for param in self.projection_k.parameters():
            param.requires_grad = False
        
        # ==================== Queue (Dictionary) ====================
        # Queue stores past key projections as negative samples
        self.register_buffer(
            "queue", 
            torch.randn(config.projection_dim, config.queue_size)
        )
        self.queue = F.normalize(self.queue, dim=0)
        
        # Queue pointer (tracks where to insert new keys)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        print(f"\nMoCo Model Initialized:")
        print(f"  - Encoder: {config.input_dim} → {config.encoder_hidden_dims} → {config.embedding_dim}")
        print(f"  - Projection: {config.embedding_dim} → {config.projection_dim}")
        print(f"  - Queue size: {config.queue_size}")
        print(f"  - Momentum: {config.momentum}")
        print(f"  - Temperature: {config.temperature}")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
    
    def _copy_params_from_q_to_k(self):
        """Initialize key encoder with query encoder parameters"""
        for param_q, param_k in zip(self.encoder_q.parameters(), 
                                     self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
        
        for param_q, param_k in zip(self.projection_q.parameters(), 
                                     self.projection_k.parameters()):
            param_k.data.copy_(param_q.data)
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of key encoder
        θ_k ← m * θ_k + (1 - m) * θ_q
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), 
                                     self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                          param_q.data * (1.0 - self.momentum)
        
        for param_q, param_k in zip(self.projection_q.parameters(), 
                                     self.projection_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                          param_q.data * (1.0 - self.momentum)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """
        Update queue: add new keys, remove oldest
        
        Args:
            keys: [batch_size, projection_dim] tensor of key projections
        """
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        
        # Check if queue is large enough
        assert self.queue_size % batch_size == 0, \
            f"Queue size {self.queue_size} should be divisible by batch size {batch_size}"
        
        # Replace oldest entries in queue
        self.queue[:, ptr:ptr + batch_size] = keys.T
        
        # Move pointer
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr
    
    def forward(self, x_q, x_k):
        """
        Forward pass for training
        
        Args:
            x_q: Query samples [batch_size, input_dim]
            x_k: Key samples [batch_size, input_dim] (augmented version)
        
        Returns:
            logits: [batch_size, 1 + queue_size]
            labels: [batch_size] (all zeros - positive is at index 0)
            emb_q: Query embeddings [batch_size, embedding_dim]
        """
        batch_size = x_q.shape[0]
        
        # ==================== Query Branch (Trainable) ====================
        emb_q = self.encoder_q(x_q)              # [batch, embedding_dim]
        q = self.projection_q(emb_q)             # [batch, projection_dim]
        q = F.normalize(q, dim=1)                # L2 normalize
        
        # ==================== Key Branch (Momentum) ====================
        with torch.no_grad():
            # Update key encoder
            self._momentum_update_key_encoder()
            
            emb_k = self.encoder_k(x_k)          # [batch, embedding_dim]
            k = self.projection_k(emb_k)         # [batch, projection_dim]
            k = F.normalize(k, dim=1)            # L2 normalize
        
        # ==================== Compute Logits ====================
        # Positive logits: [batch, 1]
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        
        # Negative logits: [batch, queue_size]
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        # Concatenate: [batch, 1 + queue_size]
        logits = torch.cat([l_pos, l_neg], dim=1)
        
        # Apply temperature
        logits /= self.temperature
        
        # Labels: positive is the first (index 0)
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        
        # ==================== Update Queue ====================
        self._dequeue_and_enqueue(k)
        
        return logits, labels, emb_q
    
    @torch.no_grad()
    def get_embedding(self, x):
        """
        Get embeddings for inference (use query encoder only)
        
        Args:
            x: [batch_size, input_dim]
        
        Returns:
            embeddings: [batch_size, embedding_dim] (L2 normalized)
        """
        self.eval()
        embeddings = self.encoder_q(x)
        embeddings = F.normalize(embeddings, dim=1)
        return embeddings


# Example usage
if __name__ == "__main__":
    from config import MoCoConfig
    
    config = MoCoConfig()
    
    # Create model
    model = MoCo(config)
    
    # Test forward pass
    batch_size = 8
    x_q = torch.randn(batch_size, config.input_dim)
    x_k = torch.randn(batch_size, config.input_dim)
    
    logits, labels, emb_q = model(x_q, x_k)
    
    print(f"\nForward pass test:")
    print(f"  - Logits shape: {logits.shape}")
    print(f"  - Labels shape: {labels.shape}")
    print(f"  - Embeddings shape: {emb_q.shape}")
    
    # Test inference
    embeddings = model.get_embedding(x_q)
    print(f"\nInference test:")
    print(f"  - Embeddings shape: {embeddings.shape}")
    print(f"  - Embeddings normalized: {torch.allclose(torch.norm(embeddings, dim=1), torch.ones(batch_size))}")