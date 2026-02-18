"""
Utility functions for MoCo Market Embeddings
Common operations and helper functions
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os


def visualize_embeddings(embeddings, market_ids, save_path=None, n_samples=1000):
    """
    Visualize embeddings using t-SNE
    
    Args:
        embeddings: Tensor of shape [N, embedding_dim]
        market_ids: List of market IDs
        save_path: Path to save plot (optional)
        n_samples: Number of samples to plot (for performance)
    """
    print(f"Visualizing embeddings using t-SNE...")
    
    # Sample if too many
    if len(embeddings) > n_samples:
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        embeddings_plot = embeddings[indices].cpu().numpy()
        market_ids_plot = [market_ids[i] for i in indices]
    else:
        embeddings_plot = embeddings.cpu().numpy()
        market_ids_plot = market_ids
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings_plot)
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    # Extract country codes for coloring
    countries = [mid.split('_')[0] for mid in market_ids_plot]
    unique_countries = list(set(countries))
    color_map = {country: i for i, country in enumerate(unique_countries)}
    colors = [color_map[c] for c in countries]
    
    scatter = plt.scatter(
        embeddings_2d[:, 0], 
        embeddings_2d[:, 1],
        c=colors,
        alpha=0.6,
        s=20,
        cmap='tab20'
    )
    
    plt.title('Market Embeddings Visualization (t-SNE)', fontsize=16)
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_training_curves(stats_path, save_path=None):
    """
    Plot training curves from stats JSON
    
    Args:
        stats_path: Path to training_stats.json
        save_path: Path to save plot (optional)
    """
    import json
    
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training loss
    axes[0, 0].plot(stats['train_loss'], label='Train Loss', linewidth=2)
    if stats['val_loss'][0] is not None:
        axes[0, 0].plot(stats['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Learning rate
    axes[0, 1].plot(stats['learning_rate'], linewidth=2, color='orange')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Learning Rate')
    axes[0, 1].set_title('Learning Rate Schedule')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Epoch time
    axes[1, 0].plot(stats['epoch_time'], linewidth=2, color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].set_title('Time per Epoch')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss zoom (last 50 epochs)
    if len(stats['train_loss']) > 50:
        axes[1, 1].plot(stats['train_loss'][-50:], label='Train Loss', linewidth=2)
        if stats['val_loss'][0] is not None:
            axes[1, 1].plot(stats['val_loss'][-50:], label='Val Loss', linewidth=2)
        axes[1, 1].set_xlabel('Epoch (last 50)')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Loss (Last 50 Epochs)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.show()


def export_embeddings_to_csv(embeddings, market_ids, save_path):
    """
    Export embeddings to CSV for external analysis
    
    Args:
        embeddings: Tensor of shape [N, embedding_dim]
        market_ids: List of market IDs
        save_path: Path to save CSV
    """
    import pandas as pd
    
    embeddings_np = embeddings.cpu().numpy()
    
    # Create DataFrame
    columns = ['market_id'] + [f'dim_{i}' for i in range(embeddings_np.shape[1])]
    data = np.column_stack([market_ids, embeddings_np])
    
    df = pd.DataFrame(data, columns=columns)
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    
    print(f"Embeddings exported to {save_path}")
    print(f"  - Shape: {df.shape}")


def compute_embedding_statistics(embeddings):
    """
    Compute statistics about embeddings
    
    Args:
        embeddings: Tensor of shape [N, embedding_dim]
    
    Returns:
        stats: Dictionary of statistics
    """
    embeddings_np = embeddings.cpu().numpy()
    
    # Compute pairwise similarities
    embeddings_norm = embeddings / embeddings.norm(dim=1, keepdim=True)
    similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.T).cpu().numpy()
    
    # Remove diagonal
    np.fill_diagonal(similarity_matrix, np.nan)
    
    stats = {
        'embedding_dim': embeddings.shape[1],
        'num_markets': embeddings.shape[0],
        'mean_norm': float(embeddings.norm(dim=1).mean()),
        'std_norm': float(embeddings.norm(dim=1).std()),
        'mean_similarity': float(np.nanmean(similarity_matrix)),
        'std_similarity': float(np.nanstd(similarity_matrix)),
        'min_similarity': float(np.nanmin(similarity_matrix)),
        'max_similarity': float(np.nanmax(similarity_matrix)),
    }
    
    return stats


def print_embedding_statistics(stats):
    """Pretty print embedding statistics"""
    print("\n" + "="*60)
    print("Embedding Statistics")
    print("="*60)
    print(f"Number of markets: {stats['num_markets']}")
    print(f"Embedding dimension: {stats['embedding_dim']}")
    print(f"\nNorm statistics:")
    print(f"  Mean: {stats['mean_norm']:.4f}")
    print(f"  Std:  {stats['std_norm']:.4f}")
    print(f"\nPairwise similarity statistics:")
    print(f"  Mean: {stats['mean_similarity']:.4f}")
    print(f"  Std:  {stats['std_similarity']:.4f}")
    print(f"  Min:  {stats['min_similarity']:.4f}")
    print(f"  Max:  {stats['max_similarity']:.4f}")
    print("="*60 + "\n")


def find_market_by_pattern(market_ids, pattern):
    """
    Find markets matching a pattern
    
    Args:
        market_ids: List of market IDs
        pattern: String pattern (e.g., 'USA', '2020', 'USA_A')
    
    Returns:
        matches: List of matching market IDs
    """
    matches = [mid for mid in market_ids if pattern in mid]
    return sorted(matches)


def batch_similarity_search(inference, query_market_ids, top_k=10):
    """
    Find similar markets for multiple queries at once
    
    Args:
        inference: MarketEmbeddingInference object
        query_market_ids: List of market IDs to query
        top_k: Number of similar markets per query
    
    Returns:
        results: Dict mapping query_id to list of similar markets
    """
    results = {}
    
    for query_id in query_market_ids:
        if query_id not in inference.market_ids:
            print(f"Warning: {query_id} not found in database")
            continue
        
        idx = inference.market_ids.index(query_id)
        query_embedding = inference.embeddings_cache[idx]
        
        similar = inference.find_similar_markets(
            query_embedding,
            top_k=top_k + 1,  # +1 to exclude self
            return_scores=True
        )
        
        # Exclude self
        results[query_id] = [s for s in similar if s[0] != query_id][:top_k]
    
    return results


def create_similarity_matrix(inference, market_ids_subset):
    """
    Create a similarity matrix for a subset of markets
    
    Args:
        inference: MarketEmbeddingInference object
        market_ids_subset: List of market IDs
    
    Returns:
        similarity_matrix: Numpy array of shape [N, N]
    """
    # Get indices
    indices = [inference.market_ids.index(mid) for mid in market_ids_subset]
    
    # Get embeddings
    embeddings = inference.embeddings_cache[indices]
    
    # Compute similarity matrix
    embeddings_norm = embeddings / embeddings.norm(dim=1, keepdim=True)
    similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.T).cpu().numpy()
    
    return similarity_matrix


def plot_similarity_matrix(similarity_matrix, market_ids, save_path=None):
    """
    Plot a similarity matrix as a heatmap
    
    Args:
        similarity_matrix: Numpy array of shape [N, N]
        market_ids: List of market IDs (for labels)
        save_path: Path to save plot (optional)
    """
    plt.figure(figsize=(10, 8))
    
    im = plt.imshow(similarity_matrix, cmap='RdYlGn', vmin=0, vmax=1)
    plt.colorbar(im, label='Similarity')
    
    # Set ticks
    plt.xticks(range(len(market_ids)), market_ids, rotation=90, fontsize=8)
    plt.yticks(range(len(market_ids)), market_ids, fontsize=8)
    
    plt.title('Market Similarity Matrix', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Similarity matrix saved to {save_path}")
    
    plt.show()


# Example usage
if __name__ == "__main__":
    print("Utility functions for MoCo Market Embeddings")
    print("\nAvailable functions:")
    print("  - visualize_embeddings()")
    print("  - plot_training_curves()")
    print("  - export_embeddings_to_csv()")
    print("  - compute_embedding_statistics()")
    print("  - find_market_by_pattern()")
    print("  - batch_similarity_search()")
    print("  - create_similarity_matrix()")
    print("  - plot_similarity_matrix()")