"""
Inference module for trained MoCo model
Handles embedding generation, similarity search, and market analysis
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import os


class MarketEmbeddingInference:
    """
    Inference engine for market embeddings
    Provides similarity search, clustering, and analysis capabilities
    """
    
    def __init__(self, model, preprocessor, device='cuda'):
        """
        Args:
            model: Trained MoCo model
            preprocessor: MarketDataPreprocessor with fitted scaler
            device: Device to run inference on
        """
        self.model = model.to(device)
        self.model.eval()
        self.preprocessor = preprocessor
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Cache for embeddings
        self.embeddings_cache = None
        self.market_ids = preprocessor.market_ids
        
        print(f"Inference engine initialized on {self.device}")
    
    @torch.no_grad()
    def get_embeddings(self, data_tensor):
        """
        Get embeddings for a batch of markets
        
        Args:
            data_tensor: Tensor of shape [N, input_dim]
        
        Returns:
            embeddings: Tensor of shape [N, embedding_dim]
        """
        self.model.eval()
        
        # Move to device
        if not data_tensor.is_cuda:
            data_tensor = data_tensor.to(self.device)
        
        # Get embeddings
        embeddings = self.model.get_embedding(data_tensor)
        
        return embeddings
    
    def build_embeddings_database(self, data_tensor, batch_size=256):
        """
        Build embeddings for entire dataset
        
        Args:
            data_tensor: Full dataset tensor [N, input_dim]
            batch_size: Batch size for inference
        
        Returns:
            embeddings: Tensor of shape [N, embedding_dim]
        """
        print(f"\nBuilding embeddings database for {len(data_tensor)} markets...")
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(data_tensor), batch_size):
            batch = data_tensor[i:i+batch_size]
            embeddings = self.get_embeddings(batch)
            all_embeddings.append(embeddings.cpu())
        
        # Concatenate
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        # Cache
        self.embeddings_cache = all_embeddings
        
        print(f"Embeddings database built: {all_embeddings.shape}")
        
        return all_embeddings
    
    def find_similar_markets(
        self, 
        query_embedding, 
        top_k=10, 
        return_scores=True
    ):
        """
        Find most similar markets to a query embedding
        
        Args:
            query_embedding: Query embedding [1, embedding_dim] or [embedding_dim]
            top_k: Number of similar markets to return
            return_scores: Whether to return similarity scores
        
        Returns:
            similar_markets: List of (market_id, similarity_score) or just market_ids
        """
        if self.embeddings_cache is None:
            raise ValueError("Embeddings database not built. Call build_embeddings_database first.")
        
        # Ensure query is 2D
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)
        
        # Move to device if needed
        if not query_embedding.is_cuda:
            query_embedding = query_embedding.to(self.device)
        
        embeddings_db = self.embeddings_cache.to(self.device)
        
        # Compute cosine similarities
        similarities = torch.mm(query_embedding, embeddings_db.T).squeeze()
        
        # Get top-k
        top_k_values, top_k_indices = torch.topk(similarities, k=min(top_k, len(similarities)))
        
        # Convert to list
        results = []
        for idx, score in zip(top_k_indices.cpu().numpy(), top_k_values.cpu().numpy()):
            market_id = self.market_ids[idx]
            if return_scores:
                results.append((market_id, float(score)))
            else:
                results.append(market_id)
        
        return results
    
    def get_embedding_for_new_market(self, market_data_raw):
        """
        Get embedding for a new market (handles preprocessing)
        
        Args:
            market_data_raw: Raw market data [input_dim] (before normalization)
        
        Returns:
            embedding: [embedding_dim] tensor
        """
        # Normalize using fitted scaler
        market_data_normalized = self.preprocessor.scaler.transform(
            market_data_raw.reshape(1, -1)
        )
        
        # Convert to tensor
        market_tensor = torch.FloatTensor(market_data_normalized)
        
        # Get embedding
        embedding = self.get_embeddings(market_tensor).squeeze()
        
        return embedding
    
    def compare_markets(self, market_id_1, market_id_2):
        """
        Compare two markets by their IDs
        
        Args:
            market_id_1: First market ID
            market_id_2: Second market ID
        
        Returns:
            similarity: Similarity score between 0 and 1
        """
        if self.embeddings_cache is None:
            raise ValueError("Embeddings database not built.")
        
        # Get indices
        idx_1 = self.market_ids.index(market_id_1)
        idx_2 = self.market_ids.index(market_id_2)
        
        # Get embeddings
        emb_1 = self.embeddings_cache[idx_1]
        emb_2 = self.embeddings_cache[idx_2]
        
        # Compute similarity
        similarity = torch.dot(emb_1, emb_2).item()
        
        return similarity
    
    def get_market_neighborhood(self, market_id, radius=0.8, max_neighbors=50):
        """
        Get all markets within a similarity radius
        
        Args:
            market_id: Query market ID
            radius: Similarity threshold (0 to 1)
            max_neighbors: Maximum number of neighbors to return
        
        Returns:
            neighbors: List of (market_id, similarity) within radius
        """
        idx = self.market_ids.index(market_id)
        query_embedding = self.embeddings_cache[idx]
        
        # Find similar markets
        all_similar = self.find_similar_markets(
            query_embedding, 
            top_k=max_neighbors,
            return_scores=True
        )
        
        # Filter by radius
        neighbors = [(mid, score) for mid, score in all_similar if score >= radius and mid != market_id]
        
        return neighbors
    
    def analyze_temporal_evolution(self, country_code, freq='A'):
        """
        Analyze how a country's economy evolved over time
        
        Args:
            country_code: Country code (e.g., 'USA', 'NLD')
            freq: Frequency ('A' for annual)
        
        Returns:
            evolution: DataFrame with year, embedding, and changes
        """
        if self.embeddings_cache is None:
            raise ValueError("Embeddings database not built.")
        
        # Find all markets for this country
        country_markets = [
            (mid, idx) for idx, mid in enumerate(self.market_ids) 
            if mid.startswith(f"{country_code}_{freq}_")
        ]
        
        if not country_markets:
            return None
        
        # Sort by year
        country_markets.sort(key=lambda x: x[0])
        
        # Extract embeddings and compute changes
        results = []
        prev_emb = None
        
        for market_id, idx in country_markets:
            # Extract year
            year = market_id.split('_')[-1]
            
            # Get embedding
            emb = self.embeddings_cache[idx]
            
            # Compute change from previous year
            change = None
            if prev_emb is not None:
                change = 1 - torch.dot(emb, prev_emb).item()  # 1 - similarity = change
            
            results.append({
                'year': year,
                'market_id': market_id,
                'embedding': emb.numpy(),
                'change_from_prev': change
            })
            
            prev_emb = emb
        
        return pd.DataFrame(results)
    
    def find_anomalies(self, threshold=0.5):
        """
        Find anomalous markets (far from all other markets)
        
        Args:
            threshold: Markets with max similarity < threshold are anomalies
        
        Returns:
            anomalies: List of (market_id, max_similarity)
        """
        if self.embeddings_cache is None:
            raise ValueError("Embeddings database not built.")
        
        print("Finding anomalous markets...")
        
        embeddings = self.embeddings_cache.to(self.device)
        
        # Compute pairwise similarities
        similarity_matrix = torch.mm(embeddings, embeddings.T)
        
        # For each market, find max similarity (excluding self)
        similarity_matrix.fill_diagonal_(-1)  # Ignore self-similarity
        max_similarities = similarity_matrix.max(dim=1)[0]
        
        # Find anomalies
        anomaly_indices = (max_similarities < threshold).nonzero().squeeze()
        
        anomalies = []
        for idx in anomaly_indices.cpu().numpy():
            market_id = self.market_ids[idx]
            max_sim = max_similarities[idx].item()
            anomalies.append((market_id, max_sim))
        
        anomalies.sort(key=lambda x: x[1])  # Sort by similarity (lowest first)
        
        print(f"Found {len(anomalies)} anomalies")
        
        return anomalies
    
    def cluster_markets(self, n_clusters=10, method='kmeans'):
        """
        Cluster markets based on embeddings
        
        Args:
            n_clusters: Number of clusters
            method: Clustering method ('kmeans' or 'hierarchical')
        
        Returns:
            clusters: Dict mapping cluster_id to list of market_ids
        """
        if self.embeddings_cache is None:
            raise ValueError("Embeddings database not built.")
        
        print(f"Clustering {len(self.market_ids)} markets into {n_clusters} clusters...")
        
        embeddings_np = self.embeddings_cache.cpu().numpy()
        
        if method == 'kmeans':
            from sklearn.cluster import KMeans
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            labels = clusterer.fit_predict(embeddings_np)
        elif method == 'hierarchical':
            from sklearn.cluster import AgglomerativeClustering
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clusterer.fit_predict(embeddings_np)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Group by cluster
        clusters = {}
        for idx, label in enumerate(labels):
            label = int(label)
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(self.market_ids[idx])
        
        print(f"Clustering complete. Cluster sizes: {[len(v) for v in clusters.values()]}")
        
        return clusters
    
    def save_embeddings(self, save_path):
        """Save embeddings database to disk"""
        if self.embeddings_cache is None:
            raise ValueError("No embeddings to save")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        torch.save({
            'embeddings': self.embeddings_cache,
            'market_ids': self.market_ids
        }, save_path)
        
        print(f"Embeddings saved to {save_path}")
    
    def load_embeddings(self, load_path):
        """Load embeddings database from disk"""
        data = torch.load(load_path)
        self.embeddings_cache = data['embeddings']
        self.market_ids = data['market_ids']
        
        print(f"Embeddings loaded from {load_path}")
        print(f"  - {len(self.market_ids)} markets")
        print(f"  - {self.embeddings_cache.shape[1]}-dimensional embeddings")


# Example usage
if __name__ == "__main__":
    print("This is an inference module. Import and use with a trained model.")
    print("\nExample usage:")
    print("""
    from inference import MarketEmbeddingInference
    from trainer import load_trained_model
    from data_preprocessing import MarketDataPreprocessor
    
    # Load model
    model, config = load_trained_model('checkpoints/best_model.pt')
    
    # Load preprocessor
    preprocessor = MarketDataPreprocessor(config)
    preprocessor.load_preprocessor('preprocessor.pkl')
    
    # Create inference engine
    inference = MarketEmbeddingInference(model, preprocessor)
    
    # Build database
    inference.build_embeddings_database(data_tensor)
    
    # Find similar markets
    similar = inference.find_similar_markets(query_embedding, top_k=10)
    """)