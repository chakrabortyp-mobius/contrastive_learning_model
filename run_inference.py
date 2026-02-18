"""
Inference script for using trained MoCo model
Provides easy-to-use interface for market similarity search and analysis
"""

import torch
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import MarketEmbeddingInference
from trainer import load_trained_model
from data_preprocessing import MarketDataPreprocessor
from config import MoCoConfig
import pandas as pd


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run inference with trained MoCo model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--preprocessor', type=str, required=True,
                       help='Path to saved preprocessor')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to data file')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    # Operation mode
    parser.add_argument('--mode', type=str, required=True,
                       choices=['similar', 'compare', 'cluster', 'anomaly', 'evolution', 'build_db'],
                       help='Inference mode')
    
    # Mode-specific arguments
    parser.add_argument('--market_id', type=str,
                       help='Market ID for similarity search or evolution analysis')
    parser.add_argument('--market_id_2', type=str,
                       help='Second market ID for comparison')
    parser.add_argument('--top_k', type=int, default=10,
                       help='Number of similar markets to return')
    parser.add_argument('--n_clusters', type=int, default=10,
                       help='Number of clusters')
    parser.add_argument('--country', type=str,
                       help='Country code for evolution analysis')
    parser.add_argument('--embeddings_path', type=str,
                       default='/home/claude/moco_market/outputs/embeddings.pt',
                       help='Path to save/load embeddings database')
    
    return parser.parse_args()


def build_embeddings_database(args):
    """Build and save embeddings database"""
    print("\n" + "="*70)
    print("Building Embeddings Database")
    print("="*70)
    
    # Load model
    model, config = load_trained_model(args.checkpoint, args.device)
    
    # Load preprocessor
    config_obj = MoCoConfig()
    preprocessor = MarketDataPreprocessor(config_obj)
    preprocessor.load_preprocessor(args.preprocessor)
    
    # Load data
    import pyarrow.parquet as pq
    table = pq.read_table(args.data_path)
    df = table.to_pandas()
    
    # Preprocess
    df_filled = df.fillna(df.mean())
    data_normalized = preprocessor.scaler.transform(df_filled)
    data_tensor = torch.FloatTensor(data_normalized)
    
    # Create inference engine
    inference = MarketEmbeddingInference(model, preprocessor, args.device)
    
    # Build embeddings
    embeddings = inference.build_embeddings_database(data_tensor)
    
    # Save
    inference.save_embeddings(args.embeddings_path)
    
    print(f"\n✓ Embeddings database saved to {args.embeddings_path}")
    print(f"  - {len(embeddings)} markets")
    print(f"  - {embeddings.shape[1]}-dimensional embeddings")


def find_similar_markets(args):
    """Find similar markets"""
    print("\n" + "="*70)
    print(f"Finding markets similar to: {args.market_id}")
    print("="*70)
    
    # Load model
    model, config = load_trained_model(args.checkpoint, args.device)
    
    # Load preprocessor
    config_obj = MoCoConfig()
    preprocessor = MarketDataPreprocessor(config_obj)
    preprocessor.load_preprocessor(args.preprocessor)
    
    # Create inference engine
    inference = MarketEmbeddingInference(model, preprocessor, args.device)
    
    # Load embeddings
    inference.load_embeddings(args.embeddings_path)
    
    # Find market
    if args.market_id not in inference.market_ids:
        print(f"Error: Market ID '{args.market_id}' not found in database")
        return
    
    idx = inference.market_ids.index(args.market_id)
    query_embedding = inference.embeddings_cache[idx]
    
    # Find similar
    similar_markets = inference.find_similar_markets(
        query_embedding, 
        top_k=args.top_k + 1,  # +1 to exclude self
        return_scores=True
    )
    
    # Print results
    print(f"\nTop {args.top_k} similar markets to {args.market_id}:")
    print("-" * 70)
    
    for i, (market_id, score) in enumerate(similar_markets[1:], 1):  # Skip self
        print(f"{i:2d}. {market_id:30s} | Similarity: {score:.4f}")


def compare_two_markets(args):
    """Compare two markets"""
    print("\n" + "="*70)
    print(f"Comparing: {args.market_id} vs {args.market_id_2}")
    print("="*70)
    
    # Load model
    model, config = load_trained_model(args.checkpoint, args.device)
    
    # Load preprocessor
    config_obj = MoCoConfig()
    preprocessor = MarketDataPreprocessor(config_obj)
    preprocessor.load_preprocessor(args.preprocessor)
    
    # Create inference engine
    inference = MarketEmbeddingInference(model, preprocessor, args.device)
    
    # Load embeddings
    inference.load_embeddings(args.embeddings_path)
    
    # Compare
    similarity = inference.compare_markets(args.market_id, args.market_id_2)
    
    print(f"\nSimilarity Score: {similarity:.4f}")
    print(f"\nInterpretation:")
    if similarity > 0.9:
        print("  → Very similar markets (almost identical)")
    elif similarity > 0.7:
        print("  → Similar markets (comparable economic profiles)")
    elif similarity > 0.5:
        print("  → Moderately similar")
    elif similarity > 0.3:
        print("  → Somewhat different")
    else:
        print("  → Very different markets")


def cluster_markets(args):
    """Cluster markets"""
    print("\n" + "="*70)
    print(f"Clustering markets into {args.n_clusters} clusters")
    print("="*70)
    
    # Load model
    model, config = load_trained_model(args.checkpoint, args.device)
    
    # Load preprocessor
    config_obj = MoCoConfig()
    preprocessor = MarketDataPreprocessor(config_obj)
    preprocessor.load_preprocessor(args.preprocessor)
    
    # Create inference engine
    inference = MarketEmbeddingInference(model, preprocessor, args.device)
    
    # Load embeddings
    inference.load_embeddings(args.embeddings_path)
    
    # Cluster
    clusters = inference.cluster_markets(n_clusters=args.n_clusters)
    
    # Print results
    print("\nCluster Summary:")
    print("-" * 70)
    
    for cluster_id, markets in sorted(clusters.items()):
        print(f"\nCluster {cluster_id} ({len(markets)} markets):")
        print("  Sample markets:", markets[:5])


def find_anomalies(args):
    """Find anomalous markets"""
    print("\n" + "="*70)
    print("Finding Anomalous Markets")
    print("="*70)
    
    # Load model
    model, config = load_trained_model(args.checkpoint, args.device)
    
    # Load preprocessor
    config_obj = MoCoConfig()
    preprocessor = MarketDataPreprocessor(config_obj)
    preprocessor.load_preprocessor(args.preprocessor)
    
    # Create inference engine
    inference = MarketEmbeddingInference(model, preprocessor, args.device)
    
    # Load embeddings
    inference.load_embeddings(args.embeddings_path)
    
    # Find anomalies
    anomalies = inference.find_anomalies(threshold=0.5)
    
    # Print results
    print(f"\nFound {len(anomalies)} anomalous markets:")
    print("-" * 70)
    
    for i, (market_id, max_sim) in enumerate(anomalies[:20], 1):  # Show top 20
        print(f"{i:2d}. {market_id:30s} | Max Similarity: {max_sim:.4f}")


def analyze_evolution(args):
    """Analyze temporal evolution"""
    if not args.country:
        print("Error: --country required for evolution analysis")
        return
    
    print("\n" + "="*70)
    print(f"Analyzing Evolution: {args.country}")
    print("="*70)
    
    # Load model
    model, config = load_trained_model(args.checkpoint, args.device)
    
    # Load preprocessor
    config_obj = MoCoConfig()
    preprocessor = MarketDataPreprocessor(config_obj)
    preprocessor.load_preprocessor(args.preprocessor)
    
    # Create inference engine
    inference = MarketEmbeddingInference(model, preprocessor, args.device)
    
    # Load embeddings
    inference.load_embeddings(args.embeddings_path)
    
    # Analyze evolution
    evolution_df = inference.analyze_temporal_evolution(args.country)
    
    if evolution_df is None or len(evolution_df) == 0:
        print(f"\nNo data found for country: {args.country}")
        return
    
    # Print results
    print(f"\nTemporal Evolution for {args.country}:")
    print("-" * 70)
    print(evolution_df[['year', 'market_id', 'change_from_prev']].to_string(index=False))
    
    # Identify major changes
    if 'change_from_prev' in evolution_df.columns:
        major_changes = evolution_df[evolution_df['change_from_prev'] > 0.1].dropna()
        if len(major_changes) > 0:
            print("\nMajor Changes (>10% shift):")
            for _, row in major_changes.iterrows():
                print(f"  {row['year']}: {row['change_from_prev']:.2%} change")


def main():
    """Main inference function"""
    args = parse_args()
    
    if args.mode == 'build_db':
        build_embeddings_database(args)
    elif args.mode == 'similar':
        if not args.market_id:
            print("Error: --market_id required for similarity search")
            return
        find_similar_markets(args)
    elif args.mode == 'compare':
        if not args.market_id or not args.market_id_2:
            print("Error: --market_id and --market_id_2 required for comparison")
            return
        compare_two_markets(args)
    elif args.mode == 'cluster':
        cluster_markets(args)
    elif args.mode == 'anomaly':
        find_anomalies(args)
    elif args.mode == 'evolution':
        analyze_evolution(args)


if __name__ == "__main__":
    main()