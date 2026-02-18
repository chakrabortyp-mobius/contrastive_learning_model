"""
Data preprocessing module for market indicator data
Handles loading, normalization, and train/val splitting
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import pickle
import os


class MarketDataPreprocessor:
    """Handles data loading and preprocessing"""
    
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.market_ids = None
        self.data_normalized = None
        self.temporal_neighbors = {}  # For temporal positive pairs
        
    def load_data(self, data_path=None):
        """
        Load parquet file and extract data
        
        Args:
            data_path: Path to parquet file (optional, uses config if None)
        
        Returns:
            df: Pandas DataFrame
        """
        if data_path is None:
            data_path = self.config.data_path
        
        print(f"Loading data from {data_path}...")
        table = pq.read_table(data_path)
        df = table.to_pandas()
        
        print(f"Data shape: {df.shape}")
        print(f"Columns: {len(df.columns)}")
        print(f"Rows (markets): {len(df)}")
        
        # Store market IDs
        self.market_ids = df.index.tolist()
        
        return df
    
    def preprocess(self, df):
        """
        Preprocess the data using only row chunking
        """
        print("\nPreprocessing data...")
        
        # Step 1: Handle missing values (do this on full DataFrame but memory-efficient)
        print("  - Filling missing values...")
        
        # Calculate column means first (this is memory efficient)
        print("    Computing column means...")
        column_means = df.mean()
        
        # Fill missing values row by row chunk
        row_chunk_size = 5000  # Process 5000 rows at a time
        n_rows = len(df)
        filled_chunks = []
        
        for start_idx in range(0, n_rows, row_chunk_size):
            end_idx = min(start_idx + row_chunk_size, n_rows)
            print(f"    Filling rows {start_idx} to {end_idx}...")
            
            # Take a chunk of rows
            chunk = df.iloc[start_idx:end_idx]
            
            # Fill missing values using pre-computed column means
            chunk_filled = chunk.fillna(column_means)
            
            # Store filled chunk
            filled_chunks.append(chunk_filled)
            
            # Clear chunk from memory
            del chunk
        
        # Combine all filled chunks
        print("  - Combining filled chunks...")
        df_filled = pd.concat(filled_chunks, axis=0)
        del filled_chunks
        
        print(f"  - Filled data shape: {df_filled.shape}")
        
        # Step 2: Fit scaler on a sample (to avoid memory issues)
        print("  - Fitting scaler on sample...")
        sample_size = min(10000, len(df_filled))
        sample_indices = np.random.choice(len(df_filled), sample_size, replace=False)
        sample_df = df_filled.iloc[sample_indices]
        
        self.scaler.fit(sample_df)
        print(f"    Scaler fitted on {sample_size} samples")
        
        # Step 3: Normalize using row chunking
        print("  - Normalizing data in row chunks...")
        normalized_chunks = []
        
        for start_idx in range(0, n_rows, row_chunk_size):
            end_idx = min(start_idx + row_chunk_size, n_rows)
            print(f"    Normalizing rows {start_idx} to {end_idx}...")
            
            # Take chunk of filled data
            chunk = df_filled.iloc[start_idx:end_idx]
            
            # Normalize chunk
            chunk_normalized = self.scaler.transform(chunk)
            
            # Store normalized chunk
            normalized_chunks.append(chunk_normalized)
            
            # Clear chunk from memory
            del chunk
        
        # Combine normalized chunks
        print("  - Combining normalized chunks...")
        self.data_normalized = np.vstack(normalized_chunks)
        del normalized_chunks
        
        # Step 4: Build temporal neighbor mapping (uses only indices, not data)
        print("  - Building temporal neighbor mapping...")
        self._build_temporal_mapping()
        
        # Step 5: Convert to tensor (this will still need full dataset in memory)
        print("  - Converting to tensor...")
        data_tensor = torch.FloatTensor(self.data_normalized)
        
        print(f"  - Preprocessed data shape: {data_tensor.shape}")
        print(f"  - Data mean: {data_tensor.mean():.4f}, std: {data_tensor.std():.4f}")
        
        return data_tensor
    
    def _build_temporal_mapping(self):
        """
        Build mapping of market_id to temporal neighbors
        Format: market_id = COUNTRY_FREQ_YEAR (e.g., ABW_A_2000)
        """
        self.temporal_neighbors = {}
        
        # Parse market IDs
        parsed = []
        for market_id in self.market_ids:
            parts = market_id.split('_')
            if len(parts) >= 3:
                country = parts[0]
                freq = parts[1]
                year = parts[2]
                parsed.append({
                    'market_id': market_id,
                    'country': country,
                    'freq': freq,
                    'year': int(year) if year.isdigit() else -1
                })
        
        # Build index mapping
        market_id_to_idx = {mid: idx for idx, mid in enumerate(self.market_ids)}
        
        # For each market, find temporal neighbors (±1 year)
        for item in parsed:
            if item['year'] == -1:
                continue
            
            market_id = item['market_id']
            country = item['country']
            freq = item['freq']
            year = item['year']
            
            neighbors = []
            
            # Look for previous year
            prev_id = f"{country}_{freq}_{year-1}"
            if prev_id in market_id_to_idx:
                neighbors.append(market_id_to_idx[prev_id])
            
            # Look for next year
            next_id = f"{country}_{freq}_{year+1}"
            if next_id in market_id_to_idx:
                neighbors.append(market_id_to_idx[next_id])
            
            if neighbors:
                self.temporal_neighbors[market_id_to_idx[market_id]] = neighbors
        
        print(f"  - Found temporal neighbors for {len(self.temporal_neighbors)} markets")
    
    def split_data(self, data_tensor, train_split=None):
        """
        Split data into train and validation sets
        
        Args:
            data_tensor: Full dataset tensor
            train_split: Train split ratio (uses config if None)
        
        Returns:
            train_data: Training tensor
            val_data: Validation tensor (None if train_split=1.0)
        """
        if train_split is None:
            train_split = self.config.train_split
        
        # Set random seed for reproducibility
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
        
        n_samples = len(data_tensor)
        
        if train_split == 1.0:
            print(f"\nUsing full dataset for training (train_split=1.0)")
            print(f"  - Train samples: {n_samples}")
            return data_tensor, None
        
        # Create random permutation
        indices = torch.randperm(n_samples)
        train_size = int(train_split * n_samples)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        train_data = data_tensor[train_indices]
        val_data = data_tensor[val_indices]
        
        print(f"\nData split (train/val = {train_split:.1%}/{1-train_split:.1%}):")
        print(f"  - Train samples: {len(train_data)}")
        print(f"  - Val samples: {len(val_data)}")
        
        return train_data, val_data
    
    def save_preprocessor(self, save_path):
        """Save preprocessor state (scaler, market_ids, etc.)"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        state = {
            'scaler': self.scaler,
            'market_ids': self.market_ids,
            'temporal_neighbors': self.temporal_neighbors
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"\nPreprocessor saved to {save_path}")
    
    def load_preprocessor(self, load_path):
        """Load preprocessor state"""
        with open(load_path, 'rb') as f:
            state = pickle.load(f)
        
        self.scaler = state['scaler']
        self.market_ids = state['market_ids']
        self.temporal_neighbors = state['temporal_neighbors']
        
        print(f"\nPreprocessor loaded from {load_path}")


class MarketDataset(Dataset):
    """PyTorch Dataset for market data"""
    
    def __init__(self, data, temporal_neighbors=None):
        """
        Args:
            data: Tensor of shape [N, input_dim]
            temporal_neighbors: Dict mapping index to list of neighbor indices
        """
        self.data = data
        self.temporal_neighbors = temporal_neighbors if temporal_neighbors else {}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns the market data at index idx
        Temporal neighbor info is stored but not returned here
        (handled by augmentation module)
        """
        return self.data[idx]


def create_dataloaders(train_data, val_data, config, temporal_neighbors=None):
    """
    Create train and validation dataloaders
    
    Args:
        train_data: Training tensor
        val_data: Validation tensor (can be None)
        config: Configuration object
        temporal_neighbors: Dict of temporal neighbor mappings
    
    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader (None if val_data is None)
    """
    # Create datasets
    train_dataset = MarketDataset(train_data, temporal_neighbors)
    
    # Create train loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        persistent_workers=True if config.num_workers > 0 else False
    )
    
    # Create validation loader if val_data exists
    val_loader = None
    if val_data is not None:
        val_dataset = MarketDataset(val_data, temporal_neighbors)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory
        )
    
    print("\nDataLoaders created:")
    print(f"  - Train batches: {len(train_loader)}")
    if val_loader:
        print(f"  - Val batches: {len(val_loader)}")
    else:
        print(f"  - Val batches: 0 (no validation)")
    
    return train_loader, val_loader


# Example usage
if __name__ == "__main__":
    from config import MoCoConfig
    
    config = MoCoConfig()
    
    # Initialize preprocessor
    preprocessor = MarketDataPreprocessor(config)
    
    # Load and preprocess
    df = preprocessor.load_data()
    data_tensor = preprocessor.preprocess(df)
    
    # Split data
    train_data, val_data = preprocessor.split_data(data_tensor)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_data, val_data, config, preprocessor.temporal_neighbors
    )
    
    # Test
    for batch in train_loader:
        print(f"Batch shape: {batch.shape}")
        break