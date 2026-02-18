# MoCo Market Embeddings

A modular implementation of Momentum Contrast (MoCo) for learning market embeddings from economic indicator data.

## Project Structure

```
moco_market/
├── config.py               # Configuration management
├── data_preprocessing.py   # Data loading and preprocessing
├── augmentation.py         # Data augmentation strategies
├── model.py               # MoCo model architecture
├── trainer.py             # Training loop and utilities
├── inference.py           # Inference engine
├── train.py               # Main training script
├── run_inference.py       # Inference script
└── README.md              # This file
```


## Quick Start

### 1. Training

#### Basic Training (90% train, 10% validation)

```bash
cd /home/claude/moco_market
python train.py --data_path /home/gaian/Downloads/tensor_values.parquet
```

#### Full Training (100% train, no validation)

```bash
python train.py \
    --data_path /home/gaian/Downloads/tensor_values.parquet \
    --train_split 1.0 \
    --num_epochs 300
```

#### Custom Configuration

```bash
python train.py \
    --data_path /home/gaian/Downloads/tensor_values.parquet \
    --train_split 0.85 \
    --batch_size 128 \
    --num_epochs 250 \
    --learning_rate 5e-4 \
    --queue_size 8192 \
    --embedding_dim 256
```

### 2. Build Embeddings Database

After training, build the embeddings database:

```bash
python run_inference.py \
    --mode build_db \
    --checkpoint /home/claude/moco_market/checkpoints/best_model.pt \
    --preprocessor /home/claude/moco_market/outputs/preprocessor.pkl \
    --data_path /home/gaian/Downloads/tensor_values.parquet
```

### 3. Find Similar Markets

```bash
python run_inference.py \
    --mode similar \
    --checkpoint /home/claude/moco_market/checkpoints/best_model.pt \
    --preprocessor /home/claude/moco_market/outputs/preprocessor.pkl \
    --data_path /home/gaian/Downloads/tensor_values.parquet \
    --market_id "NLD_A_2020" \
    --top_k 10
```

### 4. Compare Two Markets

```bash
python run_inference.py \
    --mode compare \
    --checkpoint /home/claude/moco_market/checkpoints/best_model.pt \
    --preprocessor /home/claude/moco_market/outputs/preprocessor.pkl \
    --data_path /home/gaian/Downloads/tensor_values.parquet \
    --market_id "NLD_A_2020" \
    --market_id_2 "BEL_A_2020"
```

### 5. Cluster Markets

```bash
python run_inference.py \
    --mode cluster \
    --checkpoint /home/claude/moco_market/checkpoints/best_model.pt \
    --preprocessor /home/claude/moco_market/outputs/preprocessor.pkl \
    --data_path /home/gaian/Downloads/tensor_values.parquet \
    --n_clusters 10
```

### 6. Find Anomalies

```bash
python run_inference.py \
    --mode anomaly \
    --checkpoint /home/claude/moco_market/checkpoints/best_model.pt \
    --preprocessor /home/claude/moco_market/outputs/preprocessor.pkl \
    --data_path /home/gaian/Downloads/tensor_values.parquet
```

### 7. Analyze Temporal Evolution

```bash
python run_inference.py \
    --mode evolution \
    --checkpoint /home/claude/moco_market/checkpoints/best_model.pt \
    --preprocessor /home/claude/moco_market/outputs/preprocessor.pkl \
    --data_path /home/gaian/Downloads/tensor_values.parquet \
    --country USA
```

## Using the API Programmatically

### Training

```python
from config import MoCoConfig
from data_preprocessing import MarketDataPreprocessor, create_dataloaders
from augmentation import MarketDataAugmentation
from model import MoCo
from trainer import MoCoTrainer

# Create config
config = MoCoConfig()
config.update(
    train_split=1.0,  # Use full data for training
    num_epochs=300,
    batch_size=128
)

# Load and preprocess data
preprocessor = MarketDataPreprocessor(config)
df = preprocessor.load_data()
data_tensor = preprocessor.preprocess(df)
train_data, val_data = preprocessor.split_data(data_tensor)

# Create dataloaders
train_loader, val_loader = create_dataloaders(
    train_data, val_data, config, preprocessor.temporal_neighbors
)

# Create model
model = MoCo(config)

# Create augmentation
augmentation = MarketDataAugmentation(config, preprocessor.temporal_neighbors)

# Train
trainer = MoCoTrainer(model, train_loader, val_loader, config, augmentation)
trained_model = trainer.train()
```

### Inference

```python
from inference import MarketEmbeddingInference
from trainer import load_trained_model
from data_preprocessing import MarketDataPreprocessor

# Load model
model, config = load_trained_model('checkpoints/best_model.pt')

# Load preprocessor
preprocessor = MarketDataPreprocessor(config)
preprocessor.load_preprocessor('outputs/preprocessor.pkl')

# Create inference engine
inference = MarketEmbeddingInference(model, preprocessor)

# Build embeddings database
inference.build_embeddings_database(data_tensor)

# Find similar markets
query_embedding = inference.embeddings_cache[100]  # Example
similar = inference.find_similar_markets(query_embedding, top_k=10)

# Compare markets
similarity = inference.compare_markets('NLD_A_2020', 'BEL_A_2020')

# Cluster markets
clusters = inference.cluster_markets(n_clusters=10)

# Find anomalies
anomalies = inference.find_anomalies(threshold=0.5)

# Analyze evolution
evolution = inference.analyze_temporal_evolution('USA')
```

## Configuration Parameters

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `train_split` | 0.9 | Train/val split (use 1.0 for full training) |
| `batch_size` | 64 | Batch size |
| `num_epochs` | 200 | Number of epochs |
| `learning_rate` | 3e-4 | Learning rate |
| `embedding_dim` | 128 | Embedding dimension |
| `queue_size` | 4096 | Queue size for negatives |
| `momentum` | 0.999 | Momentum coefficient |
| `temperature` | 0.07 | Temperature for InfoNCE loss |
| `mask_prob` | 0.20 | Masking probability |
| `noise_std` | 0.02 | Gaussian noise std |

### Modify Config

```python
config = MoCoConfig()
config.update(
    train_split=1.0,        # Full training
    batch_size=128,         # Larger batch
    num_epochs=300,         # More epochs
    queue_size=8192,        # Larger queue
    embedding_dim=256       # Higher dimensional embeddings
)
```

## Training for Manager (No Validation)

When your manager asks to train on full data:

```bash
# Option 1: Command line
python train.py \
    --data_path /home/gaian/Downloads/tensor_values.parquet \
    --train_split 1.0 \
    --num_epochs 300

# Option 2: Programmatically
config = MoCoConfig()
config.update(train_split=1.0, num_epochs=300)
# ... continue with training
```

The system will:
- Use 100% of data for training
- Skip validation
- Use best training loss for early stopping
- Save the best model based on training loss

## Output Files

After training, you'll find:

```
/home/claude/moco_market/
├── checkpoints/
│   ├── best_model.pt              # Best model checkpoint
│   ├── checkpoint_epoch_0.pt      # Per-epoch checkpoints
│   └── ...
├── outputs/
│   ├── preprocessor.pkl           # Fitted preprocessor
│   ├── final_model.pt            # Final model state
│   └── embeddings.pt             # Embeddings database
└── logs/
    └── training_stats.json        # Training statistics
```

## Model Architecture

```
Input: [batch, 16197]
    ↓
Query Encoder (trainable):
    Linear(16197 → 2048) + BatchNorm + GELU + Dropout
    Linear(2048 → 1024) + BatchNorm + GELU + Dropout
    Linear(1024 → 512) + BatchNorm + GELU + Dropout
    Linear(512 → 256) + BatchNorm + GELU
    Linear(256 → 128)
    ↓
Projection Head:
    Linear(128 → 64) + ReLU
    Linear(64 → 32)
    ↓
L2 Normalization
    ↓
InfoNCE Loss with Queue (4096 negatives)

Key Encoder (momentum updated): Same architecture, no gradients
Queue: Stores 4096 past key embeddings
```

## Data Augmentation

Augmentation applied to **non-zero elements only**:

1. **Masking**: Randomly mask 20% of non-zero indicators
2. **Gaussian Noise**: Add 2% noise to non-zero elements
3. **Temporal Mixing**: Mix with neighboring years (20% probability)

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
python train.py --batch_size 32

# Or use gradient accumulation (modify trainer.py)
```

### Slow Training

```bash
# Increase workers
python train.py --num_workers 8

# Use mixed precision (default: enabled)
python train.py --use_mixed_precision
```

### Queue Size Issues

Ensure queue_size is divisible by batch_size:

```bash
# Good combinations
--batch_size 64 --queue_size 4096   # 4096 / 64 = 64
--batch_size 128 --queue_size 8192  # 8192 / 128 = 64
```

## Performance Tips

1. **For 75K samples**: Default settings work well
2. **For faster training**: Increase batch_size to 128-256
3. **For better quality**: Increase queue_size to 8192
4. **For production**: Use train_split=1.0 after validating hyperparameters

## Citation

This implementation is based on:

```
MoCo: Momentum Contrast for Unsupervised Visual Representation Learning
Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, Ross Girshick
CVPR 2020
```

## License

MIT License

## Support

For issues or questions, check the code comments or contact the development team.