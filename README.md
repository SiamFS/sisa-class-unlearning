<div align="center">

# Machine Unlearning for Class Removal through SISA-based Deep Neural Network Architectures

The SISA (Sharded, Isolated, Sliced, and Aggregated) Framework is a implementation for efficient machine unlearning in deep learning models. This implementation is based on a modified version of the SISA framework that focuses on class unlearning rather than individual data point removal. While the original SISA framework was designed to delete specific data points from CNN models, this implementation addresses a different challenge: removing entire classes from trained CNN models. This approach is more relevant to real-world scenarios where organizations need to remove all data related to a particular category for data privacy compliance, rather than individual data points.
</div>

## Prerequisites

- Python 3.7+
- PyTorch
- torchvision
- NumPy
- Matplotlib
- scikit-learn
- seaborn

## Installation

```bash
pip install torch torchvision numpy matplotlib scikit-learn seaborn
```

## Usage

### 1. Data Processing

Process and partition the dataset into shards and slices:

**Configuration**
This can be modified in `config.py`.

```bash
python data_processing/entry_data_processing.py
```

### 2. Model Training

Train the SISA system with shard-specific models and gating network:

```bash
python training/entry_training.py
```

### 3. Unlearning

**Unlearn by class:**
```bash
python unlearning/sisa_unlearning.py --class-name cat
```

### 4. Prediction Analysis

Analyze model predictions for specific classes:

```bash
python search.py --class-name cat
python search.py --class-name dog --threshold 0.3 --samples 16 # max samples is 16
```

## Configuration

All system parameters can be modified in `config.py`, including:

- Shard and slice counts (`NUM_SHARDS`, `NUM_SLICES_PER_SHARD`)
- Training hyperparameters (`BATCH_SIZE`, `MAX_EPOCHS`, `LEARNING_RATE`)
- Unlearning settings (`UNLEARNING_PATIENCE`, `UNLEARNING_LEARNING_RATE`)
- Confidence thresholds and augmentation strategies

## Project Structure

```
Sisa-Framework/
├── data_processing/          # Data partitioning and preprocessing
├── training/                 # Model training and gating network
├── unlearning/              # Unlearning implementation
├── config.py                # Global configuration
├── plots.py                 # Visualization utilities
└── search.py                # Prediction analysis tool
```


## References

Bourtoule, L., Chandrasekaran, V., Choquette-Choo, C. A., Jia, H., Travers, A., Zhang, B., Lie, D., & Papernot, N. (2021). Machine Unlearning. In *2021 IEEE Symposium on Security and Privacy (SP)*. arXiv:1912.03817 [cs.CR]. https://arxiv.org/abs/1912.03817
