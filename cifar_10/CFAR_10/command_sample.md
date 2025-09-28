### 1. SISA Data Processing (Scalable sharding + adaptive slicing):

**Auto-scaling (Recommended - system decides optimal configuration):**
```bash
python data_processing/entry_data_processing.py
```

**Manual Configuration (full control):**
```bash
python data_processing/entry_data_processing.py --num-shards 2 --num-slices 5
python data_processing/entry_data_processing.py --num-shards 3 --num-slices 4
python data_processing/entry_data_processing.py --num-shards 4 --num-slices 6
```


### 2. SISA Training (Smart hyperparameter tuning + incremental training):
```bash
python training/entry_training.py
```

### 3. Machine Unlearning (Smart tuning + proper slice retraining):
Unlearning by Index: 
python unlearning/sisa_unlearning.py --index 100 
Unlearning by Class: 
python unlearning/sisa_unlearning.py --class-name cat 


### 4. Class Prediction Search & Analysis:
```bash
python search.py --class-name cat
python search.py --class-name dog --threshold 0.3 --samples 16
```