# Navigate to the CIFAR-10 directory before executing the commands below.

# Data_processing:
python data_processing/entry_data_processing.py

**Manual Configuration (full control):**
python data_processing/entry_data_processing.py --num-shards 2 --num-slices 5
python data_processing/entry_data_processing.py --num-shards 3 --num-slices 4
python data_processing/entry_data_processing.py --num-shards 4 --num-slices 6

### 2. SISA Training:
python training/entry_training.py

### 3. Machine Unlearning:

**Single Class Unlearning:**
```bash
python unlearning/sisa_unlearning.py --class-name cat
python unlearning/sisa_unlearning.py --class-name dog
python unlearning/sisa_unlearning.py --class-name ship
```

**Batch Unlearning (Multiple Classes Simultaneously):**
```bash
# Unlearn 2 classes
python unlearning/sisa_unlearning.py --class-name cat dog

# Unlearn 3 classes
python unlearning/sisa_unlearning.py --class-name cat dog ship

# Unlearn 4 classes
python unlearning/sisa_unlearning.py --class-name airplane automobile bird truck
```

**Unlearning by Index (Not implemented yet):**
```bash
python unlearning/sisa_unlearning.py --index 100
```

### 4. Class Prediction Search & Analysis:
python search.py --class-name cat
python search.py --class-name dog --threshold 0.3 --samples 16