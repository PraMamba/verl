---
name: add-dataset
description: Add a new dataset loader to verl
---

# Add Dataset Loader

This skill guides you through adding a new dataset loader to verl.

## Steps

### 1. Create Dataset File (10 min)

**Location:** `verl/data/your_dataset_name.py`

**Template:**
```python
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

from typing import List, Dict, Any
from torch.utils.data import Dataset

class YourDataset(Dataset):
    """
    Dataset for your specific data format.

    Args:
        data_path: Path to dataset file(s)
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data
        self.data = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load and parse dataset."""
        # Implement data loading logic
        data = []
        # ... load from file
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single data item.

        Returns:
            Dict with keys:
            - 'prompt': str - The prompt text
            - 'messages': List[Dict] - Conversation messages (for SFT/RL)
            - 'answer': str - Ground truth answer (optional, for RL)
        """
        item = self.data[idx]

        # Format for verl
        return {
            'prompt': item['prompt'],
            'messages': [
                {'role': 'user', 'content': item['prompt']},
                {'role': 'assistant', 'content': item['response']}
            ],
            'answer': item.get('answer', None)
        }
```

**Required Fields:**
- **For SFT:** `messages` (conversation format)
- **For RL:** `messages` + optional `answer` (for reward computation)

### 2. Register Dataset (2 min)

**File:** `verl/data/__init__.py`

```python
from verl.data.your_dataset_name import YourDataset

__all__ = [
    # ... existing datasets
    'YourDataset',
]
```

### 3. Create Dataset Config (Optional, 3 min)

**Location:** `verl/trainer/config/data/your_dataset.yaml`

```yaml
# @package _global_

data:
  train_files: /path/to/train.jsonl
  val_files: /path/to/val.jsonl
  train_batch_size: 1024
  val_batch_size: 256

  dataset_class: YourDataset
  dataset_kwargs:
    max_length: 2048
```

### 4. Add Unit Tests (10 min)

**Location:** `tests/test_your_dataset.py`

```python
import pytest
from verl.data.your_dataset_name import YourDataset
from transformers import AutoTokenizer

@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained('gpt2')

@pytest.fixture
def sample_data_path(tmp_path):
    """Create sample dataset file."""
    data_file = tmp_path / "sample.jsonl"
    with open(data_file, 'w') as f:
        f.write('{"prompt": "Test", "response": "Answer"}\n')
    return str(data_file)

def test_dataset_loading(sample_data_path, tokenizer):
    """Test dataset loads correctly."""
    dataset = YourDataset(sample_data_path, tokenizer)
    assert len(dataset) > 0

def test_dataset_item_format(sample_data_path, tokenizer):
    """Test item format is correct."""
    dataset = YourDataset(sample_data_path, tokenizer)
    item = dataset[0]

    assert 'prompt' in item
    assert 'messages' in item
    assert isinstance(item['messages'], list)
    assert all('role' in msg and 'content' in msg for msg in item['messages'])

def test_dataset_with_dataloader(sample_data_path, tokenizer):
    """Test dataset works with DataLoader."""
    from torch.utils.data import DataLoader

    dataset = YourDataset(sample_data_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=lambda x: x)

    batch = next(iter(dataloader))
    assert len(batch) <= 2
```

### 5. Integration Test (Optional, 5 min)

Test in trainer:
```python
def test_dataset_in_trainer():
    """Test dataset in training loop."""
    config = load_test_config('ppo_trainer.yaml')
    config.data.dataset_class = 'YourDataset'
    config.data.train_files = 'path/to/test_data.jsonl'

    trainer = RayPPOTrainer(config)
    trainer.init_workers()
    # Run one epoch
```

## Common Patterns

### JSONL Format
```python
import json

def _load_data(self):
    data = []
    with open(self.data_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    return data
```

### Parquet Format
```python
import pandas as pd

def _load_data(self):
    df = pd.read_parquet(self.data_path)
    return df.to_dict('records')
```

### HuggingFace Datasets
```python
from datasets import load_dataset

def _load_data(self):
    dataset = load_dataset('your_dataset_name', split='train')
    return list(dataset)
```

### Multi-Turn Conversations
```python
def __getitem__(self, idx):
    item = self.data[idx]

    messages = []
    for turn in item['conversation']:
        messages.append({
            'role': turn['role'],
            'content': turn['content']
        })

    return {
        'prompt': messages[0]['content'],  # First user message
        'messages': messages
    }
```

### Data Filtering
```python
def _load_data(self):
    raw_data = self._load_raw_data()

    # Filter by length
    filtered_data = [
        item for item in raw_data
        if len(item['prompt']) < self.max_length
    ]

    return filtered_data
```

## Verification

Run tests:
```bash
pytest tests/test_your_dataset.py -v
```

Test loading:
```python
from verl.data.your_dataset_name import YourDataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')
dataset = YourDataset('path/to/data.jsonl', tokenizer)
print(f"Loaded {len(dataset)} examples")
print(dataset[0])
```

## Troubleshooting

**Dataset loading slow:**
- Cache processed data to disk
- Use memory mapping for large files
- Parallelize loading with multiprocessing

**Memory issues:**
- Use lazy loading (load items on-demand)
- Stream data instead of loading all at once
- Reduce max_length

**Format errors:**
- Validate data format in `_load_data()`
- Add error handling for malformed items
- Log skipped items for debugging

## Related Files
- `verl/data/`: Existing dataset implementations
- `verl/trainer/config/data/`: Dataset configs
- `examples/`: Example dataset usage
