# BPE Tokenizer in Python

A lightweight and efficient implementation of the Byte Pair Encoding (BPE) tokenization algorithm in Python.

## Overview

This repository contains a Python implementation of the BPE tokenization algorithm, which is widely used in Natural Language Processing (NLP) tasks. BPE is a data compression technique that iteratively replaces the most frequent pair of bytes (or characters) in a sequence with a single, unused byte.

## Features

- Pure Python implementation of BPE algorithm
- Support for Unicode text
- Customizable vocabulary size
- Training on text files or directories
- Saving and loading trained models
- Regex pattern-based text splitting (including GPT-style tokenization pattern)

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Tokenizer-in-py.git
cd Tokenizer-in-py
```
## Usage
### Basic Usage

```python
from bpe import BPETokenizer

# Initialize tokenizer
tokenizer = BPETokenizer()

# Train on text
text = "Hello world! This is a sample text to demonstrate BPE tokenization."
tokenizer.train(text, vocab_size=300)

# Encode text
encoded = tokenizer.encode("Hello world!")
print(encoded)

# Decode tokens
decoded = tokenizer.decode(encoded)
print(decoded)
```

### Training on Files
```python
from bpe import BPETokenizer

tokenizer = BPETokenizer()

# Train on text files in a directory
tokenizer.train("path/to/text/files", vocab_size=1000, ispath=True)

# Save the trained model
tokenizer.save("my_tokenizer")

# Load a trained model
tokenizer = BPETokenizer()
tokenizer.load("my_tokenizer.model")
```

### Using Pattern Matching
```python
from bpe import BPETokenizer

tokenizer = BPETokenizer()

# Train with GPT-style pattern matching
tokenizer.train("path/to/text/files", vocab_size=1000, ispath=True, match_pattern=True)

# Or use a custom pattern
custom_pattern = r"\w+|\s+|[^\w\s]+"
tokenizer.train("path/to/text/files", vocab_size=1000, ispath=True, match_pattern=custom_pattern)
```

## How It Works
1. Initialization : Start with a vocabulary of 256 byte tokens (0-255)
2. Training :
   - Convert text to UTF-8 bytes
   - Count frequencies of adjacent byte pairs
   - Merge the most frequent pair and add to vocabulary
   - Repeat until desired vocabulary size is reached
3. Encoding :
   - Convert text to UTF-8 bytes
   - Iteratively merge byte pairs according to learned merges
4. Decoding :
   - Convert token IDs back to their corresponding byte sequences
   - Decode bytes to UTF-8 text