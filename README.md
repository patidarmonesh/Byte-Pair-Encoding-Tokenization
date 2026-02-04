# Byte-Pair Encoding (BPE) Tokenizer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A fast and efficient implementation of Byte-Pair Encoding (BPE) tokenizer in pure Python. This implementation uses priority queues and linked-list data structures for optimal performance and supports exact detokenization with full UTF-8 byte preservation.

## Overview

This project implements a production-ready BPE tokenizer from scratch using only Python's standard library. The implementation focuses on:

- **Performance**: Uses priority queues and per-word symbol lists to achieve ~450x speedup over naive implementations
- **Accuracy**: Exact byte preservation ensures lossless detokenization
- **Efficiency**: In-place merging and memoization reduce memory overhead
- **Determinism**: Lexicographic tie-breaking guarantees reproducible results

## Quick Start

```bash
git clone https://github.com/patidarmonesh/Byte-Pair-Encoding-Tokenization.git
cd Byte-Pair-Encoding-Tokenization

python BPE.py --train input_data.txt --input sample_data.txt --vocab_size 5000
```

This will generate three output files:
- `bpe_vocab_5000.txt` - Vocabulary file
- `bpe_tokens.txt` - Tokenized output
- `bpe_detokenized.txt` - Reconstructed text

## Installation

**Requirements:**
- Python 3.8 or higher
- No external dependencies (uses only standard library)

**Setup:**

```bash
git clone https://github.com/patidarmonesh/Byte-Pair-Encoding-Tokenization.git
cd Byte-Pair-Encoding-Tokenization
```

Optional: Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## Usage

### Basic Usage

```bash
python BPE.py --train <training_file> --input <input_file> --vocab_size <size>
```

### With Progress Monitoring

```bash
python BPE.py --train input_data.txt --input sample_data.txt --vocab_size 5000 --progress
```

### Command-Line Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--train` | string | Yes | Path to training corpus (UTF-8 text file) |
| `--input` | string | Yes | Path to input file for tokenization |
| `--vocab_size` | int | Yes | Target vocabulary size (includes 4 reserved tokens) |
| `--workers` | int | No | Number of workers (default: 1) |
| `--progress` | flag | No | Display progress updates during training |
| `--score_method` | string | No | Scoring method: `freq` (default) or `length` |

## Algorithm

### What is BPE?

Byte-Pair Encoding is a data compression technique adapted for subword tokenization in NLP. The algorithm iteratively merges the most frequent pairs of adjacent symbols to build a vocabulary of subword units.

### Implementation Details

**Key Data Structures:**

1. **WordSymbolList**: Linked-list representation of each unique word
   - Stores symbols as nodes with prev/next pointers
   - Supports O(1) in-place merging
   - Marks merged nodes as "dead" to avoid reallocation

2. **Priority Heap**: Max-heap of candidate symbol pairs
   - Enables O(log n) extraction of highest-frequency pair
   - Only local neighbors re-inserted after merge
   - Lexicographic tie-breaking for determinism

**Training Process:**

```
1. Split corpus into words and count frequencies
2. Initialize each word as: [BOS, byte₁, byte₂, ..., byteₙ, EOS]
3. Populate heap with all adjacent symbol pairs
4. Repeat until target vocabulary size:
   - Pop highest-scoring pair from heap
   - Validate pair is still adjacent and alive
   - Merge in-place: update left node, mark right node dead
   - Record merge operation
   - Re-insert affected local pairs
5. Build final vocabulary from merge history
```

**Tokenization:**

Words are tokenized by iteratively applying learned merge rules. Whitespace is preserved as raw bytes to ensure exact detokenization.

## Performance

### Benchmarks

Training performance on different corpus sizes:

| Corpus Size | Vocab Size | Training Time | Memory Usage |
|-------------|------------|---------------|--------------|
| 1 MB        | 5,000      | 8s            | 120 MB       |
| 10 MB       | 10,000     | 45s           | 850 MB       |
| 25 MB       | 5,000      | 54s           | 1.8 GB       |
| 100 MB      | 50,000     | 6min          | 6.5 GB       |

Test environment: Intel Core i7-10750H, 16GB RAM, Ubuntu 22.04, Python 3.10

### Optimizations

The following optimizations contribute to the ~450x overall speedup:

| Technique | Impact |
|-----------|--------|
| Per-word symbol lists | 10x faster |
| Priority queue for pair selection | 5x faster |
| Integer-only operations in hot loops | 2x faster |
| Memoization of token conversions | 3x faster |
| In-place merging with node reuse | 1.5x faster |

## Dataset

The implementation has been tested on a 25MB multilingual corpus containing:
- **Languages**: English, Hindi, and other Unicode scripts
- **Format**: UTF-8 encoded plain text
- **Size**: ~26 MB (5.2M tokens, 453K unique words)
- **Content**: Mixed domain text with diverse vocabulary

Training data: [input_data.txt](https://github.com/patidarmonesh/Byte-Pair-Encoding-Tokenization/blob/main/input_data.txt)

Sample data: [sample_data.txt](https://github.com/patidarmonesh/Byte-Pair-Encoding-Tokenization/blob/main/sample_data.txt)

## Implementation Highlights

**Byte-Level Encoding:**
- Initial vocabulary consists of 256 UTF-8 bytes plus reserved tokens
- Guarantees coverage of any Unicode text without unknown tokens
- Enables exact reconstruction of original text

**Deterministic Behavior:**
- Frequency-based pair selection with lexicographic tie-breaking
- Same input always produces identical vocabulary and tokenization
- Critical for reproducibility in ML pipelines

**Memory Efficiency:**
- In-place node updates avoid expensive array copying
- Dead nodes marked with flag rather than deleted
- Memoization caches prevent redundant computations

## Technical Details

**Token ID Layout:**

```
Range       | Purpose
------------|------------------
0-3         | Reserved tokens (<pad>, <unk>, <bos>, <eos>)
4-259       | UTF-8 bytes (256 values)
260         | BOS marker (▁)
261         | EOS marker ()
262+        | Learned merge tokens
```

**Complexity Analysis:**

- Training: O(V × log H) where V = vocab_size, H = heap size
- Tokenization: O(T × L × log L) where T = tokens, L = avg length
- Space: O(W × L) where W = unique words

## References

This implementation draws inspiration from:

1. **Sennrich, R., Haddow, B., & Birch, A. (2016)**  
   *Neural Machine Translation of Rare Words with Subword Units*  
   Proceedings of ACL 2016  
   https://arxiv.org/abs/1508.07909

2. **Gage, P. (1994)**  
   *A New Algorithm for Data Compression*  
   C Users Journal, 12(2)

3. **Guillaume Becquin (2021)**  
   *Byte Pair Encoding - Understanding the Algorithm*  
   https://guillaume-be.github.io/2021-09-16/byte_pair_encoding

## Project Structure

```
.
├── BPE.py                        # Main implementation
├── input_data.txt                # Training corpus
├── sample_data.txt               # Sample input for testing
├── README.md                     # This file
├── LICENSE                       # MIT License
└── examples/
    ├── train_example.py          # Example training script
    └── output_examples/          # Sample outputs
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Author

**Monesh Patidar**  
IIT Kanpur  
GitHub: [@patidarmonesh](https://github.com/patidarmonesh)  
Email: moeshp23@iitk.ac.in

## Acknowledgments

- Guillaume Becquin for the comprehensive BPE tutorial and visualization
- Hugging Face team for tokenizer implementation best practices
- Original BPE authors for the elegant compression algorithm

---

For questions, issues, or contributions, please open an issue on GitHub or contact via email.
