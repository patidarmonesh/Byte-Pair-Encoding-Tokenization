**Overview**

This repository implements a fast, exact, and production-minded Byte-Pair Encoding (BPE) trainer and tokenizer in Python. The implementation preserves original UTF-8 bytes (including whitespace and newlines), produces deterministic merges, and supports exact detokenization to reproduce the original input byte-for-byte.

**Quick Start**
!python 230663_assignment2_bpe.py --train train.txt --input test.txt --vocab_size 5000 --progress

**Key features**

Exact byte preservation: whitespace segments encoded byte-by-byte; words wrapped with BOS/EOS to prevent merges across whitespace.
Integer-centric core: token operations use integer ids to minimize Python overhead.
Per-unique-word representation: each unique word stored as a WordSymbolList (array of nodes with prev/next indices and an alive flag) to localize merges.
Priority queue of candidate pairs: a single heap stores adjacent-pair candidates; only local neighbors are re-inserted after a merge.
In-place merging and node reuse: left node is updated, right node is marked dead to reduce allocations.
Deterministic tie-breaking: lexicographic tie-breaks based on token-id composition ensure reproducible merges.
Two scoring modes: freq (default) and length (heuristic).

**Optimizations implemented**

Build symbol lists per unique word to avoid repeated full-corpus scans.
Maintain a single priority heap and re-evaluate only affected local pairs after each merge.
Use integer-only operations in hot loops and memoize token-to-string/bytes conversions.
Perform in-place merges and mark right nodes dead to reduce memory churn.
Encode whitespace as raw bytes to guarantee exact detokenization.
Provide a safety iteration limit during tokenization to avoid pathological loops.

**How it works (brief)**

Read training text and compute word frequencies.
Create WordSymbolList for each unique word: nodes = BOS + bytes + EOS.
Initialize the heap with all adjacent pairs (scored by frequency or length).
Repeatedly pop, validate, and perform in-place merges; append merges to merge_order.
After each merge, re-insert only local predecessor/successor pairs.
Stop when the desired vocabulary size is reached.
Tokenization applies merge_rules iteratively to each word; whitespace is appended as raw bytes.

**CLI options**
--train (required): training corpus (UTF-8)
--input (required): file to tokenize (UTF-8)
--vocab_size (required): target vocabulary size (includes reserved tokens)
--workers (optional): placeholder for parallelism (not used for heavy parallelism)
--progress (flag): print periodic progress to stderr
--score_method: freq (default) or length

**Benchmark note**

After the described optimizations, the implementation trains on a ~25 MB corpus in under one minute on the author's development hardware. Actual runtime depends on CPU, memory, and OS; include a benchmarks/ folder with machine specs and commands to reproduce timings.

