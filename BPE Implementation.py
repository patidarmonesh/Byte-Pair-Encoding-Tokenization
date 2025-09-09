import argparse
import collections
import heapq
import time
import gc
import sys
import re
from typing import Dict, List, Tuple

# ------------------------------
# Constants & token id layout
# ------------------------------
RESERVED = ["<pad>", "<unk>", "<s>", "</s>"]   # 0..3
BYTE_OFFSET = len(RESERVED)                    # 4
BOS_ID = BYTE_OFFSET + 256                     # 260 (▁)
EOS_ID = BYTE_OFFSET + 257                     # 261 (</w>)
INITIAL_VOCAB_SIZE = BYTE_OFFSET + 256 + 2     # 262

ROLLNO = "230663"  

# ------------------------------
# Small classes for clarity
# ------------------------------
class WordSymbolList:
    """
    Linked-list-like representation of symbols for a single unique word.
    Nodes stored in a list; merging reuses the left node and marks right node dead.
    Node: dict with keys: token_id, prev, next, size, alive
    """
    def __init__(self, word: str):
        self.nodes = []
        # BOS node
        self.nodes.append({"token_id": BOS_ID, "prev": -1, "next": None, "size": 1, "alive": True})
        bts = word.encode("utf-8")
        # append byte nodes, maintaining correct prev pointer
        for byte in bts:
            prev_idx = len(self.nodes) - 1
            self.nodes.append({"token_id": BYTE_OFFSET + byte, "prev": prev_idx, "next": None, "size": 1, "alive": True})
        # EOS node
        prev_idx = len(self.nodes) - 1
        self.nodes.append({"token_id": EOS_ID, "prev": prev_idx, "next": -1, "size": 1, "alive": True})
        # fix next pointers
        for i in range(len(self.nodes) - 1):
            self.nodes[i]["next"] = i + 1

    def get_node(self, idx: int):
        if idx < 0 or idx >= len(self.nodes):
            return None
        return self.nodes[idx]

    def merge_nodes(self, left_idx: int, right_idx: int, expected_size: int):
        """
        Merge left and right nodes if valid (validate size to detect staleness).
        Returns True if merge succeeded, else False.
        """
        if not (0 <= left_idx < len(self.nodes) and 0 <= right_idx < len(self.nodes)):
            return False
        left = self.nodes[left_idx]
        right = self.nodes[right_idx]
        if left is None or right is None:
            return False
        if (not left["alive"]) or (not right["alive"]):
            return False
        if left.get("next", -2) != right_idx or right.get("prev", -3) != left_idx:
            return False
        if left["size"] + right["size"] != expected_size:
            return False

        # merge into left node 
        left["size"] = left["size"] + right["size"]
        next_idx = right.get("next", -1)
        left["next"] = next_idx if next_idx is not None else -1
        if next_idx is not None and next_idx != -1:
            self.nodes[next_idx]["prev"] = left_idx

        # mark right dead
        right["alive"] = False
        right["prev"] = -1
        right["next"] = -1
        return True

# ------------------------------
# Token conversion helpers
# ------------------------------
def token_id_to_string(token_id: int, merge_order: List[Tuple[int,int]], memo: Dict[int,str]) -> str:
    """Human-readable representation used for lexicographic tie-breaks and vocab output."""
    if token_id in memo:
        return memo[token_id]
    if token_id < BYTE_OFFSET:
        s = RESERVED[token_id]
    elif token_id < BYTE_OFFSET + 256:
        b = token_id - BYTE_OFFSET
        s = chr(b) if 32 <= b <= 126 else f"<byte_{b}>"
    elif token_id == BOS_ID:
        s = "▁"
    elif token_id == EOS_ID:
        s = "</w>"
    else:
        idx = token_id - INITIAL_VOCAB_SIZE
        if 0 <= idx < len(merge_order):
            left, right = merge_order[idx]
            s = token_id_to_string(left, merge_order, memo) + token_id_to_string(right, merge_order, memo)
        else:
            s = f"<unk_{token_id}>"
    memo[token_id] = s
    return s

def token_id_to_bytes(token_id: int, merge_order: List[Tuple[int,int]], memo_bytes: Dict[int, bytes]) -> bytes:
    """Reconstruct raw bytes for a token id (so we can detokenize to exact UTF-8)."""
    if token_id in memo_bytes:
        return memo_bytes[token_id]
    if token_id < BYTE_OFFSET:
        b = b''
    elif token_id < BYTE_OFFSET + 256:
        b = bytes([token_id - BYTE_OFFSET])
    elif token_id == BOS_ID or token_id == EOS_ID:
        b = b''
    else:
        idx = token_id - INITIAL_VOCAB_SIZE
        if 0 <= idx < len(merge_order):
            left, right = merge_order[idx]
            b = token_id_to_bytes(left, merge_order, memo_bytes) + token_id_to_bytes(right, merge_order, memo_bytes)
        else:
            b = b''
    memo_bytes[token_id] = b
    return b

def detokenize_from_token_ids(token_ids: List[int], merge_order: List[Tuple[int,int]]) -> str:
    """Given token ids, reconstruct the original UTF-8 text exactly (preserving whitespace/newlines)."""
    memo = {}
    out_bytes = bytearray()

    for tid in token_ids:
        # convert each token id to its bytes and append directly
        b = token_id_to_bytes(tid, merge_order, memo)
        out_bytes.extend(b)

    try:
        return out_bytes.decode('utf-8')
    except UnicodeDecodeError:
        return out_bytes.decode('utf-8', errors='replace')

# ------------------------------
# Core trainer 
# ------------------------------
def train_bpe_priority_linked(train_text: str, vocab_size: int, workers: int = 1,
                              progress: bool = False, score_method: str = "freq"):
    """
    Train BPE merges.

    score_method: "freq" (use word frequency) or "length" (heuristic: length of pair substring)
    """
    t0 = time.time()
    words = train_text.split()
    word_freqs = collections.Counter(words)
    unique_words = list(word_freqs.keys())

    # Build per-unique-word symbol lists
    word_lists = [WordSymbolList(w) for w in unique_words]
    freqs = [word_freqs[w] for w in unique_words]

    # Heap entries: (-score, pair_str, uid, word_index, left_idx, right_idx, left_tid, right_tid, size_sum)
    heap = []
    uid = 0
    merge_order: List[Tuple[int,int]] = []
    merge_rules = {}
    token_memo: Dict[int,str] = {}

    # initialize heap with adjacent pairs from each unique word
    for wi, wlist in enumerate(word_lists):
        freq = freqs[wi]
        nodes = wlist.nodes
        for i in range(len(nodes)-1):
            left = nodes[i]; right = nodes[i+1]
            if not left["alive"] or not right["alive"]:
                continue
            left_tid = left["token_id"]
            right_tid = right["token_id"]
            if score_method == "freq":
                score = freq
            else:  # length heuristic 
                # substring length approx by sum of sizes (bytes), which initially is 1 per byte
                score = left["size"] + right["size"]
            pair_str = token_id_to_string(left_tid, merge_order, token_memo) + "|" + token_id_to_string(right_tid, merge_order, token_memo)
            heapq.heappush(heap, (-score, pair_str, uid, wi, i, i+1, left_tid, right_tid, left["size"] + right["size"]))
            uid += 1

    max_merges = max(0, vocab_size - INITIAL_VOCAB_SIZE)
    merges_done = 0
    last_report = time.time()
    report_interval = 3.0

    # main loop
    while merges_done < max_merges and heap:
        neg_score, pair_str, entry_uid, wi, li, ri, left_tid_push, right_tid_push, size_sum = heapq.heappop(heap)
        score = -neg_score

        # validate
        if not (0 <= wi < len(word_lists)):
            continue
        wlist = word_lists[wi]
        nodes = wlist.nodes
        if not (0 <= li < len(nodes) and 0 <= ri < len(nodes)):
            continue
        left_node = nodes[li]; right_node = nodes[ri]
        if (not left_node["alive"]) or (not right_node["alive"]):
            continue
        if left_node.get("next", -2) != ri or right_node.get("prev", -3) != li:
            continue
        if left_node["size"] + right_node["size"] != size_sum:
            continue

        # perform merge
        a_tid = left_node["token_id"]
        b_tid = right_node["token_id"]
        new_tid = INITIAL_VOCAB_SIZE + len(merge_order)
        merge_order.append((a_tid, b_tid))
        merge_rules[(a_tid, b_tid)] = new_tid

        # update left node token_id to new merged id and mark right dead
        left_node["token_id"] = new_tid
        left_node["size"] = left_node["size"] + right_node["size"]
        next_idx = right_node.get("next", -1)
        left_node["next"] = next_idx if next_idx is not None else -1
        if next_idx is not None and next_idx != -1:
            nodes[next_idx]["prev"] = li
        right_node["alive"] = False
        right_node["prev"] = -1
        right_node["next"] = -1

        merges_done += 1

        # push new candidate pairs
        prev_idx = left_node.get("prev", -1)
        if prev_idx is not None and prev_idx != -1:
            prev_node = nodes[prev_idx]
            if prev_node["alive"]:
                left_tid_curr = prev_node["token_id"]
                right_tid_curr = left_node["token_id"]
                s = token_id_to_string(left_tid_curr, merge_order, token_memo) + "|" + token_id_to_string(right_tid_curr, merge_order, token_memo)
                if score_method == "freq":
                    sc = freqs[wi]
                else:
                    sc = prev_node["size"] + left_node["size"]
                heapq.heappush(heap, (-sc, s, uid, wi, prev_idx, li, left_tid_curr, right_tid_curr, prev_node["size"] + left_node["size"]))
                uid += 1

        next_idx = left_node.get("next", -1)
        if next_idx is not None and next_idx != -1:
            next_node = nodes[next_idx]
            if next_node["alive"]:
                left_tid_curr = left_node["token_id"]
                right_tid_curr = next_node["token_id"]
                s = token_id_to_string(left_tid_curr, merge_order, token_memo) + "|" + token_id_to_string(right_tid_curr, merge_order, token_memo)
                if score_method == "freq":
                    sc = freqs[wi]
                else:
                    sc = left_node["size"] + next_node["size"]
                heapq.heappush(heap, (-sc, s, uid, wi, li, next_idx, left_tid_curr, right_tid_curr, left_node["size"] + next_node["size"]))
                uid += 1

        # progress
        if progress and (time.time() - last_report >= report_interval):
            elapsed = time.time() - t0
            rate = merges_done / elapsed if elapsed > 0 else 0.0
            remaining = max(0, max_merges - merges_done)
            eta = remaining / rate if rate > 0 else float("inf")
            left_s = token_id_to_string(a_tid, merge_order, token_memo)
            right_s = token_id_to_string(b_tid, merge_order, token_memo)
            print(f"[BPE] {merges_done}/{max_merges} merges; last='{left_s}+{right_s}' score={score} elapsed={elapsed:.1f}s rate={rate:.2f}/s eta={eta:.1f}s", file=sys.stderr)
            last_report = time.time()
            gc.collect()

    # build vocabulary list
    vocab: List[str] = []
    vocab.extend(RESERVED)
    for b in range(256):
        if 32 <= b <= 126:
            vocab.append(chr(b))
        else:
            vocab.append(f"<byte_{b}>")
    vocab.append("▁")
    vocab.append("</w>")

    memo = {i: vocab[i] for i in range(len(vocab))}
    for k, (left, right) in enumerate(merge_order):
        tid = INITIAL_VOCAB_SIZE + k
        s = token_id_to_string(tid, merge_order, memo)
        vocab.append(s)

    tokenizer = {
        "vocab": vocab,
        "merge_rules": { (a,b): INITIAL_VOCAB_SIZE + i for i, (a,b) in enumerate(merge_order) },
        "merge_order": merge_order
    }

    if progress:
        total_time = time.time() - t0
        print(f"[BPE] finished {len(merge_order)} merges; vocab={len(vocab)} time={total_time:.1f}s", file=sys.stderr)
    return vocab, tokenizer

# ------------------------------
# Tokenize + I/O helpers 
# ------------------------------
def get_word_token_ids(word: str) -> List[int]:
    out = [BOS_ID]
    for b in word.encode("utf-8"):
        out.append(BYTE_OFFSET + b)
    out.append(EOS_ID)
    return out

def tokenize_text_to_ids(text: str, tokenizer: Dict) -> List[int]:
    """
    Tokenize the input text **preserving whitespace and newlines**.

    Approach:
    - Split text into alternating sequences of non-whitespace (words) and whitespace using re.split(r'(\s+)').
    - For word segments: wrap with BOS/EOS and perform merges using merge_rules.
    - For whitespace segments: encode each byte directly (as BYTE_OFFSET + byte) so the exact whitespace is preserved.
    """
    merge_rules = tokenizer["merge_rules"]
    all_ids: List[int] = []

    segments = re.split(r'(\s+)', text)
    for seg in segments:
        if seg == "":
            continue
        if seg.isspace():
            # preserve whitespace exactly by encoding its raw bytes (no BOS/EOS)
            for b in seg.encode("utf-8"):
                all_ids.append(BYTE_OFFSET + b)
        else:
            # a word: wrap with BOS/EOS and apply merge rules
            toks = get_word_token_ids(seg)
            changed = True
            iter_limit = 2000
            it = 0
            while changed and it < iter_limit:
                changed = False
                new_toks = []
                i = 0
                while i < len(toks):
                    if i < len(toks)-1 and (toks[i], toks[i+1]) in merge_rules:
                        new_toks.append(merge_rules[(toks[i], toks[i+1])])
                        i += 2
                        changed = True
                    else:
                        new_toks.append(toks[i])
                        i += 1
                toks = new_toks
                it += 1
            all_ids.extend(toks)
    return all_ids

def save_vocab(vocab: List[str], rollno: str, vocab_size: int):
    fname = f"{rollno}_assignment2_bpe_vocab{vocab_size}.txt"
    with open(fname, "w", encoding="utf-8") as fh:
        for t in vocab:
            fh.write(t + "\n")

def save_tokens_from_ids(token_ids: List[int], vocab: List[str], rollno: str):
    fname = f"{rollno}_assignment2_bpe_tokens.txt"
    with open(fname, "w", encoding="utf-8") as fh:
        for tid in token_ids:
            if 0 <= tid < len(vocab):
                fh.write(vocab[tid] + "\n")
            else:
                fh.write(f"<unk_{tid}>\n")

def save_detokenized_text(detok_text: str, rollno: str):
    fname = f"{rollno}_assignment2_bpe_detokenized.txt"
    with open(fname, "w", encoding="utf-8") as fh:
        fh.write(detok_text)

# ------------------------------
# --- Boilerplate interface functions requested ---
# ------------------------------
def load_training_data(train_path: str) -> str:
    """Load and return raw text for training."""
    with open(train_path, "r", encoding="utf-8") as fh:
        return fh.read()

def train_bpe_tokenizer(text: str, vocab_size: int, workers: int = 1, progress: bool = False, score_method: str = "freq"):
    """
    Learn BPE merges and return (vocab, tokenizer) tuple.
    This function wraps the original trainer (train_bpe_priority_linked).
    """
    return train_bpe_priority_linked(text, vocab_size, workers=workers, progress=progress, score_method=score_method)

def tokenize(text: str, tokenizer: Dict) -> List[int]:
    """Tokenize input text using trained BPE model (preserving whitespace/newlines)."""
    return tokenize_text_to_ids(text, tokenizer)

def detokenize(tokens: List[int], tokenizer: Dict) -> str:
    """Detokenize tokens back to original text."""
    return detokenize_from_token_ids(tokens, tokenizer["merge_order"])

def save_tokens(tokens: List[int], rollno: str, vocab: List[str]):
    """Save tokens (one token text per line) using the provided vocab (keeps original output naming)."""
    save_tokens_from_ids(tokens, vocab, rollno)

def save_detokenized(text: str, rollno: str):
    save_detokenized_text(text, rollno)

# ------------------------------
# CLI 
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Priority-Queue + Linked-List BPE (final) - 230663")
    parser.add_argument("--train", required=True, help="Training data file (UTF-8)")
    parser.add_argument("--input", required=True, help="Input file to tokenize (UTF-8)")
    parser.add_argument("--vocab_size", required=True, type=int, help="Target vocab size (includes 4 reserved tokens)")
    parser.add_argument("--workers", type=int, default=1, help="(Optional) workers for initial population (not required)")
    parser.add_argument("--progress", action="store_true", help="Show periodic progress (do not use in final submission)")
    parser.add_argument("--score_method", choices=["freq","length"], default="freq", help="How to score candidate pairs")
    args = parser.parse_args()

    # Load training text
    train_text = load_training_data(args.train)

    # Train tokenizer (returns vocab, tokenizer)
    vocab, tokenizer = train_bpe_tokenizer(train_text, args.vocab_size, workers=args.workers, progress=args.progress, score_method=args.score_method)

    # Save vocabulary file (uses original file naming)
    save_vocab(vocab, ROLLNO, args.vocab_size)

    # Read input to tokenize (preserve exact content)
    with open(args.input, "r", encoding="utf-8") as fh:
        input_text = fh.read()

    # Tokenize and save tokens
    token_ids = tokenize(input_text, tokenizer)
    save_tokens(token_ids, ROLLNO, vocab)

    # Detokenize and save detokenized text (exact preservation)
    detok_text = detokenize(token_ids, tokenizer)
    save_detokenized(detok_text, ROLLNO)

if __name__ == "__main__":
    main()
