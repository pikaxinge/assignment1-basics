import os
import cProfile
import pstats
from pathlib import Path

from cs336_basics.pretokenization_example import parallel_pretokenize
from collections import defaultdict
"""Given the path to an input corpus, run train a BPE tokenizer and
output its vocabulary and merges.

Args:
    input_path (str | os.PathLike): Path to BPE tokenizer training data.
    vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
    special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
        These strings will never be split into multiple tokens, and will always be
        kept as a single token. If these special tokens occur in the `input_path`,
        they are treated as any other string.

Returns:
    tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        vocab:
            The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges:
            BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
"""
def expand_pair_counts_bytes(pair_counts, not_split_special_tokens=None )-> dict[tuple[bytes,...], int]:
    out = {}
    for (token_bytes,), count in pair_counts:
        # token_bytes is e.g. b' don'
        chars = tuple(bytes([b]) for b in token_bytes)  # (b' ', b'd', b'o', b'n')
        out[chars] = count
    return out


def run_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    parallel_counts: dict[tuple[bytes], int] = parallel_pretokenize(
        input_path=input_path,
        special_tokens=special_tokens,
        num_processes=4,
    )

    # Normalize keys to tuples and expand to single-byte tokens
    pair_counts = expand_pair_counts_bytes(list(parallel_counts.items()))

    # print(f"some elements of pair_counts: {list(pair_counts.items())[:10]}")

    # initilize vocab: dict[int, bytes], The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
    vocab: dict[int, bytes] = {}
    # start from special tokens then add single-byte tokens
    current_index = 0
    for special_token in special_tokens:
        vocab[current_index] = special_token.encode("utf-8")
        current_index += 1
    for b in range(256):
        if current_index >= vocab_size:
            break
        vocab[current_index] = bytes([b])
        current_index += 1

    # merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item
    # is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
    # <token2>. The merges should be ordered by order of creation
    merges: list[tuple[bytes, bytes]] = []

    maximum_vocab_size = min(vocab_size, len(pair_counts)+len(vocab))

    while current_index < maximum_vocab_size:
        bpe_cache: dict[tuple[bytes,bytes],int] = defaultdict(int)
        # count pairs from every pair_counts key, add value to it 
        for k,v in pair_counts.items():
            # split into pairs
            for i in range(len(k)-1):
                # add to bpe_cache
                bpe_cache[(k[i], k[i+1])] += v
        # get most count key in bpe_cache
        (most_count_k,most_count_v) = max(bpe_cache.items(),key = lambda x : (x[1],x[0]))

        # print(f"Most common pair: {most_count_k} with count {most_count_v}")
        # add to merges
        merges.append(most_count_k)

        new_token = most_count_k[0]+most_count_k[1]

        # add new_token to vocab
        vocab[current_index] = new_token
        current_index += 1
        # for loop pair_counts to replace all new_token pairs to new_token
        new_pair_counts = {}
        for k,v in pair_counts.items():
            new_k = k
            i = 0
            while i < len(new_k) - 1:
                if new_token == new_k[i] + new_k[i+1]:
                    # replace k[i] + k[i+1] with the merged byte token
                    new_k = new_k[:i] + (new_token,) + new_k[i+2:]
                else:
                    i += 1
            new_pair_counts[new_k] = v
        pair_counts = new_pair_counts
    return (vocab, merges)

def run_bpe_cache(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    parallel_counts: dict[tuple[bytes], int] = parallel_pretokenize(
        input_path=input_path,
        special_tokens=special_tokens,
        num_processes=4,
    )

    # Normalize keys to tuples and expand to single-byte tokens
    pair_counts = expand_pair_counts_bytes(list(parallel_counts.items()))

    # Initialize vocab: dict[int, bytes], The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
    vocab: dict[int, bytes] = {}
    # start from special tokens then add single-byte tokens
    current_index = 0
    for special_token in special_tokens:
        vocab[current_index] = special_token.encode("utf-8")
        current_index += 1
    for b in range(256):
        if current_index >= vocab_size:
            break
        vocab[current_index] = bytes([b])
        current_index += 1

    # merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item
    # is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
    # <token2>. The merges should be ordered by order of creation
    merges: list[tuple[bytes, bytes]] = []

    maximum_vocab_size = min(vocab_size, len(pair_counts) + len(vocab))

    # Represent sequences by integer IDs so multiple indexes can share them.
    sequences: dict[int, tuple[bytes, ...]] = {}
    pair_counts_by_id: dict[int, int] = {}
    for seq_id, (seq, count) in enumerate(pair_counts.items()):
        sequences[seq_id] = seq
        pair_counts_by_id[seq_id] = count

    # Build bpe_cache and index_from_bpe_to_pair outside the while loop.
    bpe_cache: dict[tuple[bytes, bytes], int] = defaultdict(int)
    # index_from_bpe_to_pair maps a BPE pair to the set of sequence IDs where it appears.
    index_from_bpe_to_pair: dict[tuple[bytes, bytes], set[int]] = {}

    for seq_id, seq in sequences.items():
        count = pair_counts_by_id[seq_id]
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            bpe_cache[pair] += count
            index_from_bpe_to_pair.setdefault(pair, set()).add(seq_id)

    while current_index < maximum_vocab_size and bpe_cache:
        # get most frequent pair in bpe_cache
        (most_count_k, most_count_v) = max(bpe_cache.items(), key=lambda x: (x[1], x[0]))

        if most_count_v <= 0:
            break

        # print(f"Most common pair: {most_count_k} with count {most_count_v}")
        merges.append(most_count_k)

        new_token = most_count_k[0] + most_count_k[1]

        # add new_token to vocab
        if current_index >= vocab_size:
            break
        vocab[current_index] = new_token
        current_index += 1

        # sequences affected by this merge
        affected_ids = index_from_bpe_to_pair.get(most_count_k, set()).copy()
        # after processing, this pair is no longer present
        bpe_cache.pop(most_count_k, None)
        index_from_bpe_to_pair.pop(most_count_k, None)

        for seq_id in affected_ids:
            seq = sequences[seq_id]
            count = pair_counts_by_id[seq_id]

            # collect neighbor pairs to update counts and indexes
            neighbor_prefix: list[tuple[bytes, bytes]] = []
            neighbor_suffix: list[tuple[bytes, bytes]] = []

            new_seq = list(seq)
            i = 0
            # We operate on a list for easier in-place merging
            while i < len(new_seq) - 1:
                if new_seq[i] == most_count_k[0] and new_seq[i + 1] == most_count_k[1]:
                    # record neighbors before merging
                    if i > 0:
                        neighbor_prefix.append((new_seq[i - 1], new_seq[i]))
                    if i < len(new_seq) - 2:
                        neighbor_suffix.append((new_seq[i + 1], new_seq[i + 2]))

                    # apply merge: replace pair with new_token
                    new_seq[i : i + 2] = [new_token]
                    # do not advance i so we can catch overlapping merges
                else:
                    i += 1

            # update sequence storage
            sequences[seq_id] = tuple(new_seq)

            # update bpe_cache and index_from_bpe_to_pair for neighbors
            for np in neighbor_prefix:
                old_pair = np
                new_pair = (np[0], new_token)
                # decrease old neighbor count
                if old_pair in bpe_cache:
                    bpe_cache[old_pair] -= count
                    if bpe_cache[old_pair] <= 0:
                        bpe_cache.pop(old_pair, None)
                # increase new neighbor count
                bpe_cache[new_pair] += count
                index_from_bpe_to_pair.setdefault(new_pair, set()).add(seq_id)

            for np in neighbor_suffix:
                old_pair = np
                new_pair = (new_token, np[1])
                if old_pair in bpe_cache:
                    bpe_cache[old_pair] -= count
                    if bpe_cache[old_pair] <= 0:
                        bpe_cache.pop(old_pair, None)
                bpe_cache[new_pair] += count
                index_from_bpe_to_pair.setdefault(new_pair, set()).add(seq_id)

    return (vocab, merges)

def profile_run_bpe(func,   
     input_path = Path("tests/fixtures/tinystories_sample_5M.txt"),
    vocab_size = 500,
    special_tokens = ["<|endoftext|>"]
                    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Profile a single run_bpe call using cProfile.

    This does not affect tests; it's only used when running this
    module as a script.
    """

    profiler = cProfile.Profile()
    profiler.enable()
    vocab, merges = func(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    profiler.disable()

    # Dump stats to file for tools like snakeviz
    stats_path = "train_bpe.prof"
    profiler.dump_stats(stats_path)

    # Also print a short summary to the console
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats(30)

    print(f"Profiling data written to {stats_path}")
    return vocab, merges


if __name__ == "__main__":
    # profile_run_bpe(run_bpe)
    profile_run_bpe(run_bpe_cache)
    # run_bpe_cache( input_path=Path("tests/fixtures/tinystories_sample_5M.txt"),
    #     vocab_size=300,
    #     special_tokens=["<|endoftext|>"])
    # run_bpe( input_path=Path("tests/fixtures/tinystories_sample_5M.txt"),
    #     vocab_size=300,
    #     special_tokens=["<|endoftext|>"])
