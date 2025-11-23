import os
import regex as re
from typing import BinaryIO
from multiprocessing import Process, Queue
import time


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def find_chunk_boundaries_text(
    text: str,
    desired_num_chunks: int,
    split_special_token: str,
) -> list[int]:
    """Divide chunk boundaries near special tokens in an in-memory string.

    Semantics similar to find_chunk_boundaries:
    - First divide evenly into desired_num_chunks parts;
    - Then scan forward from each internal boundary to find the next special token occurrence;
    - If not found, place the boundary at the end of text;
    - Finally deduplicate and sort boundaries.
    """

    length = len(text)
    if length == 0 or desired_num_chunks <= 1:
        return [0, length]

    chunk_size = length // desired_num_chunks

    boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    boundaries[-1] = length

    if not split_special_token:
        return sorted(set(boundaries))

    for bi in range(1, len(boundaries) - 1):
        start_pos = boundaries[bi]
        found_at = text.find(split_special_token, start_pos)
        if found_at == -1:
            boundaries[bi] = length
        else:
            boundaries[bi] = found_at

    return sorted(set(boundaries))

# Original "normal token" rules (excluding special tokens)
BASE_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# Regex for plain text only, excluding special token branches
BASE_PATTERN_RE = re.compile(BASE_PAT)

# Default example special token, only for demo in main()
TOKEN_STR = "<|endoftext|>"


def tokenize_text_chunk(
    text: str,
    special_tokens: list[str],
) -> dict[tuple[bytes], int]:
    """Pre-tokenize an entire text chunk in memory, processing only normal tokens."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    results: dict[tuple[bytes], int] = {}

    if special_tokens:
        escaped = [re.escape(tok) for tok in special_tokens]
        split_pattern = "(" + "|".join(escaped) + ")"
        parts = re.split(split_pattern, text)
    else:
        parts = [text]

    for part in parts:
        if not part:
            continue
        if special_tokens and part in special_tokens:
            continue
        for match in re.finditer(BASE_PATTERN_RE, part):
            token_bytes = match.group(0).encode("utf-8")
            token_tuple = (token_bytes,)
            results[token_tuple] = results.get(token_tuple, 0) + 1

    return results

def process_chunk(
    idx: int,
    start: int,
    end: int,
    file_path: str,
    pattern: "re.Pattern | None" = None,
    special_tokens: list[str] = [],
    result_queue: "Queue | None" = None,
) -> None:
    """Process one chunk in a subprocess.

    Key points:
    1. First split text by special tokens to ensure no merging across special tokens;
    2. Special tokens themselves don't participate in pre-tokenization counting;
    3. Only apply BASE_PATTERN_RE regex tokenization to normal text portions.
    """
    with open(file_path, "rb") as chunk_file:
        chunk_file.seek(start)
        raw = chunk_file.read(end - start).decode("utf-8", errors="ignore")

    results = tokenize_text_chunk(raw, special_tokens)

    # If called by parallel_pretokenize, put results in queue
    if result_queue is not None:
        result_queue.put(results)
        return

    # Default behavior: print sorted results (preserving original logic)
    sorted_results = dict(
        sorted(results.items(), key=lambda item: item[1], reverse=True)
    )
    print(sorted_results)

def parallel_pretokenize(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_processes: int,
) -> dict[tuple[bytes], int]:
    """Perform parallel pretokenization on the text file specified by input_path.

    Requirements:
    - Special tokens are only used for splitting boundaries, not for regex tokenization;
    - Don't count special token frequencies themselves;
    - Return frequency dict[tuple[bytes], int] for all normal tokens (unsorted).
    """
    # pattern parameter is kept for process_chunk, but actual matching only uses BASE_PATTERN_RE
    pattern = None

    input_path = os.fspath(input_path)

    # Calculate chunk boundaries
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f,
            num_processes,
            special_tokens[0].encode("utf-8") if special_tokens else b"",
        )

    # Start subprocesses
    q: Queue = Queue()
    processes: list[Process] = []
    for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]), 1):
        p = Process(
            target=process_chunk,
            args=(i, start, end, input_path, pattern,special_tokens, q),
        )
        p.start()
        processes.append(p)


    # Aggregate results (queue should contain exactly len(processes) partial results)
    global_counts: dict[tuple[bytes], int] = {}
    for _ in processes:
        partial = q.get()
        for tok, cnt in partial.items():
            global_counts[tok] = global_counts.get(tok, 0) + cnt

    # Wait for all processes to finish
    for p in processes:
        p.join()
    return global_counts


def parallel_pretokenize_text(
    text: str,
    special_tokens: list[str],
    num_processes: int,
) -> dict[tuple[bytes], int]:
    """Perform parallel pretokenization on an entire in-memory text.

    Uses similar approach to parallel_pretokenize:
    - Calculate boundaries on the string using find_chunk_boundaries_text;
    - Each subprocess processes a substring [start, end);
    - Subprocess internally still calls tokenize_text_chunk to share core logic.
    """

    if num_processes <= 1:
        return tokenize_text_chunk(text, special_tokens)

    split_token = special_tokens[0] if special_tokens else ""
    boundaries = find_chunk_boundaries_text(text, num_processes, split_token)

    q: Queue = Queue()
    processes: list[Process] = []

    for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]), 1):
        part = text[start:end]

        def _worker(idx: int, chunk_text: str, specials: list[str], queue: Queue) -> None:
            res = tokenize_text_chunk(chunk_text, specials)
            queue.put(res)

        p = Process(target=_worker, args=(i, part, special_tokens, q))
        p.start()
        processes.append(p)

    global_counts: dict[tuple[bytes], int] = {}
    for _ in processes:
        partial = q.get()
        for tok, cnt in partial.items():
            global_counts[tok] = global_counts.get(tok, 0) + cnt

    for p in processes:
        p.join()

    return global_counts


def remove_special_tokens_from_counts(
    counts: dict[tuple[bytes], int],
    special_tokens: list[str],
) -> dict[tuple[bytes], int]:
    """Remove special token counts from counts."""
    special_token_bytes = {tok.encode("utf-8") for tok in special_tokens}
    return {
        tok: cnt
        for tok, cnt in counts.items()
        if tok[0] not in special_token_bytes
    }

def sequential_pretokenize(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_chunks: int,
) -> dict[tuple[bytes], int]:
    """Count token frequencies single-process with same boundaries as parallel_pretokenize.

    To verify correctness of parallel implementation, this reuses the same
    "split by special tokens first, then tokenize normal text with BASE_PATTERN_RE" logic
    as process_chunk.
    """
    input_path = os.fspath(input_path)

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f,
            num_chunks,
            special_tokens[0].encode("utf-8") if special_tokens else b"",
        )

    global_counts: dict[tuple[bytes], int] = {}
    with open(input_path, "rb") as f:
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            raw = f.read(end - start).decode("utf-8", errors="ignore")
            partial = tokenize_text_chunk(raw, special_tokens)
            for tok, cnt in partial.items():
                global_counts[tok] = global_counts.get(tok, 0) + cnt

    return global_counts


def pretokenize_text(
    text: str,
    special_tokens: list[str],
) -> dict[tuple[bytes], int]:
    """Perform single-process pretokenization directly on in-memory text."""

    return tokenize_text_chunk(text, special_tokens)


def pretokenize(
    source: str | os.PathLike,
    special_tokens: list[str],
    num_processes: int = 1,
) -> dict[tuple[bytes], int]:
    """Unified entry point: supports both file paths and direct text input.

    - If source is an existing file path, process as file;
    - Otherwise treat source as text content.
    """

    path_like = os.fspath(source)

    if os.path.exists(path_like):
        if num_processes > 1:
            return parallel_pretokenize(path_like, special_tokens, num_processes)
        return sequential_pretokenize(path_like, special_tokens, num_chunks=1)

    text = str(source)
    if num_processes > 1:
        return parallel_pretokenize_text(text, special_tokens, num_processes)
    return pretokenize_text(text, special_tokens)


def main() -> None:
    file_path = r"tests/fixtures/tinystories_sample_5M.txt"  # Confirm this file exists
    num_processes = 4
    specials = [TOKEN_STR]

    # Parallel pretokenization
    t_par0 = time.perf_counter()
    parallel_counts = parallel_pretokenize(
        input_path=file_path,
        special_tokens=specials,
        num_processes=num_processes,
    )
    t_par1 = time.perf_counter()

    # Sequential pretokenization (using same chunk splitting approach)
    t_seq0 = time.perf_counter()
    sequential_counts = sequential_pretokenize(
        input_path=file_path,
        special_tokens=specials,
        num_chunks=num_processes,
    )
    t_seq1 = time.perf_counter()

    # Compare if results are completely identical
    same = parallel_counts == sequential_counts
    print(f"parallel vs sequential counts equal: {same}")
    print(f"parallel_pretokenize time:   {t_par1 - t_par0:.6f} seconds")
    print(f"sequential_pretokenize time: {t_seq1 - t_seq0:.6f} seconds")

    # Also check top tokens
    sorted_parallel = sorted(parallel_counts.items(), key=lambda kv: kv[1], reverse=True)
    print(f"parallel top token sample: {sorted_parallel[:5]}")


if __name__ == "__main__":
    main()
