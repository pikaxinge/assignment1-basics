from __future__ import annotations

from collections.abc import Iterable, Iterator
import json
import os
from pathlib import Path

import tiktoken

from cs336_basics.pretokenization_example import parallel_pretokenize, parallel_pretokenize_text, parallel_pretokenize_text_seq
from cs336_basics.train_bpe import expand_pair_counts_bytes


class Tokenizer:
	"""Byte-pair encoding tokenizer.

	This class is constructed from a vocabulary and list of merges and
	provides methods for encoding and decoding text. Special tokens can
	optionally be provided.
	"""
	reversed_vocab:dict[bytes,int]
	merges_rank:dict[tuple[bytes, bytes], int]

	def __init__(
		self,
		vocab: dict[int, bytes],
		merges: list[tuple[bytes, bytes]],
		special_tokens: list[str] | None = None,
	) -> None:
		"""Construct a tokenizer from vocabulary, merges, and special tokens."""
		self.vocab = vocab
		self.merges = merges
		self.special_tokens = special_tokens or []
		# build reverse vocab
		self.reversed_vocab = {v: k for k, v in vocab.items()}
		# build merges rank
		self.merges_rank = {merge: idx for idx, merge in enumerate(merges)}

	@classmethod
	def from_files(
		cls,
		vocab_filepath: str,
		merges_filepath: str,
		special_tokens: list[str] | None = None,
	) -> "Tokenizer":
		"""Create a tokenizer from serialized vocabulary and merges files.

		`vocab_filepath` is expected to be a JSON file mapping string token
		representations to integer IDs (e.g., GPT-2 style vocab). This method
		inverts that mapping to obtain `dict[int, bytes]`.

		`merges_filepath` is expected to be a plain-text file where each line
		contains two space-separated tokens (like `"Ġ t"`), matching the
		format produced by the BPE training script.
		"""
		# Load vocabulary (JSON mapping token -> id), convert to id -> bytes
		with open(vocab_filepath, encoding="utf-8") as vf:
			raw_vocab: dict[str, int] = json.load(vf)

		vocab: dict[int, bytes] = {
			idx: token.encode("utf-8") for token, idx in raw_vocab.items()
		}

		# Load merges (each line: "token1 token2")
		merges: list[tuple[bytes, bytes]] = []
		with open(merges_filepath, encoding="utf-8") as mf:
			for line in mf:
				line = line.strip()
				if not line:
					continue
				parts = line.split(" ")
				if len(parts) != 2:
					continue
				token1_str, token2_str = parts
				merges.append((token1_str.encode("utf-8"), token2_str.encode("utf-8")))

		return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

	def encode(self, text: str) -> list[int]:
		"""Encode an input text into a sequence of token IDs."""
		# check special tokens if overlap with vocab
		# preprocess text into tokens (bytes)
		text_seq: list[tuple[bytes, ...]] = parallel_pretokenize_text_seq(
			text,
			self.special_tokens,
			num_processes=4
		)
		result_seq_Ids: list[int] = []
		# bpe encode the tokens
		for seq in text_seq:
			# print(f"Encoding token sequence: {seq}")
			# don't split special tokens
			if self.special_tokens and seq in tuple(
				(token.encode("utf-8"),) for token in self.special_tokens
			):
				token_id = self.reversed_vocab.get(seq[0], -1)
				if token_id != -1:
					result_seq_Ids.append(token_id)
				continue
			# use merges to merge bytes in seq
			seq_merge_cache: tuple[bytes, ...] = seq

			while True:
				# find smallest merge token in pair_set according to merges order
				min_token_id = 1e9
				min_token = None
				for i in range(len(seq_merge_cache) - 1):
					pair = (seq_merge_cache[i], seq_merge_cache[i + 1])
					if pair in self.merges_rank:
						if self.merges_rank[pair] < min_token_id:
							min_token_id = self.merges_rank[pair]
							min_token = pair
				# if no merge token found, break
				if min_token_id == 1e9 or min_token == None:
					break
				# merge all occurrences of min_token in seq_merge_cache
				index = 0
				while index < len(seq_merge_cache) - 1:
					if min_token == (seq_merge_cache[index], seq_merge_cache[index+1]):
						# merge
						seq_merge_cache = seq_merge_cache[:index] + (min_token[0] + min_token[1],) + seq_merge_cache[index+2:]
						# print current seq and seq_merge_cache
						# print(f"Merged {min_token} at index {index}: {seq_merge_cache}")
					else:
						index+=1
				
			# for merge_token in self.merges:
			# 	if merge_token not in pair_set:
			# 		continue
			# 	# find matched pairs and replace with merge token
			# 	index = 0
			# 	while index < len(seq_merge_cache) - 1:
			# 		if merge_token == (seq_merge_cache[index], seq_merge_cache[index+1]):
			# 			# merge
			# 			seq_merge_cache = seq_merge_cache[:index] + (merge_token[0] + merge_token[1],) + seq_merge_cache[index+2:]
			# 			# print current seq and seq_merge_cache
			# 			# print(f"Merged {merge_token} at index {index}: {seq_merge_cache}")

			# 			# add previous and next pairs to pair_set
			# 			if index > 0:
			# 				pair_set.add((seq_merge_cache[index-1], seq_merge_cache[index]))
			# 			if index < len(seq_merge_cache) - 1:
			# 				pair_set.add((seq_merge_cache[index], seq_merge_cache[index+1]))
			# 		else:
			# 			index+=1

			# 	# pair_set.discard(merge_token)

			# Map tokens to IDs
			seq_as_Ids: list[int] = []
			for token in seq_merge_cache:
				seq_as_Ids.append(self.reversed_vocab.get(token,-1))
			result_seq_Ids.extend(seq_as_Ids)
			
		return result_seq_Ids

	def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
		"""Lazily yield token IDs for an iterable of strings."""
		for text in iterable:
			print(f"[DEBUG] Encoding text: {text}")
			yield from self.encode(text)

	def decode(self, ids: list[int]) -> str:
		"""Decode a sequence of token IDs into text."""
		raise NotImplementedError

