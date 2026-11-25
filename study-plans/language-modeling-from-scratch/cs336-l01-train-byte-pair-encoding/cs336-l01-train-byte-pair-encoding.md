# Train a Deterministic BPE Vocabulary

Byte-pair encoding (BPE) is the tokenizer training algorithm behind GPT-2 and most modern language models. It starts from raw bytes and repeatedly glues the most common neighbouring pair into a new token, so frequent pieces of text end up as single tokens. The heart of this problem is making every choice deterministic, so the same corpus always produces the same vocabulary and merge list.

## Why language models need a learned vocabulary

A language model reads integer IDs, not text. The simplest choice is one ID per byte, which covers every possible string with only 256 tokens. The cost is length, because a single English word becomes five or six tokens and the model must spend attention and compute on every one of them.

The opposite choice is one ID per word. Sequences become short, but the vocabulary explodes, and any word missing from it cannot be represented at all.

BPE sits between the two. It keeps the 256 byte tokens, so nothing is ever unrepresentable, and adds new tokens only for byte sequences that appear often. Common words become one token, while rare words fall back to smaller pieces.

## Start from UTF-8 bytes

Each string is first converted to its UTF-8 bytes. ASCII characters take one byte each, so "a" becomes 97 and "b" becomes 98. Characters outside ASCII take two to four bytes, which means a single accented letter or emoji starts out as several tokens.

The initial vocabulary maps ID $i$ to the single byte $i$ for $0 \le i \le 255$. These 256 tokens always exist and count toward vocab_size, so a target of 300 allows at most 44 learned tokens.

## Count adjacent pairs

Let each string in the corpus be a sequence of token IDs $t_1, t_2, \ldots, t_n$. A pair $(a, b)$ is counted every time $t_i = a$ and $t_{i+1} = b$ in the same string:

$$
\text{count}(a, b) = \sum_{\text{strings}} \left|\{\, i : t_i = a,\ t_{i+1} = b \,\}\right|
$$

Two details matter here.

- Pairs never cross string boundaries. The last token of one string and the first token of the next string are not neighbours, because the strings are separate documents.
- Overlapping pairs all count. In "aaaa" the pair (97, 97) starts at positions 1, 2 and 3, so its count is 3.

## Merge the winner without overlap

The pair with the highest count becomes a new token. Its ID is the next unused integer, starting at 256, and its bytes are the left token's bytes followed by the right token's bytes.

Every occurrence of the pair is then replaced, scanning each string from left to right. When a match is replaced, the scan continues after both tokens, so a token is never used by two merges in the same round. For "aaaa", the scan replaces positions 1 and 2, then positions 3 and 4, producing two copies of the new token. Counting found three pairs, but only two replacements fit.

After the replacement, pair counts are recomputed on the new sequences and the next round begins. Training stops when the vocabulary reaches vocab_size or when no string has two tokens left to pair.

## Breaking ties deterministically

Two pairs often share the highest count, especially in small corpora. A rule is needed so that training is reproducible.

This problem compares the byte strings of the two tokens, not their IDs. Pair $(a, b)$ beats pair $(c, d)$ when the bytes of $a$ are greater than the bytes of $c$, or when they are equal and the bytes of $b$ are greater than the bytes of $d$. Byte strings compare like words in a dictionary, one byte at a time.

The difference matters because learned tokens have large IDs but can hold small bytes. Token 256 might stand for "ab", which sorts before the single byte "c" even though 256 is greater than 99.

## A two-round example

Train on the corpus ["abab", "abc"] with vocab_size 258. The bytes are $a = 97$, $b = 98$ and $c = 99$.

In round one, the sequences are $[97, 98, 97, 98]$ and $[97, 98, 99]$. The counts are

$$
\text{count}(97, 98) = 3, \quad \text{count}(98, 97) = 1, \quad \text{count}(98, 99) = 1
$$

The pair (97, 98) wins clearly and becomes token 256 with bytes "ab". The sequences become $[256, 256]$ and $[256, 99]$.

In round two, the pairs (256, 256) and (256, 99) both have count 1. Their left tokens are both "ab", so the right tokens decide. The bytes "c" are greater than "ab" because 99 is greater than 97 at the first byte. The pair (256, 99) wins and becomes token 257 with bytes "abc".

Comparing IDs instead would have picked (256, 256), because 256 is greater than 99. That would have produced a different vocabulary from the same corpus.

The vocabulary now holds 258 tokens, so training stops with merges (97, 98, 256) and (256, 99, 257).

## Common mistakes to avoid

- **Counting across strings.** Joining the corpus into one sequence creates pairs between the end of one string and the start of the next.
- **Breaking ties by ID.** Learned tokens have high IDs but can hold short or early-sorting bytes, so ID order and byte order disagree.
- **Overlapping replacements.** Replacing every position where the pair starts turns "aaaa" into three new tokens that share bytes, which is impossible.
- **Forgetting the base tokens in the size.** vocab_size includes the 256 byte tokens, so the number of merges is vocab_size minus 256 at most.
- **Building new token bytes from IDs.** A merged token's bytes come from concatenating the two tokens' byte strings, not from the IDs themselves.
- **Not stopping when nothing can merge.** A corpus of single-character strings has no pairs, so training must end even if vocab_size is far away.
