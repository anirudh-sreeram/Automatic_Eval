import torch
from typing import Iterable, List


class LogitsProcessor:
    """Abstract base class for all logit processors that can be applied during generation."""

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    r"""
    N-grams are groups of "n" consecutive words, characters, or tokens taken from a sequence of text. Given the
    sentence: "She runs fast", the bi-grams (n=2) would be ("she", "runs") and ("runs", "fast"). In text generation,
    avoiding repetitions of word sequences provides a more diverse output. This [`LogitsProcessor`] enforces no
    repetition of n-grams by setting the scores of banned tokens to negative infinity which eliminates those tokens
    from consideration when further processing the scores.
    [Fairseq](https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345).
    """

    def __init__(self, ngram_size: int, tokenizer):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(
                f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}"
            )
        self.tokenizer = tokenizer
        self.ngram_size = ngram_size
        # Servicenow specific
        self.input_ngrams_size = {}
        #

    def _get_ngrams(
        self, ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int
    ):
        """
        Assume ngram_size=2 and prev_input_ids=tensor([[40, 2883, 2712, 4346]]). The output of generated ngrams look like
        this {(40,): [2883], (2883,): [2712], (2712,): [4346]}.

        Args:
            ngram_size (`int`):
                The number sequential tokens taken as a group which may only occur once before being banned.
            prev_input_ids (`torch.Tensor`):
            Generated token ids for the current hypothesis.
            num_hypos (`int`):
                The number of hypotheses for which n-grams need to be generated.

        Returns:
            generated_ngrams (`dict`):
                Dictionary of generated ngrams.
        """
        # Initialize an empty list of dictionaries, one for each hypothesis (index) in the range of num_hypos
        generated_ngrams = [{} for _ in range(num_hypos)]
        for idx in range(num_hypos):
            gen_tokens = prev_input_ids[idx].tolist()
            generated_ngram = generated_ngrams[idx]
            # Servicenow specific
            if idx not in self.input_ngrams_size:
                self.input_ngrams_size[idx] = len(gen_tokens)
                continue
            else:
                # strip the input from the generated input_ids
                gen_tokens = gen_tokens[self.input_ngrams_size[idx] :]
            #
            # Loop through each n-gram of size ngram_size in the list of tokens (gen_tokens)
            for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):
                prev_ngram_tuple = tuple(ngram[:-1])
                generated_ngram[prev_ngram_tuple] = generated_ngram.get(
                    prev_ngram_tuple, []
                ) + [ngram[-1]]
        return generated_ngrams

    def _get_generated_ngrams(self, banned_ngrams, prev_input_ids, ngram_size, cur_len):
        """
        Determines the banned tokens for the current hypothesis based on previously generated n-grams.

        Args:
            banned_ngrams (`dict`):
                A dictionary containing previously generated n-grams for each hypothesis.
            prev_input_ids (`torch.Tensor`):
                Generated token ids for the current hypothesis.
            ngram_size (`int`):
                The number sequential tokens taken as a group which may only occur once before being banned.
            cur_len (`int`):
                The current length of the token sequences for which the n-grams are being checked.

        Returns:
            List of tokens that are banned.
        """
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - ngram_size
        ngram_idx = tuple(prev_input_ids[start_idx:cur_len].tolist())
        # Servicenow specific
        # check for space - if generated ngram does not contain space, then no need to ban the next token
        decoded_ngram = self.tokenizer.batch_decode([ngram_idx])[0].strip()
        if " " not in decoded_ngram:
            return []
        #
        return banned_ngrams.get(ngram_idx, [])

    def _calc_banned_ngram_tokens(
        self,
        ngram_size: int,
        prev_input_ids: torch.Tensor,
        num_hypos: int,
        cur_len: int,
    ) -> List[Iterable[int]]:
        """Copied from fairseq for no_repeat_ngram in beam_search"""
        if cur_len + 1 < ngram_size:
            # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
            return [[] for _ in range(num_hypos)]
        generated_ngrams = self._get_ngrams(ngram_size, prev_input_ids, num_hypos)
        banned_tokens = [
            self._get_generated_ngrams(
                generated_ngrams[hypo_idx],
                prev_input_ids[hypo_idx],
                ngram_size,
                cur_len,
            )
            for hypo_idx in range(num_hypos)
        ]
        return banned_tokens

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        num_batch_hypotheses = scores.shape[0]
        cur_len = input_ids.shape[-1]
        banned_batch_tokens = self._calc_banned_ngram_tokens(
            self.ngram_size, input_ids, num_batch_hypotheses, cur_len
        )
        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")
        return scores
