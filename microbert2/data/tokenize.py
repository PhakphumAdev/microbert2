import os
import pprint
import random
import shutil
from typing import Any, Iterable, List, Literal, Optional, Tuple

from tango import DillFormat, Step
from tango.common import Lazy, Tqdm
from tango.integrations.transformers import Tokenizer

from microbert2.microbert.tasks.task import MicroBERTTask
from microbert2.tokenizers import train_tokenizer


@Step.register("microbert2.data.tokenize::subword_tokenize")
class SubwordTokenize(Step):
    """
    Use a pretrained transformer tokenizer to get inputs necessary for the language model. Also,
    note which wordpieces belong to whole tokens in the original tokenization.
    """

    DETERMINISTIC = True
    CACHEABLE = True
    FORMAT = DillFormat()

    def _intra_word_tokenize(
        self,
        string_tokens: List[str],
        tokenizer: Tokenizer,
        max_wordpieces: int,
    ) -> Tuple[List[int], List[Optional[Tuple[int, int]]], bool]:
        tokens = []
        offsets = []
        truncated = False
        for i, token_string in enumerate(string_tokens):
            # Convert None to string "None" to match tokenizer training
            token_string = "None" if token_string is None else str(token_string)
            wordpieces = tokenizer.encode_plus(
                token_string,
                add_special_tokens=False,
                return_tensors=None,
                return_offsets_mapping=False,
                return_attention_mask=False,
            )
            wp_ids = wordpieces["input_ids"]

            # Stop early if adding this token would exceed our budget
            if len(tokens) + len(wp_ids) > max_wordpieces:
                self.logger.warning(
                    f"Stopping at token {i} in sentence with {len(string_tokens)} tokens due to wordpiece limit"
                )
                truncated = True
                break

            if len(wp_ids) > 0:
                tokens.extend(wp_ids)
                offsets.append((len(tokens) - len(wp_ids), len(tokens) - 1))
            else:
                tokens.append(tokenizer.unk_token_id)
                offsets.append((len(tokens) - 1, len(tokens) - 1))
        return tokens, offsets, truncated

    @staticmethod
    def _increment_offsets(
        offsets: Iterable[Optional[Tuple[int, int]]], increment: int
    ) -> List[Optional[Tuple[int, int]]]:
        return [None if offset is None else (offset[0] + increment, offset[1] + increment) for offset in offsets]

    def intra_word_tokenize(
        self,
        string_tokens: List[str],
        tokenizer: Tokenizer,
        max_wordpieces: int,
    ) -> Tuple[List[int], List[Optional[Tuple[int, int]]], bool]:
        """
        Tokenizes each word into wordpieces separately and returns the wordpiece IDs.
        Also calculates offsets such that tokens[offsets[i][0]:offsets[i][1] + 1]
        corresponds to the original i-th token.
        This function inserts special tokens.
        """
        wp_ids, offsets, truncated = self._intra_word_tokenize(string_tokens, tokenizer, max_wordpieces - 2)
        # Handle special tokens
        wp_ids = [tokenizer.cls_token_id] + wp_ids + [tokenizer.sep_token_id]
        offsets = self._increment_offsets(offsets, 1)
        offsets = [(0, 0)] + offsets + [(offsets[-1][1] + 1,) * 2]
        return wp_ids, offsets, truncated

    def intra_word_tokenize_pair(
        self,
        src_tokens: List[str],
        tgt_tokens: List[str],
        tokenizer: Tokenizer,
        max_wordpieces: int,
    ) -> Tuple[List[int], List[int], List[Optional[Tuple[int, int]]], bool]:
        """
        Tokenize a parallel pair as [CLS] src [SEP] tgt [SEP].
        Returns (input_ids, token_type_ids, token_spans, truncated).
        token_type_ids is 0 for the [CLS]+src+[SEP] side and 1 for tgt+[SEP].
        """
        budget = max_wordpieces - 3  # [CLS], [SEP], [SEP]
        src_budget = budget // 2
        tgt_budget = budget - src_budget

        src_wp, src_offsets, src_trunc = self._intra_word_tokenize(src_tokens, tokenizer, src_budget)
        tgt_wp, tgt_offsets, tgt_trunc = self._intra_word_tokenize(tgt_tokens, tokenizer, tgt_budget)

        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id

        input_ids = [cls_id] + src_wp + [sep_id] + tgt_wp + [sep_id]
        token_type_ids = [0] * (1 + len(src_wp) + 1) + [1] * (len(tgt_wp) + 1)

        src_sep_pos = 1 + len(src_wp)
        tgt_sep_pos = src_sep_pos + 1 + len(tgt_wp)
        token_spans = (
            [(0, 0)]
            + self._increment_offsets(src_offsets, 1)
            + [(src_sep_pos, src_sep_pos)]
            + self._increment_offsets(tgt_offsets, src_sep_pos + 1)
            + [(tgt_sep_pos, tgt_sep_pos)]
        )

        return input_ids, token_type_ids, token_spans, src_trunc or tgt_trunc

    def _process_split(
        self,
        split: list,
        split_name: str,
        task_slug: str,
        tokenizer: Tokenizer,
        max_length: Optional[int],
        discard_truncated: bool = False,
    ) -> list:
        wp_count = 0
        sentence_count = 0
        token_count = 0

        sample = pprint.pformat(random.choice(split), indent=4, width=120, compact=True)
        self.logger.info(f"Sample inst from {task_slug}_{split_name}:\n\n\t{sample}\n")

        def inner():
            nonlocal wp_count, sentence_count, token_count
            for d in Tqdm.tqdm(split, desc=f"Tokenizing {task_slug} ({split_name})"):
                sentence = d["tokens"]
                if "target_tokens" in d:
                    wp_ids, token_type_ids, token_spans, truncated = self.intra_word_tokenize_pair(
                        sentence, d["target_tokens"], tokenizer, max_length
                    )
                else:
                    wp_ids, token_spans, truncated = self.intra_word_tokenize(sentence, tokenizer, max_length)
                    token_type_ids = [0] * len(wp_ids)
                flattened = []
                for pair in token_spans:
                    flattened.extend(pair)
                d = {
                    **d,
                    "input_ids": wp_ids,
                    "token_spans": flattened,
                    "attention_mask": [1] * len(wp_ids),
                    "token_type_ids": token_type_ids,
                }
                wp_count += len(wp_ids)
                sentence_count += 1
                token_count += len(token_spans) - 2
                yield d, truncated

        result = []
        num_discard = 0
        for x, truncated in inner():
            if discard_truncated and truncated:
                num_discard += 1
                continue
            result.append(x)
        self.logger.info(
            f"Split {split_name}: {sentence_count} sentences, {token_count} tokens, {wp_count} wordpieces"
            f", {num_discard} discarded"
        )
        return result

    def run(
        self,
        dataset: dict,
        tokenizer: Tokenizer,
        max_length: Optional[int] = None,
        tasks: list[MicroBERTTask] = [],
    ) -> list[dict[Literal["train", "dev", "test"], list[dict[str, Any]]]]:
        datasets = []
        for task in tasks:
            task_dataset = {
                k: self._process_split(v, k, task.slug, tokenizer, max_length, discard_truncated=True)
                for k, v in task.dataset.items()
            }
            datasets.append(task_dataset)
        return datasets


@Step.register("microbert2.data.tokenize::train_tokenizer")
class TrainTokenizer(Step):
    DETERMINISTIC = True
    CACHEABLE = True
    FORMAT = DillFormat()

    def run(
        self,
        dataset: dict,
        model_path: str,
        tasks: list[MicroBERTTask] = [],
        vocab_size: Optional[int] = None,
        lowercase: bool = True,
        nfd_normalize: bool = True,
        strip_accents: bool = False,
    ) -> Tokenizer:
        if os.path.exists(model_path):
            self.logger.info(f"Already found model at {model_path}. Removing...")
            shutil.rmtree(model_path)
        sentences = [x["tokens"] for x in dataset["train"]]
        tokens = [" ".join("None" if t is None else str(t) for t in s) for s in sentences]
        for task in tasks:
            for sentence in task.dataset["train"]:
                tokens.append(" ".join("None" if t is None else str(t) for t in sentence.get("tokens", [])))
                if "target_tokens" in sentence:
                    tokens.append(" ".join("None" if t is None else str(t) for t in sentence["target_tokens"]))
        tokenizer = train_tokenizer(
            tokens,
            model_path,
            vocab_size=vocab_size,
            lowercase=lowercase,
            nfd_normalize=nfd_normalize,
            strip_accents=strip_accents,
        )
        # simple_train_tokenizer(sentences, model_path)
        self.logger.info(f"Wrote tokenizer to {model_path}")
        return tokenizer
