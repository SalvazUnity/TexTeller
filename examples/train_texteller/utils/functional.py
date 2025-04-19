from typing import Any

import torch
from transformers import DataCollatorForLanguageModeling

from texteller.constants import MAX_TOKEN_SIZE, MIN_HEIGHT, MIN_WIDTH


def _left_move(x: torch.Tensor, pad_val):
    assert len(x.shape) == 2, "x should be 2-dimensional"
    lefted_x = torch.ones_like(x)
    lefted_x[:, :-1] = x[:, 1:]
    lefted_x[:, -1] = pad_val
    return lefted_x


def tokenize_fn(samples: dict[str, list[Any]], tokenizer=None) -> dict[str, list[Any]]:
    assert tokenizer is not None, "tokenizer should not be None"
    tokenized_formula = tokenizer(samples["latex_formula"], return_special_tokens_mask=True)
    tokenized_formula["pixel_values"] = samples["image"]
    return tokenized_formula


def collate_fn(samples: list[dict[str, Any]], tokenizer=None) -> dict[str, list[Any]]:
    assert tokenizer is not None, "tokenizer should not be None"
    pixel_values = [dic.pop("pixel_values") for dic in samples]

    clm_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    batch = clm_collator(samples)
    batch["pixel_values"] = pixel_values
    batch["decoder_input_ids"] = batch.pop("input_ids")
    batch["decoder_attention_mask"] = batch.pop("attention_mask")

    batch["labels"] = _left_move(batch["labels"], -100)

    # convert list of Image to a tensor with (B, C, H, W)
    batch["pixel_values"] = torch.stack(batch["pixel_values"], dim=0)
    return batch


def filter_fn(sample, tokenizer=None) -> bool:
    return (
        sample["image"].height > MIN_HEIGHT
        and sample["image"].width > MIN_WIDTH
        and len(tokenizer(sample["latex_formula"])["input_ids"]) < MAX_TOKEN_SIZE - 10
    )
