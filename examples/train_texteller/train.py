from functools import partial

import yaml
from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
)

from texteller import load_model, load_tokenizer
from texteller.constants import MIN_HEIGHT, MIN_WIDTH

from examples.train_texteller.utils import (
    collate_fn,
    filter_fn,
    img_inf_transform,
    img_train_transform,
    tokenize_fn,
)


def train(model, tokenizer, train_dataset, eval_dataset, collate_fn_with_tokenizer):
    training_args = TrainingArguments(**training_config)
    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn_with_tokenizer,
    )

    trainer.train(resume_from_checkpoint=None)


if __name__ == "__main__":
    dataset = load_dataset("imagefolder", data_dir="dataset")["train"]
    dataset = dataset.filter(
        lambda x: x["image"].height > MIN_HEIGHT and x["image"].width > MIN_WIDTH
    )
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.flatten_indices()

    tokenizer = load_tokenizer()
    # If you want use your own tokenizer, please modify the path to your tokenizer
    # tokenizer = load_tokenizer("/path/to/your/tokenizer")
    filter_fn_with_tokenizer = partial(filter_fn, tokenizer=tokenizer)
    dataset = dataset.filter(filter_fn_with_tokenizer, num_proc=8)

    map_fn = partial(tokenize_fn, tokenizer=tokenizer)
    tokenized_dataset = dataset.map(
        map_fn, batched=True, remove_columns=dataset.column_names, num_proc=8
    )

    # Split dataset into train and eval, ratio 9:1
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset, eval_dataset = split_dataset["train"], split_dataset["test"]
    train_dataset = train_dataset.with_transform(img_train_transform)
    eval_dataset = eval_dataset.with_transform(img_inf_transform)
    collate_fn_with_tokenizer = partial(collate_fn, tokenizer=tokenizer)

    # Train from scratch
    model = load_model()

    # If you want to train from pre-trained model, please modify the path to your pre-trained checkpoint
    # model = load_model("/path/to/your/model_checkpoint")

    enable_train = True
    training_config = yaml.safe_load(open("train_config.yaml"))
    if enable_train:
        train(model, tokenizer, train_dataset, eval_dataset, collate_fn_with_tokenizer)
