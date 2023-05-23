import json
import tempfile
from typing import Any

import valohai
from datasets import load_dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)

from models.nlp.utils.huggingface import PrintMetricsCallback
from utils.torch import get_preferred_torch_device


def prepare_train_features(
    examples: dict[str, Any],
    tokenizer: AutoTokenizer,
    max_length: int,
    doc_stride: int,
) -> dict[str, Any]:
    pad_on_right = tokenizer.padding_side == "right"
    # Some of the questions have lots of whitespace, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    examples["question"] = [q.strip() for q in examples["question"]]

    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = json.loads(examples["answers"][sample_index])

        # If no answers are given, set the cls_index as answer.
        if not answers["answer_start"]:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            start_position, end_position = compute_start_end_positions(
                start_char,
                end_char,
                input_ids,
                offsets,
                sequence_ids,
                pad_on_right,
                cls_index,
            )
            tokenized_examples["start_positions"].append(start_position)
            tokenized_examples["end_positions"].append(end_position)

    return tokenized_examples


def compute_start_end_positions(
    start_char: int,
    end_char: int,
    input_ids: list[int],
    offsets: list[tuple[int, int]],
    sequence_ids: list[int],
    pad_on_right: bool,
    cls_index: int,
) -> tuple[int, int]:
    # Start/end token index of the answer in the text
    token_start_index = 0
    while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
        token_start_index += 1

    token_end_index = len(input_ids) - 1
    while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
        token_end_index -= 1

    # Detect if the answer is out of the span
    if not (
        offsets[token_start_index][0] <= start_char
        and offsets[token_end_index][1] >= end_char
    ):
        return cls_index, cls_index

    # Otherwise move the token_start_index and token_end_index to the two ends of the answer
    while (
        token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char
    ):
        token_start_index += 1
    while offsets[token_end_index][1] >= end_char:
        token_end_index -= 1

    return token_start_index - 1, token_end_index + 1


def get_answer_positions(
    answers: dict[str, Any],
    offsets: list[tuple[int, int]],
    cls_index: int,
    token_start_index: int,
    token_end_index: int,
    start_char: int,
    end_char: int,
) -> tuple[int, int]:
    if not answers["answer_start"]:
        return cls_index, cls_index

    # Start/end character index of the answer in the text.
    start_char = answers["answer_start"][0]
    end_char = start_char + len(answers["text"][0])

    # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
    if not (
        offsets[token_start_index][0] <= start_char
        and offsets[token_end_index][1] >= end_char
    ):
        return cls_index, cls_index

    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
    # Note: we could go after the last offset if the answer is the last word (edge case).
    while (
        token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char
    ):
        token_start_index += 1
    while offsets[token_end_index][1] >= end_char:
        token_end_index -= 1
    return token_start_index - 1, token_end_index + 1


def main():
    model_name = valohai.parameters("huggingface_repository").value
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = load_dataset(valohai.inputs("dataset").dir_path())
    dataset = dataset["train"].train_test_split(
        test_size=valohai.parameters("test_split_size").value,
        seed=valohai.parameters("seed").value,
    )
    tokenized_datasets = dataset.map(
        prepare_train_features,
        batched=True,
        remove_columns=dataset["train"].column_names,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": valohai.parameters("max_length").value,
            "doc_stride": valohai.parameters("doc_stride").value,
        },
    )

    device = get_preferred_torch_device()
    model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)

    with tempfile.TemporaryDirectory() as temp_dir:
        training_args = TrainingArguments(
            output_dir=temp_dir,
            evaluation_strategy="steps",
            per_device_train_batch_size=valohai.parameters("batch_size").value,
            per_device_eval_batch_size=valohai.parameters("batch_size").value,
            eval_steps=valohai.parameters("eval_steps").value,
            max_steps=valohai.parameters("max_steps").value,
            num_train_epochs=valohai.parameters("num_train_epochs").value,
            learning_rate=valohai.parameters("learning_rate").value,
            weight_decay=valohai.parameters("weight_decay").value,
            warmup_steps=valohai.parameters("warmup_steps").value,
            adam_beta1=valohai.parameters("adam_beta1").value,
            adam_beta2=valohai.parameters("adam_beta2").value,
            adam_epsilon=valohai.parameters("adam_epsilon").value,
            max_grad_norm=valohai.parameters("max_grad_norm").value,
            seed=valohai.parameters("seed").value,
            disable_tqdm=valohai.parameters("disable_tqdm").value,
        )

        print(training_args)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            data_collator=DefaultDataCollator(),
            tokenizer=tokenizer,
            callbacks=[PrintMetricsCallback()],
        )

        trainer.train()
        model.save_pretrained(valohai.outputs().dir_path)
        tokenizer.save_pretrained(valohai.outputs().dir_path)
        valohai.outputs().compress(source="*.*", filename="model.zip")


if __name__ == "__main__":
    main()
