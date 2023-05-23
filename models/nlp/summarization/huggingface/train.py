import tempfile

import torch
import valohai
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizerBase,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from models.nlp.utils.huggingface import PrintMetricsCallback
from utils.torch import get_preferred_torch_device


def preprocess_function(
    examples: dict[str, str | list],
    tokenizer: PreTrainedTokenizerBase,
    text_max_length: int,
    summary_max_length: int,
) -> dict[str, torch.Tensor]:
    inputs = examples["text"]
    model_inputs = tokenizer(inputs, max_length=text_max_length, truncation=True)
    labels = tokenizer(
        text_target=examples["summary"],
        max_length=summary_max_length,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    model_name = valohai.parameters("huggingface_repository").value
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset(valohai.inputs("dataset").dir_path())

    dataset = dataset["train"].train_test_split(
        test_size=valohai.parameters("test_split_size").value,
        seed=valohai.parameters("seed").value,
    )

    device = get_preferred_torch_device()
    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        fn_kwargs={
            "tokenizer": tokenizer,
            "text_max_length": valohai.parameters("text_max_length").value,
            "summary_max_length": valohai.parameters("summary_max_length").value,
        },
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
    ).to(device)

    with tempfile.TemporaryDirectory() as temp_dir:
        training_args = Seq2SeqTrainingArguments(
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

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name),
            callbacks=[PrintMetricsCallback()],
        )

        trainer.train()
        model.save_pretrained(valohai.outputs().dir_path)
        tokenizer.save_pretrained(valohai.outputs().dir_path)
        valohai.outputs().compress(source="*.*", filename="model.zip")


if __name__ == "__main__":
    main()
