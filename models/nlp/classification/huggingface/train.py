import tempfile

import valohai
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from models.nlp.utils.huggingface import PrintMetricsCallback
from utils.torch import get_preferred_torch_device


def main():
    model_name = valohai.parameters("huggingface_repository").value
    num_labels = valohai.parameters("num_labels").value

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    dataset = load_dataset(valohai.inputs("dataset").dir_path())
    dataset = dataset["train"].train_test_split(
        test_size=valohai.parameters("test_split_size").value,
        seed=valohai.parameters("seed").value,
    )

    if num_labels == 0:
        print("num_labels == 0. Auto-detecting num_labels...")
        unique_labels = {example["label"] for example in dataset["train"]}
        num_labels = len(unique_labels)
        print(f"Found {num_labels} unique labels in the dataset")

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    device = get_preferred_torch_device()
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    ).to(device)

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
            callbacks=[PrintMetricsCallback()],
        )

        trainer.train()
        model.save_pretrained(valohai.outputs().dir_path)
        tokenizer.save_pretrained(valohai.outputs().dir_path)
        valohai.outputs().compress(source="*.*", filename="model.zip")


if __name__ == "__main__":
    main()
