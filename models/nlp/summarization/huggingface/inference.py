import torch
import valohai
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from models.nlp.utils.huggingface import load_huggingface_model_and_tokenizer
from utils.serializers import get_serializer


def predict(
    text: str | list[str],
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    device: torch.device,
    max_summary_length: int,
) -> str:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).input_ids
    inputs = inputs.to(device)
    outputs = model.generate(
        inputs,
        max_new_tokens=max_summary_length,
        do_sample=False,
    )
    outputs = outputs.cpu()
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    tokenizer, model, device = load_huggingface_model_and_tokenizer(
        valohai.inputs("model").path(process_archives=False),
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
    )

    data_path = valohai.inputs("data").path("*.txt")
    max_summary_length = valohai.parameters("max_summary_length").value
    log_frequency = valohai.parameters("log_frequency").value

    with open(data_path) as f:
        results = []
        for index, row in enumerate(f, 1):
            text = row.strip()
            prediction = predict(text, tokenizer, model, device, max_summary_length)
            results.append([text, prediction])
            if index % log_frequency == 0:
                print(f"Processed {index} items")

    output_path = valohai.outputs().path(valohai.parameters("output_path").value)
    serializer = get_serializer(output_path)
    serializer.serialize(results, columns=["text", "summary"])


if __name__ == "__main__":
    main()
