import torch
import valohai
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from models.nlp.utils.huggingface import (
    load_huggingface_model_and_tokenizer_from_config,
)
from utils.serializers import get_serializer
from utils.torch import get_preferred_torch_device


@torch.no_grad()
def predict(
    text: str,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    device: torch.device,
) -> int:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    return int(predictions.item())


def main():
    device = get_preferred_torch_device()
    model, tokenizer = load_huggingface_model_and_tokenizer_from_config(
        model_type=AutoModelForSequenceClassification,
        tokenizer_type=AutoTokenizer,
    )
    model.to(device)
    data_path = valohai.inputs("data").path("*.txt")
    log_frequency = valohai.parameters("log_frequency").value

    with open(data_path) as f:
        results = []
        for index, row in enumerate(f, 1):
            text = row.strip()
            prediction = predict(text, tokenizer, model, device)
            results.append([text, prediction])
            if index % log_frequency == 0:
                print(f"Processed {index} items")

    output_path = valohai.outputs().path(valohai.parameters("output_path").value)
    serializer = get_serializer(output_path)
    serializer.serialize(results, ["text", "label"])


if __name__ == "__main__":
    main()
