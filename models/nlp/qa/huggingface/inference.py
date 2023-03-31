import csv

import torch
import valohai
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from models.nlp.utils.huggingface import load_huggingface_model_and_tokenizer
from utils.serializers import get_serializer


def load_data(data_path: str) -> list[dict[str, str]]:
    data = []
    with open(data_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(
                {
                    "id": row["id"],
                    "context": row["context"],
                    "question": row["question"],
                },
            )
    return data


def predict(
    question: dict[str, str],
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    device: torch.device,
) -> str:
    inputs = tokenizer(question["question"], question["context"], return_tensors="pt")
    inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()
    predict_answer_tokens = inputs["input_ids"][
        0,
        answer_start_index : answer_end_index + 1,
    ].cpu()
    return tokenizer.decode(predict_answer_tokens)


def main():
    tokenizer, model, device = load_huggingface_model_and_tokenizer(
        valohai.inputs("model").path(process_archives=False),
        AutoModelForQuestionAnswering,
        AutoTokenizer,
    )
    data_path = valohai.inputs("data").path("*.csv")
    rows = load_data(data_path)
    log_frequency = valohai.parameters("log_frequency").value

    results = []
    for index, row in enumerate(rows, 1):
        answer = predict(row, tokenizer, model, device)
        results.append({**row, "answer": answer})

        if index % log_frequency == 0:
            print(f"Processed {index} items")

    output_path = valohai.outputs().path(valohai.parameters("output_path").value)
    serializer = get_serializer(output_path)
    serializer.serialize(results, ["id", "context", "question", "answer"])


if __name__ == "__main__":
    main()
