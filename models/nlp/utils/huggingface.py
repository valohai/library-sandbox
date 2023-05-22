import tempfile
import zipfile

import valohai
from transformers import (
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainerCallback,
)


def load_huggingface_model_and_tokenizer(
    archive_path: str,
    model_type: type[
        AutoModelForQuestionAnswering
        | AutoModelForSeq2SeqLM
        | AutoModelForSequenceClassification
    ],
    tokenizer_type: type[PreTrainedTokenizer] = AutoTokenizer,
) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
    if not archive_path.endswith(".zip"):
        raise ValueError(f"Model {archive_path} is not a zip file. Aborting.")

    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        tokenizer = tokenizer_type.from_pretrained(temp_dir)
        model = model_type.from_pretrained(temp_dir)

    return tokenizer, model


class PrintMetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        with valohai.logger() as logger:
            logger.log(**metrics)
