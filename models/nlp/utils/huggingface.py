from __future__ import annotations

import tempfile
import zipfile
from typing import TYPE_CHECKING

import valohai
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainerCallback,
)

if TYPE_CHECKING:
    from transformers.models.auto.auto_factory import _BaseAutoModelClass


def load_huggingface_model_and_tokenizer(
    model_spec: str,
    model_type: type[_BaseAutoModelClass],
    tokenizer_type: type[PreTrainedTokenizer] = AutoTokenizer,
) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
    """
    Load a HuggingFace model and tokenizer from a model spec (pathname to a .zip or a model identifier on Hub).
    """
    if model_spec.endswith(".zip"):
        return _load_from_zip(model_spec, model_type=model_type, tokenizer_type=tokenizer_type)
    tokenizer = tokenizer_type.from_pretrained(model_spec)
    model = model_type.from_pretrained(model_spec)
    return tokenizer, model


def _load_from_zip(
    model_spec: str,
    *,
    model_type: type[_BaseAutoModelClass],
    tokenizer_type: type[PreTrainedTokenizer],
):
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(model_spec, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        tokenizer = tokenizer_type.from_pretrained(temp_dir)
        model = model_type.from_pretrained(temp_dir)
    return tokenizer, model


class PrintMetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        with valohai.logger() as logger:
            logger.log(**metrics)
