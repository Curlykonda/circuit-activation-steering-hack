from enum import Enum
from typing import Tuple, Union

from transformers import GPT2LMHeadModel, GPT2Tokenizer

from .utils import get_device


class ModelType(Enum):
    GPT2 = "gpt2"


def get_model(model_name: Union[str, ModelType]):

    if isinstance(model_name, str):
        model_type = ModelType(model_name)

    if model_type == ModelType.GPT2:
        return _load_gpt2()
    else:
        raise ValueError(f"Unknown model: {model_type.value}")


def _load_gpt2() -> Tuple[GPT2LMHeadModel, GPT2Tokenizer]:

    device = get_device()

    tokenizer = GPT2Tokenizer.from_pretrained(ModelType.GPT2.value)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = GPT2LMHeadModel.from_pretrained(
        ModelType.GPT2.value, pad_token_id=tokenizer.eos_token_id
    )
    model.to(device)
    return model, tokenizer
