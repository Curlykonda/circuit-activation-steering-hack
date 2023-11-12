from dataclasses import dataclass

from data_processing.dataloader import get_dataloader
from model_interfaces.models import get_model


@dataclass
class HyperParams:
    batch_size: int
    max_seq_len: int


def _get_default_params() -> HyperParams:
    return HyperParams(
        batch_size=50,
        max_seq_len=64,
    )


def main():

    model = get_model()
    h_params = _get_default_params()

    dataloader = get_dataloader(
        data_file="data/231112_11-09_agreeableness.jsonl",
        batch_size=h_params.batch_size,
        tokenizer=model.tokenizer,
        max_length=h_params.max_seq_len,
    )

    for batch in dataloader:
        positive, negative = batch
        print(positive["input_ids"])
        print(negative["input_ids"])
        break


if __name__ == "__main__":
    main()
