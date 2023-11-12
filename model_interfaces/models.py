from transformer_lens import HookedTransformer

from model_interfaces.utils import get_device


def get_model() -> HookedTransformer:

    device = get_device()

    model = HookedTransformer.from_pretrained(
        "gpt2-small",
        center_writing_weights=False,
        center_unembed=False,
        fold_ln=False,
        device=device,
    )
    model.set_use_hook_mlp_in(True)
    model.set_use_split_qkv_input(True)
    model.set_use_attn_result(True)

    return model
