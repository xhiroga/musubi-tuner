import os

import pytest
import torch

from musubi_tuner.fpack_generate_video import load_optimized_dit_model_with_lora


@pytest.mark.adhoc
def test_load_optimized_dit_model_with_lora():
    dit_path = os.environ.get("DIT_PATH")
    lora_path = os.environ.get("LORA_PATH")
    lora_multiplier = os.environ.get("LORA_MULTIPLIER")
    optimized_model_dir = os.environ.get("OPTIMIZED_MODEL_DIR")
    if not dit_path or not lora_path or not lora_multiplier or not optimized_model_dir:
        assert False, (
            "DIT_PATH, LORA_PATH, LORA_MULTIPLIER, or OPTIMIZED_MODEL_DIR is not set"
        )

    result = load_optimized_dit_model_with_lora(
        dit_path=dit_path,
        fp8_scaled=True,
        lora_weight=[lora_path],
        lora_multiplier=[float(lora_multiplier)],
        fp8=False,
        blocks_to_swap=0,
        attn_mode="sdpa",
        rope_scaling_timestep_threshold=0,
        rope_scaling_factor=0,
        optimized_model_dir=optimized_model_dir,
        device=torch.device("cuda"),
        include_patterns=[],
        exclude_patterns=[],
        lycoris=False,
        save_merged_model=None,
    )
    assert result is not None
    assert hasattr(result, "eval")
