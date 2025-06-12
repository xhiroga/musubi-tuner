import os

import pytest
import torch

from musubi_tuner.fpack_generate_video import load_optimized_model


@pytest.mark.adhoc
def test_load_optimized_model():
    dit_path = os.environ.get("DIT_PATH")
    lora_path = os.environ.get("LORA_PATH")
    lora_multiplier = os.environ.get("LORA_MULTIPLIER")
    if not all([dit_path, lora_path, lora_multiplier]):
        pytest.skip("DIT_PATH, LORA_PATH, LORA_MULTIPLIER is not set")

    model = load_optimized_model(
        dit_path=dit_path,
        fp8_scaled=True,
        lora_weight=[lora_path],
        lora_multiplier=[float(lora_multiplier)],
        fp8=False,
        blocks_to_swap=0,
        attn_mode="sdpa",
        rope_scaling_timestep_threshold=0,
        rope_scaling_factor=0,
        optimized_model_dir="/tmp",
        device=torch.device("cuda"),
        include_patterns=[],
        exclude_patterns=[],
        lycoris=False,
        save_merged_model=None,
    )
    assert model is not None
    assert hasattr(model, "eval")
    for name, param in model.named_parameters():
        # print(f"{name}: {param.dtype}, shape: {param.shape}")
        # 最適化のロジックは [hunyuan_video_packed.py#L1958-L1959](https://github.com/xhiroga/musubi-tuner-xhiroga/blob/15800869e6d17abb915307d6c69b7d6164b28433/src/musubi_tuner/frame_pack/hunyuan_video_packed.py#L1958-L1959)
        # 入力埋め込み, Contenxt処理、出力のProjectionはfp8最適化しない
        if name.startswith("x_embedder") or name.startswith("context_embedder") or name.startswith("image_projection"):
            assert param.dtype == torch.bfloat16, f"{name=}, {param.dtype=}"
        # DiTのうち、重みは基本的にfp8最適化する
        elif (
            name.endswith(".weight")
            and (name.startswith("transformer_blocks") or name.startswith("single_transformer_blocks"))
            and "norm" not in name
        ):
            assert param.dtype == torch.float8_e4m3fn, f"{name=}, {param.dtype=}"
        else:
            assert param.dtype == torch.bfloat16, f"{name=}, {param.dtype=}"
