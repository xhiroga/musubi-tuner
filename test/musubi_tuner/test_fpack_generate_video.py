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
        lora_weight=[lora_path],
        lora_multiplier=[float(lora_multiplier)],
        fp8_scaled=True,
        fp8=False,
        blocks_to_swap=0,
        attn_mode="sdpa",
        rope_scaling_timestep_threshold=0,
        rope_scaling_factor=0,
        optimized_model_dir=None,
        device=torch.device("cuda"),
        include_patterns=[],
        exclude_patterns=[],
        lycoris=False,
        save_merged_model=None,
    )
    assert model is not None
    assert hasattr(model, "eval")

    for name, module in model.named_modules():
        # 最適化のロジックは [hunyuan_video_packed.py#L1958-L1959](src/musubi_tuner/frame_pack/hunyuan_video_packed.py#L1958-L1959)
        # DiTのうち、正規化層を除いた重みがfp8最適化される
        if (name.startswith("transformer_blocks.") or name.startswith("single_transformer_blocks.")) and "norm" not in name:
            # コンテナモジュールを除く
            if hasattr(module, "weight"):
                assert hasattr(module, "scale_weight"), f"{name=}, {module=}"
                assert module.weight.dtype == torch.float8_e4m3fn, f"{name=}, {module.weight.dtype=}"
                assert module.scale_weight.dtype == torch.bfloat16, f"{name=}, {module.scale_weight.dtype=}"
        # そのほかの場合（入力埋め込み, Contenxt処理、出力のProjection等）はfp8最適化しない
        else:
            pass

@pytest.mark.adhoc
def test_load_optimized_model_from_disk():
    dit_path = os.environ.get("DIT_PATH")
    lora_path = os.environ.get("LORA_PATH")
    lora_multiplier = os.environ.get("LORA_MULTIPLIER")
    optimized_model_dir = os.environ.get("OPTIMIZED_MODEL_DIR")
    if not all([dit_path, lora_path, lora_multiplier, optimized_model_dir]):
        pytest.skip("DIT_PATH, LORA_PATH, LORA_MULTIPLIER, OPTIMIZED_MODEL_DIR is not set")

    model = load_optimized_model(
        dit_path=dit_path,
        lora_weight=[lora_path],
        lora_multiplier=[float(lora_multiplier)],
        fp8_scaled=True,
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
    assert model is not None
    assert hasattr(model, "eval")
