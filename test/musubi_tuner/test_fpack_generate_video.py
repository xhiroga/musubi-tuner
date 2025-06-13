import os

import pytest
import torch

from musubi_tuner.fpack_generate_video import load_optimized_model


def assert_fp8_scaled_model(model: torch.nn.Module) -> None:
    for name, module in model.named_modules():
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
def test_load_optimized_model():
    dit_path = os.environ.get("DIT_PATH")
    lora_path = os.environ.get("LORA_PATH")
    lora_multiplier_str = os.environ.get("LORA_MULTIPLIER")
    if not all([dit_path, lora_path, lora_multiplier_str]):
        pytest.skip("DIT_PATH, LORA_PATH, LORA_MULTIPLIER is not set")
    
    assert dit_path is not None
    assert lora_path is not None
    assert lora_multiplier_str is not None
    lora_multiplier = float(lora_multiplier_str)

    model = load_optimized_model(
        dit_path=dit_path,
        lora_weight=[lora_path],
        lora_multiplier=[lora_multiplier],
        fp8_scaled=True,
        fp8=False,
        blocks_to_swap=0,
        attn_mode="sdpa",
        rope_scaling_timestep_threshold=0,
        rope_scaling_factor=0,
        cache_dir=None,
        device=torch.device("cuda"),
        include_patterns=[],
        exclude_patterns=[],
        lycoris=False,
        save_merged_model=None,
    )
    assert model is not None
    assert hasattr(model, "eval")
    assert_fp8_scaled_model(model)


@pytest.mark.adhoc
def test_load_optimized_model_from_disk():
    dit_path = os.environ.get("DIT_PATH")
    lora_path = os.environ.get("LORA_PATH")
    lora_multiplier_str = os.environ.get("LORA_MULTIPLIER")
    cache_dir = os.environ.get("CACHE_DIR")
    if not all([dit_path, lora_path, lora_multiplier_str, cache_dir]):
        pytest.skip("DIT_PATH, LORA_PATH, LORA_MULTIPLIER, CACHE_DIR is not set")

    assert dit_path is not None
    assert lora_path is not None
    assert lora_multiplier_str is not None
    assert cache_dir is not None
    lora_multiplier = float(lora_multiplier_str)

    model = load_optimized_model(
        dit_path=dit_path,
        lora_weight=[lora_path],
        lora_multiplier=[lora_multiplier],
        fp8_scaled=True,
        fp8=False,
        blocks_to_swap=0,
        attn_mode="sdpa",
        rope_scaling_timestep_threshold=0,
        rope_scaling_factor=0,
        cache_dir=cache_dir,
        device=torch.device("cuda"),
        include_patterns=[],
        exclude_patterns=[],
        lycoris=False,
        save_merged_model=None,
    )
    assert model is not None
    assert hasattr(model, "eval")
    assert_fp8_scaled_model(model)
