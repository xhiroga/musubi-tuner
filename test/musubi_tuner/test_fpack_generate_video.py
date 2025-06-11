import pytest
import torch
import tempfile
import os
from musubi_tuner.fpack_generate_video import load_optimized_dit_model_with_lora


@pytest.mark.adhoc
def test_load_optimized_dit_model_with_lora():
    """
    load_optimized_dit_model_with_lora関数を呼び出すテスト。
    モックを使わずに実際のロジックをすべて実行する。
    """
    # 環境変数からモデルパスを取得
    models_dir = os.environ.get("MODELS", "/path/to/models")
    dit_path = os.path.join(models_dir, "diffusion_models/FramePackI2V_HY")
    
    # テスト用の一時ディレクトリを作成
    with tempfile.TemporaryDirectory() as temp_dir:
        # テスト用パラメータ
        fp8_scaled = False
        lora_weight = []  # 空のLoRAウェイト
        lora_multiplier = []
        fp8 = False
        blocks_to_swap = 0
        attn_mode = "torch"
        rope_scaling_timestep_threshold = 1000
        rope_scaling_factor = 1.0
        optimized_model_dir = temp_dir
        device = torch.device("cpu")  # CPUを使用してGPUメモリ不足を回避
        include_patterns = None
        exclude_patterns = None
        lycoris = False
        save_merged_model = None
        
        # 実際の関数を呼び出し（モックなし）
        result = load_optimized_dit_model_with_lora(
            dit_path=dit_path,
            fp8_scaled=fp8_scaled,
            lora_weight=lora_weight,
            lora_multiplier=lora_multiplier,
            fp8=fp8,
            blocks_to_swap=blocks_to_swap,
            attn_mode=attn_mode,
            rope_scaling_timestep_threshold=rope_scaling_timestep_threshold,
            rope_scaling_factor=rope_scaling_factor,
            optimized_model_dir=optimized_model_dir,
            device=device,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            lycoris=lycoris,
            save_merged_model=save_merged_model
        )
        
        # 検証
        assert result is not None
        # モデルが正常にロードされていることを確認
        assert hasattr(result, 'eval')  # モデルオブジェクトであることを確認
        

@pytest.mark.adhoc
def test_load_optimized_dit_model_with_lora_with_lora_weights():
    """
    LoRAウェイトありでload_optimized_dit_model_with_lora関数を呼び出すテスト。
    実際のLoRAファイルが存在することを前提とする。
    """
    # 環境変数からモデルパスを取得
    models_dir = os.environ.get("MODELS", "/path/to/models")
    dit_path = os.path.join(models_dir, "diffusion_models/FramePackI2V_HY")
    
    # 実際のLoRAファイルパスを設定
    lora_path = os.path.join(models_dir, "lora/test_lora.safetensors")
    
    # LoRAファイルが存在しない場合はテストをスキップ
    if not os.path.exists(lora_path):
        pytest.skip(f"LoRA file not found: {lora_path}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # テスト用パラメータ（LoRAウェイトあり）
        fp8_scaled = False
        lora_weight = [lora_path]  # 実際のLoRAウェイト
        lora_multiplier = [1.0]
        fp8 = False
        blocks_to_swap = 0
        attn_mode = "torch"
        rope_scaling_timestep_threshold = 1000
        rope_scaling_factor = 1.0
        optimized_model_dir = temp_dir
        device = torch.device("cpu")
        include_patterns = None
        exclude_patterns = None
        lycoris = False
        save_merged_model = None
        
        # 実際の関数を呼び出し
        result = load_optimized_dit_model_with_lora(
            dit_path=dit_path,
            fp8_scaled=fp8_scaled,
            lora_weight=lora_weight,
            lora_multiplier=lora_multiplier,
            fp8=fp8,
            blocks_to_swap=blocks_to_swap,
            attn_mode=attn_mode,
            rope_scaling_timestep_threshold=rope_scaling_timestep_threshold,
            rope_scaling_factor=rope_scaling_factor,
            optimized_model_dir=optimized_model_dir,
            device=device,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            lycoris=lycoris,
            save_merged_model=save_merged_model
        )
        
        # 検証
        assert result is not None
        assert hasattr(result, 'eval')
