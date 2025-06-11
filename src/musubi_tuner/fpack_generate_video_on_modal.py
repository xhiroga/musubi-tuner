"""
FramePack I2V Video Generation on Modal with Advanced Profiling

This script provides Modal Functions for running FramePack I2V video generation in the cloud
with comprehensive profiling capabilities based on Modal's torch_profiling example.

Profiling Features:
- PyTorch Profiler with CPU and CUDA activity tracking
- Memory profiling and shape recording
- Stack trace recording for debugging
- TensorBoard trace generation
- Timing logs and performance metrics
- Local and remote profiling result management

Usage:
1. Setup models:
   modal run src/musubi_tuner/fpack_generate_video_on_modal.py::upload_models_from_local --local-model-dir /path/to/models
   
2. Test setup:
   modal run src/musubi_tuner/fpack_generate_video_on_modal.py::test_modal_setup
   
3. Generate video with profiling:
   modal run src/musubi_tuner/fpack_generate_video_on_modal.py::profile_video_generation \
       --image-path /path/to/image.jpg --prompt "rotating 360 degrees" \
       --profile-steps 3 --record-shapes --profile-memory
   
   Or use the Makefile:
   make fpack_generate_video_on_modal IMAGE_PATH=/path/to/image.jpg SAVE_PATH=/path/to/output

4. Download profiling results:
   modal run src/musubi_tuner/fpack_generate_video_on_modal.py::download_profiling_results \
       --local-output-dir ./profiling_results

5. List models:
   modal run src/musubi_tuner/fpack_generate_video_on_modal.py::list_models

6. Serve TensorBoard (experimental):
   modal deploy src/musubi_tuner/fpack_generate_video_on_modal.py

Requirements:
- Modal account and CLI setup
- Model files (either uploaded or downloaded from HF Hub)
- Input image for video generation

Environment Variables:
- HF_TOKEN: Hugging Face token (if downloading models from private repos)
"""

import modal
import os
from pathlib import Path
from typing import Optional, Dict, Any
from uuid import uuid4

# CUDA環境の設定
cuda_version = "12.8.0"
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# Modal用のイメージを構築
ml_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "libsm6", "libxrender1", "libxext6")
    .pip_install("uv")
    .run_commands([
        # fpack_generate_video.pyの全依存関係をuvで高速インストール
        """uv pip install --system --compile-bytecode \
           'torch>=2.7.1' \
           'torchvision>=0.22.1' \
           torchaudio \
           'transformers>=4.46.3' \
           'diffusers>=0.32.1' \
           'safetensors>=0.4.5' \
           'pillow>=10.2.0' \
           'numpy<2' \
           'tqdm>=4.66.5' \
           'accelerate>=1.6.0' \
           'diskcache>=5.6.3' \
           'huggingface_hub[hf_transfer]>=0.30.0' \
           'opencv-python>=4.10.0.84' \
           psutil \
           'einops>=0.7.0' \
           packaging \
           requests \
           'av==14.0.1' \
           'bitsandbytes>=0.45.0' \
           'sageattention>=1.0.6' \
           'toml>=0.10.2' \
           'voluptuous>=0.15.2' \
           'modal>=1.0.3' \
           'ftfy==6.3.1' \
           'easydict==1.13' \
           argparse \
           logging \
           datetime"""
    ])
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "CUDA_VISIBLE_DEVICES": "0"
    })
    .add_local_python_source("musubi_tuner")  # ローカルのmusubi_tunerモジュールを追加
)

app = modal.App("fpack-generate-video-on-modal")
model_volume = modal.Volume.from_name("fpack-generate-video-on-modal-models", create_if_missing=True)
# プロファイリング結果用のVolume
profiling_volume = modal.Volume.from_name("fpack-generate-video-profiling", create_if_missing=True)

PROFILING_DIR = Path("/profiling")

@app.function(
    image=ml_image,
    gpu="H100",  # 高性能GPU
    volumes={
        "/models": model_volume,
        PROFILING_DIR: profiling_volume
    },
    timeout=3600,  # 1時間のタイムアウト
    memory=32768,  # 32GB RAM
)
def profile_video_generation(
    image_bytes: bytes,
    image_filename: str,
    prompt: str = "rotating 360 degrees",
    video_size: tuple = (960, 544),
    fps: int = 30,
    infer_steps: int = 10,
    video_sections: int = 1,
    latent_window_size: int = 5,
    seed: int = 1234,
    save_path: str = "/tmp/output",
    # プロファイリング設定
    profile_steps: int = 3,
    profile_label: Optional[str] = None,
    record_shapes: bool = False,
    profile_memory: bool = True,
    with_stack: bool = True,
    print_profile_summary: int = 10
):
    """
    Modal上でFramePackによる動画生成を実行し、詳細なプロファイリングを行う関数
    Modalのtorch_profiling例に基づいて実装
    """
    import sys
    import argparse
    import torch
    import json
    import time
    from datetime import datetime
    from pathlib import Path
    
    # musubi_tunerモジュールからfpack_generate_videoをインポート
    from musubi_tuner.fpack_generate_video import get_generation_settings, generate, save_output
    
    # 一意のプロファイリングセッションIDを生成
    session_id = str(uuid4())
    timestamp = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")
    
    # プロファイリング用のディレクトリを作成
    function_name = "fpack_video_generation"
    profile_label_suffix = f"_{profile_label}" if profile_label else ""
    profile_session_dir = PROFILING_DIR / f"{function_name}{profile_label_suffix}" / f"{timestamp}_{session_id}"
    profile_session_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🔍 Starting profiled video generation on Modal...")
    print(f"📊 Profile session: {session_id}")
    print(f"📁 Profile directory: {profile_session_dir}")
    
    # 画像バイナリデータを一時ファイルに保存
    temp_dir = "/tmp/modal_input"
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_image_path = os.path.join(temp_dir, image_filename)
    with open(temp_image_path, 'wb') as f:
        f.write(image_bytes)
    
    print(f"🖼️  Saved input image to: {temp_image_path}")
    print(f"📏 Image size: {len(image_bytes)} bytes")
    
    # Makefileの引数を再現してargparse.Namespaceを構築
    args = argparse.Namespace()
    
    # モデルパス（Volume内の正しい構造に基づく）
    args.dit = "/models/models/diffusion_models/FramePackI2V_HY"
    args.vae = "/models/models/vae/diffusion_pytorch_model.safetensors"
    args.text_encoder1 = "/models/models/text_encoder/model-00001-of-00004.safetensors"
    args.text_encoder2 = "/models/models/text_encoder_2/model.safetensors"
    args.image_encoder = "/models/models/image_encoder/model.safetensors"
    
    # 入力パラメータ
    args.image_path = temp_image_path
    args.prompt = prompt
    args.video_size = list(video_size)
    args.fps = fps
    args.infer_steps = infer_steps
    args.video_sections = video_sections
    args.latent_window_size = latent_window_size
    args.seed = seed
    args.save_path = save_path
    
    # デバイス・最適化設定
    args.device = "cuda"
    args.attn_mode = "sdpa"
    args.fp8 = False
    args.fp8_llm = False
    args.fp8_scaled = True
    
    # VAE設定
    args.vae_chunk_size = 32
    args.vae_spatial_tile_sample_min_size = 128
    
    # その他の設定
    args.cache_dir = "/tmp/.cache/musubi_tuner"
    args.optimized_model_dir = "/models/models/diffusion_models/optimized"
    args.log_level = "DEBUG"
    
    # プロファイリング設定
    args.profile = True
    args.profile_shapes = record_shapes
    args.profile_memory = profile_memory
    args.profile_stack = with_stack
    
    # デフォルト値
    args.negative_prompt = None
    args.custom_system_prompt = None
    args.video_seconds = 5.0
    args.one_frame_inference = None
    args.control_image_path = None
    args.control_image_mask_path = None
    args.end_image_path = None
    args.latent_paddings = None
    args.f1 = False
    args.rope_scaling_factor = 0.5
    args.rope_scaling_timestep_threshold = None
    args.lora_weight = None
    args.lora_multiplier = 1.0
    args.include_patterns = None
    args.exclude_patterns = None
    args.save_merged_model = None
    args.lycoris = False
    args.blocks_to_swap = 0
    args.output_type = "video"
    args.no_metadata = False
    args.latent_path = None
    args.bulk_decode = False
    args.sample_solver = "unipc"
    args.embedded_cfg_scale = 10.0
    args.guidance_scale = 1.0
    args.guidance_rescale = 0.0
    
    # 出力ディレクトリを作成
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    try:
        # プロファイリングスケジュールを設定
        if profile_steps < 3:
            raise ValueError("Profile steps must be at least 3 when using default schedule")
        
        schedule = torch.profiler.schedule(
            wait=1,
            warmup=1, 
            active=profile_steps - 2,
            repeat=0
        )
        
        print(f"🎬 Starting profiled video generation...")
        print(f"📊 Profile config: steps={profile_steps}, shapes={record_shapes}, memory={profile_memory}, stack={with_stack}")
        print(f"Input image: {temp_image_path}")
        print(f"Prompt: {prompt}")
        print(f"Video size: {video_size}")
        print(f"Output path: {save_path}")
        
        # タイミングログファイルを作成
        timing_log_path = profile_session_dir / "timing.log"
        start_time = time.time()
        
        def log_timing(message):
            elapsed = time.time() - start_time
            with open(timing_log_path, 'a') as f:
                f.write(f"[T+{elapsed:.2f}s] {message}\n")
            print(f"⏱️  [T+{elapsed:.2f}s] {message}")
        
        log_timing("Profile session started")
        
        # 詳細プロファイリングの実行
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        
        with torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profile_session_dir)),
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack
        ) as prof:
            # メタデータを追加
            prof.add_metadata_json("args", json.dumps(vars(args), default=str))
            prof.add_metadata_json("session_info", json.dumps({
                "session_id": session_id,
                "timestamp": timestamp,
                "image_filename": image_filename,
                "prompt": prompt,
                "video_size": video_size,
                "profile_steps": profile_steps
            }))
            
            log_timing("Profiler initialized")
            
            for step in range(profile_steps):
                log_timing(f"Starting profile step {step + 1}/{profile_steps}")
                
                # 生成設定を取得
                gen_settings = get_generation_settings(args)
                log_timing(f"Generation settings configured")
                
                # 動画生成を実行
                result = generate(args, gen_settings, prof=prof, log_timing=log_timing)
                
                if result is None:
                    # Model was saved, no further processing needed
                    log_timing("Model save completed, skipping video generation")
                    prof.step()
                    continue
                
                vae, latent = result
                
                # 結果を保存
                save_output(args, vae, latent[0], gen_settings.device, None, log_timing=log_timing)
                
                log_timing(f"Completed profile step {step + 1}/{profile_steps}")
                prof.step()
        
        # プロファイリング結果のサマリーを生成
        if print_profile_summary > 0:
            print(f"\n📊 Profile Summary (Top {print_profile_summary} operations by CUDA time):")
            summary_table = prof.key_averages().table(
                sort_by="cuda_time_total", 
                row_limit=print_profile_summary
            )
            print(summary_table)
            
            # サマリーをファイルに保存
            summary_path = profile_session_dir / "profile_summary.txt"
            with open(summary_path, 'w') as f:
                f.write(f"Profile Summary - {timestamp}\n")
                f.write(f"Session ID: {session_id}\n")
                f.write(f"Prompt: {prompt}\n")
                f.write(f"Video Size: {video_size}\n")
                f.write(f"Profile Steps: {profile_steps}\n")
                f.write("=" * 80 + "\n")
                f.write(summary_table)
        
        log_timing("Profile analysis completed")
        
        # プロファイリング結果のファイル一覧を取得
        profile_files = []
        for file_path in profile_session_dir.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(PROFILING_DIR)
                profile_files.append(str(relative_path))
        
        # Volume にコミット
        profiling_volume.commit()
        
        print(f"✅ Profiled video generation completed successfully!")
        print(f"📊 Profile results saved to Volume: {profile_session_dir}")
        print(f"📁 Profile files: {len(profile_files)} files")
        
        # 生成されたファイルを確認してバイナリデータとして返す
        output_files = list(Path(save_path).glob("*.mp4"))
        result_files = {}
        
        for file_path in output_files:
            if file_path.is_file():
                with open(file_path, 'rb') as f:
                    file_data = f.read()
                result_files[file_path.name] = file_data
                print(f"📄 Captured output file: {file_path.name} ({len(file_data)} bytes)")
        
        log_timing("Process completed")
        
        return {
            "status": "success",
            "session_id": session_id,
            "output_path": save_path,
            "files": [str(f) for f in output_files],
            "result_files": result_files,  # バイナリデータを含む
            "profile_session_dir": str(profile_session_dir.relative_to(PROFILING_DIR)),
            "profile_files": profile_files,
            "timing_log": timing_log_path.read_text() if timing_log_path.exists() else None
        }
        
    except Exception as e:
        print(f"❌ Error during profiled video generation: {str(e)}")
        import traceback
        error_trace = traceback.format_exc()
        traceback.print_exc()
        
        # エラー情報をプロファイリングディレクトリに保存
        error_log_path = profile_session_dir / "error.log"
        with open(error_log_path, 'w') as f:
            f.write(f"Error occurred during profiled video generation\n")
            f.write(f"Session ID: {session_id}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Error: {str(e)}\n")
            f.write("=" * 80 + "\n")
            f.write(error_trace)
        
        profiling_volume.commit()
        
        return {
            "status": "error",
            "session_id": session_id,
            "error": str(e),
            "traceback": error_trace,
            "profile_session_dir": str(profile_session_dir.relative_to(PROFILING_DIR))
        }

@app.function(
    image=ml_image,
    volumes={PROFILING_DIR: profiling_volume},
    timeout=600
)
def download_profiling_results(local_output_dir: str, session_id: Optional[str] = None):
    """
    Modal Volume内のプロファイリング結果をローカルにダウンロードする関数
    """
    import shutil
    import tarfile
    from pathlib import Path
    
    print(f"📥 Downloading profiling results to: {local_output_dir}")
    
    if not PROFILING_DIR.exists():
        print("No profiling data found in Volume")
        return {"status": "no_data", "message": "No profiling data found"}
    
    # ローカル出力ディレクトリを作成
    local_path = Path(local_output_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    
    downloaded_sessions = []
    
    if session_id:
        # 特定のセッションをダウンロード
        session_dirs = list(PROFILING_DIR.rglob(f"*{session_id}*"))
        if not session_dirs:
            return {"status": "not_found", "message": f"Session {session_id} not found"}
    else:
        # 全てのプロファイリング結果をダウンロード
        session_dirs = [d for d in PROFILING_DIR.rglob("*") if d.is_dir() and len(d.parts) == 3]  # function/session_dir level
    
    for session_dir in session_dirs:
        if session_dir.is_dir():
            session_name = session_dir.name
            local_session_dir = local_path / session_name
            
            print(f"📁 Downloading session: {session_name}")
            
            # セッションディレクトリをローカルにコピー
            shutil.copytree(session_dir, local_session_dir, dirs_exist_ok=True)
            
            # セッション情報を収集
            session_info = {
                "session_name": session_name,
                "local_path": str(local_session_dir),
                "file_count": len(list(local_session_dir.rglob("*"))),
                "size_mb": sum(f.stat().st_size for f in local_session_dir.rglob("*") if f.is_file()) / (1024 * 1024)
            }
            downloaded_sessions.append(session_info)
            
            print(f"✅ Downloaded: {session_name} ({session_info['file_count']} files, {session_info['size_mb']:.1f} MB)")
    
    print(f"✅ Download completed: {len(downloaded_sessions)} sessions")
    
    return {
        "status": "success",
        "local_output_dir": local_output_dir,
        "downloaded_sessions": downloaded_sessions,
        "total_sessions": len(downloaded_sessions)
    }

@app.function(
    image=ml_image,
    volumes={PROFILING_DIR: profiling_volume}
)
def list_profiling_sessions():
    """
    Modal Volume内のプロファイリングセッション一覧を表示する関数
    """
    import os
    from pathlib import Path
    from datetime import datetime
    
    print("📊 Profiling Sessions in Modal Volume:")
    
    if not PROFILING_DIR.exists():
        print("No profiling data found")
        return {"status": "no_data", "sessions": []}
    
    sessions = []
    
    for root, dirs, files in os.walk(PROFILING_DIR):
        level = root.replace(str(PROFILING_DIR), '').count(os.sep)
        
        if level == 2:  # session level (function/label/session)
            session_path = Path(root)
            session_name = session_path.name
            function_name = session_path.parent.name
            
            # セッション情報を収集
            file_count = len([f for f in session_path.rglob("*") if f.is_file()])
            total_size = sum(f.stat().st_size for f in session_path.rglob("*") if f.is_file())
            size_mb = total_size / (1024 * 1024)
            
            # タイムスタンプを抽出
            timestamp_str = session_name.split('_')[0] if '_' in session_name else "unknown"
            
            session_info = {
                "session_name": session_name,
                "function_name": function_name,
                "timestamp": timestamp_str,
                "file_count": file_count,
                "size_mb": round(size_mb, 1),
                "path": str(session_path.relative_to(PROFILING_DIR))
            }
            sessions.append(session_info)
            
            print(f"  📁 {function_name}/{session_name}")
            print(f"     ⏰ {timestamp_str}")
            print(f"     📄 {file_count} files ({size_mb:.1f} MB)")
            print()
    
    # セッションをタイムスタンプでソート
    sessions.sort(key=lambda x: x["timestamp"], reverse=True)
    
    print(f"Total sessions: {len(sessions)}")
    
    return {
        "status": "success",
        "sessions": sessions,
        "total_sessions": len(sessions)
    }

@app.function(
    image=ml_image,
    volumes={PROFILING_DIR: profiling_volume}
)
def clear_profiling_data(confirm: bool = False):
    """
    Modal Volume内のプロファイリングデータをクリアする関数
    """
    import shutil
    
    if not confirm:
        return {
            "status": "confirmation_required",
            "message": "Set confirm=True to actually clear profiling data"
        }
    
    if PROFILING_DIR.exists():
        shutil.rmtree(PROFILING_DIR)
        PROFILING_DIR.mkdir(parents=True, exist_ok=True)
        profiling_volume.commit()
        print("🗑️  Profiling data cleared")
        return {"status": "cleared"}
    else:
        print("No profiling data to clear")
        return {"status": "no_data"}

@app.function(
    image=ml_image,
    gpu="H100",  # 高性能GPU
    volumes={"/models": model_volume},
    timeout=3600,  # 1時間のタイムアウト
    memory=32768,  # 32GB RAM
)
def generate_video_modal(
    image_bytes: bytes,
    image_filename: str,
    prompt: str = "rotating 360 degrees",
    video_size: tuple = (960, 544),
    fps: int = 30,
    infer_steps: int = 10,
    video_sections: int = 1,
    latent_window_size: int = 5,
    seed: int = 1234,
    save_path: str = "/tmp/output"
):
    """
    Modal上でFramePackによる動画生成を実行する関数
    """
    import sys
    import argparse
    from pathlib import Path
    
    # musubi_tunerモジュールからfpack_generate_videoをインポート
    from musubi_tuner.fpack_generate_video import main, parse_args
    
    # 画像バイナリデータを一時ファイルに保存
    import os
    import tempfile
    
    # 一時ディレクトリに画像を保存
    temp_dir = "/tmp/modal_input"
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_image_path = os.path.join(temp_dir, image_filename)
    with open(temp_image_path, 'wb') as f:
        f.write(image_bytes)
    
    print(f"🖼️  Saved input image to: {temp_image_path}")
    print(f"📏 Image size: {len(image_bytes)} bytes")
    
    # Makefileの引数を再現してargparse.Namespaceを構築
    args = argparse.Namespace()
    
    # モデルパス（Volume内の正しい構造に基づく）
    args.dit = "/models/models/diffusion_models/FramePackI2V_HY"
    args.vae = "/models/models/vae/diffusion_pytorch_model.safetensors"
    args.text_encoder1 = "/models/models/text_encoder/model-00001-of-00004.safetensors"
    args.text_encoder2 = "/models/models/text_encoder_2/model.safetensors"
    args.image_encoder = "/models/models/image_encoder/model.safetensors"
    
    # 入力パラメータ
    args.image_path = temp_image_path
    args.prompt = prompt
    args.video_size = list(video_size)
    args.fps = fps
    args.infer_steps = infer_steps
    args.video_sections = video_sections
    args.latent_window_size = latent_window_size
    args.seed = seed
    args.save_path = save_path
    
    # デバイス・最適化設定
    args.device = "cuda"
    args.attn_mode = "sdpa"
    args.fp8 = False
    args.fp8_llm = False
    args.fp8_scaled = True
    
    # VAE設定
    args.vae_chunk_size = 32
    args.vae_spatial_tile_sample_min_size = 128
    
    # その他の設定
    args.cache_dir = "/tmp/.cache/musubi_tuner"
    args.optimized_model_dir = "/models/models/diffusion_models/optimized"
    args.log_level = "DEBUG"
    args.profile = True
    
    # デフォルト値
    args.negative_prompt = None
    args.custom_system_prompt = None
    args.video_seconds = 5.0
    args.one_frame_inference = None
    args.control_image_path = None
    args.control_image_mask_path = None
    args.end_image_path = None
    args.latent_paddings = None
    args.f1 = False
    args.rope_scaling_factor = 0.5
    args.rope_scaling_timestep_threshold = None
    args.lora_weight = None
    args.lora_multiplier = 1.0
    args.include_patterns = None
    args.exclude_patterns = None
    args.save_merged_model = None
    args.lycoris = False
    args.blocks_to_swap = 0
    args.output_type = "video"
    args.no_metadata = False
    args.latent_path = None
    args.bulk_decode = False
    args.sample_solver = "unipc"
    args.embedded_cfg_scale = 10.0
    args.guidance_scale = 1.0
    args.guidance_rescale = 0.0
    
    # 出力ディレクトリを作成
    os.makedirs(save_path, exist_ok=True)
    
    # キャッシュディレクトリを作成
    os.makedirs(args.cache_dir, exist_ok=True)
    
    print(f"🎬 Starting video generation on Modal...")
    print(f"Input image: {temp_image_path}")
    print(f"Prompt: {prompt}")
    print(f"Video size: {video_size}")
    print(f"Output path: {save_path}")
    
    try:
        # プロファイラー用の設定
        import torch
        import json
        from datetime import datetime
        import time
        
        timestamp = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")
        log_dir = f"{save_path}/profiles/{timestamp}"
        os.makedirs(log_dir, exist_ok=True)
        
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        
        with torch.profiler.profile(
            activities=activities,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
            record_shapes=args.profile_shapes,
            profile_memory=args.profile_memory,
            with_stack=args.profile_stack
        ) as prof:
            prof.add_metadata_json("args", json.dumps(vars(args), default=str))
            
            # fpack_generate_video.pyから必要な関数をインポート
            from musubi_tuner.fpack_generate_video import get_generation_settings, generate, save_output
            
            # 生成設定を取得
            gen_settings = get_generation_settings(args)
            
            # 動画生成を実行
            result = generate(args, gen_settings, prof=prof)
            
            if result is None:
                # Model was saved, no further processing needed
                return {
                    "status": "success", 
                    "message": "Model saved successfully, no video generation performed",
                    "output_path": save_path,
                    "files": [],
                    "result_files": {},
                    "log_dir": log_dir
                }
            
            vae, latent = result
            
            # 結果を保存
            save_output(args, vae, latent[0], gen_settings.device, None)
        
        print(f"✅ Video generation completed successfully!")
        print(f"Output saved to: {save_path}")
        
        # 生成されたファイルを確認してバイナリデータとして返す
        output_files = list(Path(save_path).glob("*.mp4"))
        result_files = {}
        
        for file_path in output_files:
            if file_path.is_file():
                with open(file_path, 'rb') as f:
                    file_data = f.read()
                result_files[file_path.name] = file_data
                print(f"📄 Captured output file: {file_path.name} ({len(file_data)} bytes)")
        
        # log_dirは既に上で定義済み
        
        return {
            "status": "success",
            "output_path": save_path,
            "files": [str(f) for f in output_files],
            "result_files": result_files,  # バイナリデータを含む
            "log_dir": log_dir
        }
        
    except Exception as e:
        print(f"❌ Error during video generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@app.function(
    image=ml_image,
    volumes={"/models": model_volume},
    timeout=3600
)
def upload_models_from_local(local_model_dir: str):
    """
    ローカルのモデルファイルをModal Volumeにアップロードする関数
    """
    import shutil
    from pathlib import Path
    
    local_path = Path(local_model_dir)
    if not local_path.exists():
        raise ValueError(f"Local model directory does not exist: {local_model_dir}")
    
    print(f"📦 Uploading models from {local_model_dir} to Modal Volume...")
    
    # モデルディレクトリの構造を再現
    model_dirs = [
        "diffusion_models/FramePackI2V_HY",
        "vae",
        "text_encoder", 
        "text_encoder_2",
        "image_encoder",
        "diffusion_models/optimized"
    ]
    
    for model_dir in model_dirs:
        local_model_path = local_path / model_dir
        remote_model_path = Path("/models") / model_dir
        
        if local_model_path.exists():
            print(f"Copying {model_dir}...")
            remote_model_path.parent.mkdir(parents=True, exist_ok=True)
            if local_model_path.is_dir():
                shutil.copytree(local_model_path, remote_model_path, dirs_exist_ok=True)
            else:
                shutil.copy2(local_model_path, remote_model_path)
        else:
            print(f"⚠️  Warning: {model_dir} not found in local directory")
    
    model_volume.commit()
    print("✅ Model upload completed!")

@app.function(
    image=ml_image,
    volumes={"/models": model_volume},
    timeout=1800
)  
def download_models_from_hf():
    """
    Hugging Face Hubからモデルをダウンロードする関数
    """
    from huggingface_hub import snapshot_download
    import os
    
    print("📦 Downloading models from Hugging Face Hub...")
    
    # 環境変数からHFトークンを取得（必要に応じて）
    hf_token = os.getenv("HF_TOKEN")
    
    try:
        # FramePack I2V モデルのダウンロード例
        # 実際のリポジトリ名は適切なものに変更してください
        model_repo = "your-org/framepack-i2v-model"  # 実際のリポジトリ名に変更
        
        snapshot_download(
            repo_id=model_repo,
            local_dir="/models/diffusion_models/FramePackI2V_HY",
            token=hf_token,
            ignore_patterns=["*.git*", "README.md", "*.md"]
        )
        
        model_volume.commit()
        print("✅ Models downloaded successfully!")
        
    except Exception as e:
        print(f"❌ Error downloading models: {str(e)}")
        print("Please ensure you have the correct repository access and HF_TOKEN is set")

@app.function(
    image=ml_image,
    volumes={"/models": model_volume}
)
def list_models():
    """
    Modal Volume内のモデルファイルをリストアップする関数
    """
    import os
    from pathlib import Path
    
    print("📋 Models in Modal Volume:")
    model_root = Path("/models")
    
    if not model_root.exists():
        print("No models directory found")
        return
    
    for root, dirs, files in os.walk(model_root):
        level = root.replace(str(model_root), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            file_path = Path(root) / file
            file_size = file_path.stat().st_size if file_path.exists() else 0
            size_mb = file_size / (1024 * 1024)
            print(f"{subindent}{file} ({size_mb:.1f} MB)")

@app.function(image=ml_image)
def test_modal_setup():
    """
    Modal環境のセットアップをテストする関数
    """
    print("🧪 Testing Modal setup...")
    
    # CUDA availability check
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA device count: {torch.cuda.device_count()}")
            print(f"✅ CUDA device name: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"❌ PyTorch/CUDA test failed: {e}")
    
    # Other dependencies check
    dependencies = ["transformers", "diffusers", "safetensors", "accelerate", "diskcache"]
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep} imported successfully")
        except ImportError:
            print(f"❌ {dep} import failed")
    
    print("🎉 Modal setup test completed!")

@app.local_entrypoint()
def main(
    image_path: str = "path/to/your/image.jpg",
    prompt: str = "rotating 360 degrees",
    video_width: int = 960,
    video_height: int = 544,
    fps: int = 30,
    infer_steps: int = 10,
    video_sections: int = 1,
    latent_window_size: int = 5,
    seed: int = 1234,
    save_path: str = "/tmp/modal_output",
    # プロファイリング設定
    enable_profiling: bool = True,
    profile_steps: int = 3,
    profile_label: Optional[str] = None,
    record_shapes: bool = False,
    profile_memory: bool = True,
    with_stack: bool = True,
    print_profile_summary: int = 10,
    download_profiles: bool = True,
    local_profile_dir: str = "./profiling_results"
):
    """
    Modal上での動画生成を実行するエントリーポイント（プロファイリング対応）
    
    使用例:
    modal run src/musubi_tuner/fpack_generate_video_on_modal.py \
        --image-path "path/to/image.jpg" \
        --prompt "rotating 360 degrees" \
        --enable-profiling \
        --profile-steps 3 \
        --record-shapes \
        --download-profiles
    """
    import os
    from pathlib import Path
    
    print(f"🚀 Starting Modal video generation with profiling...")
    print(f"📸 Loading image from: {image_path}")
    
    # 入力画像をバイナリとして読み込み
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")
    
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    image_filename = Path(image_path).name
    print(f"📏 Image loaded: {image_filename} ({len(image_bytes)} bytes)")
    
    # プロファイリングが有効な場合は profile_video_generation を使用
    if enable_profiling:
        print(f"🔍 Running with profiling enabled...")
        result = profile_video_generation.remote(
            image_bytes=image_bytes,
            image_filename=image_filename,
            prompt=prompt,
            video_size=(video_width, video_height),
            fps=fps,
            infer_steps=infer_steps,
            video_sections=video_sections,
            latent_window_size=latent_window_size,
            seed=seed,
            save_path="/tmp/output",  # Modal内の一時パス
            profile_steps=profile_steps,
            profile_label=profile_label,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
            print_profile_summary=print_profile_summary
        )
    else:
        print(f"🎬 Running without profiling...")
        result = generate_video_modal.remote(
            image_bytes=image_bytes,
            image_filename=image_filename,
            prompt=prompt,
            video_size=(video_width, video_height),
            fps=fps,
            infer_steps=infer_steps,
            video_sections=video_sections,
            latent_window_size=latent_window_size,
            seed=seed,
            save_path="/tmp/output"  # Modal内の一時パス
        )
    
    print(f"📊 Generation result status: {result.get('status', 'unknown')}")
    
    if result.get("status") == "success":
        # ローカル保存ディレクトリを作成
        os.makedirs(save_path, exist_ok=True)
        
        # 生成されたファイルをローカルに保存
        if result.get("result_files"):
            for filename, file_data in result["result_files"].items():
                local_file_path = os.path.join(save_path, filename)
                with open(local_file_path, 'wb') as f:
                    f.write(file_data)
                
                print(f"💾 Saved to local: {local_file_path} ({len(file_data)} bytes)")
        
        # プロファイリング結果をダウンロード
        if enable_profiling and download_profiles and result.get("session_id"):
            print(f"📥 Downloading profiling results...")
            profile_result = download_profiling_results.remote(
                local_output_dir=local_profile_dir,
                session_id=result["session_id"]
            )
            
            if profile_result.get("status") == "success":
                print(f"✅ Profiling results downloaded to: {local_profile_dir}")
                for session in profile_result["downloaded_sessions"]:
                    print(f"   📁 {session['session_name']}: {session['file_count']} files ({session['size_mb']:.1f} MB)")
            else:
                print(f"⚠️  Failed to download profiling results: {profile_result}")
        
        print(f"✅ Video generation completed! Check output in: {save_path}")
        
        # プロファイリング情報の表示
        if enable_profiling and result.get("timing_log"):
            print(f"\n⏱️  Timing Log:")
            print(result["timing_log"])
            
    elif result.get("status") == "error":
        print(f"❌ Error occurred: {result.get('error', 'Unknown error')}")
        if result.get("traceback"):
            print(f"Traceback:\n{result['traceback']}")
    else:
        print(f"⚠️  Unexpected result: {result}")
    
    return result

if __name__ == "__main__":
    # CLIでの実行をサポート
    import sys
    if len(sys.argv) > 1:
        # コマンドライン引数がある場合は、main関数を直接呼び出し
        main()
    else:
        print("Usage: modal run src/musubi_tuner/fpack_generate_video_on_modal.py --image-path <path>")




