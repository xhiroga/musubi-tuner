include .env
export

.PHONY: fpack_generate_video_on_modal fpack_generate_video modal_setup modal_upload_models modal_upload_models_volume modal_test modal_list_models modal_download_hf test

# Modal関連のタスク
modal_setup: modal_test modal_list_models
	@echo "🎉 Modal setup completed!"

modal_upload_models_volume:
	@echo "📦 Uploading models to Modal Volume using modal volume put..."
	modal volume put fpack-generate-video-on-modal-models $(MODELS)/diffusion_models/FramePackI2V_HY /models/diffusion_models/FramePackI2V_HY
	modal volume put fpack-generate-video-on-modal-models $(MODELS)/vae /models/vae
	modal volume put fpack-generate-video-on-modal-models $(MODELS)/text_encoder /models/text_encoder
	modal volume put fpack-generate-video-on-modal-models $(MODELS)/text_encoder_2 /models/text_encoder_2
	modal volume put fpack-generate-video-on-modal-models $(MODELS)/image_encoder /models/image_encoder
	modal volume put fpack-generate-video-on-modal-models $(MODELS)/diffusion_models/optimized /models/diffusion_models/optimized
	@echo "✅ All models uploaded successfully!"

modal_upload_models:
	@echo "📦 Uploading models to Modal Volume..."
	modal run src/musubi_tuner/fpack_generate_video_on_modal.py::upload_models_from_local \
		--local-model-dir $(MODELS)

modal_download_hf:
	@echo "📦 Downloading models from Hugging Face Hub..."
	modal run src/musubi_tuner/fpack_generate_video_on_modal.py::download_models_from_hf

modal_test:
	@echo "🧪 Testing Modal environment..."
	modal run src/musubi_tuner/fpack_generate_video_on_modal.py::test_modal_setup

modal_list_models:
	@echo "📋 Listing models in Modal Volume..."
	modal run src/musubi_tuner/fpack_generate_video_on_modal.py::list_models

fpack_generate_video_on_modal:
	@echo "🚀 Generating video on Modal..."
	modal run src/musubi_tuner/fpack_generate_video_on_modal.py \
		--image-path $(IMAGE_PATH) \
		--prompt "rotating 360 degrees" \
		--video-width 960 \
		--video-height 544 \
		--fps 30 \
		--infer-steps 10 \
		--video-sections 1 \
		--latent-window-size 5 \
		--seed 1234 \
		--save-path $(SAVE_PATH)

# フルワークフロー（アップロード→テスト→生成）
modal_full_workflow: modal_upload_models_volume modal_test fpack_generate_video_on_modal
	@echo "✅ Full Modal workflow completed!"

fpack_generate_video:
	uv run --extra cu128 src/musubi_tuner/fpack_generate_video.py \
		--dit $(DIT_PATH) \
		--vae $(VAE_PATH) \
		--text_encoder1 $(TEXT_ENCODER1_PATH) \
		--text_encoder2 $(TEXT_ENCODER2_PATH) \
		--image_encoder $(IMAGE_ENCODER_PATH) \
		--image_path $(IMAGE_PATH) \
		--prompt $(PROMPT) \
		--video_size 960 544 --fps 30 --infer_steps 10 --video_sections 1 --latent_window_size 5 \
		--device cuda --attn_mode sdpa --fp8_scaled \
		--vae_chunk_size 32 --vae_spatial_tile_sample_min_size 128 \
		--save_path $(SAVE_PATH) --seed 1234  \
		--lora_weight $(LORA_PATH) --lora_multiplier $(LORA_MULTIPLIER) \
		--profile --cache_dir $(CACHE_DIR) --optimized_model_dir $(OPTIMIZED_MODEL_DIR) --log_level DEBUG

# FramePack I2V生成 (プロファイリング版)
fpack_generate_video_on_modal_profile:
	@echo "🔍 Generating video on Modal with profiling..."
	modal run src/musubi_tuner/fpack_generate_video_on_modal.py \
		--image-path $(IMAGE_PATH) \
		--prompt "$(PROMPT)" \
		--video-width $(VIDEO_WIDTH) \
		--video-height $(VIDEO_HEIGHT) \
		--fps $(FPS) \
		--infer-steps $(INFER_STEPS) \
		--video-sections $(VIDEO_SECTIONS) \
		--latent-window-size $(LATENT_WINDOW_SIZE) \
		--seed $(SEED) \
		--save-path $(SAVE_PATH) \
		--enable-profiling \
		--profile-steps 3 \
		--record-shapes \
		--profile-memory \
		--with-stack \
		--download-profiles \
		--local-profile-dir $(SAVE_PATH)/profiling_results

# プロファイリング結果のダウンロード
modal_download_profiles:
	@echo "📥 Downloading profiling results from Modal..."
	modal run src/musubi_tuner/fpack_generate_video_on_modal.py::download_profiling_results \
		--local-output-dir $(SAVE_PATH)/profiling_results

# プロファイリングセッション一覧
modal_list_profiling_sessions:
	@echo "📊 Listing profiling sessions..."
	modal run src/musubi_tuner/fpack_generate_video_on_modal.py::list_profiling_sessions

# プロファイリングデータのクリア（要確認）
modal_clear_profiling_data:
	@echo "🗑️  Clearing profiling data (requires confirmation)..."
	modal run src/musubi_tuner/fpack_generate_video_on_modal.py::clear_profiling_data --confirm

test:
	uv run pytest test/musubi_tuner/test_fpack_generate_video.py -m adhoc -sv
