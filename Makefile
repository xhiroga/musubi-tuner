include .env
export

.PHONY: fpack_generate_video

fpack_generate_video:
	uv run --extra cu128 src/musubi_tuner/fpack_generate_video.py \
		--dit $(MODELS)/diffusion_models/FramePackI2V_HY \
		--vae $(MODELS)/vae/diffusion_pytorch_model.safetensors \
		--text_encoder1 $(MODELS)/text_encoder/model-00001-of-00004.safetensors \
		--text_encoder2 $(MODELS)/text_encoder_2/model.safetensors \
		--image_encoder $(MODELS)/image_encoder/model.safetensors \
		--image_path $(IMAGE) \
		--prompt "rotating 360 degrees" \
		--video_size 960 544 --fps 30 --infer_steps 10 --video_sections 1 --latent_window_size 5 \
		--device cuda --attn_mode sdpa --fp8 --fp8_llm \
		--vae_chunk_size 32 --vae_spatial_tile_sample_min_size 128 \
		--save_path $(SAVE_PATH) --seed 1234  \
		--profile --cache_dir ~/.cache/musubi_tuner --optimized_model_dir $(MODELS)/diffusion_models/optimized
