/content/numeri-vjepa-experiment
🧠 BAGIAN 1: Menjalankan Recursive Intent Logic (LangGraph)...
--------------------------------------------------
2026-02-14 10:02:19,821 - run_recursive_system - INFO - Building Recursive Intent Graph...
2026-02-14 10:02:19,824 - v2_logic.controllers.recursive_flow - INFO - [build_recursive_graph] Graph compiled.
2026-02-14 10:02:19,980 - run_recursive_system - INFO - Session started: session_1771063339
2026-02-14 10:02:19,980 - run_recursive_system - INFO - Target Intent: ['cup']
2026-02-14 10:02:20,037 - run_recursive_system - INFO - --- Processing Frame 0 ---
2026-02-14 10:02:20,039 - v2_logic.controllers.recursive_flow - INFO - [vjepa_brain_node] Encoding latent context
2026-02-14 10:02:26,283 - v2_logic.models.v_jepa_engine - INFO - [V-JEPA] Initialized on cuda
2026-02-14 10:02:46,648 - v2_logic.models.v_jepa_engine - INFO - [V-JEPA] Successfully loaded and aligned weights from /content/numeri-vjepa-experiment/Implementation/v2_logic/models/../../checkpoints/vjepa_vitl16.pth.tar
2026-02-14 10:02:46,709 - v2_logic.controllers.recursive_flow - INFO - [Engines] VJEPAEngine initialized
/usr/lib/python3.12/contextlib.py:105: FutureWarning: `torch.backends.cuda.sdp_kernel()` is deprecated. In the future, this context manager will be removed. Please see `torch.nn.attention.sdpa_kernel()` for the new context manager, with updated signature.
  self.gen = func(*args, **kwds)
2026-02-14 10:02:48,089 - v2_logic.controllers.recursive_flow - INFO - [Engines] SLMEngine initialized
2026-02-14 10:02:55,241 - numexpr.utils - INFO - NumExpr defaulting to 2 threads.
2026-02-14 10:02:57,023 - v2_logic.models.slm_engine - INFO - [SLMEngine] Loading VLM backend...
Loading VLM (Qwen2.5-VL) components...
 - Loading Model: Qwen/Qwen2-VL-7B-Instruct (4-bit)
2026-02-14 10:02:57,178 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
2026-02-14 10:02:57,208 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen2-VL-7B-Instruct/eed13092ef92e448dd6875b2a00151bd3f7db0ac/config.json "HTTP/1.1 200 OK"
2026-02-14 10:02:57,280 - httpx - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/Qwen/Qwen2-VL-7B-Instruct/eed13092ef92e448dd6875b2a00151bd3f7db0ac/config.json "HTTP/1.1 200 OK"
config.json: 1.20kB [00:00, 2.31MB/s]
2026-02-14 10:02:57,336 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/adapter_config.json "HTTP/1.1 404 Not Found"
2026-02-14 10:02:57,376 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
2026-02-14 10:02:57,406 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen2-VL-7B-Instruct/eed13092ef92e448dd6875b2a00151bd3f7db0ac/config.json "HTTP/1.1 200 OK"
2026-02-14 10:03:01,552 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/model.safetensors "HTTP/1.1 404 Not Found"
2026-02-14 10:03:01,593 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/model.safetensors.index.json "HTTP/1.1 307 Temporary Redirect"
2026-02-14 10:03:01,633 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen2-VL-7B-Instruct/eed13092ef92e448dd6875b2a00151bd3f7db0ac/model.safetensors.index.json "HTTP/1.1 200 OK"
2026-02-14 10:03:01,675 - httpx - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/Qwen/Qwen2-VL-7B-Instruct/eed13092ef92e448dd6875b2a00151bd3f7db0ac/model.safetensors.index.json "HTTP/1.1 200 OK"
model.safetensors.index.json: 56.5kB [00:00, 26.8MB/s]
2026-02-14 10:03:01,723 - httpx - INFO - HTTP Request: GET https://huggingface.co/api/models/Qwen/Qwen2-VL-7B-Instruct/revision/main "HTTP/1.1 200 OK"
Downloading (incomplete total...): 0.00B [00:00, ?B/s]
Fetching 5 files:   0% 0/5 [00:00<?, ?it/s]2026-02-14 10:03:01,769 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/eed13092ef92e448dd6875b2a00151bd3f7db0ac/model-00001-of-00005.safetensors "HTTP/1.1 302 Found"
2026-02-14 10:03:01,799 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/eed13092ef92e448dd6875b2a00151bd3f7db0ac/model-00005-of-00005.safetensors "HTTP/1.1 302 Found"
2026-02-14 10:03:01,802 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/eed13092ef92e448dd6875b2a00151bd3f7db0ac/model-00004-of-00005.safetensors "HTTP/1.1 302 Found"
2026-02-14 10:03:01,804 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/eed13092ef92e448dd6875b2a00151bd3f7db0ac/model-00002-of-00005.safetensors "HTTP/1.1 302 Found"
2026-02-14 10:03:01,805 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/eed13092ef92e448dd6875b2a00151bd3f7db0ac/model-00003-of-00005.safetensors "HTTP/1.1 302 Found"
2026-02-14 10:03:01,834 - httpx - INFO - HTTP Request: GET https://huggingface.co/api/models/Qwen/Qwen2-VL-7B-Instruct/xet-read-token/eed13092ef92e448dd6875b2a00151bd3f7db0ac "HTTP/1.1 200 OK"
Downloading (incomplete total...):   0% 0.00/3.90G [00:00<?, ?B/s]2026-02-14 10:03:01,842 - httpx - INFO - HTTP Request: GET https://huggingface.co/api/models/Qwen/Qwen2-VL-7B-Instruct/xet-read-token/eed13092ef92e448dd6875b2a00151bd3f7db0ac "HTTP/1.1 200 OK"
Downloading (incomplete total...):   0% 0.00/4.99G [00:00<?, ?B/s]2026-02-14 10:03:01,845 - httpx - INFO - HTTP Request: GET https://huggingface.co/api/models/Qwen/Qwen2-VL-7B-Instruct/xet-read-token/eed13092ef92e448dd6875b2a00151bd3f7db0ac "HTTP/1.1 200 OK"
Downloading (incomplete total...):   0% 0.00/8.85G [00:00<?, ?B/s]2026-02-14 10:03:01,845 - httpx - INFO - HTTP Request: GET https://huggingface.co/api/models/Qwen/Qwen2-VL-7B-Instruct/xet-read-token/eed13092ef92e448dd6875b2a00151bd3f7db0ac "HTTP/1.1 200 OK"
Downloading (incomplete total...):   0% 0.00/12.7G [00:00<?, ?B/s]2026-02-14 10:03:01,846 - httpx - INFO - HTTP Request: GET https://huggingface.co/api/models/Qwen/Qwen2-VL-7B-Instruct/xet-read-token/eed13092ef92e448dd6875b2a00151bd3f7db0ac "HTTP/1.1 200 OK"
Downloading (incomplete total...):  98% 16.3G/16.6G [05:02<00:04, 72.0MB/s]
Downloading (incomplete total...): 100% 16.6G/16.6G [05:03<00:00, 161MB/s]
Fetching 5 files: 100% 5/5 [05:03<00:00, 60.75s/it] 
Download complete: 100% 16.6G/16.6G [05:03<00:00, 54.6MB/s]
Loading weights: 100% 730/730 [01:09<00:00, 10.57it/s, Materializing param=model.visual.patch_embed.proj.weight]
2026-02-14 10:09:15,713 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/generation_config.json "HTTP/1.1 307 Temporary Redirect"
2026-02-14 10:09:15,756 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen2-VL-7B-Instruct/eed13092ef92e448dd6875b2a00151bd3f7db0ac/generation_config.json "HTTP/1.1 200 OK"
2026-02-14 10:09:15,797 - httpx - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/Qwen/Qwen2-VL-7B-Instruct/eed13092ef92e448dd6875b2a00151bd3f7db0ac/generation_config.json "HTTP/1.1 200 OK"
generation_config.json: 100% 244/244 [00:00<00:00, 1.50MB/s]
 - Loading Processor...
2026-02-14 10:09:15,842 - httpx - INFO - HTTP Request: GET https://huggingface.co/api/models/Qwen/Qwen2-VL-7B-Instruct/tree/main/additional_chat_templates?recursive=false&expand=false "HTTP/1.1 404 Not Found"
2026-02-14 10:09:15,883 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/processor_config.json "HTTP/1.1 404 Not Found"
2026-02-14 10:09:15,926 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/chat_template.json "HTTP/1.1 307 Temporary Redirect"
2026-02-14 10:09:15,966 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen2-VL-7B-Instruct/eed13092ef92e448dd6875b2a00151bd3f7db0ac/chat_template.json "HTTP/1.1 200 OK"
2026-02-14 10:09:16,009 - httpx - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/Qwen/Qwen2-VL-7B-Instruct/eed13092ef92e448dd6875b2a00151bd3f7db0ac/chat_template.json "HTTP/1.1 200 OK"
chat_template.json: 1.05kB [00:00, 415kB/s]
2026-02-14 10:09:16,056 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/chat_template.jinja "HTTP/1.1 404 Not Found"
2026-02-14 10:09:16,097 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json "HTTP/1.1 404 Not Found"
2026-02-14 10:09:16,141 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/processor_config.json "HTTP/1.1 404 Not Found"
2026-02-14 10:09:16,180 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/preprocessor_config.json "HTTP/1.1 307 Temporary Redirect"
2026-02-14 10:09:16,210 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen2-VL-7B-Instruct/eed13092ef92e448dd6875b2a00151bd3f7db0ac/preprocessor_config.json "HTTP/1.1 200 OK"
2026-02-14 10:09:16,256 - httpx - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/Qwen/Qwen2-VL-7B-Instruct/eed13092ef92e448dd6875b2a00151bd3f7db0ac/preprocessor_config.json "HTTP/1.1 200 OK"
preprocessor_config.json: 100% 347/347 [00:00<00:00, 2.21MB/s]
The image processor of type `Qwen2VLImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`. 
2026-02-14 10:09:16,375 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/processor_config.json "HTTP/1.1 404 Not Found"
2026-02-14 10:09:16,417 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/preprocessor_config.json "HTTP/1.1 307 Temporary Redirect"
2026-02-14 10:09:16,447 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen2-VL-7B-Instruct/eed13092ef92e448dd6875b2a00151bd3f7db0ac/preprocessor_config.json "HTTP/1.1 200 OK"
2026-02-14 10:09:16,486 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
2026-02-14 10:09:16,515 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen2-VL-7B-Instruct/eed13092ef92e448dd6875b2a00151bd3f7db0ac/config.json "HTTP/1.1 200 OK"
2026-02-14 10:09:16,583 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/tokenizer_config.json "HTTP/1.1 307 Temporary Redirect"
2026-02-14 10:09:16,624 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen2-VL-7B-Instruct/eed13092ef92e448dd6875b2a00151bd3f7db0ac/tokenizer_config.json "HTTP/1.1 200 OK"
2026-02-14 10:09:16,654 - httpx - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/Qwen/Qwen2-VL-7B-Instruct/eed13092ef92e448dd6875b2a00151bd3f7db0ac/tokenizer_config.json "HTTP/1.1 200 OK"
tokenizer_config.json: 4.19kB [00:00, 14.2MB/s]
2026-02-14 10:09:16,705 - httpx - INFO - HTTP Request: GET https://huggingface.co/api/models/Qwen/Qwen2-VL-7B-Instruct/tree/main/additional_chat_templates?recursive=false&expand=false "HTTP/1.1 404 Not Found"
2026-02-14 10:09:16,749 - httpx - INFO - HTTP Request: GET https://huggingface.co/api/models/Qwen/Qwen2-VL-7B-Instruct/tree/main?recursive=true&expand=false "HTTP/1.1 200 OK"
2026-02-14 10:09:16,825 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/vocab.json "HTTP/1.1 307 Temporary Redirect"
2026-02-14 10:09:16,854 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen2-VL-7B-Instruct/eed13092ef92e448dd6875b2a00151bd3f7db0ac/vocab.json "HTTP/1.1 200 OK"
2026-02-14 10:09:16,907 - httpx - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/Qwen/Qwen2-VL-7B-Instruct/eed13092ef92e448dd6875b2a00151bd3f7db0ac/vocab.json "HTTP/1.1 200 OK"
vocab.json: 2.78MB [00:00, 10.3MB/s]
2026-02-14 10:09:17,220 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/merges.txt "HTTP/1.1 307 Temporary Redirect"
2026-02-14 10:09:17,260 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen2-VL-7B-Instruct/eed13092ef92e448dd6875b2a00151bd3f7db0ac/merges.txt "HTTP/1.1 200 OK"
2026-02-14 10:09:17,291 - httpx - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/Qwen/Qwen2-VL-7B-Instruct/eed13092ef92e448dd6875b2a00151bd3f7db0ac/merges.txt "HTTP/1.1 200 OK"
merges.txt: 1.67MB [00:00, 77.2MB/s]
2026-02-14 10:09:17,354 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/tokenizer.json "HTTP/1.1 307 Temporary Redirect"
2026-02-14 10:09:17,400 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen2-VL-7B-Instruct/eed13092ef92e448dd6875b2a00151bd3f7db0ac/tokenizer.json "HTTP/1.1 200 OK"
2026-02-14 10:09:17,464 - httpx - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/Qwen/Qwen2-VL-7B-Instruct/eed13092ef92e448dd6875b2a00151bd3f7db0ac/tokenizer.json "HTTP/1.1 200 OK"
tokenizer.json: 7.03MB [00:00, 22.7MB/s]
2026-02-14 10:09:17,819 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/added_tokens.json "HTTP/1.1 404 Not Found"
2026-02-14 10:09:17,864 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/special_tokens_map.json "HTTP/1.1 404 Not Found"
2026-02-14 10:09:19,110 - httpx - INFO - HTTP Request: GET https://huggingface.co/api/models/Qwen/Qwen2-VL-7B-Instruct "HTTP/1.1 200 OK"
2026-02-14 10:09:19,153 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/processor_config.json "HTTP/1.1 404 Not Found"
2026-02-14 10:09:19,194 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/video_preprocessor_config.json "HTTP/1.1 404 Not Found"
2026-02-14 10:09:19,235 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/preprocessor_config.json "HTTP/1.1 307 Temporary Redirect"
2026-02-14 10:09:19,258 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen2-VL-7B-Instruct/eed13092ef92e448dd6875b2a00151bd3f7db0ac/preprocessor_config.json "HTTP/1.1 200 OK"
2026-02-14 10:09:19,304 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/processor_config.json "HTTP/1.1 404 Not Found"
2026-02-14 10:09:19,346 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/video_preprocessor_config.json "HTTP/1.1 404 Not Found"
2026-02-14 10:09:19,384 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/preprocessor_config.json "HTTP/1.1 307 Temporary Redirect"
2026-02-14 10:09:19,408 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen2-VL-7B-Instruct/eed13092ef92e448dd6875b2a00151bd3f7db0ac/preprocessor_config.json "HTTP/1.1 200 OK"
VLM loaded on: cuda:0
2026-02-14 10:09:20,429 - v2_logic.models.slm_engine - INFO - [SLMEngine] Requesting volume prior for 'cup'
2026-02-14 10:09:23,908 - v2_logic.models.slm_engine - INFO - [SLMEngine] Volume Response: 0.00025
2026-02-14 10:09:23,909 - v2_logic.controllers.recursive_flow - INFO - [director] SLM PHYSICAL PRIOR: Estimated volume for 'cup' is 0.00025 m^3
2026-02-14 10:09:23,938 - v2_logic.controllers.recursive_flow - INFO - [density_sensing_node] Analyzing physical density
2026-02-14 10:09:23,938 - v2_logic.controllers.recursive_flow - INFO - [countvid_executor_node] Counting objects matching intent: ['cup']
2026-02-14 10:09:23,938 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Generating point cloud for frame 0
2026-02-14 10:09:23,939 - v2_logic.controllers.recursive_flow - INFO - [v2e_sensor_node] Generating events
[DINOv2Engine] Loading DINOv2 ViT-B/14 on cuda...
Downloading: "https://github.com/facebookresearch/dinov2/zipball/main" to /root/.cache/torch/hub/main.zip
[SegmentationEngine] Loading SAM2...
[SegmentationEngine] No local checkpoint. Loading from HuggingFace...
2026-02-14 10:09:25,790 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/facebook/sam2.1-hiera-tiny/resolve/main/sam2.1_hiera_tiny.pt "HTTP/1.1 302 Found"
2026-02-14 10:09:25,839 - httpx - INFO - HTTP Request: GET https://huggingface.co/api/models/facebook/sam2.1-hiera-tiny/xet-read-token/de431c4043854a71d8101e17995dfe596bf101a5 "HTTP/1.1 200 OK"
sam2.1_hiera_tiny.pt:   0% 0.00/156M [00:00<?, ?B/s]/root/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/swiglu_ffn.py:51: UserWarning: xFormers is not available (SwiGLU)
  warnings.warn("xFormers is not available (SwiGLU)")
/root/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/attention.py:33: UserWarning: xFormers is not available (Attention)
  warnings.warn("xFormers is not available (Attention)")
/root/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/block.py:40: UserWarning: xFormers is not available (Block)
  warnings.warn("xFormers is not available (Block)")
2026-02-14 10:09:26,002 - dinov2 - INFO - using MLP layer as FFN
sam2.1_hiera_tiny.pt: 100% 156M/156M [00:02<00:00, 72.8MB/s] 
Downloading: "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth" to /root/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth
  0% 0.00/330M [00:00<?, ?B/s]2026-02-14 10:09:33,798 - v2ecore.emulator - INFO - ON/OFF log_e temporal contrast thresholds: 0.2 / 0.2 +/- 0.03
  1% 4.25M/330M [00:00<00:19, 17.9MB/s]2026-02-14 10:09:33,900 - v2ecore.emulator - WARNING - cannot get screen size for window placement: No enumerators available
2026-02-14 10:09:33,904 - v2_logic.models.v2e_engine - INFO - [V2E] Initialized on cuda
2026-02-14 10:09:33,905 - v2_logic.controllers.recursive_flow - INFO - [Engines] V2EEngine initialized
 28% 93.0M/330M [00:01<00:02, 107MB/s]2026-02-14 10:09:34,582 - root - INFO - Loaded checkpoint sucessfully
 37% 122M/330M [00:01<00:01, 113MB/s][SegmentationEngine] Loaded from HuggingFace: facebook/sam2.1-hiera-tiny
[SegmentationEngine] SAM2 loaded on cuda
2026-02-14 10:09:35,129 - v2_logic.controllers.recursive_flow - INFO - [Engines] SegmentationEngine initialized with loosened ROI (0.02)
 40% 133M/330M [00:01<00:02, 71.2MB/s]2026-02-14 10:09:35,201 - dinov2 - WARNING - xFormers not available
2026-02-14 10:09:35,208 - dinov2 - WARNING - xFormers not available
2026-02-14 10:09:35,218 - v2_logic.models.depth_engine - INFO - [DepthEngine] Loading from checkpoint: /content/numeri-vjepa-experiment/Techs/Depth-Anything-V2-main/Depth-Anything-V2-main/checkpoints/depth_anything_v2_vits.pth
2026-02-14 10:09:35,231 - dinov2 - INFO - using MLP layer as FFN
100% 330M/330M [00:04<00:00, 74.9MB/s]
2026-02-14 10:09:39,068 - v2_logic.models.depth_engine - INFO - [DepthEngine] Model loaded on cuda (encoder=vits)
2026-02-14 10:09:39,070 - v2_logic.controllers.recursive_flow - INFO - [Engines] DepthEngine initialized
/usr/local/lib/python3.12/dist-packages/timm/models/layers/__init__.py:49: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
/content/numeri-vjepa-experiment/Techs/CountVid-main/CountVid-main/models/GroundingDINO/utils.py:67: SyntaxWarning: invalid escape sequence '\s'
  - memory: bs, \sum{hw}, d_model
/usr/local/lib/python3.12/dist-packages/torch/functional.py:505: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at /pytorch/aten/src/ATen/native/TensorShape.cpp:4381.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
2026-02-14 10:09:39,526 - v2_logic.controllers.recursive_flow - INFO - [Engines] DINOv2Engine initialized
2026-02-14 10:09:39,731 - v2_logic.controllers.recursive_flow - INFO - [Engines] DensityPredictor initialized
2026-02-14 10:09:39,755 - root - INFO - For numpy array image, we assume (HxWxC) format
[DensityPredictor] Warning: Model not fitted. Returning default density 1.0.
2026-02-14 10:09:39,961 - root - INFO - Computing image embeddings for the provided image...
2026-02-14 10:09:40,115 - root - INFO - Image embeddings computed.
[SegmentationEngine] Filtered 2 objects (outside ROI/area range)
final text_encoder_type: /content/numeri-vjepa-experiment/Techs/CountVid-main/CountVid-main/checkpoints/bert-base-uncased
load tokenizer done.
Loading weights: 100% 199/199 [00:00<00:00, 443.27it/s, Materializing param=pooler.dense.weight]
BertModel LOAD REPORT from: /content/numeri-vjepa-experiment/Techs/CountVid-main/CountVid-main/checkpoints/bert-base-uncased
Key                                        | Status     |  | 
-------------------------------------------+------------+--+-
cls.predictions.transform.LayerNorm.weight | UNEXPECTED |  | 
cls.predictions.transform.dense.weight     | UNEXPECTED |  | 
cls.predictions.transform.LayerNorm.bias   | UNEXPECTED |  | 
cls.seq_relationship.weight                | UNEXPECTED |  | 
cls.predictions.transform.dense.bias       | UNEXPECTED |  | 
cls.seq_relationship.bias                  | UNEXPECTED |  | 
cls.predictions.bias                       | UNEXPECTED |  | 

Notes:
- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.
[SegmentationEngine] Union Masking: merged 3 fragments → 13 physical units
final text_encoder_type: /content/numeri-vjepa-experiment/Techs/CountVid-main/CountVid-main/checkpoints/bert-base-uncased
load tokenizer done.
[SegmentationEngine] Clustered 13 masks into 8 volumetric clusters
2026-02-14 10:09:44,293 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Aggregated volume: 48.172904 m^3 (48172904.11 cm^3) from 8 clusters
2026-02-14 10:09:50,013 - v2_logic.models.count_vid_engine - INFO - [CountVid] Model loaded successfully on cuda
2026-02-14 10:09:50,027 - v2_logic.controllers.recursive_flow - INFO - [Engines] CountVidEngine initialized
[CountVid Patch] ⚠️ Smart Dispatcher: Falling back to Positional Signature.
[CountVid Patch] ⚠️ Smart Dispatcher: Falling back for get_head_mask.
/content/numeri-vjepa-experiment/Techs/CountVid-main/CountVid-main/models/GroundingDINO/transformer.py:901: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=False):
2026-02-14 10:09:51,126 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] Computing volumetric reconciliation
2026-02-14 10:09:51,828 - v2_logic.controllers.recursive_flow - INFO - [Engines] AlphaHullWrapper initialized
2026-02-14 10:09:51,828 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] 3DC Result: 192691.62 units (From V_stack: 48.172904 m^3, rho: 1.00, V_unit: 0.000250 m^3)
2026-02-14 10:09:51,828 - v2_logic.controllers.recursive_flow - WARNING - [v3_math_node] Sanity Check Failed: Count 192691.61644588178 exceeds safety buffer 1000. Clipping n_vol.
2026-02-14 10:09:51,829 - v2_logic.controllers.recursive_flow - INFO - [fusion_engine_node] Fusing sensor data & Tracking
2026-02-14 10:09:51,835 - v2_logic.controllers.recursive_flow - INFO - [Engines] FusionEngineV2 initialized
2026-02-14 10:09:51,838 - v2_logic.controllers.recursive_flow - INFO - [Engines] ReIDEngine initialized
2026-02-14 10:09:51,888 - v2_logic.models.fusion_engine_v2 - INFO - [FusionV2] Shields: spatial: ✓ (1.00) | volumetric: ✗ (0.00) | latent: ✓ (0.50) → confidence=0.600 (threshold=0.60)
2026-02-14 10:09:51,892 - v2_logic.controllers.recursive_flow - INFO - [Engines] LogicGate initialized
2026-02-14 10:09:51,892 - v2_logic.controllers.recursive_flow - INFO - [logic_gate_node] Decision: loop (Rule: Rule3_VolumetricAnomaly)
2026-02-14 10:09:51,895 - v2_logic.controllers.recursive_flow - INFO - [route] Checking decision status: loop
2026-02-14 10:09:51,895 - v2_logic.controllers.recursive_flow - INFO - [targeted_slm_node] SLM reasoning triggered
2026-02-14 10:09:51,896 - v2_logic.models.slm_engine - INFO - [SLMEngine] Prompt: You are an intelligent visual analyst. I counted 3 objects visually, but the 3D volume suggests there should be between 1000 and 1000 objects. This is a discrepancy. Are some objects occluded (hidden behind others)? Or are some counts false positives? Analyze the spatial arrangement.
Provide a concise explanation and a hypothesis.
2026-02-14 10:10:04,919 - v2_logic.models.slm_engine - INFO - [SLMEngine] Response: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table. There are three objects visible: the person's hand, the blue cup they are holding, and the two blue cups on the table. However, the 3D volume suggests there should be between 1000 and 1000 objects, which is a significant discrepancy.

The discrepancy likely arises from the following:

1. **Occlusion**: The person's body is partially obscuring the view of the objects on the table. The person's hand and body are in front of the cups and the ball
2026-02-14 10:10:04,920 - v2_logic.controllers.recursive_flow - INFO - [interpolation_node] Interpolating state to current frame
2026-02-14 10:10:04,921 - v2_logic.controllers.recursive_flow - INFO - [director] SLM Hypothesis received: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table
2026-02-14 10:10:04,921 - v2_logic.controllers.recursive_flow - INFO - [director] DISCOVERY LOOP: Added new labels: ['ball', 'person']
2026-02-14 10:10:04,922 - v2_logic.models.slm_engine - INFO - [SLMEngine] Requesting volume prior for 'cup'
2026-02-14 10:10:05,859 - v2_logic.models.slm_engine - INFO - [SLMEngine] Volume Response: 0.00025
2026-02-14 10:10:05,862 - v2_logic.controllers.recursive_flow - INFO - [director] SLM PHYSICAL PRIOR: Estimated volume for 'cup' is 0.00025 m^3
2026-02-14 10:10:05,863 - v2_logic.controllers.recursive_flow - INFO - [countvid_executor_node] Counting objects matching intent: ['cup', 'ball', 'person']
2026-02-14 10:10:05,864 - v2_logic.controllers.recursive_flow - INFO - [density_sensing_node] Analyzing physical density
2026-02-14 10:10:05,864 - v2_logic.controllers.recursive_flow - INFO - [v2e_sensor_node] Generating events
2026-02-14 10:10:05,864 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Generating point cloud for frame 0
2026-02-14 10:10:05,924 - v2ecore.emulator - WARNING - no signal events generated for frame #2 at t=0.0000s
[DensityPredictor] Warning: Model not fitted. Returning default density 1.0.
2026-02-14 10:10:06,097 - root - INFO - For numpy array image, we assume (HxWxC) format
2026-02-14 10:10:06,604 - root - INFO - Computing image embeddings for the provided image...
2026-02-14 10:10:06,629 - root - INFO - Image embeddings computed.
[SegmentationEngine] Filtered 2 objects (outside ROI/area range)
[SegmentationEngine] Union Masking: merged 3 fragments → 13 physical units
[SegmentationEngine] Clustered 13 masks into 8 volumetric clusters
2026-02-14 10:10:09,296 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Aggregated volume: 48.172904 m^3 (48172904.11 cm^3) from 8 clusters
2026-02-14 10:10:09,298 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] Computing volumetric reconciliation
2026-02-14 10:10:09,298 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] 3DC Result: 192691.62 units (From V_stack: 48.172904 m^3, rho: 1.00, V_unit: 0.000250 m^3)
2026-02-14 10:10:09,298 - v2_logic.controllers.recursive_flow - WARNING - [v3_math_node] Sanity Check Failed: Count 192691.61644588178 exceeds safety buffer 1000. Clipping n_vol.
2026-02-14 10:10:09,299 - v2_logic.controllers.recursive_flow - INFO - [fusion_engine_node] Fusing sensor data & Tracking
2026-02-14 10:10:09,324 - v2_logic.models.fusion_engine_v2 - INFO - [FusionV2] Shields: spatial: ✓ (1.00) | volumetric: ✗ (0.00) | latent: ✓ (0.50) → confidence=0.600 (threshold=0.60)
2026-02-14 10:10:09,327 - v2_logic.controllers.recursive_flow - INFO - [logic_gate_node] Decision: loop (Rule: Rule3_VolumetricAnomaly)
2026-02-14 10:10:09,327 - v2_logic.controllers.recursive_flow - INFO - [route] Checking decision status: loop
2026-02-14 10:10:09,327 - v2_logic.controllers.recursive_flow - INFO - [targeted_slm_node] SLM reasoning triggered
2026-02-14 10:10:09,327 - v2_logic.models.slm_engine - INFO - [SLMEngine] Prompt: You are an intelligent visual analyst. I counted 3 objects visually, but the 3D volume suggests there should be between 1000 and 1000 objects. This is a discrepancy. Are some objects occluded (hidden behind others)? Or are some counts false positives? Analyze the spatial arrangement.
Provide a concise explanation and a hypothesis.
2026-02-14 10:10:22,508 - v2_logic.models.slm_engine - INFO - [SLMEngine] Response: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table. There are three objects visible: the person's hand, the blue cup they are holding, and the two blue cups on the table. However, the 3D volume suggests there should be between 1000 and 1000 objects, which is a significant discrepancy.

The discrepancy likely arises from the following:

1. **Occlusion**: The person's body is partially obscuring the view of the objects on the table. The person's hand and body are in front of the cups and the ball
2026-02-14 10:10:22,509 - v2_logic.controllers.recursive_flow - INFO - [interpolation_node] Interpolating state to current frame
2026-02-14 10:10:22,510 - v2_logic.controllers.recursive_flow - INFO - [director] SLM Hypothesis received: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table
2026-02-14 10:10:22,510 - v2_logic.models.slm_engine - INFO - [SLMEngine] Requesting volume prior for 'cup'
2026-02-14 10:10:23,432 - v2_logic.models.slm_engine - INFO - [SLMEngine] Volume Response: 0.00025
2026-02-14 10:10:23,432 - v2_logic.controllers.recursive_flow - INFO - [director] SLM PHYSICAL PRIOR: Estimated volume for 'cup' is 0.00025 m^3
2026-02-14 10:10:23,433 - v2_logic.controllers.recursive_flow - INFO - [countvid_executor_node] Counting objects matching intent: ['cup', 'ball', 'person']
2026-02-14 10:10:23,434 - v2_logic.controllers.recursive_flow - INFO - [density_sensing_node] Analyzing physical density
2026-02-14 10:10:23,434 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Generating point cloud for frame 0
2026-02-14 10:10:23,434 - v2_logic.controllers.recursive_flow - INFO - [v2e_sensor_node] Generating events
2026-02-14 10:10:23,486 - v2ecore.emulator - WARNING - no signal events generated for frame #3 at t=0.0000s
[DensityPredictor] Warning: Model not fitted. Returning default density 1.0.
2026-02-14 10:10:23,713 - root - INFO - For numpy array image, we assume (HxWxC) format
2026-02-14 10:10:23,939 - root - INFO - Computing image embeddings for the provided image...
2026-02-14 10:10:23,968 - root - INFO - Image embeddings computed.
[SegmentationEngine] Filtered 2 objects (outside ROI/area range)
[SegmentationEngine] Union Masking: merged 3 fragments → 13 physical units
[SegmentationEngine] Clustered 13 masks into 8 volumetric clusters
2026-02-14 10:10:26,762 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Aggregated volume: 48.172904 m^3 (48172904.11 cm^3) from 8 clusters
2026-02-14 10:10:26,763 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] Computing volumetric reconciliation
2026-02-14 10:10:26,764 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] 3DC Result: 192691.62 units (From V_stack: 48.172904 m^3, rho: 1.00, V_unit: 0.000250 m^3)
2026-02-14 10:10:26,764 - v2_logic.controllers.recursive_flow - WARNING - [v3_math_node] Sanity Check Failed: Count 192691.61644588178 exceeds safety buffer 1000. Clipping n_vol.
2026-02-14 10:10:26,764 - v2_logic.controllers.recursive_flow - INFO - [fusion_engine_node] Fusing sensor data & Tracking
2026-02-14 10:10:26,788 - v2_logic.models.fusion_engine_v2 - INFO - [FusionV2] Shields: spatial: ✓ (1.00) | volumetric: ✗ (0.00) | latent: ✓ (0.50) → confidence=0.600 (threshold=0.60)
2026-02-14 10:10:26,789 - v2_logic.controllers.recursive_flow - INFO - [logic_gate_node] Decision: loop (Rule: Rule3_VolumetricAnomaly)
2026-02-14 10:10:26,790 - v2_logic.controllers.recursive_flow - INFO - [route] Checking decision status: loop
2026-02-14 10:10:26,790 - v2_logic.controllers.recursive_flow - INFO - [targeted_slm_node] SLM reasoning triggered
2026-02-14 10:10:26,790 - v2_logic.models.slm_engine - INFO - [SLMEngine] Prompt: You are an intelligent visual analyst. I counted 3 objects visually, but the 3D volume suggests there should be between 1000 and 1000 objects. This is a discrepancy. Are some objects occluded (hidden behind others)? Or are some counts false positives? Analyze the spatial arrangement.
Provide a concise explanation and a hypothesis.
2026-02-14 10:10:39,924 - v2_logic.models.slm_engine - INFO - [SLMEngine] Response: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table. There are three objects visible: the person's hand, the blue cup they are holding, and the two blue cups on the table. However, the 3D volume suggests there should be between 1000 and 1000 objects, which is a significant discrepancy.

The discrepancy likely arises from the following:

1. **Occlusion**: The person's body is partially obscuring the view of the objects on the table. The person's hand and body are in front of the cups and the ball
2026-02-14 10:10:39,924 - v2_logic.controllers.recursive_flow - INFO - [interpolation_node] Interpolating state to current frame
2026-02-14 10:10:39,925 - v2_logic.controllers.recursive_flow - INFO - [director] SLM Hypothesis received: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table
2026-02-14 10:10:39,925 - v2_logic.models.slm_engine - INFO - [SLMEngine] Requesting volume prior for 'cup'
2026-02-14 10:10:40,869 - v2_logic.models.slm_engine - INFO - [SLMEngine] Volume Response: 0.00025
2026-02-14 10:10:40,869 - v2_logic.controllers.recursive_flow - INFO - [director] SLM PHYSICAL PRIOR: Estimated volume for 'cup' is 0.00025 m^3
2026-02-14 10:10:40,870 - v2_logic.controllers.recursive_flow - INFO - [countvid_executor_node] Counting objects matching intent: ['cup', 'ball', 'person']
2026-02-14 10:10:40,871 - v2_logic.controllers.recursive_flow - INFO - [density_sensing_node] Analyzing physical density
2026-02-14 10:10:40,872 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Generating point cloud for frame 0
2026-02-14 10:10:40,873 - v2_logic.controllers.recursive_flow - INFO - [v2e_sensor_node] Generating events
2026-02-14 10:10:40,921 - v2ecore.emulator - WARNING - no signal events generated for frame #4 at t=0.0000s
[DensityPredictor] Warning: Model not fitted. Returning default density 1.0.
2026-02-14 10:10:41,183 - root - INFO - For numpy array image, we assume (HxWxC) format
2026-02-14 10:10:41,524 - root - INFO - Computing image embeddings for the provided image...
2026-02-14 10:10:41,554 - root - INFO - Image embeddings computed.
[SegmentationEngine] Filtered 2 objects (outside ROI/area range)
[SegmentationEngine] Union Masking: merged 3 fragments → 13 physical units
[SegmentationEngine] Clustered 13 masks into 8 volumetric clusters
2026-02-14 10:10:44,625 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Aggregated volume: 48.172904 m^3 (48172904.11 cm^3) from 8 clusters
2026-02-14 10:10:44,627 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] Computing volumetric reconciliation
2026-02-14 10:10:44,627 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] 3DC Result: 192691.62 units (From V_stack: 48.172904 m^3, rho: 1.00, V_unit: 0.000250 m^3)
2026-02-14 10:10:44,627 - v2_logic.controllers.recursive_flow - WARNING - [v3_math_node] Sanity Check Failed: Count 192691.61644588178 exceeds safety buffer 1000. Clipping n_vol.
2026-02-14 10:10:44,628 - v2_logic.controllers.recursive_flow - INFO - [fusion_engine_node] Fusing sensor data & Tracking
2026-02-14 10:10:44,651 - v2_logic.models.fusion_engine_v2 - INFO - [FusionV2] Shields: spatial: ✓ (1.00) | volumetric: ✗ (0.00) | latent: ✓ (0.50) → confidence=0.600 (threshold=0.60)
2026-02-14 10:10:44,652 - v2_logic.controllers.recursive_flow - INFO - [logic_gate_node] Decision: exit (Rule: MaxLoopSafety)
2026-02-14 10:10:44,653 - v2_logic.controllers.recursive_flow - INFO - [route] Checking decision status: exit
2026-02-14 10:10:44,654 - run_recursive_system - INFO - Frame 0 Results:
2026-02-14 10:10:44,654 - run_recursive_system - INFO -   - Count (N_visible): 3
2026-02-14 10:10:44,654 - run_recursive_system - INFO -   - Volumetric Range: (1000, 1000)
2026-02-14 10:10:44,654 - run_recursive_system - INFO -   - Spike Energy: 0.00
2026-02-14 10:10:44,654 - run_recursive_system - INFO -   - Anomaly Status: exit
2026-02-14 10:10:44,654 - run_recursive_system - INFO -   - SLM Reasoning: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table. There are three objects visible: the person's hand, the blue cup they are holding, and the two blue cups on the table. However, the 3D volume suggests there should be between 1000 and 1000 objects, which is a significant discrepancy.

The discrepancy likely arises from the following:

1. **Occlusion**: The person's body is partially obscuring the view of the objects on the table. The person's hand and body are in front of the cups and the ball
2026-02-14 10:10:44,781 - run_recursive_system - INFO - --- Processing Frame 15 ---
2026-02-14 10:10:44,783 - v2_logic.controllers.recursive_flow - INFO - [vjepa_brain_node] Encoding latent context
/usr/lib/python3.12/contextlib.py:105: FutureWarning: `torch.backends.cuda.sdp_kernel()` is deprecated. In the future, this context manager will be removed. Please see `torch.nn.attention.sdpa_kernel()` for the new context manager, with updated signature.
  self.gen = func(*args, **kwds)
2026-02-14 10:10:45,210 - v2_logic.controllers.recursive_flow - INFO - [director] SLM Hypothesis received: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table
2026-02-14 10:10:45,210 - v2_logic.models.slm_engine - INFO - [SLMEngine] Requesting volume prior for 'cup'
2026-02-14 10:10:46,137 - v2_logic.models.slm_engine - INFO - [SLMEngine] Volume Response: 0.00025
2026-02-14 10:10:46,137 - v2_logic.controllers.recursive_flow - INFO - [director] SLM PHYSICAL PRIOR: Estimated volume for 'cup' is 0.00025 m^3
2026-02-14 10:10:46,138 - v2_logic.controllers.recursive_flow - INFO - [countvid_executor_node] Counting objects matching intent: ['cup', 'ball', 'person']
2026-02-14 10:10:46,140 - v2_logic.controllers.recursive_flow - INFO - [density_sensing_node] Analyzing physical density
2026-02-14 10:10:46,141 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Generating point cloud for frame 15
2026-02-14 10:10:46,143 - v2_logic.controllers.recursive_flow - INFO - [v2e_sensor_node] Generating events
[DensityPredictor] Warning: Model not fitted. Returning default density 1.0.
2026-02-14 10:10:46,355 - root - INFO - For numpy array image, we assume (HxWxC) format
2026-02-14 10:10:46,644 - root - INFO - Computing image embeddings for the provided image...
2026-02-14 10:10:46,683 - root - INFO - Image embeddings computed.
[SegmentationEngine] Filtered 2 objects (outside ROI/area range)
[SegmentationEngine] Clustered 12 masks into 8 volumetric clusters
2026-02-14 10:10:49,515 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Aggregated volume: 41.317014 m^3 (41317013.94 cm^3) from 8 clusters
2026-02-14 10:10:49,516 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] Computing volumetric reconciliation
2026-02-14 10:10:49,516 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] 3DC Result: 192691.62 units (From V_stack: 48.172904 m^3, rho: 1.00, V_unit: 0.000250 m^3)
2026-02-14 10:10:49,516 - v2_logic.controllers.recursive_flow - WARNING - [v3_math_node] Sanity Check Failed: Count 192691.61644588178 exceeds safety buffer 1000. Clipping n_vol.
2026-02-14 10:10:49,517 - v2_logic.controllers.recursive_flow - INFO - [fusion_engine_node] Fusing sensor data & Tracking
2026-02-14 10:10:49,558 - v2_logic.models.fusion_engine_v2 - INFO - [FusionV2] Shields: spatial: ✓ (0.98) | volumetric: ✗ (0.00) | latent: ✓ (0.50) → confidence=0.590 (threshold=0.60)
2026-02-14 10:10:49,559 - v2_logic.models.fusion_engine_v2 - WARNING - [FusionV2] LOW CONFIDENCE (0.590) — SLM Audit recommended.
2026-02-14 10:10:49,560 - v2_logic.controllers.recursive_flow - INFO - [logic_gate_node] Decision: exit (Rule: MaxLoopSafety)
2026-02-14 10:10:49,561 - v2_logic.controllers.recursive_flow - INFO - [route] Checking decision status: exit
2026-02-14 10:10:49,562 - run_recursive_system - INFO - Frame 15 Results:
2026-02-14 10:10:49,562 - run_recursive_system - INFO -   - Count (N_visible): 3
2026-02-14 10:10:49,562 - run_recursive_system - INFO -   - Volumetric Range: (1000, 1000)
2026-02-14 10:10:49,562 - run_recursive_system - INFO -   - Spike Energy: 363500.00
2026-02-14 10:10:49,562 - run_recursive_system - INFO -   - Anomaly Status: exit
2026-02-14 10:10:49,562 - run_recursive_system - INFO -   - SLM Reasoning: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table. There are three objects visible: the person's hand, the blue cup they are holding, and the two blue cups on the table. However, the 3D volume suggests there should be between 1000 and 1000 objects, which is a significant discrepancy.

The discrepancy likely arises from the following:

1. **Occlusion**: The person's body is partially obscuring the view of the objects on the table. The person's hand and body are in front of the cups and the ball
2026-02-14 10:10:49,604 - run_recursive_system - INFO - --- Processing Frame 30 ---
2026-02-14 10:10:49,606 - v2_logic.controllers.recursive_flow - INFO - [vjepa_brain_node] Encoding latent context
2026-02-14 10:10:50,042 - v2_logic.controllers.recursive_flow - INFO - [director] SLM Hypothesis received: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table
2026-02-14 10:10:50,042 - v2_logic.models.slm_engine - INFO - [SLMEngine] Requesting volume prior for 'cup'
2026-02-14 10:10:50,994 - v2_logic.models.slm_engine - INFO - [SLMEngine] Volume Response: 0.00025
2026-02-14 10:10:50,994 - v2_logic.controllers.recursive_flow - INFO - [director] SLM PHYSICAL PRIOR: Estimated volume for 'cup' is 0.00025 m^3
2026-02-14 10:10:50,996 - v2_logic.controllers.recursive_flow - INFO - [countvid_executor_node] Counting objects matching intent: ['cup', 'ball', 'person']
2026-02-14 10:10:50,997 - v2_logic.controllers.recursive_flow - INFO - [density_sensing_node] Analyzing physical density
2026-02-14 10:10:50,998 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Generating point cloud for frame 30
2026-02-14 10:10:51,000 - v2_logic.controllers.recursive_flow - INFO - [v2e_sensor_node] Generating events
[DensityPredictor] Warning: Model not fitted. Returning default density 1.0.
2026-02-14 10:10:51,230 - root - INFO - For numpy array image, we assume (HxWxC) format
2026-02-14 10:10:51,409 - root - INFO - Computing image embeddings for the provided image...
2026-02-14 10:10:51,452 - root - INFO - Image embeddings computed.
[SegmentationEngine] Filtered 3 objects (outside ROI/area range)
[SegmentationEngine] Union Masking: merged 1 fragments → 12 physical units
[SegmentationEngine] Clustered 12 masks into 7 volumetric clusters
2026-02-14 10:10:54,558 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Aggregated volume: 66.826012 m^3 (66826011.80 cm^3) from 7 clusters
2026-02-14 10:10:54,562 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] Computing volumetric reconciliation
2026-02-14 10:10:54,562 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] 3DC Result: 192691.62 units (From V_stack: 48.172904 m^3, rho: 1.00, V_unit: 0.000250 m^3)
2026-02-14 10:10:54,562 - v2_logic.controllers.recursive_flow - WARNING - [v3_math_node] Sanity Check Failed: Count 192691.61644588178 exceeds safety buffer 1000. Clipping n_vol.
2026-02-14 10:10:54,563 - v2_logic.controllers.recursive_flow - INFO - [fusion_engine_node] Fusing sensor data & Tracking
2026-02-14 10:10:54,638 - v2_logic.models.fusion_engine_v2 - INFO - [FusionV2] Shields: spatial: ✓ (0.96) | volumetric: ✗ (0.00) | latent: ✓ (0.50) → confidence=0.580 (threshold=0.60)
2026-02-14 10:10:54,638 - v2_logic.models.fusion_engine_v2 - WARNING - [FusionV2] LOW CONFIDENCE (0.580) — SLM Audit recommended.
2026-02-14 10:10:54,641 - v2_logic.controllers.recursive_flow - INFO - [logic_gate_node] Decision: exit (Rule: MaxLoopSafety)
2026-02-14 10:10:54,641 - v2_logic.controllers.recursive_flow - INFO - [route] Checking decision status: exit
2026-02-14 10:10:54,643 - run_recursive_system - INFO - Frame 30 Results:
2026-02-14 10:10:54,643 - run_recursive_system - INFO -   - Count (N_visible): 3
2026-02-14 10:10:54,643 - run_recursive_system - INFO -   - Volumetric Range: (1000, 1000)
2026-02-14 10:10:54,643 - run_recursive_system - INFO -   - Spike Energy: 829775.00
2026-02-14 10:10:54,643 - run_recursive_system - INFO -   - Anomaly Status: exit
2026-02-14 10:10:54,643 - run_recursive_system - INFO -   - SLM Reasoning: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table. There are three objects visible: the person's hand, the blue cup they are holding, and the two blue cups on the table. However, the 3D volume suggests there should be between 1000 and 1000 objects, which is a significant discrepancy.

The discrepancy likely arises from the following:

1. **Occlusion**: The person's body is partially obscuring the view of the objects on the table. The person's hand and body are in front of the cups and the ball
2026-02-14 10:10:54,722 - run_recursive_system - INFO - --- Processing Frame 45 ---
2026-02-14 10:10:54,728 - v2_logic.controllers.recursive_flow - INFO - [vjepa_brain_node] Encoding latent context
2026-02-14 10:10:55,171 - v2_logic.controllers.recursive_flow - INFO - [director] SLM Hypothesis received: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table
2026-02-14 10:10:55,172 - v2_logic.models.slm_engine - INFO - [SLMEngine] Requesting volume prior for 'cup'
2026-02-14 10:10:56,438 - v2_logic.models.slm_engine - INFO - [SLMEngine] Volume Response: 0.00025
2026-02-14 10:10:56,438 - v2_logic.controllers.recursive_flow - INFO - [director] SLM PHYSICAL PRIOR: Estimated volume for 'cup' is 0.00025 m^3
2026-02-14 10:10:56,442 - v2_logic.controllers.recursive_flow - INFO - [countvid_executor_node] Counting objects matching intent: ['cup', 'ball', 'person']
2026-02-14 10:10:56,443 - v2_logic.controllers.recursive_flow - INFO - [density_sensing_node] Analyzing physical density
2026-02-14 10:10:56,445 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Generating point cloud for frame 45
2026-02-14 10:10:56,445 - v2_logic.controllers.recursive_flow - INFO - [v2e_sensor_node] Generating events
[DensityPredictor] Warning: Model not fitted. Returning default density 1.0.
2026-02-14 10:10:56,700 - root - INFO - For numpy array image, we assume (HxWxC) format
2026-02-14 10:10:57,176 - root - INFO - Computing image embeddings for the provided image...
2026-02-14 10:10:57,200 - root - INFO - Image embeddings computed.
[SegmentationEngine] Filtered 2 objects (outside ROI/area range)
[SegmentationEngine] Clustered 15 masks into 7 volumetric clusters
2026-02-14 10:10:59,927 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Aggregated volume: 89.154907 m^3 (89154907.49 cm^3) from 7 clusters
2026-02-14 10:10:59,932 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] Computing volumetric reconciliation
2026-02-14 10:10:59,932 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] 3DC Result: 192691.62 units (From V_stack: 48.172904 m^3, rho: 1.00, V_unit: 0.000250 m^3)
2026-02-14 10:10:59,932 - v2_logic.controllers.recursive_flow - WARNING - [v3_math_node] Sanity Check Failed: Count 192691.61644588178 exceeds safety buffer 1000. Clipping n_vol.
2026-02-14 10:10:59,934 - v2_logic.controllers.recursive_flow - INFO - [fusion_engine_node] Fusing sensor data & Tracking
2026-02-14 10:10:59,981 - v2_logic.models.fusion_engine_v2 - INFO - [FusionV2] Shields: spatial: ✗ (0.65) | volumetric: ✗ (0.00) | latent: ✓ (0.50) → confidence=0.426 (threshold=0.60)
2026-02-14 10:10:59,981 - v2_logic.models.fusion_engine_v2 - WARNING - [FusionV2] LOW CONFIDENCE (0.426) — SLM Audit recommended.
2026-02-14 10:10:59,983 - v2_logic.controllers.recursive_flow - INFO - [logic_gate_node] Decision: exit (Rule: MaxLoopSafety)
2026-02-14 10:10:59,984 - v2_logic.controllers.recursive_flow - INFO - [route] Checking decision status: exit
2026-02-14 10:10:59,985 - run_recursive_system - INFO - Frame 45 Results:
2026-02-14 10:10:59,985 - run_recursive_system - INFO -   - Count (N_visible): 3
2026-02-14 10:10:59,985 - run_recursive_system - INFO -   - Volumetric Range: (1000, 1000)
2026-02-14 10:10:59,985 - run_recursive_system - INFO -   - Spike Energy: 558771.00
2026-02-14 10:10:59,985 - run_recursive_system - INFO -   - Anomaly Status: exit
2026-02-14 10:10:59,985 - run_recursive_system - INFO -   - SLM Reasoning: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table. There are three objects visible: the person's hand, the blue cup they are holding, and the two blue cups on the table. However, the 3D volume suggests there should be between 1000 and 1000 objects, which is a significant discrepancy.

The discrepancy likely arises from the following:

1. **Occlusion**: The person's body is partially obscuring the view of the objects on the table. The person's hand and body are in front of the cups and the ball
2026-02-14 10:11:00,037 - run_recursive_system - INFO - --- Processing Frame 60 ---
2026-02-14 10:11:00,040 - v2_logic.controllers.recursive_flow - INFO - [vjepa_brain_node] Encoding latent context
2026-02-14 10:11:00,483 - v2_logic.controllers.recursive_flow - INFO - [director] SLM Hypothesis received: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table
2026-02-14 10:11:00,483 - v2_logic.models.slm_engine - INFO - [SLMEngine] Requesting volume prior for 'cup'
2026-02-14 10:11:01,427 - v2_logic.models.slm_engine - INFO - [SLMEngine] Volume Response: 0.00025
2026-02-14 10:11:01,427 - v2_logic.controllers.recursive_flow - INFO - [director] SLM PHYSICAL PRIOR: Estimated volume for 'cup' is 0.00025 m^3
2026-02-14 10:11:01,430 - v2_logic.controllers.recursive_flow - INFO - [countvid_executor_node] Counting objects matching intent: ['cup', 'ball', 'person']
2026-02-14 10:11:01,431 - v2_logic.controllers.recursive_flow - INFO - [density_sensing_node] Analyzing physical density
2026-02-14 10:11:01,432 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Generating point cloud for frame 60
2026-02-14 10:11:01,435 - v2_logic.controllers.recursive_flow - INFO - [v2e_sensor_node] Generating events
[DensityPredictor] Warning: Model not fitted. Returning default density 1.0.
2026-02-14 10:11:01,653 - root - INFO - For numpy array image, we assume (HxWxC) format
2026-02-14 10:11:01,767 - root - INFO - Computing image embeddings for the provided image...
2026-02-14 10:11:01,797 - root - INFO - Image embeddings computed.
[SegmentationEngine] Filtered 3 objects (outside ROI/area range)
[SegmentationEngine] Union Masking: merged 1 fragments → 10 physical units
[SegmentationEngine] Clustered 10 masks into 7 volumetric clusters
2026-02-14 10:11:05,160 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Aggregated volume: 102.389193 m^3 (102389193.05 cm^3) from 7 clusters
2026-02-14 10:11:05,166 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] Computing volumetric reconciliation
2026-02-14 10:11:05,166 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] 3DC Result: 192691.62 units (From V_stack: 48.172904 m^3, rho: 1.00, V_unit: 0.000250 m^3)
2026-02-14 10:11:05,167 - v2_logic.controllers.recursive_flow - WARNING - [v3_math_node] Sanity Check Failed: Count 192691.61644588178 exceeds safety buffer 1000. Clipping n_vol.
2026-02-14 10:11:05,168 - v2_logic.controllers.recursive_flow - INFO - [fusion_engine_node] Fusing sensor data & Tracking
2026-02-14 10:11:05,216 - v2_logic.models.fusion_engine_v2 - INFO - [FusionV2] Shields: spatial: ✓ (1.00) | volumetric: ✗ (0.00) | latent: ✓ (0.50) → confidence=0.600 (threshold=0.60)
2026-02-14 10:11:05,220 - v2_logic.controllers.recursive_flow - INFO - [logic_gate_node] Decision: exit (Rule: MaxLoopSafety)
2026-02-14 10:11:05,220 - v2_logic.controllers.recursive_flow - INFO - [route] Checking decision status: exit
2026-02-14 10:11:05,221 - run_recursive_system - INFO - Frame 60 Results:
2026-02-14 10:11:05,221 - run_recursive_system - INFO -   - Count (N_visible): 3
2026-02-14 10:11:05,221 - run_recursive_system - INFO -   - Volumetric Range: (1000, 1000)
2026-02-14 10:11:05,221 - run_recursive_system - INFO -   - Spike Energy: 1181358.00
2026-02-14 10:11:05,221 - run_recursive_system - INFO -   - Anomaly Status: exit
2026-02-14 10:11:05,221 - run_recursive_system - INFO -   - SLM Reasoning: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table. There are three objects visible: the person's hand, the blue cup they are holding, and the two blue cups on the table. However, the 3D volume suggests there should be between 1000 and 1000 objects, which is a significant discrepancy.

The discrepancy likely arises from the following:

1. **Occlusion**: The person's body is partially obscuring the view of the objects on the table. The person's hand and body are in front of the cups and the ball
2026-02-14 10:11:05,285 - run_recursive_system - INFO - --- Processing Frame 75 ---
2026-02-14 10:11:05,293 - v2_logic.controllers.recursive_flow - INFO - [vjepa_brain_node] Encoding latent context
2026-02-14 10:11:05,753 - v2_logic.controllers.recursive_flow - INFO - [director] SLM Hypothesis received: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table
2026-02-14 10:11:05,753 - v2_logic.models.slm_engine - INFO - [SLMEngine] Requesting volume prior for 'cup'
2026-02-14 10:11:06,830 - v2_logic.models.slm_engine - INFO - [SLMEngine] Volume Response: 0.00025
2026-02-14 10:11:06,831 - v2_logic.controllers.recursive_flow - INFO - [director] SLM PHYSICAL PRIOR: Estimated volume for 'cup' is 0.00025 m^3
2026-02-14 10:11:06,836 - v2_logic.controllers.recursive_flow - INFO - [countvid_executor_node] Counting objects matching intent: ['cup', 'ball', 'person']
2026-02-14 10:11:06,838 - v2_logic.controllers.recursive_flow - INFO - [density_sensing_node] Analyzing physical density
2026-02-14 10:11:06,841 - v2_logic.controllers.recursive_flow - INFO - [v2e_sensor_node] Generating events
2026-02-14 10:11:06,839 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Generating point cloud for frame 75
[DensityPredictor] Warning: Model not fitted. Returning default density 1.0.
2026-02-14 10:11:07,243 - root - INFO - For numpy array image, we assume (HxWxC) format
2026-02-14 10:11:07,751 - root - INFO - Computing image embeddings for the provided image...
2026-02-14 10:11:07,814 - root - INFO - Image embeddings computed.
[SegmentationEngine] Filtered 3 objects (outside ROI/area range)
[SegmentationEngine] Union Masking: merged 2 fragments → 9 physical units
[SegmentationEngine] Clustered 9 masks into 7 volumetric clusters
2026-02-14 10:11:10,299 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Aggregated volume: 106.654169 m^3 (106654168.68 cm^3) from 7 clusters
2026-02-14 10:11:10,309 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] Computing volumetric reconciliation
2026-02-14 10:11:10,309 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] 3DC Result: 192691.62 units (From V_stack: 48.172904 m^3, rho: 1.00, V_unit: 0.000250 m^3)
2026-02-14 10:11:10,309 - v2_logic.controllers.recursive_flow - WARNING - [v3_math_node] Sanity Check Failed: Count 192691.61644588178 exceeds safety buffer 1000. Clipping n_vol.
2026-02-14 10:11:10,312 - v2_logic.controllers.recursive_flow - INFO - [fusion_engine_node] Fusing sensor data & Tracking
2026-02-14 10:11:10,376 - v2_logic.models.fusion_engine_v2 - INFO - [FusionV2] Shields: spatial: ✗ (0.52) | volumetric: ✗ (0.00) | latent: ✓ (0.50) → confidence=0.359 (threshold=0.60)
2026-02-14 10:11:10,376 - v2_logic.models.fusion_engine_v2 - WARNING - [FusionV2] LOW CONFIDENCE (0.359) — SLM Audit recommended.
2026-02-14 10:11:10,382 - v2_logic.controllers.recursive_flow - INFO - [logic_gate_node] Decision: exit (Rule: MaxLoopSafety)
2026-02-14 10:11:10,383 - v2_logic.controllers.recursive_flow - INFO - [route] Checking decision status: exit
2026-02-14 10:11:10,384 - run_recursive_system - INFO - Frame 75 Results:
2026-02-14 10:11:10,384 - run_recursive_system - INFO -   - Count (N_visible): 3
2026-02-14 10:11:10,384 - run_recursive_system - INFO -   - Volumetric Range: (1000, 1000)
2026-02-14 10:11:10,384 - run_recursive_system - INFO -   - Spike Energy: 1258133.00
2026-02-14 10:11:10,384 - run_recursive_system - INFO -   - Anomaly Status: exit
2026-02-14 10:11:10,384 - run_recursive_system - INFO -   - SLM Reasoning: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table. There are three objects visible: the person's hand, the blue cup they are holding, and the two blue cups on the table. However, the 3D volume suggests there should be between 1000 and 1000 objects, which is a significant discrepancy.

The discrepancy likely arises from the following:

1. **Occlusion**: The person's body is partially obscuring the view of the objects on the table. The person's hand and body are in front of the cups and the ball
2026-02-14 10:11:10,436 - run_recursive_system - INFO - --- Processing Frame 90 ---
2026-02-14 10:11:10,442 - v2_logic.controllers.recursive_flow - INFO - [vjepa_brain_node] Encoding latent context
2026-02-14 10:11:10,907 - v2_logic.controllers.recursive_flow - INFO - [director] SLM Hypothesis received: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table
2026-02-14 10:11:10,908 - v2_logic.models.slm_engine - INFO - [SLMEngine] Requesting volume prior for 'cup'
2026-02-14 10:11:11,892 - v2_logic.models.slm_engine - INFO - [SLMEngine] Volume Response: 0.00025
2026-02-14 10:11:11,892 - v2_logic.controllers.recursive_flow - INFO - [director] SLM PHYSICAL PRIOR: Estimated volume for 'cup' is 0.00025 m^3
2026-02-14 10:11:11,897 - v2_logic.controllers.recursive_flow - INFO - [countvid_executor_node] Counting objects matching intent: ['cup', 'ball', 'person']
2026-02-14 10:11:11,898 - v2_logic.controllers.recursive_flow - INFO - [density_sensing_node] Analyzing physical density
2026-02-14 10:11:11,900 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Generating point cloud for frame 90
2026-02-14 10:11:11,903 - v2_logic.controllers.recursive_flow - INFO - [v2e_sensor_node] Generating events
[DensityPredictor] Warning: Model not fitted. Returning default density 1.0.
2026-02-14 10:11:12,264 - root - INFO - For numpy array image, we assume (HxWxC) format
2026-02-14 10:11:12,649 - root - INFO - Computing image embeddings for the provided image...
2026-02-14 10:11:12,677 - root - INFO - Image embeddings computed.
[SegmentationEngine] Filtered 3 objects (outside ROI/area range)
[SegmentationEngine] Union Masking: merged 1 fragments → 10 physical units
[SegmentationEngine] Clustered 10 masks into 6 volumetric clusters
2026-02-14 10:11:15,369 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Aggregated volume: 49.698081 m^3 (49698081.03 cm^3) from 6 clusters
2026-02-14 10:11:15,385 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] Computing volumetric reconciliation
2026-02-14 10:11:15,385 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] 3DC Result: 192691.62 units (From V_stack: 48.172904 m^3, rho: 1.00, V_unit: 0.000250 m^3)
2026-02-14 10:11:15,385 - v2_logic.controllers.recursive_flow - WARNING - [v3_math_node] Sanity Check Failed: Count 192691.61644588178 exceeds safety buffer 1000. Clipping n_vol.
2026-02-14 10:11:15,389 - v2_logic.controllers.recursive_flow - INFO - [fusion_engine_node] Fusing sensor data & Tracking
2026-02-14 10:11:15,444 - v2_logic.models.fusion_engine_v2 - INFO - [FusionV2] Shields: spatial: ✗ (0.58) | volumetric: ✗ (0.00) | latent: ✓ (0.50) → confidence=0.390 (threshold=0.60)
2026-02-14 10:11:15,445 - v2_logic.models.fusion_engine_v2 - WARNING - [FusionV2] LOW CONFIDENCE (0.390) — SLM Audit recommended.
2026-02-14 10:11:15,450 - v2_logic.controllers.recursive_flow - INFO - [logic_gate_node] Decision: exit (Rule: MaxLoopSafety)
2026-02-14 10:11:15,451 - v2_logic.controllers.recursive_flow - INFO - [route] Checking decision status: exit
2026-02-14 10:11:15,451 - run_recursive_system - INFO - Frame 90 Results:
2026-02-14 10:11:15,452 - run_recursive_system - INFO -   - Count (N_visible): 3
2026-02-14 10:11:15,452 - run_recursive_system - INFO -   - Volumetric Range: (1000, 1000)
2026-02-14 10:11:15,452 - run_recursive_system - INFO -   - Spike Energy: 1336177.00
2026-02-14 10:11:15,452 - run_recursive_system - INFO -   - Anomaly Status: exit
2026-02-14 10:11:15,452 - run_recursive_system - INFO -   - SLM Reasoning: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table. There are three objects visible: the person's hand, the blue cup they are holding, and the two blue cups on the table. However, the 3D volume suggests there should be between 1000 and 1000 objects, which is a significant discrepancy.

The discrepancy likely arises from the following:

1. **Occlusion**: The person's body is partially obscuring the view of the objects on the table. The person's hand and body are in front of the cups and the ball
2026-02-14 10:11:15,503 - run_recursive_system - INFO - --- Processing Frame 105 ---
2026-02-14 10:11:15,510 - v2_logic.controllers.recursive_flow - INFO - [vjepa_brain_node] Encoding latent context
2026-02-14 10:11:15,978 - v2_logic.controllers.recursive_flow - INFO - [director] SLM Hypothesis received: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table
2026-02-14 10:11:15,978 - v2_logic.models.slm_engine - INFO - [SLMEngine] Requesting volume prior for 'cup'
2026-02-14 10:11:16,947 - v2_logic.models.slm_engine - INFO - [SLMEngine] Volume Response: 0.00025
2026-02-14 10:11:16,947 - v2_logic.controllers.recursive_flow - INFO - [director] SLM PHYSICAL PRIOR: Estimated volume for 'cup' is 0.00025 m^3
2026-02-14 10:11:16,951 - v2_logic.controllers.recursive_flow - INFO - [countvid_executor_node] Counting objects matching intent: ['cup', 'ball', 'person']
2026-02-14 10:11:16,952 - v2_logic.controllers.recursive_flow - INFO - [density_sensing_node] Analyzing physical density
2026-02-14 10:11:16,954 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Generating point cloud for frame 105
2026-02-14 10:11:16,957 - v2_logic.controllers.recursive_flow - INFO - [v2e_sensor_node] Generating events
[DensityPredictor] Warning: Model not fitted. Returning default density 1.0.
2026-02-14 10:11:17,406 - root - INFO - For numpy array image, we assume (HxWxC) format
2026-02-14 10:11:17,568 - root - INFO - Computing image embeddings for the provided image...
2026-02-14 10:11:17,594 - root - INFO - Image embeddings computed.
[SegmentationEngine] Filtered 1 objects (outside ROI/area range)
[SegmentationEngine] Union Masking: merged 1 fragments → 11 physical units
[SegmentationEngine] Clustered 11 masks into 8 volumetric clusters
2026-02-14 10:11:20,800 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Aggregated volume: 27.435731 m^3 (27435731.10 cm^3) from 8 clusters
2026-02-14 10:11:20,818 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] Computing volumetric reconciliation
2026-02-14 10:11:20,818 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] 3DC Result: 192691.62 units (From V_stack: 48.172904 m^3, rho: 1.00, V_unit: 0.000250 m^3)
2026-02-14 10:11:20,818 - v2_logic.controllers.recursive_flow - WARNING - [v3_math_node] Sanity Check Failed: Count 192691.61644588178 exceeds safety buffer 1000. Clipping n_vol.
2026-02-14 10:11:20,823 - v2_logic.controllers.recursive_flow - INFO - [fusion_engine_node] Fusing sensor data & Tracking
2026-02-14 10:11:20,904 - v2_logic.models.fusion_engine_v2 - INFO - [FusionV2] Shields: spatial: ✗ (0.60) | volumetric: ✗ (0.00) | latent: ✓ (0.50) → confidence=0.399 (threshold=0.60)
2026-02-14 10:11:20,904 - v2_logic.models.fusion_engine_v2 - WARNING - [FusionV2] LOW CONFIDENCE (0.399) — SLM Audit recommended.
2026-02-14 10:11:20,915 - v2_logic.controllers.recursive_flow - INFO - [logic_gate_node] Decision: exit (Rule: MaxLoopSafety)
2026-02-14 10:11:20,915 - v2_logic.controllers.recursive_flow - INFO - [route] Checking decision status: exit
2026-02-14 10:11:20,916 - run_recursive_system - INFO - Frame 105 Results:
2026-02-14 10:11:20,917 - run_recursive_system - INFO -   - Count (N_visible): 3
2026-02-14 10:11:20,917 - run_recursive_system - INFO -   - Volumetric Range: (1000, 1000)
2026-02-14 10:11:20,917 - run_recursive_system - INFO -   - Spike Energy: 1083391.00
2026-02-14 10:11:20,917 - run_recursive_system - INFO -   - Anomaly Status: exit
2026-02-14 10:11:20,917 - run_recursive_system - INFO -   - SLM Reasoning: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table. There are three objects visible: the person's hand, the blue cup they are holding, and the two blue cups on the table. However, the 3D volume suggests there should be between 1000 and 1000 objects, which is a significant discrepancy.

The discrepancy likely arises from the following:

1. **Occlusion**: The person's body is partially obscuring the view of the objects on the table. The person's hand and body are in front of the cups and the ball
2026-02-14 10:11:20,986 - run_recursive_system - INFO - --- Processing Frame 120 ---
2026-02-14 10:11:20,993 - v2_logic.controllers.recursive_flow - INFO - [vjepa_brain_node] Encoding latent context
2026-02-14 10:11:21,468 - v2_logic.controllers.recursive_flow - INFO - [director] SLM Hypothesis received: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table
2026-02-14 10:11:21,468 - v2_logic.models.slm_engine - INFO - [SLMEngine] Requesting volume prior for 'cup'
2026-02-14 10:11:22,446 - v2_logic.models.slm_engine - INFO - [SLMEngine] Volume Response: 0.00025
2026-02-14 10:11:22,446 - v2_logic.controllers.recursive_flow - INFO - [director] SLM PHYSICAL PRIOR: Estimated volume for 'cup' is 0.00025 m^3
2026-02-14 10:11:22,452 - v2_logic.controllers.recursive_flow - INFO - [countvid_executor_node] Counting objects matching intent: ['cup', 'ball', 'person']
2026-02-14 10:11:22,454 - v2_logic.controllers.recursive_flow - INFO - [density_sensing_node] Analyzing physical density
2026-02-14 10:11:22,455 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Generating point cloud for frame 120
2026-02-14 10:11:22,457 - v2_logic.controllers.recursive_flow - INFO - [v2e_sensor_node] Generating events
[DensityPredictor] Warning: Model not fitted. Returning default density 1.0.
2026-02-14 10:11:22,685 - root - INFO - For numpy array image, we assume (HxWxC) format
2026-02-14 10:11:22,871 - root - INFO - Computing image embeddings for the provided image...
2026-02-14 10:11:22,906 - root - INFO - Image embeddings computed.
[SegmentationEngine] Filtered 2 objects (outside ROI/area range)
[SegmentationEngine] Union Masking: merged 1 fragments → 11 physical units
[SegmentationEngine] Clustered 11 masks into 5 volumetric clusters
2026-02-14 10:11:25,865 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Aggregated volume: 81.844367 m^3 (81844367.25 cm^3) from 5 clusters
2026-02-14 10:11:25,882 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] Computing volumetric reconciliation
2026-02-14 10:11:25,883 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] 3DC Result: 192691.62 units (From V_stack: 48.172904 m^3, rho: 1.00, V_unit: 0.000250 m^3)
2026-02-14 10:11:25,883 - v2_logic.controllers.recursive_flow - WARNING - [v3_math_node] Sanity Check Failed: Count 192691.61644588178 exceeds safety buffer 1000. Clipping n_vol.
2026-02-14 10:11:25,888 - v2_logic.controllers.recursive_flow - INFO - [fusion_engine_node] Fusing sensor data & Tracking
2026-02-14 10:11:25,947 - v2_logic.models.fusion_engine_v2 - INFO - [FusionV2] Shields: spatial: ✗ (0.55) | volumetric: ✗ (0.00) | latent: ✓ (0.50) → confidence=0.377 (threshold=0.60)
2026-02-14 10:11:25,948 - v2_logic.models.fusion_engine_v2 - WARNING - [FusionV2] LOW CONFIDENCE (0.377) — SLM Audit recommended.
2026-02-14 10:11:25,952 - v2_logic.controllers.recursive_flow - INFO - [logic_gate_node] Decision: exit (Rule: MaxLoopSafety)
2026-02-14 10:11:25,953 - v2_logic.controllers.recursive_flow - INFO - [route] Checking decision status: exit
2026-02-14 10:11:25,953 - run_recursive_system - INFO - Frame 120 Results:
2026-02-14 10:11:25,954 - run_recursive_system - INFO -   - Count (N_visible): 3
2026-02-14 10:11:25,954 - run_recursive_system - INFO -   - Volumetric Range: (1000, 1000)
2026-02-14 10:11:25,954 - run_recursive_system - INFO -   - Spike Energy: 932879.00
2026-02-14 10:11:25,954 - run_recursive_system - INFO -   - Anomaly Status: exit
2026-02-14 10:11:25,954 - run_recursive_system - INFO -   - SLM Reasoning: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table. There are three objects visible: the person's hand, the blue cup they are holding, and the two blue cups on the table. However, the 3D volume suggests there should be between 1000 and 1000 objects, which is a significant discrepancy.

The discrepancy likely arises from the following:

1. **Occlusion**: The person's body is partially obscuring the view of the objects on the table. The person's hand and body are in front of the cups and the ball
2026-02-14 10:11:25,999 - run_recursive_system - INFO - --- Processing Frame 135 ---
2026-02-14 10:11:26,004 - v2_logic.controllers.recursive_flow - INFO - [vjepa_brain_node] Encoding latent context
2026-02-14 10:11:26,458 - v2_logic.controllers.recursive_flow - INFO - [director] SLM Hypothesis received: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table
2026-02-14 10:11:26,458 - v2_logic.models.slm_engine - INFO - [SLMEngine] Requesting volume prior for 'cup'
2026-02-14 10:11:27,431 - v2_logic.models.slm_engine - INFO - [SLMEngine] Volume Response: 0.00025
2026-02-14 10:11:27,431 - v2_logic.controllers.recursive_flow - INFO - [director] SLM PHYSICAL PRIOR: Estimated volume for 'cup' is 0.00025 m^3
2026-02-14 10:11:27,436 - v2_logic.controllers.recursive_flow - INFO - [countvid_executor_node] Counting objects matching intent: ['cup', 'ball', 'person']
2026-02-14 10:11:27,437 - v2_logic.controllers.recursive_flow - INFO - [density_sensing_node] Analyzing physical density
2026-02-14 10:11:27,439 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Generating point cloud for frame 135
2026-02-14 10:11:27,442 - v2_logic.controllers.recursive_flow - INFO - [v2e_sensor_node] Generating events
[DensityPredictor] Warning: Model not fitted. Returning default density 1.0.
2026-02-14 10:11:27,700 - root - INFO - For numpy array image, we assume (HxWxC) format
2026-02-14 10:11:28,163 - root - INFO - Computing image embeddings for the provided image...
2026-02-14 10:11:28,228 - root - INFO - Image embeddings computed.
[SegmentationEngine] Filtered 2 objects (outside ROI/area range)
[SegmentationEngine] Clustered 11 masks into 6 volumetric clusters
2026-02-14 10:11:30,765 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Aggregated volume: 69.523882 m^3 (69523881.57 cm^3) from 6 clusters
2026-02-14 10:11:30,778 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] Computing volumetric reconciliation
2026-02-14 10:11:30,778 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] 3DC Result: 192691.62 units (From V_stack: 48.172904 m^3, rho: 1.00, V_unit: 0.000250 m^3)
2026-02-14 10:11:30,779 - v2_logic.controllers.recursive_flow - WARNING - [v3_math_node] Sanity Check Failed: Count 192691.61644588178 exceeds safety buffer 1000. Clipping n_vol.
2026-02-14 10:11:30,782 - v2_logic.controllers.recursive_flow - INFO - [fusion_engine_node] Fusing sensor data & Tracking
2026-02-14 10:11:30,827 - v2_logic.models.fusion_engine_v2 - INFO - [FusionV2] Shields: spatial: ✗ (0.56) | volumetric: ✗ (0.00) | latent: ✓ (0.50) → confidence=0.382 (threshold=0.60)
2026-02-14 10:11:30,827 - v2_logic.models.fusion_engine_v2 - WARNING - [FusionV2] LOW CONFIDENCE (0.382) — SLM Audit recommended.
2026-02-14 10:11:30,832 - v2_logic.controllers.recursive_flow - INFO - [logic_gate_node] Decision: exit (Rule: MaxLoopSafety)
2026-02-14 10:11:30,832 - v2_logic.controllers.recursive_flow - INFO - [route] Checking decision status: exit
2026-02-14 10:11:30,833 - run_recursive_system - INFO - Frame 135 Results:
2026-02-14 10:11:30,833 - run_recursive_system - INFO -   - Count (N_visible): 3
2026-02-14 10:11:30,834 - run_recursive_system - INFO -   - Volumetric Range: (1000, 1000)
2026-02-14 10:11:30,834 - run_recursive_system - INFO -   - Spike Energy: 679750.00
2026-02-14 10:11:30,834 - run_recursive_system - INFO -   - Anomaly Status: exit
2026-02-14 10:11:30,834 - run_recursive_system - INFO -   - SLM Reasoning: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table. There are three objects visible: the person's hand, the blue cup they are holding, and the two blue cups on the table. However, the 3D volume suggests there should be between 1000 and 1000 objects, which is a significant discrepancy.

The discrepancy likely arises from the following:

1. **Occlusion**: The person's body is partially obscuring the view of the objects on the table. The person's hand and body are in front of the cups and the ball
2026-02-14 10:11:30,880 - run_recursive_system - INFO - --- Processing Frame 150 ---
2026-02-14 10:11:31,415 - v2_logic.controllers.recursive_flow - INFO - [vjepa_brain_node] Encoding latent context
2026-02-14 10:11:31,863 - v2_logic.controllers.recursive_flow - INFO - [director] SLM Hypothesis received: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table
2026-02-14 10:11:31,864 - v2_logic.models.slm_engine - INFO - [SLMEngine] Requesting volume prior for 'cup'
2026-02-14 10:11:33,086 - v2_logic.models.slm_engine - INFO - [SLMEngine] Volume Response: 0.00025
2026-02-14 10:11:33,087 - v2_logic.controllers.recursive_flow - INFO - [director] SLM PHYSICAL PRIOR: Estimated volume for 'cup' is 0.00025 m^3
2026-02-14 10:11:33,091 - v2_logic.controllers.recursive_flow - INFO - [countvid_executor_node] Counting objects matching intent: ['cup', 'ball', 'person']
2026-02-14 10:11:33,092 - v2_logic.controllers.recursive_flow - INFO - [density_sensing_node] Analyzing physical density
2026-02-14 10:11:33,094 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Generating point cloud for frame 150
2026-02-14 10:11:33,099 - v2_logic.controllers.recursive_flow - INFO - [v2e_sensor_node] Generating events
[DensityPredictor] Warning: Model not fitted. Returning default density 1.0.
2026-02-14 10:11:33,315 - root - INFO - For numpy array image, we assume (HxWxC) format
2026-02-14 10:11:33,503 - root - INFO - Computing image embeddings for the provided image...
2026-02-14 10:11:33,539 - root - INFO - Image embeddings computed.
[SegmentationEngine] Filtered 1 objects (outside ROI/area range)
[SegmentationEngine] Union Masking: merged 1 fragments → 12 physical units
[SegmentationEngine] Clustered 12 masks into 9 volumetric clusters
2026-02-14 10:11:36,457 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Aggregated volume: 73.376556 m^3 (73376556.07 cm^3) from 9 clusters
2026-02-14 10:11:36,466 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] Computing volumetric reconciliation
2026-02-14 10:11:36,467 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] 3DC Result: 192691.62 units (From V_stack: 48.172904 m^3, rho: 1.00, V_unit: 0.000250 m^3)
2026-02-14 10:11:36,467 - v2_logic.controllers.recursive_flow - WARNING - [v3_math_node] Sanity Check Failed: Count 192691.61644588178 exceeds safety buffer 1000. Clipping n_vol.
2026-02-14 10:11:36,469 - v2_logic.controllers.recursive_flow - INFO - [fusion_engine_node] Fusing sensor data & Tracking
2026-02-14 10:11:36,529 - v2_logic.models.fusion_engine_v2 - INFO - [FusionV2] Shields: spatial: ✗ (0.67) | volumetric: ✗ (0.00) | latent: ✓ (0.50) → confidence=0.436 (threshold=0.60)
2026-02-14 10:11:36,529 - v2_logic.models.fusion_engine_v2 - WARNING - [FusionV2] LOW CONFIDENCE (0.436) — SLM Audit recommended.
2026-02-14 10:11:36,533 - v2_logic.controllers.recursive_flow - INFO - [logic_gate_node] Decision: exit (Rule: MaxLoopSafety)
2026-02-14 10:11:36,533 - v2_logic.controllers.recursive_flow - INFO - [route] Checking decision status: exit
2026-02-14 10:11:36,534 - run_recursive_system - INFO - Frame 150 Results:
2026-02-14 10:11:36,534 - run_recursive_system - INFO -   - Count (N_visible): 3
2026-02-14 10:11:36,534 - run_recursive_system - INFO -   - Volumetric Range: (1000, 1000)
2026-02-14 10:11:36,535 - run_recursive_system - INFO -   - Spike Energy: 863975.00
2026-02-14 10:11:36,535 - run_recursive_system - INFO -   - Anomaly Status: exit
2026-02-14 10:11:36,535 - run_recursive_system - INFO -   - SLM Reasoning: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table. There are three objects visible: the person's hand, the blue cup they are holding, and the two blue cups on the table. However, the 3D volume suggests there should be between 1000 and 1000 objects, which is a significant discrepancy.

The discrepancy likely arises from the following:

1. **Occlusion**: The person's body is partially obscuring the view of the objects on the table. The person's hand and body are in front of the cups and the ball
2026-02-14 10:11:36,605 - run_recursive_system - INFO - --- Processing Frame 165 ---
2026-02-14 10:11:36,610 - v2_logic.controllers.recursive_flow - INFO - [vjepa_brain_node] Encoding latent context
2026-02-14 10:11:37,045 - v2_logic.controllers.recursive_flow - INFO - [director] SLM Hypothesis received: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table
2026-02-14 10:11:37,046 - v2_logic.models.slm_engine - INFO - [SLMEngine] Requesting volume prior for 'cup'
2026-02-14 10:11:38,004 - v2_logic.models.slm_engine - INFO - [SLMEngine] Volume Response: 0.00025
2026-02-14 10:11:38,004 - v2_logic.controllers.recursive_flow - INFO - [director] SLM PHYSICAL PRIOR: Estimated volume for 'cup' is 0.00025 m^3
2026-02-14 10:11:38,009 - v2_logic.controllers.recursive_flow - INFO - [countvid_executor_node] Counting objects matching intent: ['cup', 'ball', 'person']
2026-02-14 10:11:38,011 - v2_logic.controllers.recursive_flow - INFO - [density_sensing_node] Analyzing physical density
2026-02-14 10:11:38,014 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Generating point cloud for frame 165
2026-02-14 10:11:38,017 - v2_logic.controllers.recursive_flow - INFO - [v2e_sensor_node] Generating events
[DensityPredictor] Warning: Model not fitted. Returning default density 1.0.
2026-02-14 10:11:38,367 - root - INFO - For numpy array image, we assume (HxWxC) format
2026-02-14 10:11:38,792 - root - INFO - Computing image embeddings for the provided image...
2026-02-14 10:11:38,829 - root - INFO - Image embeddings computed.
[SegmentationEngine] Filtered 3 objects (outside ROI/area range)
[SegmentationEngine] Union Masking: merged 1 fragments → 11 physical units
[SegmentationEngine] Clustered 11 masks into 7 volumetric clusters
2026-02-14 10:11:41,365 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Aggregated volume: 52.337074 m^3 (52337074.49 cm^3) from 7 clusters
2026-02-14 10:11:41,380 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] Computing volumetric reconciliation
2026-02-14 10:11:41,380 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] 3DC Result: 192691.62 units (From V_stack: 48.172904 m^3, rho: 1.00, V_unit: 0.000250 m^3)
2026-02-14 10:11:41,380 - v2_logic.controllers.recursive_flow - WARNING - [v3_math_node] Sanity Check Failed: Count 192691.61644588178 exceeds safety buffer 1000. Clipping n_vol.
2026-02-14 10:11:41,384 - v2_logic.controllers.recursive_flow - INFO - [fusion_engine_node] Fusing sensor data & Tracking
2026-02-14 10:11:41,445 - v2_logic.models.fusion_engine_v2 - INFO - [FusionV2] Shields: spatial: ✗ (0.49) | volumetric: ✗ (0.00) | latent: ✓ (0.50) → confidence=0.347 (threshold=0.60)
2026-02-14 10:11:41,445 - v2_logic.models.fusion_engine_v2 - WARNING - [FusionV2] LOW CONFIDENCE (0.347) — SLM Audit recommended.
2026-02-14 10:11:41,450 - v2_logic.controllers.recursive_flow - INFO - [logic_gate_node] Decision: exit (Rule: MaxLoopSafety)
2026-02-14 10:11:41,450 - v2_logic.controllers.recursive_flow - INFO - [route] Checking decision status: exit
2026-02-14 10:11:41,453 - run_recursive_system - INFO - Frame 165 Results:
2026-02-14 10:11:41,454 - run_recursive_system - INFO -   - Count (N_visible): 3
2026-02-14 10:11:41,454 - run_recursive_system - INFO -   - Volumetric Range: (1000, 1000)
2026-02-14 10:11:41,454 - run_recursive_system - INFO -   - Spike Energy: 893920.00
2026-02-14 10:11:41,454 - run_recursive_system - INFO -   - Anomaly Status: exit
2026-02-14 10:11:41,454 - run_recursive_system - INFO -   - SLM Reasoning: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table. There are three objects visible: the person's hand, the blue cup they are holding, and the two blue cups on the table. However, the 3D volume suggests there should be between 1000 and 1000 objects, which is a significant discrepancy.

The discrepancy likely arises from the following:

1. **Occlusion**: The person's body is partially obscuring the view of the objects on the table. The person's hand and body are in front of the cups and the ball
2026-02-14 10:11:41,505 - run_recursive_system - INFO - --- Processing Frame 180 ---
2026-02-14 10:11:41,511 - v2_logic.controllers.recursive_flow - INFO - [vjepa_brain_node] Encoding latent context
2026-02-14 10:11:41,947 - v2_logic.controllers.recursive_flow - INFO - [director] SLM Hypothesis received: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table
2026-02-14 10:11:41,948 - v2_logic.models.slm_engine - INFO - [SLMEngine] Requesting volume prior for 'cup'
2026-02-14 10:11:42,932 - v2_logic.models.slm_engine - INFO - [SLMEngine] Volume Response: 0.00025
2026-02-14 10:11:42,932 - v2_logic.controllers.recursive_flow - INFO - [director] SLM PHYSICAL PRIOR: Estimated volume for 'cup' is 0.00025 m^3
2026-02-14 10:11:42,937 - v2_logic.controllers.recursive_flow - INFO - [countvid_executor_node] Counting objects matching intent: ['cup', 'ball', 'person']
2026-02-14 10:11:42,938 - v2_logic.controllers.recursive_flow - INFO - [density_sensing_node] Analyzing physical density
2026-02-14 10:11:42,941 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Generating point cloud for frame 180
2026-02-14 10:11:42,946 - v2_logic.controllers.recursive_flow - INFO - [v2e_sensor_node] Generating events
[DensityPredictor] Warning: Model not fitted. Returning default density 1.0.
2026-02-14 10:11:43,282 - root - INFO - For numpy array image, we assume (HxWxC) format
2026-02-14 10:11:43,786 - root - INFO - Computing image embeddings for the provided image...
2026-02-14 10:11:43,863 - root - INFO - Image embeddings computed.
[SegmentationEngine] Filtered 3 objects (outside ROI/area range)
[SegmentationEngine] Union Masking: merged 1 fragments → 10 physical units
[SegmentationEngine] Clustered 10 masks into 6 volumetric clusters
2026-02-14 10:11:46,390 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Aggregated volume: 95.546347 m^3 (95546346.65 cm^3) from 6 clusters
2026-02-14 10:11:46,405 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] Computing volumetric reconciliation
2026-02-14 10:11:46,405 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] 3DC Result: 192691.62 units (From V_stack: 48.172904 m^3, rho: 1.00, V_unit: 0.000250 m^3)
2026-02-14 10:11:46,405 - v2_logic.controllers.recursive_flow - WARNING - [v3_math_node] Sanity Check Failed: Count 192691.61644588178 exceeds safety buffer 1000. Clipping n_vol.
2026-02-14 10:11:46,409 - v2_logic.controllers.recursive_flow - INFO - [fusion_engine_node] Fusing sensor data & Tracking
2026-02-14 10:11:46,464 - v2_logic.models.fusion_engine_v2 - INFO - [FusionV2] Shields: spatial: ✗ (0.44) | volumetric: ✗ (0.00) | latent: ✓ (0.50) → confidence=0.319 (threshold=0.60)
2026-02-14 10:11:46,465 - v2_logic.models.fusion_engine_v2 - WARNING - [FusionV2] LOW CONFIDENCE (0.319) — SLM Audit recommended.
2026-02-14 10:11:46,469 - v2_logic.controllers.recursive_flow - INFO - [logic_gate_node] Decision: exit (Rule: MaxLoopSafety)
2026-02-14 10:11:46,469 - v2_logic.controllers.recursive_flow - INFO - [route] Checking decision status: exit
2026-02-14 10:11:46,472 - run_recursive_system - INFO - Frame 180 Results:
2026-02-14 10:11:46,472 - run_recursive_system - INFO -   - Count (N_visible): 3
2026-02-14 10:11:46,472 - run_recursive_system - INFO -   - Volumetric Range: (1000, 1000)
2026-02-14 10:11:46,472 - run_recursive_system - INFO -   - Spike Energy: 977200.00
2026-02-14 10:11:46,473 - run_recursive_system - INFO -   - Anomaly Status: exit
2026-02-14 10:11:46,473 - run_recursive_system - INFO -   - SLM Reasoning: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table. There are three objects visible: the person's hand, the blue cup they are holding, and the two blue cups on the table. However, the 3D volume suggests there should be between 1000 and 1000 objects, which is a significant discrepancy.

The discrepancy likely arises from the following:

1. **Occlusion**: The person's body is partially obscuring the view of the objects on the table. The person's hand and body are in front of the cups and the ball
2026-02-14 10:11:46,529 - run_recursive_system - INFO - --- Processing Frame 195 ---
2026-02-14 10:11:46,534 - v2_logic.controllers.recursive_flow - INFO - [vjepa_brain_node] Encoding latent context
2026-02-14 10:11:46,963 - v2_logic.controllers.recursive_flow - INFO - [director] SLM Hypothesis received: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table
2026-02-14 10:11:46,963 - v2_logic.models.slm_engine - INFO - [SLMEngine] Requesting volume prior for 'cup'
2026-02-14 10:11:47,918 - v2_logic.models.slm_engine - INFO - [SLMEngine] Volume Response: 0.00025
2026-02-14 10:11:47,919 - v2_logic.controllers.recursive_flow - INFO - [director] SLM PHYSICAL PRIOR: Estimated volume for 'cup' is 0.00025 m^3
2026-02-14 10:11:47,923 - v2_logic.controllers.recursive_flow - INFO - [countvid_executor_node] Counting objects matching intent: ['cup', 'ball', 'person']
2026-02-14 10:11:47,924 - v2_logic.controllers.recursive_flow - INFO - [density_sensing_node] Analyzing physical density
2026-02-14 10:11:47,927 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Generating point cloud for frame 195
2026-02-14 10:11:47,931 - v2_logic.controllers.recursive_flow - INFO - [v2e_sensor_node] Generating events
[DensityPredictor] Warning: Model not fitted. Returning default density 1.0.
2026-02-14 10:11:48,207 - root - INFO - For numpy array image, we assume (HxWxC) format
2026-02-14 10:11:48,399 - root - INFO - Computing image embeddings for the provided image...
2026-02-14 10:11:48,475 - root - INFO - Image embeddings computed.
[SegmentationEngine] Filtered 3 objects (outside ROI/area range)
[SegmentationEngine] Clustered 10 masks into 8 volumetric clusters
2026-02-14 10:11:51,243 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Aggregated volume: 101.151433 m^3 (101151433.41 cm^3) from 8 clusters
2026-02-14 10:11:51,255 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] Computing volumetric reconciliation
2026-02-14 10:11:51,255 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] 3DC Result: 192691.62 units (From V_stack: 48.172904 m^3, rho: 1.00, V_unit: 0.000250 m^3)
2026-02-14 10:11:51,255 - v2_logic.controllers.recursive_flow - WARNING - [v3_math_node] Sanity Check Failed: Count 192691.61644588178 exceeds safety buffer 1000. Clipping n_vol.
2026-02-14 10:11:51,258 - v2_logic.controllers.recursive_flow - INFO - [fusion_engine_node] Fusing sensor data & Tracking
2026-02-14 10:11:51,310 - v2_logic.models.fusion_engine_v2 - INFO - [FusionV2] Shields: spatial: ✗ (0.39) | volumetric: ✗ (0.00) | latent: ✓ (0.50) → confidence=0.297 (threshold=0.60)
2026-02-14 10:11:51,311 - v2_logic.models.fusion_engine_v2 - WARNING - [FusionV2] LOW CONFIDENCE (0.297) — SLM Audit recommended.
2026-02-14 10:11:51,314 - v2_logic.controllers.recursive_flow - INFO - [logic_gate_node] Decision: exit (Rule: MaxLoopSafety)
2026-02-14 10:11:51,314 - v2_logic.controllers.recursive_flow - INFO - [route] Checking decision status: exit
2026-02-14 10:11:51,317 - run_recursive_system - INFO - Frame 195 Results:
2026-02-14 10:11:51,317 - run_recursive_system - INFO -   - Count (N_visible): 3
2026-02-14 10:11:51,317 - run_recursive_system - INFO -   - Volumetric Range: (1000, 1000)
2026-02-14 10:11:51,317 - run_recursive_system - INFO -   - Spike Energy: 943627.00
2026-02-14 10:11:51,317 - run_recursive_system - INFO -   - Anomaly Status: exit
2026-02-14 10:11:51,317 - run_recursive_system - INFO -   - SLM Reasoning: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table. There are three objects visible: the person's hand, the blue cup they are holding, and the two blue cups on the table. However, the 3D volume suggests there should be between 1000 and 1000 objects, which is a significant discrepancy.

The discrepancy likely arises from the following:

1. **Occlusion**: The person's body is partially obscuring the view of the objects on the table. The person's hand and body are in front of the cups and the ball
2026-02-14 10:11:51,364 - run_recursive_system - INFO - --- Processing Frame 210 ---
2026-02-14 10:11:51,369 - v2_logic.controllers.recursive_flow - INFO - [vjepa_brain_node] Encoding latent context
2026-02-14 10:11:51,801 - v2_logic.controllers.recursive_flow - INFO - [director] SLM Hypothesis received: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table
2026-02-14 10:11:51,801 - v2_logic.models.slm_engine - INFO - [SLMEngine] Requesting volume prior for 'cup'
2026-02-14 10:11:52,772 - v2_logic.models.slm_engine - INFO - [SLMEngine] Volume Response: 0.00025
2026-02-14 10:11:52,772 - v2_logic.controllers.recursive_flow - INFO - [director] SLM PHYSICAL PRIOR: Estimated volume for 'cup' is 0.00025 m^3
2026-02-14 10:11:52,776 - v2_logic.controllers.recursive_flow - INFO - [countvid_executor_node] Counting objects matching intent: ['cup', 'ball', 'person']
2026-02-14 10:11:52,777 - v2_logic.controllers.recursive_flow - INFO - [density_sensing_node] Analyzing physical density
2026-02-14 10:11:52,780 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Generating point cloud for frame 210
2026-02-14 10:11:52,783 - v2_logic.controllers.recursive_flow - INFO - [v2e_sensor_node] Generating events
[DensityPredictor] Warning: Model not fitted. Returning default density 1.0.
2026-02-14 10:11:53,165 - root - INFO - For numpy array image, we assume (HxWxC) format
2026-02-14 10:11:53,317 - root - INFO - Computing image embeddings for the provided image...
2026-02-14 10:11:53,345 - root - INFO - Image embeddings computed.
[SegmentationEngine] Filtered 3 objects (outside ROI/area range)
[SegmentationEngine] Union Masking: merged 1 fragments → 10 physical units
[SegmentationEngine] Clustered 10 masks into 7 volumetric clusters
2026-02-14 10:11:56,435 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Aggregated volume: 124.477226 m^3 (124477226.42 cm^3) from 7 clusters
2026-02-14 10:11:56,454 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] Computing volumetric reconciliation
2026-02-14 10:11:56,454 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] 3DC Result: 192691.62 units (From V_stack: 48.172904 m^3, rho: 1.00, V_unit: 0.000250 m^3)
2026-02-14 10:11:56,454 - v2_logic.controllers.recursive_flow - WARNING - [v3_math_node] Sanity Check Failed: Count 192691.61644588178 exceeds safety buffer 1000. Clipping n_vol.
2026-02-14 10:11:56,459 - v2_logic.controllers.recursive_flow - INFO - [fusion_engine_node] Fusing sensor data & Tracking
2026-02-14 10:11:56,543 - v2_logic.models.fusion_engine_v2 - INFO - [FusionV2] Shields: spatial: ✗ (0.67) | volumetric: ✗ (0.00) | latent: ✓ (0.50) → confidence=0.435 (threshold=0.60)
2026-02-14 10:11:56,543 - v2_logic.models.fusion_engine_v2 - WARNING - [FusionV2] LOW CONFIDENCE (0.435) — SLM Audit recommended.
2026-02-14 10:11:56,549 - v2_logic.controllers.recursive_flow - INFO - [logic_gate_node] Decision: exit (Rule: MaxLoopSafety)
2026-02-14 10:11:56,550 - v2_logic.controllers.recursive_flow - INFO - [route] Checking decision status: exit
2026-02-14 10:11:56,550 - run_recursive_system - INFO - Frame 210 Results:
2026-02-14 10:11:56,551 - run_recursive_system - INFO -   - Count (N_visible): 3
2026-02-14 10:11:56,551 - run_recursive_system - INFO -   - Volumetric Range: (1000, 1000)
2026-02-14 10:11:56,551 - run_recursive_system - INFO -   - Spike Energy: 1193234.00
2026-02-14 10:11:56,551 - run_recursive_system - INFO -   - Anomaly Status: exit
2026-02-14 10:11:56,551 - run_recursive_system - INFO -   - SLM Reasoning: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table. There are three objects visible: the person's hand, the blue cup they are holding, and the two blue cups on the table. However, the 3D volume suggests there should be between 1000 and 1000 objects, which is a significant discrepancy.

The discrepancy likely arises from the following:

1. **Occlusion**: The person's body is partially obscuring the view of the objects on the table. The person's hand and body are in front of the cups and the ball
2026-02-14 10:11:56,627 - run_recursive_system - INFO - --- Processing Frame 225 ---
2026-02-14 10:11:56,633 - v2_logic.controllers.recursive_flow - INFO - [vjepa_brain_node] Encoding latent context
2026-02-14 10:11:57,064 - v2_logic.controllers.recursive_flow - INFO - [director] SLM Hypothesis received: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table
2026-02-14 10:11:57,065 - v2_logic.models.slm_engine - INFO - [SLMEngine] Requesting volume prior for 'cup'
2026-02-14 10:11:58,032 - v2_logic.models.slm_engine - INFO - [SLMEngine] Volume Response: 0.00025
2026-02-14 10:11:58,032 - v2_logic.controllers.recursive_flow - INFO - [director] SLM PHYSICAL PRIOR: Estimated volume for 'cup' is 0.00025 m^3
2026-02-14 10:11:58,037 - v2_logic.controllers.recursive_flow - INFO - [countvid_executor_node] Counting objects matching intent: ['cup', 'ball', 'person']
2026-02-14 10:11:58,038 - v2_logic.controllers.recursive_flow - INFO - [density_sensing_node] Analyzing physical density
2026-02-14 10:11:58,040 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Generating point cloud for frame 225
2026-02-14 10:11:58,043 - v2_logic.controllers.recursive_flow - INFO - [v2e_sensor_node] Generating events
[DensityPredictor] Warning: Model not fitted. Returning default density 1.0.
2026-02-14 10:11:58,275 - root - INFO - For numpy array image, we assume (HxWxC) format
2026-02-14 10:11:58,465 - root - INFO - Computing image embeddings for the provided image...
2026-02-14 10:11:58,521 - root - INFO - Image embeddings computed.
[SegmentationEngine] Filtered 2 objects (outside ROI/area range)
[SegmentationEngine] Clustered 10 masks into 8 volumetric clusters
2026-02-14 10:12:01,678 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Aggregated volume: 41.652402 m^3 (41652402.00 cm^3) from 8 clusters
2026-02-14 10:12:01,691 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] Computing volumetric reconciliation
2026-02-14 10:12:01,691 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] 3DC Result: 192691.62 units (From V_stack: 48.172904 m^3, rho: 1.00, V_unit: 0.000250 m^3)
2026-02-14 10:12:01,691 - v2_logic.controllers.recursive_flow - WARNING - [v3_math_node] Sanity Check Failed: Count 192691.61644588178 exceeds safety buffer 1000. Clipping n_vol.
2026-02-14 10:12:01,696 - v2_logic.controllers.recursive_flow - INFO - [fusion_engine_node] Fusing sensor data & Tracking
2026-02-14 10:12:01,749 - v2_logic.models.fusion_engine_v2 - INFO - [FusionV2] Shields: spatial: ✗ (0.78) | volumetric: ✗ (0.00) | latent: ✓ (0.50) → confidence=0.489 (threshold=0.60)
2026-02-14 10:12:01,749 - v2_logic.models.fusion_engine_v2 - WARNING - [FusionV2] LOW CONFIDENCE (0.489) — SLM Audit recommended.
2026-02-14 10:12:01,753 - v2_logic.controllers.recursive_flow - INFO - [logic_gate_node] Decision: exit (Rule: MaxLoopSafety)
2026-02-14 10:12:01,753 - v2_logic.controllers.recursive_flow - INFO - [route] Checking decision status: exit
2026-02-14 10:12:01,754 - run_recursive_system - INFO - Frame 225 Results:
2026-02-14 10:12:01,754 - run_recursive_system - INFO -   - Count (N_visible): 3
2026-02-14 10:12:01,754 - run_recursive_system - INFO -   - Volumetric Range: (1000, 1000)
2026-02-14 10:12:01,754 - run_recursive_system - INFO -   - Spike Energy: 854100.00
2026-02-14 10:12:01,754 - run_recursive_system - INFO -   - Anomaly Status: exit
2026-02-14 10:12:01,754 - run_recursive_system - INFO -   - SLM Reasoning: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table. There are three objects visible: the person's hand, the blue cup they are holding, and the two blue cups on the table. However, the 3D volume suggests there should be between 1000 and 1000 objects, which is a significant discrepancy.

The discrepancy likely arises from the following:

1. **Occlusion**: The person's body is partially obscuring the view of the objects on the table. The person's hand and body are in front of the cups and the ball
2026-02-14 10:12:01,800 - run_recursive_system - INFO - --- Processing Frame 240 ---
2026-02-14 10:12:01,806 - v2_logic.controllers.recursive_flow - INFO - [vjepa_brain_node] Encoding latent context
2026-02-14 10:12:02,229 - v2_logic.controllers.recursive_flow - INFO - [director] SLM Hypothesis received: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table
2026-02-14 10:12:02,230 - v2_logic.models.slm_engine - INFO - [SLMEngine] Requesting volume prior for 'cup'
2026-02-14 10:12:03,178 - v2_logic.models.slm_engine - INFO - [SLMEngine] Volume Response: 0.00025
2026-02-14 10:12:03,178 - v2_logic.controllers.recursive_flow - INFO - [director] SLM PHYSICAL PRIOR: Estimated volume for 'cup' is 0.00025 m^3
2026-02-14 10:12:03,182 - v2_logic.controllers.recursive_flow - INFO - [countvid_executor_node] Counting objects matching intent: ['cup', 'ball', 'person']
2026-02-14 10:12:03,183 - v2_logic.controllers.recursive_flow - INFO - [density_sensing_node] Analyzing physical density
2026-02-14 10:12:03,185 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Generating point cloud for frame 240
2026-02-14 10:12:03,189 - v2_logic.controllers.recursive_flow - INFO - [v2e_sensor_node] Generating events
[DensityPredictor] Warning: Model not fitted. Returning default density 1.0.
2026-02-14 10:12:03,410 - root - INFO - For numpy array image, we assume (HxWxC) format
2026-02-14 10:12:03,818 - root - INFO - Computing image embeddings for the provided image...
2026-02-14 10:12:03,855 - root - INFO - Image embeddings computed.
[SegmentationEngine] Filtered 2 objects (outside ROI/area range)
[SegmentationEngine] Union Masking: merged 1 fragments → 12 physical units
[SegmentationEngine] Clustered 12 masks into 8 volumetric clusters
2026-02-14 10:12:06,550 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Aggregated volume: 71.086842 m^3 (71086842.21 cm^3) from 8 clusters
2026-02-14 10:12:06,561 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] Computing volumetric reconciliation
2026-02-14 10:12:06,561 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] 3DC Result: 192691.62 units (From V_stack: 48.172904 m^3, rho: 1.00, V_unit: 0.000250 m^3)
2026-02-14 10:12:06,562 - v2_logic.controllers.recursive_flow - WARNING - [v3_math_node] Sanity Check Failed: Count 192691.61644588178 exceeds safety buffer 1000. Clipping n_vol.
2026-02-14 10:12:06,564 - v2_logic.controllers.recursive_flow - INFO - [fusion_engine_node] Fusing sensor data & Tracking
2026-02-14 10:12:06,617 - v2_logic.models.fusion_engine_v2 - INFO - [FusionV2] Shields: spatial: ✗ (0.47) | volumetric: ✗ (0.00) | latent: ✓ (0.50) → confidence=0.335 (threshold=0.60)
2026-02-14 10:12:06,617 - v2_logic.models.fusion_engine_v2 - WARNING - [FusionV2] LOW CONFIDENCE (0.335) — SLM Audit recommended.
2026-02-14 10:12:06,620 - v2_logic.controllers.recursive_flow - INFO - [logic_gate_node] Decision: exit (Rule: MaxLoopSafety)
2026-02-14 10:12:06,620 - v2_logic.controllers.recursive_flow - INFO - [route] Checking decision status: exit
2026-02-14 10:12:06,621 - run_recursive_system - INFO - Frame 240 Results:
2026-02-14 10:12:06,621 - run_recursive_system - INFO -   - Count (N_visible): 3
2026-02-14 10:12:06,621 - run_recursive_system - INFO -   - Volumetric Range: (1000, 1000)
2026-02-14 10:12:06,621 - run_recursive_system - INFO -   - Spike Energy: 811509.00
2026-02-14 10:12:06,621 - run_recursive_system - INFO -   - Anomaly Status: exit
2026-02-14 10:12:06,622 - run_recursive_system - INFO -   - SLM Reasoning: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table. There are three objects visible: the person's hand, the blue cup they are holding, and the two blue cups on the table. However, the 3D volume suggests there should be between 1000 and 1000 objects, which is a significant discrepancy.

The discrepancy likely arises from the following:

1. **Occlusion**: The person's body is partially obscuring the view of the objects on the table. The person's hand and body are in front of the cups and the ball
2026-02-14 10:12:06,677 - run_recursive_system - INFO - --- Processing Frame 255 ---
2026-02-14 10:12:06,681 - v2_logic.controllers.recursive_flow - INFO - [vjepa_brain_node] Encoding latent context
2026-02-14 10:12:07,119 - v2_logic.controllers.recursive_flow - INFO - [director] SLM Hypothesis received: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table
2026-02-14 10:12:07,120 - v2_logic.models.slm_engine - INFO - [SLMEngine] Requesting volume prior for 'cup'
2026-02-14 10:12:08,253 - v2_logic.models.slm_engine - INFO - [SLMEngine] Volume Response: 0.00025
2026-02-14 10:12:08,253 - v2_logic.controllers.recursive_flow - INFO - [director] SLM PHYSICAL PRIOR: Estimated volume for 'cup' is 0.00025 m^3
2026-02-14 10:12:08,258 - v2_logic.controllers.recursive_flow - INFO - [countvid_executor_node] Counting objects matching intent: ['cup', 'ball', 'person']
2026-02-14 10:12:08,261 - v2_logic.controllers.recursive_flow - INFO - [density_sensing_node] Analyzing physical density
2026-02-14 10:12:08,263 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Generating point cloud for frame 255
2026-02-14 10:12:08,267 - v2_logic.controllers.recursive_flow - INFO - [v2e_sensor_node] Generating events
[DensityPredictor] Warning: Model not fitted. Returning default density 1.0.
2026-02-14 10:12:08,801 - root - INFO - For numpy array image, we assume (HxWxC) format
2026-02-14 10:12:09,129 - root - INFO - Computing image embeddings for the provided image...
2026-02-14 10:12:09,192 - root - INFO - Image embeddings computed.
[SegmentationEngine] Filtered 3 objects (outside ROI/area range)
[SegmentationEngine] Union Masking: merged 2 fragments → 10 physical units
[SegmentationEngine] Clustered 10 masks into 7 volumetric clusters
2026-02-14 10:12:11,677 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Aggregated volume: 65.190675 m^3 (65190675.16 cm^3) from 7 clusters
2026-02-14 10:12:11,686 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] Computing volumetric reconciliation
2026-02-14 10:12:11,686 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] 3DC Result: 192691.62 units (From V_stack: 48.172904 m^3, rho: 1.00, V_unit: 0.000250 m^3)
2026-02-14 10:12:11,686 - v2_logic.controllers.recursive_flow - WARNING - [v3_math_node] Sanity Check Failed: Count 192691.61644588178 exceeds safety buffer 1000. Clipping n_vol.
2026-02-14 10:12:11,689 - v2_logic.controllers.recursive_flow - INFO - [fusion_engine_node] Fusing sensor data & Tracking
2026-02-14 10:12:11,741 - v2_logic.models.fusion_engine_v2 - INFO - [FusionV2] Shields: spatial: ✗ (0.73) | volumetric: ✗ (0.00) | latent: ✓ (0.50) → confidence=0.467 (threshold=0.60)
2026-02-14 10:12:11,741 - v2_logic.models.fusion_engine_v2 - WARNING - [FusionV2] LOW CONFIDENCE (0.467) — SLM Audit recommended.
2026-02-14 10:12:11,745 - v2_logic.controllers.recursive_flow - INFO - [logic_gate_node] Decision: exit (Rule: MaxLoopSafety)
2026-02-14 10:12:11,745 - v2_logic.controllers.recursive_flow - INFO - [route] Checking decision status: exit
2026-02-14 10:12:11,746 - run_recursive_system - INFO - Frame 255 Results:
2026-02-14 10:12:11,746 - run_recursive_system - INFO -   - Count (N_visible): 3
2026-02-14 10:12:11,746 - run_recursive_system - INFO -   - Volumetric Range: (1000, 1000)
2026-02-14 10:12:11,746 - run_recursive_system - INFO -   - Spike Energy: 1075803.00
2026-02-14 10:12:11,746 - run_recursive_system - INFO -   - Anomaly Status: exit
2026-02-14 10:12:11,746 - run_recursive_system - INFO -   - SLM Reasoning: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table. There are three objects visible: the person's hand, the blue cup they are holding, and the two blue cups on the table. However, the 3D volume suggests there should be between 1000 and 1000 objects, which is a significant discrepancy.

The discrepancy likely arises from the following:

1. **Occlusion**: The person's body is partially obscuring the view of the objects on the table. The person's hand and body are in front of the cups and the ball
2026-02-14 10:12:11,795 - run_recursive_system - INFO - --- Processing Frame 270 ---
2026-02-14 10:12:11,799 - v2_logic.controllers.recursive_flow - INFO - [vjepa_brain_node] Encoding latent context
2026-02-14 10:12:12,238 - v2_logic.controllers.recursive_flow - INFO - [director] SLM Hypothesis received: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table
2026-02-14 10:12:12,238 - v2_logic.models.slm_engine - INFO - [SLMEngine] Requesting volume prior for 'cup'
2026-02-14 10:12:13,207 - v2_logic.models.slm_engine - INFO - [SLMEngine] Volume Response: 0.00025
2026-02-14 10:12:13,208 - v2_logic.controllers.recursive_flow - INFO - [director] SLM PHYSICAL PRIOR: Estimated volume for 'cup' is 0.00025 m^3
2026-02-14 10:12:13,212 - v2_logic.controllers.recursive_flow - INFO - [countvid_executor_node] Counting objects matching intent: ['cup', 'ball', 'person']
2026-02-14 10:12:13,213 - v2_logic.controllers.recursive_flow - INFO - [density_sensing_node] Analyzing physical density
2026-02-14 10:12:13,215 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Generating point cloud for frame 270
2026-02-14 10:12:13,218 - v2_logic.controllers.recursive_flow - INFO - [v2e_sensor_node] Generating events
[DensityPredictor] Warning: Model not fitted. Returning default density 1.0.
2026-02-14 10:12:13,452 - root - INFO - For numpy array image, we assume (HxWxC) format
2026-02-14 10:12:13,924 - root - INFO - Computing image embeddings for the provided image...
2026-02-14 10:12:13,957 - root - INFO - Image embeddings computed.
[SegmentationEngine] Filtered 2 objects (outside ROI/area range)
[SegmentationEngine] Union Masking: merged 1 fragments → 10 physical units
[SegmentationEngine] Clustered 10 masks into 6 volumetric clusters
2026-02-14 10:12:16,512 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Aggregated volume: 116.868882 m^3 (116868882.19 cm^3) from 6 clusters
2026-02-14 10:12:16,528 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] Computing volumetric reconciliation
2026-02-14 10:12:16,528 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] 3DC Result: 192691.62 units (From V_stack: 48.172904 m^3, rho: 1.00, V_unit: 0.000250 m^3)
2026-02-14 10:12:16,528 - v2_logic.controllers.recursive_flow - WARNING - [v3_math_node] Sanity Check Failed: Count 192691.61644588178 exceeds safety buffer 1000. Clipping n_vol.
2026-02-14 10:12:16,532 - v2_logic.controllers.recursive_flow - INFO - [fusion_engine_node] Fusing sensor data & Tracking
2026-02-14 10:12:16,585 - v2_logic.models.fusion_engine_v2 - INFO - [FusionV2] Shields: spatial: ✗ (0.77) | volumetric: ✗ (0.00) | latent: ✓ (0.50) → confidence=0.486 (threshold=0.60)
2026-02-14 10:12:16,585 - v2_logic.models.fusion_engine_v2 - WARNING - [FusionV2] LOW CONFIDENCE (0.486) — SLM Audit recommended.
2026-02-14 10:12:16,588 - v2_logic.controllers.recursive_flow - INFO - [logic_gate_node] Decision: exit (Rule: MaxLoopSafety)
2026-02-14 10:12:16,589 - v2_logic.controllers.recursive_flow - INFO - [route] Checking decision status: exit
2026-02-14 10:12:16,590 - run_recursive_system - INFO - Frame 270 Results:
2026-02-14 10:12:16,590 - run_recursive_system - INFO -   - Count (N_visible): 3
2026-02-14 10:12:16,590 - run_recursive_system - INFO -   - Volumetric Range: (1000, 1000)
2026-02-14 10:12:16,590 - run_recursive_system - INFO -   - Spike Energy: 677862.00
2026-02-14 10:12:16,590 - run_recursive_system - INFO -   - Anomaly Status: exit
2026-02-14 10:12:16,590 - run_recursive_system - INFO -   - SLM Reasoning: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table. There are three objects visible: the person's hand, the blue cup they are holding, and the two blue cups on the table. However, the 3D volume suggests there should be between 1000 and 1000 objects, which is a significant discrepancy.

The discrepancy likely arises from the following:

1. **Occlusion**: The person's body is partially obscuring the view of the objects on the table. The person's hand and body are in front of the cups and the ball
2026-02-14 10:12:16,633 - run_recursive_system - INFO - --- Processing Frame 285 ---
2026-02-14 10:12:16,636 - v2_logic.controllers.recursive_flow - INFO - [vjepa_brain_node] Encoding latent context
2026-02-14 10:12:17,070 - v2_logic.controllers.recursive_flow - INFO - [director] SLM Hypothesis received: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table
2026-02-14 10:12:17,070 - v2_logic.models.slm_engine - INFO - [SLMEngine] Requesting volume prior for 'cup'
2026-02-14 10:12:18,021 - v2_logic.models.slm_engine - INFO - [SLMEngine] Volume Response: 0.00025
2026-02-14 10:12:18,021 - v2_logic.controllers.recursive_flow - INFO - [director] SLM PHYSICAL PRIOR: Estimated volume for 'cup' is 0.00025 m^3
2026-02-14 10:12:18,025 - v2_logic.controllers.recursive_flow - INFO - [countvid_executor_node] Counting objects matching intent: ['cup', 'ball', 'person']
2026-02-14 10:12:18,026 - v2_logic.controllers.recursive_flow - INFO - [density_sensing_node] Analyzing physical density
2026-02-14 10:12:18,028 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Generating point cloud for frame 285
2026-02-14 10:12:18,034 - v2_logic.controllers.recursive_flow - INFO - [v2e_sensor_node] Generating events
[DensityPredictor] Warning: Model not fitted. Returning default density 1.0.
2026-02-14 10:12:18,241 - root - INFO - For numpy array image, we assume (HxWxC) format
2026-02-14 10:12:18,320 - root - INFO - Computing image embeddings for the provided image...
2026-02-14 10:12:18,359 - root - INFO - Image embeddings computed.
[SegmentationEngine] Filtered 3 objects (outside ROI/area range)
[SegmentationEngine] Union Masking: merged 1 fragments → 11 physical units
[SegmentationEngine] Clustered 11 masks into 6 volumetric clusters
2026-02-14 10:12:21,868 - v2_logic.controllers.recursive_flow - INFO - [sam2_depth_node] Aggregated volume: 96.871540 m^3 (96871540.48 cm^3) from 6 clusters
2026-02-14 10:12:21,883 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] Computing volumetric reconciliation
2026-02-14 10:12:21,883 - v2_logic.controllers.recursive_flow - INFO - [v3_math_node] 3DC Result: 192691.62 units (From V_stack: 48.172904 m^3, rho: 1.00, V_unit: 0.000250 m^3)
2026-02-14 10:12:21,884 - v2_logic.controllers.recursive_flow - WARNING - [v3_math_node] Sanity Check Failed: Count 192691.61644588178 exceeds safety buffer 1000. Clipping n_vol.
2026-02-14 10:12:21,888 - v2_logic.controllers.recursive_flow - INFO - [fusion_engine_node] Fusing sensor data & Tracking
2026-02-14 10:12:21,958 - v2_logic.models.fusion_engine_v2 - INFO - [FusionV2] Shields: spatial: ✗ (0.48) | volumetric: ✗ (0.00) | latent: ✓ (0.50) → confidence=0.342 (threshold=0.60)
2026-02-14 10:12:21,959 - v2_logic.models.fusion_engine_v2 - WARNING - [FusionV2] LOW CONFIDENCE (0.342) — SLM Audit recommended.
2026-02-14 10:12:21,962 - v2_logic.controllers.recursive_flow - INFO - [logic_gate_node] Decision: exit (Rule: MaxLoopSafety)
2026-02-14 10:12:21,963 - v2_logic.controllers.recursive_flow - INFO - [route] Checking decision status: exit
2026-02-14 10:12:21,963 - run_recursive_system - INFO - Frame 285 Results:
2026-02-14 10:12:21,963 - run_recursive_system - INFO -   - Count (N_visible): 3
2026-02-14 10:12:21,963 - run_recursive_system - INFO -   - Volumetric Range: (1000, 1000)
2026-02-14 10:12:21,963 - run_recursive_system - INFO -   - Spike Energy: 840630.00
2026-02-14 10:12:21,964 - run_recursive_system - INFO -   - Anomaly Status: exit
2026-02-14 10:12:21,964 - run_recursive_system - INFO -   - SLM Reasoning: The image shows a person holding a blue cup, with two other blue cups and a blue ball on a table. There are three objects visible: the person's hand, the blue cup they are holding, and the two blue cups on the table. However, the 3D volume suggests there should be between 1000 and 1000 objects, which is a significant discrepancy.

The discrepancy likely arises from the following:

1. **Occlusion**: The person's body is partially obscuring the view of the objects on the table. The person's hand and body are in front of the cups and the ball
2026-02-14 10:12:22,013 - run_recursive_system - INFO - Processing complete.


👁️ BAGIAN 2: Menghasilkan Video Visualisasi (MP4)...
--------------------------------------------------
INFO:numexpr.utils:NumExpr defaulting to 2 threads.
==================================================
 INFERENCE VISUALIZER: GLIDE-AND-COUNT
==================================================
 Source Video   : /content/numeri-vjepa-experiment/Techs/sam2-main/sam2-main/demo/data/gallery/02_cups.mp4
 Output target  : /content/output_v2.mp4
 Sensitivity    : 0.2
--------------------------------------------------
 Execution Unit : CUDA
 GPU Model      : Tesla T4
 VRAM Available : 15.64 GB
--------------------------------------------------
INFO:v2ecore.emulator:ON/OFF log_e temporal contrast thresholds: 0.2 / 0.2 +/- 0.03
WARNING:v2ecore.emulator:cannot get screen size for window placement: No enumerators available
INFO:v2_logic.models.v2e_engine:[V2E] Initialized on cuda
INFO:v2_logic.models.vl_jepa_engine:[VL-JEPA] Loading model: google/paligemma-3b-mix-224 on cuda
INFO:httpx:HTTP Request: HEAD https://huggingface.co/google/paligemma-3b-mix-224/resolve/main/processor_config.json "HTTP/1.1 404 Not Found"
INFO:httpx:HTTP Request: HEAD https://huggingface.co/google/paligemma-3b-mix-224/resolve/main/preprocessor_config.json "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: GET https://huggingface.co/google/paligemma-3b-mix-224/resolve/main/preprocessor_config.json "HTTP/1.1 200 OK"
preprocessor_config.json: 100% 699/699 [00:00<00:00, 2.40MB/s]
INFO:httpx:HTTP Request: HEAD https://huggingface.co/google/paligemma-3b-mix-224/resolve/main/processor_config.json "HTTP/1.1 404 Not Found"
INFO:httpx:HTTP Request: HEAD https://huggingface.co/google/paligemma-3b-mix-224/resolve/main/preprocessor_config.json "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: GET https://huggingface.co/api/models/google/paligemma-3b-mix-224/tree/main/additional_chat_templates?recursive=false&expand=false "HTTP/1.1 404 Not Found"
INFO:httpx:HTTP Request: HEAD https://huggingface.co/google/paligemma-3b-mix-224/resolve/main/processor_config.json "HTTP/1.1 404 Not Found"
INFO:httpx:HTTP Request: HEAD https://huggingface.co/google/paligemma-3b-mix-224/resolve/main/chat_template.json "HTTP/1.1 404 Not Found"
INFO:httpx:HTTP Request: HEAD https://huggingface.co/google/paligemma-3b-mix-224/resolve/main/chat_template.jinja "HTTP/1.1 404 Not Found"
INFO:httpx:HTTP Request: HEAD https://huggingface.co/google/paligemma-3b-mix-224/resolve/main/audio_tokenizer_config.json "HTTP/1.1 404 Not Found"
INFO:httpx:HTTP Request: HEAD https://huggingface.co/google/paligemma-3b-mix-224/resolve/main/processor_config.json "HTTP/1.1 404 Not Found"
INFO:httpx:HTTP Request: HEAD https://huggingface.co/google/paligemma-3b-mix-224/resolve/main/preprocessor_config.json "HTTP/1.1 200 OK"
The image processor of type `SiglipImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`. 
INFO:httpx:HTTP Request: HEAD https://huggingface.co/google/paligemma-3b-mix-224/resolve/main/processor_config.json "HTTP/1.1 404 Not Found"
INFO:httpx:HTTP Request: HEAD https://huggingface.co/google/paligemma-3b-mix-224/resolve/main/preprocessor_config.json "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: HEAD https://huggingface.co/google/paligemma-3b-mix-224/resolve/main/config.json "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: GET https://huggingface.co/google/paligemma-3b-mix-224/resolve/main/config.json "HTTP/1.1 200 OK"
config.json: 100% 1.03k/1.03k [00:00<00:00, 5.17MB/s]
INFO:httpx:HTTP Request: HEAD https://huggingface.co/google/paligemma-3b-mix-224/resolve/main/tokenizer_config.json "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: GET https://huggingface.co/google/paligemma-3b-mix-224/resolve/main/tokenizer_config.json "HTTP/1.1 200 OK"
tokenizer_config.json: 100% 40.0k/40.0k [00:00<00:00, 12.6MB/s]
INFO:httpx:HTTP Request: HEAD https://huggingface.co/google/paligemma-3b-mix-224/resolve/main/tokenizer_config.json "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: GET https://huggingface.co/api/models/google/paligemma-3b-mix-224/tree/main/additional_chat_templates?recursive=false&expand=false "HTTP/1.1 404 Not Found"
INFO:httpx:HTTP Request: GET https://huggingface.co/api/models/google/paligemma-3b-mix-224/tree/main?recursive=true&expand=false "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: HEAD https://huggingface.co/google/paligemma-3b-mix-224/resolve/main/tokenizer.json "HTTP/1.1 302 Found"
INFO:httpx:HTTP Request: GET https://huggingface.co/api/models/google/paligemma-3b-mix-224/xet-read-token/d1d8734c9c3ad0ccfeea4afc270faa356c2ba515 "HTTP/1.1 200 OK"
tokenizer.json: 100% 17.5M/17.5M [00:00<00:00, 41.3MB/s]
INFO:httpx:HTTP Request: HEAD https://huggingface.co/google/paligemma-3b-mix-224/resolve/main/tokenizer.model "HTTP/1.1 302 Found"
tokenizer.model: 100% 4.26M/4.26M [00:00<00:00, 39.8MB/s]
INFO:httpx:HTTP Request: HEAD https://huggingface.co/google/paligemma-3b-mix-224/resolve/main/added_tokens.json "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: GET https://huggingface.co/google/paligemma-3b-mix-224/resolve/main/added_tokens.json "HTTP/1.1 200 OK"
added_tokens.json: 100% 24.0/24.0 [00:00<00:00, 150kB/s]
INFO:httpx:HTTP Request: HEAD https://huggingface.co/google/paligemma-3b-mix-224/resolve/main/special_tokens_map.json "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: GET https://huggingface.co/google/paligemma-3b-mix-224/resolve/main/special_tokens_map.json "HTTP/1.1 200 OK"
special_tokens_map.json: 100% 607/607 [00:00<00:00, 3.28MB/s]
INFO:httpx:HTTP Request: GET https://huggingface.co/api/models/google/paligemma-3b-mix-224 "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: HEAD https://huggingface.co/google/paligemma-3b-mix-224/resolve/main/config.json "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: HEAD https://huggingface.co/google/paligemma-3b-mix-224/resolve/main/adapter_config.json "HTTP/1.1 404 Not Found"
INFO:httpx:HTTP Request: HEAD https://huggingface.co/google/paligemma-3b-mix-224/resolve/main/config.json "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: HEAD https://huggingface.co/google/paligemma-3b-mix-224/resolve/main/model.safetensors "HTTP/1.1 404 Not Found"
INFO:httpx:HTTP Request: HEAD https://huggingface.co/google/paligemma-3b-mix-224/resolve/main/model.safetensors.index.json "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: GET https://huggingface.co/google/paligemma-3b-mix-224/resolve/main/model.safetensors.index.json "HTTP/1.1 200 OK"
model.safetensors.index.json: 100% 62.6k/62.6k [00:00<00:00, 12.6MB/s]
INFO:httpx:HTTP Request: GET https://huggingface.co/api/models/google/paligemma-3b-mix-224/revision/main "HTTP/1.1 200 OK"
Downloading (incomplete total...): 0.00B [00:00, ?B/s]
Fetching 3 files:   0% 0/3 [00:00<?, ?it/s]INFO:httpx:HTTP Request: HEAD https://huggingface.co/google/paligemma-3b-mix-224/resolve/d1d8734c9c3ad0ccfeea4afc270faa356c2ba515/model-00002-of-00003.safetensors "HTTP/1.1 302 Found"
Downloading (incomplete total...):   0% 0.00/5.00G [00:00<?, ?B/s]INFO:httpx:HTTP Request: HEAD https://huggingface.co/google/paligemma-3b-mix-224/resolve/d1d8734c9c3ad0ccfeea4afc270faa356c2ba515/model-00003-of-00003.safetensors "HTTP/1.1 302 Found"
Downloading (incomplete total...):   0% 0.00/6.74G [00:00<?, ?B/s]INFO:httpx:HTTP Request: HEAD https://huggingface.co/google/paligemma-3b-mix-224/resolve/d1d8734c9c3ad0ccfeea4afc270faa356c2ba515/model-00001-of-00003.safetensors "HTTP/1.1 302 Found"
Downloading (incomplete total...):  99% 11.6G/11.7G [02:44<00:00, 225MB/s]
Downloading (incomplete total...): 100% 11.7G/11.7G [02:48<00:00, 49.0MB/s]
Fetching 3 files: 100% 3/3 [02:48<00:00, 56.33s/it]
Download complete: 100% 11.7G/11.7G [02:49<00:00, 69.2MB/s]
Loading weights: 100% 603/603 [00:45<00:00, 13.33it/s, Materializing param=model.vision_tower.vision_model.post_layernorm.weight]
INFO:httpx:HTTP Request: HEAD https://huggingface.co/google/paligemma-3b-mix-224/resolve/main/generation_config.json "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: GET https://huggingface.co/google/paligemma-3b-mix-224/resolve/main/generation_config.json "HTTP/1.1 200 OK"
generation_config.json: 100% 137/137 [00:00<00:00, 700kB/s]
INFO:v2_logic.models.vl_jepa_engine:[VL-JEPA] Model loaded successfully
INFO:v2_logic.models.v_jepa_engine:[V-JEPA] Initialized on cuda
INFO:v2_logic.models.v_jepa_engine:[V-JEPA] Successfully loaded and aligned weights from /content/numeri-vjepa-experiment/Implementation/v2_logic/models/../../checkpoints/vjepa_vitl16.pth.tar
/usr/local/lib/python3.12/dist-packages/timm/models/layers/__init__.py:49: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
/usr/local/lib/python3.12/dist-packages/torch/functional.py:505: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at /pytorch/aten/src/ATen/native/TensorShape.cpp:4381.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
final text_encoder_type: /content/numeri-vjepa-experiment/Techs/CountVid-main/CountVid-main/checkpoints/bert-base-uncased
load tokenizer done.
Loading weights: 100% 199/199 [00:00<00:00, 621.27it/s, Materializing param=pooler.dense.weight]
BertModel LOAD REPORT from: /content/numeri-vjepa-experiment/Techs/CountVid-main/CountVid-main/checkpoints/bert-base-uncased
Key                                        | Status     |  | 
-------------------------------------------+------------+--+-
cls.predictions.bias                       | UNEXPECTED |  | 
cls.predictions.transform.dense.bias       | UNEXPECTED |  | 
cls.seq_relationship.bias                  | UNEXPECTED |  | 
cls.predictions.transform.dense.weight     | UNEXPECTED |  | 
cls.seq_relationship.weight                | UNEXPECTED |  | 
cls.predictions.transform.LayerNorm.weight | UNEXPECTED |  | 
cls.predictions.transform.LayerNorm.bias   | UNEXPECTED |  | 

Notes:
- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.
final text_encoder_type: /content/numeri-vjepa-experiment/Techs/CountVid-main/CountVid-main/checkpoints/bert-base-uncased
load tokenizer done.
INFO:v2_logic.models.count_vid_engine:[CountVid] Model loaded successfully on cuda
[SegmentationEngine] Loading SAM2...
[SegmentationEngine] No local checkpoint. Loading from HuggingFace...
INFO:httpx:HTTP Request: HEAD https://huggingface.co/facebook/sam2.1-hiera-tiny/resolve/main/sam2.1_hiera_tiny.pt "HTTP/1.1 302 Found"
INFO:root:Loaded checkpoint sucessfully
[SegmentationEngine] Loaded from HuggingFace: facebook/sam2.1-hiera-tiny
[SegmentationEngine] SAM2 loaded on cuda
WARNING:dinov2:xFormers not available
WARNING:dinov2:xFormers not available
INFO:v2_logic.models.depth_engine:[DepthEngine] Loading from checkpoint: /content/numeri-vjepa-experiment/Techs/Depth-Anything-V2-main/Depth-Anything-V2-main/checkpoints/depth_anything_v2_vits.pth
INFO:dinov2:using MLP layer as FFN
INFO:v2_logic.models.depth_engine:[DepthEngine] Model loaded on cuda (encoder=vits)
You are passing both `text` and `images` to `PaliGemmaProcessor`. The processor expects special image tokens in the text, as many tokens as there are images per each text. It is recommended to add `<image>` tokens in the very beginning of your text. For this call, we will infer how many images each text has and add special tokens.
The following generation flags are not valid and may be ignored: ['temperature', 'top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
INFO:v2_logic.models.vl_jepa_engine:[VL-JEPA] Identified Intent: cup
INFO:v2_logic.pipeline.engine_v2:[Director] Initial Intent: cup
V2 Inference:   0% 0/300 [00:00<?, ?it/s]/usr/lib/python3.12/contextlib.py:105: FutureWarning: `torch.backends.cuda.sdp_kernel()` is deprecated. In the future, this context manager will be removed. Please see `torch.nn.attention.sdpa_kernel()` for the new context manager, with updated signature.
  self.gen = func(*args, **kwds)
[CountVid Patch] ⚠️ Smart Dispatcher: Falling back to Positional Signature.
[CountVid Patch] ⚠️ Smart Dispatcher: Falling back for get_head_mask.
/content/numeri-vjepa-experiment/Techs/CountVid-main/CountVid-main/models/GroundingDINO/transformer.py:901: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=False):
INFO:root:For numpy array image, we assume (HxWxC) format
INFO:root:Computing image embeddings for the provided image...
INFO:root:Image embeddings computed.
[SegmentationEngine] Filtered 2 objects (outside ROI/area range)
V2 Inference:   5% 15/300 [00:09<00:52,  5.45it/s]INFO:root:For numpy array image, we assume (HxWxC) format
INFO:root:Computing image embeddings for the provided image...
INFO:root:Image embeddings computed.
[SegmentationEngine] Filtered 3 objects (outside ROI/area range)
INFO:v2_logic.models.count_vid_engine:[CountVid] Final Tally: 3
V2 Inference:  10% 31/300 [00:13<00:33,  8.11it/s]INFO:root:For numpy array image, we assume (HxWxC) format
INFO:root:Computing image embeddings for the provided image...
INFO:root:Image embeddings computed.
[SegmentationEngine] Filtered 3 objects (outside ROI/area range)
INFO:v2_logic.models.count_vid_engine:[CountVid] Final Tally: 3
V2 Inference:  16% 47/300 [00:18<00:32,  7.67it/s]INFO:root:For numpy array image, we assume (HxWxC) format
INFO:root:Computing image embeddings for the provided image...
INFO:root:Image embeddings computed.
[SegmentationEngine] Filtered 4 objects (outside ROI/area range)
INFO:v2_logic.models.count_vid_engine:[CountVid] Final Tally: 3
V2 Inference:  21% 62/300 [00:22<00:29,  7.97it/s]INFO:root:For numpy array image, we assume (HxWxC) format
INFO:root:Computing image embeddings for the provided image...
INFO:root:Image embeddings computed.
[SegmentationEngine] Filtered 3 objects (outside ROI/area range)
INFO:v2_logic.models.count_vid_engine:[CountVid] Final Tally: 3
V2 Inference:  26% 79/300 [00:28<00:30,  7.22it/s]INFO:root:For numpy array image, we assume (HxWxC) format
INFO:root:Computing image embeddings for the provided image...
INFO:root:Image embeddings computed.
[SegmentationEngine] Filtered 3 objects (outside ROI/area range)
INFO:v2_logic.models.count_vid_engine:[CountVid] Final Tally: 3
V2 Inference:  31% 94/300 [00:33<00:24,  8.50it/s]INFO:root:For numpy array image, we assume (HxWxC) format
INFO:root:Computing image embeddings for the provided image...
INFO:root:Image embeddings computed.
[SegmentationEngine] Filtered 3 objects (outside ROI/area range)
INFO:v2_logic.models.count_vid_engine:[CountVid] Final Tally: 3
V2 Inference:  37% 111/300 [00:38<00:24,  7.77it/s]INFO:root:For numpy array image, we assume (HxWxC) format
INFO:root:Computing image embeddings for the provided image...
INFO:root:Image embeddings computed.
[SegmentationEngine] Filtered 3 objects (outside ROI/area range)
INFO:v2_logic.models.count_vid_engine:[CountVid] Final Tally: 3
V2 Inference:  42% 126/300 [00:43<00:20,  8.47it/s]INFO:root:For numpy array image, we assume (HxWxC) format
INFO:root:Computing image embeddings for the provided image...
INFO:root:Image embeddings computed.
[SegmentationEngine] Filtered 3 objects (outside ROI/area range)
INFO:v2_logic.models.count_vid_engine:[CountVid] Final Tally: 3
V2 Inference:  47% 142/300 [00:47<00:21,  7.22it/s]INFO:root:For numpy array image, we assume (HxWxC) format
INFO:root:Computing image embeddings for the provided image...
INFO:root:Image embeddings computed.
[SegmentationEngine] Filtered 4 objects (outside ROI/area range)
INFO:v2_logic.models.count_vid_engine:[CountVid] Final Tally: 3
V2 Inference:  53% 159/300 [00:53<00:16,  8.74it/s]INFO:root:For numpy array image, we assume (HxWxC) format
INFO:root:Computing image embeddings for the provided image...
INFO:root:Image embeddings computed.
[SegmentationEngine] Filtered 3 objects (outside ROI/area range)
INFO:v2_logic.models.count_vid_engine:[CountVid] Final Tally: 3
V2 Inference:  58% 175/300 [00:57<00:15,  7.95it/s]INFO:root:For numpy array image, we assume (HxWxC) format
INFO:root:Computing image embeddings for the provided image...
INFO:root:Image embeddings computed.
[SegmentationEngine] Filtered 3 objects (outside ROI/area range)
INFO:v2_logic.models.count_vid_engine:[CountVid] Final Tally: 3
V2 Inference:  64% 191/300 [01:03<00:16,  6.78it/s]INFO:root:For numpy array image, we assume (HxWxC) format
INFO:root:Computing image embeddings for the provided image...
INFO:root:Image embeddings computed.
[SegmentationEngine] Filtered 3 objects (outside ROI/area range)
INFO:v2_logic.models.count_vid_engine:[CountVid] Final Tally: 3
V2 Inference:  69% 206/300 [01:08<00:11,  8.13it/s]INFO:root:For numpy array image, we assume (HxWxC) format
INFO:root:Computing image embeddings for the provided image...
INFO:root:Image embeddings computed.
[SegmentationEngine] Filtered 3 objects (outside ROI/area range)
INFO:v2_logic.models.count_vid_engine:[CountVid] Final Tally: 3
V2 Inference:  74% 222/300 [01:12<00:10,  7.70it/s]INFO:root:For numpy array image, we assume (HxWxC) format
INFO:root:Computing image embeddings for the provided image...
INFO:root:Image embeddings computed.
[SegmentationEngine] Filtered 2 objects (outside ROI/area range)
INFO:v2_logic.models.count_vid_engine:[CountVid] Final Tally: 3
V2 Inference:  80% 239/300 [01:17<00:06,  9.37it/s]INFO:root:For numpy array image, we assume (HxWxC) format
INFO:root:Computing image embeddings for the provided image...
INFO:root:Image embeddings computed.
[SegmentationEngine] Filtered 2 objects (outside ROI/area range)
INFO:v2_logic.models.count_vid_engine:[CountVid] Final Tally: 3
V2 Inference:  85% 255/300 [01:22<00:05,  7.60it/s]INFO:root:For numpy array image, we assume (HxWxC) format
INFO:root:Computing image embeddings for the provided image...
INFO:root:Image embeddings computed.
[SegmentationEngine] Filtered 2 objects (outside ROI/area range)
INFO:v2_logic.models.count_vid_engine:[CountVid] Final Tally: 3
V2 Inference:  90% 271/300 [01:28<00:03,  7.26it/s]INFO:root:For numpy array image, we assume (HxWxC) format
INFO:root:Computing image embeddings for the provided image...
INFO:root:Image embeddings computed.
[SegmentationEngine] Filtered 2 objects (outside ROI/area range)
INFO:v2_logic.models.count_vid_engine:[CountVid] Final Tally: 3
V2 Inference:  96% 287/300 [01:32<00:01,  9.10it/s]INFO:root:For numpy array image, we assume (HxWxC) format
INFO:root:Computing image embeddings for the provided image...
INFO:root:Image embeddings computed.
[SegmentationEngine] Filtered 3 objects (outside ROI/area range)
INFO:v2_logic.models.count_vid_engine:[CountVid] Final Tally: 3
V2 Inference: 100% 299/300 [01:37<00:00,  3.07it/s]
INFO:v2_logic.pipeline.engine_v2:Visualizer complete. Saved to /content/output_v2.mp4


✅ Pengujian Selesai!
