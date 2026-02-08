
# sleep 3600  # 延迟4个小时（4小时 * 60分钟 * 60秒 = 14400秒）


# ################################ AXIS all #################################
model_names=(
    "./output/qw3b-grpo-rexcliq-zero2-2048-bs8_rerun/v18-20251118-104631/checkpoint-13838"
    'byrLLCC/ChestX-Reasoner'
)
result_tags=(
    "CheXOne"
    "ChestX-Reasoner"
)
save_dirs=(
    "./chexbench_eval_updated/CheXOne/"
    "./chexbench_eval_updated/ChestX-Reasoner/"
)
# CUDA_LAUNCH_BLOCKING=1
# replace the rexvqa-test-reason.json with the json file you want to test
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export IMAGE_MAX_TOKEN_NUM=1024
for idx in "${!model_names[@]}"; do
    model_name="${model_names[$idx]}"
    result_tag="${result_tags[$idx]}"



    NPROC_PER_NODE=1 \
    CUDA_VISIBLE_DEVICES=0 \
    IMAGE_MAX_TOKEN_NUM=1024 \
    MAX_PIXELS=262144 \
    VIDEO_MAX_PIXELS=50176 \
    FPS_MAX_FRAMES=12 \
    swift infer \
        --model ${model_name} \
        --infer_backend vllm \
        --use_hf true \
        --val_dataset ./examples/train/chexone/prepare_testing/2_json_for_testing/rexvqa-test-reason.json \
        --max_new_tokens 2048 \
        --max_length 4096 \
        --result_path ${save_dirs[$idx]}/rexvqa-test-reason_${result_tag}.json \
        --max_batch_size 16 \
        --temperature 0.0 \
        --remove_unused_columns False \
        --logprobs false \
        --system 'You are a helpful assistant.'
done