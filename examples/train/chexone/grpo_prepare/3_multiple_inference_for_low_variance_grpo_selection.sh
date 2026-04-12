### 在sft完成后，grpo之前，挑出一部分 中度难度样本，用于grpo训练； 下面的代码服务于挑出中度难度样本

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# 自动设置 WORLD_SIZE
export WORLD_SIZE=$NUM_GPUS
echo "Detected $NUM_GPUS GPUs. Setting WORLD_SIZE=$WORLD_SIZE"

###########################################################  VQA Set  ###########################################################

NPROC_PER_NODE=$NUM_GPUS \
MAX_PIXELS=262144 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
swift infer \
    --model ./output/sft_qwenvl3b_16million_nostream/v0-20250829-211503/checkpoint-58687 \
    --model_type qwen2_5_vl \
    --infer_backend vllm \
    --val_dataset ./data/grpo_prepare/iou_set/chestxagent_sft_train_all_tasks_Abnormality_Grounding_subset_standardized_repeat.json \
    --max_new_tokens 2048 \
    --max_length 4096 \
    --result_path ./data/grpo_prepare/iou_set/output/chestxagent_sft_train_all_tasks_Abnormality_Grounding_subset_standardized_repeat.json \
    --max_batch_size 16 \
    --temperature 1.0 \
    --remove_unused_columns False \
    --logprobs false \
    --system 'You are a helpful assistant.' 


NPROC_PER_NODE=$NUM_GPUS \
MAX_PIXELS=262144 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
swift infer \
    --model ./output/sft_qwenvl3b_16million_nostream/v0-20250829-211503/checkpoint-58687 \
    --model_type qwen2_5_vl \
    --infer_backend vllm \
    --val_dataset ./data/grpo_prepare/iou_set/chestxagent_sft_train_all_tasks_Chest_Tube_Segmentation_subset_standardized_repeat.json \
    --max_new_tokens 2048 \
    --max_length 4096 \
    --result_path ./data/grpo_prepare/iou_set/output/chestxagent_sft_train_all_tasks_Chest_Tube_Segmentation_subset_standardized_repeat.json \
    --max_batch_size 16 \
    --temperature 1.0 \
    --remove_unused_columns False \
    --logprobs false \
    --system 'You are a helpful assistant.' 


NPROC_PER_NODE=$NUM_GPUS \
MAX_PIXELS=262144 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
swift infer \
    --model ./output/sft_qwenvl3b_16million_nostream/v0-20250829-211503/checkpoint-58687 \
    --model_type qwen2_5_vl \
    --infer_backend vllm \
    --val_dataset ./data/grpo_prepare/iou_set/chestxagent_sft_train_all_tasks_Phrase_Grounding_subset_standardized_repeat.json \
    --max_new_tokens 2048 \
    --max_length 4096 \
    --result_path ./data/grpo_prepare/iou_set/output/chestxagent_sft_train_all_tasks_Phrase_Grounding_subset_standardized_repeat.json \
    --max_batch_size 16 \
    --temperature 1.0 \
    --remove_unused_columns False \
    --logprobs false \
    --system 'You are a helpful assistant.' 


NPROC_PER_NODE=$NUM_GPUS \
MAX_PIXELS=262144 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
swift infer \
    --model ./output/sft_qwenvl3b_16million_nostream/v0-20250829-211503/checkpoint-58687 \
    --model_type qwen2_5_vl \
    --infer_backend vllm \
    --val_dataset ./data/grpo_prepare/iou_set/chestxagent_sft_train_all_tasks_Pneumothorax_Segmentation_subset_standardized_repeat.json \
    --max_new_tokens 2048 \
    --max_length 4096 \
    --result_path ./data/grpo_prepare/iou_set/output/chestxagent_sft_train_all_tasks_Pneumothorax_Segmentation_subset_standardized_repeat.json \
    --max_batch_size 16 \
    --temperature 1.0 \
    --remove_unused_columns False \
    --logprobs false \
    --system 'You are a helpful assistant.' 

# NPROC_PER_NODE=$NUM_GPUS \
# MAX_PIXELS=262144 \
# VIDEO_MAX_PIXELS=50176 \
# FPS_MAX_FRAMES=12 \
# swift infer \
#     --model ./output/sft_qwenvl3b_16million_nostream/v0-20250829-211503/checkpoint-58687 \
#     --model_type qwen2_5_vl \
#     --infer_backend vllm \
#     --val_dataset ./data/grpo_prepare/vqa_set/VQA_ReXGradient_160K_train_valid_repeat.json \
#     --max_new_tokens 2048 \
#     --max_length 4096 \
#     --result_path ./data/grpo_prepare/vqa_set/output/VQA_ReXGradient_160K_train_valid_repeat.json \
#     --max_batch_size 16 \
#     --temperature 1.0 \
#     --remove_unused_columns False \
#     --logprobs false \
#     --system 'You are a helpful assistant.' 

# NPROC_PER_NODE=$NUM_GPUS \
# MAX_PIXELS=262144 \
# VIDEO_MAX_PIXELS=50176 \
# FPS_MAX_FRAMES=12 \
# swift infer \
#     --model ./output/sft_qwenvl3b_16million_nostream/v0-20250829-211503/checkpoint-58687 \
#     --model_type qwen2_5_vl \
#     --infer_backend vllm \
#     --val_dataset ./data/grpo_prepare/vqa_set/chestxagent_sft_train_all_tasks_Close-Ended_VQA_subset_standardized_repeat.json \
#     --max_new_tokens 2048 \
#     --max_length 4096 \
#     --result_path ./data/grpo_prepare/vqa_set/output/chestxagent_sft_train_all_tasks_Close-Ended_VQA_subset_standardized_repeat.json \
#     --max_batch_size 16 \
#     --temperature 1.0 \
#     --remove_unused_columns False \
#     --logprobs false \
#     --system 'You are a helpful assistant.' 


# NPROC_PER_NODE=$NUM_GPUS \
# MAX_PIXELS=262144 \
# VIDEO_MAX_PIXELS=50176 \
# FPS_MAX_FRAMES=12 \
# swift infer \
#     --model ./output/sft_qwenvl3b_16million_nostream/v0-20250829-211503/checkpoint-58687 \
#     --model_type qwen2_5_vl \
#     --infer_backend vllm \
#     --val_dataset ./data/grpo_prepare/vqa_set/chestxagent_sft_train_all_tasks_Image_Classification_subset_standardized_repeat.json \
#     --max_new_tokens 2048 \
#     --max_length 4096 \
#     --result_path ./data/grpo_prepare/vqa_set/output/chestxagent_sft_train_all_tasks_Image_Classification_subset_standardized_repeat.json \
#     --max_batch_size 16 \
#     --temperature 1.0 \
#     --remove_unused_columns False \
#     --logprobs false \
#     --system 'You are a helpful assistant.' 

# NPROC_PER_NODE=$NUM_GPUS \
# MAX_PIXELS=262144 \
# VIDEO_MAX_PIXELS=50176 \
# FPS_MAX_FRAMES=12 \
# swift infer \
#     --model ./output/sft_qwenvl3b_16million_nostream/v0-20250829-211503/checkpoint-58687 \
#     --model_type qwen2_5_vl \
#     --infer_backend vllm \
#     --val_dataset ./data/grpo_prepare/vqa_set/chestxagent_sft_train_all_tasks_View_Classification_subset_standardized_repeat.json \ 
#     --max_new_tokens 2048 \
#     --max_length 4096 \
#     --result_path ./data/grpo_prepare/vqa_set/output/chestxagent_sft_train_all_tasks_View_Classification_subset_standardized_repeat.json \
#     --max_batch_size 16 \
#     --temperature 1.0 \
#     --remove_unused_columns False \
#     --logprobs false \
#     --system 'You are a helpful assistant.' 

# ###########################################################  VQA Set  ###########################################################