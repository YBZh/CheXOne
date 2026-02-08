# CUDA_VISIBLE_DEVICES=0  swift rollout --model Qwen/Qwen2.5-VL-3B-Instruct  ### MASTER_PORT=50353 \


# bash examples/train/vlm_medical/grpo_3b_internal.sh
    # --gradient_accumulation_steps $((128 / ($NUM_GPUS * $per_device_train_batch_size))) \
    # --model Qwen/Qwen2.5-VL-3B-Instruct \
# export PYTHONWARNINGS="ignore::UserWarning"
# export NCCL_DEBUG=WARN ## INFO, WARN, ERROR, DEBUG, DETAIL
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=OFF ## DETAIL
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
# NUM_GPUS=8
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# NUM_GPUS=2
# export CUDA_VISIBLE_DEVICES=0,1
# 自动设置 WORLD_SIZE
export WORLD_SIZE=$NUM_GPUS
# export WANDB_MODE=offline
echo "Detected $NUM_GPUS GPUs. Setting WORLD_SIZE=$WORLD_SIZE"
per_device_train_batch_size=8
# 设置CUDA设备同步模式


# 禁用CUDA缓存（可选）
# export CUDA_CACHE_DISABLE=1
MAX_PIXELS=262144 \
NPROC_PER_NODE=$NUM_GPUS \
PYTORCH_CUDA_ALLOC_CONF='' \
swift rlhf \
    --rlhf_type grpo \
    --model ./output/sft_qwenvl3b_16million_nostream/v0-20250829-211503/checkpoint-58687 \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs external_format_reward_boxed external_vqa_orm external_iou_reward external_generation_with_reason_reward \
    --use_vllm True \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_max_model_len 4096 \
    --vllm_tensor_parallel_size 4 \
    --offload_optimizer True \
    --offload_model True \
    --sleep_level 1 \
    --padding_free True \
    --deepspeed zero2 \
    --attn_impl flash_attn \
    --train_type full \
    --torch_dtype bfloat16 \
    --truncation_strategy delete \
    --dataset ./data/grpo_prepare/generation/output/ReXGradient_160K_Findings_Generation_subset_standardized_selected_59638_reason_filter.json \
              ./data/grpo_prepare/generation/output/ReXGradient_160K_Impression_Generation_subset_standardized_selected_51837_reason_filter.json \
              ./data/grpo_prepare/generation/output/chestxagent_sft_train_all_tasks_Findings_Generation_subset_standardized_selected_95168_reason_filter.json \
              ./data/grpo_prepare/generation/output/chestxagent_sft_train_all_tasks_Impression_Generation_subset_standardized_selected_216388_reason_filter.json#100000 \
              ./data/grpo_prepare/vqa_set/output/VQA_ReXGradient_160K_train_valid_selected_87638_reason_filter.json \
              ./data/grpo_prepare/vqa_set/output/chestxagent_sft_train_all_tasks_Close-Ended_VQA_subset_standardized_selected_93527_reason_filter.json \
              ./data/grpo_prepare/vqa_set/output/chestxagent_sft_train_all_tasks_Image_Classification_subset_standardized_selected_91044_reason_filter.json \
              ./data/grpo_prepare/vqa_set/chestxagent_sft_train_all_tasks_Temporal_Image_Classification_subset_standardized.json#2000 \
              ./data/grpo_prepare/vqa_set/output/chestxagent_sft_train_all_tasks_View_Classification_subset_standardized_selected_48999_reason_filter.json#20000 \
              ./data/grpo_prepare/iou_set_old_chexagent_form/chestxagent_sft_train_all_tasks_Abnormality_Grounding_subset_standardized_onebox_14262.json#20000 \
              ./data/grpo_prepare/iou_set_old_chexagent_form/chestxagent_sft_train_all_tasks_Chest_Tube_Segmentation_subset_standardized_onebox_1262.json#3000 \
              ./data/grpo_prepare/iou_set_old_chexagent_form/chestxagent_sft_train_all_tasks_Phrase_Grounding_subset_standardized_onebox_736.json#2000 \
              ./data/grpo_prepare/iou_set_old_chexagent_form/chestxagent_sft_train_all_tasks_Pneumothorax_Segmentation_subset_standardized_onebox_4334.json#10000 \
              ./data/grpo_prepare/generation/Findings_Generation_ReXGradient_160K_train_with_indication_139999_with_filtered_137934.json#50000 \
             ./data/grpo_prepare/generation/Impression_Generation_ReXGradient_160K_train_with_indication_139999_with_filtered_137934.json#50000 \
             ./data/grpo_prepare/generation/chestxagent_sft_train_all_tasks_Findings_Generation_with_Indication_subset_148090_with_filtered_146301.json#50000 \
             ./data/grpo_prepare/generation/chestxagent_sft_train_all_tasks_Impression_Generation_with_Indication_subset_172993_with_filtered_171144_153079.json#50000 \
             ./data/grpo_prepare/generation/chestxagent_sft_train_all_tasks_Progression_Findings_Generation_subset_108796.json#50000 \
             ./data/grpo_prepare/generation/chestxagent_sft_train_all_tasks_Progression_Impression_Generation_subset_193079.json#50000 \
    --split_dataset_ratio 0.0 \
    --max_completion_length 2048 \
    --max_length 4096 \
    --num_train_epochs 1 \
    --per_device_train_batch_size $per_device_train_batch_size \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 8 \
    --save_strategy 'steps' \
    --eval_strategy 'steps' \
    --save_steps 100 \
    --save_total_limit 1 \
    --logging_steps 50 \
    --output_dir ./output/qw3b-grpo-rexcliq-zero2-2048-bs8_rerun \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --num_generations 8 \
    --temperature 1.0 \
    --repetition_penalty 1.1 \
    --system 'You are a helpful assistant.' \
    --report_to wandb \
    --run_name grpo_rexcliq_zero2_2048_bs8_rerun \
    --beta 0.001 \
    --max_grad_norm 0.5 \
    --log_completions true \
    --sequence_parallel_size 1 \
    --dataloader_drop_last true \
    --freeze_llm False \
    --freeze_vit True \
    --freeze_aligner False

