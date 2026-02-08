# bash examples/train/vlm_medical/sft.sh
# 检测 GPU 数量
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
export IMAGE_MAX_TOKEN_NUM=1024
# 自动设置 WORLD_SIZE
export WORLD_SIZE=$NUM_GPUS
# export WANDB_MODE=offline
echo "Detected $NUM_GPUS GPUs. Setting WORLD_SIZE=$WORLD_SIZE"
per_device_train_batch_size=8
NPROC_PER_NODE=$NUM_GPUS \
swift sft \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset data/revised_json_for_sft/Findings_Generation_ReXGradient_160K_139999.json \
        data/revised_json_for_sft/Findings_Generation_ReXGradient_160K_train_with_indication_139999.json \
        data/revised_json_for_sft/Findings_Summarization_ReXGradient_160K_139999.json \
        data/revised_json_for_sft/Impression_Generation_ReXGradient_160K_139999.json \
        data/revised_json_for_sft/Impression_Generation_ReXGradient_160K_train_with_indication_139999.json \
        data/MIMIC_CXR_reason_gen_all_tasks_Close-Ended_VQA_subset.json \
        data/MIMIC_CXR_reason_gen_all_tasks_Findings_Generation_subset.json \
        data/MIMIC_CXR_reason_gen_all_tasks_Findings_Generation_with_Indication_subset.json \
        data/MIMIC_CXR_reason_gen_all_tasks_Findings_Summarization_subset.json \
        data/MIMIC_CXR_reason_gen_all_tasks_Image-Text_Matching_subset.json \
        data/MIMIC_CXR_reason_gen_all_tasks_Image-Text_Selection_subset.json \
        data/MIMIC_CXR_reason_gen_all_tasks_Image_Classification_subset.json \
        data/MIMIC_CXR_reason_gen_all_tasks_Impression_Generation_subset.json \
        data/MIMIC_CXR_reason_gen_all_tasks_Impression_Generation_with_Indication_subset.json \
        data/MIMIC_CXR_reason_gen_all_tasks_Local_Findings_Generation_subset.json \
        data/MIMIC_CXR_reason_gen_all_tasks_Local_Impression_Generation_subset.json \
        data/MIMIC_CXR_reason_gen_all_tasks_Local_Progression_Findings_Generation_subset.json \
        data/MIMIC_CXR_reason_gen_all_tasks_Local_Progression_Impression_Generation_subset.json \
        data/MIMIC_CXR_reason_gen_all_tasks_Open-Ended_VQA_subset.json \
        data/MIMIC_CXR_reason_gen_all_tasks_Progression_Findings_Generation_subset.json \
        data/MIMIC_CXR_reason_gen_all_tasks_Progression_Impression_Generation_subset.json \
        data/MIMIC_CXR_reason_gen_all_tasks_View_Matching_subset.json \
        data/VQA_ReXGradient_160K_train_valid.json \
        data/chestxagent_sft_train_all_tasks_Abnormality_Detection_subset.json \
        data/chestxagent_sft_train_all_tasks_Abnormality_Grounding_subset.json \
        data/chestxagent_sft_train_all_tasks_Caption_Generation_subset.json \
        data/chestxagent_sft_train_all_tasks_Chest_Tube_Segmentation_subset.json \
        data/chestxagent_sft_train_all_tasks_Close-Ended_VQA_subset.json \
        data/chestxagent_sft_train_all_tasks_Difference_VQA_subset.json \
        data/chestxagent_sft_train_all_tasks_Findings_Generation_subset.json \
        data/chestxagent_sft_train_all_tasks_Findings_Generation_with_Indication_subset.json \
        data/chestxagent_sft_train_all_tasks_Findings_Summarization_subset.json \
        data/chestxagent_sft_train_all_tasks_Foreign_Object_Detection_subset.json \
        data/chestxagent_sft_train_all_tasks_Grounded_Captioning_subset.json \
        data/chestxagent_sft_train_all_tasks_Grounded_Diagnosis_subset.json \
        data/chestxagent_sft_train_all_tasks_Grounded_Phrase_Extraction_subset.json \
        data/chestxagent_sft_train_all_tasks_Image-Text_Matching_subset.json \
        data/chestxagent_sft_train_all_tasks_Image-Text_Selection_subset.json \
        data/chestxagent_sft_train_all_tasks_Image_Classification_subset.json \
        data/chestxagent_sft_train_all_tasks_Impression_Generation_subset.json \
        data/chestxagent_sft_train_all_tasks_Impression_Generation_with_Indication_subset.json \
        data/chestxagent_sft_train_all_tasks_Local_Findings_Generation_subset.json \
        data/chestxagent_sft_train_all_tasks_Local_Impression_Generation_subset.json \
        data/chestxagent_sft_train_all_tasks_Local_Progression_Findings_Generation_subset.json \
        data/chestxagent_sft_train_all_tasks_Local_Progression_Impression_Generation_subset.json \
        data/chestxagent_sft_train_all_tasks_Named_Entity_Recognition_subset.json \
        data/chestxagent_sft_train_all_tasks_Natural_Language_Explanation_subset.json \
        data/chestxagent_sft_train_all_tasks_Open-Ended_VQA_subset.json \
        data/chestxagent_sft_train_all_tasks_Phrase_Extraction_and_Grounding_subset.json \
        data/chestxagent_sft_train_all_tasks_Phrase_Grounding_subset.json \
        data/chestxagent_sft_train_all_tasks_Pneumothorax_Segmentation_subset.json \
        data/chestxagent_sft_train_all_tasks_Progression_Findings_Generation_subset.json \
        data/chestxagent_sft_train_all_tasks_Progression_Impression_Generation_subset.json \
        data/chestxagent_sft_train_all_tasks_Report_Generation_subset.json \
        data/chestxagent_sft_train_all_tasks_Rib_Fracture_Segmentation_subset.json \
        data/chestxagent_sft_train_all_tasks_Temporal_Image_Classification_subset.json \
        data/chestxagent_sft_train_all_tasks_Text_QA_subset.json \
        data/chestxagent_sft_train_all_tasks_View_Classification_subset.json \
        data/chestxagent_sft_train_all_tasks_View_Matching_subset.json \
    --dataset_num_proc 8 \
    --max_pixels 262144 \
    --streaming False \
    --max_epochs 1 \
    --num_train_epochs 1 \
    --split_dataset_ratio 0.0 \
    --attn_impl flash_attn \
    --padding_free true \
    --deepspeed zero3 \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size 8 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps $((256 / ($NUM_GPUS * $per_device_train_batch_size))) \
    --save_steps 1000 \
    --save_total_limit 2 \
    --logging_steps 100 \
    --max_length 4096 \
    --output_dir output/sft_chexone_qwen25vl3b \
    --report_to wandb \
    --run_name chexagent_r1_qwen25vl3b_newgrounding \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --freeze_llm False \
    --freeze_vit False \
    --freeze_aligner False

    # --resume_from_checkpoint output/sft_qwenvl3b_16million_nostream_debug/v0-20250829-195104/checkpoint-500 \
# --ignore_data_skip true ## 会跳过skip, 多跑一些数据
    # --use_hf True \