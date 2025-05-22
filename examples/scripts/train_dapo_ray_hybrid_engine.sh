set -x

python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 8 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 4 \
   --vllm_tensor_parallel_size 2 \
   --colocate_all_models \
   --vllm_gpu_memory_utilization 0.6 \
   --init_kl_coef 0.0 \
   --kl_target 0.0 \
   --gamma 1.0 \
   --advantage_estimator group_norm \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --agent_func_path examples/python/agent_func.py \
   --save_path /openrlhf/examples/test_scripts/final/llama3-8b-dapo \
   --ckpt_path /openrlhf/examples/test_scripts/ckpt/llama3-8b-dapo \
   --save_hf_ckpt \
   --micro_train_batch_size 8 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 16 \
   --rollout_batch_size 128 \
   --n_samples_per_prompt 8 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --max_samples 100000 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --normalize_reward \
   --gradient_checkpointing \
   --packing_samples \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep \
   --dapo_clip_eps_low 0.2 \
   --dapo_clip_eps_high 0.28 \
   --dynamic_filtering \
   --dapo_enable_overlong_filtering \
   --enable_dapo_overlong_reward_shaping \
   --dapo_l_max 20480 \
   --dapo_l_cache 4096

# Notes on changes from GRPO script for DAPO:
# - Removed --reward_pretrain, --reward_num_nodes, --reward_num_gpus_per_node
# - Added --agent_func_path for rule-based rewards
# - Set --init_kl_coef 0.0 and --kl_target 0.0 (or remove kl_target)
# - Removed --use_kl_loss and --kl_estimator
# - Added DAPO specific arguments:
#   --dapo_clip_eps_low, --dapo_clip_eps_high
#   --dynamic_filtering (explicitly added)
#   --dapo_enable_overlong_filtering
#   --enable_dapo_overlong_reward_shaping
#   --dapo_l_max, --dapo_l_cache
# - Kept --advantage_estimator group_norm as DAPO modifies loss/rewards, not necessarily estimator type.
# - Save paths updated to reflect 'dapo'.
# - Critic model will default to --pretrain as per logic when agent_func_path is used.
# - --eps_clip is not used by PolicyLoss if dapo_clip_eps_low/high are used, so it's implicitly removed from effect.
# - --dynamic_filtering_reward_range removed as it's not used by DAPO's dynamic filtering.
# - Ensure examples/python/agent_func.py is adapted for DAPO's +1/-1 reward logic.
