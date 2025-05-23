set -x

# Base model and reward model paths
# Using a smaller model for testing purposes.
# If agent_func_path is used, reward_pretrain might be overridden or not used.
PRETRAIN_MODEL_PATH="facebook/opt-125m"
# For the critic, it often shares the same architecture as the actor or reward model.
# If reward_pretrain is not used by the PPO script when agent_func_path is active,
# this critic_pretrain might need to be the same as PRETRAIN_MODEL_PATH or a compatible model.
CRITIC_PRETRAIN_MODEL_PATH="facebook/opt-125m" 

python3 -m openrlhf.cli.train_ppo_ray \
   --async_train \
   --agent_func_path openrlhf/envs/multiturn_env.py \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --pretrain $PRETRAIN_MODEL_PATH \
   --critic_pretrain $CRITIC_PRETRAIN_MODEL_PATH \
   --save_path ./ckpt/ppo_multiturn_env \
   --ckpt_path ./ckpt/ppo_multiturn_env/ckpt \
   --save_hf_ckpt \
   --micro_train_batch_size 2 \
   --train_batch_size 16 \
   --micro_rollout_batch_size 2 \
   --rollout_batch_size 4 \
   --n_samples_per_prompt 1 \
   --max_epochs 1 \
   --prompt_max_len 256 \
   --max_samples 1000 \
   --generate_max_len 256 \
   --zero_stage 2 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data openrlhf/datasets/multiturn_math_prompts.jsonl \
   --input_key input \
   --normalize_reward \
   --gradient_checkpointing

   # --colocate_all_models (Already commented, behavior changes with async, kept as is for now)
   # --vllm_gpu_memory_utilization 0.5 \ 
   # --reward_pretrain $PRETRAIN_MODEL_PATH \ 
   # --apply_chat_template \ 
   # --packing_samples \ 
   # --vllm_sync_backend nccl \ 
   # --enforce_eager \ 
   # --vllm_enable_sleep (Already commented, and must be for async)
   # --deepspeed_enable_sleep (Already commented, preferred for async)

echo "Training with Math Environment Prompts. To use Function Calling:"
echo "1. Change ENVIRONMENT_MODE to 'function_calling' in openrlhf/envs/multiturn_env.py"
echo "2. Change --prompt_data to openrlhf/datasets/multiturn_function_calling_prompts.jsonl in this script."
