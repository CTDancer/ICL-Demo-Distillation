export CUDA_VISIBLE_DEVICES=0,1

python distill.py \
--random_seed=42 \
--dataset="gsm8k" \
--model="gpt-3.5-turbo" \
--trainset_path="./dataset/GSM8K/train.jsonl" \
--demo_path="./logdifference_results/gsm8k_Llama-2-13b-chat-hf_4_2_trainsplit_42.txt" \
--max_tokens=1024 --api_time_interval=2 --temperature=0.7 \
--multipath=1 