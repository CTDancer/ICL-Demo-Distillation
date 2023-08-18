export CUDA_VISIBLE_DEVICES=0,1

python inference.py \
--random_seed=42 \
--dataset="gsm8k" \
--model="gpt-3.5-turbo" \
--trainset_path="./dataset/GSM8K/train.jsonl" \
--demo_path="./distilled_demos/1.2 gsm8k_Llama-2-7b-chat-hf_4_2_trainsplit_24.txt" \
--max_tokens=1024 --api_time_interval=3 --temperature=0.7 \
--multipath=1 \
--qes_limit=0