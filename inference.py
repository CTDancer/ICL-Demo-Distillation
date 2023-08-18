import time
import argparse
import json

from transformers import pipeline
from pathlib import Path
from tqdm import tqdm

import utils


def main():
    args = arg_parser()
    print('*****************************')
    print(args)
    print('*****************************')

    utils.set_random_seed(args.random_seed)

    dataloader = utils.create_dataloader(args)
    
    initial_prompt = "Follow the given examples and answer the final question step by step.\
        Note that the last sentence in your response can ONLY start with `Therefore the answer is:`, \
        and provide the numerical answer to the final answer.\n"
    
    try:
        with open(args.demo_path, "r") as file:
            demo = file.read()
    except FileNotFoundError:
        print("Your demo path doesn't exist. Please try another path.")
        
    correct = 0
    
    for count, qa in enumerate(dataloader):
        if args.qes_limit is not None and count+1 == args.qes_limit:
            break
        predictions = []
        messages_for_inference = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": (initial_prompt + demo + qa['question'])}
        ]
        for i in range(0, args.multipath):
            prediction = utils.GPT3_5_request(
                model=args.model, 
                messages=messages_for_inference,
                max_tokens=args.max_tokens,
                time_interval=args.api_time_interval,
                temperature=args.temperature
            )
            last_line = prediction.split('\n')[-1]
            print(f"prediction is: {last_line}")
            prediction = utils.answer_extraction(args, prediction).lstrip()
            predictions.append(prediction)
        prediction = max(predictions, key=predictions.count)
        print(f"Extracted answer: {prediction}")
        print(f"Ground Truth: {qa['answer']}")
        print("**************************")
        
        if prediction == qa['answer']:
            correct += 1
        
    print(f"Total correct number: {correct}")
    print(f"Percentage: {correct / (count+1)}%")


def arg_parser():
    parser = argparse.ArgumentParser(description="Inference with selected prompts.")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--dataset", type=str, default="gsm8k", choices=["gsm8k","svamp", "aqua", "csqa", "asdiv", "last_letters", "addsub", "singleeq", "strategyqa", "multiarith"], help="dataset to inference"
    )
    parser.add_argument(
        "--trainset_path", type=str, default="./dataset/GSM8K/train.jsonl", help="prompts to use"
    )
    parser.add_argument(
        "--demo_path", type=str, default="./distilled_demos/gsm8k_Llama-2-13b-chat-hf_4_2_trainsplit_42.txt", help="path to distilled demos"
    )
    parser.add_argument(
        "--selected_prompt_path_from_APS", type=str, default="./logdifference_results/gsm8k_k=10.txt", help="selected prompts from APS"
    )
    parser.add_argument(
        "--run_APS_baseline", action='store_true', help="whether to run APS_baseline function"
    )
    parser.add_argument(
        "--APS_index", type=int, default=0, help="The index for multiple APS_baseline experiments"
    )
    parser.add_argument(
        "--multiple_prompting_rounds", action='store_true', help="how to format `messages` for openai ChatCompletion"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-3.5-turbo", choices=["gpt-3.5-turbo"], help="model used for decoding."
    )
    parser.add_argument(
        "--QA_dir", type=str, default="./QA_records/", help="output directory for QA records"
    )
    parser.add_argument(
        "--wrong_list_dir", type=str, default="./wrong_lists/", help="output directory for wrong lists"
    )
    parser.add_argument(
        "--max_length_cot", type=int, default=512, help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--qes_limit", type=int, default=10, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=15, help="how many seconds to sleep between each request"
    )
    parser.add_argument(
        "--temperature", type=float, default=0, help=""
    )
    parser.add_argument(
        "--max_tokens", type=int, default=1024, help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--multipath", type=int, default=1, help="self-consistency path num"
    )
    parser.add_argument(
        "--concat_length", type=int, default=4, help='Used for task last_letters, indicates length of last letter to concat, i.e. Elon Musk -> nk, use concat length of 2'
    )
    parser.add_argument(
        "--use_code_style_prompt", type=bool, default=False, help='Use code-style prompt as mentioned in paper for last_letters dataset'
    )

    args = parser.parse_args()

    if args.multipath > 1:
        args.temperature = 0.7
    else:
        args.temperature = 0
    print(f"Temperature: {args.temperature}")

    if args.dataset == "gsm8k":
        args.dataset_path = "./dataset/GSM8K/test.jsonl"
    elif args.dataset == "svamp":
        args.dataset_path = "./dataset/SVAMP/SVAMP.json"
    elif args.dataset == "asdiv":
        args.dataset_path = "./dataset/ASDiv/ASDiv.json"
    elif args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/test.json"
    elif args.dataset == "csqa":
        args.dataset_path = "./dataset/CSQA/dev_rand_split.jsonl"
    elif args.dataset == "strategyqa":
        args.dataset_path = "./dataset/strategyQA/task.json"
    elif args.dataset == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters_test.json"
    elif args.dataset == "addsub":
        args.dataset_path = "./dataset/MAWPS/AddSub.json"
    elif args.dataset == "singleeq":
        args.dataset_path = "./dataset/MAWPS/SingleEq.json"
    elif args.dataset == "multiarith":
        args.dataset_path = "./dataset/MAWPS/MultiArith.json"
    else:
        raise ValueError("dataset is not properly defined ...")

    return args


if __name__ == "__main__":
    main()