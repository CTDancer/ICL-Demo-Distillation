import argparse
import torch
import json
import time
import os
import pdb
import utils

def main():
    args = arg_parser()
    print('*****************************')
    print(args)
    print('*****************************')
    utils.set_random_seed(args.random_seed)
    used_index = []
    done = False
    select_prompt = True
    
    dataloader = utils.create_dataloader(args)
    if args.raw is True:
        print(args.raw)
        questions, answers = utils.get_qas(args.demo_path)
        previous_demos = utils.get_demos(questions, answers)
    else:
        with open(args.demo_path, "r") as file:
            previous_demos = file.read()
            
    initial_prompt = "Follow the given examples and answer the final question step by step.\
        Note that the last sentence in your response can ONLY start with `Therefore the answer is:`, \
        and provide the numerical answer to the final answer.\n"
    prompts = utils.get_prompts()
    question, answer = utils.sample(dataloader, args)
    print("question is: ", question)
    print("answer is: ", answer)
    print(f"previous demo: {previous_demos}\n")
    previous_length = len(previous_demos.split())
    
    while True:
        prompt, done = utils.select_prompt(prompts, used_index, done)
        if done:
            break
        print(f"prompt is: {prompt}\n")
        messages_for_distillation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": (previous_demos + prompt)}
        ]
        # pdb.set_trace()
        distilled_demos = utils.GPT3_5_request(
            model=args.model, 
            messages=messages_for_distillation,
            max_tokens=args.max_tokens,
            time_interval=args.api_time_interval,
            temperature=args.temperature
        )
        print(f"distilled_demos is: {distilled_demos}\n")
        distilled_length = len(distilled_demos.split())
        print(f"previous demo length: {previous_length}")
        print(f"distilled demo length: {distilled_length}\n")
        if distilled_length >= previous_length:
            used_index.append(prompts.index(prompt))
            continue
        # 将出现次数最多的答案当成预测结果
        predictions = []
        messages_for_inference = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": (initial_prompt + distilled_demos + question)}
        ]
        for i in range(0, args.multipath):
            prediction = utils.GPT3_5_request(
                model=args.model, 
                messages=messages_for_inference,
                max_tokens=args.max_tokens,
                time_interval=args.api_time_interval,
                temperature=args.temperature
            )
            print(f"prediction is: {prediction}\n")
            prediction = utils.answer_extraction(args, prediction).lstrip()
            print(f"Extracted answer: {prediction}\n")
            predictions.append(prediction)
            print("**************************")
        prediction = max(predictions, key=predictions.count)
        # reward = utils.compute_distance(prediction, answer)
        # if reward > args.criterion:
        #     previous_demos = distilled_demos
        if prediction == answer:
            print("yes")
            previous_demos = distilled_demos
            previous_length = distilled_length
        else:
            used_index.append(prompts.index(prompt))
            
    dest = os.path.join(args.save_path, args.demo_path.split('/')[-1])
    if os.path.exists(dest):
        with open(dest, "r") as file:
            demos = file.read()
            
        words_count = len(demos.split())
    
    if os.path.exists(dest) and words_count < previous_length:
        print("It is not shorter than the previous distilled version.")
    else:
        with open(dest, "w") as file: 
            file.write(previous_demos)

def arg_parser():
    parser = argparse.ArgumentParser(description="Inference with selected prompts.")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--dataset", type=str, default="gsm8k", choices=["gsm8k","svamp", "aqua", "csqa", "asdiv", "last_letters", "addsub", "singleeq", "strategyqa", "multiarith"], help="dataset to inference"
    )
    parser.add_argument(
        "--trainset_path", type=str, default="./dataset/GSM8K/train.jsonl", help="prompts to use"
    )
    parser.add_argument(
        "--demo_path", type=str, default="./logdifference_results/gsm8k_baichuan7b_8-1_trainsplit-val.txt", help="path to demos"
    )
    parser.add_argument(
        "--save_path", type=str, default="./distilled_demos", help="path to save demos"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-3.5-turbo", help="model used for decoding."
    )
    parser.add_argument(
        "--output_dir", type=str, default="./QA_records/", help="output directory for QA records"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=1024, help="maximum length of output tokens by model for reasoning extraction"
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
        "--multipath", type=int, default=1, help="self-consistency path num"
    )
    parser.add_argument(
        "--concat_length", type=int, default=4, help='Used for task last_letters, indicates length of last letter to concat, i.e. Elon Musk -> nk, use concat length of 2'
    )
    parser.add_argument(
        "--use_code_style_prompt", type=bool, default=False, help='Use code-style prompt as mentioned in paper for last_letters dataset'
    )
    parser.add_argument(
        "--raw", action='store_true', help='Use raw demonstrations or distilled demonstrations'
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