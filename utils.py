import random
import sys
import json
import re
import time
import pdb

import numpy as np
import torch
import openai

API_KEY = " "
# define for no solution if GPT cannot generate a valid solution
# here define a magic number for the convenience of variance calculation
NO_SOLUTION = '-10086'


# set the random seed for reproducibility
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# pass in a prompt and returns a response body contains response
def GPT3_request(model:str, input_prompt:list, max_tokens:int, time_interval, temperature=0.7, stop=None):
    resp = None
    done = False
    while not done:
        try:
            openai.api_key = API_KEY
            resp = openai.Completion.create(
                model=model,
                prompt=input_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop = stop
            )
            done = True
        except:
            errno = sys.exc_info()[:2]
            if errno[0] == openai.error.InvalidRequestError:
                print(f"Invalid Request\nPrompt: {input_prompt}\n")
                print(f"Reason: {errno[1]}")
                assert False
            else:
                print(f"Error: {errno[0]}\n")
                print(f"Reason: {errno[1]}\n")
        # pause between each request to avoid rate limit
        time.sleep(time_interval)
    return resp

def GPT3_5_request(model:str, messages:list, max_tokens:int, time_interval=2, temperature=0.7, stop=None):
    ''''''
    API_KEY = "sk-AAMOOJ4kAVI8NeZKE066De9947874dF39aD8C804Dd89Be38"    # for api.dqwang.group
    resp = None
    done = False
    while not done:
        try:
            openai.api_key = API_KEY
            openai.api_base = "https://api.dqwang.group/v1"
            resp = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop = stop
            )
            done = True
        except:
            errno = sys.exc_info()[:2]
            if errno[0] == openai.error.InvalidRequestError:
                print(f"Invalid Request\nPrompt: {messages}\n")
                print(f"Reason: {errno[1]}")
                assert False
            else:
                print(f"Error: {errno[0]}\n")
                print(f"Reason: {errno[1]}\n")
        # pause between each request to avoid rate limit
        time.sleep(time_interval)
    
    # print("response is: ", resp)
    return resp['choices'][0]['message']['content']


def load_data(args):
    questions = []
    answers = []
    decoder = json.JSONDecoder()

    if args.dataset == "gsm8k":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                questions.append(json_res["question"].strip())
                answers.append(json_res["answer"].split("#### ")[-1].replace(",", ""))
    elif args.dataset == "aqua":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                qes = json_res["question"].strip() + " Answer Choices:"

                for opt in json_res["options"]:
                    opt = opt.replace(')', ') ')
                    qes += f" ({opt}"

                questions.append(qes)
                answers.append(json_res["correct"])
    elif args.dataset == "svamp":
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["Body"].strip() + " " + line["Question"].strip()
                a = str(line["Answer"])
                if a[-2:] == ".0":
                    a = a[:-2]
                questions.append(q)
                answers.append(a)
    elif args.dataset == "asdiv":
        with open(args.dataset_path) as f:
            json_data = json.load(f)["Instances"]
            for line in json_data:
                q = line['input'].strip()
                a = line['output'][0]
                questions.append(q)
                answers.append(a)
    elif args.dataset in ("addsub", "singleeq", "multiarith"):
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["sQuestion"].strip()
                a = str(line["lSolutions"][0])
                if a[-2:] == ".0":
                    a = a[:-2]
                questions.append(q)
                answers.append(a)
    elif args.dataset == "csqa":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                choice = "Answer Choices:"
                for c in json_res["question"]["choices"]:
                    choice += " ("
                    choice += c["label"]
                    choice += ") "
                    choice += c["text"]
                questions.append(json_res["question"]["stem"].strip() + " " + choice)
                answers.append(json_res["answerKey"])
    elif args.dataset == "strategyqa":
        if 'task' in args.dataset_path:
            with open(args.dataset_path) as f:
                json_data = json.load(f)["examples"]
                for line in json_data:
                    q = line["input"].strip()
                    a = int(line["target_scores"]["Yes"])
                    if a == 1:
                        a = "yes"
                    else:
                        a = "no"
                    questions.append(q)
                    answers.append(a)
        else:
            with open(args.dataset_path, encoding='utf-8') as f:
                json_data = json.load(f)
                for line in json_data:
                    q = line["question"].strip() 
                    if line['answer']:
                        a = 'yes'
                    else:
                        a = 'no'
                    questions.append(q)
                    answers.append(a)
    elif args.dataset in ("coin_flip", "last_letters"):
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            json_data = json_data["examples"]
            for line in json_data:
                q = line["question"]
                a = line["answer"]
                questions.append(q)
                answers.append(a)
    else:
        raise NotImplementedError

    print(f"dataset: {args.dataset}")
    print(f"dataset_size: {len(answers)}")
    args.dataset_size = len(answers)
    return questions, answers


def create_dataloader(args) -> list:
    '''Not a PyTorch dataloader. It supprts random index(slice) access'''
    set_random_seed(args.random_seed)
    questions, answers = load_data(args)
    dataset = []
    for idx in range(len(questions)):
        dataset.append({"question":questions[idx], "answer":answers[idx], "question_idx":idx})

    random.shuffle(dataset)
    print(f"dataloader size: {len(dataset)}")
    return dataset


def create_input_prompt(args, qa_pairs, val_flag:bool)->str:
    '''return the combination of validation prompts and already-selected prompts'''
    x, y = [], []
    if val_flag:
        with open(args.prompt_path, encoding="utf-8") as f:
            json_data = json.load(f)
            json_data = json_data["prompt"]
            for line in json_data:
                x.append(line["question"])
                y.append(line["pred_ans"])
            if qa_pairs:
                for qa_pair in qa_pairs:
                    x.append(qa_pair["question"])
                    y.append(qa_pair["answer"])  
    else:
        if qa_pairs:
            for qa_pair in qa_pairs:
                x.append(qa_pair["question"])
                y.append(qa_pair["answer"])          

    index_list = list(range(len(x)))

    prompt_text = ""
    for i in index_list:
        prompt_text += x[i] + " " + y[i] + "\n\n"
    return prompt_text


def answer_extraction(args, responses):
    pred_ans = ""
    temp = responses
    if args.dataset in ("gsm8k", "svamp", "asdiv", "addsub", "singleeq", "multiarith"):
        temp = temp.replace(",", "")
        temp = [s for s in re.findall(r'-?\d+\.?\d*', temp)]
    elif args.dataset in ("aqua", "csqa"):
        temp = re.findall(r'A|B|C|D|E', temp)
    elif args.dataset in ("strategyqa", "coin_flip"):
        temp = temp.lower()
        temp = re.sub("\"|\'|\n|\.|\s|\:|\,"," ", temp)
        temp = temp.split(" ")
        temp = [i for i in temp if i in ("yes", "no")]
    elif args.dataset in ("last_letters"):
        temp = re.sub("\"|\'|\n|\.|\s","", temp)
        temp = [temp]
    
    if len(temp) != 0:
        answer = temp[-1]
        # if there is . at the end of answer, remove it
        # e.g. answer = 64.
        if answer != "":
            if answer[-1] == ".":
                answer = answer[:-1]

        # round the answer to nearest integer
        if args.dataset in ("gsm8k", "svamp"):
            try:
                answer = str(round(float(answer)))
            except:
                answer = "" # no sol or sol doesn't have valid format
        elif args.dataset in ("last_letters"):
            try:
                answer = answer[-args.concat_length:]
            except:
                answer = ""
        pred_ans = answer
    else:
        pred_ans = ""
    return pred_ans


def create_gpt_test_input_prompt(args) -> str:
    x, y, z = [], [], []
    with open(args.selected_prompt_path, encoding="utf-8") as f:
        json_data = json.load(f)
        for line in json_data:
            z.append(line["dataset_idx"])

        with open(args.trainset_path, encoding="utf-8") as f2:
            for z_val in z:
                f2.seek(0)  # redirect fp to the beginning of file
                for i, line in enumerate(f2):
                    json_data = json.loads(line)
                    if i == z_val:
                        x.append(json_data["question"])
                        combine = json_data["answer"].split("\n")
                        one_prompt = ". ".join(combine)
                        one_prompt = one_prompt.replace('####', 'Therefore the answer is')
                        # one_prompt = one_prompt.replace('$', '')
                        y.append(one_prompt)

    index_list = list(range(len(x)))
    prompt_text = ""

    for i in index_list:
        prompt_text += "Q: " + x[i] + "\n"+ "A: "  + y[i] + "." + "\n\n"

    prompt_text = re.sub(r'<<.*?>>', '', prompt_text)   # delete calculator annotation

    return prompt_text

def get_qas(demo_path):
    ''' get the question list and answer list from the given file '''
    
    with open(demo_path, encoding="utf-8") as f:
        qas = json.load(f)
        questions = [qa["question"] for qa in qas]
        answers = [qa["answer"] for qa in qas]
    
    return questions, answers

def get_demos(questions=None, answers=None):
    ''' format the demonstration '''
    
    assert len(questions) == len(answers), "number of questions should be equal to number of answers"
    
    demonstrations = ""
    for i in range(len(questions)):
        demonstrations += ( "Question: " + questions[i] + "\n" + "Answer: " + answers[i] + "\n" )
        
    return demonstrations

def get_prompts():
    ''' define the prompts for distillation '''
    
    prompts = []
    prompts.append("Rephrase or edit the text above so as to delete any unimportant information as possible and to shorten its total length. Thank you.")
    
    prompts.append("Edit the given text to remove any redundant or repetitive information, ensuring the core message remains intact. Thank you.")
    
    prompts.append("Reduce wordiness in the text, making it more concise without losing its essential meaning. Thank you.")
    
    prompts.append("Craft a succinct version of the text that omits redundant information while retaining its core essence. Thank you.")
    
    prompts.append("Revise the text using abbreviations and shortening where appropriate, ensuring the essential details remain intact. Thank you.")
    
    return prompts

def select_prompt(prompts, used_index, done):
    ''' select a prompt from prompts '''

    candidates = [x for x in list(range(len(prompts))) if x not in used_index]
    if len(candidates) == 0:
        done = True
        return None, done
    
    index = random.choice(candidates)    
    return prompts[index], done

def sample(dataloader, args):
    ''' randomly sample a question-answer pair from the dataloader '''
    
    set_random_seed(args.random_seed)
    random.shuffle(dataloader)
    
    return dataloader[0]['question'], dataloader[0]['answer']

def compute_distance(prediction, answer):
    ''' compute how much does the prediction differ from the answer '''
    
    pass

