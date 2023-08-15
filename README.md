# ICL-Demo-Distillation
![image](https://github.com/CTDancer/ICL-Demo-Distillation/assets/89793506/f5f0ed03-d97a-4be9-84e6-c8b855993ba7)

目标：让GPT帮我们蒸馏demonstrations（直接在文本上进行edit/rephrase），且蒸馏完的demonstrations能够泛化给其他input and task

价值：
1. 蒸馏得到的结果语义上可解释
2. 蒸馏成本低
3. 便于保留核心逻辑，并去除冗余信息，这样更能泛化（直觉上应该是这样）

思路：
1. 自行构建一组prompts，能够让LLM能够自行edit/compress/rephrase整个demonstration，使得在缩短整个token长度的同时保留最核心的信息和逻辑
2. 给定demostrations，选一个prompt给GPT，让它压缩得到distilled demonstration
3. 将distilled demonstration和input给另一个GPT，让它生成output
4. 将这个output和标签ground truth给reward model计算reward，根据reward：
  1. 如果reward达到某个标准，就保留这步distillation，再选一个prompt，让GPT进一步压缩这个distilled demonstration
  2. 如果没有达到这个标准，就舍弃这步distillation，再选另一个prompt，让它压缩

需要考虑的具体细节：
1. prompts如何构建？
2. 每次的prompt应该如何选取？
3. 判断reward是否达标的标准应该如何选取？
4. 根据同一个prompt，同一个demonstration，GPT每次可能会给出不同的答案，其中可能带有不稳定性，是否会产生影响？
5. 如何决定整个蒸馏过程是否应该终止？
