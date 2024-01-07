import os
import openai

# Set the proxy based on your environment if needed
# os.environ["http_proxy"] = "http://127.0.0.1:58591"
# os.environ["https_proxy"] = "http://127.0.0.1:58591"

# load different types of prompt
from visual_programming_prompt.object_query_prompt import PROMPT as object_query_prompt
from visual_programming_prompt.visual_programm_prompt import (
    PROMPT as visual_programm_prompt,
)

full_prompt_file = "visual_programming_prompt/full_prompt.ini"
with open(full_prompt_file, "r") as f:
    full_prompt_i2a = f.readlines()
full_prompt_i2a = "".join(full_prompt_i2a)

openai.api_key = ""
# this api key is only for demo, please use your own api key

prompt_style = "VISPROG"
prompts = {
    "instruct2act": full_prompt_i2a,
    "VISPROG": visual_programm_prompt,
}
# You can choose different prompt here
# visual_programm_prompt: VISPROG style prompt
# full_prompt_i2a: VISPROG + ViperGPT style prompt
prompt_base = prompts[prompt_style]

folder = "visual_programming_prompt/output/" + prompt_style + "/"
if not os.path.exists(folder):
    os.makedirs(folder)

def turn_list_to_string(all_result):
    if not isinstance(all_result, list):
        return all_result
    all_in_one_str = ""
    for r in all_result:
        all_in_one_str = all_in_one_str + r + "\n"
    if all_in_one_str.endswith("\n"):
        all_in_one_str = all_in_one_str[:-1]
    if all_in_one_str.endswith("."):
        all_in_one_str = all_in_one_str[:-1]
    return all_in_one_str

def result_preprocess(results):
    """
        Only used for the result with full_prompt_i2a
    """
    codes = []
    if isinstance(results, list):
        results = results[0]
    for code in results.splitlines():
        if "main" in code or len(code) < 2:
            continue
        if  "(" not in code and ")" not in code:
            continue
        if code.startswith("#"):
            continue
        codes.append(code.strip())
    
    # codes = turn_list_to_string(codes)
    return codes

def insert_task_into_prompt(task, prompt_base, insert_index="INSERT TASK HERE"):
    full_prompt = prompt_base.replace(insert_index, task)
    return full_prompt



def exec_steps(instruction_task, task_id=None):
    curr_prompt = insert_task_into_prompt(instruction_task, prompt_base)
    response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=curr_prompt,
                temperature=0.99,
                max_tokens=512,
                n=1,
                stop=".",
            )

    all_result = []
    result = response["choices"][0]["text"]
    if prompt_style == "instruct2act":
        all_result.append(result)
    else:
        all_result.append(result.replace("\n\n", ""))
        all_result = all_result[0]
        
    all_result = result_preprocess(all_result)
    all_result = turn_list_to_string(all_result)
    return all_result