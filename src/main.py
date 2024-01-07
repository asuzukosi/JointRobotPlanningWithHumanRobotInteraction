# import required packages
import openai
from typing import List

# configure openai key and informaation
NAGA_AI_BASE = "https://api.naga.ac/v1"
NAGA_AI_KEY = "VN7eDdNzbkQkrmEmIr1Gj1Kci3Ed_g6a_atrW14jq6c"

openai.api_base = NAGA_AI_BASE
openai.api_key = NAGA_AI_KEY

from src.prompt import SYSTEM_PROMPT
# import function to get camera image from camera module
from vision import *
from actions import *

detection_mode = "vild" # sam_clip

if detection_mode == "vild":
    from vild import *
if detection_mode == "sam_clip":
    from sam import *
    

def prepare_message(command, scene_description=None):
    if scene_description:
        full_prompt = f"""
    Scene: In the scene there are {scene_description}
    Instruction: {command}
    Program:
    """
    else:
        full_prompt = f"""
    Instruction: {command}
    Program:
        """
    return {
            "role": "user",
            "content": full_prompt
            }

MAX_STREAM_SIZE = 10
MESSAGE_STREAM = [{"role": "system", 
                   "content": SYSTEM_PROMPT}]

def removeFromMessageStream():
    global MESSAGE_STREAM
    MESSAGE_STREAM = MESSAGE_STREAM[:1] + MESSAGE_STREAM[2:]
    return

def clearMessageStream():
    global MESSAGE_STREAM
    MESSAGE_STREAM = MESSAGE_STREAM[:1]
    return

def removeLastMessage():
    global MESSAGE_STREAM
    MESSAGE_STREAM = MESSAGE_STREAM[:-1]
    return

def addMessage(message):
    global MESSAGE_STREAM
    if len(MESSAGE_STREAM) >= MAX_STREAM_SIZE:
        removeFromMessageStream()
    MESSAGE_STREAM.append(message)
    return


def LLMPlanGenerator(instruction):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=MESSAGE_STREAM,
        temperature=0.99,
        max_tokens=512,
        n=1,
    )
    plan = response["choices"][0]["message"]["content"]
    return plan


def RobotExecute(plan:str):
    print("** EXECUTING AI GENERATED ROBOT PLAN** \n")
    for code in plan.splitlines():
        exec(code)
    print("** AI GENERATE ROBOT PLAN EXECUTED SUCCESSFULLY **")


def RequestApproval(plan):
    """
    Request user to reject or approve LLM generated robot plan
    """
    print("**AI GENERATED ROBOT PLAN**")
    print(plan)
    resp = input("would you like to execute this plan? y or n")
    if resp == 'y':
        return True
    return False

def RequestFeedback(plan):
    resp = input("Please enter the feedback for the robot \n")
    addMessage({"role": "assistant", "content": plan })
    addMessage({"role": "user", "content": resp})
    
def Instruct2ActFeedbackVild(num_iterations: int=10):
    """
    Implemenation of our joint robot planning with human interaction
    """
    
    MaxFeedback = 3
    for i in range(0, num_iterations):
        print(f"=======Iteration {i+1} ===========")
        instruction = input("Enter robot command: ")
        print(f"You have enterd the instructurion '{instruction}'")
        image, _ = get_camera_image()
        scene_description = SceneDescription(image)
        print(scene_description)
        message = prepare_message(instruction, scene_description)
        addMessage(message)
        plan = LLMPlanGenerator(instruction)
        addMessage({"role": "assistant", "content": plan})
        InstructionApproved = RequestApproval(plan)
        NumFeedback = 0
        while (not InstructionApproved) and (NumFeedback < MaxFeedback):
            feedback = RequestFeedback(plan)
            plan = LLMPlanGenerator(feedback)
            InstructionApproved = RequestApproval(plan)
        RobotExecute(plan)

def Instruct2ActNoFeedbackVild(num_iterations:int=10):
    for i in range(0, num_iterations):
        print(f"=======Iteration {i+1} ===========")
        instruction = input("Enter robot command: ")
        print(f"You have enterd the instructurion '{instruction}'")
        image = get_camera_image()
        scene_description = SceneDescription(image)
        print(scene_description)
        message = prepare_message(instruction, scene_description)
        addMessage(message)
        plan = LLMPlanGenerator(instruction)
        print("**AI GENERATED ROBOT PLAN**")
        print(plan)
        addMessage({"role": "assistant", "content": plan})
        RobotExecute(plan)

def Instruct2ActFeedbacKSamClip(num_iterations:int=10):
    MaxFeedback = 3
    for i in range(0, num_iterations):
        print(f"=======Iteration {i+1} ===========")
        instruction = input("Enter robot command: ")
        print(f"You have enterd the instructurion '{instruction}'")
        image = get_camera_image()
        scene_description = SceneDescription(image)
        print(scene_description)
        message = prepare_message(instruction, scene_description)
        addMessage(message)
        plan = LLMPlanGenerator(instruction)
        print("**AI GENERATED ROBOT PLAN**")
        print(plan)
        addMessage({"role": "assistant", "content": plan})
        InstructionApproved = RequestApproval(plan)
        NumFeedback = 0
        while (not InstructionApproved) and (NumFeedback < MaxFeedback):
            feedback = RequestFeedback(plan)
            plan = LLMPlanGenerator(feedback)
            addMessage({"role": "assistant", "content": plan})

            InstructionApproved = RequestApproval(plan)
        # use sam models instead of clip models
        # maybe try adding the import statement here so the imports are dynamic
        RobotExecute(plan)

def Instruct2ActFeedbackVildNoSceneDescription(num_iterations:int = 10):
    MaxFeedback = 3
    for i in range(0, num_iterations):
        print(f"=======Iteration {i+1} ===========")
        instruction = input("Enter robot command: ")
        print(f"You have enterd the instructurion '{instruction}'")
        scene_description = None
        print(scene_description)
        message = prepare_message(instruction, scene_description)
        addMessage(message)
        plan = LLMPlanGenerator(instruction)
        print("**AI GENERATED ROBOT PLAN**")
        print(plan)
        addMessage({"role": "assistant", "content": plan})
        InstructionApproved = RequestApproval(plan)
        NumFeedback = 0
        while (not InstructionApproved) and (NumFeedback < MaxFeedback):
            Feedback = RequestFeedback(plan)
            plan = LLMPlanGenerator(Feedback)
            addMessage({"role": "assistant", "content": plan})

            InstructionApproved = RequestApproval(plan)
        RobotExecute(plan)

