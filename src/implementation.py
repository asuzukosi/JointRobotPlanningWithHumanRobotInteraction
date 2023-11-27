# import required packages
import openai
import pyniryo
import time
from typing import List
from vision_funcs import Location, getAllObjectLocation, getObjectLocation, get_camera_image

# configure openai key and informaation
NAGA_AI_BASE = "https://api.naga.ac/v1"
NAGA_AI_KEY = "VN7eDdNzbkQkrmEmIr1Gj1Kci3Ed_g6a_atrW14jq6c"

openai.api_base = NAGA_AI_BASE
openai.api_key = NAGA_AI_KEY

# import vision tools and prompt
# from vision_funcs import build_scene_description, Location
from program_prompt import SYSTEM_PROMPT

# configure the robot IP address
ROBOT_IP_ADDRESS = "172.20.10.3"

# connect to the robot though its IP address
def connectRobot():
    try:
        Robot = pyniryo.NiryoRobot(ROBOT_IP_ADDRESS)
        Robot.calibrate(calibrate_mode=pyniryo.CalibrateMode.AUTO)
        hardware_status = Robot.get_hardware_status()
        print("Robot hardware status: \n")
        print(hardware_status)
        Robot.update_tool()
        tool = Robot.get_current_tool_id()
        print(tool)
        
        print("Moving robot to home pose...")
        Robot.set_learning_mode(False)
        # adjust the pose to move to the home location
        Robot.move_to_home_pose()
        # set the leanring mode back to true to allow external adjustment
        Robot.set_learning_mode(True)
        print("Done connecting to robot")
        return Robot
    except Exception as e:
        print("You are not connectied to the same network as the robot")
        raise e

# close the connection to the robot
def closeRobotConnection(robot: pyniryo.NiryoRobot):
    robot.close_connection()
    print("Robot connection closed successfully")
    

Robot: pyniryo.NiryoRobot = connectRobot()
print("Robot connection complete")

def calculate_robot_x_axis(pixel_y, pixel_y_base=50, 
                           robot_x_base=0.1413):
    """
    This will take the y axis of the pixel image and calculate the robot x axis
    The pixel y base is the base / minimum value which the robot can reach this should be set at the beginning of the experiment
    THe robot x base is robot x pose value for the provided pixel base
    """
    CONVERSTION_VALUE = 0.0009135
    # check if pixel value is within range
    if (pixel_y < pixel_y_base or pixel_y > 315):
        raise Exception("Pixel Y is out of robot range")
    # get difference from base
    difference = abs(pixel_y - pixel_y_base)
    
    # multiply difference by conversion factor
    robot_axis_difference = difference * CONVERSTION_VALUE
    
    # add robot axis difference to robot axis base
    robot_x_axis = robot_x_base + robot_axis_difference
    return robot_x_axis

def calculate_robot_y_axis(pixel_x, 
                           pixel_x_base=110, 
                           robot_y_base=-0.2440):
    """
    This will take the x axis of the pixel image and calculate the robot y axis
    The pixel x base is the base / minimum value which the robot can reach this should be set at the beginning of the experiment
    THe robot y base is robot x pose value for the provided pixel base
    """
    
    CONVERSTION_VALUE = 0.001006968
    # check if pixel is within the desired range
    if (pixel_x < 110 or pixel_x > 640):
        raise Exception("Pixel X is out of robot range")
    
    # calculate difference from base
    pixel_difference = abs(pixel_x - pixel_x_base)
    
    # multiply difference by conversion factor
    robot_axis_difference = pixel_difference * CONVERSTION_VALUE
    
    # add robot axis difference to robot base
    robot_y_axis = robot_y_base + robot_axis_difference
    return robot_y_axis
    

def getRobotPoseFromPixelValues(pixel_x, pixel_y, action="pick"):
    y = calculate_robot_y_axis(pixel_x)
    x = calculate_robot_x_axis(pixel_y)
    z = 0.063 if action == "pick" else 0.055
    
    return pyniryo.PoseObject(
        x=x, y=y, z=z,
        roll=-0.1, pitch=1.57, yaw=0.0,
        )


def Pick(loc: Location, shift_x=0, shift_y=0):
    if not loc:
        return
    pose = getRobotPoseFromPixelValues(loc.x+shift_x, loc.y+shift_y, "pick")
    Robot.pick_from_pose(pose)
    
    
def Place(loc:Location, shift_x=0, shift_y=0):
    if not loc:
        return
    pose = getRobotPoseFromPixelValues(loc.x+shift_x, loc.y+shift_y, "place")
    Robot.place_from_pose(pose)

  
def PickAndPlace(loc1:Location, loc2:Location):
    if not loc1 or not loc2:
        print("** One of the item was not found in the location, aborting action **")
        return
    Pick(loc1)
    time.sleep(3)
    Place(loc2)
    time.sleep(3)
    Robot.move_to_home_pose()
    time.sleep(3)

def PickAndPlaceAll(locations: List[Location], loc2:Location):
    if len(locations) == 0 or not loc2:
        print("** One of the item was not found in the location, aborting action **")
        return
    for location in locations:
        Pick(location)
        time.sleep(3)
        Place(loc2)
        time.sleep(3)
    Robot.move_to_home_pose()
    time.sleep(3)
        
    
def MoveLeft(loc1:Location, move_value = 0.2):
    if not loc1:
        print("** Item was not found in the location, aborting action **")
        return
    Pick(loc1)
    time.sleep(3)
    loc1.x += move_value
    Place(loc1)
    time.sleep(3)
    Robot.move_to_home_pose()
    time.sleep(3)


def MoveRight(loc1:Location, move_value = 0.2):
    if not loc1:
        print("** Item was not found in the location, aborting action **")
        return
    Pick(loc1)
    time.sleep(3)
    loc1.x -= move_value
    Place(loc1)
    time.sleep(3)
    Robot.move_to_home_pose()
    time.sleep(3)
    
def MoveLeftAll(locations:List[Location], move_value=0.2):
    if len(locations) == 0:
        print("** One of the item was not found in the location, aborting action **")
        return

    for location in locations:
        Pick(location)
        time.sleep(3)
        location.x += move_value
        Place(location)
        time.sleep(3)
    
    Robot.move_to_home_pose()
    time.sleep(3)

def MoveRightAll(locations:List[Location], move_value =20):
    if len(locations) == 0:
        print("** One of the item was not found in the location, aborting action **")
        return

    for location in locations:
        Pick(location)
        time.sleep(3)
        location.x -= move_value
        Place(location)
        time.sleep(3)
    
    Robot.move_to_home_pose()
    time.sleep(3)
    

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

def shouldAddSceneDescription():
    resp = input("Should we add the scene description to the prompt y or n?")
    if resp != 'y':
        return False
    return True

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
        

def generate_response():
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=MESSAGE_STREAM,
        temperature=0.99,
        max_tokens=512,
        n=1,
    )
    result = response["choices"][0]["message"]["content"]
    return result


def performCycle():
    command = input("Enter robot command: ")
    print(f"You have enterd the instructurion '{command}'")
    # build scene description with the vision model
    # scene_description = build_scene_description()
    scene_description = "There are 4 blocks in the scene \n 1 red block \n 1 green block, \n 1 blue block \n 1 black block \n there are 2 bowls in the scene \n 1 blue bowl \n 1 green bowl"
    print(scene_description)
    scene_description = scene_description if shouldAddSceneDescription() else None
    message = prepare_message(command, scene_description)
    addMessage(message)    
    while True:
        exec_response = generate_response()
        print("**AI GENERATED ROBOT PLAN**")
        print(exec_response)
        resp = input("would you like to execute this plan? y or n")
        if resp == 'y':
            try:
                print("** EXECUTING AI GENERATED ROBOT PLAN** \n")
                exec(exec_response)
                print("** AI GENERATE ROBOT PLAN EXECUTED SUCCESSFULLY **")
                return
            except Exception as e:
                print("** Failed to execute Robot plan with exception: %s" % e)
                addMessage({"role": "assistant", "content": exec_response })
                addMessage({"role": "system", "content": f"you failed to execute the exception {e}"})
        else:
            resp = input("Would you like to provide feedback for the robot? y or n")
            if resp != 'y':
                removeLastMessage()
                return
            else:
                resp = input("Please enter the feedback for the robot \n")
                addMessage({"role": "assistant", "content": exec_response })
                addMessage({"role": "user", "content": resp})
            

def startInteraction():
    # print start message
    print("Hello, Welcome to Ned Interactive. An interactive robot to execute tasks")
    while True:
        performCycle()



# startInteraction()
# closeRobotConnection()
Robot.set_learning_mode(True)
