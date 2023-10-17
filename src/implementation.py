import openai
import pyniryo
import time
from typing import List

NAGA_AI_BASE = ""
NAGA_AI_KEY = ""

openai.api_base = NAGA_AI_BASE
openai.api_key = NAGA_AI_KEY
# from vision_funcs import getObjectLocation, build_scene_description, findObjectInScene, Location
from program_prompt import PROMPT

ROBOT_IP_ADDRESS = "10.10.10.10"

def connectRobot(robot_ip_address):
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
        return Robot
    except Exception as e:
        print("You are not connectied to the same network as the robot")
        raise e

def closeRobotConnection(robot: pyniryo.NiryoRobot):
    robot.close_connection()
    print("Robot connection closed successfully")
    

    
# Robot = connectRobot(ROBOT_IP_ADDRESS)

def calculate_robot_y_axis(pixel_x):
    # Calculate the y axis for the robot arm mmovement given the
    # pixel x axis
    
    # check if pixel is within the desired range
    if (pixel_x < 0 or pixel_x > 640):
        raise Exception("Pixel X is out of robot range")
    
    # calculate difference from base
    PIXEL_BASE = 0
    pixel_differnce = abs(pixel_x - PIXEL_BASE)
    
    # multiply difference by conversion factor
    robot_axis_difference = pixel_differnce * 0.00093578
    
    # add robot axis difference to robot base
    ROBOT_AXIS_BASE = 0.2858
    robot_y_axis = ROBOT_AXIS_BASE - robot_axis_difference
    return robot_y_axis


def calculate_robot_x_axis(pixel_y):
    # Calculates the x axis for the robots movement given the pixel y axis
    
    # check if pixel value is within range
    if (pixel_y < 215 or pixel_y > 415):
        raise Exception("Pixel Y is out of robot range")
    # get difference from base
    PIXEL_BASE = 415
    difference = abs(PIXEL_BASE - pixel_y)
    
    # multiply difference by conversion factor
    robot_axis_difference = difference * 0.00093578
    
    # add robot axis difference to robot axis base
    ROBOT_AXIS_BASE = 0.1495
    robot_x_axis = ROBOT_AXIS_BASE + robot_axis_difference
    return robot_x_axis
    

def getRobotPoseFromPixelValues(pixel_x, pixel_y):
    y = calculate_robot_y_axis(pixel_x)
    x = calculate_robot_x_axis(pixel_y)
    
    return pyniryo.PoseObject(
        x=x, y=y, z=0.065,
        roll=-0.1, pitch=1.57, yaw=0.0,
        )
    



# def Pick(loc: Location, shift_x=-30, shift_y=20):
#     if not loc:
#         return
#     pose = getRobotPoseFromPixelValues(loc.x+shift_x, loc.y+shift_y)
#     Robot.pick_from_pose(pose)
    
    
# def Place(loc:Location, shift_x=-30, shift_y=20):
#     if not loc:
#         return
#     pose = getRobotPoseFromPixelValues(loc.x+shift_x, loc.y+shift_y)
#     Robot.place_from_pose(pose)

  
# def PickAndPlace(loc1:Location, loc2:Location):
#     if not loc1 or not loc2:
#         print("** One of the item was not found in the location, aborting action **")
#         return
#     Pick(loc1)
#     time.sleep(3)
#     Place(loc2)
#     time.sleep(3)
#     Robot.move_to_home_pose()
#     time.sleep(3)

# def PickAndPlaceAll(locations: List[Location], loc2:Location):
#     if len(locations) == 0 or not loc2:
#         print("** One of the item was not found in the location, aborting action **")
#         return
    
#     for location in locations:
#         Pick(location)
#         time.sleep(3)
#         Place(loc2)
#         time.sleep(3)
#     Robot.move_to_home_pose()
#     time.sleep(3)
        
    
# def MoveLeft(loc1:Location, move_value = 20):
#     if not loc1:
#         print("** Item was not found in the location, aborting action **")
#         return
#     Pick(loc1)
#     time.sleep(3)
#     loc1.x += move_value
#     Place(loc1)
#     time.sleep(3)
#     Robot.move_to_home_pose()
#     time.sleep(3)


# def MoveRight(loc1:Location, move_value = 20):
#     if not loc1:
#         print("** Item was not found in the location, aborting action **")
#         return
#     Pick(loc1)
#     time.sleep(3)
#     loc1.x -= move_value
#     Place(loc1)
#     time.sleep(3)
#     Robot.move_to_home_pose()
#     time.sleep(3)
    
# def MoveLeftAll(locations:List[Location], move_value =20):
#     if len(locations) == 0:
#         print("** One of the item was not found in the location, aborting action **")
#         return

#     for location in locations:
#         Pick(location)
#         time.sleep(3)
#         location.x += move_value
#         Place(location)
#         time.sleep(3)
    
#     Robot.move_to_home_pose()
#     time.sleep(3)

# def MoveRightAll(locations:List[Location], move_value =20):
#     if len(locations) == 0:
#         print("** One of the item was not found in the location, aborting action **")
#         return

#     for location in locations:
#         Pick(location)
#         time.sleep(3)
#         location.x -= move_value
#         Place(location)
#         time.sleep(3)
    
#     Robot.move_to_home_pose()
#     time.sleep(3)
    

def prepare_prompt(command, scene_description=None):
    full_prompt = PROMPT.replace("INSERT TASK HERE", command)
    if scene_description:
        full_prompt = full_prompt.replace("Scene: In the scene there are:",
                                          f"Scene: In the scene there are: {scene_description}")
    else:
        full_prompt = full_prompt.replace("Scene: In the scene there are:",
                                          "")
    return full_prompt

def generate_response(prompt, additional_messages=[]):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[{"role": "system", "content": "You are an assistive arm robot, you are to generate instructions in the syle of the examples provided, do not create any new functions or use for loops."},
                  {"role": "user", "content": prompt}] + additional_messages,
        temperature=0.99,
        max_tokens=512,
        n=1,
    )
    result = response["choices"][0]["message"]["content"]
    return result
    
while True:
    command = input("Enter robot command: ")
    # print command
    print("You have entered the command: ", command)
    # use LLM to generate goal and intermediate plans to accomplish set goal
    # TODO scene_description = build_scene_description()
    scene_description = "There are 4 blocks in the scene \n 2 red blocks \n 2 green blocks, there is 1 bowl in the scene \n 1 blue bowl"
    print("This is the description of the scene: ", scene_description)
    # request the user to know if we should add the scene description to the scene
    resp = input("Should we add the scene description to the prompt y or n?")
    if resp != 'y':
        scene_description=None
    full_prompt = prepare_prompt(command, scene_description)
    # generate robot execution plan
    result = generate_response(full_prompt)
    # confirm if robot should execute the plan by the user
    print("**AI GENERATED ROBOT PLAN**")
    print(result)
    resp = input("would you like to execute this plan? y or n")
    if resp == 'y':
        try:
            print("** EXECUTING AI GENERATED ROBOT PLAN** \n")
            # TODO exec(result)
            print("** AI GENERATE ROBOT PLAN EXECUTED SUCCESSFULLY **")
        except Exception as e:
            print("** Failed to execute Robot plan with exception: %s" % e)
        
    else:
        # request user to provide feedback for the robot
        resp = input("Would you like to provide feedback for the robot? y or n")
        if resp != 'y':
            continue
        else:
            resp = input("Please enter the feedback for the robot \n")
            additional_messages = [
                {"role": "assistant", "content":result },
                {"role": "user", "content": resp}
            ]
            result =  generate_response(result, additional_messages)
            print("**AI GENERATED ROBOT PLAN** \n")
            print(result)
            resp = input("would you like to execute this plan? y or n")
            if resp != 'y':
                try:
                    print("** EXECUTING AI GENERATED ROBOT PLAN**")
                    # TODO exec(result)
                    print("** AI GENERATE ROBOT PLAN EXECUTED SUCCESSFULLY **")
                except Exception as e:
                    print("** Failed to execute Robot plan with exception: %s" % e)
            else:
                print("** Seems like you are going to have to request a different command")
  