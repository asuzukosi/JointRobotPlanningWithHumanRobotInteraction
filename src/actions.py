"""
This sections define the functions that are used to control the robots actions
"""
import pyniryo
import time
from vision import Location
from typing import List


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
    z = 0.063 if action == "pick" else 0.1
    
    return pyniryo.PoseObject(
        x=x, y=y, z=z,
        roll=-0.1, pitch=1.57, yaw=0.0,
        )


def Pick(loc: Location, shift_x=0, shift_y=33):
    if not loc:
        return
    pose = getRobotPoseFromPixelValues(loc.x+shift_x, loc.y+shift_y, "pick")
    Robot.pick_from_pose(pose)
    
    
def Place(loc:Location, shift_x=0, shift_y=33):
    if not loc:
        return
    pose = getRobotPoseFromPixelValues(loc.x+shift_x, loc.y+shift_y, "place")
    Robot.place_from_pose(pose)

  
def PickAndPlace(loc1:Location, loc2:Location):
    if not loc1 or not loc2:
        print("** One of the item was not found in the location, aborting action **")
        return
    Pick(loc1)
    time.sleep(1)
    Robot.move_to_home_pose()
    time.sleep(1)
    Place(loc2)
    time.sleep(1)
    Robot.move_to_home_pose()
    time.sleep(1)

def PickAndPlaceAll(locations: List[Location], loc2:Location):
    if len(locations) == 0 or not loc2:
        print("** One of the item was not found in the location, aborting action **")
        return
    for location in locations:
        Pick(location)
        time.sleep(1)
        Robot.move_to_home_pose()
        time.sleep(1)
        Place(loc2)
        time.sleep(1)
    Robot.move_to_home_pose()
    time.sleep(1)
        
    
def MoveLeft(loc1:Location, move_value = 0.2):
    if not loc1:
        print("** Item was not found in the location, aborting action **")
        return
    Pick(loc1)
    time.sleep(1)
    loc1.x += move_value
    Place(loc1)
    time.sleep(1)
    Robot.move_to_home_pose()
    time.sleep(1)


def MoveRight(loc1:Location, move_value = 0.2):
    if not loc1:
        print("** Item was not found in the location, aborting action **")
        return
    Pick(loc1)
    time.sleep(1)
    loc1.x -= move_value
    Place(loc1)
    time.sleep(1)
    Robot.move_to_home_pose()
    time.sleep(1)
    
def MoveLeftAll(locations:List[Location], move_value=0.2):
    if len(locations) == 0:
        print("** One of the item was not found in the location, aborting action **")
        return

    for location in locations:
        Pick(location)
        time.sleep(1)
        location.x += move_value
        Place(location)
        time.sleep(1)
    
    Robot.move_to_home_pose()
    time.sleep(1)

def MoveRightAll(locations:List[Location], move_value =20):
    if len(locations) == 0:
        print("** One of the item was not found in the location, aborting action **")
        return

    for location in locations:
        Pick(location)
        time.sleep(1)
        location.x -= move_value
        Place(location)
        time.sleep(1)
    
    Robot.move_to_home_pose()
    time.sleep(1)