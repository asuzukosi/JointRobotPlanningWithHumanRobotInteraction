from pyniryo import *
from typing import List

# set the robot ip address
ROBOT_IP_ADDRESS = "10.10.10.10"

# connect to the robot using the IP address
robot = NiryoRobot(ROBOT_IP_ADDRESS)
# calibrate the robot
robot.calibrate()

# move the joint to a specific joint position
robot.move_joints(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
# test how the difference in joints affects the robots movements
robot.move_joints(0.2, -0.3, 0.1, 0.0, 0.5, -0.8)

# turn on learning mode (the question is what is learning mode used for?)
robot.set_learning_mode(True)

# stop the tcp connection (Why on earth would you want to stop the connection?)
robot.close_connection()

# we can use this commnand to update the robot tool, but how does the tool update differ the logic we used in the 
# provided pick and place algorithm
robot.update_tool()

# release with robot tool, whatever that tool might be
robot.release_with_tool()

# grasp with robot tool
robot.grasp_with_tool()

# we can either use the joints as a spread list of values or we can pass in a list containing all the 6 joints
# as a spread list of values
robot.move_joints(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

# as a list of values
robot.move_joints([-0.5, -0.6, 0.0, 0.3, 0.0, 0.0])

# You can also retreive joint coordinates using getters and functions
# with function
joints_read = robot.get_joints()

# with getter
joints = robot.joints()

# we can move to the home position using the move to home pose functionn
robot.move_to_home_pose()

# we can move to a specific object using the move to object function
robot.move_to_object()

# Working with poses, you can pass the poss as a spread set of values or as a list of 6 values or as a pose object
# the pose is made up of 6 values x, y, z, roll, pitch, yaw 
pose_target = [0.2, 0.0, 0.2, 0.0, 0.0, 0.0]
pose_target_obj = PoseObject(0.2, 0.0, 0.2, 0.0, 0.0, 0.0)

# move pose with function
robot.move_pose(0.2, 0.0, 0.2, 0.0, 0.0, 0.0)
robot.move_pose(pose_target)
robot.move_pose(pose_target_obj)

# we can also move to a specific pose using setters
robot.pose = (0.2, 0.0, 0.2, 0.0, 0.0, 0.0)
robot.pose = pose_target
robot.pose = pose_target_obj

# get pose information this will return a pose object, we can write a helper function to convert a pose object to a list
# get pose using function
pose_read = robot.get_pose()

# get pose using getter
pose_read = robot.pose

# The PoseObject has a .to_list() method

# The pose object gives us some form of granular control for the pose we would like or robots to perform
'''
It also allows to create new PoseObject with some offset, much easier than copying list and editing
only 1 or 2 values. For instance, imagine that we want to shift the
place pose by 5 centimeters at each iteration of a loop, you can use the copy_with_offsets() method:
'''

# This is the initial pick pose we should always revert to
pick_pose = PoseObject(
x=0.30, y=0.0, z=0.15,
roll=0, pitch=1.57, yaw=0.0
)

# this is the place pose that is updated after every iteration of the loop
place_pose = PoseObject(
    x=0.0, y=0.2, z=0.15,
    roll=0, pitch=1.57, yaw=0.0
)
# loop through a number of times
for i in range(5):
    # the robot moves to the initial pick pose
    robot.move_pose(pick_pose)
    # the place pose is slightly modified
    new_place_pose = place_pose.copy_with_offsets(x_offset=0.05 * i)
    # the robot moves to the place pose
    robot.move_pose(new_place_pose)
    
    
# We can manipulate the tools of the robot using the following API
# the update tool function scans the robot mechanics to determine which tool is in use
robot.update_tool()

# to perform a tool agnostic gransping you can use the function
robot.grasp_with_tool()

tool_used = ToolID.GRIPPER_1
# perform action for specific tool based on tool used
if tool_used in [ToolID.GRIPPER_1, ToolID.GRIPPER_2, ToolID.GRIPPER_3]:
    robot.close_gripper(speed=500)
elif tool_used == ToolID.ELECTROMAGNET_1:
    pin_electromagnet = PinID.XXX
    robot.setup_electromagnet(pin_electromagnet)
    robot.activate_electromagnet(pin_electromagnet)
elif tool_used == ToolID.VACUUM_PUMP_1:
    robot.pull_air_vacuum_pump()
    
    
# performing tool release action based on the tool that is set
if tool_used in [ToolID.GRIPPER_1, ToolID.GRIPPER_2, ToolID.GRIPPER_3]:
    robot.open_gripper(speed=500)
elif tool_used == ToolID.ELECTROMAGNET_1:
    pin_electromagnet = PinID.XXX
    robot.setup_electromagnet(pin_electromagnet)
    robot.deactivate_electromagnet(pin_electromagnet)
elif tool_used == ToolID.VACUUM_PUMP_1:
    robot.push_air_vacuum_pump()
    

def pick_and_place1():
    height_offset = 0.05  # Offset according to Z-Axis to go over pick & place poses
    gripper_speed = 400

    # Going Over Object
    robot.move_pose(pick_pose.x, pick_pose.y, pick_pose.z + height_offset,
                               pick_pose.roll, pick_pose.pitch, pick_pose.yaw)
    # Opening Gripper
    robot.open_gripper(gripper_speed)
    # Going to picking place and closing gripper
    robot.move_pose(pick_pose)
    robot.close_gripper(gripper_speed)

    # Raising
    robot.move_pose(pick_pose.x, pick_pose.y, pick_pose.z + height_offset,
                               pick_pose.roll, pick_pose.pitch, pick_pose.yaw)

    # Going Over Place pose
    robot.move_pose(place_pose.x, place_pose.y, place_pose.z + height_offset,
                               place_pose.roll, place_pose.pitch, place_pose.yaw)
    # Going to Place pose
    robot.move_pose(place_pose)
    # Opening Gripper
    robot.open_gripper(gripper_speed)
    # Raising
    robot.move_pose(place_pose.x, place_pose.y, place_pose.z + height_offset,
                               place_pose.roll, place_pose.pitch, place_pose.yaw)
    

def pick_and_place2():
    height_offset = 0.05  # Offset according to Z-Axis to go over pick & place poses
    gripper_speed = 400
    
    pick_pose_high = pick_pose.copy_with_offsets(z_offset=height_offset)
    place_pose_high = place_pose.copy_with_offsets(z_offset=height_offset)

    # Going Over Object
    robot.move_pose(pick_pose_high)
    # Opening Gripper
    robot.release_with_tool()
    # Going to picking place and closing gripper
    robot.move_pose(pick_pose)
    robot.grasp_with_tool()
    # Raising
    robot.move_pose(pick_pose_high)

    # Going Over Place pose
    robot.move_pose(place_pose_high)
    # Going to Place pose
    robot.move_pose(place_pose)
    # Opening Gripper
    robot.release_with_tool(gripper_speed)
    # Raising
    robot.move_pose(place_pose_high)


def pick_n_place_version_3():
    # Pick
    robot.pick_from_pose(pick_pose)
    # Place
    robot.place_from_pose(place_pose)
    
def pick_n_place_version_4():
    # Pick & Place
    robot.pick_and_place(pick_pose, place_pose)

# The ned python library also comes with a conveyor system
# its not directly valuable but it would help in better understanding the robot
# and API if we need to perform so more customized development in future

# activate the connection with the conveyor belt
conveyor_id =  robot.set_conveyor()

# run the conveyor at 50% speed
robot.run_conveyor(conveyor_id, speed=50, direction=ConveyorDirection.FORWARD)

# pause the entire robot for 3 seconds
robot.wait(3)

# stop the conveyor motor
robot.stop_conveyor(conveyor_id)

# disconnect the conveyor
robot.unset_conveyor(conveyor_id)

# -- Setting variables
sensor_pin_id = PinID.GPIO_1A

catch_nb = 5

for i in range(catch_nb):
    robot.run_conveyor(conveyor_id)
    while robot.digital_read(sensor_pin_id) == PinState.LOW:
        robot.wait(0.1)

    # Stopping robot's motor
    robot.stop_conveyor(conveyor_id)
    # Making a pick & place
    robot.pick_and_place(pick_pose, place_pose)

# Deactivating connexion with the Conveyor Belt
robot.unset_conveyor(conveyor_id)


# Using Ned Niryo vision API, although we will not be using their vision API as our setup will have the camera connected to 
# the macbook directly

# Calibrate robot if the robot needs calibration
robot.calibrate_auto()
# Updating tool
robot.update_tool()

## Template for robot module

local_mode = False # Or True
tool_used = ToolID.GRIPPER_1
# Set robot address
robot_ip_address_rpi = "x.x.x.x"
robot_ip_address_local = "127.0.0.1"

robot_ip_address = robot_ip_address_local if local_mode else robot_ip_address_rpi


def process(niryo_edu):
    # --- --------- --- #
    # --- YOUR CODE --- #
    # --- --------- --- #
    pass

if __name__ == '__main__':
    # Connect to robot
    robot = NiryoRobot(robot_ip_address)
    # Calibrate robot if robot needs calibration
    robot.calibrate_auto()
    # Equip tool
    robot.update_tool()
    # Launching main process
    process(robot)
    # Ending
    robot.go_to_sleep()
    # Releasing connection
    robot.close_connection()
    
# Set connection to robot
ip = ""
robot2 = NiryoRobot(ip)
# connet robot to specific ip address
robot2.connect(ip)

# close the connection with a robot
robot2.close_connection()

# calibrate robot, you can do auto calibration or manual calibration
# what is calibration and why is it important to start with calibration
robot2.calibrate_auto()
# manual calibration
robot2.calibrate(CalibrateMode.MANUAL)

# you can check if the robot motors need calibration
calibration_needed = robot2.need_calibration()

# get the learning mode of a robot
# what is learning mode and why is it important
learning_mode = robot2.get_learning_mode()

# set the learning mode state
robot2.set_learning_mode(True)

# the API even allows us to adjust hte arm velocity
robot2.set_arm_max_velocity(50)

# we can adjust jog control if we want
# what does it mean for jog control to be set or not?
robot2.set_jog_control(True)


# we can make the entire robot wait for a specific amount of time
robot2.wait(10) # this will make the entire robot wait for 10 seconds


# workign with joints and poses in robots
# we can retreive joint information from the robot
joints = robot2.get_joints()
joints = robot2.joints

# get robot pose
pose = robot2.get_pose()
pose = robot.pose

# get robot pose quaterion
pose_quat = robot2.get_pose_quat()

# move robot joint values, joint values are expressed in radians
robot.joints = [0.2, 0.1, 0.3, 0.0, 0.5, 0.0]
robot.move_joints([0.2, 0.1, 0.3, 0.0, 0.5, 0.0])
robot.move_joints(0.2, 0.1, 0.3, 0.0, 0.5, 0.0)

# the pose is the pose of the robot end effector
# we can set pose objects using lists, distributed values or pose objects
# PoseObjects allow us to adjust the robot pose through methods without having to 
# the manual value manipulations
robot.pose = [0.2, 0.1, 0.3, 0.0, 0.5, 0.0]
robot.move_pose([0.2, 0.1, 0.3, 0.0, 0.5, 0.0])
robot.move_pose(0.2, 0.1, 0.3, 0.0, 0.5, 0.0)
robot.move_pose(PoseObject(0.2, 0.1, 0.3, 0.0, 0.5, 0.0))

# Move robot pose in a linear trajectory
robot.move_linear_pose([0.2, 0.1, 0.3, 0.0, 0.5, 0.0])

# shift robot pose along a particular axis by a specific amount
robot.shift_pose(RobotAxis.X, 19)

# shift robot pose along a particular axis with linear trajectory
robot.shift_linear_pose(RobotAxis.Y, 12)

# We can adjust jog values
# Jog corresponds to a shift without motion
# We can jog joints
robot2.jog_joints([0.2, 0.1, 0.3, 0.0, 0.5, 0.0])

# We can jog poses
robot2.jog_pose([0.2, 0.1, 0.3, 0.0, 0.5, 0.0])

# We can move to home pose which is the robot resting position
# I think learning mode is where we can adjust the robots actuators and they are not stif
robot2.move_to_home_pose()

# We can move to home pose and sleep, which will set the learning mode to true
robot2.go_to_sleep()

# compute the forward and inverse kinematics automatically
# the forward kinematics uses the joint positions to estimate the pose
# the inverse kinematics uses the pose to testimate the joint values

# FORWARD KINEMATICS
pose: PoseObject =  robot2.forward_kinematics([0.2, 0.1, 0.3, 0.0, 0.5, 0.0])

# INVERSE KINEMATICS
joint_positions:List = robot2.inverse_kinematics(place_pose)

# We can save specific poses in the ned memory
pose_name = "pose_key"
saved_poses = robot2.get_pose_saved(pose_name)

# this saves the pose of a robot with a speicific key
robot2.save_pose("pose_key")

# delete saved pose
robot2.delete_pose("pose_key")

# get the list of all saved poses
pose_list = robot2.get_saved_pose_list()

# The ned robot also provides high level APIs for actions like pick and place
# we can pick from a specific pose
robot2.pick_from_pose(pick_pose)

# We can place from pose
robot2.place_from_pose(place_pose)

# Full robot pick and place with robot high level APIs
# dist smoothing is for distance from waypoints before smoothing trajectory
robot2.pick_and_place(pick_pose, place_pose, dist_smoothing=0.0)

# Handling trajectories
# similar to saved poses we can save trajectories
trajectory = robot2.get_trajectory_saved("trajectory_key")

# A trajectory is a list of poses
# We can execute a trajectory by providing a list of poses
robot2.execute_trajectory_from_poses([pick_pose.to_list(), place_pose.to_list()])

# you can execute a trajectory based on both poses and joints, pretty weird 
# but excited to try it out
robot2.execute_trajectory_from_poses_and_joints()

# execute a previously saved trajectory
robot2.execute_trajectory_saved("trajectory_key")

# save a strajectory as a list of poses
pose_list = []
robot2.save_trajectory("trajectory_key", pose_list)

# get all saved trajectories
saved_trajectories: List = robot2.get_saved_trajectory_list()


# tool use
# The tool is what is connect to the end effector for operations
# get the id of the tool that is currently connected
tool_id = robot2.get_current_tool_id()

# update the tool that is equiped this is valuable if you change the tool while the robot is still in operation
robot2.update_tool()

# we can perform high level grasp and releas with tool
robot2.grasp_with_tool()
robot2.release_with_tool()

# we can also perform low level grasp and releas with a specific tool
robot2.open_gripper()
robot2.close_gripper()

# if an overload occurs you can reboot the motor of the tools
robot2.tool_reboot()

# we can get the full hardware status of the robot
robot2.get_hardware_status()


# How can we leverage some of the vision to pose calculations used in the robot to perform 
# robot actions
# Also what is a robot workspace






