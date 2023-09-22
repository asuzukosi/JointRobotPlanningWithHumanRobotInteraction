from pyniryo import *

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









