# # This file is the file that takes in the user prompt and generates the action that should be performed by the robot

# import os
# import subprocess

# # import rospy
# # subscriber = rospy.Subscriber("/ned_lang/action_complete", bool, callback=allow_proceed)

# PROCEED = True

# ACTION_FILE_HEADERS = '''
# import rospy
# from primitives import pick_and_place, move_left, move_right, view_scene
# '''

# ACTION_FILE_FOOTER = '''
# if __name__ == '__main__':
#     rospy.init_node('robot_action)
#     main()
#     publisher = rospy.Publisher("/ned_lang/action_complete", bool)
#     publisher.publish(True)

# '''

# def pixel_to_location(pixel):
#     pass

# def allow_proceed(data):
#     PROCEED = True

# def generate_action_file(code):
#     pass

# def prompt_language_model(prompt):
#     pass

# def loop_action():
#     rospy.loginfo("Enter your prompt: ")
#     prompt = input("Enter your prompt: ")
#     code = prompt_language_model(prompt)
#     rospy.loginfo("Action function generated")
#     rospy.loginfo(f"The generated action function is: \n {code}")
#     generate_action_file(code)
#     rospy.loginfo("Action function file has been generated and is being executed.")
    
    
    


# if __name__ == '__main__':
#     rospy.init_node("ned_lang")
#     rospy.loginfo("Welcome to language based robot control")
    
#     while PROCEED:
#         PROCEED = False
#         loop_action()
        
    
    
    
    
    