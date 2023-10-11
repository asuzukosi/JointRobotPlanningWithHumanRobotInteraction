import pyniryo
import numpy
import cv2
import time
# if you are connected to the robot through the robot's hotspot this is the 
# default ip address for the hotspot connection
# although we need to explore the possibility of connecting the robot to a shared network
# to computer make network requests while connected to the robot
ROBOT_IP_ADDRESS = "10.10.10.10"

# # If the computer is not connected to the same network as the robot it raises a network connection exception
# try:
#     Robot = pyniryo.NiryoRobot(ROBOT_IP_ADDRESS)
# except Exception as e:
#     print("You are not connectied to the same network as the robot")

# any two images of the same planar surface in 
# space are related by a homography
# we are trying to map the homography between the two images

def capture_image():
    """
    Uses intelrealsense camera to capture the 
    snapshot of what is going on in the scene
    a sample of such image is lego_blocks5 file
    """
    
    # determine frame rate of image capture
    return numpy.random.normal(size=(3, 64, 64))


# we would be using background substraction to find the exact pixel location of
# an object in an image 

# BACKGROUND SUBTRACTION
# The idea behind background subtraction is that once you have 
# a model of the background, you can detect objects by examining the difference 
# between the current video frame and the background frame.


def turn_gray(image):
    """
    Gets the gray scaled version of an image to allow absolute difference 
    background subtraction
    """
    # Create kernel for morphological operation. You can tweak
    # the dimensions of the kernel.
    # e.g. instead of 20, 20, you can try 30, 30
    kernel = numpy.ones((20,20),numpy.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     
    # Close gaps using closing
    gray = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel)
       
    # Remove salt and pepper noise with a median filter
    gray = cv2.medianBlur(gray,5)
    
    return gray

# this is better suited to detect static object that enter into a scene and stay fixed
def absolute_difference_background_subtraction(base_frame, current_frame):
    """
    Calculate the absolute difference between the current video frame
    and he original video frame
    
    Runs very fast, but is sensitive to noise and shadows
    """
    # get the grayed version of the original frame
    base_gray = turn_gray(base_frame)
    # get the grayed version of the current video frame
    current_gray = turn_gray(current_frame)
    # get the absolute differene between base frame and current frame
    absolute_difference = cv2.absdiff(base_gray, current_gray)
    # adjust the absolute difference based on specific threshold
    # here we are using the threshold of 100 and a maximum value of 255
    _, absolute_difference = cv2.threshold(absolute_difference, 
                                           100, 255, 
                                           cv2.THRESH_BINARY)
    
    # Find the contours of the object inside the binary image
    contours, hierarchy = cv2.findContours(absolute_difference,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:] # takes only the last two items from the result
    # find all the areas that are different from the original image
    areas = [cv2.contourArea(c) for c in contours]
    
    # if there are no moving objects then return none
    if len(areas) < 1:
        # no new objects were found in the image
        return False, []

    # Find the largest moving object in the image
    max_index = numpy.argmax(areas)
    
    centroids = []
    for index in range(len(contours)):
    
        # Draw the bounding box
        cnt = contours[index]
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(current_frame,(x,y),(x+w,y+h),(0,255,0),3)
        
        # get the center points of the image
        # Draw circle in the center of the bounding box
        center_x = x + int(w/2)
        center_y = y + int(h/2)
        centroids.append((center_x, center_y))
        
        # OPTIONAL
        # cv2.circle(current_frame,(x2,y2),4,(0,255,0),-1)
        
        # # Print the centroid coordinates (we'll use the center of the
        # # bounding box) on the image
        # text = "x: " + str(x2) + ", y: " + str(y2)
        # cv2.putText(current_frame, text, (x2 - 10, y2 - 10),
        # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        # # Display the resulting frame
        # cv2.imshow("Frame",current_frame)
    return True, centroids

def calculate_pick_position_givne_bounded_box(x, y, width, height):
    # Given a set of bounding box parameters determine the 
    # exact center point or pick location of the object
    pass

# this is more compulationally intensive that absolute substraction
# but it hanldes shadows and noise better
# This is better suited for moving objects
def background_subtractor_mog2(base_frame, current_frame):
    # This algorithm detects objects in a video stream
    # using the Gaussian Mixture Model background subtraction method
    bg_subtractor = cv2.BackgroundSubtractorMOG2(history=150,
                                                 varThreshold=25, detectShadows=True)
    time.sleep(0.1)
    # Create kernel for morphological operation. You can tweak
    # the dimensions of the kernel.
    # e.g. instead of 20, 20, you can try 30, 30
    kernel = numpy.ones((20,20),numpy.uint8)
    
    # Convert to foreground mask
    fg_mask = bg_subtractor.apply(current_frame)
     
    # Close gaps using closing
    fg_mask = cv2.morphologyEx(fg_mask,cv2.MORPH_CLOSE,kernel)
       
    # Remove salt and pepper noise with a median filter
    fg_mask = cv2.medianBlur(fg_mask,5)
    
    # If a pixel is less than ##, it is considered black (background). 
    # Otherwise, it is white (foreground). 255 is upper limit.
    # Modify the number after fg_mask as you see fit.
    _, fg_mask = cv2.threshold(fg_mask, 
                               127, 255, 
                               cv2.THRESH_BINARY)
    # Find the contours of the object inside the binary image
    contours, hierarchy = cv2.findContours(fg_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    areas = [cv2.contourArea(c) for c in contours]
    
     # if there are no moving objects then return none
    if len(areas) < 1:
        # no new objects were found in the image
        return False, []
    
    # get the list of all the center points of the contour objects
    centroids = []
    for index in range(len(contours)):
    
        # Draw the bounding box
        cnt = contours[index]
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(current_frame,(x,y),(x+w,y+h),(0,255,0),3)
        
        # get the center points of the image
        # Draw circle in the center of the bounding box
        center_x = x + int(w/2)
        center_y = y + int(h/2)
        centroids.append((center_x, center_y))
        
        # OPTIONAL
        # cv2.circle(current_frame,(x2,y2),4,(0,255,0),-1)
        
        # # Print the centroid coordinates (we'll use the center of the
        # # bounding box) on the image
        # text = "x: " + str(x2) + ", y: " + str(y2)
        # cv2.putText(current_frame, text, (x2 - 10, y2 - 10),
        # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        # # Display the resulting frame
        # cv2.imshow("Frame",current_frame)
    return True, centroids
  

def find_obj_location(image: numpy.ndarray):
    """
    Find the pixel location of an object in an image
    We would want to use the central point of the image as the 
    image location
    Args:
        image (numpy.ndarray): input image
    """
    pass


def pixel_to_pose(pixel):
    """
    Moves the robot to a specific pixel location given
    a fixed camera location relative to the pixel location
    Args:
        pixel ((x, y)): specific pixel value 
    """
    pass

# actions that will be generated by the llm will be executed by the python exec function


# exec() function is used for the dynamic execution of Python 
# programs which can either be a string or object code.
# If it is a string, the string is parsed as a suite of 
# Python statements which is then executed unless a syntax
# error occurs and if it is an object code, it is simply 
# executed. 
robot_action = None

generated ='''
def robot_action(x, y):
  return x + y
'''

generated2 = '''
def robot_action(x, y):
  return x * y
'''

class Test:
    def __init__(self):
        exec(generated, globals())
        print(robot_action(1, 2))
        exec(generated2, globals())
        print(robot_action(1, 2))

# t = Test()

# image = capture_image()

# Converting pixel coordinates to coordinates 
# relative to the robot baseframe
