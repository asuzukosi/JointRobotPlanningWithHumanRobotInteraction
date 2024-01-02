# from sam import getObjectLocation
from vild import getObjectLocation
from vision import SceneDescription
from PIL import Image, ImageDraw
import time





def evaluateModel(image_path, query):
    print("STARTING NOW")
    start = time.time()
    location = getObjectLocation(query, image_path)
    end = time.time()
    print("The location is : ", location)
    print("Total time elapsed : ", end - start)

    try:
        # Load the image
        image = Image.open(image_path)

        # Create a drawing object
        draw = ImageDraw.Draw(image)

        # Define the coordinates of the bounding box (x, y, width, height)
        bbox = (100, 50, 200, 150)

        # Draw the bounding box on the image
        draw.rectangle([location.x -5, location.y - 5, location.x + 5, location.y + 5], outline=(0, 255, 0), width=2)

        # Save the result to a new file
        output_path = f'out_{query}_{image_path}'
        image.save(output_path)

        # Display the result
        # image.show()
    except:
        print("Failed to save image with bounding box")



examples = [
{
    "image_path": "example1.png",
    "query": "red block"
},
{
    "image_path": "example2.png",
    "query": "blue block"
},
{
    "image_path": "example2.png",
    "query": "red block"
},
{
    "image_path": "example1.png",
    "query": "blue bowl"
},
{
    "image_path": "example3.png",
    "query": "green block"
},
{
    "image_path": "example4.jpg",
    "query": "green block"
},
]

# for example in examples:
#     evaluateModel(**example)


start = time.time()
import numpy as np
image = np.array(Image.open("liveimage.jpg"))
SceneDescription(image)
end = time.time()
print("Total time elapsed : ", end - start)

