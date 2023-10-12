PROMPT = """Think step by step to carry out the instruction.
Instruction: Put the red block in the green bowl.
Program:

loc1 = getObjectLocation('red block')
loc2 = getObjectLocation('green bowl')
PickAndPlace(loc1, loc2)

Instruction: Put the green block into the red bowl.
Program:

loc1 = getObjectLocation('green block')
loc2 = getObjectLocation('red bowl')
PickAndPlace(loc1, loc2)

Instruction: Move the red block to the left
Program:

loc1 = getObjectLocation('red block)
MoveRight(loc1)

Instruction: Move all the green blocks to the blue bowl.
Program:

locations = getAllObjectLocation('green block')
loc2 = getObjectLocation('blue bowl')
PickAndPlaceAll(locations, loc2)

Instruction: Put the red and blue block in green bowl
Program:
locations = getAllObjectLocation('red block', 'blue block')
loc2 = getObjectLocation('green bowl')
PickAndPlaceAll(locations, loc2)


Instruction: Move all the green blocks to the left
Program:
locations = getAllObjectLocation('green block')
MoveLeftAll(locations)

Instruction: Put the green blocks to the blue block
Program:
loc1 = getObjectLocation('green block')
loc2 = getObjectLocation('blue block')
PickAndPlace(loc1, loc2)

Scene: In the scene there are: 
Instruction: INSERT TASK HERE.
Program:
"""