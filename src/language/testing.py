from llama_cpp import Llama
MODEL_PATH = "codellama-7b-instruct.Q2_K.gguf"

def load_llm():
    llm = Llama(model_path=MODEL_PATH, n_ctx=1024)
    return llm

def llm_response(prompt):
    llm = load_llm()
    llm_pipeline = llm(prompt, max_tokens=1000, temperature=0.4)
    text_from_choices = llm_pipeline['choices'][0]['text']
    return text_from_choices


prompt = '''
Please solve the following instruction step-by-step. You should implement ONLY
    main() function do not call the main function, and output in the Python-code style. do not import any libraries
You have access to the following functions.
- ViewScene() # views the entire scene
- FindObjectInScene(object_name) # returns list of locations of object
- PickAndPlace(loc_1, loc_2) # pick and object from loc_1 and move to loc_2
- MoveLeft(loc_1) # move object in loc_1 to the left
- MoveRight(loc_1) # move object in loc_2 to the right
Example:
def main():
    toy = FindObjectInScene(toy)
    box = FindObjectInScene(box)
    PickAndPlace(toy, box)
    
Instruction: move all the lego blocks to the right.
'''

response = llm_response(prompt)
print("RESPONSE -------------------------------------------- RESPONSE")
print(response)


