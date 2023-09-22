## In this section we will be exploring using language models to generate action sequences for our robot to perform tasks

### We will first try setting up Llama2 7b and lay out the steps used to run the application on mac os

We will be using ggml as the backbone for our inference and llama.cpp as the architecture. 

We will be carrying out the following steps to setup ggml
- clone ggml from the git repo at https://github.com/ggerganov/llama.cpp i.e `git clone https://github.com/ggerganov/llama.cpp`
- move into the cloned repository and build the tool using make `cd llama.cpp && LLAMA_METAL=1 make`
- convert downloaded weights to gguf format `python convert.py EvolCodeLlama-7b --outtype f16 --outfile EvolCodeLlama-7b/evolcodellama-7b.gguf.fp16.bin`
- Get the code llmaa model from hugging face `wget https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF/resolve/main/codellama-7b-instruct.Q2_K.gguf`
- The 13b model memory requirement is too large, and the higher quantized models for the 7b are too slow, although we will experiment with trying them individually to find the minimum acceptable performance
- Install llama.cpp python library `pip install llama-cpp-python`