import openai

# Tips and tricks to improve llm performance

# LLM caching, we cache the response from an LLM in order not to make repeated calls
# We do this by using the query, model, temperature, maxtoken, logprobs and echo as the key

def get_cache(
    LLM_CACHE = {},
    model="text-ada-001",
    prompt="",
    max_tokens=128,
    temperature=0,
    logprobs=1,
    echo=False
):

    id = ((model, prompt, max_tokens, temperature, logprobs, echo))

    if id in LLM_CACHE.keys():
        return LLM_CACHE[id]
    return None

def cache_response(LLM_CACHE = {},
    model="text-ada-001",
    prompt="",
    max_tokens=128,
    temperature=0,
    logprobs=1,
    echo=False,
    response=""):
    
    id  = ((model, prompt, max_tokens, temperature, logprobs, echo))
    
    LLM_CACHE[id] = response
    return 