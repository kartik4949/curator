import time
from bespokelabs import curator
from datasets import Dataset

start = time.time()
dataset = Dataset.from_dict(
    {"prompt": ["Write a poem about the beauty of computer science"] * 10}
)
poet = curator.Prompter(
    prompt_func=lambda row: row["prompt"], model_name="gpt-4o-mini", n=100
)
poem = poet(dataset)
end = time.time()
assert len(poem["response"]) == 1000
print(f"Time taken for 1000 completions with N=100: {end - start} seconds")

start = time.time()
dataset = Dataset.from_dict(
    {"prompt": ["Write a poem about the beauty of computer science"] * 1000}
)
poet = curator.Prompter(
    prompt_func=lambda row: row["prompt"],
    model_name="gpt-4o-mini",
)
poem = poet(dataset)
end = time.time()
assert len(poem["response"]) == 1000
print(f"Time taken for 1000 completions with N=1: {end - start} seconds")
