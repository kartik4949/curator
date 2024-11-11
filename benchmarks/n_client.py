import os
import time
from openai import OpenAI

client = OpenAI()

for n in [1, 2, 4, 6, 8, 10, 20, 30, 40, 50, 60, 80, 100, 200]:
    start = time.time()
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Write me a poem",
            }
        ],
        model="gpt-4o-mini",
        n=n,
    )
    end = time.time()
    assert len(chat_completion.choices) == n
    print(f"Time taken for {n} completions: {end - start} seconds")
