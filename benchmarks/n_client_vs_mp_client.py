import time
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

client = OpenAI()

for _ in range(5):
    n = 50
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
    print(f"Time taken for {n} completions with API: {end - start} seconds")

    start = time.time()
    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = [
            executor.submit(
                client.chat.completions.create,
                messages=[
                    {
                        "role": "user",
                        "content": "Write me a poem",
                    }
                ],
                model="gpt-4o-mini",
            )
            for _ in range(n)
        ]
        chat_completions = [future.result() for future in futures]
    end = time.time()
    total_choices = sum(len(completion.choices) for completion in chat_completions)
    assert total_choices == n
    print(f"Time taken for {n} completions with MP: {end - start} seconds")
    print("")
