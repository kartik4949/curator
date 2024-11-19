import os
from litellm import completion

# set env - [OPTIONAL] replace with your anthropic key
os.environ["ANTHROPIC_API_KEY"] = ""

messages = [{"role": "user", "content": "Hey! how's it going?"}]
response = completion(model="claude-3-haiku-20240307", messages=messages)
print(response)
