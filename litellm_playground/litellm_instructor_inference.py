import instructor
from litellm import completion, get_supported_openai_params
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int


client = instructor.from_litellm(completion)

resp, completion = client.chat.completions.create_with_completion(
    model="gpt-4o-mini",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "Extract Jason is 25 years old.",
        }
    ],
    response_model=User,
)


params = get_supported_openai_params(model="gpt-4o-mini")
print(params)

assert isinstance(resp, User)
assert resp.name == "Jason"
assert resp.age == 25
print(resp)
print(completion)