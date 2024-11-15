from bespokelabs import curator
from datasets import load_dataset, Dataset, concatenate_datasets
from pydantic import BaseModel, Field

remaining_dataset = load_dataset("allenai/WildChat", split="train")
remaining_dataset = remaining_dataset.select(range(5))
remaining_dataset = remaining_dataset.map(
    lambda row: {"instruction": row["conversation"][0]["content"]}, num_proc=2
)
remaining_dataset = remaining_dataset.select_columns(["instruction"])
success_dataset = []


# Response
def response_prompt_func(row):
    if "improvements" in row:
        prompt = f"For the given instruction {row['instruction']} and response {row['new_response']}. " \
            "Generate a revised response based on the feedback {row['improvements']}."
    else:
        prompt = f"{row['instruction']}"
    return prompt


def response_parse_func_(row, response):
    return {"instruction": row["instruction"], "new_response": response}


response_prompter = curator.Prompter(
    prompt_func=response_prompt_func,
    parse_func=response_parse_func_,
    model_name="gpt-3.5-turbo",
)

# Evaluation
MINIMUM_SCORE = 9
print(f"Minimum evaluation score: {MINIMUM_SCORE}\n")


class Evaluation(BaseModel):
    coherence: int = Field(description="A coherence score from 0 to 10")
    relevance: int = Field(description="A relevance score from 0 to 10")
    correctness: int = Field(description="A correctness score from 0 to 10")
    naturalness: int = Field(description="A naturalness score from 0 to 10")


def evaluation_prompt_func(row):
    prompt = f"For the given instruction {row['instruction']}. " \
        "Generate an evaluation of the response {row['new_response']}."
    return prompt


evaluation_prompter = curator.Prompter(
    prompt_func=evaluation_prompt_func,
    parse_func=lambda original_row, evaluation: {
        "instruction": original_row["instruction"],
        "new_response": original_row["new_response"],
        "evaluation": evaluation.model_dump(),
    },
    model_name="gpt-4o",
    response_format=Evaluation,
)


def check_scores(dataset: Dataset):
    def filter_func(row):
        return (
            row["evaluation"]["coherence"] >= MINIMUM_SCORE
            and row["evaluation"]["relevance"] >= MINIMUM_SCORE
            and row["evaluation"]["correctness"] >= MINIMUM_SCORE
            and row["evaluation"]["naturalness"] >= MINIMUM_SCORE
        )

    success_dataset = dataset.filter(filter_func)
    print("SUCCEEDED SAMPLES SCORES:")
    for row in success_dataset:
        print(row["evaluation"])

    failure_dataset = dataset.filter(lambda row: not filter_func(row))
    print("FAILED SAMPLES SCORES:")
    for row in failure_dataset:
        print(row["evaluation"])

    return success_dataset, failure_dataset


# Improvement
def improvements_prompt_func(row):
    prompt = f"For the given instruction {row['instruction']} and response {row['new_response']}. " \
        "Generate some feedback. Including points of improvement, specific suggestions, and examples if needed."
    return prompt


improvements_prompter = curator.Prompter(
    prompt_func=improvements_prompt_func,
    parse_func=lambda original_row, improvements: {
        "instruction": original_row["instruction"],
        "new_response": original_row["new_response"],
        "evaluation": original_row["evaluation"],
        "improvements": improvements,
    },
    model_name="gpt-4o",
)

# Main loop
iterations = 0
max_iterations = 4
while len(remaining_dataset) > 0 and iterations < max_iterations:
    iterations += 1

    print("Generating model responses...")
    response_dataset = response_prompter(remaining_dataset)

    print("Generating evaluations...")
    evaluation_dataset = evaluation_prompter(response_dataset)
    newly_successful_dataset, remaining_dataset = check_scores(evaluation_dataset)
    print(f"Iteration {iterations} complete. " \
            f"{len(newly_successful_dataset)} newly successful. " \
            f"{len(success_dataset)} total successful. " \
            f"{len(remaining_dataset)} remaining.\n"
    )

    if len(newly_successful_dataset) > 0:
        if len(success_dataset) > 0:
            success_dataset = concatenate_datasets([success_dataset, newly_successful_dataset])
        else:
            success_dataset = newly_successful_dataset

    if len(remaining_dataset) > 0:
        print("Generating improvements...")
        improvements_dataset = improvements_prompter(remaining_dataset)

# Print results
if remaining_dataset.num_rows > 0:
    print(f"Reached max iterations ({max_iterations}). " \
            f"Success dataset size ({len(success_dataset)}). " \
            f"Remaining dataset size ({remaining_dataset.num_rows}):")
else:
    print(f"Successfully completed in {iterations} iterations. " \
            f"Success dataset size ({len(success_dataset)}):")
