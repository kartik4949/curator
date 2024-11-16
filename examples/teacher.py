"""
Example of using a teacher/evaluator/feedback model to improve a student model.


We maintain a success dataset of responses that meet the minimum score,
and a needs-improvement dataset of responses that do not meet the minimum score.
We repeat until all student responses are sufficiently good
or max iterations are reached.

1. Student generates responses to instructions
2. Evaluator scores the responses across multiple dimensions
3. If responses meet the minimum score, add them to the success dataset.
Otherwise, add responses to the needs-improvement dataset.
4. Feedback generator provides improvement suggestions for needs-improvement dataset
5. Student generates revised responses based on feedback
"""

from datasets import Dataset, concatenate_datasets, load_dataset
from pydantic import BaseModel

from bespokelabs import curator

# Load a small subset of the WildChat dataset for training
needs_improvement_dataset = load_dataset("allenai/WildChat", split="train")
needs_improvement_dataset = needs_improvement_dataset.select(
    range(50)
)  # Select first 5 examples
# Extract the first message from each conversation as the instruction
needs_improvement_dataset = needs_improvement_dataset.map(
    lambda row: {"instruction": row["conversation"][0]["content"]}, num_proc=2
)
needs_improvement_dataset = needs_improvement_dataset.select_columns(
    ["instruction"]
)
success_dataset = []  # Will store successfully improved responses


# Create a student model that generates responses to instructions
def create_student(model_name: str):
    def response_prompt_func(row):
        # If improvements exist, ask for a revised response based on feedback
        # Otherwise, generate an initial response
        if "improvements" in row:
            return f"For the given instruction {row['instruction']} and response {row['new_response']}. Generate a revised response based on the feedback {row['improvements']}."
        else:
            return f"For the given instruction {row['instruction']}. Generate a response."

    def response_parse_func(row, response):
        return {"instruction": row["instruction"], "new_response": response}

    return curator.Prompter(
        prompt_func=response_prompt_func,
        parse_func=response_parse_func,
        model_name=model_name,
    )


# Create an evaluator model that scores responses on multiple criteria
def create_evaluator(model_name: str):
    class Evaluation(BaseModel):
        # Define scoring criteria, each on a scale of 0-10
        coherence: int
        relevance: int
        correctness: int
        naturalness: int

    def evaluation_prompt_func(row):
        return [
            {
                "role": "system",
                "content": (
                    "You are an expert evaluator of AI assistant responses. You will be given an instruction and a response. "
                    "Evaluate the response based on the following criteria: coherence, relevance, correctness, and naturalness. "
                    "Each criterion should be scored on a scale of 0 to 10. "
                ),
            },
            {
                "role": "user",
                "content": f"Evaluate the following instruction and response: Instruction: {row['instruction']}\nResponse: {row['new_response']}. Scores:",
            },
        ]

    return curator.Prompter(
        prompt_func=evaluation_prompt_func,
        parse_func=lambda original_row, evaluation: {
            "instruction": original_row["instruction"],
            "new_response": original_row["new_response"],
            "evaluation": evaluation.model_dump(),
        },
        model_name=model_name,
        response_format=Evaluation,
    )


# Create a feedback generator model that provides specific improvement suggestions
def create_feedback_generator(model_name: str):
    def improvements_prompt_func(row):
        return f"For the given instruction {row['instruction']} and response {row['new_response']}. Generate some feedback. Including points of improvement, specific suggestions, and examples if needed."

    return curator.Prompter(
        prompt_func=improvements_prompt_func,
        parse_func=lambda original_row, improvements: {
            "instruction": original_row["instruction"],
            "new_response": original_row["new_response"],
            "evaluation": original_row["evaluation"],
            "improvements": improvements,
        },
        model_name=model_name,
    )


# Function to separate successful responses from those needing improvement
def check_scores(dataset: Dataset, minimum_score: int):
    def filter_func(row):
        # Check if all evaluation scores meet the minimum threshold
        return (
            row["evaluation"]["coherence"] >= minimum_score
            and row["evaluation"]["relevance"] >= minimum_score
            and row["evaluation"]["correctness"] >= minimum_score
            and row["evaluation"]["naturalness"] >= minimum_score
        )

    # Print evaluation scores for successful and failed responses
    success_dataset = dataset.filter(filter_func)

    failure_dataset = dataset.filter(lambda row: not filter_func(row))

    assert len(success_dataset) + len(failure_dataset) == len(dataset)
    return success_dataset, failure_dataset


# Initialize the models
student = create_student("gpt-3.5-turbo")  # Student model (gpt-3.5-turbo)
evaluator = create_evaluator(
    "gpt-4o-mini"
)  # Teacher/evaluator model (gpt-4o-mini)
feedback_generator = create_feedback_generator(
    "gpt-4o-mini"
)  # Teacher/feedback model (gpt-4o-mini)

# Main improvement loop
iterations = 0
max_iterations = 4
print(
    f"Starting with {needs_improvement_dataset.num_rows} responses to improve."
)

while len(needs_improvement_dataset) > 0 and iterations < max_iterations:
    iterations += 1

    # Step 1: Student generates responses for remaining instructions
    response_dataset = student(needs_improvement_dataset)

    # Step 2: Evaluator scores the responses
    evaluation_dataset = evaluator(response_dataset)
    # Step 3: Separate successful responses (all 4 criteria score >= 9) from those needing improvement
    newly_successful_dataset, needs_improvement_dataset = check_scores(
        evaluation_dataset, 9
    )
    print(
        f"Iteration {iterations} complete. {len(newly_successful_dataset)} newly successful. {len(success_dataset)} total successful. {len(needs_improvement_dataset)} remaining."
    )

    # Add newly successful responses to the success dataset
    if len(newly_successful_dataset) > 0:
        if len(success_dataset) > 0:
            success_dataset = concatenate_datasets(
                [success_dataset, newly_successful_dataset]
            )
        else:
            success_dataset = newly_successful_dataset

    # Step 4: For remaining responses, get improvement feedback
    if len(needs_improvement_dataset) > 0:
        improvements_dataset = feedback_generator(needs_improvement_dataset)
        # This dataset will be used in the next iteration by the student

# Print final results
if needs_improvement_dataset.num_rows > 0:
    print(
        f"Reached max iterations ({max_iterations}). Success dataset size ({len(success_dataset)}). Needs-improvement dataset size ({needs_improvement_dataset.num_rows}):"
    )
else:
    print(
        f"Successfully completed in {iterations} iterations. Success dataset size ({len(success_dataset)}):"
    )
