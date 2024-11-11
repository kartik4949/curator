from typing import List
from pydantic import BaseModel, Field
from bespokelabs import curator
from datasets import Dataset
import time


class Subject(BaseModel):
    subject: str = Field(description="A subject")


class Subjects(BaseModel):
    subjects: List[Subject] = Field(description="A list of subjects")


class Topic(BaseModel):
    topic: str = Field(description="A topic")


class Topics(BaseModel):
    topics: List[Topic] = Field(description="A list of topics")


class Subtopic(BaseModel):
    subtopic: str = Field(description="A subtopic")


class Subtopics(BaseModel):
    subtopics: List[Subtopic] = Field(description="A list of subtopics")


class QA(BaseModel):
    question: str = Field(description="A question")
    answer: str = Field(description="An answer")


class QAs(BaseModel):
    qas: List[QA] = Field(description="A list of QAs")


subject_prompter = curator.Prompter(
    prompt_func=lambda: f"Generate a diverse list of 3 subjects. Keep it high-level (e.g. Math, Science).",
    parse_func=lambda _, subjects: [subject for subject in subjects.subjects],
    model_name="gpt-4o-mini",
    response_format=Subjects,
)
subject_dataset = subject_prompter()
print(subject_dataset.to_pandas())

topic_prompter = curator.Prompter(
    prompt_func=lambda row: f"For the given subject {row['subject']}. Generate 25 diverse topics. No explanation.",
    parse_func=lambda row, topics: [
        {"subject": row["subject"], "topic": topic.subject} for topic in topics.subjects
    ],
    model_name="gpt-4o-mini",
    response_format=Subjects,
)
topic_dataset = topic_prompter(subject_dataset)
print(topic_dataset.to_pandas())

subtopic_prompter = curator.Prompter(
    prompt_func=lambda row: f"For the given topic {row['topic']} in the subject {row['subject']}. Generate 25 diverse subtopics. No explanation.",
    parse_func=lambda row, subtopics: [
        {
            "subject": row["subject"],
            "topic": row["topic"],
            "subtopic": subtopic.subtopic,
        }
        for subtopic in subtopics.subtopics
    ],
    model_name="gpt-4o-mini",
    response_format=Subtopics,
)
subtopic_dataset = subtopic_prompter(topic_dataset)
print(subtopic_dataset.to_pandas())


start = time.time()
qa_prompter = curator.Prompter(
    prompt_func=lambda row: f"For the given subtopic {row['subtopic']} in the topic {row['topic']} in the subject {row['subject']}. Generate 3 diverse questions and answers. No explanation.",
    model_name="gpt-4o-mini",
    response_format=QAs,
    parse_func=lambda row, qas: [
        {
            "subject": row["subject"],
            "topic": row["topic"],
            "subtopic": row["subtopic"],
            "question": qa.question,
            "answer": qa.answer,
        }
        for qa in qas.qas
    ],
    n=80,
)
qa_dataset = qa_prompter(subtopic_dataset)
end = time.time()
print(qa_dataset.to_pandas())
print(f"Time taken for completions with N=80: {end - start} seconds")

subtopic_dataset_repeated = subtopic_dataset.map(
    lambda x: {k: [v] * 80 for k, v in x.items()}, batched=True, batch_size=1
)
start = time.time()
qa_prompter = curator.Prompter(
    prompt_func=lambda row: f"For the given subtopic {row['subtopic']} in the topic {row['topic']} in the subject {row['subject']}. Generate 3 diverse questions and answers. No explanation.",
    model_name="gpt-4o-mini",
    response_format=QAs,
    parse_func=lambda row, qas: [
        {
            "subject": row["subject"],
            "topic": row["topic"],
            "subtopic": row["subtopic"],
            "question": qa.question,
            "answer": qa.answer,
        }
        for qa in qas.qas
    ],
)
qa_dataset = qa_prompter(subtopic_dataset_repeated)
print(qa_dataset.to_pandas())
print(len(qa_dataset))
print(f"Time taken for completions with N=1: {end - start} seconds")
