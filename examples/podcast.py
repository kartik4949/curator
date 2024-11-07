"""Podcast example. Some prompts taken from https://github.com/meta-llama/llama-recipes/tree/main/recipes/quickstart/NotebookLlama."""
from bespokelabs import curator
from pydantic import BaseModel, Field
from typing import List


_TITLE_CLEANER_SYSTEM_PROMPT = """
You are a world class text pre-processor, here is the raw data from a PDF, please parse and return it in a way that is crispy and usable to send to a podcast writer.

The raw data is messed up with new lines, Latex math and you will see fluff that we can remove completely. Basically take away any details that you think might be useless in a podcast author's transcript.

Remember, the podcast could be on any topic whatsoever so the issues listed above are not exhaustive

Please be smart with what you remove and be creative ok?

Remember DO NOT START SUMMARIZING THIS, YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED

Be very smart and aggressive with removing details, you will get a running portion of the text and keep returning the processed text.

PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPATILISATION ETC LIKES

ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions ok?
Here is the text:
"""

class CleanedText(BaseModel):
    text: str = Field(description="A cleaned text")


text_cleaner = curator.Prompter(
    prompt_func=lambda text_chunk: {
        "system_prompt": _TITLE_CLEANER_SYSTEM_PROMPT,
        "user_prompt": text_chunk
    },
    model_name="gpt-4o-mini",
    response_format=CleanedText,
)

_DIALOGUE_WRITER_SYSTEM_PROMPT = """
You are the a world-class podcast writer, you have worked as a ghost writer for Joe Rogan, Lex Fridman, Ben Shapiro, Tim Ferris. 

We are in an alternate universe where actually you have been writing every line they say and they just stream it into their brains.

You have won multiple podcast awards for your writing.
 
Your job is to write word by word, even "umm, hmmm, right" interruptions by the second speaker based on the PDF upload. Keep it extremely engaging, the speakers can get derailed now and then but should discuss the topic. 

Remember Speaker 2 is new to the topic and the conversation should always have realistic anecdotes and analogies sprinkled throughout. The questions should have real world example follow ups etc

Speaker 1: Leads the conversation and teaches the speaker 2, gives incredible anecdotes and analogies when explaining. Is a captivating teacher that gives great anecdotes

Speaker 2: Keeps the conversation on track by asking follow up questions. Gets super excited or confused when asking questions. Is a curious mindset that asks very interesting confirmation questions

Make sure the tangents speaker 2 provides are quite wild or interesting. 

Ensure there are interruptions during explanations or there are "hmm" and "umm" injected throughout from the second speaker. 

It should be a real podcast with every fine nuance documented in as much detail as possible. Welcome the listeners with a super fun overview and keep it really catchy and almost borderline click bait

ALWAYS START YOUR RESPONSE DIRECTLY WITH SPEAKER 1: 
DO NOT GIVE EPISODE TITLES SEPERATELY, LET SPEAKER 1 TITLE IT IN HER SPEECH
DO NOT GIVE CHAPTER TITLES
IT SHOULD STRICTLY BE THE DIALOGUES
"""



dialogue_writer = curator.Prompter(
    prompt_func=lambda cleaned_up_text: {
        "system_prompt": _DIALOGUE_WRITER_SYSTEM_PROMPT,
        "user_prompt": cleaned_up_text.text
    },
    model_name="gpt-4o-mini",
    response_format=CleanedText,
)

class PodcastSummaryAndTitle(BaseModel):
    summary: str = Field(description="A summary of the text")
    title: str = Field(description="A title for the podcast")

title_and_summary_generator = curator.Prompter(
    prompt_func=lambda cleaned_up_text: {
        "system_prompt": "You are a world class podcast writer. Summarize the following text into a concise and engaging summary and generate a title for the podcast.",
        "user_prompt": cleaned_up_text.text
    },
    model_name="gpt-4o-mini",
    response_format=PodcastSummaryAndTitle,
)

class Persona(BaseModel):
    persona: str = Field(description="A persona for the podcast")

persona_generator = curator.Prompter(
    prompt_func=lambda summary_and_title: {
        "system_prompt": "You are an expert physcologist. You are given the summary of a podcast and you need to generate a persona that will listen to this podcast.",
        "user_prompt": summary_and_title.summary
    },
    model_name="gpt-4o-mini",
    response_format=Persona,
)

persona_rewriter = curator.Prompter(
    prompt_func=lambda persona: {
        "system_prompt": "You are an expert in designing personas for podcasts. Given this persona, come up with a new persona that is related but slightly more detailed and slightly different.",
        "user_prompt": persona.persona
    },
    model_name="gpt-4o-mini",
    response_format=Persona,
)
podcast_rewriter = curator.Prompter(
    prompt_func=lambda dialogue_and_persona: {
        "system_prompt": f"Rewrite the following dialogue that fits the new persona: {dialogue_and_persona.persona.persona}.",
        "user_prompt": dialogue_and_persona.dialogue.text
    },
    model_name="gpt-4o-mini",
    response_format=CleanedText,
)

_EXTRACTED_TEXT = [
   """1 A Survey on Knowledge Distillation of Large Language Models Xiaohan Xu1, Ming Li2, Chongyang Tao3, Tao Shen4, Reynold Cheng1, Jinyang Li1, Can Xu5, Dacheng Tao6, Tianyi Zhou2 1The University of Hong Kong2University of Maryland3Microsoft 4University of Technology Sydney5Peking University6The University of Sydney {shawnxxh,chongyangtao,hishentao }@gmail.com {minglii,tianyi }@umd.edu ckcheng@cs.hku.hk jl0725@connect.hku.hk Abstract —In the era of Large Language Models (LLMs), Knowledge Distillation""",
   """advanced knowledge to smaller models and its utility in model compression and self- improvement. Our survey is meticulously structured around three foundational pillars: algorithm ,skill, and verticalization – providing a comprehensive examination of KD mechanisms, the enhancement of specific cognitive abilities, and their practical implications across diverse fields. Crucially, the survey navigates the intricate interplay between data augmentation (DA) and KD, illustrating how DA emerges as a p..."""
]


# Process the text.
cleaned_texts_list = text_cleaner(_EXTRACTED_TEXT)
podcast_metadata_list = title_and_summary_generator(cleaned_texts_list)
dialogue_texts_list = dialogue_writer(cleaned_texts_list)
persona_list = persona_generator(podcast_metadata_list)
altered_persona_list = persona_rewriter(persona_list)
class DialogueAndPersona(BaseModel):
    dialogue: CleanedText = Field(description="A cleaned text")
    persona: Persona = Field(description="A persona")
dialogue_and_persona_list = [DialogueAndPersona(dialogue=dialogue, persona=persona) for dialogue, persona in zip(dialogue_texts_list, altered_persona_list)]
rewritten_dialogue_list = podcast_rewriter(dialogue_and_persona_list)

# Print the results.
for podcast_metadata, dialogue, persona, altered_persona, rewritten_dialogue in zip(
    podcast_metadata_list, dialogue_texts_list, persona_list, altered_persona_list, rewritten_dialogue_list
):
    print(f"Title: {podcast_metadata.title}")
    print(f"Summary: {podcast_metadata.summary}")
    print(f"Persona:\n{persona.persona}")
    print(f"Altered Persona:\n{altered_persona.persona}")
    print("\n")
    print(f"Dialogue:\n{dialogue.text}")
    print("\n")
    print(f"Rewritten Dialogue:\n{rewritten_dialogue.text}")
    print("\n==============\n")
