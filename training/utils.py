"""
This module contains utility functions for training.
"""
import json
import os
from openai import OpenAI

# Use tokenizer to ensure that the prompt fits within the model's token limit
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Set API keys
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

openai_client = OpenAI(
    api_key = os.environ['OPENAI_API_KEY']
)


def save_json(data, filename):
    """
    Save data to a JSON file.

    Args:
        data (any): Data to save.
        filename (str): Name of the file to save the data.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_json(filename):
    """
    Load data from a JSON file.

    Args:
        filename (str): Name of the file to load the data.
    """
    with open(filename, encoding = 'utf-8') as f:
        data = json.load(f)
    return data

def evaluate_with_gpt3(question, true_answer, predicted_answer):
    """
    Evaluate the predicted answer with GPT-3.5-turbo.

    Args:
        question (str): The question.
        true_answer (str): The ground truth answer.
        predicted_answer (str): The predicted answer.
    
    Returns:
        int: The score given by the evaluator.
    """
    while True:
        try:
            prompt = (
                f"Evaluate the following question and answers. If the predicted answer is correct in meaning, respond with '1'. "
                f"If it is incorrect, respond with '0'.\n\n"
                f"Question: {question}\n"
                f"Ground Truth Answer: {true_answer}\n"
                f"Predicted Answer: {predicted_answer}\n"
                f"Is the predicted answer correct? (1 for yes, 0 for no):"
            )

            completion = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI. Please score the following question and answers.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0.5,
                max_tokens=1000,
                n=1,
            )
            score = int(completion.choices[0].message.content)
            return score
        except Exception as e:
            print(e)
            continue

def truncate_prompt(prompt, question, retrieved_context, max_tokens_allowed):
    """
    Truncate the prompt if it exceeds the maximum token limit.

    Args:
        prompt (str): Prompt to be truncated.
        question (str): Question to be used in the prompt.
        retrieved_context (str): Retrieved context to be used in the prompt.
        max_tokens_allowed (int): Maximum number of tokens allowed in the prompt.

    Returns:
        str: Truncated prompt.
    """
    tokens = tokenizer.encode(prompt)
    if len(tokens) > max_tokens_allowed:
        excess_tokens = len(tokens) - max_tokens_allowed
        context_tokens = tokenizer.encode(retrieved_context)
        truncated_context_tokens = context_tokens[:-excess_tokens]
        truncated_context = tokenizer.decode(truncated_context_tokens)
        prompt = (
            f"<s>[INST] <<SYS>>\nYou are an AI. Please answer the following question as briefly and concisely as possible, using as few words as possible. "
            f"For example, if the question is 'What year was George Washington born?' please answer '1732'. Use the following context to help answer the question:<</SYS>>\n"
            f"Context:\n{truncated_context}\nQuestion:\n{question} [/INST]"
        )
    return prompt