#!/usr/bin/env python3
import json
import os
import re
from typing import List

import requests
import openai
from openai import OpenAI
from typing import Optional

import yfinance as yf
import pandas as pd
import ast

# -----------------------------------------------------------------------------
# Extract query from Action string
# -----------------------------------------------------------------------------
def extract_query(action_string: str) -> Optional[str]:
    """
    Extract query from action strings like:
    - "Search[Paris population]"
    - "Search[capital of France]"
    
    Returns the text inside brackets, or None if not found.
    """
    match = re.search(r'Lookup\[(.*?)\]$', action_string)
    if match:
        return match.group(1).strip()
    
    match = re.search(r'Calculate\[(.*?)\]$', action_string)
    if match:
        return match.group(1).strip()
    
    return None



# Option 1: Using DuckDuckGo (free, no API key needed)
def instrument_lookup(ticker_symbol, max_results: int = 10) -> list:
    """
    Use Yahoo Finance to look for instruments price histories.
    """
    try:
        results = yf.download(ticker_symbol, period="5d")['Close'].values
        return results
    except Exception as e:
        return f"Search error: {str(e)}"


def computation(calculation_string:str) -> str:
    #print(f"CALCULATE {calculation_string}")
    match = re.search(r'([A-Z]+)\, \[(.*?)\]$', calculation_string)
    if match:
        cs1= match.group(1).strip()
        cs2="["+ match.group(2).strip()+"]"
        prices = ast.literal_eval(cs2)
        average = sum(prices) / len(prices)
        return f"The average price of {cs1} is {average}"
    return "The average price is 0.0"

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_NAME = "gpt-4o"
MAX_TOKENS = 1024

def extract_action(text):
    # Parse "Action: Search[query]" format
    match = re.search(r'\"Action\":\s*\"(\w+\[.*?\])\"', text)
    return match.group(1) if match else None

def extract_answer(content):
    # Parse "Answer: ..." format
    return content['Answer']

def execute_action(action_string):
    if action_string.startswith("Lookup["):
        query = extract_query(action_string)
        print("EXTRACTED TICKER",query)
        return instrument_lookup(query)
    if action_string.startswith("Calculate["):
        query = extract_query(action_string)
        return computation(query)

    return "Nothing"
# -----------------------------------------------------------------------------
# 2. ReAct loop
# -----------------------------------------------------------------------------
PROMPT_PATH = os.path.join(os.path.dirname(__file__), "react_raven_prompt.txt")
with open(PROMPT_PATH, "r", encoding="utf-8") as fp:
    SYSTEM_PROMPT = fp.read()

def is_total_complete(in_text: str) -> bool:
    if str("TASK_COMPLETE") in in_text:
        return True
    return False

def react_agent(question: str, *, max_steps: int = 10) -> str:
    ## BEGIN SOLUTION
    client = OpenAI()

    # Example API usage:
    # response = client.chat.completions.create(
    #     model=MODEL_NAME,
    #     messages=messages,
    #     temperature=0.0,
    #     response_format={"type": "json_object"}
    # )
    # content = response.choices[0].message.content
    content = ""
    n_steps=10
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}  # Add the initial question
        ]   
    print("Starting Agent with Messages:", messages)
    
    for step in range(n_steps):
        response = client.chat.completions.create(
         model=MODEL_NAME,
         messages=messages,
         temperature=0.0,
         response_format={"type": "json_object"} 
         )
        
        #content = next_iter_input
        content=response.choices[0].message.content

        if "Answer" in content:
            print("ANSWER FOUND!!", content)
            return(json.loads(content)['Answer'])
            
        
        print("Contnet", content)
        action = extract_action(content)  # e.g., "Search[Paris population]"
        print("Action", action)
        observation = execute_action(action)
        
        # Add to conversation history
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": f"Observation: {observation}"})


        print ("Completed Iteration:",messages)
        user_input = input("Please HIT enter ")
        
    ## END SOLUTION

# -----------------------------------------------------------------------------
# 3. Demo – run:  python web_nav_agent.py
# -----------------------------------------------------------------------------

def main() -> None:
    """Tiny demo to illustrate usage."""
    #q = "What is the average closing price of AMZN at the close of market in the last week?"
    #q = "Which of the magnificent 7 stocks has shown a higher mean in the closing price of their stock last week?"
    q = "Which of these companies: Apple, Amazon or Microsoft has shown a higher mean in the closing price of their stock last week?"
    #q = "Is Ethereum price correlated with Bitcoin price?" 
    # q = "What papers did Nicholas Tomlin write in 2024 according to his academic website?"
    print("Question:", q)
    print("Answer:", react_agent(q))


if __name__ == "__main__":
    main()