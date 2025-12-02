#!/usr/bin/env python3
import json
import os
import re
from typing import List
from statistics import variance
from openai import OpenAI
from typing import Optional

import yfinance as yf
import pandas as pd
import ast
import numpy as np


class FinancialWorld:
    def __init__(self,input_span):
        self.span=input_span

    @property
    def span(self):
        return self._span
    
    @span.setter
    def span(self,input_span):
        self._span=input_span

# -----------------------------------------------------------------------------
# Extract query from Action string
# -----------------------------------------------------------------------------
def extract_query(action_string: str) -> Optional[str]:
    """
    Extract query from action strings like:
    - "Lookup[AAPL]" --> "AAPL"
    - "Calculate[[1.0,2.0,3.0]]]"--> "[1.0,2.0,3.0]"
    Returns the text inside brackets, or None if not found.
    """

    match = re.search(r'Lookup\[(.*?)\]$', action_string)
    if match:
        return match.group(1).strip()
    
    match = re.search(r'Average\[(.*?)\]$', action_string)
    if match:
        return match.group(1).strip()
    
    match = re.search(r'Momentum\[(.*?)\]$', action_string)
    if match:
        return match.group(1).strip()
    
    match = re.search(r'Ratio\[(.*?)\]$', action_string)
    if match:
        return match.group(1).strip()
    
    match = re.search(r'Volatility\[(.*?)\]$', action_string)
    if match:
        return match.group(1).strip()
    
    match = re.search(r'TimeDifference\[(.*?)\]$', action_string)
    if match:
        return match.group(1).strip()
    

    return None


# Instrument Lookup using Yahoo finance. Provide Ticker Symbol.
def instrument_lookup(ticker_symbol, max_results ,context ) -> list:
    """
    Use Yahoo Finance to look for instruments price histories.
    """
    try:
        #print(f"LOOKING UP<<<< {context.span}")
        if context is not None:
            span=str(context.span)+"d"
        else:
            span="5d"

        results = yf.download(ticker_symbol, period=span)['Close'].values
        return results
    except Exception as e:
        return f"Search error: {str(e)}"


def average(calculation_string:str) -> str:
    """
    Computes basic time series average.
    """
    #print(f"CALCULATE {calculation_string}")
    match = re.search(r'([A-Z]+)\,\s*\[(.*?)\]$', calculation_string)
    if match:
        cs1= match.group(1).strip()
        cs2="["+ match.group(2).strip()+"]"
        prices = ast.literal_eval(cs2)
        average = sum(prices) / len(prices)
        return f"The average price of {cs1} is {average}"
    return "The average price is 0.0"


def volatility(calculation_string:str) -> str:
    """
    Computes basic time series volatility.
    """
    #print(f"CALCULATE {calculation_string}")
    match = re.search(r'([A-Z]+)\,\s*\[(.*?)\]$', calculation_string)
    if match:
        cs1= match.group(1).strip()
        cs2="["+ match.group(2).strip()+"]"
        prices = ast.literal_eval(cs2)
        volatility = variance(prices) 
        return f"The average volatility of {cs1} is {volatility}"
    return f"The average volatility of {cs1} is 0.0"


def momentum(calculation_string:str) -> str:
    """
    Computes basic momentum. Takes a time series of prices as input.
    """
    epsilon=0.9
    #print(f"MOMENTUM {calculation_string}")
    match = re.search(r'([A-Z]+)\,\s*\[(.*?)\]$', calculation_string)
    if match:
        cs1= match.group(1).strip()
        cs2="["+ match.group(2).strip()+"]"
        prices = ast.literal_eval(cs2)
        momentum = prices[-1]+ epsilon*(prices[-1]-prices[0])
        return f"The momentum of {cs1} is {momentum}"
    return f"The momentum of {cs1} is 0.0"



def time_difference(calculation_string:str) -> str:
    """
    Computes time differences. Takes a time series and finds time differences.
    """
    epsilon=0.9
    match = re.search(r'([A-Z]+)\,\s*\[(.*?)\]$', calculation_string)
    if match:
        cs1= match.group(1).strip()
        cs2="["+match.group(2).strip()+"]"
        print(f"CS2!!!! {cs2}")
    
        td = np.diff(np.array(ast.literal_eval(cs2)).flatten()).tolist()
    
        return f"The time difference of {cs1} is {td}"
    return f"The momentum of {cs1} is 0.0"



def ratio(calculation_string:str) -> str:
    """
    Computes  ratio. Takes ticker name,  value1, ticker name 2, value 2 and returns the ratio of the two values.
    """
    match = re.search(r'([A-Z]+)\,\s*([\-0-9\.]+)\,\s*([A-Z]+)\,\s*([\-0-9\.]+).*?$', calculation_string)
    if match:
        try:
            t1= match.group(1).strip()
            v1=float(match.group(2).strip())
            t2= match.group(3).strip()
            v2=float(match.group(4).strip())
            ratio=v1/v2
            print(f"Im here!!!{ratio}")
            return f"The ratio of {t1} to {t2} is {ratio}"
        except:
            return f"The ratio of {t1} to {t2}  is 0.0"


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_NAME = "gpt-4o"
#MODEL_NAME = "gpt-5-nano"
MAX_TOKENS = 1024

def extract_action(text):
    # Parse "Action: Search[query]" format
    match = re.search(r'\"Action\":\s*\"(\w+\[.*?\])\"', text)
    return match.group(1) if match else None

def extract_answer(content):
    # Parse "Answer: ..." format
    return content['Answer']

def execute_action(action_string,my_context):
    if action_string.startswith("Lookup["):
        query = extract_query(action_string)
        print("EXTRACTED TICKER",query)
        return instrument_lookup(query,10,my_context)
    if action_string.startswith("Average["):
        query = extract_query(action_string)
        return average(query)
    if action_string.startswith("Momentum["):
        query = extract_query(action_string)
        return momentum(query)
    if action_string.startswith("Ratio["):
        query = extract_query(action_string)
        return ratio(query)
    if action_string.startswith("Volatility["):
        query = extract_query(action_string)
        return volatility(query)
    if action_string.startswith("TimeDifference["):
        query = extract_query(action_string)
        return time_difference(query)

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

def react_agent(question: str, max_steps , my_context) -> str:
    ## BEGIN SOLUTION
    client = OpenAI()
    content = ""
    n_steps=max_steps
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
        action = extract_action(content)  
        print("Action", action)
        
        if "Answer" in content:
            #print("ANSWER FOUND!!", content)
            return(json.loads(content)['Answer'])
        elif action is None:
            print(f"REACHED A None Action. Content follows \n{content}")
            return(json.loads(content))
            
        observation = execute_action(action,my_context)
        print(f"observation {observation} step {step}")
        
        # Add to conversation history
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": f"Observation: {observation}"})


        #print ("Completed Iteration:",messages)
        #user_input = input("Please HIT enter ")
        
   

def main() -> None:
    """RAVEN Examples."""
    my_context=FinancialWorld(10)
    #q = "What is the average closing price of AMZN at the close of market in the last week?"
    #q = "Which of the magnificent 7 stocks has shown a higher mean in the closing price of their stock last week?"
    #q = "Which of these companies: Apple, Amazon or Microsoft has shown a higher mean in the closing price of their stock last week?"
    #q = "Which of these two companies: Apple and  Amazon has higher momentum in their closing prices in the last 7 days?"
    #q = "Which of these two companies: Apple and Amazon has a higher ratio of momentum to average closing price in the last 7 days?"
    #q = "Find the company in the 3 top hyperscalars that has the highest ratio of momentum to average in the last 7 days and the one with the lowest."
    #q = "Find the company in the 3 top hyperscalars that has the highest ratio of momentum to volatility in the last 7 days and the one with the lowest."
    #q = "Find the company in the 3 top hyperscalars that has the highest momentum of the time difference in its closing prices in the last 7 days and the one with the lowest."
    q = "Is Nvidia mean of the time difference of the closing day prices for the last 10 days higher than Oracle?"
    print("Question:", q)
    print("Answer:", react_agent(q,30,my_context))


if __name__ == "__main__":
    main()