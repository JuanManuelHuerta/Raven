#!/usr/bin/env python3
import json
import os
import re
from typing import List

import requests
from bs4 import BeautifulSoup
import openai
from openai import OpenAI
import re
import requests
from typing import Optional

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
    match = re.search(r'Search\[(.*?)\]', action_string)
    if match:
        return match.group(1).strip()
    return None



# Option 1: Using DuckDuckGo (free, no API key needed)
def web_search(query: str, max_results: int = 10) -> str:
    """
    Search the web using DuckDuckGo's HTML interface.
    Returns a formatted string with top results.
    """
    try:
        from duckduckgo_search import DDGS
        print("DUCKDUCK SEARCH",query)
        results = []
        with DDGS() as ddgs:
            for i, result in enumerate(ddgs.text(query, max_results=max_results)):
                title = result.get('title', 'No title')
                snippet = result.get('body', 'No description')
                url = result.get('href', '')
                results.append(f"{i+1}. {title}\n   {snippet}\n   URL: {url}")
        
        if not results:
            return "No results found."
        
        return "\n\n".join(results)
    
    except Exception as e:
        return f"Search error: {str(e)}"


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_NAME = "gpt-4o"
MAX_TOKENS = 1024
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ReActNav/1.0)"}

# -----------------------------------------------------------------------------
# 1. Accessibility-tree extraction
# -----------------------------------------------------------------------------
def get_accessibility_tree(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    lines: List[str] = []
    EXCLUDE = {"script", "style", "head", "meta", "link", "iframe", "noscript",
               "form", "select", "option", "nav", "header", "footer"}
    def traverse(node, depth: int = 0):
        for child in node.children:
            name = getattr(child, "name", None)
            if not name:
                continue
            if name in EXCLUDE:
                traverse(child, depth + 1)
                continue
            text = ""
            if child.string and isinstance(child.string, str):
                text = child.string.strip()[:60]
            role = child.attrs.get("role")
            aria = child.attrs.get("aria-label")
            # skip nodes without any visible text, role, or aria-label
            if not (text or role or aria):
                traverse(child, depth + 1)
                continue
            indent = " " * (depth * 2)
            line = f"{indent}<{name}>"
            if role:
                line += f" role={role}"
            if aria:
                line += f" aria-label=\"{aria}\""
            if text:
                line += f" “{text}”"
            lines.append(line)
            traverse(child, depth + 1)
    root = soup.body or soup
    traverse(root)
    return "\n".join(lines)


def extract_action(text):
    # Parse "Action: Search[query]" format
    match = re.search(r'\"Action\":\s*\"(\w+\[.*?\])\"', text)
    return match.group(1) if match else None

def extract_answer(text):
    # Parse "Answer: ..." format
    match = re.search(r'Answer:\s*(.*)', text, re.DOTALL)
    return match.group(1).strip() if match else None

def execute_action(action_string):
    if action_string.startswith("Search["):
        query = extract_query(action_string)
        return web_search(query)
    return "Nothing"
# -----------------------------------------------------------------------------
# 2. ReAct loop
# -----------------------------------------------------------------------------
PROMPT_PATH = os.path.join(os.path.dirname(__file__), "react_prompt.txt")
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
        if "Answer:" in content:
            print(extract_answer(content))
            break
    
        print("CONTENT:",content)

        action = extract_action(content)  # e.g., "Search[Paris population]"
        print("Contnet", content)
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
    # q = "How many states have a population larger than Beijing?"
    q = "Is Ethereum price correlated with Bitcoin price?" 
    # q = "What papers did Nicholas Tomlin write in 2024 according to his academic website?"
    print("Question:", q)
    print("Answer:", react_agent(q))


if __name__ == "__main__":
    main()