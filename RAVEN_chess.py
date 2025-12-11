import chess
import chess.engine
import time
import random
import os
import numpy as np
import lmstudio as lms
#import matplotlib.pyplot as plt
#from terminalplot import plot

from openai import OpenAI
from typing import Optional

from pydantic import BaseModel

# A class based schema for a book
class MoveSchema(BaseModel):
    move: str

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_NAME = "gpt-4o"
#MODEL_NAME = "gpt-5-nano"
MAX_TOKENS = 1024
PLAY_HUMAN=False
USE_LOCAL_LLM=True
PURE_LLM=False
SYSTEM_PROMPT = "You are an algorithmic chess player. You provide answers in JSON format.  You will receive updates of the game and will follow algorithmic instructions. Good luck."



all_scores=[]


def chess_agent(a: list) -> str:
    """Given a list of legal moves returns one."""
    columns_weights={'a':0.1,'b':0.2, 'c':0.3, 'd':0.4, 'e':0.4, 'f':0.3, 'g':0.2, 'h':0.1}
    rows_weights={'1':0.1,'2':0.2, '3':0.3, '4':0.4, '5':0.4, '6':0.3, '7':0.2, '8':0.1}
    a_w=[]
    for i in a:
        a_w.append(columns_weights[i[2]]*rows_weights[i[3]])
    next_move_2=random.choices(a,weights=a_w)[0]
    return next_move_2

def raven_chess(model,legal_moves_list) -> list:
    response=model.act(
                        f"Pick a legal move from {legal_moves_list}",
                        [chess_agent],
                        on_message=print,
                        )
    next_move_2="test"
    print("OBJECT",dir(model._session))
    '''
    for message in response.history:
        if message["role"] == "assistant":
            print(f"Assistant's response from history: {message['content']}")
        elif message["role"] == "tool_result":
            print(f"Tool result from history: {message['content']}")
    '''

    ## TODO: still need to handle correctly the action output of the model
    #print("RESPONSE", assistant_reponse)
    return next_move_2

def play_timed_chess_match(engine_path, time_per_player_seconds=300):
    """
    Plays a timed chess match between a RaVEN Agent and a UCI-compatible chess engine.

    Args:
        engine_path (str): The path to the UCI-compatible chess engine executable.
        time_per_player_seconds (int): The initial time in seconds for each player.
    """
    question = "Pick a move at random. I will keep updating the options. Just pick one."
    board = chess.Board()
    engine = chess.engine.SimpleEngine.popen_uci("/opt/homebrew/bin/stockfish")
    global all_scores

    #plt.ion() 
    #plt.plot([0])
    #plt.show(block=False)    
    if USE_LOCAL_LLM is False:

        client = OpenAI()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}  # Add the initial question
            ]   
        print("Starting OpenAI Agent with Messages:", messages)
    else:
        model = lms.llm("gemma-3-12b-it-qat")
  

    raven_time_left = time_per_player_seconds
    engine_time_left = time_per_player_seconds

    print("Welcome to RaVEN Chess!")
    print(f"Each player starts with {time_per_player_seconds // 60} minutes.")
    print("Moves are specified in standard algebraic notation (e.g., 'e2e4', 'Nf3').")

    while not board.is_game_over():
        print("\n" + "=" * 30)
        print(board)
        print(f"RaVEN Time Left: {raven_time_left:.1f}s | Engine Time Left: {engine_time_left:.1f}s")

        if board.turn == chess.WHITE:
            # RaVEN's turn
            start_time = time.time()
            while True:
                try:
                    legal_moves_iterator = board.legal_moves
                    legal_moves_list = [move.uci() for move in legal_moves_iterator]
                    info = engine.analyse(board, chess.engine.Limit(depth=20))
                    current_score=info['score'].white().score()
                    if type(current_score) is int:
                        all_scores.append(current_score)
                    #plt.plot(all_scores)
                    #plt.draw()
                    #plot(all_scores)
                    print(f"Info!!! {current_score}, {all_scores}")
                    print(f"Info!!! ",np.mean(all_scores), np.std(all_scores))
                    print("Legal moves:",legal_moves_list)
                    if PLAY_HUMAN == True:
                        next_move_2 = input("Human RaVEN move: ")
                    else:
                        print("RaVEN Thinking")
                        #next_move_2=raven_chess(model,legal_moves_list)    
                        if USE_LOCAL_LLM is False:

                            messages.append({"role": "user", "content": str(legal_moves_list)})
                            print("MESSAGES",messages)
                            response = client.chat.completions.create(
                                model=MODEL_NAME,
                                messages=messages,
                                temperature=0.0,
                                response_format={"type": "json_object"} 
                                )

                            #content = next_iter_input
                            content=eval(response.choices[0].message.content)
                            print("RESPONESE:",content)
                            #next_move_2=random.choice(legal_moves_list).replace("'","").replace("\"","")
                            next_move_2=content["move"].replace("'","").replace("\"","")
                        else:

                            if PURE_LLM is True:
                                result = model.respond(f"Select the next move from this list. Return just the move: {legal_moves_list}",response_format=MoveSchema)
                                next_move_2=(result.parsed)["move"].replace("'","").replace("\"","")
                                print(f"LocalLLM picked {next_move_2}")

                            else:
                                next_move_2=chess_agent(legal_moves_list)

                        print(f"RaVEN Move: {next_move_2}")
                    
                    move = board.parse_san(next_move_2)
                    if move in board.legal_moves:
                        board.push(move)
                        break
                    else:
                        print("Illegal move. Try again.")
                except ValueError:
                    print("Invalid move format or illegal move. Try again.")
            end_time = time.time()
            raven_time_left -= (end_time - start_time)
            if raven_time_left <= 0:
                print("Time's up! RaVEN loses on time.")
                break
        else:
            # Engine's turn
            start_time = time.time()
            result = engine.play(board, chess.engine.Limit(time=engine_time_left / 100)) # Adjust engine thinking time
            board.push(result.move)
            end_time = time.time()
            engine_time_left -= (end_time - start_time)
            print(f"Engine played: {result.move}")
            if engine_time_left <= 0:
                print("Engine loses on time!")
                break

    print("\n" + "=" * 30)
    print("Game Over!")
    print(board.result())
    if board.result()=="1-0":
        result=1
    else:
        result=0
    engine.quit()
    return result

if __name__ == "__main__":

    # Path to Stockfish executable
    stockfish_path = "/opt/homebrew/bin/stockfish" 
    results=[]    

    for i in range(100):
        try:
            result=play_timed_chess_match(stockfish_path, time_per_player_seconds=300)
            results.append(result)
            long_mean=np.mean(results)
            if i>10:
                short_mean=np.mean(results[-10:])
            else:
                short_mean=long_mean
            print(f"Game {i}, {long_mean} {short_mean}")
        except FileNotFoundError:
            print(f"Error: Chess engine not found at '{stockfish_path}'.")
            print("Please ensure the path is correct and the engine is installed.")
        except Exception as e:
            print(f"An error occurred: {e}")
