import chess
import chess.engine
import time
import random
import os
import numpy as np
import lmstudio as lms
import math
#import matplotlib.pyplot as plt
#from terminalplot import plot

from openai import OpenAI
from typing import Optional
from pydantic import BaseModel

# A class based schema for a book
class MoveSchema(BaseModel):
    move: str

# -----------------------------------------------------------------------------
# Configuration & Parameters
# -----------------------------------------------------------------------------
MODEL_NAME = "gpt-4o"
#MODEL_NAME = "gpt-5-nano"
MAX_TOKENS = 1024
NUM_GAMES=10


#PLAY_HUMAN=False
#USE_LOCAL_LLM=True
#PURE_LLM=False


ALGORITHM_B='ENGINE'
#ALGORITHM_B='HUMAN'

#ALGORITHM_A='ALPHA-BETA'
ALGORITHM_A='ENGINE'
#ALGORITHM_B='1-STEP-LOOKAHEAD'
#ALGORITHM_B='HUMAN'
#ALGORITHM_B='LOCAL_LLM'
#ALGORITHM_B='CLOUD_AGENT'
#ALGORITHM_B='RANDOM_MOVE'

CURRICULAR_SPEED=200## AS the model learns, reduce the speed.
HANDICAP_FACTOR=1
MAX_TURNS_PER_GAME=200
GIVES_CHECK_HEURISTIC=0.90
TEMPERATURE=0.5

SYSTEM_PROMPT = "You are an algorithmic chess player. You provide answers in JSON format.  You will receive updates of the game and will follow algorithmic instructions. Good luck."


all_scores=[]


# Piece-Square Tables (White's perspective)
# Indexed as [rank][file] where rank 0 = rank 1, rank 7 = rank 8
# file 0 = a-file, file 7 = h-file

piece_square_table={
chess.PAWN : [
    [0,  0,  0,  0,  0,  0,  0,  0],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [10, 10, 20, 30, 30, 20, 10, 10],
    [5,  5, 10, 25, 25, 10,  5,  5],
    [0,  0,  0, 20, 20,  0,  0,  0],
    [5, -5,-10,  0,  0,-10, -5,  5],
    [5, 10, 10,-20,-20, 10, 10,  5],
    [0,  0,  0,  0,  0,  0,  0,  0]
],

chess.KNIGHT :  [
    [-50,-40,-30,-30,-30,-30,-40,-50],
    [-40,-20,  0,  0,  0,  0,-20,-40],
    [-30,  0, 10, 15, 15, 10,  0,-30],
    [-30,  5, 15, 20, 20, 15,  5,-30],
    [-30,  0, 15, 20, 20, 15,  0,-30],
    [-30,  5, 10, 15, 15, 10,  5,-30],
    [-40,-20,  0,  5,  5,  0,-20,-40],
    [-50,-40,-30,-30,-30,-30,-40,-50]
],

chess.BISHOP : [
    [-20,-10,-10,-10,-10,-10,-10,-20],
    [-10,  0,  0,  0,  0,  0,  0,-10],
    [-10,  0,  5, 10, 10,  5,  0,-10],
    [-10,  5,  5, 10, 10,  5,  5,-10],
    [-10,  0, 10, 10, 10, 10,  0,-10],
    [-10, 10, 10, 10, 10, 10, 10,-10],
    [-10,  5,  0,  0,  0,  0,  5,-10],
    [-20,-10,-10,-10,-10,-10,-10,-20]
],

chess.ROOK :  [
    [0,  0,  0,  0,  0,  0,  0,  0],
    [5, 10, 10, 10, 10, 10, 10,  5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [0,  0,  0,  5,  5,  0,  0,  0]
],

chess.QUEEN :  [
    [-20,-10,-10, -5, -5,-10,-10,-20],
    [-10,  0,  0,  0,  0,  0,  0,-10],
    [-10,  0,  5,  5,  5,  5,  0,-10],
    [-5,  0,  5,  5,  5,  5,  0, -5],
    [0,  0,  5,  5,  5,  5,  0, -5],
    [-10,  5,  5,  5,  5,  5,  0,-10],
    [-10,  0,  5,  0,  0,  0,  0,-10],
    [-20,-10,-10, -5, -5,-10,-10,-20]
],

chess.KING :  [
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-20,-30,-30,-40,-40,-30,-30,-20],
    [-10,-20,-20,-20,-20,-20,-20,-10],
    [20, 20,  0,  0,  0,  0, 20, 20],
    [20, 30, 10,  0,  0, 10, 30, 20]
],

'king_end_game' : [
    [-50,-40,-30,-20,-20,-30,-40,-50],
    [-30,-20,-10,  0,  0,-10,-20,-30],
    [-30,-10, 20, 30, 30, 20,-10,-30],
    [-30,-10, 30, 40, 40, 30,-10,-30],
    [-30,-10, 30, 40, 40, 30,-10,-30],
    [-30,-10, 20, 30, 30, 20,-10,-30],
    [-30,-30,  0,  0,  0,  0,-30,-30],
    [-50,-30,-30,-30,-30,-30,-30,-50]
]

}

def evaluate(board):
    """Evaluates a board from the WHITE perspective based on basic materiel and position."""

    # If game is over, return definitive scores
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            return -1000000  # Black won
        else:
            return +1000000  # White won
    
    if board.is_stalemate() or board.is_insufficient_material():
        return 0  # Draw
    
    score = 0
    
    # 1. MATERIAL COUNTING
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0  # King has no material value
    }
    
    for square_index, piece in board.piece_map().items():
        value = piece_values[piece.piece_type]
        #square_index = chess.parse_square(square) 
        
        positional_value = piece_square_table[piece.piece_type][chess.square_file(square_index)][chess.square_rank(square_index)]
        if piece.color == chess.WHITE:
            score += (value + positional_value)
        else:
            score -= (value + positional_value)
    
    # 3. POSITIONAL BONUSES (optional for "basic")
    # Mobility: count legal moves (more mobility = better)
    # Pawn structure: penalize doubled/isolated pawns
    # King safety: penalize exposed king in middlegame
    
    return score


'''
Next is work in progres
'''

def alphabeta(board, depth, alpha, beta, maximizing_player):
    """
    Alpha-beta pruning algorithm implementation.
    
    Args:
        board: chess.Board object representing the current position
        depth: Maximum search depth (plies)
        alpha: Best value maximizing player can guarantee
        beta: Best value minimizing player can guarantee
        maximizing_player: True if maximizing, False if minimizing
    
    Returns:
        The heuristic value of the position
    """
    # Terminal condition: depth reached or game over
    if depth == 0 or board.is_game_over():
        return evaluate(board)
        #info = engine.analyse(board, chess.engine.Limit(depth=20))
        #this_score=info['score'].white().score()
    
    if maximizing_player:
        value = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            value = max(value, alphabeta(board, depth - 1, alpha, beta, False))
            board.pop()
            
            if value >= beta:
                break  # Beta cutoff
            alpha = max(alpha, value)
        return value
    else:
        value = float('inf')
        for move in board.legal_moves:
            board.push(move)
            value = min(value, alphabeta(board, depth - 1, alpha, beta, True))
            board.pop()
            
            if value <= alpha:
                break  # Alpha cutoff
            beta = min(beta, value)
        return value


def find_best_move(board, depth):
    """
    Find the best move using alpha-beta pruning.
    
    Args:
        board: chess.Board object
        depth: Search depth (plies)
    
    Returns:
        Best move found
    """
    best_move = None
    best_value = float('-inf') if board.turn == chess.WHITE else float('inf')
    alpha = float('-inf')
    beta = float('inf')
    
    for move in board.legal_moves:
        board.push(move)
        
        if board.turn == chess.BLACK:  # We just made a white move
            value = alphabeta(board, depth - 1, alpha, beta, False)
            if value > best_value:
                best_value = value
                best_move = move
            alpha = max(alpha, value)
        else:  # We just made a black move
            value = alphabeta(board, depth - 1, alpha, beta, True)
            if value < best_value:
                best_value = value
                best_move = move
            beta = min(beta, value)
        
        board.pop()
    
    return best_move, best_value



'''
def alpha_beta_search(board, depth):
    """
    Main search function - finds the best move for the current position
    Returns: best_move (chess.Move object)
    """
    best_move = None
    alpha = float('-inf')
    beta = float('inf')
    
    for move in board.legal_moves:
        board.push(move)
        score = -alpha_beta(board, depth - 1, -beta, -alpha)
        board.pop()
        
        if score > alpha:
            alpha = score
            best_move = move
    
    return best_move


def alpha_beta(board, depth, alpha, beta):
    """
    Recursive alpha-beta search
    Returns: evaluation score from current player's perspective
    
    Args:
        board: chess.Board object
        depth: remaining search depth
        alpha: best score for maximizing player
        beta: best score for minimizing player
    """
    # Base case: reached maximum depth or game over
    if depth == 0 or board.is_game_over():
        return evaluate(board)
    
    max_score = float('-inf')
    
    for move in board.legal_moves:
        board.push(move)
        
        # Negamax framework: negate score from opponent's perspective
        score = -alpha_beta(board, depth - 1, -beta, -alpha)
        
        board.pop()
        
        max_score = max(max_score, score)
        alpha = max(alpha, score)
        
        # Beta cutoff: opponent won't allow this line
        if alpha >= beta:
            break  # Prune remaining moves
    
    return max_score


'''

def chess_agent(board,engine,a: list) -> str:
    """Given a list of legal moves returns one."""

    ## Openings
    ## Midgame
    ## Endgame << ---- Looks like winning needs to be addressed separately.
    

    columns_weights={'a':0.1,'b':0.2, 'c':0.3, 'd':0.4, 'e':0.4, 'f':0.3, 'g':0.2, 'h':0.1}
    rows_weights={'1':0.1,'2':0.2, '3':0.3, '4':0.4, '5':0.4, '6':0.3, '7':0.2, '8':0.1}
    #piece_weights={'R':0.15,'N':0.45, 'B':0.4, 'Q':0.3, 'K':0.01, 'P':0.3}
    piece_weights={'R':0.99,'N':0.99, 'B':0.99, 'Q':0.99, 'K':0.99, 'P':0.99}
    a_w=[]
    
    for i in a:
        p_type=board.piece_at(chess.parse_square(i[0:2])).symbol()
        
        move = chess.Move.from_uci(i)
        board.push(move)
        info = engine.analyse(board, chess.engine.Limit(depth=20))
        this_score=info['score'].white().score()
        try:
            counterfactual_score=math.exp(this_score/TEMPERATURE)
        except:
            counterfactual_score=0.0
        
        board.pop()

        #print("Counterfactual ", counterfactual_score)
        gives_check = GIVES_CHECK_HEURISTIC if  board.gives_check(move) else (1.0 - GIVES_CHECK_HEURISTIC)
        

        ##. This is an alternative basic score

        #print(p_type,type(p_type))
        #a_w.append(columns_weights[i[2]]*rows_weights[i[3]]*piece_weights[p_type]*gives_check*counterfactual_score)
        a_w.append(0.0*columns_weights[i[2]]*rows_weights[i[3]]*piece_weights[p_type]*gives_check+ counterfactual_score)
        #print(a_w)
    print("PROBABILITIES",a_w)
    print(">>>>>>>. EVAL. EVAL EVAL >>>>>>",evaluate(board))

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

def play_timed_chess_match(engine_path, time_per_player_seconds=3):
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

    algorithm_b=ALGORITHM_B
    algorithm=ALGORITHM_A
    
    #plt.ion() 
    #plt.plot([0])
    #plt.show(block=False)    

    if algorithm == 'CLOUD_AGENT':

        client = OpenAI()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}  # Add the initial question
            ]   
        print("Starting OpenAI Agent with Messages:", messages)
 
    if algorithm == 'LOCAL_LLM':
        model = lms.llm("gemma-3-12b-it-qat")
  

    raven_time_left = time_per_player_seconds
    engine_time_left = time_per_player_seconds/HANDICAP_FACTOR
    current_game_moves=0

    print("Welcome to RaVEN Chess!")
    print(f"Each player starts with {time_per_player_seconds // 60} minutes.")
    print("Moves are specified in standard algebraic notation (e.g., 'e2e4', 'Nf3').")
    found_error=False

    while (not board.is_game_over()) and current_game_moves<MAX_TURNS_PER_GAME and found_error is False:
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
                    print(f"ALGORITHM {algorithm}")

                    if algorithm == 'HUMAN':
                        next_move_2 = input("Human RaVEN move: ")
                        move = board.parse_san(next_move_2)

                    elif algorithm == 'ENGINE':
                        result = engine.play(board, chess.engine.Limit(time=raven_time_left / 100)) # Adjust engine thinking time
                        next_move_2 = result.move
                        move=result.move


                    elif algorithm == 'CLOUD_AGENT':
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
                        move = board.parse_san(next_move_2)

                    
                    elif algorithm == 'LOCAL_LLM':
                        result = model.respond(f"Select the next move from this list. Return just the move: {legal_moves_list}",response_format=MoveSchema)
                        next_move_2=(result.parsed)["move"].replace("'","").replace("\"","")
                        move = board.parse_san(next_move_2)


                    elif algorithm == 'ALPHA-BETA':
                        #next_move_2 = alpha_beta_search(board, depth=4)
                        next_move_2, s2 = find_best_move(board,depth=6)
                        move=next_move_2


                    elif algorithm == '1-STEP-LOOKAHEAD':
                        next_move_2=chess_agent(board,engine,legal_moves_list)
                        move = board.parse_san(next_move_2)
                    

                    else:
                        next_move_2=random.choice(legal_moves_list)
                        move = board.parse_san(next_move_2)

                    print(f"RaVEN {algorithm} Move: {next_move_2}")
                    
                    
                    if move in board.legal_moves:
                        print("Im here")
                        board.push(move)
                        break
                    else:
                        print("Illegal move. Bailing.")
                        found_error=True
                        break

                except ValueError:
                    print("Invalid move format or illegal move. Bailing out.")
                    found_error=True
                    break

            end_time = time.time()
            raven_time_left -= (end_time - start_time)
            if raven_time_left <= 0:
                print("Time's up! RaVEN loses on time.")
                break
        else:
            # Engine's turn
            start_time = time.time()
            
            if algorithm_b == 'ENGINE':
                result = engine.play(board, chess.engine.Limit(time=engine_time_left / 100)) # Adjust engine thinking time
                next_move_2 = result.move
                board.push(result.move)

            if algorithm_b == 'HUMAN':
                next_move_2 = input("Human ENBINE move: ")
                move = board.parse_san(next_move_2)
                board.push(move)
            
            end_time = time.time()
            engine_time_left -= (end_time - start_time)

            print(f"Engine played: {next_move_2} with algorithm {algorithm_b}")
            if engine_time_left <= 0:
                print("Engine loses on time!")
                break
        current_game_moves+=1

    print("\n" + "=" * 30)
    print("Game Over!")
    if (current_game_moves>=MAX_TURNS_PER_GAME) or found_error is True:
        result=0.1
    else:
        text_result=str(board.result())
        print(text_result)
        if text_result=="1-0":
            result=1
        elif text_result=="1/2-1/2":
            result=0.5
        else:
            result=0
    engine.quit()
    return result

if __name__ == "__main__":

    # Path to Stockfish executable
    stockfish_path = "/opt/homebrew/bin/stockfish" 
    results=[]    

    for i in range(NUM_GAMES):
        try:
            result=play_timed_chess_match(stockfish_path, time_per_player_seconds=CURRICULAR_SPEED)
            results.append(result)
            long_mean=np.mean(results)
            if i>10:
                short_mean=np.mean(results[-10:])
            else:
                short_mean=long_mean
            print(f"Game {i}, {long_mean} {short_mean} {results}")
        except FileNotFoundError:
            print(f"Error: Chess engine not found at '{stockfish_path}'.")
            print("Please ensure the path is correct and the engine is installed.")
        except Exception as e:
            print(f"An error occurred: {e}")
