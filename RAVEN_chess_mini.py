import chess
import chess.engine
import chess.polyglot
import time
import random
import os
import numpy as np
import math
from typing import Optional
from pydantic import BaseModel

# A class based schema for a book
class MoveSchema(BaseModel):
    move: str

# -----------------------------------------------------------------------------
# Configuration & Parameters
# -----------------------------------------------------------------------------
NUM_GAMES=10

#------------------------------------------------------------
# KOMODO, baby!
#------------------------------------------------------------
PATH_TO_OPENINGS = "data/polyglot/komodo.bin"

CURRICULAR_SPEED=2000.    ## As the model learns, reduce the speed.
MAX_SECS_FOR_ENGINE=0.5  ## Cap the engine's time
ALPHA_BETA_DEPTH=3
HANDICAP_FACTOR=100
MAX_TURNS_PER_GAME=200
N_STEP_LOOKAHEAD=18
GIVES_CHECK_HEURISTIC=0.90
TEMPERATURE=0.5


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
        rank = chess.square_rank(square_index)
        file = chess.square_file(square_index)

    # Flip board for black pieces
        if piece.color == chess.BLACK:
            rank = 7 - rank

        value = piece_values[piece.piece_type]
        #square_index = chess.parse_square(square) 
        
        #positional_value = piece_square_table[piece.piece_type][chess.square_file(square_index)][chess.square_rank(square_index)]
        positional_value = piece_square_table[piece.piece_type][rank][file]
        if piece.color == chess.WHITE:
            score += (value + positional_value)
        else:
            score -= (value + positional_value)
    
    # 3. POSITIONAL BONUSES (optional for "basic")
    # Mobility: count legal moves (more mobility = better)
    # Pawn structure: penalize doubled/isolated pawns
    # King safety: penalize exposed king in middlegame
    
    return score




def get_weighted_book_move(board, book_path):
    """
    Get a move from the opening book with weighted random selection.
    This adds variety to the opening play.
    
    Args:
        board: chess.Board object
        book_path: Path to the polyglot opening book file (.bin)
    
    Returns:
        A weighted random move from the book, or None if position not in book
    """
    try:
        with chess.polyglot.open_reader(book_path) as reader:
            # weighted_choice will pick a move based on weights in the book
            move = reader.weighted_choice(board).move
            return move
    except (FileNotFoundError, IndexError):
        pass
    return None


def play_with_book(board, book_path):
    """
    Make a move using opening book if available, otherwise use alpha-beta.
    
    Args:
        board: chess.Board object
        depth: Search depth for alpha-beta (if not using book)
        book_path: Path to the polyglot opening book file
        use_weighted: If True, use weighted random selection from book
    
    Returns:
        (move, source) tuple where source is "book" or "search"
    """
    # Try to get a move from the opening book
    book_move = get_weighted_book_move(board, book_path)
     
    if book_move:
        print("!!!!!!!!!!!!!!!!!!!!!  DOING BOOK MOVE !!!!!!!!!!!!!!!!!!!!!!!!!!")
        return book_move, "book"
    else:
        return -1, "book"
    # If not in book, use alpha-beta search
    #best_move, _ = find_best_move(board, depth)
    #return best_move, "search"


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
        info = engine.analyse(board, chess.engine.Limit(depth=N_STEP_LOOKAHEAD))
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
def play_timed_chess_match(engine_path, time_per_player_seconds=3):
    """
    Plays a timed chess match between a RaVEN Agent and a UCI-compatible chess engine.

    Args:
        engine_path (str): The path to the UCI-compatible chess engine executable.
        time_per_player_seconds (int): The initial time in seconds for each player.
    """
    global transposition_table

    question = "Pick a move at random. I will keep updating the options. Just pick one."
    board = chess.Board()
    engine = chess.engine.SimpleEngine.popen_uci("/opt/homebrew/bin/stockfish")
    current_game_moves=0
    global all_scores


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
        transposition_table={}
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
                    print(f"Info!!! {current_score}, {all_scores}")
                    print(f"Info!!! ",np.mean(all_scores), np.std(all_scores))
                    print("Legal moves:",legal_moves_list)
                    next_move_2, s2 = play_with_book(board,book_path=PATH_TO_OPENINGS)
                    move=next_move_2
                    if next_move_2 == -1:
                        next_move_2=chess_agent(board,engine,legal_moves_list)
                        move = board.parse_san(next_move_2)

                    else:
                        next_move_2=random.choice(legal_moves_list)
                        move = board.parse_san(next_move_2)
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
            
            move_time=min(raven_time_left/100,MAX_SECS_FOR_ENGINE)
            #result = engine.play(board, chess.engine.Limit(move_time)) # Adjust engine thinking time
            result = engine.play(board, chess.engine.Limit(depth=4)) # Adjust engine thinking time
            #chess.engine.Limit(depth=20)
            next_move_2 = result.move
            board.push(result.move)
            
            end_time = time.time()
            engine_time_left -= (end_time - start_time)

            print(f"Engine played: {next_move_2}")
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
        if text_result=="0-1":
            result=0
        elif text_result=="1/2-1/2":
            result=0.5
        else:
            result=1.0  # Most likely a timeout for the blacks.
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
