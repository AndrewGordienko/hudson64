import chess
import torch
from tqdm import tqdm

from hudson64.core._mcts.mcts import mcts_search
from hudson64.core._network.model import AlphaZeroNet


def play_single_game(model_white, model_black, sims=50, game_index=None):
    board = chess.Board()

    move_num = 1
    while not board.is_game_over():
        model = model_white if board.turn == chess.WHITE else model_black
        move = mcts_search(board, model, simulations=sims, return_pi=False)

        if move is None:
            break

        board.push(move)

        move_num += 1

    result = board.result()

    if result == "1-0":
        return 1
    elif result == "0-1":
        return -1
    else:
        return 0


def evaluate_models(old_path, new_path, games=20, sims=50):
    """
    Returns: win_rate_new (float in [0,1])
    """

    print(f"\nLoading old model: {old_path}")
    old_model = AlphaZeroNet()
    old_model.load_state_dict(torch.load(old_path, map_location="cpu"))
    old_model.eval()

    print(f"Loading new model: {new_path}")
    new_model = AlphaZeroNet()
    new_model.load_state_dict(torch.load(new_path, map_location="cpu"))
    new_model.eval()

    new_score = 0
    wins = 0
    losses = 0
    draws = 0

    print(f"\n=== Arena Evaluation: {games} games, {sims} sims per move ===\n")

    for g in tqdm(range(games), desc="Arena Games"):
        if g % 2 == 0:
            # new = White
            result = play_single_game(new_model, old_model, sims=sims, game_index=g+1)
            new_score += result
            if result == 1: wins += 1
            elif result == -1: losses += 1
            else: draws += 1
        else:
            # new = Black
            result = play_single_game(old_model, new_model, sims=sims, game_index=g+1)
            # result is POV old model, so invert for new-scoring
            new_score -= result
            if result == -1: wins += 1     # new won
            elif result == 1: losses += 1  # new lost
            else: draws += 1

    # Convert to win-rate
    win_rate = (new_score + games) / (2 * games)

    print("\n=== Final Arena Results ===")
    print(f"New model wins:   {wins}")
    print(f"New model losses: {losses}")
    print(f"Draws:            {draws}")
    print(f"Win rate (new):   {win_rate:.3f}")

    return win_rate
