import torch
import chess
import numpy as np
from tqdm import tqdm

from hudson64.util._dataset.data_encoding import encode_board_with_history
from hudson64.core._mcts.mcts import mcts_search
from hudson64.core._network.model import AlphaZeroNet

POLICY_DIM = 4672


def play_one_game(model,
                  simulations=200,
                  temperature_moves=20,
                  move_limit=80):
    """
    Plays ONE AlphaZero-style self-play game.

    Returns:
        list of (state_planes, pi_vector, z_value)
    """

    board = chess.Board()
    history_boards = []
    samples = []
    move_number = 0

    while True:
        # -------------------------------
        # Check for end of game
        # -------------------------------
        if board.is_game_over():
            result = board.result()
            break

        if board.fullmove_number >= move_limit:
            result = "1/2-1/2"
            break

        # -------------------------------
        # Run MCTS
        # -------------------------------
        move, pi = mcts_search(
            board,
            model,
            simulations=simulations,
            temperature=1.0 if move_number < temperature_moves else 1e-6,
            return_pi=True,
            add_dirichlet=(move_number < temperature_moves)
        )

        # -------------------------------
        # Encode BEFORE the move
        # -------------------------------
        turn = board.turn  # save which side is making the decision
        state_planes = encode_board_with_history(board, list(history_boards))

        # Store training sample
        samples.append((state_planes, pi, turn))

        # -------------------------------
        # Apply move
        # -------------------------------
        board.push(move)
        history_boards.append(board.copy())

        move_number += 1

    # -------------------------------
    # Final game result
    # -------------------------------
    if result == "1-0":
        z_final = 1.0
    elif result == "0-1":
        z_final = -1.0
    else:
        z_final = 0.0

    # -------------------------------
    # Convert each sample's value to POV
    # -------------------------------
    final_samples = []
    for state_planes, pi, turn in samples:
        z = z_final if turn == chess.WHITE else -z_final
        final_samples.append((state_planes, pi, z))

    return final_samples



def selfplay_many_games(model_path, out_path, games=50, simulations=200):

    print(f"Loading model: {model_path}")
    model = AlphaZeroNet()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    all_states = []
    all_pi = []
    all_z = []

    for _ in tqdm(range(games), desc="Self-play"):
        samples = play_one_game(
            model,
            simulations=simulations
        )
        for s, pi, z in samples:
            all_states.append(s)
            all_pi.append(pi)
            all_z.append(z)

    all_states = torch.tensor(np.array(all_states), dtype=torch.float32)
    all_pi = torch.tensor(np.array(all_pi), dtype=torch.float32)
    all_z = torch.tensor(np.array(all_z), dtype=torch.float32).view(-1, 1)

    print(f"Collected {len(all_states)} training positions")

    torch.save({
        "inputs": all_states,
        "policy": all_pi,
        "value": all_z
    }, out_path)

    print(f"Saved RL dataset to: {out_path}")
