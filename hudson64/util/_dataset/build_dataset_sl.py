import chess
import chess.pgn
import numpy as np
import torch
from tqdm import tqdm

from hudson64.util._dataset.data_encoding import encode_board_with_history, HISTORY
from hudson64.util.move_indexing import move_to_index

POLICY_DIM = 4672
RESULT_MAP = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}

# ---- disk / size controls ----
CHUNK_SIZE = 10_000         # samples per shard file
MAX_SAMPLES_TOTAL = 150_000 # global cap across all shards

def mirror_square(sq: int) -> int:
    return chess.square(7 - chess.square_file(sq), chess.square_rank(sq))


def mirror_move(move: chess.Move) -> chess.Move:
    if move.promotion:
        return chess.Move(
            mirror_square(move.from_square),
            mirror_square(move.to_square),
            promotion=move.promotion,
        )
    return chess.Move(
        mirror_square(move.from_square),
        mirror_square(move.to_square),
    )


def mirror_planes(planes: np.ndarray) -> np.ndarray:
    """Flip horizontally: planes are [C, 8, 8]."""
    return np.flip(planes, axis=2).copy()


def mirror_policy_index(idx: int, board: chess.Board) -> int:
    """Mirror policy target using move_to_index() on the mirrored move."""
    for mv in board.legal_moves:
        try:
            if move_to_index(mv) == idx:
                return move_to_index(mirror_move(mv))
        except Exception:
            continue
    return idx

def _flush_shard(shard_idx: int, X_list, Pi_list, V_list) -> None:
    if not X_list:
        return

    print(f"\n[Shard {shard_idx:03d}] Converting {len(X_list)} samples to tensors...")

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    Pi = torch.tensor(np.array(Pi_list), dtype=torch.int64)
    V = torch.tensor(np.array(V_list), dtype=torch.float32).view(-1, 1)

    fname = f"dataset_sl_chunk_{shard_idx:03d}.pt"
    torch.save({"inputs": X, "policy": Pi, "value": V}, fname)

    print(f"[Shard {shard_idx:03d}] Saved: {fname} ({len(X)} samples)\n")

def build_dataset(pgn_path: str, max_games: int = 5000, max_positions: int = 1_000_000):

    X = []
    Pi = []
    V = []

    shard_idx = 0
    samples_in_shard = 0
    total_samples = 0

    game_count = 0
    pos_count = 0

    print(f"\nBuilding dataset from: {pgn_path}")
    print(f"Max games: {max_games} | Max positions: {max_positions}")
    print(f"Chunk size: {CHUNK_SIZE} | Max samples total: {MAX_SAMPLES_TOTAL}\n")

    with open(pgn_path, "r", encoding="utf-8") as f:

        pbar_games = tqdm(total=max_games, desc="Games", unit="game")

        while game_count < max_games and total_samples < MAX_SAMPLES_TOTAL:

            game = chess.pgn.read_game(f)
            if game is None:
                break

            game_count += 1
            pbar_games.update(1)

            result = game.headers.get("Result")
            if result not in RESULT_MAP:
                continue

            value = RESULT_MAP[result]
            board = game.board()
            history = []

            for move in game.mainline_moves():

                if pos_count >= max_positions or total_samples >= MAX_SAMPLES_TOTAL:
                    break

                # Encode board state [C, 8, 8]
                planes = encode_board_with_history(board, history)

                # Move → index
                try:
                    idx = move_to_index(move)
                except Exception:
                    board.push(move)
                    history.append(board.copy())
                    continue

                # Original
                X.append(planes)
                Pi.append(idx)
                V.append(value)
                samples_in_shard += 1
                total_samples += 1

                # Mirrored
                if total_samples < MAX_SAMPLES_TOTAL:
                    X.append(mirror_planes(planes))
                    Pi.append(mirror_policy_index(idx, board))
                    V.append(value)
                    samples_in_shard += 1
                    total_samples += 1

                pos_count += 1

                # Flush shard if needed
                if samples_in_shard >= CHUNK_SIZE:
                    _flush_shard(shard_idx, X, Pi, V)
                    shard_idx += 1
                    X.clear()
                    Pi.clear()
                    V.clear()
                    samples_in_shard = 0

                # Advance game
                board.push(move)
                history.append(board.copy())

                if total_samples >= MAX_SAMPLES_TOTAL:
                    break

        pbar_games.close()

    # Flush any remaining samples
    if samples_in_shard > 0:
        _flush_shard(shard_idx, X, Pi, V)

    print("\n================================")
    print("Dataset sharding complete.")
    print(f"Games processed: {game_count}")
    print(f"Positions (pre-aug): {pos_count}")
    print(f"Total samples saved across shards: {total_samples}")
    print("Files: dataset_sl_chunk_XXX.pt")
    print("================================\n")


if __name__ == "__main__":
    build_dataset(
        pgn_path="lichess_elite_2020-06.pgn",
        max_games=10_000,
        max_positions=1_000_000,
    )
