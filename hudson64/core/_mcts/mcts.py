import math
import random
import chess
import torch
import numpy as np

from hudson64.util._dataset.data_encoding import encode_board_with_history
from hudson64.util.move_indexing import move_to_index
from hudson64.core._network.model import POLICY_DIM   # 4672


# ============================================================
# Utility: detect model device
# ============================================================

def _get_model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


# ============================================================
# MCTS Node
# ============================================================

class MCTSNode:
    """
    Stores stats for each legal move from this position.
    """

    def __init__(self, board: chess.Board, priors=None):
        self.board = board
        self.children: dict[chess.Move, "MCTSNode"] = {}

        self.N: dict[chess.Move, int] = {}      # visit count
        self.W: dict[chess.Move, float] = {}    # total value
        self.Q: dict[chess.Move, float] = {}    # mean value
        self.P: dict[chess.Move, float] = {}    # prior prob

        legal = list(board.legal_moves)
        if not legal:
            return

        if priors is None:
            # Uniform priors if the network fails
            p = 1.0 / len(legal)
            for mv in legal:
                self.N[mv] = 0
                self.W[mv] = 0.0
                self.Q[mv] = 0.0
                self.P[mv] = p
        else:
            total = sum(priors.get(mv, 0.0) for mv in legal)
            if total <= 1e-12:
                p = 1.0 / len(legal)
                for mv in legal:
                    self.N[mv] = 0
                    self.W[mv] = 0.0
                    self.Q[mv] = 0.0
                    self.P[mv] = p
            else:
                inv = 1.0 / total
                for mv in legal:
                    p = priors.get(mv, 0.0) * inv
                    self.N[mv] = 0
                    self.W[mv] = 0.0
                    self.Q[mv] = 0.0
                    self.P[mv] = p


# ============================================================
# Network evaluation (batched)
# ============================================================

def _evaluate_batch(boards, model):
    """
    Evaluate a list of boards in a single forward pass.

    Returns:
        list of (priors_dict, value_float)
    """

    if not boards:
        return []

    device = _get_model_device(model)

    # Encode each board with empty history for search
    planes_list = [
        encode_board_with_history(b, history_boards=[])
        for b in boards
    ]
    x = torch.tensor(np.stack(planes_list), dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        logits, values = model(x)

    # logits: [B, POLICY_DIM], values: [B, 1]
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    values = values.view(-1).cpu().numpy()

    results = []
    for board, prob_row, v in zip(boards, probs, values):
        priors: dict[chess.Move, float] = {}
        total = 0.0

        for mv in board.legal_moves:
            try:
                idx = move_to_index(mv)
                p = float(prob_row[idx])
            except Exception:
                p = 0.0
            priors[mv] = p
            total += p

        if total <= 1e-12:
            # Fallback to uniform
            legal = list(board.legal_moves)
            if legal:
                u = 1.0 / len(legal)
                for mv in legal:
                    priors[mv] = u
        else:
            inv = 1.0 / total
            for mv in priors:
                priors[mv] *= inv

        results.append((priors, float(v)))

    return results


def _evaluate_single(board: chess.Board, model):
    """
    Convenience wrapper: evaluate a single board using the batched routine.
    """
    return _evaluate_batch([board], model)[0]


# ============================================================
# PUCT Selection Rule
# ============================================================

def select_child(node: MCTSNode, c_puct: float = 1.75) -> chess.Move:
    """
    Strong PUCT: Q + c_puct * P * sqrt(sum(N)) / (1+N).
    """

    best_score = -1e18
    best_move = None

    total_N = sum(node.N.values()) + 1

    for mv in node.N:
        Q = node.Q[mv]
        U = c_puct * node.P[mv] * math.sqrt(total_N) / (1 + node.N[mv])
        score = Q + U

        if score > best_score:
            best_score = score
            best_move = mv

    return best_move


# ============================================================
# Dirichlet noise (for RL self-play)
# ============================================================

def apply_dirichlet_noise(node: MCTSNode, alpha=0.3, eps=0.25):
    """
    Mix Dirichlet noise into root priors for exploration during self-play.
    """

    moves = list(node.P.keys())
    if not moves:
        return

    noise = np.random.dirichlet([alpha] * len(moves))
    for mv, n in zip(moves, noise):
        node.P[mv] = (1 - eps) * node.P[mv] + eps * float(n)


# ============================================================
# Helper: backprop
# ============================================================

def _backprop(path, value: float):
    """
    Backpropagate value along the selection path.
    path: list of (parent_node, move_taken)
    value: value at leaf, from POV of side-to-move at leaf.
    """
    for parent, mv in reversed(path):
        parent.N[mv] += 1
        parent.W[mv] += value
        parent.Q[mv] = parent.W[mv] / parent.N[mv]
        value = -value   # flip POV at each ply


# ============================================================
# Main MCTS with batched leaf evaluation
# ============================================================

def mcts_search(
    board: chess.Board,
    model,
    simulations: int = 800,
    return_pi: bool = False,
    root: MCTSNode | None = None,
    c_puct: float = 1.75,
    add_dirichlet: bool = False,   # False for human play, True for RL self-play
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
    temperature: float = 0.0,      # 0 → argmax-visits
    batch_size: int = 16,          # batched leaf evals
):
    """
    AlphaZero-style MCTS with batched network evaluation.

    Args:
        board:        current chess.Board
        model:        policy+value network
        simulations:  number of simulations from the root
        return_pi:    if True, also returns π(s) over POLICY_DIM
        root:         optional existing root node (for tree reuse)
        c_puct:       exploration constant
        add_dirichlet:add Dirichlet noise at root (self-play)
        temperature:  softmax temperature on visit counts for π
        batch_size:   number of leaf nodes to evaluate per network forward

    Returns:
        if return_pi == False:
            best_move

        if return_pi == True:
            (best_move, pi_vector[POLICY_DIM])
    """

    # Optional tree reuse
    if root is None or root.board.board_fen() != board.board_fen():
        priors, _ = _evaluate_single(board, model)
        root = MCTSNode(board, priors)

    # Optional Dirichlet at root (for RL early moves)
    if add_dirichlet and sum(root.N.values()) == 0:
        apply_dirichlet_noise(root, dirichlet_alpha, dirichlet_eps)

    if not root.N:
        # No legal moves (mate / stalemate)
        if return_pi:
            return None, np.zeros(POLICY_DIM, dtype=np.float32)
        return None

    sims_done = 0

    while sims_done < simulations:
        # ----------------------------------------------------
        # 1) Build a batch of leaf nodes via selection
        # ----------------------------------------------------
        leaf_boards = []
        leaf_paths = []
        terminal_indices = []
        terminal_values = []

        batch_slots = min(batch_size, simulations - sims_done)

        for _ in range(batch_slots):
            node = root
            b = board.copy()
            path = []

            # Selection phase: down until first unexplored child or terminal
            while node.children and not b.is_game_over():
                mv = select_child(node, c_puct)
                path.append((node, mv))
                b.push(mv)
                child = node.children.get(mv)
                if child is None:
                    # Found an edge that has never been expanded
                    break
                node = child

            # If terminal, handle later without network call
            if b.is_game_over():
                res = b.result()
                if res == "1-0":
                    val = 1.0
                elif res == "0-1":
                    val = -1.0
                else:
                    val = 0.0

                # value is from POV of side-to-move at leaf
                if not b.turn:
                    val = -val

                terminal_indices.append(len(leaf_boards))
                terminal_values.append(val)

                leaf_boards.append(b)
                leaf_paths.append(path)
            else:
                # Non-terminal leaf, will be evaluated by network
                leaf_boards.append(b)
                leaf_paths.append(path)

            sims_done += 1
            if sims_done >= simulations:
                break

        # ----------------------------------------------------
        # 2) Evaluate all NON-terminal leaves in a single batch
        # ----------------------------------------------------
        non_terminal_indices = [
            i for i in range(len(leaf_boards)) if i not in terminal_indices
        ]

        non_terminal_boards = [leaf_boards[i] for i in non_terminal_indices]

        priors_and_values = _evaluate_batch(non_terminal_boards, model) if non_terminal_boards else []

        # Map: index -> (priors, value)
        eval_map = {}
        for idx, (priors, v) in zip(non_terminal_indices, priors_and_values):
            eval_map[idx] = (priors, v)

        # ----------------------------------------------------
        # 3) Expansion + backprop for each leaf in the batch
        # ----------------------------------------------------
        # Handle network-evaluated leaves
        for idx in non_terminal_indices:
            b = leaf_boards[idx]
            path = leaf_paths[idx]
            priors, value = eval_map[idx]

            # Create new node for this leaf
            new_node = MCTSNode(b, priors)

            if path:
                parent, mv = path[-1]
                parent.children[mv] = new_node
            else:
                root = new_node

            # Value is from POV of side-to-move at leaf
            if not b.turn:
                value = -value

            _backprop(path, value)

        # Handle terminal leaves
        for idx, val in zip(terminal_indices, terminal_values):
            path = leaf_paths[idx]
            _backprop(path, val)

    # ========================================================
    # Choose move with highest visit count at root
    # ========================================================

    visits = root.N
    max_v = max(visits.values())
    best_moves = [m for m in visits if visits[m] == max_v]
    best_move = random.choice(best_moves)

    if not return_pi:
        return best_move

    # ========================================================
    # Build π(s) over full move space
    # ========================================================

    pi = np.zeros(POLICY_DIM, dtype=np.float32)

    if temperature <= 1e-6:
        # Deterministic: argmax-visits
        try:
            idx = move_to_index(best_move)
            pi[idx] = 1.0
        except Exception:
            pass
        return best_move, pi

    # Soft distribution from visit counts
    exp_counts = {}
    for mv, n in visits.items():
        exp_counts[mv] = n ** (1.0 / temperature)

    Z = sum(exp_counts.values())
    if Z <= 1e-12:
        # Fallback to argmax
        try:
            idx = move_to_index(best_move)
            pi[idx] = 1.0
        except Exception:
            pass
        return best_move, pi

    for mv, val in exp_counts.items():
        try:
            idx = move_to_index(mv)
            pi[idx] = float(val / Z)
        except Exception:
            continue

    return best_move, pi
