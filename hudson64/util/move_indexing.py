import chess

POLICY_DIM = 4672

# Knight move vectors
KNIGHT_DIRS = [
    (2, 1), (2, -1), (-2, 1), (-2, -1),
    (1, 2), (1, -2), (-1, 2), (-1, -2)
]

# Sliding move directions (queen-like)
SLIDING_DIRS = [
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (1, -1), (-1, 1), (-1, -1)
]


def move_to_index(move: chess.Move) -> int:
    """
    Convert a chess.Move into an index in [0, 4671].
    Must match dataset + MCTS.
    """

    from_sq = move.from_square
    to_sq = move.to_square

    fr = chess.square_rank(from_sq)
    ff = chess.square_file(from_sq)
    tr = chess.square_rank(to_sq)
    tf = chess.square_file(to_sq)

    dr = tr - fr
    df = tf - ff

    for dir_id, (rr, cc) in enumerate(KNIGHT_DIRS):
        if dr == rr and df == cc:
            return 64 * dir_id + from_sq

    base = 512
    offset = 0

    for d_id, (rr, cc) in enumerate(SLIDING_DIRS):
        for dist in range(1, 8):
            if dr == rr * dist and df == cc * dist:
                offset = (d_id * 7 + (dist - 1)) * 64
                return base + offset + from_sq

    if move.promotion is not None:
        return POLICY_DIM - 1   # 4671

    raise ValueError(f"Move {move.uci()} cannot be indexed!")


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """
    Reverse of move_to_index (not strictly required for MCTS, but useful).
    """

    if index < 512:
        # knight move
        dir_id = index // 64
        from_sq = index % 64

        rr, cc = KNIGHT_DIRS[dir_id]
        fr = chess.square_rank(from_sq)
        ff = chess.square_file(from_sq)
        tr = fr + rr
        tf = ff + cc

        if 0 <= tr < 8 and 0 <= tf < 8:
            mv = chess.Move(from_sq, chess.square(tf, tr))
            if mv in board.legal_moves:
                return mv

        raise ValueError("Knight reverse-index invalid.")

    # sliding moves
    index -= 512

    sliding_dir = index // (7 * 64)
    index %= (7 * 64)

    dist = index // 64
    from_sq = index % 64

    rr, cc = SLIDING_DIRS[sliding_dir]

    fr = chess.square_rank(from_sq)
    ff = chess.square_file(from_sq)
    tr = fr + rr * (dist + 1)
    tf = ff + cc * (dist + 1)

    if 0 <= tr < 8 and 0 <= tf < 8:
        mv = chess.Move(from_sq, chess.square(tf, tr))
        if mv in board.legal_moves:
            return mv

    # promotions bucket — guess from legal moves
    for mv in board.legal_moves:
        if mv.promotion is not None:
            return mv

    raise ValueError("index does not correspond to a legal move.")
