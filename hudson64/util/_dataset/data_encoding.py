import chess
import numpy as np

HISTORY = 8   # number of past states to include
BOARD_SIZE = 8

PIECE_TYPES = [
    chess.PAWN, chess.KNIGHT, chess.BISHOP,
    chess.ROOK, chess.QUEEN, chess.KING
]

def encode_single_board(board: chess.Board):
    """
    Encode 12 piece planes for a single board position.
    Output: [12, 8, 8]
    """
    planes = np.zeros((12, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    for piece_type in PIECE_TYPES:
        for color in [chess.WHITE, chess.BLACK]:
            plane_index = (0 if color == chess.WHITE else 6) + (piece_type - 1)
            for sq in board.pieces(piece_type, color):
                r = 7 - chess.square_rank(sq)
                c = chess.square_file(sq)
                planes[plane_index, r, c] = 1.0

    return planes


def encode_aux(board: chess.Board):
    """7 auxiliary planes."""
    aux = np.zeros((7, 8, 8), dtype=np.float32)

    # Plane 0: side to move
    aux[0] = 1.0 if board.turn == chess.WHITE else 0.0

    # Planes 1–4: castling rights
    aux[1] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    aux[2] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    aux[3] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    aux[4] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    # Plane 5: fifty-move counter normalized
    aux[5] = board.halfmove_clock / 100.0

    # Plane 6: en-passant file
    if board.ep_square is not None:
        file = chess.square_file(board.ep_square)
        aux[6, :, file] = 1.0

    return aux


def encode_board_with_history(board, history_boards):
    """
    Simplified encoder — ALWAYS returns 18 planes:
        12 piece planes
         1 side-to-move plane
         4 castling planes
         1 en-passant plane
    TOTAL = 18   (matches AlphaZeroNet(in_channels=18))

    History is intentionally ignored for speed.
    """

    # ---- 12 Piece planes ----
    planes = [encode_single_board(board)]   # shape [12, 8, 8]

    # ---- Aux planes (6 planes) ----
    aux = np.zeros((6, 8, 8), dtype=np.float32)

    # Plane 0 — side to move
    aux[0] = 1.0 if board.turn == chess.WHITE else 0.0

    # Planes 1–4 — castling rights
    aux[1] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    aux[2] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    aux[3] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    aux[4] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    # Plane 5 — en-passant file
    if board.ep_square is not None:
        file = chess.square_file(board.ep_square)
        aux[5, :, file] = 1.0

    # Stack = (12 + 6 = 18)
    planes = np.concatenate([planes[0], aux], axis=0)

    return planes.astype(np.float32)

