import pygame
import chess
import torch
import numpy as np

# ===== HUDSON64 imports =====
from hudson64.core._network.model import AlphaZeroNet
from hudson64.util._dataset.data_encoding import encode_board_with_history
from hudson64.core._mcts.mcts import mcts_search   # <--- MCTS FOR PLAY

# ============================================================
# VISUAL CONSTANTS
# ============================================================

SQUARE_SIZE = 80
BOARD_SIZE = SQUARE_SIZE * 8
EVAL_BAR_WIDTH = 60
WINDOW_WIDTH = BOARD_SIZE + EVAL_BAR_WIDTH
WINDOW_HEIGHT = BOARD_SIZE

PIECE_IMAGES = {}

FILENAME_MAP = {
    "P": "wp.png", "N": "wn.png", "B": "wb.png",
    "R": "wr.png", "Q": "wq.png", "K": "wk.png",
    "p": "p.png", "n": "n.png", "b": "b.png",
    "r": "r.png", "q": "q.png", "k": "k.png",
}

# ============================================================
# NETWORK HELPERS
# ============================================================

def load_network(path="policy_value_final.pt"):
    print("Loading network:", path)
    model = AlphaZeroNet()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

def network_eval(model, board, history):
    """Return network value only, for the eval bar."""
    planes = encode_board_with_history(board, history)
    x = torch.tensor(planes, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        _, value = model(x)

    return float(value.item())


# ============================================================
# MCTS MOVE SELECTION
# ============================================================

def engine_move(model, board):
    """Run MCTS for move selection."""
    best_move = mcts_search(
        board,
        model,
        simulations=300,      # GOOD DEFAULT FOR ~2200–2400 LEVEL
        return_pi=False,
        add_dirichlet=False   # No exploration during GUI play
    )
    return best_move


# ============================================================
# GRAPHICS
# ============================================================

def load_piece_images():
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    # Move to *project-root*/pieces
    piece_dir = os.path.abspath(os.path.join(base, "..", "..", "..", "pieces"))

    print("Piece directory:", piece_dir)

    for sym, fn in FILENAME_MAP.items():
        path = os.path.join(piece_dir, fn)
        print("Loading:", path)
        img = pygame.image.load(path)
        img = pygame.transform.scale(img, (SQUARE_SIZE, SQUARE_SIZE))
        PIECE_IMAGES[sym] = img


def draw_board(screen, board):
    light = (240, 217, 181)
    dark = (181, 136, 99)

    for r in range(8):
        for c in range(8):
            color = light if (r + c) % 2 == 0 else dark
            pygame.draw.rect(
                screen, color,
                pygame.Rect(c * SQUARE_SIZE, (7 - r) * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            )

    for sq, piece in board.piece_map().items():
        sym = piece.symbol()
        c = chess.square_file(sq)
        r = chess.square_rank(sq)
        screen.blit(PIECE_IMAGES[sym], (c * SQUARE_SIZE, (7 - r) * SQUARE_SIZE))


def draw_eval_bar(screen, score):
    """score ∈ [-1,1]"""
    h = int((score + 1) / 2 * WINDOW_HEIGHT)

    pygame.draw.rect(screen, (245, 245, 245),
                     pygame.Rect(BOARD_SIZE, WINDOW_HEIGHT - h, EVAL_BAR_WIDTH, h))
    pygame.draw.rect(screen, (30, 30, 30),
                     pygame.Rect(BOARD_SIZE, 0, EVAL_BAR_WIDTH, WINDOW_HEIGHT - h))


# ============================================================
# HELPER
# ============================================================

def mouse_square(pos):
    x, y = pos
    f = x // SQUARE_SIZE
    r = 7 - (y // SQUARE_SIZE)
    if f < 0 or f > 7 or r < 0 or r > 7:
        return None
    return chess.square(f, r)


# ============================================================
# MAIN GAME LOOP
# ============================================================

def play_game(play_as_white=True):

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Hudson64 — MCTS Engine")

    load_piece_images()
    model = load_network()

    board = chess.Board()
    history = []
    selected_sq = None
    eval_score = 0.0

    # Engine opens if user is black
    if not play_as_white:
        mv = engine_move(model, board)
        if mv:
            board.push(mv)
            history.append(board.copy())
            eval_score = network_eval(model, board, history)

    running = True

    while running:

        draw_board(screen, board)
        draw_eval_bar(screen, eval_score)
        pygame.display.flip()

        if board.is_game_over():
            print("Game Over:", board.result())
            pygame.time.wait(2500)
            break

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:

                is_human_turn = (board.turn == (chess.WHITE if play_as_white else chess.BLACK))

                if is_human_turn:
                    sq = mouse_square(event.pos)
                    if sq is None:
                        continue

                    if selected_sq is None:
                        selected_sq = sq
                    else:
                        mv = chess.Move(selected_sq, sq)

                        if mv in board.legal_moves:
                            board.push(mv)
                            history.append(board.copy())

                            if not board.is_game_over():
                                eng_mv = engine_move(model, board)
                                if eng_mv:
                                    board.push(eng_mv)
                                    history.append(board.copy())

                            eval_score = network_eval(model, board, history)

                        selected_sq = None

    pygame.quit()


def main():
    print("Play as:")
    print("1 = White")
    print("2 = Black")
    side = input("> ").strip()
    play_as_white = (side == "1")
    play_game(play_as_white)


if __name__ == "__main__":
    main()
