import pygame
import chess
import torch
import numpy as np

from hudson64.core._network.model import AlphaZeroNet
from hudson64.util._dataset.data_encoding import encode_board_with_history
from hudson64.util.move_indexing import move_to_index


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

def load_network(path="policy_value_best.pt"):
    model = AlphaZeroNet()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def network_eval_and_policy(model, board, history):
    """
    Returns:
        logits: [4672]
        value: float in [-1, 1]
    """
    planes = encode_board_with_history(board, history)
    inp = torch.tensor(planes, dtype=torch.float32).unsqueeze(0)  # [1, C, 8, 8]

    with torch.no_grad():
        logits, value = model(inp)

    logits = logits.squeeze(0).numpy()
    value = float(value.item())
    return logits, value


def select_network_move(model, board, history):
    logits, _ = network_eval_and_policy(model, board, history)

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    # Pick legal move with highest policy logit
    best_move = None
    best_score = -1e9

    for mv in legal_moves:
        try:
            idx = move_to_index(mv)
        except:
            continue

        score = logits[idx]
        if score > best_score:
            best_score = score
            best_move = mv

    return best_move


# ============================================================
# GRAPHICS
# ============================================================

def load_piece_images():
    import os
    base = os.path.dirname(os.path.abspath(__file__))

    # FIXED: go up THREE directories to reach the project root, then into /pieces
    piece_dir = os.path.abspath(os.path.join(base, "..", "..", "..", "pieces"))

    print("Using piece directory:", piece_dir)

    for symbol, filename in FILENAME_MAP.items():
        path = os.path.join(piece_dir, filename)
        print("Loading:", path)
        
        img = pygame.image.load(path)
        img = pygame.transform.scale(img, (SQUARE_SIZE, SQUARE_SIZE))
        PIECE_IMAGES[symbol] = img



def draw_board(screen, board):
    light = (240, 217, 181)
    dark = (181, 136, 99)

    for r in range(8):
        for c in range(8):
            color = light if (r + c) % 2 == 0 else dark
            rect = pygame.Rect(c * SQUARE_SIZE, (7 - r) * SQUARE_SIZE,
                               SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(screen, color, rect)

    for square, piece in board.piece_map().items():
        symbol = piece.symbol()
        c = chess.square_file(square)
        r = chess.square_rank(square)
        x = c * SQUARE_SIZE
        y = (7 - r) * SQUARE_SIZE
        screen.blit(PIECE_IMAGES[symbol], (x, y))


def draw_eval_bar(screen, eval_score):
    """
    eval_score ∈ [-1, 1]
    +1 = white winning
    -1 = black winning
    """

    # Convert [-1,1] to pixel height
    bar_height = int((eval_score + 1) / 2 * WINDOW_HEIGHT)

    # White section
    pygame.draw.rect(screen, (245, 245, 245),
                     pygame.Rect(BOARD_SIZE, WINDOW_HEIGHT - bar_height,
                                 EVAL_BAR_WIDTH, bar_height))

    # Black section
    pygame.draw.rect(screen, (30, 30, 30),
                     pygame.Rect(BOARD_SIZE, 0,
                                 EVAL_BAR_WIDTH, WINDOW_HEIGHT - bar_height))


# ============================================================
# HELPERS
# ============================================================

def mouse_square(pos):
    x, y = pos
    file = x // SQUARE_SIZE
    rank = 7 - (y // SQUARE_SIZE)
    if file < 0 or file > 7 or rank < 0 or rank > 7:
        return None
    return chess.square(file, rank)


# ============================================================
# MAIN GAME LOOP
# ============================================================

def play_game(play_as_white=True):

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Hudson64 - NN-Only Engine")

    load_piece_images()
    model = load_network()

    board = chess.Board()
    history = []
    selected = None
    eval_score = 0.0

    # Engine opens if user is black
    if not play_as_white:
        mv = select_network_move(model, board, history)
        if mv:
            board.push(mv)
            history.append(board.copy())
            _, eval_score = network_eval_and_policy(model, board, history)

    running = True

    while running:

        draw_board(screen, board)
        draw_eval_bar(screen, eval_score)
        pygame.display.flip()

        if board.is_game_over():
            print("Game Over:", board.result())
            pygame.time.wait(3000)
            break

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:

                # Human turn?
                if board.turn == (chess.WHITE if play_as_white else chess.BLACK):
                    sq = mouse_square(event.pos)

                    if sq is None:
                        continue

                    if selected is None:
                        selected = sq
                    else:
                        mv = chess.Move(selected, sq)

                        if mv in board.legal_moves:
                            board.push(mv)
                            history.append(board.copy())

                            # Engine responds
                            if not board.is_game_over():
                                engine_mv = select_network_move(model, board, history)
                                if engine_mv:
                                    board.push(engine_mv)
                                    history.append(board.copy())

                            _, eval_score = network_eval_and_policy(model, board, history)

                        selected = None

    pygame.quit()


def main():
    print("1 = Play White")
    print("2 = Play Black")
    side = input("> ").strip()
    play_white = (side == "1")
    play_game(play_as_white=play_white)


if __name__ == "__main__":
    main()
