import os
from flask import Flask, render_template, request, jsonify, session, make_response
import chess
import torch

from hudson64.core._network.model import AlphaZeroNet
from hudson64.util._dataset.data_encoding import encode_board_with_history
from hudson64.util.move_indexing import move_to_index

MODEL_PATH = os.environ.get("HUDSON64_MODEL_PATH", "policy_value_best.pt")

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

# ------------------------------------------------------------
# Model (loaded once)
# ------------------------------------------------------------
def load_network(path=MODEL_PATH):
    model = AlphaZeroNet()
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

MODEL = load_network()

# ------------------------------------------------------------
# Engine helpers
# ------------------------------------------------------------
def build_board_and_history_from_moves(moves_uci: list[str]) -> tuple[chess.Board, list[chess.Board]]:
    board = chess.Board()
    history_boards: list[chess.Board] = []
    for u in moves_uci:
        mv = chess.Move.from_uci(u)
        if mv not in board.legal_moves:
            # If something went out of sync, stop replaying
            break
        board.push(mv)
        history_boards.append(board.copy())
    return board, history_boards

def network_eval_and_policy(model, board: chess.Board, history: list[chess.Board]):
    planes = encode_board_with_history(board, history)
    inp = torch.tensor(planes, dtype=torch.float32).unsqueeze(0)  # [1, C, 8, 8]
    with torch.no_grad():
        logits, value = model(inp)
    logits = logits.squeeze(0).cpu().numpy()
    value = float(value.item())
    return logits, value

def select_network_move(model, board: chess.Board, history: list[chess.Board]):
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None, 0.0

    logits, value = network_eval_and_policy(model, board, history)

    best_move = None
    best_score = -1e18

    for mv in legal_moves:
        try:
            idx = move_to_index(mv)
        except Exception:
            continue
        score = float(logits[idx])
        if score > best_score:
            best_score = score
            best_move = mv

    return best_move, value

# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------
def get_moves() -> list[str]:
    return session.get("moves_uci", [])

def set_moves(moves: list[str]):
    session["moves_uci"] = moves

def reset_game(play_as_white=True):
    session["play_as_white"] = bool(play_as_white)
    set_moves([])

# ------------------------------------------------------------
# Security / embed friendliness
# ------------------------------------------------------------
@app.after_request
def add_headers(resp):
    # Allow embedding in iframes (adjust to your domain later if you want strict)
    resp.headers["X-Frame-Options"] = "ALLOWALL"
    resp.headers["Content-Security-Policy"] = "frame-ancestors *"
    return resp

# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@app.get("/")
def index():
    if "moves_uci" not in session:
        reset_game(play_as_white=True)
    return render_template("index.html")

@app.post("/api/new")
def api_new():
    data = request.get_json(force=True, silent=True) or {}
    play_as_white = bool(data.get("play_as_white", True))
    reset_game(play_as_white=play_as_white)

    # If user plays black, engine makes first move
    if not play_as_white:
        board, hist = build_board_and_history_from_moves(get_moves())
        mv, value = select_network_move(MODEL, board, hist)
        if mv:
            moves = get_moves()
            moves.append(mv.uci())
            set_moves(moves)

    return api_state()

@app.get("/api/state")
def api_state():
    board, _ = build_board_and_history_from_moves(get_moves())
    return jsonify({
        "fen": board.fen(),
        "turn": "w" if board.turn == chess.WHITE else "b",
        "moves_uci": get_moves(),
        "play_as_white": session.get("play_as_white", True),
        "is_game_over": board.is_game_over(),
        "result": board.result() if board.is_game_over() else None,
    })

@app.post("/api/move")
def api_move():
    data = request.get_json(force=True, silent=True) or {}
    uci = (data.get("uci") or "").strip()

    moves = get_moves()
    board, hist = build_board_and_history_from_moves(moves)

    if board.is_game_over():
        return jsonify({"ok": False, "error": "game_over"}), 400

    # Enforce "human to move" based on chosen side
    play_as_white = session.get("play_as_white", True)
    human_color = chess.WHITE if play_as_white else chess.BLACK
    if board.turn != human_color:
        return jsonify({"ok": False, "error": "not_human_turn"}), 400

    try:
        mv = chess.Move.from_uci(uci)
    except Exception:
        return jsonify({"ok": False, "error": "bad_uci"}), 400

    if mv not in board.legal_moves:
        return jsonify({"ok": False, "error": "illegal"}), 400

    # Apply human move
    board.push(mv)
    moves.append(mv.uci())
    hist.append(board.copy())

    engine_move_uci = None
    eval_value = 0.0

    # Engine responds if game not over
    if not board.is_game_over():
        eng_mv, eval_value = select_network_move(MODEL, board, hist)
        if eng_mv:
            board.push(eng_mv)
            moves.append(eng_mv.uci())
            engine_move_uci = eng_mv.uci()

    set_moves(moves)

    return jsonify({
        "ok": True,
        "fen": board.fen(),
        "moves_uci": moves,
        "engine_move_uci": engine_move_uci,
        "eval": eval_value,  # [-1,1]
        "is_game_over": board.is_game_over(),
        "result": board.result() if board.is_game_over() else None,
    })

if __name__ == "__main__":
    # Dev server
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)), debug=True)
