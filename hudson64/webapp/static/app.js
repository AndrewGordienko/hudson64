const elBoard = document.getElementById("board");
const elStatus = document.getElementById("status");
const elOverlay = document.getElementById("overlay");

const btnPlayWhite = document.getElementById("playWhite");
const btnPlayBlack = document.getElementById("playBlack");

const elEvalFill = document.getElementById("evalFill");
const elEvalText = document.getElementById("evalText");

let selectedFrom = null;
let orientation = "white"; // "white" or "black"

// We keep the latest parsed pieces so we can know if a move is a pawn move
let lastPieceMap = {};

const PIECE_SRC = {
  "P": "/static/pieces/wp.png",
  "N": "/static/pieces/wn.png",
  "B": "/static/pieces/wb.png",
  "R": "/static/pieces/wr.png",
  "Q": "/static/pieces/wq.png",
  "K": "/static/pieces/wk.png",

  "p": "/static/pieces/p.png",
  "n": "/static/pieces/n.png",
  "b": "/static/pieces/b.png",
  "r": "/static/pieces/r.png",
  "q": "/static/pieces/q.png",
  "k": "/static/pieces/k.png",
};

function setStatus(msg) {
  elStatus.textContent = msg;
}

function setEval(v) {
  if (typeof v !== "number" || Number.isNaN(v)) v = 0;

  // backend eval assumed in [-1, 1]
  const clamped = Math.max(-1, Math.min(1, v));
  const pct = (clamped + 1) / 2;

  elEvalFill.style.height = `${Math.round(pct * 100)}%`;
  elEvalText.textContent = clamped.toFixed(2);
}

function showOverlay() {
  elOverlay.classList.remove("hidden");
}

function hideOverlay() {
  elOverlay.classList.add("hidden");
}

function clearHighlights() {
  elBoard.querySelectorAll(".square").forEach((sq) => {
    sq.classList.remove("highlight");
  });
}

function highlightSquare(squareName) {
  const sq = elBoard.querySelector(`.square[data-square="${squareName}"]`);
  if (sq) sq.classList.add("highlight");
}

function makeSquareName(fileIdx, rankIdx) {
  const file = String.fromCharCode("a".charCodeAt(0) + fileIdx);
  const rank = String(rankIdx + 1);
  return file + rank;
}

function parseFenBoard(fen) {
  const boardPart = fen.split(" ")[0];
  const rows = boardPart.split("/");
  if (rows.length !== 8) throw new Error("Bad FEN");

  const map = {};

  for (let r = 0; r < 8; r++) {
    const row = rows[r];
    let file = 0;

    for (const ch of row) {
      if (ch >= "1" && ch <= "8") {
        file += Number(ch);
      } else {
        const rankFen = 8 - r;       // 8..1
        const rankIdx = rankFen - 1; // 7..0
        const sq = makeSquareName(file, rankIdx);
        map[sq] = ch;
        file += 1;
      }
    }
  }

  return map;
}

/**
 * Promotion should ONLY be appended for pawns reaching last rank.
 * This fixes castling being broken (e1g1q is invalid).
 */
function uciFromMove(from, to) {
  const piece = lastPieceMap[from]; // "P" or "p" etc
  const toRank = to[1];

  const isPawn = piece === "P" || piece === "p";
  const isPromotionSquare = (toRank === "8" || toRank === "1");

  if (isPawn && isPromotionSquare) {
    return from + to + "q"; // auto queen
  }

  return from + to;
}

async function apiGetState() {
  const r = await fetch("/api/state");
  return await r.json();
}

async function apiNew(playAsWhite) {
  const r = await fetch("/api/new", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ play_as_white: playAsWhite }),
  });
  return await r.json();
}

async function apiMove(uci) {
  const r = await fetch("/api/move", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ uci }),
  });
  return await r.json();
}

async function tryMove(fromSq, toSq) {
  if (!fromSq || !toSq || fromSq === toSq) return;

  setStatus("Thinking…");

  const uci = uciFromMove(fromSq, toSq);
  const res = await apiMove(uci);

  if (!res.ok) {
    setStatus(`Rejected (${res.error})`);
    const st = await apiGetState();
    renderPositionFromFen(st.fen);
    setEval(st.eval ?? 0);
    return;
  }

  renderPositionFromFen(res.fen);
  setEval(res.eval ?? 0);

  if (res.is_game_over) {
    setStatus(`Game over: ${res.result}`);
    return;
  }

  if (res.engine_move_uci) setStatus(`Engine: ${res.engine_move_uci}`);
  else setStatus("Your turn");
}

function onSquareClick(squareName) {
  if (!selectedFrom) {
    selectedFrom = squareName;
    clearHighlights();
    highlightSquare(selectedFrom);
    setStatus(`Selected ${selectedFrom}`);
    return;
  }

  const from = selectedFrom;
  const to = squareName;

  selectedFrom = null;
  clearHighlights();

  tryMove(from, to);
}

/**
 * Captures:
 * If a from-square is selected and you click ANY piece,
 * that should act as destination square (capture attempt).
 */
function onPieceClick(squareName) {
  if (selectedFrom && selectedFrom !== squareName) {
    const from = selectedFrom;
    const to = squareName;

    selectedFrom = null;
    clearHighlights();

    tryMove(from, to);
    return;
  }

  selectedFrom = squareName;
  clearHighlights();
  highlightSquare(squareName);
  setStatus(`Selected ${squareName}`);
}

/**
 * Board orientation:
 * - white: normal
 * - black: flipped
 */
function buildBoardSkeleton() {
  elBoard.innerHTML = "";

  let ranks;
  let files;

  if (orientation === "white") {
    ranks = [7, 6, 5, 4, 3, 2, 1, 0];
    files = [0, 1, 2, 3, 4, 5, 6, 7];
  } else {
    ranks = [0, 1, 2, 3, 4, 5, 6, 7];
    files = [7, 6, 5, 4, 3, 2, 1, 0];
  }

  for (const rankIdx of ranks) {
    for (const fileIdx of files) {
      const sqName = makeSquareName(fileIdx, rankIdx);

      const sq = document.createElement("div");
      sq.className = "square " + (((fileIdx + rankIdx) % 2 === 0) ? "light" : "dark");
      sq.dataset.square = sqName;

      sq.addEventListener("click", () => onSquareClick(sqName));

      elBoard.appendChild(sq);
    }
  }
}

function renderPositionFromFen(fen) {
  lastPieceMap = parseFenBoard(fen);

  // clear pieces
  elBoard.querySelectorAll(".square").forEach((sq) => {
    sq.innerHTML = "";
  });

  // place pieces
  for (const [sqName, pieceChar] of Object.entries(lastPieceMap)) {
    const src = PIECE_SRC[pieceChar];
    if (!src) continue;

    const sq = elBoard.querySelector(`.square[data-square="${sqName}"]`);
    if (!sq) continue;

    const holder = document.createElement("div");
    holder.className = "piece";
    holder.dataset.square = sqName;

    holder.addEventListener("click", (e) => {
      e.stopPropagation();
      onPieceClick(sqName);
    });

    const img = document.createElement("img");
    img.src = src;
    img.alt = pieceChar;

    holder.appendChild(img);
    sq.appendChild(holder);
  }
}

async function startGame(playAsWhite) {
  orientation = playAsWhite ? "white" : "black";

  selectedFrom = null;
  clearHighlights();

  buildBoardSkeleton();

  setEval(0);
  setStatus(playAsWhite ? "You are White." : "You are Black. Engine opening…");
  hideOverlay();

  const st = await apiNew(playAsWhite);
  renderPositionFromFen(st.fen);
  setEval(st.eval ?? 0);

  setStatus("Your turn.");
}

async function init() {
  orientation = "white";
  buildBoardSkeleton();
  showOverlay();
  setEval(0);

  // show starting pieces behind overlay
  const st = await apiGetState();
  renderPositionFromFen(st.fen);
  setEval(st.eval ?? 0);

  btnPlayWhite.addEventListener("click", () => startGame(true));
  btnPlayBlack.addEventListener("click", () => startGame(false));
}

window.addEventListener("load", init);
