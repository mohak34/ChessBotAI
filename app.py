# app.py
import chess
import numpy as np
from flask import Flask, jsonify, render_template, request
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

app = Flask(__name__)


def one_hot_encode_piece(piece):
    pieces = list("rnbqkpRNBQKP.")
    arr = np.zeros(len(pieces))
    piece_to_index = {p: i for i, p in enumerate(pieces)}
    index = piece_to_index[piece]
    arr[index] = 1
    return arr


def encode_board(board):
    board_str = str(board)
    board_str = board_str.replace(" ", "")
    board_list = []
    for row in board_str.split("\n"):
        row_list = []
        for piece in row:
            row_list.append(one_hot_encode_piece(piece))
        board_list.append(row_list)
    return np.array(board_list)


model = Sequential(
    [
        Flatten(),
        Dense(128, activation="relu"),
        Dense(1),
    ]
)

model.compile(optimizer="rmsprop", loss="mean_squared_error")
X_train = np.random.random((1000, 8 * 8 * 13))  # 8x8 board, 13 piece types
y_train = np.random.random((1000, 1))

model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)


def play_nn(fen, show_move_evaluations=False, player="b"):
    board = chess.Board(fen=fen)
    moves = []

    for move in board.legal_moves:
        candidate_board = board.copy()
        candidate_board.push(move)
        input_vector = (
            encode_board(str(candidate_board)).astype(np.int32).flatten()
        )
        score = model.predict(np.expand_dims(input_vector, axis=0), verbose=0)[
            0
        ][0]
        moves.append((score, move))

    best_move = sorted(moves, reverse=player == "b")[0][1]
    return str(best_move)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/make_move", methods=["POST"])
def make_move():
    fen = request.json.get("fen")
    try:
        ai_move = play_nn(fen)
        return jsonify({"move": ai_move})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)

