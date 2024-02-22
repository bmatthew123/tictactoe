from sklearn.ensemble import RandomForestClassifier
from copy import deepcopy
import uvicorn
import os
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import pandas as pd

app = FastAPI()
templates = Jinja2Templates(directory="templates")

def retrain(new_record=None):
    df = pd.read_csv("data.csv")
    if new_record != None:
        df.loc[len(df)] = new_record
        df.to_csv("data.csv", index=False)
    X = df.drop("move", axis=1)
    y = df["move"]
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

model = retrain()
print("Probability:", model.predict_proba([[0,0,0,0,0,0,0,0,0]]))
print("Prediction:", model.predict([[0,0,0,0,0,0,0,0,0]]))


def get_ai_move(board):
    probabilities = model.predict_proba(board)[0]
    moves = [i[1] for i in sorted(zip(probabilities, range(9)), reverse=True) if board[0][i[1]] == 0]
    if len(moves) > 0:
        return moves[0]
    return None


def get_winner(brd):
    board = brd[0]
    if board[0] == board[1] == board[2] and board[0] != 0:
        return board[0]
    if board[3] == board[4] == board[5] and board[3] != 0:
        return board[3]
    if board[6] == board[7] == board[8] and board[6] != 0:
        return board[6]
    if board[0] == board[3] == board[6] and board[0] != 0:
        return board[0]
    if board[1] == board[4] == board[7] and board[1] != 0:
        return board[1]
    if board[2] == board[5] == board[8] and board[2] != 0:
        return board[2]
    if board[0] == board[4] == board[8] and board[0] != 0:
        return board[0]
    if board[2] == board[4] == board[6] and board[2] != 0:
        return board[2]
    if all(board) > 0:
        return 0
    return -1

def create_record(move, board):
    features = deepcopy(board[0])
    m = {0: 0, 1: 2, 2: 1}  # Flip human record to be a training record
    features = [m[i] for i in features]
    features.append(int(move))
    return features

board = [[0, 0, 0, 0, 0, 0, 0, 0, 0]]
ai_first = False

@app.get("/")
@app.post("/")
async def homepage(request: Request):
    print("board:", globals()["board"][0])
    params = list(request.query_params.keys())
    if len(params) > 0:
        if params[0] == "reset":
            globals()["ai_first"] = not globals()["ai_first"]
            globals()["board"] = [[0] * 9]
            if globals()["ai_first"]:
                move = get_ai_move(board)
                globals()["board"][0][move] = 2
        else:
            move = params[0]
            features = create_record(move, globals()["board"])
            globals()["model"] = retrain(features)
            globals()["board"][0][int(move)] = 1
            winner = get_winner(board)
            if winner < 1:
                move = get_ai_move(board)
                if move is not None:
                    globals()["board"][0][move] = 2
    winner = get_winner(globals()["board"])
    print("board after:", globals()["board"][0])
    return templates.TemplateResponse("index.html", {
        "request": request,
        "board": globals()["board"][0],
        "winner": winner,
    })

if __name__ == "__main__":
    print("Starting webserver...")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
