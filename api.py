from fastapi import FastAPI, HTTPException
from pydantic import BaseModel,conlist,constr
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
import cv2 
import numpy as np
import base64
from board_reader.board_finder import BoardFinder
from board_reader.config import *
import os


model = load_model("./scrabble_cnn_model.keras")
finder = BoardFinder(False)
app = FastAPI()

centers = []
for i in range(15):
        for j in range(15):
            centers.append((BORDER_SIZE +int(GAP/2) + i * GAP, BORDER_SIZE +int(GAP/2) + j * GAP))

Board = conlist(conlist(constr(min_length=0, max_length=1), min_length=15, max_length=15), min_length=15, max_length=15)

class BoardReq(BaseModel):
    image: str
    currentBoard: Board    


ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
if ENVIRONMENT == "development":
    allowed_origins = [
        "http://localhost:5173",
        "http://192.168.1.78:5173",
    ]
else:  # production
    allowed_origins = [
        "https://your-react-app-domain.com",
    ]


app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#just to test the server
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/")
async def root(data: BoardReq):
    encoded_data = data.image.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    copy = img.copy()
    print(data.currentBoard)
    board = finder.extract_scrabble_board(copy)
    if(board is None):
        cv2.imwrite("debug.jpeg",copy)
        raise HTTPException(status_code=422, detail="Invalid board")

    tiles,tilepos = finder.extract_tiles(board)
    #blank_centers contains the center positions of ALL tiles, so we need to remove the centers that coincide with tiles that have letters
    blank_centers = finder.extract_blanks(board)
    processed_tiles = []
    for tile in tiles: 
        tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        tile = cv2.resize(tile, (50, 50), interpolation=cv2.INTER_LINEAR)
        tile = np.expand_dims(tile, axis=2)
        processed_tiles.append(tile)
    tiles_dataset = np.stack(processed_tiles, axis=0)
    predictions = model.predict(tiles_dataset)

    for prediction,position in zip(predictions,tilepos):
        charPrediction = chr(np.argmax(prediction) + 65)
        closestCenter=min(centers, key=lambda x: (x[0] - position[0]) ** 2 + (x[1] - position[1]) ** 2)
        xPos = (closestCenter[0]-BORDER_SIZE)//GAP
        yPos = (closestCenter[1]-BORDER_SIZE)//GAP
        if(data.currentBoard[yPos][xPos] != ""):
            #this is a tile that has already been placed
            continue
        data.currentBoard[(closestCenter[1]-BORDER_SIZE)//GAP][(closestCenter[0]-BORDER_SIZE)//GAP] = charPrediction
        #remove the item from blank_centers iff the center is in blank_centers
        for center in blank_centers:
            if(center[0] <= position[0] <= center[2] and center[1] <= position[1] <= center[3]):
                blank_centers.remove(center)
                break
    for center in blank_centers:
        xpos=(((center[0]+center[2])//2)-BORDER_SIZE)//GAP
        ypos=(((center[1]+center[3])//2)-BORDER_SIZE)//GAP
        data.currentBoard[ypos][xpos] = " "
    return JSONResponse(content={"board":data.currentBoard})