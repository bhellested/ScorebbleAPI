#first load up the scrabble_cnn_model.h5
#then load up the image
import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model
from board_reader.board_finder import BoardFinder
from board_reader.config import *
import os
import sys

input_image="./IMG_0762.jpeg"
model = load_model("./scrabble_cnn_model.keras")

image = cv.imread(input_image)
if image is None:
    print("Error: Image not found")
    sys.exit(1)
finder = BoardFinder(False)
image=finder.extract_scrabble_board(image)
cv.imshow("board", image)
cv.waitKey(0)
centers = []
for i in range(15):
        for j in range(15):
            centers.append((BORDER_SIZE +int(GAP/2) + i * GAP, BORDER_SIZE +int(GAP/2) + j * GAP))
tiles,tilepos = finder.extract_tiles(image)

processed_tiles = []
for tile in tiles:
    tile = cv.cvtColor(tile, cv.COLOR_BGR2GRAY)
    tile = cv.resize(tile, (50, 50), interpolation=cv.INTER_LINEAR)
    tile = np.expand_dims(tile, axis=2) 
    processed_tiles.append(tile)

tiles_dataset = np.stack(processed_tiles, axis=0)
predictions = model.predict(tiles_dataset)
boardConfiguration = [["" for x in range(15)] for y in range(15)]
for prediction,position in zip(predictions,tilepos):
    charPrediction = chr(np.argmax(prediction) + 65)
    closestCenter=min(centers, key=lambda x: (x[0] - position[0]) ** 2 + (x[1] - position[1]) ** 2)
    boardConfiguration[(closestCenter[1]-BORDER_SIZE)//GAP][(closestCenter[0]-BORDER_SIZE)//GAP] = charPrediction
for row in boardConfiguration:
    print(row)

    
