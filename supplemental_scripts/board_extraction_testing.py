import cv2 as cv
import numpy as np
import os
from board_reader.board_finder import BoardFinder
boardfinder = BoardFinder(False)

for filename in os.listdir("./"):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        print(filename)
        image = cv.imread( filename)
        image = boardfinder.extract_scrabble_board(image)
        if image is not None:
            cv.imshow("board", image)
            cv.waitKey(0)
        boardfinder.extract_blanks(image)
        continue
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        blur = cv.medianBlur(gray, 5)
        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpen = cv.filter2D(blur, -1, sharpen_kernel)
        thresh = cv.threshold(sharpen, 160, 255, cv.THRESH_BINARY_INV)[1]
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
        close = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=2)
        cnts = cv.findContours(close, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(image, cnts[0], -1, (0,255,0), 3)
        cv.imshow("board", image)
        cv.waitKey(0)