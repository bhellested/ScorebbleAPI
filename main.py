import cv2 as cv
import numpy as np
from board_reader.board_finder import BoardFinder
from board_reader.config import *


cv.namedWindow("board", cv.WINDOW_NORMAL)
debug=True
def debug_show(image):
    if debug:
        cv.imshow("board", image)
        cv.waitKey(0)


def extract_tiles(board,centers):
    tiles = []
    #create a mask with the black area
    hsv = cv.cvtColor(board, cv.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([255, 255, 50])
    mask = cv.inRange(hsv, lower_black, upper_black)
    debug_show(mask)
    #find the contours
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    contours = [contour for contour in contours if cv.contourArea(contour) > 200]
    for contour in contours:
        #board = cv.drawContours(board, [contour], -1, (0, 255, 0), 3)
        #find the center of the contour
        M = cv.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        #draw a circle at the center
        #cv.circle(board, (cX, cY), 7, (255, 255, 255), -1)
        #find the distance to the closest center
        closest = min(centers, key=lambda x: (x[0] - cX) ** 2 + (x[1] - cY) ** 2)
        distance = ((closest[0] - cX) ** 2 + (closest[1] - cY) ** 2) ** 0.5
        print(distance)
        #create a 91x91 square around the center and grab that into a new image
        tile = mask[cY - 45:cY + 45, cX - 45:cX + 45]
        debug_show(tile)
        tiles.append(tile)
    return tiles


board = cv.imread("/Users/brianhellested/dev/scrabbleSolver/images/rotate.jpg")
finder = BoardFinder(False)
board = finder.extract_scrabble_board(board)
print("showing board")
cv.imshow("board", board)
cv.waitKey(0)
# oriented_board = extract_scrabble_board(board)
# cv.imshow("board", oriented_board)
# cv.waitKey(0)
# play_area,centers = extract_play_area(oriented_board)
# cv.imshow("board", play_area)
# cv.waitKey(0)
# extract_tiles(play_area,centers)

cv.destroyAllWindows()