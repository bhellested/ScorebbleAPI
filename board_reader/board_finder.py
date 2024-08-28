import cv2 as cv
import numpy as np
from .config import *

def nothing(x):
    pass

def create_hsv_trackbars(window_name):
    cv.createTrackbar('H_low', window_name, 0, 179, nothing)
    cv.createTrackbar('S_low', window_name, 0, 255, nothing)
    cv.createTrackbar('V_low', window_name, 0, 255, nothing)
    cv.createTrackbar('H_high', window_name, 179, 179, nothing)
    cv.createTrackbar('S_high', window_name, 255, 255, nothing)
    cv.createTrackbar('V_high', window_name, 255, 255, nothing)

def saturation_find_helper(board):
    window_name = 'HSV Adjuster'
    cv.namedWindow(window_name)
    create_hsv_trackbars(window_name)

    while True:
        hsv = cv.cvtColor(board, cv.COLOR_BGR2HSV)
        
        h_low = cv.getTrackbarPos('H_low', window_name)
        s_low = cv.getTrackbarPos('S_low', window_name)
        v_low = cv.getTrackbarPos('V_low', window_name)
        h_high = cv.getTrackbarPos('H_high', window_name)
        s_high = cv.getTrackbarPos('S_high', window_name)
        v_high = cv.getTrackbarPos('V_high', window_name)

        lower_wood = np.array([h_low, s_low, v_low])
        upper_wood = np.array([h_high, s_high, v_high])

        mask = cv.inRange(hsv, lower_wood, upper_wood)

        result = cv.bitwise_and(board, board, mask=mask)

        cv.imshow('Original', board)
        cv.imshow('Mask', mask)
        cv.imshow('Result', result)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        cv.destroyAllWindows()
        print(f"Final HSV range: [{h_low}, {s_low}, {v_low}] to [{h_high}, {s_high}, {v_high}]")

class BoardFinder:
    def __init__(self, debug=False):
        self.debug = debug
        if self.debug:
            cv.namedWindow("board", cv.WINDOW_NORMAL)

    def debug_show(self, image):
        if self.debug:
            cv.imshow("board", image)
            cv.waitKey(0)

    def force_show(self, image):
        cv.imshow("board", image)
        cv.waitKey(0)
    
    def extract_blanks(self, board):
        hsv = cv.cvtColor(board, cv.COLOR_BGR2HSV)
        lower_wood = np.array([5, 73, 120])
        upper_wood = np.array([40, 126, 255])
        mask = cv.inRange(hsv, lower_wood, upper_wood)
        #self.force_show(mask)
        centerPositions = []
        #we need to do a scan of the board to find the blank tiles
        test = board.copy()
        for i in range(15):
            for j in range(15):
                cutout = mask[BORDER_SIZE + i * GAP:BORDER_SIZE + (i + 1) * GAP, BORDER_SIZE + j * GAP:BORDER_SIZE + (j + 1) * GAP]
                # cv.imshow("cutout", cutout)
                # cv.waitKey(0)
                #if more than 50% of the pixels are white, add it
                if np.sum(cutout > 10) > 0.5 * cutout.size:
                    centerPositions.append((BORDER_SIZE + j * GAP,BORDER_SIZE + i * GAP,BORDER_SIZE + (j + 1) * GAP,BORDER_SIZE + (i + 1) * GAP))
                    #draw a square over the mask
                    test = cv.rectangle(test, (BORDER_SIZE + j * GAP, BORDER_SIZE + i * GAP), (BORDER_SIZE + (j + 1) * GAP, BORDER_SIZE + (i + 1) * GAP), (0, 0, 255), 3)
        return centerPositions

    def extract_tiles(self,board):
        tiles = []
        centerPositions=[]
        #create a mask with the black area
        hsv = cv.cvtColor(board, cv.COLOR_BGR2HSV)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 70])
        mask = cv.inRange(hsv, lower_black, upper_black)

        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv.contourArea, reverse=True)
        contours = [contour for contour in contours if cv.contourArea(contour) > 200]
        for contour in contours:
            #find the center of the contour
            M = cv.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            minY = max(0, cY - 45)
            maxY = min(BOARD_SIZE, cY + 45)
            minX = max(0, cX - 45)
            maxX = min(BOARD_SIZE, cX + 45)
            centerPositions.append((cX, cY));
            tile = board[minY:maxY, minX:maxX]
            tile = cv.resize(tile, (50,50),interpolation=cv.INTER_LINEAR)
            tiles.append(tile)
        return tiles,centerPositions
    

    def extract_scrabble_board(self, board):
        hsv = cv.cvtColor(board, cv.COLOR_BGR2HSV)
        #Create three masks, one for both sides of red and a third for white.
        lower_red = np.array([0, 50, 100])
        upper_red = np.array([8, 255, 255])
        mask = cv.inRange(hsv, lower_red, upper_red)
        self.debug_show(mask)
        lower_red = np.array([175,50,50])
        upper_red = np.array([180,255,255])
        mask += cv.inRange(hsv, lower_red, upper_red)
        self.debug_show(mask)
        lower_white = np.array([0, 0, 230])
        upper_white = np.array([255, 255, 255])
        mask += cv.inRange(hsv, lower_white, upper_white)

        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv.contourArea, reverse=True)
        max_contour = contours[0]
        #draw this contour out
        if self.debug:
            boarderContour = cv.drawContours(board, [max_contour], -1, (0, 255, 0), 3)
            self.debug_show(boarderContour)
        play_area_contour = contours[1]
        epsilon = 0.1 * cv.arcLength(max_contour, True)
        approx = cv.approxPolyDP(max_contour, epsilon, True)

        #ensure the polygon has 4 sides
        if len(approx) == 4:
            board_corners = approx
        else:
            print(approx)
            print("Error: Board not found")
            return None
        
        #apply perspective transform to isolate the board
        pts1 = np.float32([board_corners[0], board_corners[1], board_corners[2], board_corners[3]])
        pts2 = np.float32([[0, 0], [0, BOARD_SIZE], [BOARD_SIZE, BOARD_SIZE], [BOARD_SIZE, 0]])
        matrix = cv.getPerspectiveTransform(pts1, pts2)
        transformed = cv.warpPerspective(board, matrix, (BOARD_SIZE, BOARD_SIZE))
        transformed_mask = cv.warpPerspective(mask, matrix, (BOARD_SIZE, BOARD_SIZE))
        self.debug_show(transformed_mask)
        #now lets find the play area and properly rotate the image
        hsv = cv.cvtColor(transformed, cv.COLOR_BGR2HSV)
        lower_red = np.array([0, 50, 100])
        upper_red = np.array([8, 255, 255])
        mask = cv.inRange(hsv, lower_red, upper_red)
        self.debug_show(mask)
        lower_red = np.array([175,50,50])
        upper_red = np.array([180,255,255])
        mask += cv.inRange(hsv, lower_red, upper_red)
        self.debug_show(mask)
        lower_white = np.array([0, 0, 230])
        upper_white = np.array([255, 255, 255])

        mask = cv.bitwise_not(mask)
        #create a mask for the red color
        self.debug_show(mask)
        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)
        play_area_contour = sorted_contours[0]
        if self.debug:
            boarderContour = cv.drawContours(transformed, [play_area_contour], -1, (0, 255, 0), 3)
            self.debug_show(boarderContour)
        epsilon = 0.1 * cv.arcLength(play_area_contour, True)
        approx = cv.approxPolyDP(play_area_contour, epsilon, True)

        if len(approx) == 4:
            board_corners = approx
        else:
            print("Error: Play area not found")
            return None
        #now we find the line with the longest distance to the border
        #this will get aligned to the left side
        maxIndex =0
        maxDistFound = 0
        for i in range(4):
            center = ((board_corners[i][0]) + (board_corners[(i + 1) % 4][0])) / 2
            minDist = min( min(center[0], 1500 - center[0]), min(center[1], 1500 - center[1]))
            if minDist > maxDistFound:
                maxDistFound = minDist
                maxIndex = i
        
        center = (board_corners[maxIndex][0] + board_corners[(maxIndex + 1) % 4][0]) / 2
        if center[0] > 600 and center[0] < 950:
            if center[1] < 750:
                angle = 90
            else:
                angle = 270
        else:
            if center[0] < 750:
                angle = 0
            else:
                angle = 180
        #rotate
        rotated = cv.getRotationMatrix2D((750, 750), angle, 1)
        transformed = cv.warpAffine(transformed, rotated, (1500, 1500))
        self.debug_show(transformed)
        #basically the above, but we use a white mask to find the play area
        hsv = cv.cvtColor(transformed, cv.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 0])
        upper_white = np.array([255, 100, 255])
        mask = cv.inRange(hsv, lower_white, upper_white)
        self.debug_show(mask)
        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)
        play_area_contour = sorted_contours[0]
        epsilon = 0.1 * cv.arcLength(play_area_contour, True)
        approx = cv.approxPolyDP(play_area_contour, epsilon, True)
        #ensure the polygon has 4 sides
        if len(approx) == 4:
            board_corners = approx
        else:
            print("Error: Play area not found")
            return None
        #apply perspective transform to isolate the play area
        #sort bord corners by y value
        board_corners = sorted(board_corners, key=lambda x: x[0][1])
        #ensure that index 0 is the top left corner
        if board_corners[0][0][0] > board_corners[1][0][0]:
            temp = board_corners[0]
            board_corners[0] = board_corners[1]
            board_corners[1] = temp
        if board_corners[3][0][0] > board_corners[2][0][0]:
            temp = board_corners[2]
            board_corners[2] = board_corners[3]
            board_corners[3] = temp
        print (board_corners)
        pts1 = np.float32([board_corners[0], board_corners[1], board_corners[2], board_corners[3]])
        pts2 = np.float32([[0, 0], [1500, 0], [1500, 1500], [0, 1500]])
        #apply perspective transform
        matrix = cv.getPerspectiveTransform(pts1, pts2)
        transformed = cv.warpPerspective(transformed, matrix, (1500, 1500))
        self.debug_show(transformed)
        return transformed
