import cv2 as cv
import numpy as np
from board_reader.board_finder import BoardFinder
from board_reader.config import *
import os

def parse_folder_into_data(basefolder,outputfolder):
    for filename in os.listdir(basefolder):
        if filename in parsed_images:
            continue
        image = cv.imread(basefolder + filename)
        finder = BoardFinder(False)
        image=finder.extract_scrabble_board(image)
        centers = []
        for i in range(15):
            for j in range(15):
                centers.append((BORDER_SIZE +int(gap/2) + i * gap, BORDER_SIZE +int(gap/2) + j * gap))

        if image is not None:
            letters = finder.extract_tiles(image,centers)
            print("Found " + str(len(letters)) + " letters")
            for letter in letters:
                cv.imshow("board", letter)
                key = chr(cv.waitKey(0))
                key = key.lower()
                if key == " ":
                    continue
                if not os.path.exists(outputfolder + key):
                    os.makedirs(outputfolder + key)
                count = len(os.listdir(outputfolder + key))
                if count < 50:
                    cv.imwrite(outputfolder + key + "/" + str(count)+".jpg", letter)
                else:
                    print("Skipping " + key + " as it has enough images")
            parsed_images_file.write(filename + "\n")
        else:
            print("Error: Board not found")


cv.namedWindow("board", cv.WINDOW_NORMAL)
parsed_images_file = open("./data_collection/parsed_images.txt", "r+")
parsed_images = parsed_images_file.readlines()
parsed_images_file.seek(0, os.SEEK_END)
parsed_images = [x.strip() for x in parsed_images]

parse_folder_into_data("./images/training/","./data_collection/training_data/")
parse_folder_into_data("./images/validation/","./data_collection/testing_data/")

