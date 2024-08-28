import os
import shutil
import sys


target = 50
dict = {}
for filename in os.listdir("./data_collection/training_data/"):
    if os.path.isdir("./data_collection/training_data/" + filename):
        length = len(os.listdir("./data_collection/training_data/" + filename))
        print(filename+"-Additional Tiles needed: "+str(target - length))
        dict[filename] = length

sorted_dict = dict.items()
sorted_dict = sorted(sorted_dict, key=lambda x: x[1])
for item in sorted_dict:
    if item[1]!=50:
        print(item[0]+str(item[1])+"\n")
        