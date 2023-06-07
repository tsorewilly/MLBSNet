import os
import numpy as np
import pandas as pd
import random 

# In[] IDENTIFY DATASET AND RENAME THEM
jpgs = os.listdir("gw_23/jpg")
pngs = os.listdir("gw_23/png")

with open("train_data.txt",'w') as f:
    for jpg in jpgs:
        png = jpg.replace("jpg","png")
        #Generate file of partitioned training and testing set
        if png in pngs:
            f.write(jpg+";"+png+"\n")
#file should contain both a jpg and its corresponding png file

print("Files Renaming Done! ____________!")

# In[] SPLIT DATASET INTO TRAINING AND TESTING SETS
with open("gw_23/train_data.txt",'r') as file:
    lines = file.readlines()
    np.random.shuffle(lines)

with open("gw_23/train.txt",'w') as file:
    for line in lines[:int(len(lines)*0.9)]:
        file.write(line)

with open("gw_23/test.txt",'w') as file:
    for line in lines[int(len(lines)*0.9):]:
        file.write(line.split('.')[0]+"\n")

"""
dataSet = open('train_data.txt', 'rb') 
trainSet = open("train.txt", 'wb') 
testSet = open("test.txt", 'wb')

for line in dataSet: 
    r = random.randint(7000, 9699) 
    if (0.0 <=  r <= 0.75): 
        trainSet.write(line) 
    elif (0.75 < r <= 0.875): 
        testSet.write(line) 
    
dataSet.close() 
trainSet.close() 
testSet.close() 
"""

#training_data = df.sample(frac=0.8, random_state=25)
"""
pic_path = "TMI Data/"

with open("train.txt", "w", encoding="UTF-8") as ff:
    for name_0 in os.listdir(jpgs):
        for name_1 in os.listdir(pngs + "/" + name_0):
            for name_2 in os.listdir(pic_path + "/" + name_0 + "/" + name_1):
                pic_input_path = name_0 + "/" + name_1 + "/" + name_2
                ff.write(pic_input_path + "\n")
    ff.close()
"""

print("well done____________!")
