import csv
from shutil import copy
import os
import sys

shuf=sys.argv[5]

with open('./'+shuf+'_labels.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    train_female = int(sys.argv[1])
    train_male = int(sys.argv[2])
    test_male = int(sys.argv[3])
    test_female = int(sys.argv[4])
    line_count = 0
    parent = shuf+"/"+shuf+"_TRAIN_F_"+str(train_female)+"_M_"+str(train_male)+"_TEST_F"+str(test_female)+"_M_"+str(test_male)
    os.mkdir(parent)
    os.mkdir(parent + "/train")
    os.mkdir(parent + "/train/female")
    os.mkdir(parent + "/train/male")
    os.mkdir(parent + "/test")
    os.mkdir(parent + "/test/female")
    os.mkdir(parent + "/test/male")
    feat_no = 21 # MALE
    
    for row in csv_reader:
        if line_count < 1:
            print(row[feat_no])
            line_count += 1
            continue
        filename = "/home/vkonton/filteringNN/celebA/data/full-data/" + row[0]
        if row[feat_no] == "-1":
            if train_female > 0: 
                copy(filename, parent +"/train/female")
                train_female -= 1
            elif test_female > 0:
                copy(filename, parent + "/test/female")
                test_female -= 1
        else: 
            if train_male > 0: 
                copy(filename, parent + "/train/male")
                train_male -= 1
            elif test_male > 0: 
                copy(filename, parent + "/test/male")
                test_male -= 1
        if train_female == 0 and train_male ==0 and test_female ==0 and test_male ==0:
            break
