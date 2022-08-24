# ## Data Split
# 
# This script splits data in train, validation and testing

#Libraries
import argparse
import os
import os.path as osp
import random

parser = argparse.ArgumentParser(description='Generador de máscaras a partir de anotaciones VIA.')

parser.add_argument('--data_root', type=str, help='Dirección de la carpeta de los datos.')
parser.add_argument('--masks_in', type=str, help='Dirección de la carpeta de las mascaras.')
parser.add_argument('--masks_fails', type=str, help='Dirección del file .txt con las máscaras defectuosas.')

args = parser.parse_args()

#Get filenames
filenames=os.listdir(args.masks_in)
random.shuffle(filenames)

#Delete failed masks from dataset
mask_fails=[]
with open(args.masks_fails, 'r') as f:
    for line in f:
        if line[-1:] == "\n":
            mask_fails.append(line[:-1])
        else:
            mask_fails.append(line)
for i in mask_fails:
    if i in filenames:
        filenames.remove(i)

#Split data in train, validation and testing
with open(args.data_root + "/splits" + '/train.txt', 'w') as f:
    # select first 3/5 as train set
    train_length = int(len(filenames)*3/5)
    f.writelines(osp.splitext(line)[0] + '\n' for line in filenames[:train_length])

with open(args.data_root + "/splits" + '/val.txt', 'w') as f:
    # select 1/5 as validation set
    valid_length = int(len(filenames)*4/5)
    f.writelines(osp.splitext(line)[0] + '\n' for line in filenames[train_length:valid_length])
    
with open(args.data_root + "/splits" + '/test.txt', 'w') as f:
    # select last 1/5 as testing set
    f.writelines(osp.splitext(line)[0] + '\n' for line in filenames[valid_length:])

