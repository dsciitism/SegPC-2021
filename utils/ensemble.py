#This script was used to ensemble/aggregrate the outputs of three models for submission

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model1_predictions',type=str,required=True,help="path to the predictions of first model")
parser.add_argument('--model2_predictions',type=str,required=True,help="path to the predictions of second model")
parser.add_argument('--model3_predictions',type=str,required=True,help="path to the predictions of third model")
parser.add_argument('--final_predictions',type=str,required=True,help="path where the ensembled outputs should be saved")

args=parser.parse_args()

import cv2,glob,os,shutil
from tqdm import tqdm
import concurrent.futures

path1 = args.model1_predictions+"/"
path2 = args.model2_predictions+"/"
path3 = args.model3_predictions+"/"

new_path = args.final_predictions+"/"
if not os.path.exists(new_path):
  os.system("mkdir {}".format(new_path))

imgs = sorted([name.split('_')[0] for name in os.listdir(path1)])

def work(it):
  img = imgs[it]
  imgs1 = glob.glob(path1+img+"_*.bmp")
  imgs2 = glob.glob(path2+img+"_*.bmp")
  imgs3 = glob.glob(path3+img+"_*.bmp")

  count = 1
  for im in imgs1:
    shutil.copyfile(im,new_path+img+"_"+str(count)+".bmp")
    count = count + 1
  
  for im in imgs2:
    shutil.copyfile(im,new_path+img+"_"+str(count)+".bmp")
    count = count + 1

  for im in imgs3:
    shutil.copyfile(im,new_path+img+"_"+str(count)+".bmp")
    count = count + 1
    
  print(it)

with concurrent.futures.ThreadPoolExecutor(4) as executor:
    executor.map(work,range(len(imgs)))
