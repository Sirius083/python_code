# read file from txt file
import pandas as pd
bbox_file = r'E:\tiny_imagenet\tiny-imagenet-200\train\n01443537\n01443537_boxes.txt'
data = pd.read_csv(bbox_file, sep=" ")
