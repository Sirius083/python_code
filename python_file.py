# read file from txt file
import pandas as pd
bbox_file = r'E:\tiny_imagenet\tiny-imagenet-200\train\n01443537\n01443537_boxes.txt'
data = pd.read_csv(bbox_file, sep=" ")


# save dict to file
def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
