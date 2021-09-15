import os
import pdb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--folder')  # path data folders
parser.add_argument('--output')  # path to save as txt
parser.add_argument('--postfix', default=".jpg")  # file

opt = parser.parse_args()

save_name = opt.output #"imagenet_image_list.txt"
path_data = opt.folder #'../../data/datasets/ILSVRC12_image_train'
file_postfix =opt.postfix  #".JPEG"
path_data = os.path.abspath(path_data)
if path_data.endswith("/"): path_data=path_data[:-1]
if not file_postfix.startswith("."): file_postfix="."+file_postfix

folders = os.listdir(path_data)

file_list = []

def get_filename(path, list_save):
    print(len(file_list))
    if os.path.isdir(path):
        for folder in os.listdir(path):
            get_filename(os.path.join(path, folder), list_save)
    elif path.endswith(file_postfix):
        _path = path.replace(file_postfix, "")
        _path = _path.replace(path_data+"/", "")
        list_save.append(_path)

get_filename(path_data, file_list)
file_list = list(map(lambda x: x.replace(path_data+"/", "")+'\n', file_list))
with open(save_name, "w") as f:
    f.writelines(file_list)
