from data.base_dataset import get_params, get_transform, BaseDataset
from PIL import Image
from data.image_folder import make_dataset
import os
import pdb


class TrainImageDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--train_image_dir', type=str, required=True,
                            help='path to the directory that contains photo images')
        parser.add_argument('--train_image_list', type=str, required=True,
                            help='path to the directory that contains photo images')
        parser.add_argument('--train_image_postfix', type=str, default="",
                            help='path to the directory that contains photo images')
        return parser

    def initialize(self, opt):
        self.opt = opt
        image_paths = self.get_paths(opt)

        self.image_paths = image_paths

        size = len(self.image_paths)
        self.dataset_size = size

    def get_paths(self, opt):
        image_dir = opt.train_image_dir
        image_list = opt.train_image_list
        names = open(image_list).readlines()
        filenames = list(map(lambda x: x.strip('\n')+opt.train_image_postfix, names))
        image_paths = list(map(lambda x: os.path.join(image_dir, x), filenames))
        return image_paths

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        # input image (real images)
        try:
            image_path = self.image_paths[index]
            image = Image.open(image_path)
            image = image.convert('RGB')
            params = get_params(self.opt, image.size)
            transform_image = get_transform(self.opt, params)
            image_tensor = transform_image(image)
            input_dict = {
                          'image': image_tensor,
                          'path': image_path,
                          }
            return input_dict
        except:
            print(f"skip {image_path}")
            return self.__getitem__((index+1)%self.__len__())
