import os
import splitfolders

input_path = './Projeto/images_full/'
output_path = './Projeto/images_split/'


def rename_images(path):
    if os.path.exists(path):
        for dirpath, dirnames, filenames in os.walk(path):
            for index, file in enumerate(filenames):
                full_path = os.path.join(dirpath, file)
                extension = '.'+full_path.split('.')[-1]
                folder_name = os.path.basename(dirpath)
                newfilename = ''.join([folder_name+str(index), extension])
                os.rename(full_path, os.path.join(dirpath, newfilename))


if __name__ == '__main__':
    rename_images(input_path)
    splitfolders.ratio(input=input_path, output=output_path, seed=1337, ratio=(.8, .2), group_prefix=None, move=False)
