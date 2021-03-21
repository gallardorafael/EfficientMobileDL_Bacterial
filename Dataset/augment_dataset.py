"""
This script augments the original DIBaS dataset, which consists
in:
* 32 classes (species of bacteria)
* Aprox. 660 examples of colonies (aprox. 20 examples per specie)

Params:
    input_dir: Directory with the original images, i.e the vanilla version
        of the DIBaS dataset.
    output_dir: Directory to save the augmented dataset.
    size: Output size of each sample.
"""

import argparse
import os
from PIL import Image
from torchvision import transforms
import torch

# Arguments
parser = argparse.ArgumentParser(description='Options for the augmentation')
parser.add_argument('--input_dir', default='./')
parser.add_argument('--output_dir', default='./')
parser.add_argument('--size', default=224)

args = parser.parse_args()

# Function to FiveCrop the full dataset
def FiveCropFull(max_zoom):
    id = 0

    shapes_list = [value for value in range(100, max_zoom + 100, 100)]
    print(shapes_list)
    # If the augmented folder does not exist
    try:
        os.mkdir(args.output_dir)
    except OSError as error:
        print(error)

    toPILImage = transforms.Compose([transforms.ToPILImage()])
    for specie in os.listdir(args.input_dir):
        specie_path = os.path.join(args.input_dir, specie)
        out_specie_path = os.path.join(args.output_dir, specie)
        # If the augmented specie folder does not exist
        try:
            os.mkdir(out_specie_path)
        except OSError as error:
            print(error)
        for filename in os.listdir(specie_path):
            print('FiveCropping:',os.path.join(specie_path, filename))
            image = Image.open(os.path.join(specie_path, filename))

            for shape_d in shapes_list:
                print("Shape:", shape_d)
                transform = transforms.Compose([transforms.FiveCrop(shape_d),
                                                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))])
                crops_t = transform(image) # 4 dimensions tensor
                crops_ts = torch.split(crops_t, 1, 0) # List of 3 dimension tensors
                for i, img_t in enumerate(crops_ts):
                    new_img_t = torch.squeeze(img_t)
                    img = toPILImage(new_img_t)
                    r_name = filename + str(id)
                    id += 1
                    path = os.path.join(out_specie_path, r_name)
                    print('Saving:',path+'.tif')
                    img.save(fp=path+'.tif')

# Function to resize the full dataset
def resizeFull(shape):
    try:
        os.mkdir(args.output_dir)
    except OSError as error:
        print(error)
    transform = transforms.Compose([transforms.Resize(shape, interpolation=Image.LANCZOS)])
    for specie in os.listdir(args.input_dir):
        specie_path = os.path.join(args.input_dir, specie)
        out_specie_path = os.path.join(args.output_dir, specie)
        # If the augmented specie folder does not exist
        try:
            os.mkdir(out_specie_path)
        except OSError as error:
            print(error)
        for filename in os.listdir(specie_path):
            print('Resizing:',os.path.join(specie_path, filename))
            image = Image.open(os.path.join(specie_path, filename))
            resized = transform(image)
            r_name = filename + '_resized_'
            path = os.path.join(out_specie_path, r_name)
            print('Saving:',path+'.tif')
            resized.save(fp=path+'.tif')

def main():
    FiveCropFull(max_zoom=700)
    resizeFull(args.size)

if __name__ == "__main__":
    main()
