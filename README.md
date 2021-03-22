# Efficient and Mobile Deep Learning Architectures for Fast Identification of Bacterial Strains in Resource-Constrained Devices: Code and instructions.
Repository for the paper: [Efficient and Mobile Deep Learning Architectures for Fast Identification of Bacterial Strains in Resource-Constrained Devices](https://gallardorafael.github.io/assets/docs/DRAFT_EffMobDIBaS.pdf)

Authors: Rafael Gallardo García, Sofía Jarquín Rodríguez, Beatriz Beltrán Martínez, Rodolfo Martínez Torres, Carlos Hernández Gracidas

Note: We highly recommend reading the paper to get a deeper understanding of the work.

## Requirements
Hardware:

- A CUDA capable GPU is recommended.
- At least 15 GB of storage.

Software:

- Python 3.8.5
- matplotlib 3.3.0
- torch 1.5.1
- torchvision 0.6.1
- scikit-learn==0.23.2
- numpy 1.19.1
- seaborn 0.10.1
- efficientnet-pytorch==0.7.0
- Pillow==7.2.0
- Jupyter (to open and run notebooks)

We are also adding a requirements.txt file, but it is not optimized (yet), so may be it will install some not useful stuff. To install from the requirements file (we recommend using a virtual environment):
```
pip3 install -r requirements.txt
```


## Digital Image of Bacterial Species (DIBaS) dataset
The original version of DIBaS dataset contains a total of 33 species of microorganisms, approximately 20 RGB images (of 2048x1532 pixels) per specie. We remove the set of images of Candida albicans colonies as it is considered fungi. The dataset was collected by the Chair of Microbiology of the Jagiellonian University in Krakow. The samples were stained using the Gramm’s method. All images were taken with an Olympus CX31 Upright Biological Microscope and a SC30 camera with a 100 times objective under oil-immersion. The DIBaS dataset is publicly available in the following [link](http://misztal.edu.pl/software/databases/dibas/).

Note: We checked the original link in January/2021 and it was not working, here you have an alternative method to download it (run the command in the folder you want):
1. Download the zip file:
```
https://drive.google.com/file/d/1wCNmQMA3pdHeU1rTrHeLehqIYZ2rF_uy/view?usp=sharing
```
2. Extract:
```
unzip DIBaS.zip
```

It is recommended to put the data into the Dataset/ folder. The Jupyter Notebooks use defined paths to load the files, but you can modify it to your needs.

## Augmented version of the DIBaS dataset
Our data augmentation strategy is aimed to provide atleast 35 different images per each sample in the orig-inal dataset. The proposed approach tries to simulate different levels of zoom for the same bacterial sample,which is achieved by cropping different regions of different sizes of the full image. The full process is described in the following algorithm:

<img src="repo/assets/Algorithm1.png" width=350 align=center>

We provide a script to augment the dataset in the same way. It is available on: Dataset/augment_dataset.py, you can augmente your data as follows:
```
cd Dataset/
python3 augment_dataset.py --input_dir DIBaS/ --output_dir DIBaS_augmented --size 224
```

The following table shows the distribution of samples before and after the augmentation.

<img src="repo/assets/ClassDistrib.png" width=400 align=center>
![ClassDistrib](repo/assets/ClassDistrib.png)

The next figure illustrates a sample of the DIBaS dataset, after applying our augmentation method:

<img src="repo/assets/AugmentationSample.png" width=600 align=center>
![AugmentedSample](repo/assets/AugmentationSample.png)

## Pre-trained models
The best models, both for the original and the augmented version of the dataset are publicy available:

1. original32x224x224in50.pth: Trained over the original version of the dataset.
2. 2fc_augmented32x224x224in10.pth: Trained over the augmented version of the dataset.

Both pretrained models are available on the folder: models/

## Examples and code: The jupyter Notebook
We provide both Jupyter Notebooks, to work with the original and augmented versions of the dataset, respectively:

1. BEST_MobileNetv2_original.ipynb: To work with the original version of the dataset.
2. BEST_test5_MobileNetv2_augmented_2fc.ipynb: To work with the augmented version of the dataset.

## Results
The following table summarizes the results of each trained model, with different hyperparameters and for both versions of the dataset. The first column indicates the number of fully connected layers in the classifier block of the network, the second column is for the version of the dataset, the third column shows the number of training epochs and the fourth presents the accuracy score obtained by the model in that row.

![Results](res/accuracy.png)

### Some graphical visualizations
#### Identification of Escherichia coli

![Example1](res/Ecoli.png)

#### Identification of Neisseria gonorrhoeae

![Example1](res/gonorrhoea.png)

#### Identification of Listeria monocytogenes

![Example1](res/listeria.png)

#### Identification of Lactobacillus plantarum

![Example1](res/plantarum.png)

#### Identification of Staphylococcus aureus

![Example1](res/staph_aureus.png)

## Citations
If you use/modify our code:
```
@inproceedings{gallardo2020bacterialident,
	title={Deep Learning for Fast Identification of Bacterial Strains in Resource Constrained Devices},
	author={Rafael Gallardo-García, Sofía Jarquín-Rodríguez, Beatriz Beltrán-Martínez and Rodolfo Martínez},
	booktitle={Aplicaciones Científicas y Tecnológicas de las Ciencias Computacionales},
	pages={67--78},
	year={2020},
	organization={BUAP}
}
```
If you use the DIBaS dataset, please cite the authors as following:
```
@article{zielinski2017,
	title={Deep learning approach to bacterial colony classification},
	author={Zieli{\'n}ski, Bartosz and Plichta, Anna and Misztal, Krzysztof and Spurek, Przemys{\l}aw and Brzychczy-W{\l}och, Monika and Ocho{\'n}ska, Dorota},
	journal={PloS one},
	volume={12},
	number={9},
	pages={e0184554},
	year={2017},
	publisher={Public Library of Science San Francisco, CA USA}
}
