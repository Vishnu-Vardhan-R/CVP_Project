import os
import random
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt



class CVPdataset(Dataset):
    
    def __init__(self, rootdir:str, num_classes:int, transform, mode:str = "train", class_list:list = None):
        """
        Args:
            rootdir (str): Root directory of the dataset.
            num_classes (int): Number of classes to sample.
            transform: Transformation to be applied on the images.
            mode (str): Mode of the dataset ('train' or 'val').
            class_list (list, optional): List of classes to sample. Defaults to None.
        """
        self.rootdir = rootdir
        self.num_classes = num_classes
        self.transform = transform
        self.mode = mode
        self.traindir = os.path.join(self.rootdir, "train/")
        
        random.seed(10)
        filenames = [filename for filename in os.listdir(self.traindir) if filename.startswith("n")]
        if self.num_classes > len(filenames):
            raise ValueError(f"Number of classes requested is greater than the total classes(={len(filenames)}) available in the dataset!")
        
        # decide which classes to sample
        if class_list is not None:
            for label in class_list:
                if label not in filenames:
                    raise ValueError(f"Class {label} not found in the dataset!")
                if len(class_list) != self.num_classes:
                    raise ValueError(f"Number of classes in class_list({len(class_list)}) does not match num_classes({self.num_classes})!")
            # use the provided class_list
            classes = class_list
        else:
            if self.num_classes > len(filenames):
                raise ValueError(f"Number of classes requested is greater than the total classes(={len(filenames)}) available in the dataset!")
            
            # extract a random subset of classes
            classes = random.sample(filenames, self.num_classes)

        if self.mode == 'train':
            # load the training data as dataframe
            self.train_df = pd.DataFrame([(im,label, classes.index(label)) for label in classes for im in os.listdir(self.traindir+label+'/images/')])
            
            # save the train set as a csv file in rootdir
            self.train_df.to_csv(rootdir+'train_set.csv', header=False, index=False)
            
            self.train_data = [(Image.open(self.traindir+row[1]+'/images/'+row[0]), row[2]) for idx, row in self.train_df.iterrows()]
            
            # print the classes and their corresponding labels
            print(f"Sampled classes(class, label): {dict( zip(classes, range(len(classes))) )}\n")
            print(f"{self.mode}Dataset is fetched!")
            print("Data structure: [(PIL Image, Label)]")
            print(f"Length of train data: {len(self.train_data)}")
            
        elif self.mode == 'val':
            self.valdir = os.path.join(self.rootdir, "val/")
            # load the validation data as a dataframe
            self.val_df = pd.read_csv(self.valdir+'val_annotations.txt', sep="\t", header=None)
            self.val_df = self.val_df[self.val_df[1].isin(classes)]
            
            # save the validation set as a csv file in rootdir
            savedf = self.val_df.iloc[:, :2]
            savedf[2] = [classes.index(label) for label in self.val_df[1]]
            savedf.to_csv(rootdir+'val_set.csv', header=False, index=False)
            
            self.val_data = [(Image.open(self.valdir+'images/'+im), classes.index(label)) for im,label in zip(self.val_df[0], self.val_df[1])]
            print(f"{self.mode}Dataset is fetched!")
            print("Data structure: [(PIL Image, Label)]")
            print(f"Length of val data: {len(self.val_data)}")
            
        else:
            raise ValueError("Invalid Mode. Choose either 'train' or 'val'.")


    def __sample__(self, index:int):
        """
        Display a sample image and its corresponding label.
        Args:
            index (int): Index of the image to display.
        """
        if self.mode == 'train':
            img, label =  self.train_data[index]
            filename = self.train_df.iloc[index, 0]
            print("(Filename, class, label)")
            print(f"({self.train_df.iloc[index, 0]}, {filename.split('_')[0]}, {label})")
        elif self.mode == 'val':
            img, label = self.val_data[index]
            print("(Filename, class, label)")
            print(f"({self.val_df.iloc[index, 0]}, {self.val_df.iloc[index, 1]}, {label})")
        
        # convert any grayscale to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # apply transformation
        if self.transform:
            img = self.transform(img)
        
        plt.imshow(img.permute(1, 2, 0))


    def __len__(self):
        """
        Returns the length of the dataset based on the mode.
        Returns:
            int: Length of the dataset.
        """
        if self.mode == 'train':
            return len(self.train_data)
        elif self.mode == 'val':
            return len(self.val_data)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the image to fetch.
        Returns:
            tuple: Tuple containing the image and its corresponding label.
        """
        if self.mode == 'train':
            img, label = self.train_data[index]
        elif self.mode == 'val':
            img, label = self.val_data[index]
        
        # convert any grayscale to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        label = torch.tensor(int(label))
        
        if self.transform:
            img = self.transform(img)
        
        return (img,label)
