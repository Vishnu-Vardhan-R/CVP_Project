import wandb
import torch
import os
import random
from math import ceil
from timeit import default_timer as timer
# import custom modules
from DataLoader import dataloader
from model import engine


def train_model(modelName:str, rootdir:str, 
                num_classes:int, batch_size:int, 
                num_epochs:int, transforms:list, 
                ages:list=[48], class_list:list=None, Model_info=None) -> tuple:
    """
    Function to train the model with different transformations and ages.
    Args:
        modelName (str): Name of the model.
        rootdir (str): Root directory of the dataset.
        num_classes (int): Number of classes in the dataset.
        batch_size (int): Batch size for training.
        num_epochs (int): Total number of epochs for training.
        transforms (list): List of transformations to apply.
        ages (list, optional): List of ages for curriculum learning. Defaults to None.
        class_list (list, optional): List of classes to sample. Defaults to None.
        Model_info (tuple, optional): Tuple containing model and evaluation metrics. Defaults to None.
    Returns:
        tuple: Tuple containing model evaluation metrics and the trained model.
    """
    
    print(f"Training initiated for {modelName} with {num_classes} classes")
    print("wandb initialised!\n")
    # wandb is used for logging and tracking experiments
    wandb.login()
    wandb.init(project="CVP_Project", name=f"{modelName}")
    
    start_epoch = 1
    EpochsPerDataset = int(ceil(num_epochs/(len(transforms)*len(ages))))
    
    for age in ages:
        for transform in transforms:
            # Check if the number of epochs exceeds the total number of epochs
            if start_epoch+EpochsPerDataset > num_epochs+1:
                EpochsPerDataset = num_epochs - start_epoch + 1
                
            start = timer()
            print(f"Dataset: applied transform = {transform}, age = {age} month(s)\n")
            trainData = dataloader(rootdir, num_classes, transform, age, mode='train', classlist=class_list, batch_size=batch_size)
            valData = dataloader(rootdir, num_classes, transform, age, mode='val', classlist=class_list, batch_size=batch_size)
            print("Datasets are loaded to the engine!")
            print(f"Time taken to load the dataset: {timer()-start:.2f}s\n")
            
            Model_info = engine(train_data=trainData, val_data=valData,
                                    Num_classes=num_classes, Batchsize=batch_size,
                                    StartEpoch=int(start_epoch), Num_epochs=int(EpochsPerDataset),
                                    model_info=Model_info, model_name=modelName)
            print(f"Model finished training with Dataset: {transform} ~ {age} month(s)\n\n")
            
            start_epoch += EpochsPerDataset

    model_eval, model = Model_info
    torch.save({'model_state_dict': model.state_dict(), 'metrics': model_eval}, f"{num_classes}{modelName}.pth")
    print(f"Model saved as {num_classes}{modelName}.pth.\n")
    wandb.finish()
    
    return Model_info


def sample_classes(num_classes:int, rootdir:str) -> list:
    """
    Function to sample classes from the dataset.
    Args:
        num_classes (int): Number of classes to sample.
        rootdir (str): Root directory of the dataset.
    Returns:
        list: List of sampled classes.
    """
    filenames = [filename for filename in os.listdir(os.path.join(rootdir, "train/")) if filename.startswith("n")]
    if num_classes > len(filenames):
        raise ValueError(f"Number of classes requested is greater than the total classes(={len(filenames)}) available in the dataset!")

    return random.sample(filenames, num_classes)


if __name__ == "__main__":
        
    # Define the root directory of the dataset
    ROOT_DIR = '/Users/vishnu/FAU/Semester 2/Computational visual perception/Project/tiny-imagenet-200/'
    
    NUM_CLASSES = 5
    random.seed(10)
    # Sample classes from the dataset
    CLASS_LIST = sample_classes(NUM_CLASSES, ROOT_DIR)
    print(f"Sampled classes(class, label): {dict( zip(CLASS_LIST, range(len(CLASS_LIST))) )}\n")

    BATCH_SIZE = 64
    NUM_EPOCHS = 60

    # training with no transforms and no curriculum
    M1 = train_model('M1none', ROOT_DIR, NUM_CLASSES, BATCH_SIZE, NUM_EPOCHS, ["none"], class_list=CLASS_LIST)

    # developmental curriculum with acuity transform
    M2 = train_model('M2acuity_1_3_5_48', ROOT_DIR, NUM_CLASSES, BATCH_SIZE, NUM_EPOCHS, ["acuity"], [1.0, 3.0, 5.0, 48.0], CLASS_LIST)

    # developmental curriculum with cs transform
    M3 = train_model('M3cs_1_3_8_48', ROOT_DIR, NUM_CLASSES, BATCH_SIZE, NUM_EPOCHS, ["cs"], [1.0, 3.0, 8.0, 48.0], CLASS_LIST)

    # curriculum learning with both acuity and cs transform
    M4 = train_model('M4acuity_cs_1_3_48', ROOT_DIR, NUM_CLASSES, BATCH_SIZE, NUM_EPOCHS, ["acuity", "cs"], [1.0, 3.0, 48.0], CLASS_LIST)