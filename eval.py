import math
import random
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torchvision.models import efficientnet_b2
from torchvision import transforms
# import custom module
from CVPDataset import CVPdataset
from DataLoader import visual_acuity_transform, contrast_sensitivity_transform



def show_images(images:list, labels:list = None, save_path:str = None) -> None:
    """
    Display a grid of images.
    Args:
        images (list): List of images to display.
        labels (list): Optional list of labels for the images.
        save_path (str): Optional path to save the displayed images.
    """
    plt.ion()  # Turn on interactive mode
    images_per_row = 5
    num_rows = math.ceil(len(images) / images_per_row)

    fig = plt.figure()  # Create a new figure
    fig, axes = plt.subplots(num_rows, images_per_row, figsize=(15, num_rows * 3))
    # Flatten axes for easier indexing
    axes = axes.flatten()

    for idx, im in enumerate(images):
        axes[idx].imshow(im.permute(1, 2, 0))  # Permute to convert (C, H, W) to (H, W, C)
        axes[idx].axis('off')  # Turn off axis for better visualization
        if labels is not None:
            axes[idx].set_title(f"Image {idx+1}, Label: {labels[idx]}")
        else:
            axes[idx].set_title(f"Image {idx+1}")
    
    # Hide any unused subplots
    for idx in range(len(images), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    
    # Save the figure if a save path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create directory if it doesn't exist
        plt.savefig(save_path)
        print(f"Saved figure to: {save_path}")
    
    plt.show()
    plt.pause(0.001)  # Pause to allow the figure to update


def load_images(num_imgs:int = 6, transform:str = "none", age:int = 3, mode:str = 'val') -> list:
    """
    Load images from the dataset and apply transformations.
    Args:
        num_imgs (int): Number of images to load.
        transform (str): Type of transformation to apply ('acuity', 'cs', or 'none').
        age (int): Age in months for the transformation.
        mode (str): Mode of the dataset ('train' or 'val').
    Returns:
        list: List of loaded images.
    """
    if transform == "acuity":
        dataset = CVPdataset(ROOT_DIR, NUM_CLASSES, visual_acuity_transform(age), mode, CLASS_LIST)
        print(f"{transform} transformation applied!")
    
    elif transform == "cs":
        dataset = CVPdataset(ROOT_DIR, NUM_CLASSES, contrast_sensitivity_transform(age), mode, CLASS_LIST)
        print(f"{transform} transformation applied!")
    
    elif transform == "none":
        dataset = CVPdataset(ROOT_DIR, NUM_CLASSES, transforms.ToTensor(), mode, CLASS_LIST)
        print(f"{transform} transformation applied!")
    
    else:
        raise ValueError("Invalid transform. Choose from 'acuity', 'cs', or 'none'.")
    
    print(f"Loading {num_imgs} images from {mode} set")
    
    images = []
    labels = []
    random.seed()
    # Randomly sample images from the dataset
    for idx in list(random.sample(range(len(dataset)), num_imgs)):
        img, label = dataset.__getitem__(idx)
        images.append(img)
        labels.append(label)
    
    # save the images
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    show_images(images, labels, save_path=SAVE_PATH+f"{timestamp}_{transform}_images.png")
    print(f"Images are saved to images/{timestamp}_{transform}_images.png")
    
    return images


def get_activation(img:torch.tensor, layer_idx:str) -> tuple:
    """
    Get the activation of a specific layer in the model for a given image.
    Args:
        img (torch.tensor): Input image tensor.
        layer_idx (str): Index of the layer to get the activation from.
    Returns:
        tuple: Activation of the specified layer and the predicted class.

    # to get the architecture of the model
    print(model.features)
    
    # hooks the activation for the entire sequential layer (Sequential[1])
    model.features[1].register_forward_hook(hook)
    
    # hooks the activation for the entire submodule (MBConv2d)
    model.features[1][0].register_forward_hook(hook)
    
    # hooks the acivation of first conv layer in a submodule (Sequential[1]->MBConv2d->Conv2d)
    list(model.features[1][0].children())[0][0][0].register_forward_hook(hook)
    """

    activation = {}
    def hook(model, input, output):
        activation[layer_idx] = output.detach()

    # Register the hook for the specified layer
    # Note: Adjust the layer index based on your model architecture
    if layer_idx == 'features0':
        model.features[0][0].register_forward_hook(hook)
    elif layer_idx == 'features1':
        list(model.features[2][0].children())[0].register_forward_hook(hook)
    elif layer_idx == 'features2':
        list(model.features[3][0].children())[0][0].register_forward_hook(hook)
    
    with torch.no_grad():
        output = model(img.unsqueeze(0).to(torch.float32))
    _, predicted_class = torch.max(output, 1)  # Get the class with the highest score

    return torch.mean(activation[layer_idx], dim=1), predicted_class.item()


def RDM(images:list, layer_idx:str) -> torch.tensor:
    """
    Compute the representational dissimilarity matrix (RDM) for a given layer of the model.
    Args:
        images (list): List of images to compute the RDM for.
        layer_idx (str): denotes the layer to compute the RDM for.
    Returns:
        torch.tensor: Representational dissimilarity matrix.
    """
    # layer activations for the input image
    layer_activations = []
    # prediction class for the input image
    predictions = []
    # flattened representation vectors of the layer activations
    layer_fvs = []
    for im in images:
        act, pred = get_activation(im, layer_idx)
        layer_activations.append(act)
        predictions.append(pred)
        layer_fvs.append(torch.flatten(act, start_dim=0, end_dim=-1))
    
    # display and save the layer activations
    print(f"Layer activations for {layer_idx}:")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    show_images(layer_activations, predictions, save_path=SAVE_PATH+f"{timestamp}_{layer_idx}_activations.png")
    
    # Stack the representations of all images
    rep_stack = torch.stack(layer_fvs, dim=0)
    
    # compute 1 - pearson correlation
    dissimilarity = 1 - torch.corrcoef(rep_stack)
    
    return dissimilarity    


def load_model(model_path:str) -> tuple:
    """
    Load the model from the given path.
    Args:
        model_path (str): Path to the .pth file.
    Returns:
        tuple: Loaded model and its evaluation metrics.
    """
    model = efficientnet_b2()
    model.classifier = nn.Sequential( nn.Dropout(p=0.2, inplace=True),
                                nn.Linear(in_features=1408, out_features=NUM_CLASSES, bias=True))
    # Load the model state dict
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_info = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(model_info['model_state_dict'])
    model_eval = model_info['metrics']
    # Set the model to evaluation mode
    model.eval()
    
    return model, model_eval


def display_heatmaps(rdms):
    """
    Display heatmaps for the RDMs.
    Args:
        rdms (dict): Dictionary containing RDMs for each model and layer.
    """
    for path, rdm in rdms.items():
        # Determine the grid size for the heatmaps  
        fig, axes = plt.subplots(1, len(rdm), figsize=(15, 5))
        axes = axes.flatten()  # Flatten axes for easier indexing
        
        for idx, (layer, matrix) in enumerate(rdm.items()):
            sns.heatmap(matrix.numpy(), annot=False, cmap="Greens", cbar=True, ax=axes[idx])
            axes[idx].set_title(f"{layer}")

        fig.suptitle(f"Dissimilarity Heatmaps for Model: {path}", fontsize=16)
        
        # Hide any unused subplots
        for idx in range(len(rdm), len(axes)):
            axes[idx].axis('off')
        
        # Save the figures
        filename = os.path.basename(path).split('.')[0] # Extract the filename without extension
        os.makedirs(os.path.dirname(SAVE_PATH+f"{filename}_heatmaps.png"), exist_ok=True)  # Create directory if it doesn't exist
        plt.savefig(SAVE_PATH+f"{filename}_heatmaps.png") 

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the subtitle
        plt.show()



if __name__ == "__main__":

    #### Hyperparameters #####
    
    # root directory of the dataset to load the images
    ROOT_DIR = '/Users/vishnu/FAU/Semester 2/Computational visual perception/Project/tiny-imagenet-200/'
    # path to save the images
    SAVE_PATH = os.path.join(os.getcwd(), "images/")
    
    # provide class list that model was trained on
    # the classes are in the order of the labels in the dataset
    CLASS_LIST = ['n01644900', 'n01443537', 'n01774384', 'n01770393', 'n01945685']
    NUM_CLASSES = 5
    
    # provide the path to the .pth files of the models
    PTH_PATHS = ['pth files/5M1none.pth', 
                'pth files/5M2acuity_1_3_5_48.pth',
                'pth files/5M3cs_1_3_8_48.pth',
                'pth files/5M4acuity_cs_1_3_48.pth',]
    
    # number of images to capture the RDM
    NUM_IMGS = 9

    rdms = {}
    rdvs = {}
    for path in PTH_PATHS:
        print(f"Loading model from {path}")
        # Load the model and its evaluation metrics
        model, eval = load_model(path)

        # load images from the dataset with different transforms
        imgs1 = load_images(int(NUM_IMGS/3))
        imgs2 = load_images(int(NUM_IMGS/3), "acuity")
        imgs3 = load_images(int(NUM_IMGS/3), "cs")
        imgs = imgs1 + imgs2 + imgs3
            
        # layer indices to compute the RDM
        layer_indices = ['features0', 'features1', 'features2'] 
        rdvec = {}
        rdmatrix = {}

        for index in layer_indices:
            rdmatrix[index] = RDM(imgs, index)
            
            # extract lower triangular part of the matrix excluding the diagonal
            lower_tri = torch.tril(rdmatrix[index], diagonal=-1)
            # Flatten the non-zero values
            flattened = lower_tri[lower_tri != 0]
            rdvec[index] = flattened
        
        # store the RDM and lower-triangle flattened vector for each model
        rdms[path] = rdmatrix
        rdvs[path] = rdvec
        
        if os.path.isdir(SAVE_PATH):
            os.rename(SAVE_PATH, f"M{PTH_PATHS.index(path)+1}")
    
    # display the heatmaps for the RDMs
    print("Displaying heatmaps for the RDMs...")
    display_heatmaps(rdms) 