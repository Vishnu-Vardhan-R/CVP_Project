from torchvision import transforms
from torch.utils.data import DataLoader
# import custom modules
from CVPDataset import CVPdataset
from CSFTransform import CSF_transform



def map_age(age:float) -> int:
    """Map age in months to the corresponding value.
    Args:
        age (float): Age in months.
    Returns:
        int: Mapped value based on age.
    Raises:
        ValueError: If age is not in the specified ranges.
    """
    mapping = {(0.0, 2.5): 1, (2.5, 6.0): 3, (6.0, 12.0): 8, (12.0, float("inf")): 48}
    for (low, high), value in mapping.items():
        if low < age <= high:
            return value
    raise ValueError("Possible age values(in months) are 1, 3, 8, 48 only!")


def contrast_sensitivity_transform(age:float) -> transforms.Compose:
    """Apply contrast sensitivity transformation based on age.
    Args:
        age (float): Age in months.
    Returns:
        transform: Transformation to be applied.
    """
    transform = transforms.Compose([CSF_transform(map_age(age)), transforms.ToTensor()])
    return transform


def visual_acuity_transform(age:float) -> transforms.Compose:
    """Apply visual acuity transformation based on age.
    Args:
        age (float): Age in months.
    Returns:
        transform: Transformation to be applied.
    """
    if age > 6.0:
        transform = transforms.ToTensor()
    elif (age >4.5) & (age <=6.0):
        transform = transforms.Compose([transforms.GaussianBlur(kernel_size=15, sigma=1), transforms.ToTensor()])
    elif (age >2.5) & (age <=4.5):
        transform = transforms.Compose([transforms.GaussianBlur(kernel_size=15, sigma=2), transforms.ToTensor()])
    elif (age >1.5) & (age <=2.5):
        transform = transforms.Compose([transforms.GaussianBlur(kernel_size=15, sigma=3), transforms.ToTensor()])
    elif (age >0.0) & (age <=1.5):
        transform = transforms.Compose([transforms.GaussianBlur(kernel_size=15, sigma=4), transforms.ToTensor()])
    else:
        transform = transforms.ToTensor()
    return transform


def dataloader(rootdir:str, num_classes:int, 
                transform:str="none", age:float=48.0, 
                mode:str="train", classlist:list=None, batch_size:int=64,
                shuffle:bool=True, num_workers:int=2,
                pin_memory:bool=True) -> DataLoader:
    """Create a DataLoader for the CVP dataset.
    Args:
        rootdir (str): Root directory of the dataset.
        num_classes (int): Number of classes in the dataset.
        transform (str): Type of transformation to apply ('acuity', 'cs', or 'none').
        age (float): Age in months.
        mode (str): Mode of the dataset ('train' or 'val').
        class_list (list, optional): List of classes to sample. Defaults to None.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of worker threads for data loading.
        pin_memory (bool): Whether to pin memory for faster data transfer.
    Returns:
        DataLoader: DataLoader for the CVP dataset.
    """
    if transform == "acuity":
        # Visual acuity transformation
        dataset = CVPdataset(rootdir, num_classes, visual_acuity_transform(age), mode, classlist)
        print(f"{transform} transformation applied!")
    
    elif transform == "cs":
        # Contrast sensitivity transformation
        dataset = CVPdataset(rootdir, num_classes, contrast_sensitivity_transform(age), mode, classlist)
        print(f"{transform} transformation applied!")
    
    elif transform == "none":
        # No transformations
        dataset = CVPdataset(rootdir, num_classes, transforms.ToTensor(), mode, classlist)
        print(f"{transform} transformation applied!")
    
    else:
        raise ValueError("Invalid transform. Choose from 'acuity', 'cs', or 'none'.")

    print(f"Image tensor shape: {dataset.__getitem__(1)[0].shape}")
    print(f"Label tensor shape: {dataset.__getitem__(1)[1].shape}\n")
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
