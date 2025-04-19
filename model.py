import random
from math import ceil
from timeit import default_timer as timer
import torch
from torch import nn
from torchvision.models import efficientnet_b2
import tqdm
import wandb


#### Hyperparameters #####
DROPOUT_P = 0.2
LEARNING_RATE = 0.01


def engine(train_data, val_data, Num_classes:int, Batchsize:int,
            StartEpoch:int, Num_epochs:int, model_info:tuple = None, model_name:str='Model_1') -> tuple:
    """
    Function to train and evaluate the model.
    Args:
        train_data (DataLoader): Training data loader.
        val_data (DataLoader): Validation data loader.
        Num_classes (int): Number of classes in the dataset.
        Batchsize (int): Batch size for training.
        StartEpoch (int): Starting epoch for training.
        Num_epochs (int): Total number of epochs for training.
        model_info (tuple, optional): Tuple containing model and evaluation metrics. Defaults to None.
        model_name (str, optional): Name of the model. Defaults to 'Model_1'.
    Returns:
        tuple: Tuple containing model evaluation metrics and the trained model.
    """
    
    random.seed(0)
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_info is None:
        model = efficientnet_b2()
        model.classifier = nn.Sequential(
            nn.Dropout(p=DROPOUT_P, inplace=True),
            nn.Linear(in_features=1408, out_features=Num_classes, bias=True)
        )
        # tracks the model evaluation metrics
        model_eval = {"train_losses": [],
                "train_accuracy": [],
                "val_losses": [],
                "val_accuracy": []}
    else:
        model_eval, model = model_info
    
    
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    
    print(f"\nTraining {model_name} for {StartEpoch}:{StartEpoch+Num_epochs-1} epochs\n")
    start_time = timer()

    for epoch in range(StartEpoch, StartEpoch+Num_epochs):
        epochStart = timer()
        
        # loop over both train and validation data
        for mode, data, num_samples in [("train",train_data,len(train_data)), ("val",val_data,len(val_data))]:
            
            num_batches = int(ceil(num_samples/Batchsize))
            pbar = tqdm.tqdm(total=num_batches, desc=f"{mode} epoch {epoch}")

            if mode == "train":
                model.train() # calculate gradients in training mode
            else:
                model.eval() # do not calculate gradients in validation mode

            # initial values of the metrics (loss, accurracy)
            runningLoss = 0.
            correct_predictions = 0
            # total number of images that were processed
            total_samples = 0

            # loop over the batches
            for i_batch, batch in enumerate(data):
                
                images, labels = batch
                # converting image type before passing to the model, to match with model's weight data type.
                images = images.to(device).type(torch.float32)
                labels = labels.to(device)
                predictions = model.forward(images)
                loss = loss_fn(predictions, labels)

                # calculate the metrics for the progress bar
                num_batchsamples = len(images)
                runningLoss += loss.item() * num_batchsamples
                correct_predictions += (torch.argmax(predictions, dim=-1) == labels).sum().item()
                total_samples += num_batchsamples
                pbar.update(1)
                pbar.set_postfix({"loss": runningLoss / total_samples,
                                "accuracy": correct_predictions / total_samples})

                # weight update
                if mode == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # calculate the epoch metrics
            epoch_accuracy = correct_predictions / total_samples
            epoch_loss = runningLoss / total_samples
            if mode == "train":
                model_eval["train_losses"].append(epoch_loss)
                model_eval["train_accuracy"].append(epoch_accuracy)
            else:
                model_eval["val_losses"].append(epoch_loss)
                model_eval["val_accuracy"].append(epoch_accuracy)

            # Log metrics to W&B
            wandb.log({f"{mode}/Accuracy": epoch_accuracy, 
                        f"{mode}/Loss": epoch_loss},
                        step=epoch)
            
            pbar.close()
        
        print(f"Epoch {epoch} runtime: {timer()-epochStart:.2f}s\n")

    print("\nTraining completed!")
    print(f"Total runtime({Num_epochs} epochs): {timer()-start_time:.2f}s")
    
    return model_eval, model
