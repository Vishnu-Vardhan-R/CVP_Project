# Computational model of an infant vision

## Repo structure
```
├── Computational_model_of_infant_vision.ipynb  # Main notebook
├── DataLoader.py                               # Custom dataloader for TinyImageNet
├── model.py                                    # Model training engine
├── requirements.txt                            # (Optional) Package requirements
└── README.md                                   # You're here!
```

A few minutes after an infant is born, their eyes start to open and look around. Though the vision is premature at this stage and continues to develop throughout the years, the early vision characteristics have a huge influence over shaping the adult vision. With enough literature background, the following work attempts to study, implement and evaluate the developmental aspects of an infant vision using a deep neural network model.

## Charateristics of vision

Here, emphasis is laid on studying the following characteristics of vision. Images were subjected to visual acuity and contrast sensitivity transformations for different ages(in months) to simulate the progressive development in infants. 

* **Visual Acuity**: Visual acuity transform is implemented by applying gaussian filters with age-specific sigma(σ) values. A normal developing infant is born with a very poor visual acuity(below 20/600 which is beyond the criterion for legal blindness) [[1]](#1). Though this poor acuity comes across as a limitation at first, the High initial acuity hypothesis [*(Vogelsang et al., 2018)*](#1) suggests that it’s a feature of the system in developmental progression of vision. Children who underwent treatment for congenital cataracts during the first year of infancy, commence their vision with high initial acuity. This high initial acuity and lack of low acuity in the development progression lead to impaired face-discrimination performance in their adolescence. Children with initially high acuity would be biased toward local processing and are impaired on tasks that rely on extended spatial processing (for example, detection of configural changes, holistic face processing) [[1]](#1).

* **Contrast sensitivity**: Contrast in infant vision has a progressive relation with spatial frequencies. In the study by [*(Banks and Salapatek, 1978)*](#3),contrast sensitivity was measured using several test stimuli (vertical sinewave grating and unpatterned stimuli) of varying spatial frequencies and contrast levels. The results were expressed as *contrast sensitivity functions(CSF)*, showing sensitivity across different spatial frequencies for different age groups. 

    <br>
    <p align="center">
    <a href="https://github.com/Vishnu-Vardhan-R/CVP_Project/blob/main/imgs/csf%20curves.png">
        <img src="https://github.com/Vishnu-Vardhan-R/CVP_Project/blob/main/imgs/csf%20curves.png"
            alt="Contrast sensitivty curves" width="400" height="300">
    </a>
    </p>
    <br>

    This contrast sensitivity function is implemented in `CSFtransform.py`. The associated parameters of CSF - peak gain (**𝛄<sub>max</sub>**), peak spatial frequency (**𝑓<sub>max​</sub>**), bandwidth (**𝛽**), and truncation value (**𝛿**), is extracted from [[2](*Fig. 1.A*)](#2). When provided with these parameter values, the CSF curves for different ages (see above figure) are obtained, which are then applied to the Fourier domain of the input images. This modifies the contrast values for particular frequencies of the input image, ultimately mimicking the perception of an infant's vision.

<br>
<p align="center">
<a href="https://github.com/Vishnu-Vardhan-R/CVP_Project/blob/main/imgs/transform.png">
    <img src="https://github.com/Vishnu-Vardhan-R/CVP_Project/blob/main/imgs/transform.png"
        alt="Image transformations" width="500" height="500">
</a>
</p>
<br>

<details><summary><b>Instructions for CSFtransform.py</b></summary>
<br>

1. You need to specify the age in months. Input age is mapped to one of the following values - `1.0`, `3.0`, `8.0`, and `48.0` months

    ```py
    from CSFTransform import CSF_transform

    csf = CSF_transform(age=8)
    ```

2. The input image should be a NumPy array (e.g., loaded using OpenCV or converted from a PIL image).

    ```py
    image = cv2.imread("path/to/image.jpg")
    
    transformed_image = csf(image)
    cv2.imshow("Transformed Image", transformed_image)
    ```

</details>


## Custom dataset modules

A custom dataset class `CVPDataset` is utilized to transform a collection of images from the `tiny-imagenet-200` dataset. The `DataLoader.py` file provides utility functions to create a PyTorch DataLoader for the `CVPdataset` with optional transformations based on visual acuity or contrast sensitivity. These transformations simulate human visual perception at different ages.

<br>
<details><summary><b>Instructions for CVPdataset.py</b></summary>
<br>

1. Initialize the dataset for either training or validation mode by specifying the required parameters. If particular classes are required, use the class_list parameter.

    ```py
    from torchvision import transforms
    from CVPDataset import CVPdataset
    
    rootdir = "/path/to/dataset" 
    classlist=["class1", "class2", "class3"]    
    num_classes = 3
    mode = "train"                  # Options: "train", "val"
    transform = transforms.ToTensor()

    dataset = CVPdataset(rootdir=rootdir, num_classes=num_classes, transform=transform, mode=mode, class_list=class_list)
    ```

2. Fetch a specific sample using the dataset's `__getitem__` method or by indexing.

    ```py
    img, label = dataset.__getitem__(5)
    plt.imshow(img.permute(1, 2, 0))
    ```

3. Use the `__sample__` method to display an image and its corresponding label.

    ```py
    dataset.__sample__(index=5) 
    ```

### Notes
The dataset expects the following directory structure:
```
rootdir/
├── train/
│   ├── class1/
│   │   ├── images/
│   │   │   ├── image1.jpg
│   │   │   ├── image2.jpg
│   │   │   └── ...
│   ├── class2/
│   │   └── images/
│   └── ...
└── val/
    ├── images/
    └── val_annotations.txt
```
* The `val_annotations.txt` file should contain tab-separated values with image names and their corresponding classes.
* The `train_set.csv` and `val_set.csv` files are automatically generated in the `rootdir` for reference.
* Ensure the `num_classes` does not exceed the total number of available classes in the dataset.

</details>
<br>

<details><summary><b>Instructions for DataLoader.py</b></summary>
<br>

1. Initialize the Dataloader. Ensure the dataset directory structure matches the requirements of the CVPdataset class.

    ```py
    from DataLoader import dataloader

    num_classes = 3
    transform = "acuity"    # Options: "acuity", "cs", "none"
    age = 8.0
    mode = "train"          # Options: "train", "val"
    classlist=["class1", "class2", "class3"]    # Optional

    train_loader = dataloader(rootdir=rootdir, num_classes=num_classes, transform=transform, age=age, mode=mode)

    ```

2. Iterate through the DataLoader to access batches of images and labels.

    ```py
    for images, labels in train_loader:
        print(f"Batch of images: {images.shape}")
        print(f"Batch of labels: {labels.shape}")
    ```

### Notes
* The `transform` parameter determines the type of transformation applied to the images:
    - `"acuity"`: Applies Gaussian blur based on age to simulate visual acuity.
    - `"cs"`: Applies the Contrast Sensitivity Function (CSF) transformation based on age.
    - `"none"`: No transformations are applied; images are converted to tensors.
* The dataset should follow the directory structure outlined in the `CVPDataset.py` instructions.
* Ensure the `age` parameter is within the supported ranges: `1`, `3`, `8`, or `48` months.
* The `dataloader` function internally uses the `CVPdataset` class from `CVPDataset.py`.

</details>


## Training the network

This work utilises **EfficientNet-B2** [[5]](#5) as the training model due to its well-balanced tradeoff between size and accuracy. The classifier layer of the **EfficientNet-B2** has been replaced by a custom fully connected layer and includes a dropout layer with rate 0.2 to prevent overfitting. 

As stated, infant vision parameters mature progressively with age. Consequently, a developmental curriculum(youngest to eldest) is created to train the deep network model. Training the model with data in this natural sequence not only mimics this behaviour, but could also potentially outperform random data sequences [[6]](#6).  Four training conditions were defined (see Table below). The `model.py` file contains the `engine()` function, which is used to train the model on a given dataset. The `train.py` script is the main entry point for training models using the engine function from `model.py`. It supports training regimen with curriculum learning with transformations such as visual acuity and contrast sensitivity.


| **Model** | **Transform** | **Epochs** |
|----------|----------------|------------|
| M1 | No transforms | 60 |
| M2 | Visual Acuity Curriculum | 0-15: Age 1mo<br>16-30: Age 3mo<br>31-45: Age 5mo<br>46-60: Age 48mo |
| M3 | Contrast Sensitivity Curriculum | 0-15: Age 1mo<br>16-30: Age 3mo<br>31-45: Age 8mo<br>46-60: Age 48mo |
| M4 | CS + VA Shuffle Curriculum |  0-10: Age 1mo(VA)<br>11-20: Age 1mo(CS)<br>21-30: Age 3mo(VA)<br>31-40: Age 3mo(CS)<br>41-50: Age 48mo(VA)<br>51-60: Age 48mo(CS) |

<br>
<details><summary><b>Instructions for model.py</b></summary>
<br>

1. Ensure you have DataLoader objects for both training and validation datasets. You can use the dataloader function from DataLoader.py to create them.

    ```py
    from DataLoader import dataloader

    train_data = dataloader(rootdir="/path/to/dataset", num_classes=3, transform="acuity", age=8.0, mode="train")
    val_data = dataloader(rootdir="/path/to/dataset", num_classes=3, transform="acuity", age=8.0, mode="val")
    ```

2. Call the engine function with the required parameters to train and evaluate the model.

    ```py
    from model import engine

    Num_classes = 10
    Batchsize = 64
    StartEpoch = 0
    Num_epochs = 10

    model_eval, trained_model = engine(train_data=train_data, val_data=val_data,
    Num_classes=Num_classes, Batchsize=Batchsize,
    StartEpoch=StartEpoch, Num_epochs=Num_epochs,
    model_name="M1")
    ```

3. Training progress and metrics (loss and accuracy) are logged using the wandb library. Ensure you have initialized a W&B project before running the script.

4. The `engine` function returns:
* `model_eval` (dict): A dictionary containing training and validation metrics:
    - `train_losses`
    - `train_accuracy`
    - `val_losses`
    - `val_accuracy`
* `trained_model` (torch.nn.Module): The trained PyTorch model.

</details>
<br>


<details><summary><b>Instructions for train.py</b></summary>
<br>

The `train.py` file is the main script for training models using the CVP dataset. It supports different transformations (e.g., visual acuity and contrast sensitivity) and curriculum learning strategies.

1. Define Dataset and Training Parameters. The script uses the `sample_classes` function to randomly sample classes from the dataset. This ensures that only the specified number of classes is used for training.

    ```py
    from train import train_model

    ROOT_DIR = '/path/to/dataset'
    NUM_CLASSES = 3
    CLASS_LIST = sample_classes(NUM_CLASSES, ROOT_DIR)
    
    BATCH_SIZE = 64
    NUM_EPOCHS = 60
    ```

2. The script trains models with different configurations. For example,

    No transformation: Trains the model on the dataset without any transformations.
    ```py
    M1 = train_model('M1_none', rootdir, NUM_CLASSES, BATCH_SIZE, NUM_EPOCHS, ["none"])
    ```

    Curriculum learning with contrast sensitivity transformation: Trains the model with progressively increasing contrast sensitivity based on age.
    ```py
    M3 = train_model('M3_cs_1_3_8_48', rootdir, NUM_CLASSES, BATCH_SIZE, NUM_EPOCHS, ["cs"], [1.0, 3.0, 12.0])
    ```

3. Each trained model is saved as a `.pth` file in the current directory. The filename includes the number of classes and the model name.

    ```bash
    3M1_none.pth
    3M3_cs_1_3_8_48.pth
    ```

4. The `train_model` function internally uses the `engine` function from `model.py` and the `dataloader` function from `DataLoader.py`.

5. The script uses Weights & Biases (W&B) for experiment tracking. Ensure you have login credentials for W&B before running the script. 

</details>


## Model evaluation

The script `eval.py` loads trained models, extracts activations from selected layers for a small batch of images, computes pairwise dissimilarities based on the Pearson correlation, and visualizes the results using heatmaps. 

<br>
<details><summary><b>Instructions for eval.py</b></summary>
<br>

1. `show_images()` displays a grid of images with labels and saves the grid to a specified path.

2. For each model, load the trained weights and evaluation metrics, and load images with different transformations.

    ```python
    NUM_IMGS = 9        # total no. of images to calculate RDMs

    model, eval = load_model(path)
    imgs1 = load_images(int(NUM_IMGS/3))
    imgs2 = load_images(int(NUM_IMGS/3), "acuity")
    imgs3 = load_images(int(NUM_IMGS/3), "cs")
    imgs = imgs1 + imgs2 + imgs3
    ```

3. Use the `get_activation` function to extract activations from specific layers of the model for a given image. This is achieved with the help of a `hook()` function. To access a particular layer activations, register the hook function onto the desired layer using `model.features()` and `register_forward_hook()`

    ```py
    from eval import load_model, get_activation

    model, eval_metrics = load_model('path/to/model.pth')

    image = images[0]  # Use the first image
    activation, predicted_class = get_activation(image, layer_idx='features1')
    print(f"Predicted class: {predicted_class}")
    ```

</details>


## Inferences and conclusions

<br>
<p align="center">
<a href="https://github.com/Vishnu-Vardhan-R/CVP_Project/blob/main/imgs/train%20acc.png">
    <img src="https://github.com/Vishnu-Vardhan-R/CVP_Project/blob/main/imgs/train%20acc.png"
        alt="Training Accuracy" width="500" height="320">
</a>

<a href="https://github.com/Vishnu-Vardhan-R/CVP_Project/blob/main/imgs/train%20loss.png">
    <img src="https://github.com/Vishnu-Vardhan-R/CVP_Project/blob/main/imgs/train%20loss.png"
        alt="Training Loss" width="500" height="320">
</a>
</p>
<br>


<p align="center">
<a href="https://github.com/Vishnu-Vardhan-R/CVP_Project/blob/main/imgs/val%20accu.png">
    <img src="https://github.com/Vishnu-Vardhan-R/CVP_Project/blob/main/imgs/val%20accu.png"
        alt="Validation Accuracy" width="500" height="320">
</a>

<a href="https://github.com/Vishnu-Vardhan-R/CVP_Project/blob/main/imgs/val%20loss.png">
    <img src="https://github.com/Vishnu-Vardhan-R/CVP_Project/blob/main/imgs/val%20loss.png"
        alt="Validation Loss" width="500" height="320">
</a>
</p>
<br>

## References

<a id="1">[1]</a> 
L. Vogelsang, S. Gilad-Gutnick, E. Ehrenberg, A. Yonas, S. Diamond, R. Held, P. Sinha, Potential downside of high initial visual acuity, Proc. Natl. Acad. Sci. U.S.A. 115 (44) 11333-11338, https://doi.org/10.1073/pnas.1800901115

<a id="2">[2]</a> 
Lukas Vogelsang, Marin Vogelsang, Gordon Pipa, Sidney Diamond, Pawan Sinha, Butterfly effects in perceptual development: A review of the ‘adaptive initial degradation’ hypothesis (Developmental Review, Volume 71, March 2024, 101117), https://doi.org/10.1016/j.dr.2024.101117

<a id="3">[3]</a> 
Banks MS, Salapatek P, Acuity and contrast sensitivity in 1-, 2-, and 3-month-old human infants, Invest Ophthalmol Vis Sci. 1978 Apr;17(4):361-5. 

<a id="4">[4]</a> 
Min SH, Reynaud A. Applying Resampling and Visualization Methods in Factor Analysis to Model Human Spatial Vision. Invest Ophthalmol Vis Sci. 2024 Jan 2;65(1):17. PMID: 38180771; PMCID: PMC10785955, https://doi.org/10.1167/iovs.65.1.17

<a id="5">[5]</a> 
Tan, M., & Le, Q.V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ArXiv, abs/1905.11946, https://doi.org/10.48550/arXiv.1905.11946

<a id="6">[6]</a> 
Saber Sheybani, Himanshu Hansaria, Justin N. Wood, Linda B. Smith, and Zoran Tiganj. (2024). Curriculum learning with infant egocentric videos


