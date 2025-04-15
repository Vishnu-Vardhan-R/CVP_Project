# Computational model of an infant vision

A few minutes after an infant is born, their eyes start to open and look around. Though the vision is premature at this stage and continues to develop throughout the years, the early vision characteristics have a huge influence over shaping the adult vision. With enough literature background, the following work attempts to study, implement and evaluate the developmental aspects of an infant vision using a deep neural network model.

## Charateristics of vision

Here, emphasis is laid on studying the following characteristics of vision:

* **Visual Acuity**: Visual acuity transform is implemented by applying gaussian filters with age-specific sigma(σ) values. A normal developing infant is born with a very poor visual acuity(below 20/600 which is beyond the criterion for legal blindness) [[1]](link). Though this poor acuity comes across as a limitation at first, the High initial acuity hypothesis [*(Vogelsang et al., 2018)*](link) suggests that it’s a feature of the system in developmental progression of vision. Children who underwent treatment for congenital cataracts during the first year of infancy, commence their vision with high initial acuity. This high initial acuity and lack of low acuity in the development progression lead to impaired face-discrimination performance in their adolescence. Children with initially high acuity would be biased toward local processing and are impaired on tasks that rely on extended spatial processing (for example, detection of configural changes, holistic face processing) [[1]](link).

* **Contrast sensitivity**: Contrast in infant vision has a progressive relation with spatial frequencies. In the study by [*(Banks and Salapatek, 1978)*](link),contrast sensitivity was measured using several test stimuli (vertical sinewave grating and unpatterned stimuli) of varying spatial frequencies and contrast levels. The results were expressed as *contrast sensitivity functions(CSF)*, showing sensitivity across different spatial frequencies for different age groups. 

    <p align="center">
    <a href="https://github.com/Vishnu-Vardhan-R/CVP_Project/blob/main/imgs/Screenshot%202025-04-15%20at%2018.45.41.png">
        <img src="https://github.com/Vishnu-Vardhan-R/CVP_Project/blob/main/imgs/Screenshot%202025-04-15%20at%2018.45.41.png"
            alt="Contrast sensitivty curves" width="400" height="300">
    </a>
    </p>

    This contrast sensitivity function is implemented in `CSFtransform.py`. The associated parameters of CSF - peak gain (**𝛄max**), peak spatial frequency (**𝑓max​**), bandwidth (**𝛽**), and truncation value (**𝛿**), is extracted from the literature [[2](*Fig. 1.A*)](link). When provided with these parameter values, the CSF curves for different ages (see below figure) are obtained, which are then applied to the Fourier domain of the input images. This modifies the contrast values for particular frequencies of the input image, ultimately mimicking the perception of an infant's vision.


<br>
<details><summary><b>Usage instructions for CSFtransform</b></summary>
<br>

1. You need to specify the age in months. Input age is mapped to one of the following values - `1.0`, `3.0`, `8.0`, and `48.0` months

    ```py
    csf = CSF_transform(age=8)
    ```

2. The input image should be a NumPy array (e.g., loaded using OpenCV or converted from a PIL image).

    ```py
    image = cv2.imread("path/to/image.jpg")
    
    transformed_image = csf(image)
    ```

3. The input image should be in RGB format. If using OpenCV, ensure the image is converted from BGR to RGB before applying the transform.

4. The output image is normalized to the range [0, 1]. You may need to scale it back to [0, 255] for saving or further processing.

<br>

</details>

<p align="center">
<a href="https://github.com/Vishnu-Vardhan-R/CVP_Project/blob/main/imgs/Screenshot%202025-04-15%20at%2018.46.51.png">
    <img src="https://github.com/Vishnu-Vardhan-R/CVP_Project/blob/main/imgs/Screenshot%202025-04-15%20at%2018.46.51.png"
        alt="Image transformations" width="400" height="300">
</a>
</p>


## Custom dataset modules

A custom dataset class `CVPDataset` is utilized to transform a collection of images from the `tiny-imagenet-200` dataset. Images were subjected to visual acuity and contrast sensitivity transformations for different ages(in months) to simulate the progressive development in infants. 

<br>
<details><summary><b>Usage instructions for CVPdataset</b></summary>
<br>

2. **Initialize the Dataset**

    You can initialize the dataset for either training or validation mode by specifying the required parameters.

    ```py
    rootdir = "/path/to/dataset"  # Root directory of the dataset
    num_classes = 10             # Number of classes to sample
    transform = None             # Transformation to apply (e.g., torchvision transforms)
    mode = "train"               # Mode: "train" or "val"
    class_list = None            # Optional: List of specific classes to sample

    # Create a dataset instance
    dataset = CVPdataset(rootdir=rootdir, num_classes=num_classes, transform=transform, mode=mode, class_list=class_list)
    ```

3. **Access Dataset Length**
    You can get the number of samples in the dataset using the `len()` function.

    ```py
    print(f"Number of samples: {len(dataset)}")
    ```

4. **Fetch a Sample**
    You can fetch a specific sample using the dataset's `__getitem__` method or by indexing.

    ```py
    img, label = dataset[0]  # Fetch the first sample
    ```

5. **Display a Sample**
    Use the `__sample__` method to display an image and its corresponding label.

    ```py
    dataset.__sample__(index=0)  # Display the first sample
    ```

6. **Use with a DataLoader**
   To iterate over the dataset in batches, use `DataLoader.py`

    ```py
    dataset.__sample__(index=0)  # Display the first sample
    ```

</details>
<br>



## Training the network

This work utilises EfficientNet-B2[1] as the training model due to its well-balanced tradeoff between size and accuracy. The classifier layer of the EfficientNet-B2 has been replaced by a custom fully connected layer and includes a dropout layer with rate 0.2 to prevent overfitting. 

As stated, infant vision parameters mature progressively with age. Consequently, a developmental curriculum(youngest to eldest) is created to train the deep network model effectively. Training the model with data in this natural sequence not only mimics this behaviour, but could also potentially outperform random data sequences[2].  Four training conditions were defined (see TABLE). 


######################

* **ES modules** and **tree-shaking** support.
* Add Size Limit to **GitHub Actions**, **Circle CI** or another CI system
  to know if a pull request adds a massive dependency.
* **Modular** to fit different use cases: big JS applications
  that use their own bundler or small npm libraries with many files.
* Can calculate **the time** it would take a browser
  to download and **execute** your JS. Time is a much more accurate
  and understandable metric compared to the size in bytes.
  Additionally, you can [customize time plugin] via config
  for every check with network speed, latency and so on.
* Calculations include **all dependencies and polyfills**
  used in your JS.



With **[GitHub action]** Size Limit will post bundle size changes as a comment
in pull request discussion.



With `--why`, Size Limit can tell you *why* your library is of this size
and show the real cost of all your internal dependencies.
We are using [Statoscope] for this analysis.



[GitHub action]: https://github.com/andresz1/size-limit-action
[Statoscope]:    https://github.com/statoscope/statoscope
[cult-img]:      http://cultofmartians.com/assets/badges/badge.svg
[cult]:          http://cultofmartians.com/tasks/size-limit-config.html
[customize time plugin]: https://github.com/ai/size-limit/packages/time#customize-network-speed







## Config

### Plugins and Presets

Plugins or plugin presets will be loaded automatically from `package.json`.
For example, if you want to use `@size-limit/webpack`, you can just use
`npm install --save-dev @size-limit/webpack`, or you can use our preset
`@size-limit/preset-big-lib`.

Plugins:

* `@size-limit/file` checks the size of files with Brotli (default), Gzip
  or without compression.
* `@size-limit/webpack` adds your library to empty webpack project
  and prepares bundle file for `file` plugin.


#### Third-Party Plugins

Third-party plugins and presets named starting with `size-limit-` are also supported.
For example:

* [`size-limit-node-esbuild`](https://github.com/un-ts/size-limit/tree/main/packages/node-esbuild)
  is like `@size-limit/esbuild` but for Node libraries.



Each section in the config can have these options:

* **path**: relative paths to files. The only mandatory option.
  It could be a path `"index.js"`, a [pattern] `"dist/app-*.js"`
  or an array `["index.js", "dist/app-*.js", "!dist/app-exclude.js"]`.
* **import**: partial import to test tree-shaking. It could be `"{ lib }"`
  to test `import { lib } from 'lib'`, `*` to test all exports,
  or `{ "a.js": "{ a }", "b.js": "{ b }" }` to test multiple files.
* **limit**: size or time limit for files from the `path` option. It should be
  a string with a number and unit, separated by a space.
  Format: `100 B`, `10 kB`, `500 ms`, `1 s`.
* **name**: the name of the current section. It will only be useful
  if you have multiple sections.
* **message**: an optional custom message to display additional information,
  such as guidance for resolving errors, relevant links, or instructions
  for next steps when a limit is exceeded.
