# deepfake-detector

This was part of a project for university: creating a model for deepfake detection of Indian faces, with a test accuracy of 99.68%.
This repository contains code to train a custom ResNet-18 model with various augmentations, MixUp, and Test-Time Augmentation (TTA) applied on an image classification task. The model is trained using a custom dataset and evaluated on both validation and test sets.

The dataset contains 65483 images of which 31483 are real images and 34000 are fakes. The fakes contain 29000 faceswaps using facefusion and 5000 downloads from thispersondoesnotexist.com

# Custom ResNet with Augmentation, MixUp, and TTA

The .ipynb file contains the model architecture. The .py file is a Web API UI using streamlit.

Link to storage: [Data and Model](https://drive.google.com/drive/folders/1lodTcVemGSLfRHavzTNpzgQmLKuYpyk9?usp=sharing)

## Features
- **Data Augmentation**: Includes various augmentations like Random Horizontal Flip, Random Rotation, and Color Jitter.
- **MixUp**: Applies the MixUp technique for better regularization and improved generalization during training.
- **Test-Time Augmentation (TTA)**: Uses a set of transformations at inference time to improve prediction accuracy.
- **Transfer Learning**: Fine-tunes a pre-trained ResNet-18 model on your custom dataset.
- **Dropout**: Prevents overfitting during training.
- **Learning Rate Scheduling**: Uses Cyclic Learning Rate for better convergence.

## Requirements

To run this code, you need the following Python libraries:

- `torch` (PyTorch)
- `torchvision`
- `PIL` (Python Imaging Library)
- `matplotlib`

You can install these dependencies using `pip`:

```bash
pip install torch torchvision matplotlib Pillow
```

## File Structure

```
.
├── README.md               # Documentation for the repository
├── train.py                # Training script
├── test.py                 # Testing and inference script
├── path/
│   └── to/
│       └── your/
│           └── data/      # Your dataset (place your images here)
└── model.py                # Custom ResNet-18 model definition
```

## How to Use

### 1. Dataset Preparation
Place your dataset in the appropriate folder. The dataset should be structured as follows:
```
data/
    train/
        class_1/
            image1.jpg
            image2.jpg
            ...
        class_2/
            image1.jpg
            image2.jpg
            ...
    val/
        class_1/
            image1.jpg
            image2.jpg
            ...
        class_2/
            image1.jpg
            image2.jpg
            ...
    test/
        class_1/
            image1.jpg
            image2.jpg
            ...
        class_2/
            image1.jpg
            image2.jpg
            ...
```
Make sure that the dataset is split into `train`, `val`, and `test` directories, each containing subdirectories for each class.

### 2. Training the Model

Run the training script (`train.py`) to train the model on your custom dataset:

```bash
python train.py
```

During training, the following will happen:
- The model will be trained using a combination of augmentations, MixUp, and dropout.
- The model's performance will be evaluated on the validation set after every epoch.
- The learning rate will be adjusted according to the Cyclic Learning Rate schedule.

### 3. Testing the Model

After training, you can test the model's performance on the test dataset:

```bash
python test.py
```

The test accuracy will be printed at the end of the evaluation.

### 4. Inference on Custom Image

To perform inference on a single image, make sure to modify the `image_path` in the inference code and run the following:

```bash
python inference.py
```

The predicted class will be displayed along with the image.

### 5. Saving and Loading the Model

The trained model will be saved to a file (adjust the path in the code):

```python
torch.save(model.state_dict(), "path/to/your/model")
```

You can load the saved model with:

```python
model.load_state_dict(torch.load("path/to/your/model", map_location=device))
```

## Code Details

- **Data Augmentation**: The `train_transform` applies multiple augmentations such as horizontal flipping, rotation, and color jitter. The validation and test sets do not apply augmentations, only resizing and normalization.
- **MixUp**: During training, the `MixUp` technique is applied using the `torchvision.transforms.v2.MixUp` class with an alpha of 0.2.
- **Test-Time Augmentation (TTA)**: At inference time, the `tta_predict` function averages the results over several augmentations to get more robust predictions.

## Model Architecture

The model is based on the ResNet-18 architecture. The final fully connected layer has been replaced with a custom classifier that includes a dropout layer for regularization.

```python
class CustomResNet(nn.Module):
    def __init__(self, base_model):
        super(CustomResNet, self).__init__()
        self.base = nn.Sequential(*list(base_model.children())[:-1])  # Remove final FC layer
        self.dropout = nn.Dropout(0.5)  # 50% Dropout to prevent overfitting
        self.fc = nn.Linear(base_model.fc.in_features, 2)  # Change output size to number of classes

    def forward(self, x):
        x = self.base(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)
```

## Hyperparameters

- **Batch Size**: 32
- **Optimizer**: Adam with weight decay of `1e-4`
- **Learning Rate**: Cyclic Learning Rate ranging from `1e-5` to `1e-2`
- **Loss Function**: CrossEntropyLoss with label smoothing (0.1)
- **Dropout**: 50%

## Acknowledgments

This repository uses the following libraries:
- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/vision/stable/index.html)
- [Pillow](https://pillow.readthedocs.io/en/stable/)
- [Matplotlib](https://matplotlib.org/)

## License

This code is provided under the MIT License. See the LICENSE file for more details.
