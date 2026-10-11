# Smart Plate ML

## Purpose
The `ml/` directory is dedicated to the machine learning components of the Smart Plate project. It contains datasets, training scripts, inference code, and notebooks for experimentation.

## Model Training Strategy
YOLO training will be performed externally on platforms like **Google Colab** or **Kaggle GPU**. This laptop will be used strictly for development, dataset preparation, and inference testing due to the lack of a dedicated NVIDIA GPU.

## Deployment
Once a YOLO model is trained successfully, the trained model weights (e.g., `.pt` files) will eventually be copied to the `backend/models/` directory for integration with the Flask API.

## Datasets
**IMPORTANT**: Datasets should **not** be committed to Git. All images and label files should reside within the `datasets/` folder, which should be excluded from version control.

## Final ML Pipeline
The final ML pipeline will follow this architecture:
1. **Image Upload**
2. **YOLO Food Detection**
3. **Portion/Volume Estimation**
4. **Nutrition Mapping**
