
# Sign Language Recognition Using Deep Learning Models on WLASL Dataset

## Project Overview

This project aims to build a model to recognize American Sign Language (ASL) gestures from video datasets. The focus is on the **WLASL dataset**, which contains over 12,000 videos representing 2,000 classes (signs). Due to class imbalance and memory constraints, the dataset was reduced to the **100 most frequent words**, comprising approximately **1,200 videos**. The project explores multiple deep learning models for spatiotemporal gesture recognition.

## Dataset Description

- **Dataset**: [WLASL Dataset](https://dx.doi.org/10.21227/ps9m-0w52)
  - 12,000 labeled videos of ASL gestures
  - 2,000 different words (glosses)
  - Videos are 1-3 seconds long, 25 frames per second
- **Preprocessed Dataset**: Reduced to 100 most frequent words (1,200 videos) to address class imbalance.

## Preprocessing Methods

Three distinct preprocessing methods were applied to address challenges such as varying video lengths, class imbalance, and the need for spatial and temporal consistency:

1. **Pose Keypoint Extraction**:
   - **Tool**: MediaPipe
   - **Keypoints**: 33 keypoints per frame (e.g., shoulders, elbows, wrists)
   - **Normalization**: To account for different resolutions
   - **Truncation/Padding**: Videos fixed to 90 frames

2. **Holistic Keypoint Extraction**:
   - **Dataset**: Top 100 words
   - **Approach**: Pose, hands, and face keypoints
   - **Variants**:
     - Pose only
     - Pose + Hands (best performing)
     - Pose + Hands + Face

3. **Full-Frame Extraction for I3D**:
   - **Frame Standardization**: 64 frames per video
   - **Resolution**: 224x224 pixels
   - **Normalization**: For consistency with I3D input requirements

## Models Explored

1. **Default CNN**: Baseline for spatial feature extraction.
2. **Default LSTM**: Baseline for temporal modeling.
3. **CNN-LSTM Hybrid**: Combines spatial and temporal features.
4. **Transformer-Based Models**:
   - **PoseFormer** and **TimeSFormer** for advanced feature capture.
   - Best results achieved with Pose + Hand keypoints.
5. **I3D Model**: For spatiotemporal patterns (limited by memory constraints).

## Results and Findings

- **Initial Results**:
  - Using 2,000 classes resulted in accuracy < 1%.
  - Reduced to 100 classes for balanced evaluation.
- **Top Model**: Transformer achieved **~9.82% accuracy**.
- **Best Dataset Configuration**: Pose + Hand keypoints.
- **I3D Model**: Demonstrated potential but faced GPU memory constraints.

## Challenges

1. **Memory Constraints**:
   - Limited by Google Colab's A100 GPU (40GB RAM).
   - **Solutions**: Batch processing, using AWS H100 GPUs.

2. **Class Imbalance**:
   - Even in the top 100 words, the most frequent word had only 16 videos.

3. **Complexity of Sign Language**:
   - Nuanced gestures require precise temporal and spatial modeling.

## Potential Applications

- **Interactive ASL Education Apps**: Real-time feedback for learners.
- **Communication Tools**: For deaf and hearing individuals.
- **Inclusive Classrooms and Workplaces**: Enhancing communication and understanding.

## Team Contributions

- **Aaron Ramirez**: Preprocessing, modeling, presentation, and draft paper.
- **Anitra Hernandez**: EDA, modeling, results graphs.
- **Devin Eror**: Validation, performance metrics, modeling, draft paper.

## Installation and Usage

### Prerequisites

- **Python 3.x**
- **Google Colab** (Recommended for GPU support)
- **Libraries**:
  ```bash
  pip install tensorflow torch mediapipe matplotlib opencv-python
  ```

### Clone the Repository

```bash
git clone https://github.com/LazyTriceratops/AAI-521-Group-6-Final-Team-Project.git
cd AAI-521-Group-6-Final-Team-Project
```

### Running the Code

1. **Preprocessing**:
   - Follow the preprocessing scripts in `preprocessing/`.

2. **Model Training**:
   - Each model can be trained using the scripts in `models/`.
   - Example:
     ```bash
     python train_transformer.py
     ```

3. **Evaluation**:
   - Evaluate models using the scripts in `evaluation/`.

### Project Structure

```
.
├── preprocessing/
│   ├── preprocess_pose.py
│   ├── preprocess_holistic.py
│   └── preprocess_i3d.py
├── models/
│   ├── train_cnn.py
│   ├── train_lstm.py
│   ├── train_transformer.py
│   └── train_cnn_lstm.py
├── evaluation/
│   └── evaluate_models.py
├── README.md
└── requirements.txt
```

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Acknowledgments

- **Dataset**: [WLASL Dataset](https://dx.doi.org/10.21227/ps9m-0w52)
- **Tools**: MediaPipe, TensorFlow, PyTorch
