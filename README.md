# Hand Sign Detection

A deep learning project for recognizing hand signs and gestures, including digits (0-9) and letters (A-Z), with an interactive web interface.

## Project Overview

This project uses convolutional neural networks (CNN) to detect and classify hand signs from images. It includes both training and inference capabilities with a user-friendly web application.

## Features

- **Hand Sign Recognition**: Detects and classifies hand gestures for:
  - Digits: 0-9
  - Letters: A-Z
  - Special character: _ (underscore)
  
- **Pre-processed Dataset**: Includes both raw and pre-processed gesture image data
- **Trained Models**: Multiple trained models available for inference
- **Web Interface**: Interactive Flask-based web application for real-time predictions
- **Training Pipeline**: Scripts for training and fine-tuning models

## Project Structure

```
HandSignDetection-main/
├── src/
│   ├── app.py                 # Flask web application
│   └── train_model.py         # Model training script
├── data/
│   ├── Gesture Image Data/    # Raw gesture images organized by label
│   └── Gesture Image Pre-Processed Data/  # Pre-processed images
├── model/
│   ├── asl_model.h5           # Trained ASL model (H5 format)
│   ├── asl_model.keras        # Trained ASL model (Keras format)
│   ├── best_model.h5          # Best performing model
│   └── label_mapping.json     # Label to class mapping
├── templates/
│   └── index.html             # Web interface
└── requirements.txt           # Project dependencies
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/HandSignDetection.git
   cd HandSignDetection-main
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Web Application

```bash
python src/app.py
```

Then open your browser and navigate to `http://localhost:5000` to access the web interface.

### Training a Model

To train the model with your data:

```bash
python src/train_model.py
```

## Model Information

- **Input**: Gesture images (pre-processed for optimal performance)
- **Output**: Classification for digits (0-9), letters (A-Z), and underscore (_)
- **Framework**: TensorFlow/Keras
- **Architecture**: Convolutional Neural Network (CNN)

## Dataset

The project includes two versions of the gesture dataset:

1. **Raw Images**: Original gesture photographs
2. **Pre-processed Images**: Optimized images for training (recommended for best results)

Both are organized in subdirectories by label (0-9, A-Z, _).

## Requirements

See `requirements.txt` for all dependencies. Main requirements include:

- Python 3.7+
- TensorFlow/Keras
- Flask
- NumPy
- Pillow (PIL)
- OpenCV (cv2)

## Model Files

- `asl_model.h5` / `asl_model.keras`: Trained model in different formats
- `best_model.h5`: Best performing model during training
- `label_mapping.json`: Maps numeric labels to hand sign classes

## Contributing

Feel free to fork this repository and submit pull requests for improvements.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Inspired by American Sign Language (ASL) recognition projects
- Built with TensorFlow and Keras
- Web interface powered by Flask

## Contact

For questions or suggestions, please open an issue on the GitHub repository.
