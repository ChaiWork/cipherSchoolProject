
# Built with love with the help of CipherSchools

# License Plate Detection with YOLO

This project implements a license plate detection system using YOLO (specifically YOLO26m model) from Ultralytics. The system is designed for training and validating a custom object detection model for license plate recognition.

## Project Structure

- `train.py` - Script for training the YOLO model
- `validate.py` - Script for validating the trained model
- `READ.md` - This documentation file

## Requirements

- Python 3.7+
- PyTorch
- Ultralytics YOLO
- CUDA-capable GPU (recommended)

## Installation

1. Clone this repository
2. Install required packages:

```bash
pip install ultralytics torch
```

## Training

To train the model, run:

```bash
python train.py
```

### Training Configuration

The training script uses the following configuration:

- **Model**: YOLO26m (pre-trained on COCO dataset)
- **Epochs**: 200
- **Image Size**: 640x640 pixels
- **Batch Size**: 12 (optimized for RTX 4030)
- **Device**: GPU (device=0)
- **Optimizer**: AdamW
- **Learning Rate**: 0.001
- **Weight Decay**: 0.01
- **Warmup Epochs**: 3
- **Patience**: 40 epochs
- **Workers**: 4
- **Cache**: Disk (for faster and stable training)

### Data Augmentation

The training includes light realism augmentations:
- **Mosaic**: 0.5 probability
- **Rotation**: ±2 degrees
- **Translation**: 0.05
- **Scale**: 0.25
- **Close Mosaic**: Enabled for last 5 epochs

## Validation

To validate the trained model, run:

```bash
python validate.py
```

### Validation Configuration

- **Batch Size**: 1
- **Image Size**: 640x640 pixels
- **Device**: GPU (device=0)
- **Workers**: 0 (for Windows compatibility)

### Metrics

The validation script outputs the following metrics:
- Precision
- Recall
- mAP50 (mean Average Precision at 50% IoU)
- mAP50-95 (mean Average Precision from 50% to 95% IoU)

## Usage Notes

### Configuration Files

Before running the scripts, you need to:

1. **Update `train.py`**: Replace `"ur own config path"` with the path to your dataset configuration file (YAML format)
2. **Update `validate.py`**: 
   - Replace `r"load ur own model"` with the path to your trained model
   - Replace `r"load ur own config path"` with the path to your dataset configuration file

### Windows Compatibility

The validation script includes GPU cache clearing for Windows systems and uses `if __name__ == "__main__":` guard for proper execution.

### Project Structure

Trained models and results are saved in:
- `license_plate_detection/yolo26m_balanced/` (training outputs)
- Model checkpoints and metrics are saved automatically

## Model Architecture

The project uses YOLOv8 architecture with:
- **Backbone**: CSPDarknet
- **Neck**: PANet
- **Head**: YOLO detection head
- **Activation**: SiLU
- **Loss**: Complete IoU, Distribution Focal Loss

## Performance Considerations

- **Batch Size**: Set to 12 for RTX 4030 GPU memory constraints
- **Image Size**: 640 provides better balance than 576
- **Cache**: Disk caching improves training speed and stability
- **Mixed Precision**: Automatic Mixed Precision (AMP) enabled for faster training

## Customization

To customize the training for your specific use case:

1. **Dataset**: Prepare your dataset in YOLO format with annotations
2. **Configuration**: Create a YAML file with dataset paths and class names
3. **Hyperparameters**: Adjust learning rate, batch size, and augmentation in `train.py`
4. **Model**: Change the base model in `train.py` (line 4) for different YOLO variants

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in `train.py`
2. **Slow Training**: Ensure disk cache is working properly
3. **Poor Validation Results**: Check dataset quality and annotation consistency
4. **Windows Execution Issues**: Use the `if __name__ == "__main__":` guard as shown

### GPU Memory Management

The validation script includes `torch.cuda.empty_cache()` to clear GPU memory before validation, which is especially important for Windows systems.

## License

This project is for educational and research purposes. Please ensure you have the right to use any datasets and comply with local regulations regarding license plate detection.

## Acknowledgments

- Ultralytics for the YOLOv8 implementation
- PyTorch team for the deep learning framework
- COCO dataset for pre-trained weights

