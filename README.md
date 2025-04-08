# Vision-Language Model (VLM) with CIFAR10 Dataset

This project implements a Vision-Language Model (VLM) that combines image understanding with natural language processing. The model is trained on an enhanced CIFAR10 dataset and uses a combination of CLIP for vision and Phi-2 for language processing.

## Live Demo

Try the model live on Hugging Face Spaces:
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/padmanabhbosamia/Vision_with_Cifar)

## Project Structure

The project consists of several key components:

1. **Dataset Collection and Enhancement**
   - `cifar10_dataset_collector.py`: Creates enhanced versions of CIFAR10 images
   - `enhanced_cifar10_dataset/`: Contains the processed images

2. **Model Training**
   - `train_siglip_phi3.py`: Trains SigLIP with frozen Phi-2 model
   - `train_vlm_sft_qlora.py`: Implements SFT with QLoRA for VLM training
   - `losses.py`: Contains custom loss functions for training

3. **Model Testing and Inference**
   - `app.py`: Gradio interface for model inference
   - `test_vlm_qlora.py`: Scripts for testing the trained model
   - `test_siglip.py`: Scripts for testing the SigLIP model

4. **Prompt Templates**
   - `prompt_templates.py`: Contains various prompt templates for different types of queries

## Tasks Completed

1. **Task 1 & 2: Dataset Creation**
   - Created enhanced CIFAR10 images with various transformations
   - Generated multiple samples per class
   - Implemented image analysis features

2. **Task 3: SigLIP Model Training**
   - Trained SigLIP with frozen Phi-2 model
   - Implemented contrastive learning
   - Saved best model checkpoints

3. **Task 4: VLM Training with SFT and QLoRA**
   - Implemented Supervised Fine-Tuning (SFT)
   - Used QLoRA for efficient training
   - Trained model from scratch on enhanced dataset

4. **Task 5: Deployment Interface**
   - Created Gradio web interface
   - Integrated sample images from CIFAR10
   - Implemented multiple prompt types
   - Added image analysis features

## Features

- **Image Analysis**
  - Color statistics
  - Brightness and contrast analysis
  - Edge detection
  - Regional brightness analysis

- **Prompt Types**
  - Basic prompts
  - Analytical prompts
  - Technical prompts
  - Comparative prompts
  - Focused prompts

- **Model Integration**
  - CLIP for vision processing
  - Phi-2 for language processing
  - LoRA adapters for efficient fine-tuning

## Usage

1. **Running the Interface**
   ```bash
   python app.py
   ```

2. **Training the Model**
   ```bash
   python train_vlm_sft_qlora.py
   ```

3. **Testing the Model**
   ```bash
   python test_vlm_qlora.py
   ```

## Requirements

Install the required packages:
```bash
pip install -r requirements.txt
```

## Directory Structure

```
.
├── app.py                  # Gradio interface
├── train_vlm_sft_qlora.py  # SFT training script
├── train_siglip_phi3.py    # SigLIP training script
├── cifar10_dataset_collector.py  # Dataset creation
├── prompt_templates.py     # Prompt templates
├── losses.py              # Loss functions
├── enhanced_cifar10_dataset/  # Processed images
├── best_vlm_model_*/      # Trained model checkpoints
└── requirements.txt       # Dependencies
```

## Future Improvements

1. Enhanced vision-language integration
2. Additional prompt templates
3. Improved image analysis features
4. Deployment to Hugging Face Spaces
5. Support for more image datasets

## Contributing

Feel free to submit issues and enhancement requests!

## License

MIT

## Acknowledgments

- CIFAR10 dataset
- HuggingFace Transformers
- PHI-2 model
- Gradio for the web interface 