# Install required packages (uncomment these lines when running in Colab)
# !pip uninstall -y transformers
# !pip install torch torchvision pillow
# !pip install git+https://github.com/huggingface/transformers.git

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForImageTextToText
from torchvision.transforms import functional as F

def enhance_image(image):
    """Apply a series of enhancements to make the image clearer"""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.uint8(image))
    
    image = image.convert('RGB')
    
    # Apply sharpening
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.5)
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2)
    
    # Enhance color
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(1.1)
    
    # Enhance brightness
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.1)
    
    return image

# Define transforms
transform = transforms.Compose([
    transforms.Resize((64, 64), antialias=True),
    transforms.Resize((384, 384), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load CIFAR10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)

# Get multiple images and select the clearest one
num_candidates = 10
best_image = None
best_clarity_score = -float('inf')

for i in range(num_candidates):
    idx = np.random.randint(len(trainset))
    image, label = trainset[idx]
    image = Image.fromarray(np.uint8(image))
    img_array = np.array(image)
    gray = np.mean(img_array, axis=2)
    clarity_score = np.var(gray)
    
    if clarity_score > best_clarity_score:
        best_clarity_score = clarity_score
        best_image = image
        best_label = label

image = best_image
label = best_label
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Enhanced image processing
enhanced_image = enhance_image(image)
small_image = enhanced_image.resize((64, 64), Image.Resampling.LANCZOS)
small_image = enhance_image(small_image)

# Display images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.imshow(image)
ax1.axis('off')
ax1.set_title('Original Image')
ax2.imshow(small_image)
ax2.axis('off')
ax2.set_title(f'Enhanced Small Image\nTrue Class: {classes[label]}')
plt.tight_layout()
plt.show()

# Load SmolVLM2 model and tokenizer
print("Loading SmolVLM2 model...")
model_id = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id)

def get_response(image, question):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # More specific and direct prompt
    prompt = f"""User: This is a small but clear image of a {classes[label]}. {question}
Assistant: Looking at this {classes[label]}, I can see the following specific details:"""
    
    img_tensor = transform(image)
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    
    try:
        inputs = {
            "input_ids": tokenizer.encode(prompt, return_tensors="pt"),
            "pixel_values": img_tensor
        }
        
        outputs = model.generate(
            **inputs,
            max_length=100,  # Shorter length to avoid repetition
            num_beams=5,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            no_repeat_ngram_size=3  # Prevent repetition
        )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        print(f"Image tensor shape: {img_tensor.shape}")
        raise e

# More focused questions
questions = [
    "What is the exact color and appearance of this {classes[label]}?",
    "What specific pose or position is this {classes[label]} in?",
    "What distinguishing features can you see clearly?",
    "How does this {classes[label]} stand out from its background?",
    "What makes this particular {classes[label]} unique or interesting?"
]

# Get responses for each question
print(f"\nAnalyzing the enhanced image (Class: {classes[label]})...")
print("=" * 50)
for question in questions:
    response = get_response(small_image, question)
    print(f"\nQ: {question}")
    print(f"A: {response}")
    print("-" * 50) 