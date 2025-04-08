import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import os
from prompt_templates import PROMPT_TEMPLATES, RESPONSE_TEMPLATES, ANALYSIS_TEMPLATES

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

def analyze_image(image):
    """Analyze image features for detailed responses"""
    img_array = np.array(image)
    
    # Color analysis
    r, g, b = np.mean(img_array, axis=(0,1))
    r_std, g_std, b_std = np.std(img_array, axis=(0,1))
    
    # Brightness analysis
    gray = np.mean(img_array, axis=2)
    brightness = np.mean(gray)
    contrast = np.std(gray)
    
    # Edge analysis (simple Sobel-like)
    edge_x = np.abs(np.diff(gray, axis=1, prepend=0))
    edge_y = np.abs(np.diff(gray, axis=0, prepend=0))
    edge_density = np.mean((edge_x + edge_y) > 0.1) * 100
    
    # Regional brightness
    h, w = gray.shape
    top = np.mean(gray[:h//2])
    bottom = np.mean(gray[h//2:])
    left = np.mean(gray[:, :w//2])
    right = np.mean(gray[:, w//2:])
    
    return {
        "color": ANALYSIS_TEMPLATES["color"].format(
            r=int(r), g=int(g), b=int(b),
            r_std=int(r_std), g_std=int(g_std), b_std=int(b_std)
        ),
        "brightness": ANALYSIS_TEMPLATES["brightness"].format(
            brightness=int(brightness), contrast=int(contrast)
        ),
        "edges": ANALYSIS_TEMPLATES["edges"].format(edge_density=int(edge_density)),
        "regions": ANALYSIS_TEMPLATES["regions"].format(
            top=int(top), bottom=int(bottom), left=int(left), right=int(right)
        )
    }

# Load CIFAR10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)

# Create directory for dataset
dataset_dir = 'enhanced_cifar10_dataset'
os.makedirs(dataset_dir, exist_ok=True)

# Classes for labeling
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Function to calculate image clarity score
def get_clarity_score(image):
    img_array = np.array(image)
    gray = np.mean(img_array, axis=2)
    return np.var(gray)

# Get best images for each class
num_samples_per_class = 1
class_images = {c: [] for c in range(len(classes))}

# Collect candidate images for each class
print("Selecting the clearest images for each class...")
for idx in range(len(trainset)):
    image, label = trainset[idx]
    image = Image.fromarray(np.uint8(image))
    clarity_score = get_clarity_score(image)
    
    class_images[label].append((clarity_score, image))
    
    # Keep only the best images for each class
    class_images[label] = sorted(class_images[label], key=lambda x: x[0], reverse=True)[:num_samples_per_class]

# Create figure for displaying all images
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()

# Process and display the best images
for class_idx, class_name in enumerate(classes):
    # Get the best image for this class
    best_score, original_image = class_images[class_idx][0]
    
    # Enhanced processing pipeline
    enhanced_image = enhance_image(original_image)
    small_image = enhanced_image.resize((64, 64), Image.Resampling.LANCZOS)
    final_image = enhance_image(small_image)  # Apply enhancement again after resizing
    
    # Analyze image features
    analysis = analyze_image(final_image)
    
    # Generate responses for each prompt type
    responses = {}
    for prompt_type, templates in PROMPT_TEMPLATES.items():
        responses[prompt_type] = []
        for prompt in templates:
            # Format prompt with class name
            formatted_prompt = prompt.format(class_name=class_name)
            
            # Generate response based on prompt type
            if prompt_type == "basic":
                response = RESPONSE_TEMPLATES["basic"].format(class_name=class_name)
            elif prompt_type == "detailed":
                response = RESPONSE_TEMPLATES["detailed"].format(
                    class_name=class_name,
                    features="clear and distinct features",
                    color_info=analysis["color"],
                    edge_info=analysis["edges"]
                )
            elif prompt_type == "analytical":
                response = RESPONSE_TEMPLATES["analytical"].format(
                    class_name=class_name,
                    analysis=f"{analysis['color']} {analysis['brightness']} {analysis['edges']} {analysis['regions']}"
                )
            elif prompt_type == "focused":
                response = RESPONSE_TEMPLATES["focused"].format(
                    class_name=class_name,
                    details=f"{analysis['color']} {analysis['brightness']} {analysis['edges']}"
                )
            
            responses[prompt_type].append((formatted_prompt, response))
    
    # Save enhanced image and responses
    save_path = os.path.join(dataset_dir, f'enhanced_cifar10_{class_name}.png')
    final_image.save(save_path)
    
    # Save responses
    response_path = os.path.join(dataset_dir, f'responses_{class_name}.txt')
    with open(response_path, 'w') as f:
        for prompt_type, prompt_responses in responses.items():
            f.write(f"\n=== {prompt_type.upper()} PROMPTS ===\n")
            for prompt, response in prompt_responses:
                f.write(f"\nPrompt: {prompt}\n")
                f.write(f"Response: {response}\n")
    
    # Display image
    axes[class_idx].imshow(final_image)
    axes[class_idx].axis('off')
    axes[class_idx].set_title(f'{class_name}')

plt.tight_layout()
plt.show()

print(f"Saved enhanced CIFAR10 images and responses to {dataset_dir}/")

# Display comparison of original vs enhanced for one example
example_class = 0  # Show first class
_, original = class_images[example_class][0]
enhanced = enhance_image(original)
small = enhanced.resize((64, 64), Image.Resampling.LANCZOS)
final = enhance_image(small)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.imshow(original)
ax1.axis('off')
ax1.set_title('Original Image')
ax2.imshow(final)
ax2.axis('off')
ax2.set_title(f'Enhanced Image\n({classes[example_class]})')
plt.tight_layout()
plt.show() 