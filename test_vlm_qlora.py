import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torchvision.transforms as transforms

def load_model():
    # Load the trained model
    model = AutoModelForCausalLM.from_pretrained("./vlm_qlora_output/final")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    return model, tokenizer

def generate_description(model, tokenizer, image_path):
    # Prepare image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    # Create prompt
    prompt = "### Instruction: Describe this image in detail.\n### Input: <image>\n### Response:"
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    # Generate response
    outputs = model.generate(
        input_ids=inputs["input_ids"].to(model.device),
        attention_mask=inputs["attention_mask"].to(model.device),
        pixel_values=image_tensor.to(model.device),
        max_length=200,
        num_beams=5,
        temperature=0.7,
        do_sample=True
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    model, tokenizer = load_model()
    
    # Test on a few images
    test_images = [
        "enhanced_cifar10_dataset/enhanced_cifar10_plane.png",
        "enhanced_cifar10_dataset/enhanced_cifar10_car.png",
        "enhanced_cifar10_dataset/enhanced_cifar10_bird.png"
    ]
    
    for image_path in test_images:
        print(f"\nGenerating description for {image_path}...")
        description = generate_description(model, tokenizer, image_path)
        print(f"Description: {description}")

if __name__ == "__main__":
    main() 