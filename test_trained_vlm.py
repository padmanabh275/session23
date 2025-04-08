import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torchvision.transforms as transforms

def load_trained_model():
    model = AutoModelForCausalLM.from_pretrained(
        "./final_vlm_model",
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("./final_vlm_model")
    return model, tokenizer

def generate_description(model, tokenizer, image_path):
    # Prepare image
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    # Prepare prompt
    prompt = """<image>
User: Describe what you see in this image.
Assistant:"""
    
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
    # Load model and tokenizer
    model, tokenizer = load_trained_model()
    
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