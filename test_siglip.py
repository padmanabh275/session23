import torch
from PIL import Image
import torchvision.transforms as transforms
from train_siglip_phi3 import SigLIPModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model():
    # Load PHI-3 model
    phi_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-3")
    
    # Create and load trained SigLIP model
    model = SigLIPModel(phi_model)
    model.load_state_dict(torch.load('siglip_final_model.pt'))
    model.eval()
    
    return model

def encode_image(model, image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        image_embedding = model.encode_image(image_tensor)
    
    return image_embedding

def test_model():
    model = load_model()
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3")
    
    # Test images
    test_images = [
        "enhanced_cifar10_dataset/enhanced_cifar10_plane.png",
        "enhanced_cifar10_dataset/enhanced_cifar10_car.png",
        "enhanced_cifar10_dataset/enhanced_cifar10_bird.png"
    ]
    
    # Test descriptions
    descriptions = [
        "A clear image of an airplane.",
        "A detailed photograph of a car.",
        "A sharp image of a bird."
    ]
    
    # Encode all images
    image_embeddings = torch.cat([encode_image(model, img_path) for img_path in test_images])
    
    # Encode all texts
    text_inputs = tokenizer(descriptions, padding=True, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = model.encode_text(model.phi_model(text_inputs.input_ids).last_hidden_state)
    
    # Calculate similarity matrix
    similarity = torch.matmul(image_embeddings, text_embeddings.t())
    
    # Print results
    print("\nSimilarity Matrix:")
    print("================")
    for i, img_path in enumerate(test_images):
        print(f"\nImage: {img_path}")
        for j, desc in enumerate(descriptions):
            print(f"Similarity with '{desc}': {similarity[i][j]:.4f}")

if __name__ == "__main__":
    test_model() 