import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
from PIL import Image
import os
import json
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from losses import ContrastiveLoss, InfoNCELoss
from prompt_templates import PROMPT_TEMPLATES, RESPONSE_TEMPLATES

# Set memory management configurations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'

class EnhancedCIFAR10Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.image_files = [f for f in os.listdir(root_dir) if f.startswith('enhanced_cifar10_') and f.endswith('.png')]
        self.response_files = [f for f in os.listdir(root_dir) if f.startswith('responses_') and f.endswith('.txt')]
        
        # Load all responses
        self.responses = {}
        for response_file in self.response_files:
            class_name = response_file.replace('responses_', '').replace('.txt', '')
            with open(os.path.join(root_dir, response_file), 'r') as f:
                self.responses[class_name] = f.read()
    
    def __len__(self):
        return len(self.image_files) * len(PROMPT_TEMPLATES)
    
    def __getitem__(self, idx):
        # Calculate which image and prompt type to use
        image_idx = idx // len(PROMPT_TEMPLATES)
        prompt_type_idx = idx % len(PROMPT_TEMPLATES)
        
        # Get image
        image_file = self.image_files[image_idx]
        class_name = image_file.replace('enhanced_cifar10_', '').replace('.png', '')
        image_path = os.path.join(self.root_dir, image_file)
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get responses for this class
        class_responses = self.responses[class_name]
        
        # Extract prompt and response for this prompt type
        prompt_type = list(PROMPT_TEMPLATES.keys())[prompt_type_idx]
        prompt = PROMPT_TEMPLATES[prompt_type][0].format(class_name=class_name)
        
        # Find the corresponding response in the text file
        response = ""
        lines = class_responses.split('\n')
        for i, line in enumerate(lines):
            if f"=== {prompt_type.upper()} PROMPTS ===" in line:
                for j in range(i+1, len(lines)):
                    if "Prompt:" in lines[j] and prompt in lines[j]:
                        response = lines[j+1].replace("Response: ", "").strip()
                        break
                break
        
        return image, prompt, response, class_name

class SigLIPModel(nn.Module):
    def __init__(self, image_encoder_name="google/siglip-base-patch16-224", text_encoder_name="microsoft/phi-2"):
        super().__init__()
        # Initialize SigLIP model
        self.siglip = AutoModel.from_pretrained(image_encoder_name)
        self.siglip_tokenizer = AutoTokenizer.from_pretrained(image_encoder_name)
        
        # Initialize Phi-2 for text
        self.phi2 = AutoModelForCausalLM.from_pretrained(
            text_encoder_name,
            output_hidden_states=True,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        self.phi2_tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        
        # Fix the padding token issue for Phi-2
        self.phi2_tokenizer.pad_token = self.phi2_tokenizer.eos_token
        self.phi2.config.pad_token_id = self.phi2_tokenizer.eos_token_id
        
        # Projection heads
        self.image_projection = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Linear(512, 256)
        )
        self.text_projection = nn.Sequential(
            nn.Linear(2560, 512),  # Updated to match Phi-2's hidden size
            nn.GELU(),
            nn.Linear(512, 256)
        )
        
    def forward(self, images, prompts):
        # Image encoding with SigLIP
        # Ensure images are in the correct format for SigLIP (B, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension if missing
        if images.shape[1] != 3:  # If channels last, convert to channels first
            images = images.permute(0, 3, 1, 2)
        
        # Get image features from SigLIP
        # Create dummy text inputs for SigLIP
        dummy_text = ["a photo"] * len(images)
        siglip_text_inputs = self.siglip_tokenizer(
            dummy_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=77,
            return_attention_mask=True
        ).to(images.device)
        
        # Get image features
        image_outputs = self.siglip(
            pixel_values=images,
            input_ids=siglip_text_inputs["input_ids"],
            attention_mask=siglip_text_inputs.get("attention_mask", None)
        )
        
        # Extract image embeddings from SigLIP output
        # SigLIP returns image_embeds directly
        image_embeddings = image_outputs.image_embeds
        image_embeddings = self.image_projection(image_embeddings)
        
        # Text encoding with Phi-2
        # Tokenize prompts and move to device
        text_inputs = self.phi2_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        text_inputs = {k: v.to(images.device) for k, v in text_inputs.items()}
        
        with torch.no_grad():
            text_outputs = self.phi2(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
                output_hidden_states=True
            )
            text_features = text_outputs.hidden_states[-1].mean(dim=1)
        text_embeddings = self.text_projection(text_features)
        
        return image_embeddings, text_embeddings

def train_siglip():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Enable memory efficient settings
    torch.backends.cudnn.benchmark = True
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Initialize model
    model = SigLIPModel().to(device)
    
    # Initialize losses
    contrastive_loss = ContrastiveLoss(temperature=0.07)
    infonce_loss = InfoNCELoss(temperature=0.07)
    
    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler()
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and dataloader
    dataset = EnhancedCIFAR10Dataset(root_dir='enhanced_cifar10_dataset', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    
    # Initialize best loss tracking
    best_loss = float('inf')
    best_epoch = 0
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (images, prompts, responses, class_names) in enumerate(dataloader):
            # Move data to device
            images = images.to(device)
            
            # Forward pass with mixed precision
            with autocast():
                image_embeddings, text_embeddings = model(images, prompts)
                
                # Calculate losses
                loss_contrastive = contrastive_loss(image_embeddings, text_embeddings)
                loss_infonce = infonce_loss(image_embeddings, text_embeddings)
                
                # Combine losses
                loss = loss_contrastive + loss_infonce
            
            # Backward pass with gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], "
                      f"Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
        
        # Save best checkpoint if current loss is better
        if avg_loss < best_loss:
            # Delete previous best model if it exists
            if os.path.exists('best_model.pt'):
                os.remove('best_model.pt')
            
            best_loss = avg_loss
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'best_model.pt')
            print(f"New best model saved with loss: {best_loss:.4f}")
    
    print(f"Training completed! Best model saved from epoch {best_epoch} with loss {best_loss:.4f}")

if __name__ == "__main__":
    train_siglip() 