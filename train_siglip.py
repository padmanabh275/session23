import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoProcessor
from PIL import Image
import os
import torchvision.transforms as transforms
from peft import prepare_model_for_kbit_training
from prompt_templates import PROMPT_TEMPLATES, RESPONSE_TEMPLATES

class TinyCIFAR10Dataset(Dataset):
    def __init__(self, image_dir, processor):
        self.image_dir = image_dir
        self.processor = processor
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.prompt_templates = PROMPT_TEMPLATES["technical"]
        self.response_templates = RESPONSE_TEMPLATES
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.image_files) * len(self.prompt_templates)

    def __getitem__(self, idx):
        img_idx = idx // len(self.prompt_templates)
        prompt_idx = idx % len(self.prompt_templates)
        
        image_path = os.path.join(self.image_dir, self.image_files[img_idx])
        image = Image.open(image_path).convert('RGB')
            
        # Get class name from filename
        class_name = os.path.splitext(self.image_files[img_idx])[0].split('_')[-1]
        
        # Format prompt and response
        prompt = self.prompt_templates[prompt_idx]
        response = self.response_templates["analytical"].format(
            class_name=class_name,
            analysis="detailed visual analysis"
        )
        
        # Process image using SigLIP processor
        image_features = self.processor(images=image, return_tensors="pt")
        
        # Process text using Phi-2 tokenizer
        text_features = self.tokenizer(
            prompt,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "pixel_values": image_features.pixel_values.squeeze(),
            "input_ids": text_features["input_ids"].squeeze(),
            "attention_mask": text_features["attention_mask"].squeeze(),
            "labels": text_features["input_ids"].squeeze()
        }

class SigLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        self.text_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
        
    def forward(self, pixel_values, input_ids, attention_mask):
        vision_outputs = self.vision_model(pixel_values)
        text_outputs = self.text_model(input_ids, attention_mask=attention_mask)
        return vision_outputs, text_outputs

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SigLIP().to(device)
    
    # Load dataset with correct processor
    processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    dataset = TinyCIFAR10Dataset('enhanced_cifar10_dataset', processor)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            vision_outputs, text_outputs = model(pixel_values, input_ids, attention_mask)
            
            # Calculate loss
            loss = criterion(text_outputs.logits.view(-1, text_outputs.logits.size(-1)), 
                           input_ids.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"siglip_checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    train() 