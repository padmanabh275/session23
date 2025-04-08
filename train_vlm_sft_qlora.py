import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
from PIL import Image
import os
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPVisionModel,
    CLIPImageProcessor,
    BitsAndBytesConfig
)
from datasets import load_dataset
import bitsandbytes as bnb
from tqdm import tqdm
from prompt_templates import PROMPT_TEMPLATES, RESPONSE_TEMPLATES  # Import prompt templates

class VLMQLoRADataset(Dataset):
    def __init__(self, dataset_dir, tokenizer, image_processor, max_length=512):
        self.dataset_dir = dataset_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        
        # Load image files and responses
        self.image_files = [f for f in os.listdir(dataset_dir) if f.startswith('enhanced_cifar10_') and f.endswith('.png')]
        self.response_files = [f for f in os.listdir(dataset_dir) if f.startswith('responses_') and f.endswith('.txt')]
        
        # Load all responses
        self.responses = {}
        for response_file in self.response_files:
            class_name = response_file.replace('responses_', '').replace('.txt', '')
            with open(os.path.join(dataset_dir, response_file), 'r') as f:
                self.responses[class_name] = f.read()
    
    def __len__(self):
        return len(self.image_files) * len(PROMPT_TEMPLATES["basic"])  # Multiply by number of prompt templates
    
    def __getitem__(self, idx):
        # Calculate which image and prompt template to use
        img_idx = idx // len(PROMPT_TEMPLATES["basic"])
        prompt_idx = idx % len(PROMPT_TEMPLATES["basic"])
        
        # Get image
        image_file = self.image_files[img_idx]
        class_name = image_file.replace('enhanced_cifar10_', '').replace('.png', '')
        image_path = os.path.join(self.dataset_dir, image_file)
        image = Image.open(image_path).convert('RGB')
        
        # Process image
        image = self.image_processor(image, return_tensors="pt")["pixel_values"][0]
        
        # Get responses for this class
        class_responses = self.responses[class_name]
        
        # Get prompt template and format it
        prompt_template = PROMPT_TEMPLATES["basic"][prompt_idx]
        instruction = prompt_template.format(class_name=class_name)
        
        # Get response template and format it
        response_template = RESPONSE_TEMPLATES["detailed"]
        response = response_template.format(
            class_name=class_name,
            features="distinctive visual features",
            color_info="typical color distribution",
            edge_info="characteristic edge patterns"
        )
        
        # Format full text
        full_text = f"### Instruction: {instruction}\n### Response: {response}"
        
        # Tokenize
        tokenized = self.tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": tokenized["input_ids"][0],
            "attention_mask": tokenized["attention_mask"][0],
            "pixel_values": image
        }

class VLMQLoRAModel(nn.Module):
    def __init__(self, vision_model_name="openai/clip-vit-base-patch32", 
                 language_model_name="microsoft/phi-2"):
        super().__init__()
        
        # Initialize vision model with gradient checkpointing
        self.vision_model = CLIPVisionModel.from_pretrained(
            vision_model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.vision_model.gradient_checkpointing_enable()
        self.vision_model.train()  # Ensure training mode
        for param in self.vision_model.parameters():
            if param.dtype.is_floating_point:
                param.requires_grad = True
        self.image_processor = CLIPImageProcessor.from_pretrained(vision_model_name)
        
        # Initialize language model with 4-bit quantization and memory efficient settings
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        self.language_model = AutoModelForCausalLM.from_pretrained(
            language_model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        self.language_model.gradient_checkpointing_enable()
        self.language_model.train()  # Ensure training mode
        for param in self.language_model.parameters():
            if param.dtype.is_floating_point:
                param.requires_grad = True
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Projection layer
        self.projection = nn.Linear(
            self.vision_model.config.hidden_size,
            self.language_model.config.hidden_size
        ).half()  # Use half precision for projection
        self.projection.train()  # Ensure training mode
        for param in self.projection.parameters():
            if param.dtype.is_floating_point:
                param.requires_grad = True
        
        # Initialize LoRA with memory efficient settings
        lora_config = LoraConfig(
            r=8,  # Reduced rank
            lora_alpha=16,  # Reduced alpha
            target_modules=["q_proj", "k_proj", "v_proj", "dense"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.language_model = get_peft_model(self.language_model, lora_config)
        
    def forward(self, pixel_values, input_ids, attention_mask):
        # Get vision features
        vision_outputs = self.vision_model(pixel_values)
        vision_features = vision_outputs.last_hidden_state.mean(dim=1)
        vision_features = self.projection(vision_features)
        
        # Get language model outputs
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
            vision_features=vision_features
        )
        
        return outputs.loss

def train_vlm_sft_qlora():
    # Set device and memory settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    print(f"Using device: {device}")
    
    # Initialize model
    model = VLMQLoRAModel().to(device)
    model.train()  # Ensure model is in training mode
    
    # Initialize optimizer with memory efficient settings
    optimizer = bnb.optim.AdamW8bit(
        model.parameters(), 
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler()
    
    # Create dataset and dataloader with reduced batch size
    dataset = VLMQLoRADataset(
        dataset_dir='enhanced_cifar10_dataset',
        tokenizer=model.tokenizer,
        image_processor=model.image_processor
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=2,  # Reduced batch size
        shuffle=True, 
        num_workers=2,  # Reduced workers
        pin_memory=True
    )
    
    # Initialize best loss tracking
    best_loss = float('inf')
    best_model_path = 'best_vlm_model'
    
    # Training loop with gradient accumulation
    num_epochs = 10
    accumulation_steps = 2  # Gradient accumulation steps
    
    for epoch in range(num_epochs):
        model.train()  # Ensure model is in training mode
        total_loss = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for i, batch in enumerate(progress_bar):
            # Move data to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            
            # Forward pass with mixed precision
            with autocast():
                loss = model(pixel_values, input_ids, attention_mask)
                if not torch.isnan(loss) and not torch.isinf(loss):
                    loss = loss / accumulation_steps  # Scale loss for gradient accumulation
                else:
                    print("Warning: Loss is NaN or Inf, skipping this batch")
                    continue
            
            # Backward pass with gradient scaling
            if loss.requires_grad:  # Check if loss requires gradient
                scaler.scale(loss).backward()
                
                # Step optimizer only after accumulation steps
                if (i + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
            
            total_loss += loss.item() * accumulation_steps
            progress_bar.set_postfix({"loss": loss.item() * accumulation_steps})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
        
        # Save best checkpoint if current loss is better
        if avg_loss < best_loss:
            import time
            import os
            import shutil
            
            # Create a temporary directory for the new model
            timestamp = int(time.time())
            temp_path = f"{best_model_path}_temp_{timestamp}"
            new_path = f"{best_model_path}_{timestamp}"
            
            try:
                # Save the new model to temporary directory
                model.language_model.save_pretrained(temp_path)
                
                # If there's an existing best model, try to remove it
                if os.path.exists(best_model_path):
                    shutil.rmtree(best_model_path, ignore_errors=True)
                
                # Rename the temporary directory to the final name
                if os.path.exists(temp_path):
                    if os.path.exists(best_model_path):
                        # If we couldn't remove the old directory, use the timestamped name
                        os.rename(temp_path, new_path)
                        print(f"Warning: Could not remove old model directory. Saved new model as: {new_path}")
                    else:
                        os.rename(temp_path, best_model_path)
                
                best_loss = avg_loss
                print(f"New best model saved with loss: {best_loss:.4f}")
                
            except Exception as e:
                print(f"Error saving model: {e}")
                # Clean up temporary directory if it exists
                if os.path.exists(temp_path):
                    shutil.rmtree(temp_path, ignore_errors=True)
        
        # Clear memory after each epoch
        torch.cuda.empty_cache()
    
    print(f"Training completed! Best model saved with loss {best_loss:.4f}")

if __name__ == "__main__":
    train_vlm_sft_qlora() 