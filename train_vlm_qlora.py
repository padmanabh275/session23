import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import os
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from prompt_templates import PROMPT_TEMPLATES, RESPONSE_TEMPLATES

class ImageCaptioningDataset(Dataset):
    def __init__(self, image_dir, tokenizer, max_length=512):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.prompt_templates = PROMPT_TEMPLATES["basic"]
        self.response_templates = RESPONSE_TEMPLATES
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_files) * len(self.prompt_templates)

    def __getitem__(self, idx):
        img_idx = idx // len(self.prompt_templates)
        prompt_idx = idx % len(self.prompt_templates)
        
        # Load and process image
        img_path = os.path.join(self.image_dir, self.image_files[img_idx])
        image = Image.open(img_path).convert('RGB')
        image_tensor = self.transform(image)
        
        # Get class name from filename
        class_name = os.path.splitext(self.image_files[img_idx])[0].split('_')[-1]
        
        # Format prompt and response
        prompt = self.prompt_templates[prompt_idx]
        response = self.response_templates["detailed"].format(
            class_name=class_name,
            features="distinctive visual features",
            color_info="typical color distribution",
            edge_info="characteristic edge patterns"
        )
        
        # Format full prompt
        full_prompt = f"""### Instruction: {prompt}
### Input: <image>
### Response: {response}"""
        
        # Tokenize
        encodings = self.tokenizer(
            full_prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "pixel_values": image_tensor,
            "labels": encodings["input_ids"].squeeze()
        }

def create_qlora_model():
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "dense"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    return model

def train():
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create model
    model = create_qlora_model()
    
    # Create dataset
    dataset = ImageCaptioningDataset(
        image_dir="enhanced_cifar10_dataset",
        tokenizer=tokenizer
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="vlm_qlora_output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda data: {
            "input_ids": torch.stack([f["input_ids"] for f in data]),
            "attention_mask": torch.stack([f["attention_mask"] for f in data]),
            "pixel_values": torch.stack([f["pixel_values"] for f in data]),
            "labels": torch.stack([f["labels"] for f in data])
        }
    )
    
    # Train
    trainer.train()
    
    # Save final model
    trainer.save_model("vlm_qlora_output/final")

if __name__ == "__main__":
    train() 