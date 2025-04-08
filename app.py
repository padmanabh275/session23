import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPVisionModel,
    CLIPImageProcessor
)
from peft import PeftModel
import gradio as gr
from PIL import Image
import os
from prompt_templates import PROMPT_TEMPLATES, RESPONSE_TEMPLATES, ANALYSIS_TEMPLATES
import numpy as np
import cv2

class VLMInference:
    def __init__(self):
        # Initialize vision model
        self.vision_model = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Initialize language model
        self.language_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Find the most recent trained model
        model_dirs = [d for d in os.listdir(".") if d.startswith("best_vlm_model")]
        if model_dirs:
            # Sort directories by timestamp if available, otherwise by name
            def get_timestamp(d):
                try:
                    return int(d.split("_")[-1])
                except ValueError:
                    return 0  # For directories without timestamps
            
            latest_model = sorted(model_dirs, key=get_timestamp)[-1]
            model_path = latest_model
            print(f"Loading trained model from: {model_path}")
            
            # Load the trained LoRA weights
            self.language_model = PeftModel.from_pretrained(
                self.language_model,
                model_path
            )
            print("Successfully loaded trained LoRA weights")
        else:
            print("No trained model found. Using base model.")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Projection layer
        self.projection = torch.nn.Linear(
            self.vision_model.config.hidden_size,
            self.language_model.config.hidden_size
        ).half()
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vision_model.to(self.device)
        self.language_model.to(self.device)
        self.projection.to(self.device)
        
    def analyze_image(self, image):
        # Convert PIL Image to numpy array
        if isinstance(image, torch.Tensor):
            # If image is a tensor, move to CPU and convert to numpy
            img_np = image.cpu().numpy().transpose(1, 2, 0)
            # Denormalize if needed (assuming image was normalized)
            img_np = (img_np * 255).astype(np.uint8)
        else:
            # If image is PIL Image, convert directly
            img_np = np.array(image)
        
        # Basic color statistics
        r, g, b = np.mean(img_np, axis=(0,1))
        r_std, g_std, b_std = np.std(img_np, axis=(0,1))
        
        # Convert to grayscale for brightness and edge analysis
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Edge detection
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1]) * 100
        
        # Regional brightness analysis
        h, w = gray.shape
        top = np.mean(gray[:h//2, :])
        bottom = np.mean(gray[h//2:, :])
        left = np.mean(gray[:, :w//2])
        right = np.mean(gray[:, w//2:])
        
        return {
            "color": ANALYSIS_TEMPLATES["color"].format(
                r=int(r), g=int(g), b=int(b),
                r_std=int(r_std), g_std=int(g_std), b_std=int(b_std)
            ),
            "brightness": ANALYSIS_TEMPLATES["brightness"].format(
                brightness=int(brightness),
                contrast=int(contrast)
            ),
            "edges": ANALYSIS_TEMPLATES["edges"].format(
                edge_density=int(edge_density)
            ),
            "regions": ANALYSIS_TEMPLATES["regions"].format(
                top=int(top), bottom=int(bottom),
                left=int(left), right=int(right)
            )
        }
        
    def process_image(self, image):
        # Process image
        image = self.image_processor(image, return_tensors="pt")["pixel_values"][0].to(self.device)
        
        # Get vision features
        with torch.no_grad():
            vision_outputs = self.vision_model(image.unsqueeze(0))
            vision_features = vision_outputs.last_hidden_state.mean(dim=1)
            vision_features = self.projection(vision_features)
        
        return vision_features
    
    def generate_response(self, image, prompt_type, custom_prompt=None):
        # Process image
        image = self.image_processor(image, return_tensors="pt")["pixel_values"][0].to(self.device)
        
        # Get vision features
        with torch.no_grad():
            vision_outputs = self.vision_model(image.unsqueeze(0))
            vision_features = vision_outputs.last_hidden_state.mean(dim=1)
            vision_features = self.projection(vision_features)
        
        # Analyze image
        analysis = self.analyze_image(image)
        
        # Format prompt based on type
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = np.random.choice(PROMPT_TEMPLATES[prompt_type])
        
        # Format full prompt with analysis
        full_prompt = f"### Instruction: {prompt}\n\nImage Analysis:\n"
        for key, value in analysis.items():
            full_prompt += f"{value}\n"
        full_prompt += "\n### Response:"
        
        # Tokenize
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            # Generate using the base model
            outputs = self.language_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=512,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("### Response:")[1].strip()
        
        return response, analysis

def create_interface():
    # Initialize model
    model = VLMInference()
    
    def process_image_and_prompt(image, prompt_type, custom_prompt):
        try:
            response, analysis = model.generate_response(image, prompt_type, custom_prompt)
            
            # Format the output
            output = f"Response:\n{response}\n\nImage Analysis:\n"
            for key, value in analysis.items():
                output += f"{value}\n"
            
            return output
        except Exception as e:
            return f"Error: {str(e)}"
    
    # Load sample images from enhanced CIFAR10 dataset
    sample_images = []
    sample_labels = []
    dataset_dir = "enhanced_cifar10_dataset"
    
    if os.path.exists(dataset_dir):
        for filename in os.listdir(dataset_dir):
            if filename.startswith("enhanced_cifar10_") and filename.endswith(".png"):
                class_name = filename.replace("enhanced_cifar10_", "").replace(".png", "")
                image_path = os.path.join(dataset_dir, filename)
                try:
                    # Load and verify the image
                    img = Image.open(image_path)
                    img.verify()  # Verify it's a valid image
                    sample_images.append(image_path)
                    sample_labels.append(class_name)
                except Exception as e:
                    print(f"Error loading image {image_path}: {str(e)}")
    
    # Create Gradio interface
    with gr.Blocks(title="Vision-Language Model Demo") as interface:
        gr.Markdown("# Vision-Language Model Demo")
        gr.Markdown("Select a sample image from the enhanced CIFAR10 dataset or upload your own image.")
        
        with gr.Row():
            with gr.Column():
                # Sample images gallery
                if sample_images:
                    gr.Markdown("### Sample Images from Enhanced CIFAR10 Dataset")
                    sample_gallery = gr.Gallery(
                        value=[(img, label) for img, label in zip(sample_images, sample_labels)],
                        label="Select a sample image",
                        columns=5,
                        height="auto",
                        object_fit="contain"
                    )
                else:
                    gr.Markdown("No sample images found in the enhanced CIFAR10 dataset.")
                
                # Image input
                image_input = gr.Image(type="pil", label="Upload Image")
                
                # Prompt selection
                prompt_type = gr.Dropdown(
                    choices=list(PROMPT_TEMPLATES.keys()),
                    value="basic",
                    label="Select Prompt Type"
                )
                custom_prompt = gr.Textbox(
                    label="Custom Prompt (optional)",
                    placeholder="Enter your own prompt here..."
                )
                submit_btn = gr.Button("Generate Response")
            
            with gr.Column():
                output_text = gr.Textbox(
                    label="Model Response and Analysis",
                    lines=15
                )
        
        # Add click event for sample gallery
        if sample_images:
            def load_selected_image(evt: gr.SelectData):
                if evt.index < len(sample_images):
                    return Image.open(sample_images[evt.index])
                return None
            
            sample_gallery.select(
                fn=load_selected_image,
                inputs=[],
                outputs=[image_input]
            )
        
        submit_btn.click(
            fn=process_image_and_prompt,
            inputs=[image_input, prompt_type, custom_prompt],
            outputs=[output_text]
        )
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True) 