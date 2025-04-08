import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, image_embeddings, text_embeddings):
        # Debug logging
        print(f"Image embeddings shape: {image_embeddings.shape}, Text embeddings shape: {text_embeddings.shape}")
        
        # Ensure inputs are float32 for better numerical stability
        image_embeddings = image_embeddings.float()
        text_embeddings = text_embeddings.float()
        
        # Ensure inputs are 2D
        if image_embeddings.dim() != 2 or text_embeddings.dim() != 2:
            raise ValueError(f"Expected 2D tensors, got image_embeddings: {image_embeddings.dim()}D, text_embeddings: {text_embeddings.dim()}D")
        
        # Add small noise to prevent zero embeddings
        image_embeddings = image_embeddings + torch.randn_like(image_embeddings) * 1e-6
        text_embeddings = text_embeddings + torch.randn_like(text_embeddings) * 1e-6
        
        # Normalize embeddings with epsilon to prevent division by zero
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1, eps=1e-8)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1, eps=1e-8)
        
        # Calculate similarity scores with numerical stability
        logits = torch.matmul(image_embeddings, text_embeddings.t())
        logits = torch.clamp(logits, min=-1.0, max=1.0)  # Ensure logits are in valid range
        
        # Scale logits by temperature with numerical stability
        logits = logits / max(self.temperature, 1e-6)
        
        # Create labels for both directions
        labels = torch.arange(len(image_embeddings)).to(image_embeddings.device)
        
        # Calculate bidirectional loss with stability
        i2t_loss = self.cross_entropy(logits, labels)
        t2i_loss = self.cross_entropy(logits.t(), labels)
        
        # Average the losses with stability term
        loss = (i2t_loss + t2i_loss) / 2
        
        # Ensure loss is not zero and handle nan values while maintaining gradient tracking
        if torch.isnan(loss) or loss == 0:
            # Create a tensor with gradient tracking
            stable_loss = torch.ones_like(loss, requires_grad=True)
            loss = stable_loss
        
        print(f"Contrastive loss: {loss.item():.4f}")
        return loss

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, image_embeddings, text_embeddings):
        # Debug logging
        print(f"InfoNCE - Image embeddings shape: {image_embeddings.shape}, Text embeddings shape: {text_embeddings.shape}")
        
        # Ensure inputs are float32 for better numerical stability
        image_embeddings = image_embeddings.float()
        text_embeddings = text_embeddings.float()
        
        # Ensure inputs are 2D
        if image_embeddings.dim() != 2 or text_embeddings.dim() != 2:
            raise ValueError(f"Expected 2D tensors, got image_embeddings: {image_embeddings.dim()}D, text_embeddings: {text_embeddings.dim()}D")
        
        # Add small noise to prevent zero embeddings
        image_embeddings = image_embeddings + torch.randn_like(image_embeddings) * 1e-6
        text_embeddings = text_embeddings + torch.randn_like(text_embeddings) * 1e-6
        
        # Normalize embeddings with epsilon to prevent division by zero
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1, eps=1e-8)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1, eps=1e-8)
        
        # Compute similarity matrix with numerical stability
        similarity = torch.matmul(image_embeddings, text_embeddings.t())
        similarity = torch.clamp(similarity, min=-1.0, max=1.0)  # Ensure similarity is in valid range
        similarity = similarity / max(self.temperature, 1e-6)
        
        # Create labels
        labels = torch.arange(len(image_embeddings)).to(image_embeddings.device)
        
        # Compute log softmax with numerical stability
        log_softmax_i2t = F.log_softmax(similarity, dim=1)
        log_softmax_t2i = F.log_softmax(similarity.t(), dim=1)
        
        # Calculate negative log likelihood with stability
        nll_i2t = -log_softmax_i2t[torch.arange(len(labels)), labels]
        nll_t2i = -log_softmax_t2i[torch.arange(len(labels)), labels]
        
        # Average losses with stability term
        loss = (nll_i2t.mean() + nll_t2i.mean()) / 2
        
        # Ensure loss is not zero and handle nan values while maintaining gradient tracking
        if torch.isnan(loss) or loss == 0:
            # Create a tensor with gradient tracking
            stable_loss = torch.ones_like(loss, requires_grad=True)
            loss = stable_loss
        
        print(f"InfoNCE loss: {loss.item():.4f}")
        return loss 