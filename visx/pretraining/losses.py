"""Loss functions for pretraining methods."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any


class PretrainingLoss(ABC):
    """Abstract base class for pretraining loss functions."""
    
    @abstractmethod
    def __call__(self, *args, **kwargs) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Compute the loss and return loss value with optional metrics."""
        pass


class SimoLoss(PretrainingLoss):
    """SIMO loss function for self-supervised learning.
    
    Based on the YAT (Yet Another Transformation) approach where:
    - Intra-class pairs should have high similarity
    - Inter-class pairs should have low similarity
    """
    
    def __init__(self, epsilon: float = 1e-3):
        self.epsilon = epsilon
    
    def __call__(self, embeddings: jnp.ndarray, labels: jnp.ndarray) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """Compute SIMO loss.
        
        Args:
            embeddings: Normalized embeddings of shape [batch_size, embedding_dim]
            labels: Labels of shape [batch_size]
            
        Returns:
            Tuple of (total_loss, (intra_loss, inter_loss))
        """
        batch_size = embeddings.shape[0]
        
        # Compute pairwise dot products and distances
        dot_prods = jnp.matmul(embeddings, embeddings.T)
        sum_sq = jnp.sum(embeddings**2, axis=1)
        dist_sq = sum_sq[:, None] + sum_sq[None, :] - 2 * dot_prods
        
        # YAT transformation: emphasis on similarity for similar items
        yat = np.sqrt(2) * (dot_prods**2) / (self.epsilon + dist_sq)
        
        # Create masks for intra-class and inter-class pairs
        labels_expanded = labels[:, None]
        intra_class_mask = jnp.triu((labels_expanded == labels_expanded.T), k=1)
        inter_class_mask = jnp.triu((labels_expanded != labels_expanded.T), k=1)
        
        # Compute losses
        total_intra_loss = (1 / (1 + yat) * intra_class_mask).sum()
        total_inter_loss = (yat * inter_class_mask).sum()
        
        # Normalize by batch size
        intra_loss = total_intra_loss / batch_size
        inter_loss = total_inter_loss / batch_size
        total_loss = intra_loss + inter_loss
        
        return total_loss, (intra_loss, inter_loss)


class SimCLRLoss(PretrainingLoss):
    """SimCLR contrastive loss function."""
    
    def __init__(self, temperature: float = 0.1):
        self.temperature = temperature
    
    def __call__(self, z1: jnp.ndarray, z2: jnp.ndarray) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Compute SimCLR contrastive loss.
        
        Args:
            z1: First view projections of shape [batch_size, projection_dim]
            z2: Second view projections of shape [batch_size, projection_dim]
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        batch_size = z1.shape[0]
        
        # Normalize projections
        z1_norm = z1 / jnp.linalg.norm(z1, axis=1, keepdims=True)
        z2_norm = z2 / jnp.linalg.norm(z2, axis=1, keepdims=True)
        
        # Concatenate all projections
        z = jnp.concatenate([z1_norm, z2_norm], axis=0)  # 2*batch_size x projection_dim
        
        # Compute similarity matrix
        sim_matrix = jnp.matmul(z, z.T) / self.temperature
        
        # Create labels for positive pairs
        labels = jnp.arange(2 * batch_size)
        labels = jnp.where(labels < batch_size, labels + batch_size, labels - batch_size)
        
        # Remove self-similarities from consideration
        mask = jnp.eye(2 * batch_size)
        sim_matrix = sim_matrix - mask * 1e9
        
        # Compute contrastive loss
        loss = jax.scipy.special.logsumexp(sim_matrix, axis=1) - sim_matrix[jnp.arange(2 * batch_size), labels]
        loss = loss.mean()
        
        return loss, {"loss": loss}


class BYOLLoss(PretrainingLoss):
    """BYOL (Bootstrap Your Own Latent) loss function."""
    
    def __call__(self, online_pred1: jnp.ndarray, online_pred2: jnp.ndarray,
                 target_proj1: jnp.ndarray, target_proj2: jnp.ndarray) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Compute BYOL loss.
        
        Args:
            online_pred1: Online network predictions for view 1
            online_pred2: Online network predictions for view 2  
            target_proj1: Target network projections for view 1
            target_proj2: Target network projections for view 2
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        
        def cosine_similarity(x, y):
            x_norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
            y_norm = jnp.linalg.norm(y, axis=-1, keepdims=True)
            return jnp.sum(x * y, axis=-1) / (x_norm * y_norm + 1e-8)
        
        # BYOL loss: negative cosine similarity
        loss1 = -cosine_similarity(online_pred1, jax.lax.stop_gradient(target_proj2)).mean()
        loss2 = -cosine_similarity(online_pred2, jax.lax.stop_gradient(target_proj1)).mean()
        
        total_loss = (loss1 + loss2) / 2
        return total_loss, {"loss": total_loss}