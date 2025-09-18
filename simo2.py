"""
SimO2 (Self-supervised Image MOdel) Pre-training in Flax NNX

This script provides a JAX/Flax NNX implementation of the SimO2
pre-training paradigm for self-supervised representation learning.

Key Features:
- Model Architecture:
  - Supports DenseNet-121 and a ResNet-18 with Layer Normalization.
  - Uses the built-in flax.nnx.LayerNorm.
- Data Handling:
  - Integrates Hugging Face datasets with streaming to handle
    large-scale datasets like ImageNet-1k and Tiny ImageNet.
  - Implements a two-crop augmentation strategy using torchvision transforms.
  - Custom data generator for streaming and batching (NHWC format).
- Training:
  - A dedicated Trainer class to encapsulate the training and evaluation logic.
  - A vectorized SimO2 contrastive loss function implemented in JAX.
  - JIT-compiled training and evaluation steps for performance.
  - Supports multi-device (GPU/TPU) training via JAX sharding.
  - Logging to Weights & Biases (wandb) for experiment tracking.
  - Checkpointing with Orbax.
- Evaluation & Visualization:
  - A comprehensive evaluation suite to analyze embedding quality.
  - t-SNE and PCA visualizations of the embedding space.
  - Clustering metrics: Silhouette Score, Davies-Bouldin Index, k-means ARI.
  - Post-training linear probing.
"""
import os
import argparse
import random
import time
from functools import partial
import itertools

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
import numpy as np
import flax.nnx as nnx
import optax
import orbax.checkpoint as ocp

from datasets import load_dataset
import torchvision.transforms as T
from PIL import Image

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, adjusted_rand_score,
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

import wandb
from tqdm import tqdm

# For reproducibility
random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------- #
#                           JAX MESH CONFIGURATION                             #
# ---------------------------------------------------------------------------- #

def setup_mesh_from_config(config_dict=None):
    """Setup mesh configuration from config dict or use defaults.
    
    Args:
        config_dict: Optional dictionary with mesh configuration.
        
    Returns:
        Optional[Mesh]: JAX mesh or None if single device.
    """
    # Try to use the new mesh utilities if available
    try:
        from visx.utils.mesh import create_mesh_from_config, setup_distributed_training
        from visx.config.config import MeshConfig
        
        if config_dict and 'mesh' in config_dict:
            # Create MeshConfig from dict
            mesh_config = MeshConfig(**config_dict['mesh'])
        elif config_dict:
            # Create default mesh config but allow override of some settings
            mesh_config = MeshConfig()
            # Check for legacy mesh settings in the main config
            if 'mesh_enabled' in config_dict:
                mesh_config.enabled = config_dict['mesh_enabled']
            if 'mesh_auto_detect' in config_dict:
                mesh_config.auto_detect = config_dict['mesh_auto_detect']
            if 'mesh_shape' in config_dict:
                mesh_config.shape = config_dict['mesh_shape']
            if 'mesh_axis_names' in config_dict:
                mesh_config.axis_names = config_dict['mesh_axis_names']
        else:
            # Use default auto-detection
            mesh_config = MeshConfig()
        
        mesh, device_info = setup_distributed_training(mesh_config)
        return mesh
        
    except ImportError:
        # Fallback to legacy mesh setup if visx modules not available
        print("VISX mesh utilities not available, using legacy mesh setup")
        return setup_legacy_mesh()


def setup_legacy_mesh():
    """Legacy mesh setup for backward compatibility."""
    if jax.device_count() > 1:
        try:
            devices = mesh_utils.create_device_mesh((jax.device_count() // 2, 2))
            mesh = Mesh(devices, axis_names=('data', 'model'))
            print(f"Using device mesh with shape: {mesh.shape}")
            return mesh
        except Exception:
            print("Could not create 2D mesh. Falling back to 1D data parallelism.")
            devices = mesh_utils.create_device_mesh((jax.device_count(),))
            mesh = Mesh(devices, axis_names=('data',))
            return mesh
    else:
        print("Running on a single device.")
        return None


# Setup mesh - this will be set later in main() based on config
mesh = None

# ---------------------------------------------------------------------------- #
#                                    MODELS                                    #
# ---------------------------------------------------------------------------- #

class ProjectionHead(nnx.Module):
    def __init__(self, input_dim: int, embedding_dim: int, *, rngs: nnx.Rngs):
        self.projection = nnx.Linear(
            input_dim,
            embedding_dim,
            use_bias=False,
            rngs=rngs
        )

    def __call__(self, x):
        return self.projection(x)

class SimoModel(nnx.Module):
    def __init__(self, backbone: nnx.Module, feature_dim: int, embedding_dim: int, *, rngs: nnx.Rngs):
        self.backbone = backbone
        self.projection_head = ProjectionHead(feature_dim, embedding_dim, rngs=rngs)

    def __call__(self, x, training=False, return_features=False):
        features = self.backbone(x, training=training)
        if return_features:
            return features
        embeddings = self.projection_head(features)
        norm = jnp.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / jnp.maximum(norm, 1e-12)
        return embeddings

# --- Custom ResNet with Layer Normalization (using nnx.LayerNorm) ---
class BasicBlockLN(nnx.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int, *, rngs: nnx.Rngs):
        super().__init__()
        self.conv1 = nnx.Conv(in_planes, planes, kernel_size=(3, 3), strides=stride, padding=1, use_bias=False, rngs=rngs)
        self.ln1 = nnx.LayerNorm(num_features=planes, rngs=rngs)
        self.conv2 = nnx.Conv(planes, planes, kernel_size=(3, 3), strides=1, padding=1, use_bias=False, rngs=rngs)
        self.ln2 = nnx.LayerNorm(num_features=planes, rngs=rngs)

        self.shortcut = nnx.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nnx.Sequential(
                nnx.Conv(in_planes, self.expansion * planes, kernel_size=(1, 1), strides=stride, use_bias=False, rngs=rngs),
                nnx.LayerNorm(num_features=self.expansion * planes, rngs=rngs)
            )

    def __call__(self, x, training: bool): # Added training flag for API consistency
        out = nnx.relu(self.ln1(self.conv1(x)))
        out = self.ln2(self.conv2(out))
        out += self.shortcut(x)
        out = nnx.relu(out)
        return out

class ResNetLN(nnx.Module):
    def __init__(self, block, num_blocks, *, rngs: nnx.Rngs):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nnx.Conv(3, 64, kernel_size=(3, 3), strides=1, padding=1, use_bias=False, rngs=rngs)
        self.ln1 = nnx.LayerNorm(num_features=64, rngs=rngs)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, rngs=rngs)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, rngs=rngs)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, rngs=rngs)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, rngs=rngs)

    def _make_layer(self, block, planes, num_blocks, stride, *, rngs: nnx.Rngs):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, stride=s, rngs=rngs))
            self.in_planes = planes * block.expansion
        return layers

    def __call__(self, x, training: bool): # Added training flag
        out = nnx.relu(self.ln1(self.conv1(x)))
        for layer in self.layer1: out = layer(out, training)
        for layer in self.layer2: out = layer(out, training)
        for layer in self.layer3: out = layer(out, training)
        for layer in self.layer4: out = layer(out, training)
        out = out.mean(axis=(1, 2)) # Global Average Pooling for NHWC
        return out

def ResNet18LN(rngs: nnx.Rngs):
    return ResNetLN(BasicBlockLN, [2, 2, 2, 2], rngs=rngs)

# --- DenseNet Implementation ---
class Bottleneck(nnx.Module):
    def __init__(self, in_planes: int, growth_rate: int, *, rngs: nnx.Rngs):
        super().__init__()
        self.bn1 = nnx.BatchNorm(in_planes, use_running_average=True, rngs=rngs)
        self.conv1 = nnx.Conv(in_planes, 4 * growth_rate, kernel_size=(1, 1), use_bias=False, rngs=rngs)
        self.bn2 = nnx.BatchNorm(4 * growth_rate, use_running_average=True, rngs=rngs)
        self.conv2 = nnx.Conv(4 * growth_rate, growth_rate, kernel_size=(3, 3), padding=1, use_bias=False, rngs=rngs)

    def __call__(self, x, training: bool):
        out = self.conv1(nnx.relu(self.bn1(x, use_running_average=not training)))
        out = self.conv2(nnx.relu(self.bn2(out, use_running_average=not training)))
        out = jnp.concatenate([out, x], axis=-1) # Concat on channel axis for NHWC
        return out

class Transition(nnx.Module):
    def __init__(self, in_planes: int, out_planes: int, *, rngs: nnx.Rngs):
        super().__init__()
        self.bn = nnx.BatchNorm(in_planes, use_running_average=True, rngs=rngs)
        self.conv = nnx.Conv(in_planes, out_planes, kernel_size=(1, 1), use_bias=False, rngs=rngs)

    def __call__(self, x, training: bool):
        out = self.conv(nnx.relu(self.bn(x, use_running_average=not training)))
        out = nnx.avg_pool(out, window_shape=(2, 2), strides=(2, 2))
        return out

class DenseNet(nnx.Module):
    def __init__(self, block, nblocks, growth_rate=32, reduction=0.5, *, rngs: nnx.Rngs):
        super().__init__()
        self.growth_rate = growth_rate
        num_planes = 2 * growth_rate
        self.conv1 = nnx.Conv(3, num_planes, kernel_size=(7, 7), strides=2, padding=3, use_bias=False, rngs=rngs)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0], rngs=rngs)
        num_planes += nblocks[0] * growth_rate
        out_planes = int(np.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes, rngs=rngs)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1], rngs=rngs)
        num_planes += nblocks[1] * growth_rate
        out_planes = int(np.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes, rngs=rngs)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2], rngs=rngs)
        num_planes += nblocks[2] * growth_rate
        out_planes = int(np.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes, rngs=rngs)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3], rngs=rngs)
        num_planes += nblocks[3] * growth_rate

        self.bn = nnx.BatchNorm(num_planes, use_running_average=True, rngs=rngs)
        self.feature_dim = num_planes

    def _make_dense_layers(self, block, in_planes, nblock, *, rngs: nnx.Rngs):
        layers = []
        for _ in range(nblock):
            layers.append(block(in_planes, self.growth_rate, rngs=rngs))
            in_planes += self.growth_rate
        return layers

    def __call__(self, x, training: bool = False):
        out = self.conv1(x)
        out = nnx.max_pool(out, window_shape=(3, 3), strides=(2, 2), padding=1)

        for layer in self.dense1: out = layer(out, training=training)
        out = self.trans1(out, training=training)
        for layer in self.dense2: out = layer(out, training=training)
        out = self.trans2(out, training=training)
        for layer in self.dense3: out = layer(out, training=training)
        out = self.trans3(out, training=training)
        for layer in self.dense4: out = layer(out, training=training)

        out = self.bn(out, use_running_average=not training)
        out = nnx.relu(out)
        out = out.mean(axis=(1, 2)) # Global Average Pooling for NHWC
        return out

def DenseNet121(rngs: nnx.Rngs):
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32, rngs=rngs)

def get_model(config, rngs):
    model_name = config['model_name']
    if model_name == 'resnet18_layernorm':
        backbone = ResNet18LN(rngs=rngs)
        feature_dim = 512
    elif model_name == 'densenet121':
        backbone = DenseNet121(rngs=rngs)
        feature_dim = backbone.feature_dim
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    model = SimoModel(backbone, feature_dim, config['embedding_dim'], rngs=rngs)
    return model

# ---------------------------------------------------------------------------- #
#                                DATA HANDLING                                 #
# ---------------------------------------------------------------------------- #
def get_hf_data_generator(config):
    image_size = config['image_size']
    # Transforms still output CHW tensors, so we transpose to HWC for Flax.
    pretrain_transform = T.Compose([
        T.RandomResizedCrop(size=image_size, scale=(0.8, 1.0)), T.RandomHorizontalFlip(),
        T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    probe_transform = T.Compose([
        T.Resize((image_size, image_size)), T.RandomHorizontalFlip(), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_transform = T.Compose([
        T.Resize((image_size, image_size)), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset_name, train_split, test_split = ('Maysee/tiny-imagenet', 'train', 'valid') \
        if config['dataset'] == 'tiny-imagenet' else ('imagenet-1k', 'train', 'validation')

    def create_generator(split, transform, two_crop=False):
        dataset = load_dataset(dataset_name, split=split, streaming=True)
        dataset = dataset.shuffle(buffer_size=config['shuffle_buffer_size'], seed=42)
        for example in dataset:
            img = example['image'].convert('RGB')
            if two_crop:
                view1 = np.array(transform(img)).transpose(1, 2, 0)
                view2 = np.array(transform(img)).transpose(1, 2, 0)
                yield {'view1': view1, 'view2': view2, 'label': example['label']}
            else:
                img_transformed = np.array(transform(img)).transpose(1, 2, 0)
                yield {'image': img_transformed, 'label': example['label']}

    def collate_and_batch(generator_fn, batch_size, two_crop=False):
        def _generator():
            batch = []
            for item in generator_fn():
                batch.append(item)
                if len(batch) == batch_size:
                    if two_crop:
                        view1 = jnp.stack([x['view1'] for x in batch])
                        view2 = jnp.stack([x['view2'] for x in batch])
                        labels = jnp.array([x['label'] for x in batch])
                        yield (view1, view2), labels
                    else:
                        images = jnp.stack([x['image'] for x in batch])
                        labels = jnp.array([x['label'] for x in batch])
                        yield images, labels
                    batch = []
        return _generator

    pretrain_gen = collate_and_batch(lambda: create_generator(train_split, pretrain_transform, True), config['batch_size'], True)
    probe_gen = collate_and_batch(lambda: create_generator(train_split, probe_transform), config['batch_size'])
    test_gen = collate_and_batch(lambda: create_generator(test_split, test_transform), 256)
    return pretrain_gen, probe_gen, test_gen

# ---------------------------------------------------------------------------- #
#                                LOSS FUNCTION                                 #
# ---------------------------------------------------------------------------- #
def simo_loss_fn(embeddings, labels, epsilon=1e-3):
    batch_size = embeddings.shape[0]
    dot_prods = jnp.matmul(embeddings, embeddings.T)
    sum_sq = jnp.sum(embeddings**2, axis=1)
    dist_sq = sum_sq[:, None] + sum_sq[None, :] - 2 * dot_prods
    yat = np.sqrt(2)*(dot_prods**2) / (epsilon + dist_sq)
    labels_expanded = labels[:, None]
    intra_class_mask = jnp.triu((labels_expanded == labels_expanded.T), k=1)
    inter_class_mask = jnp.triu((labels_expanded != labels_expanded.T), k=1)
    total_intra_loss = (1 / (1 + yat) * intra_class_mask).sum()
    total_inter_loss = (yat * inter_class_mask).sum()
    total_loss = (total_intra_loss + total_inter_loss) / batch_size
    return total_loss, total_intra_loss / batch_size, total_inter_loss / batch_size

# ---------------------------------------------------------------------------- #
#                           TRAINING & EVAL STEPS                              #
# ---------------------------------------------------------------------------- #
def pretrain_loss_fn(model: SimoModel, batch, training: bool):
    (view1, view2), labels_og = batch
    images = jnp.concatenate([view1, view2], axis=0)
    labels = jnp.concatenate([labels_og, labels_og], axis=0)
    embeddings = model(images, training=training)
    return simo_loss_fn(embeddings, labels)

def linear_probe_loss_fn(backbone, classifier, batch, training: bool):
    images, labels = batch
    features = backbone(images, training=training)
    logits = classifier(features)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    return loss, logits

@nnx.jit
def train_step(model: SimoModel, optimizer: nnx.Optimizer, batch):
    grad_fn = nnx.value_and_grad(pretrain_loss_fn, has_aux=True)
    (loss, (intra, inter)), grads = grad_fn(model, batch, training=True)
    optimizer.update(grads)
    return loss, intra, inter

@nnx.jit
def get_embeddings_step(model: SimoModel, images):
    return model(images, training=False, return_features=False)

# ---------------------------------------------------------------------------- #
#                         EVALUATION & VISUALIZATION                         #
# ---------------------------------------------------------------------------- #
def compute_embeddings(model, dataloader_fn, device_count, limit=5000):
    all_embeddings, all_labels = [], []
    count = 0
    pbar = tqdm(dataloader_fn(), desc="Computing embeddings")
    for images, labels in pbar:
        if device_count > 1:
            batch_size_per_device = images.shape[0] // device_count
            images = images.reshape((device_count, batch_size_per_device) + images.shape[1:])
        embeddings = get_embeddings_step(model, images)
        all_embeddings.append(np.array(jax.device_get(embeddings.reshape(-1, embeddings.shape[-1]))))
        all_labels.append(np.array(jax.device_get(labels)))
        count += images.shape[0] * (device_count if device_count > 1 else 1)
        if count >= limit: break
    return np.concatenate(all_embeddings), np.concatenate(all_labels)

def visualize_embeddings(embeddings, labels, step, epoch):
    print(f"Visualizing embeddings for epoch {epoch}...")
    if len(embeddings) > 2000:
        idx = np.random.choice(len(embeddings), 2000, replace=False)
        embeddings, labels = embeddings[idx], labels[idx]

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1), n_iter=300)
    tsne_results = tsne.fit_transform(embeddings)
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='tab10', alpha=0.7, s=10)
    ax.legend(*scatter.legend_elements(), title="Classes")
    ax.set_title(f't-SNE of Embeddings (Epoch {epoch})')
    wandb.log({"t-SNE": wandb.Image(fig)}, step=step)
    plt.close(fig)

def linear_probe(model, probe_loader_fn, test_loader_fn, config, rngs):
    print("--- Starting Linear Probing ---")
    feature_dim = model.projection_head.projection.in_features
    classifier = nnx.Linear(feature_dim, config['num_classes'], rngs=rngs)
    probe_optimizer = nnx.Optimizer(classifier, optax.adam(1e-3))

    @nnx.jit
    def probe_train_step(backbone, classifier, optimizer, batch):
        grad_fn = nnx.value_and_grad(linear_probe_loss_fn, argnums=1, has_aux=True)
        (loss, _), grads = grad_fn(backbone, classifier, batch, training=False)
        optimizer.update(grads)
        return loss

    @nnx.jit
    def probe_eval_step(backbone, classifier, batch):
        _, logits = linear_probe_loss_fn(backbone, classifier, batch, training=False)
        return logits.argmax(axis=-1)

    pbar = tqdm(itertools.islice(probe_loader_fn(), config['steps_per_epoch']),
                total=config['steps_per_epoch'], desc="Linear Probing Training")
    for images, labels in pbar:
        loss = probe_train_step(model.backbone, classifier, probe_optimizer, (images, labels))
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    all_preds, all_labels = [], []
    for images, labels in tqdm(test_loader_fn(), desc="Linear Probing Testing"):
        preds = probe_eval_step(model.backbone, classifier, (images, labels))
        all_preds.append(np.array(jax.device_get(preds)))
        all_labels.append(np.array(jax.device_get(labels)))
        if sum(map(len, all_labels)) >= 50000: break

    accuracy = accuracy_score(np.concatenate(all_labels), np.concatenate(all_preds)) * 100
    print(f"Linear Probe Accuracy: {accuracy:.2f}%")
    wandb.log({'final_linear_probe_accuracy': accuracy})

# ---------------------------------------------------------------------------- #
#                                    TRAINER                                   #
# ---------------------------------------------------------------------------- #
class Trainer:
    def __init__(self, model, optimizer, pretrain_gen, probe_gen, test_gen, config, rngs):
        self.model = model
        self.optimizer = optimizer
        self.pretrain_gen = pretrain_gen
        self.probe_gen = probe_gen
        self.test_gen = test_gen
        self.config = config
        self.rngs = rngs
        wandb.init(project=config['project_name'], config=config)
        self.checkpoint_dir = f"checkpoints/{wandb.run.name}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpointer = ocp.PyTreeCheckpointer(self.checkpoint_dir)
        self.global_step = 0
        self.device_count = jax.device_count()

    def train(self):
        print("Starting training...")
        total_steps = self.config['num_epochs'] * self.config['steps_per_epoch']
        pbar = tqdm(range(total_steps), desc="Training")
        train_iter = iter(self.pretrain_gen())
        
        for step in pbar:
            epoch = step // self.config['steps_per_epoch']
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.pretrain_gen())
                batch = next(train_iter)

            if self.device_count > 1:
                (v1, v2), labels = batch
                sharded_v1 = jax.device_put(v1, NamedSharding(mesh, P('data')))
                sharded_v2 = jax.device_put(v2, NamedSharding(mesh, P('data')))
                sharded_labels = jax.device_put(labels, NamedSharding(mesh, P('data')))
                sharded_batch = ((sharded_v1, sharded_v2), sharded_labels)
            else:
                sharded_batch = batch

            loss, intra, inter = train_step(self.model, self.optimizer, sharded_batch)

            if jnp.isnan(loss):
                print(f"NaN loss detected at step {self.global_step}. Skipping batch.")
                continue

            loss, intra, inter = jax.device_get([loss, intra, inter])
            pbar.set_postfix({'loss': f'{loss:.4f}', 'epoch': epoch + 1})
            wandb.log({'train/batch_loss': loss, 'train/batch_intra_loss': intra,
                       'train/batch_inter_loss': inter, 'epoch': epoch + 1}, step=self.global_step)
            self.global_step += 1
            
            if (self.global_step % self.config['eval_interval_steps']) == 0:
                print(f"\n--- Starting Evaluation for Step {self.global_step} ---")
                embeddings, labels = compute_embeddings(self.model, self.test_gen, self.device_count)
                visualize_embeddings(embeddings, labels, self.global_step, epoch + 1)
                print(f"--- Finished Evaluation for Step {self.global_step} ---\n")

            if (self.global_step % self.config['save_interval_steps']) == 0:
                self.save_checkpoint(self.global_step)

        print("--- Pre-training finished ---")
        self.save_checkpoint('final')
        print("\n--- Running Final Evaluations ---")
        linear_probe(self.model, self.probe_gen, self.test_gen, self.config, self.rngs)
        wandb.finish()

    def save_checkpoint(self, step):
        _, params, _ = nnx.split(self.model, nnx.Param, ...)
        self.checkpointer.save(os.path.join(self.checkpoint_dir, f"model_step_{step}"), args=ocp.args.StandardSave(params))
        print(f"Checkpoint saved for step {step}")

# ---------------------------------------------------------------------------- #
#                                MAIN FUNCTION                                 #
# ---------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="SIMO Pre-training in Flax/NNX")
    parser.add_argument('--dataset', type=str, default='tiny-imagenet', choices=['imagenet-1k', 'tiny-imagenet'])
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--project_name', type=str, default='simo-flax-nnx')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--shuffle_buffer_size', type=int, default=10000)
    parser.add_argument('--model_name', type=str, default='densenet121', choices=['densenet121', 'resnet18_layernorm'])
    parser.add_argument('--eval_interval_steps', type=int, default=5000)
    parser.add_argument('--save_interval_steps', type=int, default=10000)
    # Add mesh configuration arguments
    parser.add_argument('--mesh_enabled', type=bool, default=True, help='Enable mesh configuration')
    parser.add_argument('--mesh_auto_detect', type=bool, default=True, help='Auto-detect mesh configuration')
    parser.add_argument('--mesh_shape', type=int, nargs='+', help='Custom mesh shape, e.g., --mesh_shape 4 2')
    parser.add_argument('--mesh_axis_names', type=str, nargs='+', help='Mesh axis names, e.g., --mesh_axis_names batch model')
    args = parser.parse_args()
    config = vars(args)

    if config['dataset'] == 'tiny-imagenet':
        config['num_classes'] = 200
        config['steps_per_epoch'] = 100_000 // config['batch_size']
    else: # imagenet-1k
        config['num_classes'] = 1000
        config['steps_per_epoch'] = 1_281_167 // config['batch_size']
    
    print("--- Configuration ---", config, "---------------------", sep="\n")
    
    # Setup mesh configuration
    global mesh
    mesh = setup_mesh_from_config(config)
    
    pretrain_gen, probe_gen, test_gen = get_hf_data_generator(config)
    rngs = nnx.Rngs(param=0, dropout=1)
    model = get_model(config, rngs=rngs)
    
    if mesh:
        model = nnx.shard(model, mesh)

    optimizer = nnx.Optimizer(model, optax.adamw(config['learning_rate']))
    trainer = Trainer(model, optimizer, pretrain_gen, probe_gen, test_gen, config, rngs)
    trainer.train()

if __name__ == '__main__':
    main()






