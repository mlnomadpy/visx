"""Training utilities and loops."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds
import optax
from functools import partial
from flax import nnx
from typing import Tuple, Dict, Any

from ..config import Config
from .registry import ModelRegistry


def loss_fn(model, batch):
    """Compute loss for a batch."""
    logits = model(batch['image'], training=True)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label']
    ).mean()
    return loss, logits


@nnx.jit
def train_step(model, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    """Train for a single step."""
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])
    optimizer.update(grads)


@nnx.jit
def eval_step(model, metrics: nnx.MultiMetric, batch):
    """Evaluate for a single step."""
    loss, logits = loss_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])


def create_dataset(config: Config, split: str = 'train') -> tf.data.Dataset:
    """Create dataset based on configuration."""
    dataset_cfg = config.dataset
    
    def preprocess_fn(sample):
        return {
            'image': tf.cast(sample[dataset_cfg.image_key], tf.float32) / 255.0,
            'label': sample[dataset_cfg.label_key]
        }
    
    split_name = dataset_cfg.train_split if split == 'train' else dataset_cfg.test_split
    
    dataset = tfds.load(
        dataset_cfg.name, 
        split=split_name, 
        as_supervised=False
    ).map(
        preprocess_fn, 
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(
        dataset_cfg.batch_size, 
        drop_remainder=True
    ).prefetch(tf.data.AUTOTUNE)
    
    return dataset


def create_optimizer(config: Config, model: nnx.Module) -> nnx.Optimizer:
    """Create optimizer based on configuration."""
    training_cfg = config.training
    
    if training_cfg.optimizer.lower() == 'adamw':
        optimizer_fn = optax.adamw(learning_rate=training_cfg.learning_rate)
    elif training_cfg.optimizer.lower() == 'sgd':
        optimizer_fn = optax.sgd(learning_rate=training_cfg.learning_rate, momentum=training_cfg.momentum)
    elif training_cfg.optimizer.lower() == 'adam':
        optimizer_fn = optax.adam(learning_rate=training_cfg.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {training_cfg.optimizer}")
    
    return nnx.Optimizer(model, optimizer_fn)


def train_model(config: Config) -> Tuple[nnx.Module, Dict[str, Any]]:
    """Train a model based on configuration."""
    tf.random.set_seed(config.training.rng_seed)
    
    # Create model
    rngs = nnx.Rngs(config.training.rng_seed)
    model = ModelRegistry.create_model(config.model.name, config, rngs)
    
    # Create optimizer and metrics
    optimizer = create_optimizer(config, model)
    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average('loss'),
    )
    
    # Create datasets
    train_ds = create_dataset(config, 'train')
    test_ds = create_dataset(config, 'test')
    
    # Training loop
    train_ds_iter = iter(train_ds)
    test_ds_iter = iter(test_ds)
    
    # Calculate training steps
    dataset_info = tfds.load(config.dataset.name, with_info=True)[1]
    train_size = dataset_info.splits[config.dataset.train_split].num_examples
    total_train_steps = (train_size // config.dataset.batch_size) * config.dataset.num_epochs
    
    # History tracking
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'test_loss': [],
        'test_accuracy': []
    }
    
    if config.verbose:
        print(f"Starting training for {config.dataset.num_epochs} epochs...")
        print(f"Total training steps: {total_train_steps}")
    
    for step in range(total_train_steps):
        # Training step
        try:
            batch = next(train_ds_iter)
        except StopIteration:
            train_ds_iter = iter(train_ds)
            batch = next(train_ds_iter)
        
        train_step(model, optimizer, metrics, batch)
        
        # Evaluation
        if step % config.dataset.eval_every == 0:
            # Compute metrics on training batch
            train_loss = metrics.compute()['loss']
            train_accuracy = metrics.compute()['accuracy']
            
            # Reset metrics for test evaluation
            metrics.reset()
            
            # Evaluate on test set (few batches)
            test_batches = 0
            for test_batch in test_ds_iter:
                eval_step(model, metrics, test_batch)
                test_batches += 1
                if test_batches >= 10:  # Evaluate on 10 batches
                    break
            
            test_loss = metrics.compute()['loss']
            test_accuracy = metrics.compute()['accuracy']
            
            # Store history
            history['train_loss'].append(float(train_loss))
            history['train_accuracy'].append(float(train_accuracy))
            history['test_loss'].append(float(test_loss))
            history['test_accuracy'].append(float(test_accuracy))
            
            if config.verbose:
                print(f"Step {step:4d}: train_loss={train_loss:.4f}, train_acc={train_accuracy:.4f}, "
                      f"test_loss={test_loss:.4f}, test_acc={test_accuracy:.4f}")
            
            # Reset metrics for next training
            metrics.reset()
            
            # Recreate test iterator
            test_ds_iter = iter(test_ds)
    
    if config.verbose:
        print("Training completed!")
    
    return model, history