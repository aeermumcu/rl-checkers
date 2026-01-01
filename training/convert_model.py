#!/usr/bin/env python3
"""Convert Keras 3 model to TensorFlow.js format"""
import os
import sys

print("Loading TensorFlow...")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
import tensorflow as tf

model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'model.keras')
output_dir = os.path.join(os.path.dirname(__file__), '..', 'model', 'tfjs')

print(f"Loading model from: {model_path}")
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")
print(f"Input shape: {model.input_shape}")
print(f"Output shapes: {[o.shape for o in model.outputs]}")

print(f"\nConverting to TensorFlow.js format...")
os.makedirs(output_dir, exist_ok=True)

import tensorflowjs as tfjs
tfjs.converters.save_keras_model(model, output_dir)
print(f"\n✅ Model exported to: {output_dir}")
print(f"   Files: {os.listdir(output_dir)}")
