"""
AlphaZero-style Neural Network for Checkers
Dual-head network with policy and value outputs.
"""

import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("TensorFlow not found - model creation disabled")


def create_residual_block(x, filters, name_prefix):
    """Create a residual block with batch normalization."""
    shortcut = x
    
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False, 
                      name=f'{name_prefix}_conv1')(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn1')(x)
    x = layers.ReLU(name=f'{name_prefix}_relu1')(x)
    
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False,
                      name=f'{name_prefix}_conv2')(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn2')(x)
    
    x = layers.Add(name=f'{name_prefix}_add')([shortcut, x])
    x = layers.ReLU(name=f'{name_prefix}_relu2')(x)
    
    return x


def create_checkers_model(num_residual_blocks=6, filters=128, policy_size=1024):
    """
    Create the AlphaZero-style neural network for checkers.
    
    Input: 8x8x4 board tensor
    Outputs:
        - policy: probability distribution over moves (1024 possible)
        - value: win probability [-1, 1]
    """
    if not HAS_TF:
        raise RuntimeError("TensorFlow is required to create the model")
    
    # Input layer
    input_layer = layers.Input(shape=(8, 8, 4), name='board_input')
    
    # Initial convolution
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False, name='initial_conv')(input_layer)
    x = layers.BatchNormalization(name='initial_bn')(x)
    x = layers.ReLU(name='initial_relu')(x)
    
    # Residual tower
    for i in range(num_residual_blocks):
        x = create_residual_block(x, filters, f'res_block_{i}')
    
    # Policy head
    policy = layers.Conv2D(32, 1, use_bias=False, name='policy_conv')(x)
    policy = layers.BatchNormalization(name='policy_bn')(policy)
    policy = layers.ReLU(name='policy_relu')(policy)
    policy = layers.Flatten(name='policy_flatten')(policy)
    policy = layers.Dense(policy_size, activation='softmax', name='policy_output')(policy)
    
    # Value head
    value = layers.Conv2D(32, 1, use_bias=False, name='value_conv')(x)
    value = layers.BatchNormalization(name='value_bn')(value)
    value = layers.ReLU(name='value_relu')(value)
    value = layers.Flatten(name='value_flatten')(value)
    value = layers.Dense(256, activation='relu', name='value_dense1')(value)
    value = layers.Dense(1, activation='tanh', name='value_output')(value)
    
    model = Model(inputs=input_layer, outputs=[policy, value], name='checkers_alphazero')
    
    return model


def compile_model(model, learning_rate=0.001):
    """Compile the model with appropriate losses."""
    if not HAS_TF:
        raise RuntimeError("TensorFlow is required")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            'policy_output': 'categorical_crossentropy',
            'value_output': 'mse'
        },
        loss_weights={
            'policy_output': 1.0,
            'value_output': 1.0
        },
        metrics={
            'policy_output': 'accuracy',
            'value_output': 'mae'
        }
    )
    return model


def export_to_tfjs(model, output_dir):
    """Export the model to TensorFlow.js format."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        import tensorflowjs as tfjs
        tfjs.converters.save_keras_model(model, output_dir)
        print(f"Model exported to TF.js format: {output_dir}")
    except ImportError:
        # Fallback: save as Keras format for later conversion
        keras_path = os.path.join(output_dir, "model.keras")
        model.save(keras_path)
        print(f"tensorflowjs not installed - saved as Keras format: {keras_path}")
        print("Convert later with: tensorflowjs_converter --input_format keras model.keras tfjs_model/")


class CheckersNetwork:
    """Wrapper class for the neural network with prediction utilities."""
    
    def __init__(self, model=None, model_path=None):
        self.model = model
        if model_path is not None:
            self.load(model_path)
        elif model is None:
            self.model = create_checkers_model()
            compile_model(self.model)
    
    def predict(self, board_tensor):
        """
        Predict policy and value for a single board state.
        
        Args:
            board_tensor: 8x8x4 numpy array
        
        Returns:
            (policy, value) tuple
        """
        if board_tensor.ndim == 3:
            board_tensor = board_tensor[np.newaxis, ...]
        
        policy, value = self.model.predict(board_tensor, verbose=0)
        return policy[0], value[0, 0]
    
    def predict_batch(self, board_tensors):
        """Predict for a batch of board states."""
        return self.model.predict(board_tensors, verbose=0)
    
    def save(self, filepath):
        """Save model weights."""
        self.model.save_weights(filepath)
    
    def load(self, filepath):
        """Load model weights."""
        if self.model is None:
            self.model = create_checkers_model()
            compile_model(self.model)
        self.model.load_weights(filepath)
    
    def export_tfjs(self, output_dir):
        """Export to TensorFlow.js format."""
        export_to_tfjs(self.model, output_dir)


if __name__ == "__main__":
    # Test model creation
    print("Creating model...")
    model = create_checkers_model()
    model.summary()
    
    # Test prediction
    print("\nTesting prediction...")
    compile_model(model)
    test_input = np.random.randn(1, 8, 8, 4).astype(np.float32)
    policy, value = model.predict(test_input, verbose=0)
    
    print(f"Policy shape: {policy.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Policy sum: {policy.sum():.4f}")
    print(f"Value: {value[0, 0]:.4f}")
    
    # Test with CheckersNetwork wrapper
    print("\nTesting CheckersNetwork wrapper...")
    network = CheckersNetwork()
    policy, value = network.predict(test_input[0])
    print(f"Policy shape: {policy.shape}, Value: {value:.4f}")
