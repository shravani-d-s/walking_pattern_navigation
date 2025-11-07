"""
Walking Pattern Recognition and Adaptive Navigation System
Deep Learning-Based Implementation using TensorFlow/Keras

Project: Dynamically adjust navigation voice prompts based on walking patterns
Author: Shravani Sawai
Date: October 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pyttsx3
import time
from datetime import datetime

# ==================== 1. DATA GENERATION ====================

def generate_walking_data(pattern_type, num_samples=100, sequence_length=50):
    """
    Generate synthetic sensor data for different walking patterns
    
    Parameters:
    - pattern_type: 'normal', 'slow', 'fast', 'cane'
    - num_samples: number of sequences to generate
    - sequence_length: length of each time series sequence
    
    Returns:
    - sensor_data: shape (num_samples, sequence_length, 6)
    - labels: pattern type for each sample
    """
    
    # Define characteristics for each walking pattern
    patterns = {
        'normal': {'speed': 1.0, 'variance': 0.1, 'stop_freq': 0.05},
        'slow': {'speed': 0.5, 'variance': 0.15, 'stop_freq': 0.2},
        'fast': {'speed': 1.5, 'variance': 0.08, 'stop_freq': 0.02},
        'cane': {'speed': 0.7, 'variance': 0.25, 'stop_freq': 0.3}
    }
    
    params = patterns[pattern_type]
    data = []
    
    for _ in range(num_samples):
        sequence = []
        for t in range(sequence_length):
            # Simulate accelerometer data (X, Y, Z axes)
            base_speed = params['speed']
            noise = np.random.normal(0, params['variance'])
            
            # Add occasional stops
            if np.random.random() < params['stop_freq']:
                base_speed *= 0.2
            
            accel_x = base_speed * np.sin(2 * np.pi * t / 20) + noise
            accel_y = 1.0 + np.random.normal(0, 0.1)  # Gravity component
            accel_z = base_speed * np.cos(2 * np.pi * t / 20) + noise
            
            # Simulate gyroscope data (rotation rates)
            gyro_x = base_speed * np.random.normal(0, 0.2)
            gyro_y = base_speed * np.random.normal(0, 0.2)
            gyro_z = base_speed * np.random.normal(0, 0.15)
            
            sequence.append([accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z])
        
        data.append(sequence)
    
    return np.array(data), [pattern_type] * num_samples


def create_dataset():
    """Create complete dataset with all walking patterns"""
    print("Generating synthetic walking pattern dataset...")
    
    patterns = ['normal', 'slow', 'fast', 'cane']
    all_data = []
    all_labels = []
    
    for pattern in patterns:
        data, labels = generate_walking_data(pattern, num_samples=500)
        all_data.append(data)
        all_labels.extend(labels)
        print(f"Generated {len(data)} samples for {pattern} walking pattern")
    
    X = np.vstack(all_data)
    y = np.array(all_labels)
    
    print(f"\nTotal dataset shape: {X.shape}")
    print(f"Classes: {np.unique(y)}")
    
    return X, y


# ==================== 2. MODEL ARCHITECTURE ====================

def build_cnn_lstm_model(input_shape, num_classes):
    """
    Build CNN-LSTM hybrid model for time series classification
    
    Architecture:
    - Conv1D layers for feature extraction
    - LSTM layers for temporal pattern recognition
    - Dense layers for classification
    """
    
    model = keras.Sequential([
        # CNN layers for feature extraction
        layers.Conv1D(filters=64, kernel_size=3, activation='relu', 
                     input_shape=input_shape, name='conv1'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        
        layers.Conv1D(filters=128, kernel_size=3, activation='relu', name='conv2'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        
        # LSTM layers for temporal patterns
        layers.LSTM(64, return_sequences=True, name='lstm1'),
        layers.Dropout(0.3),
        
        layers.LSTM(32, name='lstm2'),
        layers.Dropout(0.3),
        
        # Dense layers for classification
        layers.Dense(64, activation='relu', name='dense1'),
        layers.Dropout(0.2),
        
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ==================== 3. TRAINING ====================

def train_model(X_train, y_train, X_val, y_val):
    """Train the CNN-LSTM model"""
    
    print("\n" + "="*60)
    print("TRAINING DEEP LEARNING MODEL")
    print("="*60)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(np.unique(y_train_encoded))
    
    model = build_cnn_lstm_model(input_shape, num_classes)
    
    print(f"\nModel Architecture:")
    model.summary()
    
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    # Train
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train_encoded,
        validation_data=(X_val, y_val_encoded),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate
    print("\nEvaluating model...")
    train_loss, train_acc = model.evaluate(X_train, y_train_encoded, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val_encoded, verbose=0)
    
    print(f"\nTraining Accuracy: {train_acc*100:.2f}%")
    print(f"Validation Accuracy: {val_acc*100:.2f}%")
    
    return model, label_encoder, history


def plot_training_history(history):
    """Plot training and validation metrics"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss
    ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("\nTraining history plot saved as 'training_history.png'")
    plt.show()


# ==================== 4. ADAPTIVE NAVIGATION SYSTEM ====================

class AdaptiveNavigationSystem:
    """
    Real-time navigation system that adapts prompts based on walking patterns
    """
    
    def __init__(self, model, label_encoder):
        self.model = model
        self.label_encoder = label_encoder
        self.engine = pyttsx3.init()
        
        # Configure voice properties
        self.engine.setProperty('rate', 150)  # Default speed
        self.engine.setProperty('volume', 0.9)
        
        # Navigation prompts
        self.prompts = {
            'standard': [
                "Turn right in 50 meters",
                "Continue straight for 100 meters",
                "Turn left at the intersection",
                "Destination ahead in 200 meters"
            ],
            'detailed': [
                "Prepare to turn right in 50 meters. There is a curb ahead.",
                "Continue straight for 100 meters. The pavement is smooth with no obstacles.",
                "Turn left at the intersection. A traffic light with audio signal is present.",
                "Destination ahead in 200 meters. The entrance has a ramp on the right side."
            ]
        }
        
        # Timing configurations
        self.timing_config = {
            'normal': {'delay': 3, 'rate': 150, 'detailed': False},
            'slow': {'delay': 5, 'rate': 130, 'detailed': True},
            'fast': {'delay': 2, 'rate': 170, 'detailed': False},
            'cane': {'delay': 6, 'rate': 120, 'detailed': True}
        }
    
    def predict_pattern(self, sensor_sequence):
        """Predict walking pattern from sensor data"""
        # Reshape for model input
        X = sensor_sequence.reshape(1, sensor_sequence.shape[0], sensor_sequence.shape[1])
        
        # Predict
        predictions = self.model.predict(X, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        # Decode label
        pattern = self.label_encoder.inverse_transform([predicted_class])[0]
        
        return pattern, confidence
    
    def speak_prompt(self, prompt, pattern):
        """Speak navigation prompt with adaptive settings"""
        config = self.timing_config[pattern]
        
        # Adjust speech rate
        self.engine.setProperty('rate', config['rate'])
        
        # Speak
        print(f"\n🔊 [{pattern.upper()}] {prompt}")
        self.engine.say(prompt)
        self.engine.runAndWait()
    
    def get_adaptive_prompt(self, pattern):
        """Get appropriate prompt based on walking pattern"""
        config = self.timing_config[pattern]
        prompt_type = 'detailed' if config['detailed'] else 'standard'
        return np.random.choice(self.prompts[prompt_type])
    
    def run_simulation(self, duration_seconds=30):
        """Run real-time simulation of adaptive navigation"""
        
        print("\n" + "="*60)
        print("ADAPTIVE NAVIGATION SYSTEM - LIVE SIMULATION")
        print("="*60)
        print("\nSimulating real-time walking pattern detection...")
        print("The system will adapt voice prompts based on detected patterns.\n")
        
        start_time = time.time()
        iteration = 0
        
        while (time.time() - start_time) < duration_seconds:
            iteration += 1
            
            # Generate random sensor data (simulating real-time input)
            random_pattern = np.random.choice(['normal', 'slow', 'fast', 'cane'])
            sensor_data, _ = generate_walking_data(random_pattern, num_samples=1, sequence_length=50)
            sensor_sequence = sensor_data[0]
            
            # Predict pattern
            predicted_pattern, confidence = self.predict_pattern(sensor_sequence)
            
            print(f"\n{'─'*60}")
            print(f"Iteration {iteration} | Time: {time.time() - start_time:.1f}s")
            print(f"Detected Pattern: {predicted_pattern.upper()} (Confidence: {confidence*100:.1f}%)")
            
            # Get adaptive prompt
            prompt = self.get_adaptive_prompt(predicted_pattern)
            
            # Speak with adaptive timing
            self.speak_prompt(prompt, predicted_pattern)
            
            # Wait based on pattern-specific delay
            delay = self.timing_config[predicted_pattern]['delay']
            print(f"Next prompt in {delay} seconds...")
            time.sleep(delay)
        
        print("\n" + "="*60)
        print("SIMULATION COMPLETED")
        print("="*60)


# ==================== 5. MAIN EXECUTION ====================

def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print(" WALKING PATTERN RECOGNITION & ADAPTIVE NAVIGATION SYSTEM")
    print(" Deep Learning-Based Voice Prompt Adaptation")
    print("="*70)
    
    # Step 1: Generate Dataset
    X, y = create_dataset()
    
    # Step 2: Split data
    print("\nSplitting dataset...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Step 3: Train Model
    model, label_encoder, history = train_model(X_train, y_train, X_val, y_val)
    
    # Step 4: Plot results
    plot_training_history(history)
    
    # Step 5: Test on test set
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    y_test_encoded = label_encoder.transform(y_test)
    test_loss, test_acc = model.evaluate(X_test, y_test_encoded, verbose=0)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Step 6: Save model
    model.save('walking_pattern_model.h5')
    print("\nModel saved as 'walking_pattern_model.h5'")
    
    # Step 7: Run Adaptive Navigation System
    print("\n" + "="*60)
    print("STARTING ADAPTIVE NAVIGATION DEMO")
    print("="*60)
    
    nav_system = AdaptiveNavigationSystem(model, label_encoder)
    
    # Run simulation for 30 seconds
    nav_system.run_simulation(duration_seconds=30)
    
    print("\n✓ Project demonstration completed successfully!")
    print("\nGenerated files:")
    print("  - walking_pattern_model.h5 (trained model)")
    print("  - training_history.png (training metrics)")


if __name__ == "__main__":
    main()