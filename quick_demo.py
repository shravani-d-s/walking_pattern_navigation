"""
Quick Demo Script - For Fast Presentations
Loads pre-trained model and demonstrates adaptive navigation
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pyttsx3
import time
import pickle

def generate_test_sample(pattern_type):
    """Generate a single test sample for given pattern"""
    patterns = {
        'normal': {'speed': 1.0, 'variance': 0.1},
        'slow': {'speed': 0.5, 'variance': 0.15},
        'fast': {'speed': 1.5, 'variance': 0.08},
        'cane': {'speed': 0.7, 'variance': 0.25}
    }
    
    params = patterns[pattern_type]
    sequence = []
    
    for t in range(50):
        base_speed = params['speed']
        noise = np.random.normal(0, params['variance'])
        
        accel_x = base_speed * np.sin(2 * np.pi * t / 20) + noise
        accel_y = 1.0 + np.random.normal(0, 0.1)
        accel_z = base_speed * np.cos(2 * np.pi * t / 20) + noise
        
        gyro_x = base_speed * np.random.normal(0, 0.2)
        gyro_y = base_speed * np.random.normal(0, 0.2)
        gyro_z = base_speed * np.random.normal(0, 0.15)
        
        sequence.append([accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z])
    
    return np.array(sequence)


def demo_adaptive_navigation():
    """
    Quick demo of the adaptive navigation system
    """
    
    print("\n" + "="*70)
    print("  ADAPTIVE NAVIGATION SYSTEM - QUICK DEMO")
    print("  Deep Learning Walking Pattern Recognition")
    print("="*70)
    
    # Load model (if exists, otherwise train a small one)
    try:
        print("\nLoading trained model...")
        model = keras.models.load_model('walking_pattern_model.h5')
        print("✓ Model loaded successfully!")
    except:
        print("\n⚠ Model not found. Please run the main script first.")
        print("  Command: python walking_pattern_recognition.py")
        return
    
    # Initialize text-to-speech
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    
    # Pattern configurations
    configs = {
        'normal': {'delay': 3, 'rate': 150, 'prompt': 'Turn right in 50 meters'},
        'slow': {'delay': 5, 'rate': 130, 'prompt': 'Prepare to turn right in 50 meters. There is a curb ahead.'},
        'fast': {'delay': 2, 'rate': 170, 'prompt': 'Turn right in 50 meters'},
        'cane': {'delay': 6, 'rate': 120, 'prompt': 'Prepare to turn right in 50 meters. There is a curb ahead. Ramp available on right.'}
    }
    
    patterns = ['normal', 'slow', 'fast', 'cane']
    label_map = {0: 'cane', 1: 'fast', 2: 'normal', 3: 'slow'}  # Update based on your encoder
    
    print("\n" + "─"*70)
    print("DEMONSTRATING 4 WALKING PATTERNS")
    print("─"*70)
    
    for pattern in patterns:
        print(f"\n{'='*70}")
        print(f"  Testing Pattern: {pattern.upper()}")
        print("="*70)
        
        # Generate test sample
        sensor_data = generate_test_sample(pattern)
        X = sensor_data.reshape(1, 50, 6)
        
        # Predict
        predictions = model.predict(X, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class] * 100
        
        # Map to label
        predicted_pattern = label_map.get(predicted_class, 'normal')
        
        print(f"\n📊 Sensor Data Shape: {sensor_data.shape}")
        print(f"🤖 Predicted: {predicted_pattern.upper()}")
        print(f"📈 Confidence: {confidence:.1f}%")
        
        # Get config
        config = configs[predicted_pattern]
        
        print(f"\n⚙️  Adaptive Settings:")
        print(f"   • Speech Rate: {config['rate']} words/min")
        print(f"   • Prompt Delay: {config['delay']} seconds")
        print(f"   • Prompt Type: {'Detailed' if config['delay'] > 3 else 'Standard'}")
        
        # Speak
        print(f"\n🔊 Voice Prompt:")
        print(f"   \"{config['prompt']}\"")
        
        engine.setProperty('rate', config['rate'])
        engine.say(config['prompt'])
        engine.runAndWait()
        
        print(f"\n⏱️  Waiting {config['delay']} seconds before next prompt...")
        time.sleep(2)  # Shortened for demo
    
    print("\n" + "="*70)
    print("  DEMO COMPLETED ✓")
    print("="*70)
    print("\nKey Observations:")
    print("  • Slow/Cane users get detailed prompts with longer delays")
    print("  • Fast walkers get quick, concise prompts")
    print("  • Speech rate adapts to walking speed")
    print("  • System provides better accessibility for all users")
    

def show_model_info():
    """Display model information"""
    try:
        model = keras.models.load_model('walking_pattern_model.h5')
        
        print("\n" + "="*70)
        print("  MODEL ARCHITECTURE")
        print("="*70)
        model.summary()
        
        print("\n" + "="*70)
        print("  MODEL DETAILS")
        print("="*70)
        print(f"Total Parameters: {model.count_params():,}")
        print(f"Input Shape: (50, 6) - 50 timesteps × 6 sensors")
        print(f"Output Classes: 4 (normal, slow, fast, cane)")
        print(f"Architecture: CNN-LSTM Hybrid")
        
    except:
        print("\n⚠ Model not found. Run main script first.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--info':
        show_model_info()
    else:
        demo_adaptive_navigation()