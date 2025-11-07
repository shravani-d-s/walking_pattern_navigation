"""
Real-Time Sensor Data Server
Receives live sensor data from smartphone and makes predictions

Setup:
1. Install Flask: pip install flask
2. Run this script: python realtime_server.py
3. Configure phone app to send data to your computer's IP
"""

from flask import Flask, request, jsonify, render_template_string
import json
import numpy as np
from collections import deque
import tensorflow as tf
from datetime import datetime
import pyttsx3
import threading

app = Flask(__name__)

# Global variables
sensor_buffer = deque(maxlen=50)  # Store last 50 readings
model = None
label_encoder = None
engine = None
last_prediction = {"pattern": "Unknown", "confidence": 0, "timestamp": ""}
prediction_lock = threading.Lock()

# Pattern configurations
timing_config = {
    'normal': {'delay': 3, 'rate': 150, 'detailed': False},
    'slow': {'delay': 5, 'rate': 130, 'detailed': True},
    'fast': {'delay': 2, 'rate': 170, 'detailed': False},
    'cane': {'delay': 6, 'rate': 120, 'detailed': True}
}

prompts = {
    'standard': [
        "Turn right in 50 meters",
        "Continue straight for 100 meters",
        "Turn left at the intersection",
        "Destination ahead in 200 meters"
    ],
    'detailed': [
        "Prepare to turn right in 50 meters. There is a curb ahead.",
        "Continue straight for 100 meters. Smooth pavement, no obstacles.",
        "Turn left at the intersection. Traffic light with audio signal present.",
        "Destination ahead in 200 meters. Entrance has ramp on right side."
    ]
}

def load_model():
    """Load trained model"""
    global model
    try:
        model = tf.keras.models.load_model('walking_pattern_model.h5')
        print("✅ Model loaded successfully!")
        return True
    except Exception as e:
        print(f"⚠️  Model not found: {e}")
        print("Please run 'python walking_pattern_recognition.py' first to train the model.")
        return False

def initialize_tts():
    """Initialize text-to-speech engine"""
    global engine
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)
        print("✅ Text-to-speech initialized!")
    except Exception as e:
        print(f"⚠️  TTS initialization failed: {e}")

def speak_prompt(prompt, pattern):
    """Speak navigation prompt in separate thread"""
    if engine is None:
        return
    
    def _speak():
        try:
            config = timing_config.get(pattern, timing_config['normal'])
            engine.setProperty('rate', config['rate'])
            engine.say(prompt)
            engine.runAndWait()
        except:
            pass
    
    thread = threading.Thread(target=_speak)
    thread.daemon = True
    thread.start()

# Web interface HTML
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Real-Time Walking Pattern Detection</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: white;
            margin-bottom: 30px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .card h2 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.5em;
        }
        .value {
            font-size: 2.5em;
            font-weight: bold;
            color: #333;
            margin: 10px 0;
        }
        .sensor-data {
            font-family: 'Courier New', monospace;
            font-size: 1.1em;
            color: #555;
            line-height: 1.6;
        }
        .status {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            margin-top: 10px;
        }
        .status.active {
            background: #10b981;
            color: white;
        }
        .status.waiting {
            background: #f59e0b;
            color: white;
        }
        .confidence-bar {
            width: 100%;
            height: 30px;
            background: #e5e7eb;
            border-radius: 15px;
            overflow: hidden;
            margin-top: 10px;
        }
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #10b981, #059669);
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
        .instructions {
            background: rgba(255,255,255,0.95);
            border-radius: 15px;
            padding: 25px;
            margin-top: 20px;
        }
        .instructions h3 {
            color: #667eea;
            margin-bottom: 15px;
        }
        .instructions ol {
            margin-left: 20px;
            line-height: 1.8;
        }
        .instructions code {
            background: #f3f4f6;
            padding: 2px 8px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .pulse {
            animation: pulse 2s infinite;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚶 Real-Time Walking Pattern Detection</h1>
        
        <div class="dashboard">
            <div class="card">
                <h2>📊 Current Pattern</h2>
                <div class="value" id="pattern">Waiting...</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" id="confidence" style="width: 0%">0%</div>
                </div>
                <div class="status waiting pulse" id="status">Waiting for data</div>
            </div>
            
            <div class="card">
                <h2>📱 Sensor Data</h2>
                <div class="sensor-data">
                    <div><strong>Accelerometer:</strong></div>
                    <div id="accel">X: 0.00, Y: 0.00, Z: 0.00</div>
                    <div style="margin-top: 10px"><strong>Gyroscope:</strong></div>
                    <div id="gyro">X: 0.00, Y: 0.00, Z: 0.00</div>
                </div>
                <div style="margin-top: 15px; color: #6b7280;">
                    Buffer: <span id="buffer">0</span>/50
                </div>
            </div>
            
            <div class="card">
                <h2>🔊 Navigation Prompt</h2>
                <div class="sensor-data" id="prompt" style="font-size: 1.2em; color: #667eea;">
                    Waiting for pattern detection...
                </div>
                <div style="margin-top: 15px; color: #6b7280;">
                    Last update: <span id="timestamp">--:--:--</span>
                </div>
            </div>
        </div>
        
        <div class="instructions">
            <h3>📲 How to Connect Your Phone:</h3>
            <ol>
                <li>Download <strong>"Sensor Logger"</strong> app from Google Play Store</li>
                <li>Open the app and tap on <strong>"Stream"</strong> tab</li>
                <li>Enable <strong>"Push to web server"</strong></li>
                <li>Enter server URL: <code>http://{{ server_ip }}:5000/sensor</code></li>
                <li>Start streaming sensors (accelerometer + gyroscope)</li>
                <li>Walk around and watch the predictions update in real-time!</li>
            </ol>
            <p style="margin-top: 15px; color: #ef4444; font-weight: bold;">
                ⚠️ Make sure your phone and computer are on the same WiFi network!
            </p>
        </div>
    </div>
    
    <script>
        // Auto-refresh status every 500ms
        setInterval(async () => {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                document.getElementById('pattern').textContent = data.pattern;
                document.getElementById('confidence').style.width = (data.confidence * 100) + '%';
                document.getElementById('confidence').textContent = (data.confidence * 100).toFixed(1) + '%';
                document.getElementById('buffer').textContent = data.buffer_size;
                document.getElementById('timestamp').textContent = data.timestamp;
                
                if (data.accel) {
                    document.getElementById('accel').textContent = 
                        `X: ${data.accel[0]}, Y: ${data.accel[1]}, Z: ${data.accel[2]}`;
                    document.getElementById('gyro').textContent = 
                        `X: ${data.gyro[0]}, Y: ${data.gyro[1]}, Z: ${data.gyro[2]}`;
                }
                
                if (data.prompt) {
                    document.getElementById('prompt').textContent = data.prompt;
                }
                
                const statusEl = document.getElementById('status');
                if (data.buffer_size >= 50) {
                    statusEl.textContent = '🟢 Active - Predicting';
                    statusEl.className = 'status active';
                } else {
                    statusEl.textContent = `🟡 Buffering ${data.buffer_size}/50`;
                    statusEl.className = 'status waiting pulse';
                }
                
            } catch (error) {
                console.error('Error fetching status:', error);
            }
        }, 500);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Main dashboard"""
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    return render_template_string(HTML_TEMPLATE, server_ip=local_ip)

@app.route('/sensor', methods=['POST'])
def receive_sensor_data():
    """
    Receive sensor data from phone
    
    Expected JSON format:
    {
        "accelX": 0.5,
        "accelY": 9.8,
        "accelZ": 0.2,
        "gyroX": 0.1,
        "gyroY": -0.05,
        "gyroZ": 0.02
    }
    """
    global last_prediction
    
    try:
        data = request.get_json()
        
        # Extract sensor values (handle different field names)
        accel_x = data.get('accelX', data.get('accelerometerX', 0))
        accel_y = data.get('accelY', data.get('accelerometerY', 0))
        accel_z = data.get('accelZ', data.get('accelerometerZ', 0))
        gyro_x = data.get('gyroX', data.get('gyroscopeX', 0))
        gyro_y = data.get('gyroY', data.get('gyroscopeY', 0))
        gyro_z = data.get('gyroZ', data.get('gyroscopeZ', 0))
        
        # Add to buffer
        sensor_reading = [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
        sensor_buffer.append(sensor_reading)
        
        print(f"📡 Received: Buffer={len(sensor_buffer)}/50", end='\r')
        
        # If we have enough data, make prediction
        if len(sensor_buffer) == 50 and model is not None:
            # Prepare data for model
            X = np.array(list(sensor_buffer)).reshape(1, 50, 6)
            
            # Predict
            predictions = model.predict(X, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            # Map to pattern (adjust based on your label encoder)
            patterns = ['cane', 'fast', 'normal', 'slow']
            pattern = patterns[predicted_class]
            
            # Get appropriate prompt
            config = timing_config[pattern]
            prompt_list = prompts['detailed'] if config['detailed'] else prompts['standard']
            prompt = np.random.choice(prompt_list)
            
            # Update last prediction
            with prediction_lock:
                last_prediction = {
                    'pattern': pattern,
                    'confidence': confidence,
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'prompt': prompt,
                    'accel': [float(f"{accel_x:.2f}"), float(f"{accel_y:.2f}"), float(f"{accel_z:.2f}")],
                    'gyro': [float(f"{gyro_x:.2f}"), float(f"{gyro_y:.2f}"), float(f"{gyro_z:.2f}")]
                }
            
            print(f"\n🚶 Detected: {pattern.upper()} (Confidence: {confidence*100:.1f}%)")
            print(f"🔊 Prompt: {prompt}")
            
            # Speak the prompt
            speak_prompt(prompt, pattern)
            
            return jsonify({
                'status': 'success',
                'pattern': pattern,
                'confidence': confidence,
                'prompt': prompt
            })
        
        return jsonify({
            'status': 'buffering',
            'buffer_size': len(sensor_buffer)
        })
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current system status"""
    with prediction_lock:
        return jsonify({
            **last_prediction,
            'buffer_size': len(sensor_buffer),
            'model_loaded': model is not None
        })

if __name__ == '__main__':
    print("\n" + "="*70)
    print("  🌐 REAL-TIME WALKING PATTERN DETECTION SERVER")
    print("="*70)
    
    # Load model
    if not load_model():
        print("\n⚠️  WARNING: Model not loaded. Please train the model first:")
        print("   python walking_pattern_recognition.py")
        print("\nServer will still start, but predictions won't work.\n")
    
    # Initialize TTS
    initialize_tts()
    
    # Get local IP
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print(f"\n✅ Server Configuration:")
    print(f"   • Local IP: {local_ip}")
    print(f"   • Port: 5000")
    print(f"   • Model Status: {'Loaded ✓' if model else 'Not Loaded ⚠️'}")
    
    print(f"\n📱 Phone Setup:")
    print(f"   1. Install 'Sensor Logger' app (Android)")
    print(f"   2. Connect phone to same WiFi as this computer")
    print(f"   3. In app, set URL to: http://{local_ip}:5000/sensor")
    print(f"   4. Enable accelerometer + gyroscope streaming")
    
    print(f"\n🖥️  Web Dashboard:")
    print(f"   Open in browser: http://localhost:5000")
    print(f"   Or from phone: http://{local_ip}:5000")
    
    print("\n" + "="*70)
    print("  Server starting... Press Ctrl+C to stop")
    print("="*70 + "\n")
    
    # Start server
    app.run(host='0.0.0.0', port=5000, debug=False)