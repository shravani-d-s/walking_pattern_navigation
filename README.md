# 🚶 Walking Pattern Recognition & Adaptive Navigation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)
![License](https://img.shields.io/badge/License-MIT-green)

**Deep learning system that dynamically adjusts navigation voice prompts based on individual walking patterns**

## 🎯 Project Overview

This system uses CNN-LSTM neural networks to recognize walking patterns from smartphone sensors and adapts voice navigation in real-time:

| Pattern | Speech Rate | Delay | Prompt Style |
|---------|-------------|-------|-------------|
| **Normal** | 150 wpm | 3s | Standard |
| **Slow** | 130 wpm | 5s | Detailed + warnings |
| **Fast** | 170 wpm | 2s | Quick, concise |
| **Cane** | 120 wpm | 6s | Very detailed + accessibility |

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/shravani-d-s/walking-pattern-recognition.git
cd walking-pattern-recognition
pip install -r requirements.txt
