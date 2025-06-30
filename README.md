# CattleIDNet 🐄

An AI-powered cow face detection and identification system using YOLOv8 and a hybrid EfficientNet + Transformer model for livestock management.

## Features
- YOLOv8-based face detection
- Hybrid EfficientNet + Transformer classification
- Known/unknown class filtering
- Ready for deployment in app/web environments

## Project Structure
- `models/`: Trained models
- `src/`: All Python source code (detection, identification, preprocessing)
- `app/`: Web or mobile app interface
- `data/`: Raw and processed image data
- `notebooks/`: Training experiments
- `results/`: Evaluation reports
- `assets/`: Sample images or diagrams

## How to Run

```bash
pip install -r requirements.txt
python src/detect.py --img sample.jpg
