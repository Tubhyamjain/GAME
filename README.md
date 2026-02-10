# Vehicle Speed & Tracking Analyzer

A Streamlit app that analyzes uploaded videos to:

- detect and track moving vehicles (car, bike, bus, truck, etc.)
- draw dynamic bounding boxes that follow each vehicle
- estimate per-vehicle speed in km/h with perspective-aware calibration
- produce a live event log with vehicle type + speed range
- optionally detect number plates and run OCR

## Highlights

- **Works with arbitrary FPS videos**: speed is derived from true timestamps.
- **Road orientation agnostic**: supports horizontal/vertical/angled roads by using world-coordinate mapping through homography.
- **Dynamic speed estimation**: rolling speed estimate with smoothing + robust outlier clipping.
- **Extensible pipeline**: modular detector, tracker, speed estimator, and plate OCR stages.

## Tech Stack

- Python 3.10+
- OpenCV
- Ultralytics YOLOv8 (detection + tracking)
- NumPy, Pandas
- EasyOCR (optional number plate OCR)
- Streamlit UI

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the URL printed by Streamlit and upload a video.

## Calibration for Better Speed Accuracy

The app uses a **meters-per-pixel** fallback and supports a more accurate **4-point perspective calibration**:

1. Provide 4 image points from the road plane (clockwise order).
2. Provide the real-world coordinates (meters) for those 4 points.
3. Speed estimation then happens in world space using homography, improving robustness on perspective views.

If calibration points are not provided, the app falls back to scalar meters-per-pixel speed estimation.

## Notes

- Plate OCR quality depends on video resolution and plate visibility.
- For best results, use clear daylight videos and adjust confidence threshold.
- First-time run may download YOLO weights.
