# NPP-People-Tracking

A multi-object tracking framework featuring YOLOv8 detection, DeepSORT-based association, and three evolutionary optimizers (QPSO, NSGA-II, MOPSO). Designed for robust benchmarking, automatic tuning, and performance visualization across multiple feature extractors and tracking configurations.

---

## 🚀 Features

* **Track-by-detection framework** using YOLOv8 + DeepSORT
* **GPU-accelerated inference** via PyTorch & Ultralytics
* **Three evolutionary optimizers:**

  * QPSO (Quantized Particle Swarm Optimization)
  * NSGA-II (Non-dominated Sorting Genetic Algorithm)
  * MOPSO (Multi-objective PSO)
* **Three feature extractors (embedders):**

  * MobileNetV2
  * ShuffleNetV2
  * ResNet
* **Automatic batch optimization** across all 3×3 (embedder × optimizer) combinations
* **Mock MOTA, IDF1, and FPS evaluation** for fast prototyping
* **Pareto front visualizations** (2D and 3D)
* **Modular codebase** for easy extension & full GPU compatibility

---

## 📁 Folder Structure

```
NPP-People-Tracking/
├── detectors/               # YOLOv8 detection wrapper
├── embeddings/              # Feature extractors
├── evaluation/              # Evaluation logic
├── optimization/            # QPSO, NSGA, MOPSO + plotting
├── results/                 # Logs, metrics, plots
├── sample_videos/          # Input test videos
├── trackers/                # DeepSORT / ByteTrack wrappers
├── utils/                   # Drawing and utility helpers
├── main.py                  # Visual demo runner
├── batch_runner.py          # Full automation across all configs
├── run_plot_pareto.py       # Manual plotting script
├── config.yaml              # Tuning bounds for all optimizers
```

---

## ⚙️ Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download YOLOv8 Weights

```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

Place it in the root directory as `yolov8n.pt`

### 3. Add a test video

Place your input video in `sample_videos/test_video.mp4`

---

## 🧪 Running the System

### 🔁 Run Full Batch (All 3×3 Combinations)

```bash
python batch_runner.py
```

This will:

* Run QPSO, NSGA-II, and MOPSO for MobileNetV2, ShuffleNetV2, and ResNet
* Save logs, metrics, plots
* Output a combined summary CSV at `results/summary_table.csv`

### ▶️ Run Demo Tracker

```bash
python main.py
```

Visualize tracking with live FPS overlay and bounding boxes.

### 📊 Generate Plots Manually

```bash
python run_plot_pareto.py
```

---

## 📈 Output Artifacts

* `metrics.json`: Best config + result for each combo
* `log.csv`: All 50 generations of evaluations
* `*_2d.png`, `*_3d.png`: Saved Pareto plots (FPS vs MOTA, FPS vs MOTA vs IDF1)
* `summary_table.csv`: Global table across all runs

---

## 🧠 Technical Notes

* All tracking logic and evaluation runs on **GPU** by default
* Mock metrics are used for rapid testing, customizable in `evaluation.py`
* Optimizers operate over bounds defined in `optimization/config.yaml`
* Supports future upgrades for real GT-based evaluation using `py-motmetrics`

---

## 🤝 Contributors

* [Abhinav Shukla](https://github.com/AbhinavAI)
  Project design, optimization logic, performance benchmarking


