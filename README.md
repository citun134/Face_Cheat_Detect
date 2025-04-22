# 🛡️ Face Cheat Detection using Gaze and Anti-Spoofing

This project aims to detect face cheating and spoofing attempts using head pose estimation and gaze detection. It combines **L2CSNet** for gaze estimation and **Silent Face Anti-Spoofing** for liveness detection.

## 📂 Project Structure

```
Face_Rec/
├── cheat_detection.py          # Main script for cheat detection
├── Silent_Face_Anti_Spoofing/ # Anti-spoofing module
├── L2CSNet_gaze360.pkl                   # Gaze estimation model
├── ...
```

## 🚀 Features

- 🧠 **Head Pose Estimation**  
- 👁️ **Gaze Estimation using L2CSNet**  
- 🧬 **Liveness Detection using Silent-Face-Anti-Spoofing**  
- 🔍 **Cheating Detection based on gaze and spoofing cues**

---

## 🔧 Installation

1. **Clone the repository**
```bash
git clone https://github.com/citun134/Face_Cheat_Detect.git
cd Face_Cheat_Detect
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 📥 Pre-trained Model

Download the L2CSNet gaze estimation model (`L2CSNet_gaze360.pkl`) from the following link:

📎 [Download from Google Drive](https://drive.google.com/drive/folders/1qDzyzXO6iaYIMDJDSyfKeqBx8O74mF8s)

After downloading, place the `L2CSNet_gaze360.pkl` file in the appropriate folder (e.g. `L2CSNet/checkpoints/`).

---

## ▶️ Usage

Once all dependencies and the model are set up, you can run the detection script:

```bash
python Silent_Face_Anti_Spoofing/cheat_detection.py
```

---

## 📌 Notes

- Make sure your webcam or input video is properly configured.
- The gaze and head pose detection work best under good lighting conditions.
- The anti-spoofing module is trained on real and spoofed faces; results may vary based on input quality.

---

## 🙌 Acknowledgements

- [L2CSNet](https://github.com/Ahmednull/L2CS-Net.git)
- [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing.git)

---

## 📃 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
