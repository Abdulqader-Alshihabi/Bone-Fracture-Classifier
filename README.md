# Bone Fracture Diagnostic Classifier — Medical AI Studio

[![Live Demo](https://img.shields.io/badge/Demo-Live_GitHub_Pages-0ea5e9?style=for-the-badge&logo=googlechrome&logoColor=white)](https://abdulqader-alshihabi.github.io/Bone-Fracture-Classifier/)
[![Colab](https://img.shields.io/badge/Notebook-Open_in_Colab-f59e0b?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/Abdulqader-Alshihabi/Bone-Fracture-Classifier/blob/main/Bone_Fracture_Classifier.ipynb)
[![Model](https://img.shields.io/badge/Backbone-EfficientNet--B0-14b8a6?style=for-the-badge&logo=pytorch&logoColor=white)](https://github.com/Abdulqader-Alshihabi/Bone-Fracture-Classifier)
[![Task](https://img.shields.io/badge/Task-12_Class_Classification-6366f1?style=for-the-badge)](https://github.com/Abdulqader-Alshihabi/Bone-Fracture-Classifier)
[![License](https://img.shields.io/badge/License-MIT-gray?style=for-the-badge)](LICENSE)

An open-source musculoskeletal computer vision system trained on clinical digital radiographs to classify and differentiate twelve distinct orthopedic fracture types. Features an interactive web-based PACS radiograph viewer with window-leveling, contrast inversion, and multi-class probability scoring.

---

## Clinical Scope & Radiological Context

Accurate categorization of orthopedic fractures from emergency digital X-rays is vital for determining whether non-operative casting, closed reduction, or open reduction and internal fixation (ORIF) surgery is indicated. 

This model investigates the diagnostic capacity of deep convolutional networks to learn morphological fracture geometry across 12 specific trauma classifications:

| Fracture Class | Radiological Definition & Trauma Mechanism |
| :--- | :--- |
| **Avulsion Fracture** | Fragment of cortical bone pulled away from main body by forceful tendon or ligament traction. |
| **Comminuted Fracture** | Bone shattered or splintered into three or more distinct fragments under high-energy trauma. |
| **Compression-Crush Fracture** | Cortical collapse and spongy bone impaction, commonly in osteoporotic vertebrae or calcaneus. |
| **Fracture Dislocation** | Co-occurring severe fracture with total loss of congruity and dislocation across the joint surface. |
| **Greenstick Fracture** | Pediatric incomplete bending fracture where the cortex breaks on the tension side while intact on compression. |
| **Hairline Fracture** | Non-displaced stress micro-fracture appearing as a fine, translucent cortical line. |
| **Impacted Fracture** | Fracture fragments forcibly driven directly into each other by severe axial compression. |
| **Intra-articular Fracture** | Fracture line directly breaches the articular hyaline cartilage joint space. |
| **Longitudinal Fracture** | Linear fracture plane extending parallel to the longitudinal anatomical axis of the bone shaft. |
| **Oblique Fracture** | Slanted fracture line traversing the bone cortex at an angle greater than 30 degrees to perpendicular. |
| **Pathological Fracture** | Structural cortical failure occurring under physiological stress due to pre-existing bone weakness or lesion. |
| **Spiral Fracture** | Helical fracture line winding around the cortical cylinder caused by severe rotational torque. |

---

## Neural Network Architecture

```
[ Input Radiograph (224 x 224 x 3) ]
                 |
                 v
+-------------------------------------------------+
|  EfficientNet-B0 Backbone (Pretrained ImageNet) |
|  - MBConv (Mobile Inverted Bottleneck Conv)     |
|  - Squeeze-and-Excitation (SE) Attention Blocks |
|  - Swish Activation (SiLU)                      |
|  - Compound Coefficient Scaling                 |
+-------------------------------------------------+
                 |
                 v  (1,280 feature channels)
+-------------------------------------------------+
|  Adaptive Global Average Pooling (1x1)          |
+-------------------------------------------------+
                 |
                 v
+-------------------------------------------------+
|  Classification Head                            |
|  - Dropout (p = 0.3)                            |
|  - Linear Projection (1,280 -> 12 Logits)       |
|  - Softmax Probability Distribution             |
+-------------------------------------------------+
                 |
                 v
[ 12-Class Differential Diagnosis Probabilities ]
```

---

## Benchmark Results & Evaluation Protocol

- **Dataset Protocol:** Held-out testing on 428 clinical radiograph images strictly partitioned to prevent data leakage.
- **Input Preprocessing:** Contrast-preserving resize, normalized to ImageNet channel statistics (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`).
- **Optimization:** AdamW optimizer with cosine learning rate scheduling and cross-entropy loss.
- **Inference Speed:** ~18ms per radiograph on standard desktop GPU, ~40ms on client-side CPU.

---

## Radiologist PACS Inspection Studio

The accompanying web viewer includes simulated Picture Archiving and Communication System (PACS) tools designed for clinical usability:
- **Instant 12-Case Benchmark:** Pre-loaded clinical X-rays representing all 12 fracture classes for immediate diagnostic evaluation.
- **Window Leveling & Brightness Control:** Interactive real-time slider adjusting radiograph brightness and dynamic range.
- **Contrast Inversion (Negative View):** One-click toggle switching between traditional white-bone radiograph and dark-bone soft-tissue view.
- **Differential Probabilities:** Bar chart visualization showing primary diagnosis and top differential alternatives.
- **Custom Upload:** Allows testing personal or academic DICOM/JPEG radiographs.

---

## Repository Structure

```
Bone-Fracture-Classifier/
├── docs/                               # PACS Radiograph Studio (GitHub Pages)
│   ├── index.html                      # Clinical viewer interface
│   ├── styles.css                      # Hospital-grade dark PACS styling
│   ├── app.js                          # Image manipulation, classification engine
│   ├── samples.json                    # Benchmark metadata and verified ground truths
│   ├── samples/                        # 12 representative clinical radiograph images
│   └── model/
│       └── fracture.onnx               # Exported EfficientNet ONNX model
├── Bone_Fracture_Classifier.ipynb      # Complete PyTorch training & evaluation notebook
└── README.md                           # Detailed technical report
```

---

## Quickstart & Local Setup

### 1. Launch the PACS Web Studio
```bash
git clone https://github.com/Abdulqader-Alshihabi/Bone-Fracture-Classifier.git
cd Bone-Fracture-Classifier
python -m http.server 8080 --directory docs
```
Open `http://localhost:8080` in your web browser.

### 2. Python Diagnostic Script
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

pip install torch torchvision onnxruntime pillow

python -c "
import onnxruntime as ort, numpy as np
from PIL import Image

sess = ort.InferenceSession('docs/model/fracture.onnx')
print('Diagnostic ONNX model loaded successfully. Inputs:', [i.name for i in sess.get_inputs()])
"
```

---

## Author & Engineering Credentials

**Abdulqader Alshihabi**  
Computer Science Graduate · Islamic University of Madinah (GPA: 4.62 / 5.00, First Class Honors)  
Software & AI Engineer specializing in Deep Learning, Medical Imaging, and Full-Stack Systems.

- **Portfolio:** [https://abdulqader-alshihabi.github.io/](https://abdulqader-alshihabi.github.io/)
- **GitHub:** [@Abdulqader-Alshihabi](https://github.com/Abdulqader-Alshihabi)
- **LinkedIn:** [https://www.linkedin.com/in/Abdulqader-Alshihabi/](https://www.linkedin.com/in/Abdulqader-Alshihabi/)
- **Email:** [abdulqader.moh.2003@gmail.com](mailto:abdulqader.moh.2003@gmail.com)

---

## Disclaimer & Notice

This project is developed solely for academic, educational, and computational vision research. It is **not** a certified medical diagnostic device and must not be used as a substitute for professional clinical radiological evaluation.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
