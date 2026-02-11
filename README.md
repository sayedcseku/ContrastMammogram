# Contrasting Multiple Mammogram Views for Breast Cancer Classification

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<img src="header.png" width="500" alt="Multi-view Mammogram Analysis">

## Abstract

Implementation of **Anatomy-aware Graph R-CNN (AGR-CNN)** for multi-view mammogram fusion in breast cancer classification. Building on "Act Like a Radiologist" by Liu et al., this work extends graph convolution beyond mass segmentation to **microcalcification detection** and incorporates temporal reasoning for longitudinal breast cancer screening.

**Authors**: Nasif Zaman, Md Abu Sayed
**Institution**: University of Nevada, Reno

## 🎯 Key Features

- **Multi-view Mammogram Fusion**: Integrates CC and MLO views using graph neural networks
- **Mass & Microcalcification Detection**: Extended AGN framework for both mass and calcification analysis
- **Temporal Reasoning**: Longitudinal analysis using prior and recent mammographic exams
- **Anatomical Graph Mapping**: 65 regions for CC views, 82 regions for MLO views based on pseudo-landmarks
- **Bilateral & Ipsilateral Reasoning**: Exploits rarity of bilateral malignancy for improved classification

## 🏗️ Architecture

The AGR-CNN follows the Anatomy-aware Graph Network (AGN) framework with key extensions:

### Graph Region Mapping
- **Pseudo-landmarks**: Generated using nipple and pectoral muscle localization
- **CC views**: Divided into **65 regions**, pectoral muscle appears consistently at image edge
- **MLO views**: Divided into **82 regions**, images rotated to detect nipple extremity and muscle line

### Graph Construction & Convolution
1. **Node Representation**: Each anatomical region becomes a graph node
2. **Edge Relationships**: Spatial relationships between regions form graph edges
3. **Two-layer GCN**: 
   - Downsample: 256 → 16 dimensions (compact encoding)
   - Upsample: 16 → 256 dimensions (feature reconstruction)
4. **Cross-view Correspondence**: Models examined, auxiliary, and contralateral views

### Calcification-Specific Adaptations
- **Spatial Voting Mechanism**: Regions at similar distances from nipple receive weighted votes
- **Geometric Correspondence**: Improves calcification detection without exhaustive search
- Unlike masses, calcifications lack size variance requiring specialized correspondence methods

## 📊 Performance

### Experimental Setup
- **Baseline Comparison**: Mask R-CNN and Faster R-CNN vs AGN variants
- **Evaluation Metric**: Recall @ t false positives per image (R@t), IOU > 0.2
- **Training**: 30 epochs, SGD optimizer with Nesterov momentum (0.9)

### Key Results
- **AG-RCNN significantly outperforms single-view baselines**
- **Multi-view fusion**: Improved lesion localization through cross-view correspondence  
- **Temporal Analysis**: Limited performance on extremely small calcifications requires higher-resolution inputs
- **CBIS-DDSM**: Effective on both mass and calcification detection (84 four-view calcification cases)

## 🛠️ Installation

### Prerequisites

- Python 3.7+
- CUDA 10.2+ (for GPU support)
- PyTorch 1.9+
- Detectron2

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/ContrastMammogram.git
   cd ContrastMammogram
   ```

2. **Install dependencies**
   ```bash
   chmod +x install.sh
   ./install.sh
   ```

3. **Manual dependency installation** (if install.sh fails)
   ```bash
   # Install PyTorch (adjust CUDA version as needed)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # Install Detectron2
   python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
   
   # Install additional requirements
   pip install pydicom pyyaml==5.1 torch-geometric torch-scatter torch-sparse
   ```

## 📁 Project Structure

```
ContrastMammogram/
├── src/
│   ├── model/
│   │   ├── agrcnn.py          # Main AGR-CNN implementation
│   │   └── linear.py          # Graph linear layers
│   └── utils/
│       ├── dataloader.py      # Data loading and preprocessing
│       ├── preprocessing.py   # Image preprocessing utilities
│       └── visualization.py   # Visualization tools
├── CBIS/                      # CBIS-DDSM dataset files
├── Temporal/                  # Temporal analysis utilities
├── train.py                   # Training script
├── MammoTrainer.py           # Custom trainer implementation
└── install.sh                # Installation script
```

## 🚀 Quick Start

### 1. Data Preparation

**For CBIS-DDSM Dataset:**
```bash
# Download CBIS-DDSM dataset from TCIA
# Ensure the following structure:
CBIS/
├── mass_case_description_train_set.csv
├── calc_case_description_train_set.csv  
├── mass_case_description_test_set.csv
├── calc_case_description_test_set.csv
├── train.json
└── test.json
```

**For Temporal Microcalcification Dataset:**
```bash
# Organize temporal mammogram data as:
Temporal/
├── Normal_cases/
├── Suspicious_cases/
├── patient_info_dict.json
├── detectron_regions_.json
└── [patient_folders with CC_prior.dcm, CC_recent.dcm, MLO_prior.dcm, MLO_recent.dcm]
```

### 2. Data Preprocessing

The system includes specialized preprocessing for multi-view analysis:
- **Aspect ratio preservation** via padding to avoid registration distortion
- **Architectural distortion augmentation** using lens-style distortions
- **Binary mask extraction** from color annotations where necessary

### Training

**Train AGR-CNN model:**
```bash
python train.py
```

**Training Configuration:**
- **Optimizer**: SGD with Nesterov momentum (0.9)
- **Learning Rate**: 0.02
- **Weight Decay**: 1e-4  
- **Epochs**: 30
- **Batch Size**: 3 images per GPU

**Experiments conducted:**
1. Mask R-CNN segmentation on CBIS-DDSM
2. Anatomy-aware Graph RCNN segmentation on CBIS-DDSM  
3. Graph-based pathology classification
4. Temporal calcification segmentation

### 3. Evaluation

The training script automatically evaluates the model using COCO metrics. Key metrics include:
- Average Precision (AP) at IoU 0.5:0.95
- Average Recall (AR) at IoU 0.5:0.95
- Class-specific performance for mass vs. calcification detection

## 🔧 Configuration

### Model Configuration

The AGR-CNN model configuration follows AGN framework:

```python
# Graph Region Settings
cfg.MODEL.NODE.CC = 65              # CC view regions
cfg.MODEL.NODE.MLO = 82             # MLO view regions  
cfg.MODEL.NODE.F = 256              # Node feature dimensions
cfg.MODEL.NODE.ENCODED_F = 16       # Compact encoding dimensions

# Training Configuration
cfg.SOLVER.BASE_LR = 0.02           # Learning rate
cfg.SOLVER.MAX_ITER = 2650          # Training iterations  
cfg.SOLVER.IMS_PER_BATCH = 3        # Batch size
cfg.SOLVER.MOMENTUM = 0.9           # Nesterov momentum
cfg.SOLVER.WEIGHT_DECAY = 1e-4      # Weight decay
```

### Data Configuration

```python
# Image preprocessing
cfg.INPUT.MIN_SIZE_TRAIN = (800, 800)
cfg.INPUT.MAX_SIZE_TRAIN = 800
cfg.INPUT.RANDOM_FLIP = "none"      # No flipping for mammograms
```

## 📈 Usage Examples

### Custom Dataset Training

```python
from src.model.agrcnn import AGRCNN_Trainer
from src.utils import dataloader

# Register custom dataset
parent_dir = "/path/to/your/dataset"
train_dict, _, _ = dataloader.registerCatalogs(parent_dir)

# Initialize trainer
trainer = AGRCNN_Trainer("custom_train", "custom_val", "/path/to/output")
trainer.train()
```

### Inference on New Images

```python
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

# Load trained model
cfg = get_cfg()
cfg.MODEL.WEIGHTS = "/path/to/trained/model.pth"
cfg.MODEL.META_ARCHITECTURE = "AGRCNN"

predictor = DefaultPredictor(cfg)

# Run inference
outputs = predictor(image)
```

## 📚 Dataset Information

### CBIS-DDSM (Curated Breast Imaging Subset of DDSM)
- **Size**: 2,620 scanned mammographic studies
- **Multi-view Support**: CC and MLO for each breast
- **Annotations**: Expert-annotated masses and calcifications
- **Calcifications**: Only **84 four-view calcification cases** were usable
- **Format**: DICOM with ROI annotations

### Temporal Microcalcification Dataset  
- **Structure**: Prior and recent mammograms for longitudinal analysis
- **Views**: CC_prior.dcm, CC_recent.dcm, MLO_prior.dcm, MLO_recent.dcm
- **Ground Truth**: Binary masks extracted from color annotations (CC_prior_GT.jpg, etc.)
- **Synthetic Data**: Contralateral views generated by flipping where necessary
- **Categories**: Normal and suspicious cases

### Data Format
Expected annotation format following COCO-style:
```json
{
  "image_id": "P_00001_CC_LEFT", 
  "annotations": [{
    "bbox": [x, y, width, height],
    "category_id": 0,
    "segmentation": [[x1, y1, ...]],
    "graph_examined_node": 45
  }]
}
```

## 🧪 Experimental Setup

### Training Details
- **Optimizer**: SGD with Nesterov momentum (0.9)
- **Learning Rate**: 0.02 
- **Epochs**: 30 epochs
- **Weight Decay**: 1e-4
- **Batch Size**: 3 images per GPU
- **Image Preprocessing**: Resized with aspect ratio preservation via padding

### Experiments Conducted
1. **Mask R-CNN segmentation** on CBIS-DDSM
2. **Anatomy-aware Graph RCNN** segmentation on CBIS-DDSM
3. **Graph-based pathology classification** 
4. **Temporal calcification segmentation** with longitudinal data

### Hardware Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **RAM**: 16GB+ system memory
- **Storage**: 50GB+ for datasets and model checkpoints

## 🔬 Research & Citation

This work extends the Anatomy-aware Graph Network (AGN) framework from Liu et al. to microcalcification detection and incorporates temporal reasoning for longitudinal breast cancer screening.


### Key References
- Liu et al., "Act Like a Radiologist", IEEE TPAMI, 2021
- Wu et al., IEEE TMI, 2019  
- Yang et al., "MommiNet-v2", Medical Image Analysis, 2021

### Future Work
- Comparison with MommiNet-v2
- Incorporation of reconstruction losses
- Improved temporal fusion using prior exams with differing views
- Higher-resolution inputs for small calcification detection

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`) 
3. Commit changes (`git commit -am 'Add feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## ⚠️ Disclaimer

This software is for research purposes only. Not intended for clinical diagnosis without proper validation and regulatory approval. Consult healthcare professionals for medical decisions.

## 🐛 Troubleshooting

**Common Issues:**
1. **CUDA out of memory**: Reduce batch size or use mixed precision training
2. **Dataset loading errors**: Verify data paths and DICOM file accessibility  
3. **Detectron2 installation**: Match CUDA versions between PyTorch and Detectron2

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **CBIS-DDSM Dataset**: Thanks to the creators and maintainers
- **Detectron2**: Facebook AI Research for the excellent framework  
- **PyTorch Geometric**: For graph neural network implementations

---

**Keywords**: Multimodal Mammography, Temporal Analysis, Contrast Mammogram, Graph Neural Networks, Medical Imaging, Breast Cancer Screening
