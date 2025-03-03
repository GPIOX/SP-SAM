# SP-SAM: Semantic Information-Driven Self-Prompt SAM for Underwater Image Semantic Segmentation

The paper is currently under peer review. A portion of the code has been made publicly available. Upon acceptance, we will release the complete codebase and the corresponding training dataset.

## Abstract
Underwater Image Semantic Segmentation (UISS) is a critical pre-processing task in the marine intelligent perception system. While the Segment Anything Model (SAM) demonstrates remarkable segmentation versatility, its dependency on manual prompts and limited robustness to underwater degradation hinders autonomous deployment. 
We propose SP-SAM, a novel automated adaptation framework incorporating three key innovations: 1) A convolutional adapter injecting spatial inductive bias into ViT-based encoding, enhancing feature extraction in turbid underwater environments; 2) A semantic prompt generator extracting multi-scale representations from ViT features to guide biological target localization; 3) An adaptive frequency encoder employing learnable high-pass filters to recover edge details through high-frequency feature extraction. These components establish a self-prompting mechanism where semantic guidance and implicit edge cues are autonomously derived from input images. Comprehensive evaluations across four UISS benchmarks demonstrate state-of-the-art performance with only 3M additional parameters, achieving real-time inference speeds crucial for underwater robotics. The framework's balanced efficiency and precision advance practical applications in embodied intelligence systems and marine biodiversity monitoring.


## Installation 
```bash
pip install -r requirements.txt
```

## Prepare
Download the dataset from [here](Fantastic Animals and Where to Find Them Segment Any Marine Animal with Dual SAM) and move in to the dir: ./data


## Train
```bash
python train --config_path config/UFO120_vit_b.yaml
```

## Test
```bash
python train --config_path config/UFO120_vit_b.yaml
```