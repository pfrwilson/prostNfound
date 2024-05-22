# ProstNFound: Enhancing Foundation Models for Prostate Cancer Detection

ProstNFound is an innovative method designed to improve the accuracy of prostate cancer (PCa) detection using high-resolution micro-ultrasound data and deep learning techniques. By combining the robustness of medical foundation models with specialized domain-specific knowledge, ProstNFound demonstrates significant improvements in detecting PCa, offering performance competitive with expert radiologists.

## Overview

Medical foundation models, pre-trained on extensive and diverse datasets, provide a strong knowledge base that can be adapted to various downstream tasks. However, their generality often limits their effectiveness in specialized domains like PCa detection. ProstNFound addresses this challenge by integrating domain-specific knowledge into foundation models, specifically tailored for ultrasound imaging and PCa detection.

### Key Features

- **Domain-Specific Knowledge Integration**: Specialized auxiliary networks embed high-resolution textural features and clinical markers into the foundation model, enhancing its performance.
- **Improved Detection Accuracy**: ProstNFound achieves 90% sensitivity at 40% specificity, demonstrating significant improvements over state-of-the-art models.
- **Clinical Relevance**: Performance competitive with expert radiologists reading multi-parametric MRI or micro-ultrasound images, suggesting significant promise for clinical application.

## Methodology

![Method3](./.github/Method3.png)

The figure above illustrates the methodology of ProstNFound. 
ProstNFound integrates a B-mode image encoder with a conditional prompt module that embeds raw RF patch data (using a CNN patch encoder) and patient metadata. These embeddings are fed into a mask decoder to generate a cancer likelihood heatmap. The training process uses histopathology labels in the needle region. The patch encoder is initialized through self-supervised pretraining, while the image encoder and mask decoder are initialized from a medical image foundation model.

## Dataset

ProstNFound was evaluated using a multi-center micro-ultrasound dataset comprising 693 patients. This diverse dataset provided a robust basis for training and testing the model, ensuring its generalizability across different clinical settings.

## Results

ProstNFound's integration of specialized auxiliary networks and foundation models has led to:

- **90% Sensitivity at 40% Specificity**: Indicating a high detection rate of prostate cancer with a reasonable level of false positives.
- **Comparable Performance to Expert Radiologists**: Demonstrating the potential for ProstNFound to assist or even outperform human experts in clinical settings.

![Heatmap Predictions](./.github/heatmap_predictions.png)

This figure shows a demo of the model outputs, where the model activations localize supicious regions of cancer (right column) confirmed by histopathology (Gleason grade), while showing little to no activations for benign examples (left column).

## Installation

To install and use ProstNFound, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/ProstNFound.git
    cd ProstNFound
    ```

2. Create a virtual environment with Python 3.11 or higher:
    ```bash
    conda create --name myEnv --python=3.11
    ```

3. Activate the virtual environment:
    ```bash
    conda activate myEnv
    ```

4. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

5. Install torch=2.0+ from their website: https://pytorch.org/get-started/locally/

## Usage - Details TBD

To use ProstNFound for PCa detection, follow these steps:

1. Prepare your micro-ultrasound data according to the dataset format specified in the documentation (see the data folder for more details)

2. (Optional - only required for using the patch CNN prompts): Run the self-supervised training stage for the CNN: 
    ```bash
    FOLD=$SLURM_ARRAY_TASK_ID
    N_FOLDS=10
    SPLITS_PATH=splits/ssl_fold${FOLD}:${N_FOLDS}.json
    DATA_TYPE=rf
    CHECKPOINT_PATH=/checkpoint/$USER/$SLURM_JOB_ID/checkpoint.pt

    srun python train_patch_ssl.py \
        --splits_file $SPLITS_PATH \
        --batch_size 64 \
        --lr 1e-4 \
        --data_type $DATA_TYPE \
        --name patch_ssl_${CENTER}_${DATA_TYPE}_${VAL_SEED} \
        --checkpoint_path=$CHECKPOINT_PATH \
        --save_weights_path=ssl_checkpoints/fold${FOLD}:${N_FOLDS}_rf_ssl_weights.pt
    ```

3. Run the training script to fine-tune the foundation model with domain-specific knowledge (see `scripts/run_main_training.sh` for an example)

3. Use the trained model for inference (see `scripts/test_prostnfound.sh` for an example)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or support, please open an issue in this repository or contact us at [pfrwilson@gmail.com](mailto:pfrwilson@gmail.com).

---

We hope ProstNFound will significantly enhance the capabilities of medical foundation models in the detection of prostate cancer, ultimately improving clinical outcomes. Thank you for your interest and contribution!