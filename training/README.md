# Training

Scripts used to build and train the binary screening models (**Model A** and **Model A1**)
on PTB-XL. They run on a GPU (developed on Google Vertex AI, NVIDIA L4).

## Steps

1. **Prepare the labels** (reuses the shared PTB-XL arrays, just relabels):
   ```bash
   python prepare_dataset_modelA.py     # NORM vs MI+STTC
   python prepare_dataset_modelA1.py    # NORM vs MI+STTC+CD
   ```

2. **Train** the hybrid ensemble (ResNet+SimCLR / Inception / TCN, 5-fold OOF + pseudo-labeling):
   ```bash
   python train_binary_hybrid.py
   ```
   The same script trains **A** or **A1** — only edit at the top of the file:
   - `DATA_DIR`   → the dataset folder produced in step 1
   - `OUTPUT_DIR` → where to write the checkpoints / metrics

## Notes

- Paths inside the scripts point to the original Vertex AI environment
  (`/home/jupyter/...`) — adapt them to your machine.
- The PTB-XL arrays (`ptbxl_ecg.npy`, …) and the SimCLR pretrained encoder are **not**
  included in this repo (size / licensing). See the main
  [README](../README.md#model-weights--data).
- The multiclass model (Model B, MI/STTC/CD) reuses the same architectures with a
  3-class softmax head and transfer learning from the binary checkpoints.
