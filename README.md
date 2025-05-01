# HYDRA-FL Experiments

This repository contains two versions of the HYDRA-FL (Hybrid Knowledge Distillation for Robust and Accurate Federated Learning) framework:

## 📁 Repository Structure

- `cuda-version/` – CUDA-enabled version (for Google Colab, GPU machines)
- `non-cuda-version/` – CPU/Mac-compatible version (no CUDA)

## 🚀 CUDA Version

Use this on Colab or any GPU system:
```
python main.py --alg moon --model cnn --dataset mnist --device cuda ...
```

## 🍏 Non-CUDA Version

Use this on Mac/M1 or CPU-only systems:
```
python main.py --alg moon --model cnn --dataset mnist --device cpu ...
```

## 📊 Replication Result (MNIST)

| Round | Accuracy |
|-------|----------|
| 0     | 0.737    |
| 10    | 0.975    |
| 19    | 0.981    |

Final Accuracy: ✅ 98.1%

## 📌 Notes

- Base repo: https://github.com/momin-ahmad-khan/HYDRA-FL
- CUDA version tested in Colab (Tesla T4)
- CPU version tested on Mac M2
