# HYDRA-FL Experiments

This repository contains three versions of the HYDRA-FL (Hybrid Knowledge Distillation for Robust and Accurate Federated Learning) framework:

## 📁 Repository Structure

- `cuda-version/` – CUDA-enabled version (for Google Colab, GPU machines)
- `non-cuda-version/` – CPU/Mac-compatible version (no CUDA)
- `Rakib-version/` – Customized version contributed by Rakib, including experiment modifications and additional metrics

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

## 🧪 Rakib Version

This version contains modifications for advanced experimentation, such as:
- Attack indicator computation
- Gradient trimming
- Dynamic loss adjustments

To run:
```
cd Rakib-version
python main.py --alg moon --model cnn --dataset mnist --device cuda ...
```

## 📊 Replication Result (MNIST)

| Round | Accuracy |
|-------|----------|
| 0     | 0.737    |
| 10    | 0.975    |
| 19    | 0.981    |

Final Accuracy:  98.1%

## 📁 Experiment Artifacts: MOON Accuracy

-  `moon_accuracy_output.csv`: Tabular output of round-wise accuracy
- 📈 `moon_static_accuracy_plot.png`: Accuracy vs Communication Round plot

This data corresponds to the MOON algorithm run on MNIST with:
```
python main.py \
  --alg moon \
  --model cnn \
  --dataset mnist \
  --n_parties 10 \
  --batch-size 64 \
  --comm_round 20 \
  --local_max_epoch 5 \
  --mu 1.0 \
  --lr 0.01 \
  --device cuda
```

Final accuracy achieved: 98.1%

**Summary:** The MOON model achieved a mean accuracy of approximately 96.7% across 20 communication rounds. The attack indicator function confirmed that all rounds had zero adversarial drift (Aₜ = 0.0), indicating robust training. Test accuracy steadily improved from ~88% in early rounds to 97.7%+ in later rounds, demonstrating strong generalization and consistent performance across clients (10 clients × 367,418 model parameters).

## 🔗 View the Colab Notebook

To reproduce the MOON experiment in Colab, click here: [Open in Colab](https://colab.research.google.com/drive/1lWY-5y9GJxdGMRoU8uHncO3YPmak6EZh?usp=sharing)
