# SSD4Rec

It is a novel generic and efficient sequential recommendation backbone, which explores the seamless adaptation of **Mamba** for sequential recommendations. Specifically, SSD4Rec marks the variable- and long-length item sequences with sequence registers and processes the item representations with bidirectional Structured State Space Duality (SSD) blocks. This not only allows for hardware-aware matrix multiplication but also empowers outstanding capabilities in **variable-length and long-range sequence modeling**.

> **SSD4Rec: A Structured State Space Duality Model for Efficient Sequential Recommendation**\
> Haohao Qu $$\dagger$$, Yifeng Zhang $$\dagger$$, Liangbo Ning, Wenqi Fan*, Qing Li.\
> $$\dagger$$ The authors contribute equally to this paper.\
> \* Corresponding author.\
> Paper: https://arxiv.org/abs/2409.01192

## Usage

### Enviroment Requirement
You can refer to the required environment specifications in environment.yaml.

### Implement

An simple example to run SSD4Rec on the ML1M (Default) dataset:

```shell
python main.py
```

We provide four processed datasets: ML1M, Amazon-Beauty, Amazon-Games, and KuaiRand-Pure. If you want to run experiments on other datasets, you should go to `config.yaml` and replace the variable of `dataset` correspondly.


## Citation

```bibtex
@article{qu2024ssd4rec,
  title={Ssd4rec: a structured state space duality model for efficient sequential recommendation},
  author={Qu, Haohao and Zhang, Yifeng and Ning, Liangbo and Fan, Wenqi and Li, Qing},
  journal={arXiv preprint arXiv:2409.01192},
  year={2024}
}
```


## Acknowledgment

This project is based on [Mamba](https://github.com/state-spaces/mamba) and [RecBole](https://github.com/RUCAIBox/RecBole). Thanks for their excellent works.

More updates will be posed in the near future! Thank you for your interest.

