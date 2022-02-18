# IGD
This repo contains the Pytorch implementation of our paper:
> [**Deep One-Class Classification via Interpolated Gaussian Descriptor**](https://arxiv.org/pdf/2101.10043.pdf)
>
> Yuanhong Chen*, [Yu Tian*](https://yutianyt.com/), [Guansong Pang](https://sites.google.com/site/gspangsite/home?authuser=0), [Gustavo Carneiro](https://cs.adelaide.edu.au/~carneiro/).

- **Accepted at AAAI 2022 (Oral).**  

## Dataset

[**Please download the MVTec AD dataset**](https://www.mvtec.com/company/research/datasets/mvtec-ad)

## Train and Test IGD 
After the setup, simply run the following command to train/test the global/local model: 
```shell
./job.sh
```


## Citation

If you find this repo useful for your research, please consider citing our paper:

```bibtex
@misc{chen2021deep,
      title={Deep One-Class Classification via Interpolated Gaussian Descriptor}, 
      author={Yuanhong Chen and Yu Tian and Guansong Pang and Gustavo Carneiro},
      year={2021},
      eprint={2101.10043},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
---
