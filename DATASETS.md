# Datasets

This project trains and evaluates on three public datasets of real and
AI-generated video. **None of them are redistributed in this repository.** To
reproduce or extend the work, obtain each dataset from its original source and
comply with its license **and** with the terms of the underlying generators
whose outputs appear in the data (e.g. **Sora**, **KLing**, **Pika**). This
obligation is part of why the trained weights are released under a
**non-commercial** (CC BY-NC 4.0) license.

The exact splits, support/query manifests, and cached frame tensors used for the
reported runs are likewise **not** shipped — this repository represents the
method and results, it is not a turnkey reproduction package.

Per-domain record counts used for the `run6` results (for context only):
AEGIS total 436 (train 50 / val 50 / test 336); DVF total 1004 (200 / 200 / 604);
GenVideo total 2971 (200 / 200 / 2571).

---

## DVF — Diffusion Video Forensics (from MM-Det)

- **Source:** [github.com/SparkleXFantasy/MM-Det](https://github.com/SparkleXFantasy/MM-Det)
- **Dataset (HF):** [huggingface.co/datasets/sparklexfantasy/DVF](https://huggingface.co/datasets/sparklexfantasy/DVF)
- **Paper:** [arXiv:2410.23623](https://arxiv.org/abs/2410.23623)
- **License note:** Released as part of the MM-Det project (NeurIPS 2024). Use is
  subject to the terms stated in the MM-Det repository and the licenses of the
  diffusion generators whose outputs it contains. Confirm before redistribution.

```bibtex
@inproceedings{song2024mmdet,
  title     = {On Learning Multi-Modal Forgery Representation for Diffusion Generated Video Detection},
  author    = {Song, Xiufeng and Guo, Xiao and Zhang, Jiache and Li, Qirui and Bai, Lei and Liu, Xiaoming and Zhai, Guangtao and Liu, Xiaohong},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024},
  note      = {arXiv:2410.23623; introduces the Diffusion Video Forensics (DVF) dataset}
}
```

## GenVideo / GenVideo-100K (from DeMamba)

- **Source:** [github.com/chenhaoxing/DeMamba](https://github.com/chenhaoxing/DeMamba)
- **Paper:** [arXiv:2405.19707](https://arxiv.org/abs/2405.19707)
- **License note:** GenVideo is released with the DeMamba benchmark. The
  lightweight **GenVideo-100K** subset was used here. Use is subject to the
  DeMamba repository terms and the licenses of the underlying generators.

```bibtex
@article{chen2024demamba,
  title   = {DeMamba: AI-Generated Video Detection on Million-Scale GenVideo Benchmark},
  author  = {Chen, Haoxing and Hong, Yan and Huang, Zizheng and Xu, Zhuoer and Gu, Zhangxuan and Li, Yaohui and Lan, Jun and Zhu, Huijia and Zhang, Jianfu and Wang, Weiqiang and Li, Huaxiong},
  journal = {arXiv preprint arXiv:2405.19707},
  year    = {2024}
}
```

## AEGIS

- **Source (HF):** [huggingface.co/datasets/Clarifiedfish/AEGIS](https://huggingface.co/datasets/Clarifiedfish/AEGIS)
- **Paper:** [arXiv:2508.10771](https://arxiv.org/abs/2508.10771) · [ACM DL](https://dl.acm.org/doi/10.1145/3746027.3758295)
- **License note:** AEGIS (ACM MM 2025). The Hugging Face dataset page does not
  expose a license tag; confirm the terms from the paper / authors before use or
  redistribution. Contains outputs from modern video generators (e.g. Sora,
  KLing, Pika), whose own terms also apply.

```bibtex
@inproceedings{li2025aegis,
  title     = {AEGIS: Authenticity Evaluation Benchmark for AI-Generated Video Sequences},
  author    = {Li, Jieyu and Zhang, Xin and Zhou, Joey Tianyi},
  booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia (ACM MM)},
  year      = {2025},
  doi       = {10.1145/3746027.3758295},
  note      = {arXiv:2508.10771}
}
```

---

> **Downstream obligation.** Anyone using these datasets — or the model trained
> on them — must independently comply with each dataset's terms and with the
> terms of the underlying generators (Sora / KLing / Pika and others).
