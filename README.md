
# Structure–Color Disentangled Anime Hair Synthesis via Flow Line

**SIGGRAPH Asia 2026**

Tianyu Zhang, I-Chao Shen, Haoran Xie

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://zty0304.github.io/FlowLines-web/)
[![Paper](https://img.shields.io/badge/Paper-ACM-red)](https://doi.org/10.1145/3829340.3842227)
![SIGGRAPH Asia](https://img.shields.io/badge/SIGGRAPH%20Asia-2026-orange)

![Teaser](assets/teaser.png)

## Overview

This is the official repository for our SIGGRAPH Asia 2026 paper,
**Structure–Color Disentangled Anime Hair Synthesis via Flow Line**.
We introduce **Flow Line**, a sparse and intuitive representation
that captures the fine-grained structure and color intent of anime
hairstyles.
Our framework enables independent control of hair structure and
color, supporting flexible hairstyle synthesis, transfer,
and manipulation.

## Release Status

- [ ] Dataset
- [ ] Inference code
- [ ] Pretrained models
- [ ] Training code

## Dataset

We introduce the **Flow Line Dataset**, a dataset designed for
flow line-guided anime hair synthesis and flow line extraction.

The dataset used in our paper consists of two subsets:

- **Hair Synthesis Dataset**: paired character images before
  and after hairstyle synthesis, together with flow line annotations.
- **Flow Line Extraction Dataset**: anime hairstyle images
  paired with corresponding flow line annotations.

### Dataset Statistics

| Subset | Training | Testing | Total |
| --- | ---: | ---: | ---: |
| Hair Synthesis | 898 | 167 | 1,065 |
| Flow Line Extraction | 973 | 200 | 1,173 |

All images are processed at a resolution of 1024 × 1024.

The statistics above describe the datasets used in the paper.
Please refer to the release notes for the contents of each
publicly available version.

### Dataset Preview

![Dataset Examples](assets/dataset_examples.png)

### Download

| Dataset | Download |
| --- | --- |
| Hair Synthesis | Coming soon |
| Flow Line Extraction | Coming soon |

### Data Organization

Please refer to [Dataset Documentation](dataset/README.md)
for the directory structure, annotation format, data splits,
and instructions for using the dataset.

### Data Sources and Usage

The dataset includes character imagery, hairstyle reference
materials, and flow line annotations.

Flow line annotations were created specifically for this
research by consenting participants.

Please refer to the dataset documentation for detailed
information about data provenance, released components,
and applicable usage restrictions.

Third-party materials remain subject to their respective
rights and licenses.

## Applications

Our framework supports various anime hairstyle synthesis
and manipulation tasks:

- Structure-guided hairstyle synthesis
- Color-guided hairstyle synthesis
- Independent structure and color manipulation
- Hairstyle transfer
- Flow line extraction

![Applications](assets/applications.png)

## License

Please refer to [DATASET_LICENSE.md](DATASET_LICENSE.md)
for the terms applicable to the released dataset.

Code and model weights, if released, will be accompanied
by their respective licenses.

## Citation

If you find our work or dataset useful in your research,
please consider citing our paper:

```bibtex
@article{zhang2026flowline,
  title={Structure--Color Disentangled Anime Hair Synthesis via Flow Line},
  author={Zhang, Tianyu and Shen, I-Chao and Xie, Haoran},
  journal={ACM Transactions on Graphics},
  year={2026},
  doi={10.1145/3829340.3842227}
}
```

## Acknowledgments

We thank all participants who contributed to the creation
of our flow line annotations.

## Contact

For questions regarding the paper or dataset, please
open an issue in this repository.
