
# Official Implementation for GTA (NeurIPS 2023)
> **Global-correlated 3D-decoupling Transformer for Clothed Avatar Reconstruction (NeurIPS 2023)** [[Paper](https://arxiv.org/abs/2309.13524)] [[Website](https://river-zhang.github.io/GTA-projectpage/)]

# News 
- **[2023/9/26]** We release the arXiv version ([Paper in arXiv](https://arxiv.org/abs/2309.13524)).

# TODO

- [ ] Hugging Face
- [ ] Release code
- [x] Release paper


# Introduction
Reconstructing 3D clothed human avatars from single images is a challenging task, especially when encountering complex poses and loose clothing. Current methods exhibit limitations in performance, largely attributable to their dependence on insufficient 2D image features and inconsistent query methods. Owing to this, we present the Global-correlated 3D-decoupling Transformer for clothed Avatar reconstruction (GTA), a novel transformer-based architecture that reconstructs clothed human avatars from monocular images. Our approach leverages transformer architectures by utilizing a Vision Transformer model as an encoder for capturing global-correlated image features. Subsequently, our innovative 3D-decoupling decoder employs cross-attention to decouple tri-plane features, using learnable embeddings as queries for cross-plane generation. To effectively enhance feature fusion with the tri-plane 3D feature and human body prior, we propose a hybrid prior fusion strategy combining spatial and prior-enhanced queries, leveraging the benefits of spatial localization and human body prior knowledge. Comprehensive experiments on CAPE and THuman2.0 datasets illustrate that our method outperforms state-of-the-art approaches in both geometry and texture reconstruction, exhibiting high robustness to challenging poses and loose clothing, and producing higher-resolution textures.

![framework](docs/src/model-structure_small.jpg)



# Bibtex
If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@misc{zhang2023globalcorrelated,
      title={Global-correlated 3D-decoupling Transformer for Clothed Avatar Reconstruction}, 
      author={Zechuan Zhang and Li Sun and Zongxin Yang and Ling Chen and Yi Yang},
      year={2023},
      eprint={2309.13524},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# Acknowledgement 
Our implementation is mainly based on [ICON](https://github.com/YuliangXiu/ICON) and many thanks to the following open-source projects:
* [ICON](https://github.com/YuliangXiu/ICON)
* [ECON](https://github.com/YuliangXiu/ECON)
* [PIFu](https://github.com/shunsukesaito/PIFu)

In addition, we sincerely thank [Yuliang Xiu](https://github.com/YuliangXiu), the author of [ICON](https://github.com/YuliangXiu/ICON) and [ECON](https://github.com/YuliangXiu/ECON) for resolving many of our concerns in GitHub Issues.

More related papers about 3D avatars: https://github.com/pansanity666/Awesome-Avatars
