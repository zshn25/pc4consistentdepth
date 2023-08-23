# pc4consistentdepth

This is the reference PyTorch implementation for training and testing temporally consistent depth estimation models using the method described in

> **Pose Constraints for Consistent Self-supervised Monocular Depth and Ego-motion**
>
> [Zeeshan Khan Suri](https://www.linkedin.com/in/zshn25/)
>
> [SCIA 2023 (arXiv pdf)](https://arxiv.org/pdf/2304.08916), [Video](https://youtu.be/AN1AGR85N2A), [Blog](https://zshn25.github.io/pc4consistentdepth/)

[kitti](https://zshn25.github.io/images/3dreco/out.gif)


## ⚙️ Requirements

Run `setup.sh`, which does the following steps respectively

1. Clone [monodepth2](https://github.com/nianticlabs/monodepth2).
2. Copy this file in the monodepth2 directory.
3. Edit train.py from `from trainer import Trainer` to `from pc4consistentdepth_trainer import Trainer`.

Then, you're ready to go. 

1. By default, uses cyclic consistency with 0.1 weight. You can change options by adding `use_pose_consistency_loss` and `pose_consistency_loss_weight` options or by editing `pc4consistentdepth_trainer.py`'s `__init__` function.
2. Follow instructions from [monodepth2](https://github.com/nianticlabs/monodepth2/blob/master/README.md).


## 👩‍⚖️ License

The `pc4consistentdepth_trainer.py` code is released under MIT License, Copyright © Zeeshan Khan Suri, Denso ADAS Engineering Services GmbH, 2023.

`so3_utils.py` is taken from [Pytorch3D](https://pytorch3d.org/), with it's respective [BSD-style license](https://github.com/facebookresearch/pytorch3d/blob/main/LICENSE).

___

If you find our work useful in your research please consider citing our paper:

```
@InProceedings{10.1007/978-3-031-31438-4_23,
author="Suri, Zeeshan Khan",
editor="Gade, Rikke and Felsberg, Michael and K{\"a}m{\"a}r{\"a}inen, Joni-Kristian",
title="Pose Constraints for Consistent Self-supervised Monocular Depth and Ego-Motion",
booktitle="Image Analysis",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="340--353",
isbn="978-3-031-31438-4",
doi={10.1007/978-3-031-31438-4_23}
}
```
