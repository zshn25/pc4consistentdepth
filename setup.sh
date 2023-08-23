#
# COPYRIGHT (c) 2023 - Denso ADAS Engineering Services GmbH, MIT License
# Authors: Zeeshan Khan Suri (z.suri@eu.denso.com)
#
# Setup to run the code from the paper:
# Pose Constraints for Self-supervised Monocular Depth and Ego-Motion (https://zshn25.github.io/pc4consistentdepth/)
#
# Clone monodepth2
git clone https://github.com/nianticlabs/monodepth2

# Copy the required files
cp pc4consistentdepth_trainer.py monodepth2
cp so3_utils.py monodepth2
cd monodepth2

# replace trainer with pc4consistentdepth_trainer
sed -i 's/trainer/pc4consistentdepth_trainer/g' train.py

# ready to go!