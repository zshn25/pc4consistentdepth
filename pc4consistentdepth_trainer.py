#
# COPYRIGHT (c) 2023 - Denso ADAS Engineering Services GmbH, MIT License
# Authors: Zeeshan Khan Suri (z.suri@eu.denso.com)
#
# monodepth2 wrapper code for the following paper:
# Pose Constraints for Self-supervised Monocular Depth and Ego-Motion (https://zshn25.github.io/pc4consistentdepth/)
#
# 1. Clone [monodepth2](https://github.com/nianticlabs/monodepth2)
# 2. Copy this file in the monodepth2 directory.
# 3. Edit train.py from `from trainer import Trainer` to `from pc4consistentdepth_trainer import Trainer`
#

import torch
from layers import get_translation_matrix, transformation_from_parameters
import trainer
from so3_utils import (so3_relative_angle, so3_rotation_angle)  # Pytorch3D functions


def invert_transformation(T):
    """Return inverted 4x4 SE3 transformation matrix
    """

    R = T[..., :3, :3]
    t = T[..., :-1, -1]

    T_inverted = torch.zeros_like(T)
    T_inverted[..., :3, :3] = R.transpose(-2, -1)
    T_inverted[..., :-1, -1] = -t
    T_inverted[..., -1, -1] = 1

    return T_inverted


class Trainer(trainer.Trainer):
    def __init__(self, options):
        super().__init__(options)

        if not hasattr(self.opt, "use_pose_consistency_loss"):
            # "cyclic", "identity", "reverse" (can also be multiple: "cyclic_reverse")
            self.opt.use_pose_consistency_loss = "cyclic"
        if not hasattr(self.opt, "pose_consistency_loss_weight"):
            self.opt.pose_consistency_loss_weight = 0.1  # weight for all pose constraints

        self.l2_loss = torch.nn.MSELoss(reduction='none')
        self.l1_loss = torch.nn.L1Loss(reduction='none')

    def predict_poses(self, inputs, features):
        # Collect poses for Pose consistency losses
        outputs = super().predict_poses(inputs, features)

        if self.num_pose_frames == 2:
            # Another pass through the pose network is ineffecient. You can skip the next 5 lines if you copy paste the rest function in monodepth2 code
            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            if self.opt.use_pose_consistency_loss:
                # Collect poses required for Pose Consistency Losses
                if "cycl" in self.opt.use_pose_consistency_loss:
                    # pose from (t-1 to t+1) should be same as (t-1 to t) and (t to t+1)
                    pose_inputs = [pose_feats[-1], pose_feats[1]]  # Inputs: (t-1 to t+1)
                    pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    outputs[("axisangle", -1, 1)], outputs[("translation", -1, 1)
                                                           ] = self.models["pose"](pose_inputs)  # pose from -1 to +1
                    outputs[("cam_T_cam", -1, 1)] = transformation_from_parameters(
                        outputs[("axisangle", -1, 1)][:, 0], outputs[("translation", -1, 1)][:, 0], invert=False)

                if "identity" in self.opt.use_pose_consistency_loss:
                    # Pose from frame t to t should be identity
                    pose_inputs = [pose_feats[0], pose_feats[0]]  # Inputs: (t to t)
                    pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    outputs[("axisangle", 0, 0)], outputs[("translation", 0, 0)] = self.models["pose"](pose_inputs)
                    outputs[("cam_T_cam", 0, 0)] = transformation_from_parameters(
                        outputs[("axisangle", 0, 0)][:, 0], outputs[("translation", 0, 0)][:, 0], invert=False)

                if "reverse" in self.opt.use_pose_consistency_loss:
                    # Pose from a to b should be inverse of pose from b to a
                    for f_i in self.opt.frame_ids[1:]:
                        # Pass frames in reversed temporal order
                        if f_i < 0:
                            pose_inputs = [pose_feats[0], pose_feats[f_i]]  # pose(t -> t-1) transforms points forward
                        else:
                            pose_inputs = [pose_feats[f_i], pose_feats[0]]  # pose(t+1 -> t) transforms points forward

                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]

                        axisangle, translation = self.models["pose"](pose_inputs)

                        # Invert the matrix if the frame id is positive
                        outputs[("invinv_cam_T_cam", 0, f_i)] = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=(f_i > 0))   # Pose from target to source

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    outputs[("cam_T_cam", f_i, 0)] = invert_transformation(outputs[("cam_T_cam", 0, f_i)])   # inv pose

            return outputs

    def compute_losses(self, inputs, outputs):
        losses = super().compute_losses(inputs, outputs)

        # Pose consistency losses. Only at one scale
        # Apply after some initial dry-run (same as in Can scale-consistent
        # monocular depth be learned in a self-supervised scale-invariant manner?)
        if self.opt.use_pose_consistency_loss and self.step > 2000:  # pose output is uni-scale. So, apply only at scale==0
            losses['consistency_loss/pose'] = torch.zeros_like(losses["loss"]).to(self.device)    # 0

            if "cycl" in self.opt.use_pose_consistency_loss:
                # Pose from (t-1 to t+1) should be same as (t-1 to t) and (t to t+1)
                points_T_points_tplus1_tminus1 = outputs[("cam_T_cam", -1, 1)]
                points_T_points_tplus1_t = outputs[("cam_T_cam", 0, 1)]
                points_T_points_t_minus1 = invert_transformation(outputs[("cam_T_cam", 0, 1)])

                # Calculate aggregate pose (t-1 to t+1)
                T_aggregate = points_T_points_t_minus1 @ points_T_points_tplus1_t  # T_t_tplus1@T_tminus1_t
                # Minimize pose deviations with the distance measure mentioned in the paper
                losses['consistency_loss/pose'] += self.l1_loss(T_aggregate[:,
                                                                :-1, -1], points_T_points_tplus1_tminus1[:, :-1, -1]).mean()  # translation
                # losses['consistency_loss/pose'] += self.l2_loss(T_aggregate[:, :3, :3],
                #                                                 points_T_points_tplus1_tminus1[:, :3, :3]).mean()  # rotation
                losses['consistency_loss/pose'] += (1. - so3_relative_angle(T_aggregate[:, :3, :3].T,
                                                                            points_T_points_tplus1_tminus1[:, :3, :3].T, cos_angle=True)).abs().mean()  # rotation
                del points_T_points_tplus1_tminus1, points_T_points_tplus1_t, points_T_points_t_minus1, T_aggregate

            if "identity" in self.opt.use_pose_consistency_loss and torch.rand() < 0.1:  # enough to apply this constraint sparsely to speed-up training
                # Pose from frame t to t should be identity. Minimize any residual pose
                losses['consistency_loss/pose'] += outputs[("translation", 0, 0)][:, 0].abs().mean()  # translation
                losses['consistency_loss/pose'] += so3_rotation_angle(outputs[("cam_T_cam", 0, 0)][:, :3, :3].T).abs().mean()  # rotation

            if "reverse" in self.opt.use_pose_consistency_loss:
                # Pose from a to b should be inverse of pose from b to a
                for f_i in self.opt.frame_ids[1:]:
                    # Minimize pose deviations with the distance measure mentioned in the paper
                    losses['consistency_loss/pose'] += self.l1_loss(outputs[("cam_T_cam", 0, f_i)][:, :-1, -1],
                                                                    outputs[("invinv_cam_T_cam", 0, f_i)][:, :-1, -1]).mean()  # translation
                    # losses['consistency_loss/pose'] += self.l2_loss(outputs[("cam_T_cam", 0, f_i)][:, :3, :3],
                    #                                                 outputs[("invinv_cam_T_cam", 0, f_i)][:, :3, :3]).mean()  # rotation
                    losses['consistency_loss/pose'] += (1. - so3_relative_angle(outputs[("cam_T_cam", 0, f_i)][:, :3, :3].T,
                                                                                outputs[("invinv_cam_T_cam", 0, f_i)][:, :3, :3].T, cos_angle=True)).abs().mean()  # rotation
            
            # default pose constraint weight = 0.1 as mentioned in the paper
            losses['consistency_loss/pose'] *= self.opt.pose_consistency_loss_weight
            losses["loss"] += losses['consistency_loss/pose']

        return losses
