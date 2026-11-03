# import atexit
import os

# from operator import is_
# import line_profiler

# import config as cfg
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as torch_f
# from .base_loss import LossBase, LossConfig, loss_builder
# from ...utils.path import PACKAGE_EXTERNAL_DIR
# from pytorch3d.transforms import axis_angle_to_matrix

# profile = line_profiler.LineProfiler()
# profile.print_stats(output_unit=1)
# atexit.register(profile.print_stats)


def plot_hull(theta, hull):
    del_rdp_hull = hull.detach().cpu().numpy()
    theta = theta.detach().cpu().numpy()

    fig = plt.figure()
    figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()

    ax = fig.add_subplot(111)
    ax.scatter(theta[:, 0], theta[:, 1], s=10, c="r")
    ax.set_xlabel("flexion")
    ax.set_ylabel("abduction")

    plt.plot(del_rdp_hull[:, 0], del_rdp_hull[:, 1], "-yo", linewidth=2)

    plt.xticks(np.arange(-3, 4, 1))
    plt.yticks(np.arange(-2, 2, 0.5))
    plt.show()


def two_norm(a):
    """

    Args:
        a: B*M*2 or B*M*3

    Returns:

    """
    return torch.norm(a, dim=-1)


def one_norm(a):
    """

    Args:
        a: B*M*2 or B*M*3

    Returns:

    """
    return torch.norm(a, dim=-1, p=1)


# @profile
def calculate_joint_angle_loss(thetas, hulls, verts):
    """

    Args:
        Theta: B, 15, 2
        hulls: 15, M, 2

    Returns:

    """

    loss = torch.zeros(thetas.shape[0]).cuda()
    v = verts.unsqueeze(0)  # (1, 15, 25, 2)
    w = -hulls.unsqueeze(0) + thetas.unsqueeze(2)  # (B, 15, M, 2)
    cross_product_2d = w[:, :, :, 0] * v[:, :, :, 1] - w[:, :, :, 1] * v[:, :, :, 0]
    tmp = torch.sum(cross_product_2d < 1e-6, dim=-1)
    is_outside = tmp != hulls.shape[1]
    if not torch.sum(is_outside):
        loss = torch.zeros(thetas.shape[0]).cuda()
    else:
        loss = torch.zeros(thetas.shape[0]).cuda()
        is_outside_batch = is_outside.sum(1) != 0  # (B)
        # is_outside_batch = torch.ones(thetas.shape[0]).bool().cuda()
        is_outside = is_outside[is_outside_batch]  # (B, 15) -> (K, 15)

        is_outside_joint = is_outside.sum(0) != 0  # (K)
        is_outside = is_outside[:, is_outside_joint]  # (K, J)

        outside_theta = (
            thetas[is_outside_batch][:, is_outside_joint]
            .unsqueeze(2)
            .repeat(
                1, 1, hulls.shape[1], 1
            )  # (K, 15, 25, 2)    .repeat(1, hulls.shape[1], 1)
        )
        w_outside = w[is_outside_batch][:, is_outside_joint]  # (K, K, M, 2)
        v = v[:, is_outside_joint]  # (1, 15, 25, 2) -> (1, J, 25, 2)
        t = (
            torch.clamp(inner_product(w_outside, v) / (two_norm(v) ** 2), min=0, max=1)
            .unsqueeze(-1)
            .nan_to_num(0)
        )
        p = hulls[is_outside_joint].unsqueeze(0) + t * v

        D = one_norm(torch.cos(outside_theta) - torch.cos(p)) + one_norm(
            torch.sin(outside_theta) - torch.sin(p)
        )
        loss[is_outside_batch] = (torch.min(D, dim=-1)[0] * is_outside).sum(
            -1
        ) / 15  # torch.sum(torch.min(D, dim=-1)[0], dim=-1)

        # vis = 0
        # if vis:
        #     print(theta)
        #     plot_hull(theta, hull)

        # loss += sub_loss

    # loss /= 15  # * thetas.shape[0]
    # loss = loss.clip(0, 1e3)
    # loss.nan_to_num_(0)

    return loss


def calculate_joint_angle_loss_batch(thetas, hulls):
    """

    Args:
        Theta: B*15*2
        hulls: list

    Returns:

    """

    loss = torch.zeros(thetas.shape[0]).cuda()
    for i in range(15):
        # print("i=",i)
        hull = hulls[i]  # (M*2)
        theta = thetas[:, i]  # (B*2)
        hull = torch.cat((hull, hull[0].unsqueeze(0)), dim=0)

        v = (hull[1:] - hull[:-1]).unsqueeze(0)  # (M-1)*2
        w = -hull[:-1].unsqueeze(0) + theta.unsqueeze(1).repeat(
            1, hull[:-1].shape[0], 1
        )  # B*(M-1)*2

        cross_product_2d = w[:, :, 0] * v[:, :, 1] - w[:, :, 1] * v[:, :, 0]
        tmp = torch.sum(cross_product_2d < 1e-6, dim=-1)

        is_outside = tmp != (hull.shape[0] - 1)
        # is_outside = torch.ones(thetas.shape[0]).bool().cuda()
        if not torch.sum(is_outside):
            sub_loss = torch.zeros(thetas.shape[0]).cuda()
        else:
            sub_loss = torch.zeros(thetas.shape[0]).cuda()
            outside_theta = theta[is_outside]
            outside_theta = outside_theta.unsqueeze(1).repeat(1, hull[:-1].shape[0], 1)
            w_outside = -hull[:-1].unsqueeze(0) + outside_theta  # B*(M-1)*2
            t = torch.clamp(
                inner_product(w_outside, v) / (two_norm(v) ** 2), min=0, max=1
            ).unsqueeze(2)
            p = hull[:-1] + t * v

            D = one_norm(torch.cos(outside_theta) - torch.cos(p)) + one_norm(
                torch.sin(outside_theta) - torch.sin(p)
            )
            sub_loss[is_outside] = torch.min(D, dim=-1)[
                0
            ]  # torch.sum(torch.min(D, dim=-1)[0], dim=-1)

        vis = 0
        if vis:
            print(theta)
            plot_hull(theta, hull)

        loss += sub_loss

    loss /= 15  # * thetas.shape[0]
    # loss = loss.clip(0, 1e3)
    # loss.nan_to_num_(0)

    return loss


def angle_between(v1, v2):
    epsilon = 1e-7
    cos = torch_f.cosine_similarity(v1, v2, dim=-1).clamp(
        -1 + epsilon, 1 - epsilon
    )  # (B)
    theta = torch.acos(cos)  # (B)
    return theta


def normalize(vec):
    return torch_f.normalize(vec, p=2, dim=-1)


def inner_product(x1, x2):
    return torch.sum(x1 * x2, dim=-1)


def cross_product(x1, x2):
    return torch.cross(x1, x2, dim=-1)


def axangle2mat_torch(axis, angle, is_normalized=False):
    """Rotation matrix for rotation angle `angle` around `axis`
    Parameters
    ----------
    axis : [B,M, 3] element sequence
       vector specifying axis for rotation.
    angle :[B,M, ] scalar
       angle of rotation in radians.
    is_normalized : bool, optional
       True if `axis` is already normalized (has norm of 1).  Default False.
    Returns
    -------
    mat : array shape (B, M,3,3)
       rotation matrix for specified rotation
    Notes
    -----
    From: http://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
    """
    B = axis.shape[0]
    M = axis.shape[1]

    if not is_normalized:
        norm_axis = axis.norm(p=2, dim=-1, keepdim=True)
        normed_axis = axis / norm_axis
    else:
        normed_axis = axis
    x, y, z = normed_axis[:, :, 0], normed_axis[:, :, 1], normed_axis[:, :, 2]
    c = torch.cos(angle)
    s = torch.sin(angle)
    C = 1 - c

    xs = x * s
    ys = y * s
    zs = z * s  # noqa
    xC = x * C
    yC = y * C
    zC = z * C  # noqa
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC  # noqa

    TMP = torch.stack(
        [
            x * xC + c,
            xyC - zs,
            zxC + ys,
            xyC + zs,
            y * yC + c,
            yzC - xs,
            zxC - ys,
            yzC + xs,
            z * zC + c,
        ],
        dim=-1,
    )
    return TMP.reshape(B, M, 3, 3)


def interval_loss(value, min, max):
    """
    calculate interval loss
    Args:
        value: B*M
        max: M
        min: M

    Returns:

    """

    # batch_3d_size = value.shape[0]

    _min = min.repeat(value.shape[0], 1)
    _max = max.repeat(value.shape[0], 1)

    loss1 = torch.max(_min - value, torch.Tensor([0]).cuda())
    loss2 = torch.max(value - _max, torch.Tensor([0]).cuda())

    loss = (loss1 + loss2).mean(-1)

    # loss /= value.shape[1]

    return loss


class BMCLossConfig:
    path: str = "data/BMC"
    lambda_bl: float = 1.0
    lambda_rb: float = 0.0
    lambda_a: float = 1.0
    scale: float = 1.0


# @loss_builder.register(BMCLossConfig, "bmc")
class BMCLoss:
    SNAP_PARENT: list[int] = [
        0,  # 0's parent
        0,  # 1's parent
        1,
        2,
        3,
        0,  # 5's parent
        5,
        6,
        7,
        0,  # 9's parent
        9,
        10,
        11,
        0,  # 13's parent
        13,
        14,
        15,
        0,  # 17's parent
        17,
        18,
        19,
    ]

    JOINT_ROOT_IDX: int = 9

    REF_BONE_LINK: tuple = (0, 9)  # mid mcp

    # bone indexes in 20 bones setting
    ID_ROOT_bone: list[int] = [0, 4, 8, 12, 16]  # ROOT_bone from wrist to MCP
    ID_PIP_bone: list[int] = [1, 5, 9, 13, 17]  # PIP_bone from MCP to PIP
    ID_DIP_bone: list[int] = [2, 6, 10, 14, 18]  # DIP_bone from  PIP to DIP
    ID_TIP_bone: list[int] = [3, 7, 11, 15, 19]  # TIP_bone from DIP to TIP

    def __init__(
        self,
        config: BMCLossConfig,
    ):
        super().__init__()

        self.config = config
        self.lambda_bl = config.lambda_bl
        self.lambda_rb = config.lambda_rb
        self.lambda_a = config.lambda_a

        self.SNAP_PARENT = self.SNAP_PARENT
        self.JOINT_ROOT_IDX = self.JOINT_ROOT_IDX
        self.REF_BONE_LINK = self.REF_BONE_LINK
        self.ID_ROOT_bone = np.array(self.ID_ROOT_bone)
        self.ID_PIP_bone = np.array(self.ID_PIP_bone)  # PIP_bone from MCP to PIP
        self.ID_DIP_bone = np.array(self.ID_DIP_bone)  # DIP_bone from  PIP to DIP
        self.ID_TIP_bone = np.array(self.ID_TIP_bone)
        self.scale = config.scale

        # self.lp = "../BMC"
        self.lp = config.path

        self.bone_len_max = np.load(os.path.join(self.lp, "bone_len_max.npy"))
        self.bone_len_min = np.load(os.path.join(self.lp, "bone_len_min.npy"))
        self.rb_curvatures_max = np.load(os.path.join(self.lp, "curvatures_max.npy"))
        self.rb_curvatures_min = np.load(os.path.join(self.lp, "curvatures_min.npy"))
        self.rb_PHI_max = np.load(os.path.join(self.lp, "PHI_max.npy"))
        self.rb_PHI_min = np.load(os.path.join(self.lp, "PHI_min.npy"))

        # self.joint_angle_limit = np.load(os.path.join(self.lp, "CONVEX_HULLS.npy"),
        #                                  allow_pickle=True)
        self.joint_angle_limit = torch.load(
            os.path.join(self.lp, "CONVEX_HULLS.pt"), weights_only=False
        )
        LEN_joint_angle_limit = len(self.joint_angle_limit)

        self.bl_max = torch.from_numpy(self.bone_len_max).float().cuda()
        self.bl_min = torch.from_numpy(self.bone_len_min).float().cuda()
        self.rb_curvatures_max = torch.from_numpy(self.rb_curvatures_max).float().cuda()
        self.rb_curvatures_min = torch.from_numpy(self.rb_curvatures_min).float().cuda()
        self.rb_PHI_max = torch.from_numpy(self.rb_PHI_max).float().cuda()
        self.rb_PHI_min = torch.from_numpy(self.rb_PHI_min).float().cuda()

        self.joint_angle_limit = [
            torch.from_numpy(self.joint_angle_limit[i]).float().cuda()
            for i in range(LEN_joint_angle_limit)
        ]

        pad_joint_angle_limit = [
            torch.cat(
                (hull, hull[-1].unsqueeze(0).repeat(25 - hull.shape[0], 1)), dim=0
            )
            for hull in self.joint_angle_limit
        ]
        pad_joint_angle_limit = torch.stack(pad_joint_angle_limit, dim=0)  # (15, 25, 2)

        self.pad_joint_angle_limit = pad_joint_angle_limit  # (15, 25, 2)

        cat_joint_angle_limit = torch.cat(
            (pad_joint_angle_limit, pad_joint_angle_limit[:, 0].unsqueeze(1)), dim=1
        )  # (15, 26, 2)
        self.hull_verts = (
            cat_joint_angle_limit[:, 1:] - cat_joint_angle_limit[:, :-1]
        )  # (15, 25, 2)
        # for i in range(LEN_joint_angle_limit):
        #     plot_hull(torch.zeros(1, 2), pad_joint_angle_limit[i])

        # hulls = [
        #     torch.cat((hull, hull[0].unsqueeze(0)), dim=0)
        #     for hull in self.joint_angle_limit
        # ]
        # self.hull_verts = [
        #     hulls[i][1:] - hulls[i][:-1] for i in range(LEN_joint_angle_limit)
        # ]

    # @profile
    def compute_loss_full(self, joints: torch.Tensor):
        """

        Args:
            joints: B*21*3

        Returns:

        """
        joints = joints * 10

        batch_size = joints.shape[0]
        final_loss = torch.zeros(joints.shape[0]).cuda()

        BMC_losses = {
            "bmc_bl": torch.zeros(joints.shape[0]).cuda(),
            "bmc_rb": torch.zeros(joints.shape[0]).cuda(),
            "bmc_a": torch.zeros(joints.shape[0]).cuda(),
            "bmc_loss": torch.zeros(joints.shape[0]).cuda(),
        }

        if (
            (self.lambda_bl < 1e-6)
            and (self.lambda_rb < 1e-6)
            and (self.lambda_a < 1e-6)
        ):
            return BMC_losses

        parent_joints = joints[:, self.SNAP_PARENT, :]  # (B,21,3)
        ALL_bones = (joints - parent_joints)[:, 1:, :]  # (B,20,3)
        # ALL_bones = [
        #     (joints[:, i, :] - joints[:, self.SNAP_PARENT[i], :]) for i in range(21)
        # ]
        # ALL_bones = torch.stack(ALL_bones[1:], dim=1)  # (B,20,3)
        ROOT_bones = ALL_bones[:, self.ID_ROOT_bone]  # (B,5,3)
        PIP_bones = ALL_bones[:, self.ID_PIP_bone]
        DIP_bones = ALL_bones[:, self.ID_DIP_bone]
        TIP_bones = ALL_bones[:, self.ID_TIP_bone]

        ALL_Z_axis = ALL_bones / torch.norm(ALL_bones, dim=-1, keepdim=True)
        # ALL_Z_axis = normalize(ALL_bones)
        PIP_Z_axis = ALL_Z_axis[:, self.ID_ROOT_bone]
        DIP_Z_axis = ALL_Z_axis[:, self.ID_PIP_bone]
        TIP_Z_axis = ALL_Z_axis[:, self.ID_DIP_bone]  # (B, 5, 3)

        normals = normalize(
            cross_product(ROOT_bones[:, 1:], ROOT_bones[:, :-1])
        )  # (B,4,3)

        # compute loss of bone length
        bl_loss = torch.zeros(joints.shape[0]).cuda()
        if self.lambda_bl:
            bls = two_norm(ALL_bones)  # (B,20,1)
            # print(bls)
            bl_loss = interval_loss(value=bls, min=self.bl_min, max=self.bl_max)
            # print(bl_loss)
            # final_loss += self.lambda_bl * bl_loss
        BMC_losses["bmc_bl"] = bl_loss

        # compute loss of Root bones
        rb_loss = torch.zeros(joints.shape[0]).cuda()
        if self.lambda_rb:
            edge_normals = torch.zeros_like(ROOT_bones).cuda()  # (B,5,3)
            edge_normals[:, [0, 4]] = normals[:, [0, 3]]
            edge_normals[:, 1:4] = normalize(normals[:, 1:4] + normals[:, :3])

            curvatures = inner_product(
                edge_normals[:, 1:] - edge_normals[:, :4],
                ROOT_bones[:, 1:] - ROOT_bones[:, :4],
            ) / (two_norm(ROOT_bones[:, 1:] - ROOT_bones[:, :4]) ** 2)
            PHI = angle_between(ROOT_bones[:, :4], ROOT_bones[:, 1:])  # (B)

            rb_loss = interval_loss(
                value=curvatures, min=self.rb_curvatures_min, max=self.rb_curvatures_max
            ) + interval_loss(value=PHI, min=self.rb_PHI_min, max=self.rb_PHI_max)
            # final_loss += self.lambda_rb * rb_loss
        BMC_losses["bmc_rb"] = rb_loss

        # compute loss of Joint angles
        a_loss = torch.zeros(joints.shape[0]).cuda()
        if self.lambda_a:
            # PIP bones
            PIP_X_axis = torch.zeros([batch_size, 5, 3]).cuda()  # (B,5,3)
            PIP_X_axis[:, [0, 1, 4], :] = -normals[:, [0, 1, 3]]
            PIP_X_axis[:, 2:4] = -normalize(
                normals[:, 2:4] + normals[:, 1:3]
            )  # (B,2,3)
            PIP_Y_axis = normalize(cross_product(PIP_Z_axis, PIP_X_axis))  # (B,5,3)

            PIP_bones_xz = (
                PIP_bones
                - inner_product(PIP_bones, PIP_Y_axis).unsqueeze(2) * PIP_Y_axis
            )

            temp_axis = normalize(cross_product(PIP_Z_axis, PIP_bones))
            temp_alpha = angle_between(
                PIP_Z_axis, PIP_bones
            )  # alpha belongs to [pi/2, pi]
            temp_R = axangle2mat_torch(
                axis=temp_axis, angle=temp_alpha, is_normalized=True
            )

            # DIP bones
            DIP_X_axis = torch.matmul(temp_R, PIP_X_axis.unsqueeze(3)).squeeze(-1)
            DIP_Y_axis = torch.matmul(temp_R, PIP_Y_axis.unsqueeze(3)).squeeze(-1)

            DIP_bones_xz = (
                DIP_bones
                - inner_product(DIP_bones, DIP_Y_axis).unsqueeze(2) * DIP_Y_axis
            )

            temp_axis = normalize(cross_product(DIP_Z_axis, DIP_bones))
            temp_alpha = angle_between(
                DIP_Z_axis, DIP_bones
            )  # alpha belongs to [pi/2, pi]
            temp_R = axangle2mat_torch(
                axis=temp_axis, angle=temp_alpha, is_normalized=True
            )

            # TIP bones
            TIP_X_axis = torch.matmul(temp_R, DIP_X_axis.unsqueeze(3)).squeeze(-1)
            TIP_Y_axis = torch.matmul(temp_R, DIP_Y_axis.unsqueeze(3)).squeeze(-1)
            TIP_bones_xz = (
                TIP_bones
                - inner_product(TIP_bones, TIP_Y_axis).unsqueeze(2) * TIP_Y_axis
            )

            bones_xz = torch.cat(
                [PIP_bones_xz, DIP_bones_xz, TIP_bones_xz], dim=1
            )  # (B,15,3)
            Z_axis = torch.cat([PIP_Z_axis, DIP_Z_axis, TIP_Z_axis], dim=1)
            Y_axis = torch.cat([PIP_Y_axis, DIP_Y_axis, TIP_Y_axis], dim=1)
            X_axis = torch.cat([PIP_X_axis, DIP_X_axis, TIP_X_axis], dim=1)
            bones = torch.cat([PIP_bones, DIP_bones, TIP_bones], dim=1)
            # x-component of the bone vector
            tmp = inner_product(bones, X_axis)
            theta_flexion = angle_between(bones_xz, Z_axis)
            theta_flexion = torch.where(tmp < 1e-6, -theta_flexion, theta_flexion)
            tmp = inner_product(bones, Y_axis)
            theta_abduction = angle_between(bones_xz, bones)
            theta_abduction = torch.where(tmp < 1e-6, -theta_abduction, theta_abduction)

            # # ALL
            # ALL_theta_flexion = torch.cat(
            #     (PIP_theta_flexion, DIP_theta_flexion, TIP_theta_flexion), dim=-1
            # )
            # ALL_theta_abduction = torch.cat(
            #     (PIP_theta_abduction, DIP_theta_abduction, TIP_theta_abduction), dim=-1
            # )
            ALL_theta = torch.stack((theta_flexion, theta_abduction), dim=-1)

            a_loss = calculate_joint_angle_loss(ALL_theta, self.joint_angle_limit)

        final_loss = (
            self.lambda_a * a_loss + self.lambda_rb * rb_loss + self.lambda_bl * bl_loss
        )

        BMC_losses["bmc_a"] = a_loss
        BMC_losses["bmc_loss"] = final_loss * self.scale

        return BMC_losses

    # @profile
    def compute_loss(self, joints: torch.Tensor):
        """
        only constrain on joint angles

        Args:
            joints: B*21*3

        Returns:

        """
        # joints = joints * 10

        batch_size = joints.shape[0]
        # final_loss = torch.zeros(joints.shape[0]).cuda()

        BMC_losses = {
            "bmc_loss": torch.zeros(joints.shape[0]).cuda(),
        }

        if (
            (self.lambda_bl < 1e-6)
            and (self.lambda_rb < 1e-6)
            and (self.lambda_a < 1e-6)
        ):
            return BMC_losses

        parent_joints = joints[:, self.SNAP_PARENT, :]  # (B,21,3)
        ALL_bones = (joints - parent_joints)[:, 1:, :]  # (B,20,3)
        # ALL_bones = [
        #     (joints[:, i, :] - joints[:, self.SNAP_PARENT[i], :]) for i in range(21)
        # ]
        # ALL_bones = torch.stack(ALL_bones[1:], dim=1)  # (B,20,3)
        ROOT_bones = ALL_bones[:, self.ID_ROOT_bone]  # (B,5,3)
        PIP_bones = ALL_bones[:, self.ID_PIP_bone]
        DIP_bones = ALL_bones[:, self.ID_DIP_bone]
        TIP_bones = ALL_bones[:, self.ID_TIP_bone]

        # ALL_Z_axis = ALL_bones / torch.norm(ALL_bones, dim=-1, keepdim=True)
        ALL_Z_axis = normalize(ALL_bones)
        PIP_Z_axis = ALL_Z_axis[:, self.ID_ROOT_bone]
        DIP_Z_axis = ALL_Z_axis[:, self.ID_PIP_bone]
        TIP_Z_axis = ALL_Z_axis[:, self.ID_DIP_bone]  # (B, 5, 3)

        normals = normalize(
            cross_product(ROOT_bones[:, 1:], ROOT_bones[:, :-1])
        )  # (B,4,3)

        # compute loss of Joint angles
        a_loss = torch.zeros(joints.shape[0]).cuda()
        # if self.lambda_a:
        # PIP bones
        PIP_X_axis = torch.zeros([batch_size, 5, 3]).cuda()  # (B,5,3)
        PIP_X_axis[:, [0, 1, 4], :] = -normals[:, [0, 1, 3]]
        PIP_X_axis[:, 2:4] = -normalize(normals[:, 2:4] + normals[:, 1:3])  # (B,2,3)
        PIP_Y_axis = normalize(cross_product(PIP_Z_axis, PIP_X_axis))  # (B,5,3)

        # PIP_bones_xz = (
        #     PIP_bones - inner_product(PIP_bones, PIP_Y_axis).unsqueeze(2) * PIP_Y_axis
        # )

        temp_axis = normalize(cross_product(PIP_Z_axis, PIP_bones))
        temp_alpha = angle_between(PIP_Z_axis, PIP_bones)  # alpha belongs to [pi/2, pi]
        # pytorch_angle = axis_angle_to_matrix(temp_axis * temp_alpha.unsqueeze(-1))
        temp_R = axangle2mat_torch(axis=temp_axis, angle=temp_alpha, is_normalized=True)

        # DIP bones
        # DIP_X_axis = torch.einsum("nbij,nbj->nbi", temp_R, PIP_X_axis)
        # DIP_Y_axis = torch.einsum("nbij,nbj->nbi", temp_R, PIP_Y_axis)
        DIP_X_axis = torch.matmul(temp_R, PIP_X_axis.unsqueeze(3)).squeeze(-1)
        DIP_Y_axis = torch.matmul(temp_R, PIP_Y_axis.unsqueeze(3)).squeeze(-1)

        # DIP_bones_xz = (
        #     DIP_bones - inner_product(DIP_bones, DIP_Y_axis).unsqueeze(2) * DIP_Y_axis
        # )

        temp_axis = normalize(cross_product(DIP_Z_axis, DIP_bones))
        temp_alpha = angle_between(DIP_Z_axis, DIP_bones)  # alpha belongs to [pi/2, pi]
        temp_R = axangle2mat_torch(axis=temp_axis, angle=temp_alpha, is_normalized=True)

        # TIP bones
        TIP_X_axis = torch.matmul(temp_R, DIP_X_axis.unsqueeze(3)).squeeze(-1)
        TIP_Y_axis = torch.matmul(temp_R, DIP_Y_axis.unsqueeze(3)).squeeze(-1)
        # TIP_bones_xz = (
        #     TIP_bones - inner_product(TIP_bones, TIP_Y_axis).unsqueeze(2) * TIP_Y_axis
        # )

        # bones_xz = torch.cat(
        #     [PIP_bones_xz, DIP_bones_xz, TIP_bones_xz], dim=1
        # )  # (B,15,3)
        Z_axis = torch.cat([PIP_Z_axis, DIP_Z_axis, TIP_Z_axis], dim=1)
        Y_axis = torch.cat([PIP_Y_axis, DIP_Y_axis, TIP_Y_axis], dim=1)
        X_axis = torch.cat([PIP_X_axis, DIP_X_axis, TIP_X_axis], dim=1)
        bones = torch.cat([PIP_bones, DIP_bones, TIP_bones], dim=1)
        # bones_xz = bones - inner_product(bones, Y_axis).unsqueeze(2) * Y_axis

        rotation = torch.stack([X_axis, Y_axis, Z_axis], dim=-2)
        bone_inv = torch.einsum("nbj, nbij->nbi", bones, rotation)
        # x-component of the bone vector
        # tmp = bone_inv[:, :, 0]  # inner_product(bones, X_axis)

        theta_flexion = torch.arctan2(bone_inv[:, :, 0], bone_inv[:, :, 2])
        theta_abduction = torch.arctan2(
            bone_inv[:, :, 1],
            torch.norm(
                torch.stack([bone_inv[:, :, 0], bone_inv[:, :, 2]], dim=-1), dim=-1
            ),
        )
        # theta_flexion = angle_between(bones_xz, Z_axis)
        # theta_flexion = torch.where(tmp < 1e-6, -theta_flexion, theta_flexion)
        # tmp = bone_inv[:, :, 1]  # - inner_product(bones, Y_axis)
        # theta_abduction = angle_between(bones_xz, bones)
        # theta_abduction = torch.where(tmp < 1e-6, -theta_abduction, theta_abduction)

        # # ALL
        # ALL_theta_flexion = torch.cat(
        #     (PIP_theta_flexion, DIP_theta_flexion, TIP_theta_flexion), dim=-1
        # )
        # ALL_theta_abduction = torch.cat(
        #     (PIP_theta_abduction, DIP_theta_abduction, TIP_theta_abduction), dim=-1
        # )
        ALL_theta = torch.stack((theta_flexion, theta_abduction), dim=-1)

        # a_loss = calculate_joint_angle_loss(
        #     ALL_theta, self.pad_joint_angle_limit, self.hull_verts
        # )

        a_loss = calculate_joint_angle_loss_batch(
            ALL_theta,
            self.joint_angle_limit,  # , self.hull_verts
        )

        # print("a_loss_1", a_loss_1)
        # print("a_loss", a_loss)
        # print(a_loss_1 - a_loss)

        # final_loss = (
        #     self.lambda_a
        #     * a_loss  # + self.lambda_rb * rb_loss + self.lambda_bl * bl_loss
        # )

        # BMC_losses["bmc_a"] = a_loss
        # BMC_losses["bmc_loss"] = self.lambda_a * a_loss * self.scale

        return {"bmc_loss": self.lambda_a * a_loss * self.scale}


if __name__ == "__main__":
    from dexrl.configs.default_bmc_loss import _C as bmc_cfg
    from dexrl.utils.path import PACKAGE_EXTERNAL_DIR

    _C = bmc_cfg
    _C.scale = 1.0
    _C.path = f"{PACKAGE_EXTERNAL_DIR}/Hand-BMC-pytorch/BMC"
    _C.lambda_bl = 0.0  # 1.0
    _C.lambda_rb = 0.0  # 1.0
    _C.lambda_a = 1.0

    bmc = BMCLoss(_C)
    joints = torch.load()["j3d"].cuda()

    # joints = torch.rand(10 * 63).reshape(-1, 21, 3).float().cuda()  # (100,21,3)
    for i in range(100):
        bmc_loss = bmc.compute_loss(joints)
    for k, v in bmc_loss.items():
        print(k, v)

    print("")
