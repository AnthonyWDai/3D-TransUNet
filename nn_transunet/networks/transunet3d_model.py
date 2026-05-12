# 3D version of TransUNet; Copyright Johns Hopkins University
# Modified from nnUNet
# Revised for numerical stability / NaN safety by ChatGPT5.4

import torch
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
from torch import nn
from torch.amp import autocast
from scipy.optimize import linear_sum_assignment

from ..networks.neural_network import SegmentationNetwork
from .vit_modeling import Transformer
from .vit_modeling import CONFIGS as CONFIGS_ViT


softmax_helper = lambda x: F.softmax(x, dim=1)


def sanitize_logits(x: torch.Tensor, clamp: float = 20.0) -> torch.Tensor:
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=clamp, neginf=-clamp)
    return x.clamp(min=-clamp, max=clamp)


def sanitize_probs(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=1.0, neginf=0.0)
    return x.clamp(min=eps, max=1.0 - eps)


def sanitize_targets(x: torch.Tensor) -> torch.Tensor:
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=1.0, neginf=0.0)
    return x.clamp(0.0, 1.0)


def assert_finite(x: torch.Tensor, name: str):
    if not torch.isfinite(x).all():
        raise RuntimeError(
            f"{name} has non-finite values: "
            f"nan={torch.isnan(x).any().item()}, inf={torch.isinf(x).any().item()}, "
            f"shape={tuple(x.shape)}, dtype={x.dtype}"
        )


class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


class ConvDropoutNormNonlin(nn.Module):
    """
    Conv -> Dropout -> Norm -> Nonlinearity
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        conv_op=nn.Conv2d,
        conv_kwargs=None,
        norm_op=nn.BatchNorm2d,
        norm_op_kwargs=None,
        dropout_op=nn.Dropout2d,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs=None,
    ):
        super().__init__()

        if nonlin_kwargs is None:
            nonlin_kwargs = {"negative_slope": 1e-2, "inplace": True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {"p": 0.5, "inplace": True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {"eps": 1e-5, "affine": True, "momentum": 0.1}
        if conv_kwargs is None:
            conv_kwargs = {"kernel_size": 3, "stride": 1, "padding": 1, "dilation": 1, "bias": True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = deepcopy(conv_kwargs)
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)

        if self.dropout_op is not None and self.dropout_op_kwargs.get("p", 0) not in (None, 0):
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None

        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.instnorm(x)
        x = self.lrelu(x)
        return x


class ConvDropoutNonlinNorm(ConvDropoutNormNonlin):
    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.lrelu(x)
        x = self.instnorm(x)
        return x


class StackedConvLayers(nn.Module):
    def __init__(
        self,
        input_feature_channels,
        output_feature_channels,
        num_convs,
        conv_op=nn.Conv2d,
        conv_kwargs=None,
        norm_op=nn.BatchNorm2d,
        norm_op_kwargs=None,
        dropout_op=nn.Dropout2d,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs=None,
        first_stride=None,
        basic_block=ConvDropoutNormNonlin,
    ):
        super().__init__()

        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        if nonlin_kwargs is None:
            nonlin_kwargs = {"negative_slope": 1e-2, "inplace": True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {"p": 0.5, "inplace": True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {"eps": 1e-5, "affine": True, "momentum": 0.1}
        if conv_kwargs is None:
            conv_kwargs = {"kernel_size": 3, "stride": 1, "padding": 1, "dilation": 1, "bias": True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = deepcopy(conv_kwargs)
        self.conv_op = conv_op
        self.norm_op = norm_op

        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv["stride"] = first_stride
        else:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)

        blocks = [
            basic_block(
                input_feature_channels,
                output_feature_channels,
                self.conv_op,
                self.conv_kwargs_first_conv,
                self.norm_op,
                self.norm_op_kwargs,
                self.dropout_op,
                self.dropout_op_kwargs,
                self.nonlin,
                self.nonlin_kwargs,
            )
        ]

        blocks += [
            basic_block(
                output_feature_channels,
                output_feature_channels,
                self.conv_op,
                self.conv_kwargs,
                self.norm_op,
                self.norm_op_kwargs,
                self.dropout_op,
                self.dropout_op_kwargs,
                self.nonlin,
                self.nonlin_kwargs,
            )
            for _ in range(num_convs - 1)
        ]

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


def print_module_training_status(module):
    if isinstance(
        module,
        (
            nn.Conv2d,
            nn.Conv3d,
            nn.Dropout3d,
            nn.Dropout2d,
            nn.Dropout,
            nn.InstanceNorm3d,
            nn.InstanceNorm2d,
            nn.InstanceNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.BatchNorm1d,
        ),
    ):
        print(str(module), module.training)


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super().__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        kwargs = {
            "size": self.size,
            "scale_factor": self.scale_factor,
            "mode": self.mode,
        }
        if self.mode in ("linear", "bilinear", "bicubic", "trilinear"):
            kwargs["align_corners"] = self.align_corners
        return F.interpolate(x, **kwargs)


def c2_xavier_fill(module: nn.Module) -> None:
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


class Generic_TransUNet_max_ppbp(SegmentationNetwork):
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320
    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480
    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000

    def __init__(
        self,
        input_channels,
        base_num_features,
        num_classes,
        num_pool,
        num_conv_per_stage=2,
        feat_map_mul_on_downscale=2,
        conv_op=nn.Conv2d,
        norm_op=nn.BatchNorm2d,
        norm_op_kwargs=None,
        dropout_op=nn.Dropout2d,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs=None,
        deep_supervision=True,
        dropout_in_localization=False,
        final_nonlin=softmax_helper,
        weightInitializer=InitWeights_He(1e-2),
        pool_op_kernel_sizes=None,
        conv_kernel_sizes=None,
        upscale_logits=False,
        convolutional_pooling=False,
        convolutional_upsampling=False,
        max_num_features=None,
        basic_block=ConvDropoutNormNonlin,
        seg_output_use_bias=False,
        patch_size=None,
        is_vit_pretrain=False,
        vit_depth=12,
        vit_hidden_size=768,
        vit_mlp_dim=3072,
        vit_num_heads=12,
        max_msda="",
        is_max_ms=True,
        is_max_ms_fpn=False,
        max_n_fpn=4,
        max_ms_idxs=None,
        max_ss_idx=0,
        is_max_bottleneck_transformer=False,
        max_seg_weight=1.0,
        max_hidden_dim=256,
        max_dec_layers=10,
        mw=0.5,
        is_max=True,
        is_masked_attn=False,
        is_max_ds=False,
        is_masking=False,
        is_masking_argmax=False,
        is_fam=False,
        fam_k=5,
        fam_reduct_ratio=8,
        is_max_hungarian=False,
        num_queries=None,
        is_max_cls=False,
        point_rend=False,
        num_point_rend=None,
        no_object_weight=None,
        is_mhsa_float32=False,
        no_max_hw_pe=False,
        max_infer=None,
        cost_weight=None,
        vit_layer_scale=False,
        decoder_layer_scale=False,
    ):
        super().__init__()

        if max_ms_idxs is None:
            max_ms_idxs = [-4, -3, -2]
        if cost_weight is None:
            cost_weight = [2.0, 5.0, 5.0]

        self.is_fam = is_fam
        self.is_max = is_max
        self.max_msda = max_msda
        self.is_max_ms = is_max_ms
        self.is_max_ms_fpn = is_max_ms_fpn
        self.max_n_fpn = max_n_fpn
        self.max_ss_idx = max_ss_idx
        self.mw = mw
        self.max_ms_idxs = max_ms_idxs
        self.is_max_cls = is_max_cls
        self.is_masked_attn = is_masked_attn
        self.is_max_ds = is_max_ds
        self.is_max_bottleneck_transformer = is_max_bottleneck_transformer
        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits

        if nonlin_kwargs is None:
            nonlin_kwargs = {"negative_slope": 1e-2, "inplace": True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {"p": 0.5, "inplace": True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {"eps": 1e-5, "affine": True, "momentum": 0.1}

        self.conv_kwargs = {"stride": 1, "dilation": 1, "bias": True}
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = deepcopy(dropout_op_kwargs)
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

        if conv_op == nn.Conv2d:
            upsample_mode = "bilinear"
            pool_op = nn.MaxPool2d
            transpconv = nn.ConvTranspose2d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
        elif conv_op == nn.Conv3d:
            upsample_mode = "trilinear"
            pool_op = nn.MaxPool3d
            transpconv = nn.ConvTranspose3d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
        else:
            raise ValueError(f"unknown convolution dimensionality, conv op: {str(conv_op)}")

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        if max_num_features is None:
            self.max_num_features = self.MAX_NUM_FILTERS_3D if self.conv_op == nn.Conv3d else self.MAX_FILTERS_2D
        else:
            self.max_num_features = max_num_features

        conv_blocks_context = []
        conv_blocks_localization = []
        td = []
        tu = []
        fams = []

        output_features = base_num_features
        input_features = input_channels

        for d in range(num_pool):
            if d != 0 and self.convolutional_pooling:
                first_stride = pool_op_kernel_sizes[d - 1]
            else:
                first_stride = None

            self.conv_kwargs["kernel_size"] = self.conv_kernel_sizes[d]
            self.conv_kwargs["padding"] = self.conv_pad_sizes[d]

            conv_blocks_context.append(
                StackedConvLayers(
                    input_features,
                    output_features,
                    num_conv_per_stage,
                    self.conv_op,
                    self.conv_kwargs,
                    self.norm_op,
                    self.norm_op_kwargs,
                    self.dropout_op,
                    self.dropout_op_kwargs,
                    self.nonlin,
                    self.nonlin_kwargs,
                    first_stride,
                    basic_block=basic_block,
                )
            )

            if not self.convolutional_pooling:
                td.append(pool_op(pool_op_kernel_sizes[d]))

            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))
            output_features = min(output_features, self.max_num_features)

        if self.convolutional_pooling:
            first_stride = pool_op_kernel_sizes[-1]
        else:
            first_stride = None

        if self.convolutional_upsampling:
            final_num_features = output_features
        else:
            final_num_features = conv_blocks_context[-1].output_channels

        self.conv_kwargs["kernel_size"] = self.conv_kernel_sizes[num_pool]
        self.conv_kwargs["padding"] = self.conv_pad_sizes[num_pool]

        conv_blocks_context.append(
            nn.Sequential(
                StackedConvLayers(
                    input_features,
                    output_features,
                    num_conv_per_stage - 1,
                    self.conv_op,
                    self.conv_kwargs,
                    self.norm_op,
                    self.norm_op_kwargs,
                    self.dropout_op,
                    self.dropout_op_kwargs,
                    self.nonlin,
                    self.nonlin_kwargs,
                    first_stride,
                    basic_block=basic_block,
                ),
                StackedConvLayers(
                    output_features,
                    final_num_features,
                    1,
                    self.conv_op,
                    self.conv_kwargs,
                    self.norm_op,
                    self.norm_op_kwargs,
                    self.dropout_op,
                    self.dropout_op_kwargs,
                    self.nonlin,
                    self.nonlin_kwargs,
                    basic_block=basic_block,
                ),
            )
        )

        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs["p"]
            self.dropout_op_kwargs["p"] = 0.0

        for u in range(num_pool):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = conv_blocks_context[-(2 + u)].output_channels
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            if u != num_pool - 1 and not self.convolutional_upsampling:
                final_num_features = conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip

            if not self.convolutional_upsampling:
                tu.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u + 1)], mode=upsample_mode))
            else:
                tu.append(
                    transpconv(
                        nfeatures_from_down,
                        nfeatures_from_skip,
                        pool_op_kernel_sizes[-(u + 1)],
                        pool_op_kernel_sizes[-(u + 1)],
                        bias=False,
                    )
                )

            self.conv_kwargs["kernel_size"] = self.conv_kernel_sizes[-(u + 1)]
            self.conv_kwargs["padding"] = self.conv_pad_sizes[-(u + 1)]

            conv_blocks_localization.append(
                nn.Sequential(
                    StackedConvLayers(
                        n_features_after_tu_and_concat,
                        nfeatures_from_skip,
                        num_conv_per_stage - 1,
                        self.conv_op,
                        self.conv_kwargs,
                        self.norm_op,
                        self.norm_op_kwargs,
                        self.dropout_op,
                        self.dropout_op_kwargs,
                        self.nonlin,
                        self.nonlin_kwargs,
                        basic_block=basic_block,
                    ),
                    StackedConvLayers(
                        nfeatures_from_skip,
                        final_num_features,
                        1,
                        self.conv_op,
                        self.conv_kwargs,
                        self.norm_op,
                        self.norm_op_kwargs,
                        self.dropout_op,
                        self.dropout_op_kwargs,
                        self.nonlin,
                        self.nonlin_kwargs,
                        basic_block=basic_block,
                    ),
                )
            )

        self.fams = nn.ModuleList(fams)

        if self.do_ds:
            seg_outputs = []
            for ds in range(len(conv_blocks_localization)):
                seg_outputs.append(
                    conv_op(
                        conv_blocks_localization[ds][-1].output_channels,
                        num_classes,
                        1,
                        1,
                        0,
                        1,
                        1,
                        seg_output_use_bias,
                    )
                )
            self.seg_outputs = nn.ModuleList(seg_outputs)
        else:
            self.seg_outputs = nn.ModuleList([])

        upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]

        for usl in range(num_pool - 1):
            if self.upscale_logits:
                upscale_logits_ops.append(
                    Upsample(
                        scale_factor=tuple(int(i) for i in cum_upsample[usl + 1]),
                        mode=upsample_mode,
                    )
                )
            else:
                upscale_logits_ops.append(nn.Identity())

        if not dropout_in_localization:
            self.dropout_op_kwargs["p"] = old_dropout_p

        self.conv_blocks_localization = nn.ModuleList(conv_blocks_localization)
        self.conv_blocks_context = nn.ModuleList(conv_blocks_context)
        self.td = nn.ModuleList(td)
        self.tu = nn.ModuleList(tu)
        self.upscale_logits_ops = nn.ModuleList(upscale_logits_ops)

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)

        # Transformer configuration
        if self.is_max_bottleneck_transformer:
            self.patch_size = patch_size
            config_vit = CONFIGS_ViT["R50-ViT-B_16"]
            config_vit.transformer.num_layers = vit_depth
            config_vit.hidden_size = vit_hidden_size
            config_vit.transformer.mlp_dim = vit_mlp_dim
            config_vit.transformer.num_heads = vit_num_heads

            self.conv_more = nn.Conv3d(config_vit.hidden_size, output_features, 1)

            num_pool_per_axis = np.prod(np.array(pool_op_kernel_sizes), axis=0)
            num_pool_per_axis = np.log2(num_pool_per_axis).astype(np.uint8)
            feat_size = [
                int(self.patch_size[0] / 2 ** num_pool_per_axis[0]),
                int(self.patch_size[1] / 2 ** num_pool_per_axis[1]),
                int(self.patch_size[2] / 2 ** num_pool_per_axis[2]),
            ]
            self.transformer = Transformer(
                config_vit,
                feat_size=feat_size,
                vis=False,
                feat_channels=output_features,
                use_layer_scale=vit_layer_scale,
            )
            if is_vit_pretrain:
                self.transformer.load_from(weights=np.load(config_vit.pretrained_path))

        if self.is_max:
            cfg = {
                "num_classes": num_classes,
                "hidden_dim": max_hidden_dim,
                "num_queries": num_classes if num_queries is None else num_queries,
                "nheads": 8,
                "dim_feedforward": max_hidden_dim * 8,
                "dec_layers": max_dec_layers,
                "pre_norm": False,
                "enforce_input_project": False,
                "mask_dim": max_hidden_dim,
                "non_object": is_max_cls,
                "use_layer_scale": decoder_layer_scale,
            }

            input_proj_list = []
            decoder_channels = [320, 320, 256, 128, 64, 32]

            if self.is_max_ms:
                if self.is_max_ms_fpn:
                    for in_channels in decoder_channels[:max_n_fpn]:
                        input_proj_list.append(
                            nn.Sequential(
                                nn.Conv3d(in_channels, max_hidden_dim, kernel_size=1),
                                nn.GroupNorm(32, max_hidden_dim),
                                nn.Upsample(
                                    size=(int(patch_size[0] / 2), int(patch_size[1] / 4), int(patch_size[2] / 4)),
                                    mode="trilinear",
                                    align_corners=False,
                                ),
                            )
                        )
                    self.input_proj = nn.ModuleList(input_proj_list)
                    self.linear_encoder_feature = nn.Conv3d(max_hidden_dim * max_n_fpn, max_hidden_dim, 1, 1)
                else:
                    for i in self.max_ms_idxs:
                        in_channels = decoder_channels[i]
                        input_proj_list.append(
                            nn.Sequential(
                                nn.Conv3d(in_channels, max_hidden_dim, kernel_size=1),
                                nn.GroupNorm(32, max_hidden_dim),
                            )
                        )
                    self.input_proj = nn.ModuleList(input_proj_list)

                self.linear_mask_features = nn.Conv3d(
                    decoder_channels[-1], cfg["mask_dim"], kernel_size=1, stride=1, padding=0
                )
            else:
                self.linear_encoder_feature = nn.Conv3d(
                    decoder_channels[max_ss_idx], cfg["mask_dim"], kernel_size=1
                )
                self.linear_mask_features = nn.Conv3d(
                    decoder_channels[-1], cfg["mask_dim"], kernel_size=1, stride=1, padding=0
                )

            if self.is_masked_attn:
                from .mask2former_modeling.transformer_decoder.mask2former_transformer_decoder3d import (
                    MultiScaleMaskedTransformerDecoder3d,
                )

                cfg["num_feature_levels"] = 1 if (not self.is_max_ms or self.is_max_ms_fpn) else 3
                cfg["is_masking"] = bool(is_masking)
                cfg["is_masking_argmax"] = bool(is_masking_argmax)
                cfg["is_mhsa_float32"] = bool(is_mhsa_float32)
                cfg["no_max_hw_pe"] = bool(no_max_hw_pe)

                self.predictor = MultiScaleMaskedTransformerDecoder3d(
                    in_channels=max_hidden_dim, mask_classification=is_max_cls, **cfg
                )
            else:
                from .mask2former_modeling.transformer_decoder.maskformer_transformer_decoder3d import (
                    StandardTransformerDecoder,
                )

                cfg["dropout"] = 0.1
                cfg["enc_layers"] = 0
                cfg["deep_supervision"] = False
                self.predictor = StandardTransformerDecoder(
                    in_channels=max_hidden_dim, mask_classification=is_max_cls, **cfg
                )

    def forward(self, x):
        skips = []
        seg_outputs = []

        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        assert_finite(x, "input")

        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
            assert_finite(x, f"encoder_stage_{d}")
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)

        x = self.conv_blocks_context[-1](x)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        assert_finite(x, "bottleneck")

        if self.is_max_bottleneck_transformer:
            x, attn = self.transformer(x)
            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
            assert_finite(x, "transformer_out")
            x = self.conv_more(x)
            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
            assert_finite(x, "conv_more_out")

        ds_feats = [x]

        for u in range(len(self.tu)):
            if u < len(self.tu) - 1 and isinstance(self.is_fam, str) and self.is_fam.startswith('fam_down'):
                skip = skips[-(u + 1)]
                if x.shape[2:] != skip.shape[2:]:
                    skip_down = F.interpolate(skip, size=x.shape[2:], mode='trilinear', align_corners=False)
                else:
                    skip_down = skip
                x_align = self.fams[u](x, x_l=skip_down)
                x_align = torch.nan_to_num(x_align, nan=0.0, posinf=1e4, neginf=-1e4)
                x = x + x_align

            x = self.tu[u](x)
            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
            assert_finite(x, f"up_{u}")

            if isinstance(self.is_fam, bool) and self.is_fam:
                x_align = self.fams[u](x, x_l=skips[-(u + 1)])
                x_align = torch.nan_to_num(x_align, nan=0.0, posinf=1e4, neginf=-1e4)
                x = x + x_align

            skip = skips[-(u + 1)]
            skip = torch.nan_to_num(skip, nan=0.0, posinf=1e4, neginf=-1e4)

            x = torch.cat((x, skip), dim=1)
            x = self.conv_blocks_localization[u](x)
            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
            assert_finite(x, f"decoder_stage_{u}")

            if self.do_ds:
                logits = self.seg_outputs[u](x)
                logits = sanitize_logits(logits)
                assert_finite(logits, f"seg_logits_{u}")
                seg_outputs.append(logits)  # RETURN LOGITS, NOT SOFTMAX

            ds_feats.append(x)

        if self.is_max:
            if self.is_max_ms:
                multi_scale_features = []
                ms_pixel_feats = ds_feats[:self.max_n_fpn] if self.is_max_ms_fpn else [ds_feats[i] for i in self.max_ms_idxs]

                for idx, f in enumerate(ms_pixel_feats):
                    f = self.input_proj[idx](f)
                    f = torch.nan_to_num(f, nan=0.0, posinf=1e4, neginf=-1e4)
                    assert_finite(f, f"input_proj_{idx}")
                    multi_scale_features.append(f)

                transformer_decoder_in_feature = (
                    self.linear_encoder_feature(torch.cat(multi_scale_features, dim=1))
                    if self.is_max_ms_fpn else multi_scale_features
                )
                mask_features = self.linear_mask_features(ds_feats[-1])
            else:
                transformer_decoder_in_feature = self.linear_encoder_feature(ds_feats[self.max_ss_idx])
                mask_features = self.linear_mask_features(ds_feats[-1])

            if isinstance(transformer_decoder_in_feature, list):
                transformer_decoder_in_feature = [
                    torch.nan_to_num(t, nan=0.0, posinf=1e4, neginf=-1e4) for t in transformer_decoder_in_feature
                ]
                for i, t in enumerate(transformer_decoder_in_feature):
                    assert_finite(t, f"transformer_decoder_in_feature_{i}")
            else:
                transformer_decoder_in_feature = torch.nan_to_num(
                    transformer_decoder_in_feature, nan=0.0, posinf=1e4, neginf=-1e4
                )
                assert_finite(transformer_decoder_in_feature, "transformer_decoder_in_feature")

            mask_features = torch.nan_to_num(mask_features, nan=0.0, posinf=1e4, neginf=-1e4)
            assert_finite(mask_features, "mask_features")

            predictions = self.predictor(transformer_decoder_in_feature, mask_features, mask=None)

            # sanitize decoder outputs
            if isinstance(predictions, dict):
                if "pred_logits" in predictions:
                    predictions["pred_logits"] = sanitize_logits(predictions["pred_logits"])
                    assert_finite(predictions["pred_logits"], "predictions[pred_logits]")
                if "pred_masks" in predictions:
                    predictions["pred_masks"] = sanitize_logits(predictions["pred_masks"])
                    assert_finite(predictions["pred_masks"], "predictions[pred_masks]")
                if "aux_outputs" in predictions:
                    for i, aux in enumerate(predictions["aux_outputs"]):
                        if "pred_logits" in aux:
                            aux["pred_logits"] = sanitize_logits(aux["pred_logits"])
                            assert_finite(aux["pred_logits"], f"aux_pred_logits_{i}")
                        if "pred_masks" in aux:
                            aux["pred_masks"] = sanitize_logits(aux["pred_masks"])
                            assert_finite(aux["pred_masks"], f"aux_pred_masks_{i}")

            if self.is_max_cls and self.is_max_ds:
                if self._deep_supervision and self.do_ds:
                    return [predictions] + [i(j) for i, j in zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])]
                return predictions
            elif self.is_max_ds and not self.is_max_ms and self.mw == 1.0:
                aux_out = [sanitize_logits(p['pred_masks']) for p in predictions['aux_outputs']]
                all_out = [sanitize_logits(predictions["pred_masks"])] + aux_out[::-1]
                return tuple(all_out)
            elif not self.is_max_ds and self.mw == 1.0:
                raise NotImplementedError
            else:
                raise NotImplementedError

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]

    @staticmethod
    def compute_approx_vram_consumption(
        patch_size,
        num_pool_per_axis,
        base_num_features,
        max_num_features,
        num_modalities,
        num_classes,
        pool_op_kernel_sizes,
        deep_supervision=False,
        conv_per_stage=2,
    ):
        if not isinstance(num_pool_per_axis, np.ndarray):
            num_pool_per_axis = np.array(num_pool_per_axis)

        npool = len(pool_op_kernel_sizes)
        map_size = np.array(patch_size).astype(np.float64)

        tmp = np.int64(
            (conv_per_stage * 2 + 1) * np.prod(map_size, dtype=np.int64) * base_num_features
            + num_modalities * np.prod(map_size, dtype=np.int64)
            + num_classes * np.prod(map_size, dtype=np.int64)
        )

        num_feat = base_num_features
        for p in range(npool):
            for pi in range(len(num_pool_per_axis)):
                map_size[pi] /= pool_op_kernel_sizes[p][pi]
            num_feat = min(num_feat * 2, max_num_features)
            num_blocks = (conv_per_stage * 2 + 1) if p < (npool - 1) else conv_per_stage
            tmp += num_blocks * np.prod(map_size, dtype=np.int64) * num_feat
            if deep_supervision and p < (npool - 2):
                tmp += np.prod(map_size, dtype=np.int64) * num_classes
        return tmp


default_dict = {
    "base_num_features": 32,
    "conv_per_stage": 2,
    "initial_lr": 0.01,
    "lr_scheduler": None,
    "lr_scheduler_eps": 0.001,
    "lr_scheduler_patience": 30,
    "lr_threshold": 1e-06,
    "max_num_epochs": 1000,
    "net_conv_kernel_sizes": [[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
    "net_num_pool_op_kernel_sizes": [[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    "net_pool_per_axis": [4, 5, 5],
    "num_batches_per_epoch": 250,
    "num_classes": 17,
    "num_input_channels": 1,
    "transpose_backward": [0, 1, 2],
    "transpose_forward": [0, 1, 2],
}


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    inputs = sanitize_logits(inputs).sigmoid()
    targets = sanitize_targets(targets)

    inputs = inputs.flatten(1)
    targets = targets.flatten(1)

    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1.0) / (denominator + 1.0)
    return torch.nan_to_num(loss, nan=1.0, posinf=1.0, neginf=1.0)


batch_dice_loss_jit = torch.jit.script(batch_dice_loss)


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    inputs = sanitize_logits(inputs)
    targets = sanitize_targets(targets)

    hw = max(inputs.shape[1], 1)

    with autocast("cuda", enabled=False):
        pos = F.binary_cross_entropy_with_logits(
            inputs.float(), torch.ones_like(inputs.float()), reduction="none"
        )
        neg = F.binary_cross_entropy_with_logits(
            inputs.float(), torch.zeros_like(inputs.float()), reduction="none"
        )

    pos = torch.nan_to_num(pos, nan=0.0, posinf=100.0, neginf=100.0)
    neg = torch.nan_to_num(neg, nan=0.0, posinf=100.0, neginf=100.0)

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum("nc,mc->nm", neg, (1 - targets))
    loss = loss / hw
    return torch.nan_to_num(loss, nan=1e6, posinf=1e6, neginf=1e6)


batch_sigmoid_ce_loss_jit = torch.jit.script(batch_sigmoid_ce_loss)


class HungarianMatcher3D(nn.Module):
    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

    def compute_cls_loss(self, inputs, targets):
        raise NotImplementedError

    def compute_dice_loss(self, inputs, targets, eps=1e-6):
        inputs = sanitize_logits(inputs).sigmoid()
        targets = sanitize_targets(targets)

        inputs = inputs.flatten(1)
        targets = targets.flatten(1)

        num_masks = max(len(inputs), 1)
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        dice = (numerator + eps) / (denominator + eps)
        dice = torch.nan_to_num(dice, nan=0.0, posinf=0.0, neginf=0.0)
        loss = 1 - dice
        loss = torch.nan_to_num(loss, nan=1.0, posinf=1.0, neginf=1.0)
        return loss.sum() / num_masks

    def compute_ce_loss(self, inputs, targets):
        with autocast("cuda", enabled=False):
            inputs = sanitize_logits(inputs)
            targets = sanitize_targets(targets)

            inputs = inputs.flatten(1)
            targets = targets.flatten(1)

            if inputs.shape[0] == 0:
                return inputs.new_tensor(0.0)

            loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
            loss = torch.nan_to_num(loss, nan=0.0, posinf=100.0, neginf=100.0)
            loss = loss.mean(1).sum() / inputs.shape[0]
            loss = torch.nan_to_num(loss, nan=0.0, posinf=100.0, neginf=100.0)
        return loss

    def compute_dice_loss(self, inputs, targets, eps=1e-6):
        inputs = sanitize_logits(inputs).sigmoid()
        targets = sanitize_targets(targets)

        inputs = inputs.flatten(1)
        targets = targets.flatten(1)

        num_masks = max(len(inputs), 1)
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        dice = (numerator + eps) / (denominator + eps)
        dice = torch.nan_to_num(dice, nan=0.0, posinf=0.0, neginf=0.0)
        loss = 1 - dice
        return torch.nan_to_num(loss.sum() / num_masks, nan=1.0, posinf=1.0, neginf=1.0)

    def compute_ce(self, inputs, targets):
        with autocast("cuda", enabled=False):
            inputs = sanitize_logits(inputs)
            targets = sanitize_targets(targets)

            inputs = inputs.flatten(1)
            targets = targets.flatten(1)
            hw = max(inputs.shape[1], 1)

            pos = F.binary_cross_entropy_with_logits(
                inputs, torch.ones_like(inputs), reduction="none"
            )
            neg = F.binary_cross_entropy_with_logits(
                inputs, torch.zeros_like(inputs), reduction="none"
            )

            pos = torch.nan_to_num(pos, nan=0.0, posinf=100.0, neginf=100.0)
            neg = torch.nan_to_num(neg, nan=0.0, posinf=100.0, neginf=100.0)

            loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
                "nc,mc->nm", neg, (1 - targets)
            )
            loss = loss / hw
            loss = torch.nan_to_num(loss, nan=1e6, posinf=1e6, neginf=1e6)
        return loss

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        indices = []

        for b in range(bs):
            out_logits = sanitize_logits(outputs["pred_logits"][b])
            out_prob = sanitize_probs(out_logits.softmax(-1))
            out_mask = sanitize_logits(outputs["pred_masks"][b])

            tgt_ids = targets[b]["labels"]
            tgt_mask = sanitize_targets(targets[b]["masks"].to(out_mask))

            if tgt_ids.numel() == 0 or tgt_mask.shape[0] == 0:
                indices.append((
                    torch.empty(0, dtype=torch.int64),
                    torch.empty(0, dtype=torch.int64)
                ))
                continue

            tgt_ids = tgt_ids.clamp(min=0, max=out_prob.shape[-1] - 1)

            with autocast("cuda", enabled=False):
                cost_class = -out_prob[:, tgt_ids]
                cost_dice = self.compute_dice(out_mask, tgt_mask)
                cost_mask = self.compute_ce(out_mask, tgt_mask)

            C = self.cost_class * cost_class + self.cost_mask * cost_mask + self.cost_dice * cost_dice
            C = torch.nan_to_num(C, nan=1e6, posinf=1e6, neginf=-1e6)
            C = C.reshape(num_queries, -1).cpu().numpy()

            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets):
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            f"cost_class: {self.cost_class}",
            f"cost_mask: {self.cost_mask}",
            f"cost_dice: {self.cost_dice}",
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)

    def _get_src_permutation_idx(self, indices):
        src_list = [src for (src, _) in indices if src.numel() > 0]
        if len(src_list) == 0:
            return (
                torch.empty(0, dtype=torch.int64),
                torch.empty(0, dtype=torch.int64),
            )
        batch_idx = torch.cat([
            torch.full_like(src, i) for i, (src, _) in enumerate(indices) if src.numel() > 0
        ])
        src_idx = torch.cat(src_list)
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        tgt_list = [tgt for (_, tgt) in indices if tgt.numel() > 0]
        if len(tgt_list) == 0:
            return (
                torch.empty(0, dtype=torch.int64),
                torch.empty(0, dtype=torch.int64),
            )
        batch_idx = torch.cat([
            torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices) if tgt.numel() > 0
        ])
        tgt_idx = torch.cat(tgt_list)
        return batch_idx, tgt_idx


def compute_loss_hungarian(
    outputs, targets, idx, matcher, num_classes,
    point_rend=False, num_points=12544,
    oversample_ratio=3.0, importance_sample_ratio=0.75,
    no_object_weight=None, cost_weight=None
):
    if cost_weight is None:
        cost_weight = [2, 5, 5]

    indices = matcher(outputs, targets)
    src_idx = matcher._get_src_permutation_idx(indices)
    tgt_idx = matcher._get_tgt_permutation_idx(indices)

    num_total_targets = sum(len(t["masks"]) for t in targets)

    src_logits = sanitize_logits(outputs["pred_logits"])
    target_classes = torch.full(
        src_logits.shape[:2], num_classes, dtype=torch.int64, device=src_logits.device
    )

    if num_total_targets > 0 and src_idx[0].numel() > 0:
        target_classes_o = torch.cat([t["labels"] for t in targets], dim=0)
        target_classes_o = target_classes_o.clamp(min=0, max=num_classes)
        target_classes[src_idx] = target_classes_o

    if no_object_weight is not None:
        empty_weight = torch.ones(num_classes + 1, device=src_logits.device)
        empty_weight[-1] = no_object_weight
        loss_cls = F.cross_entropy(src_logits.transpose(1, 2), target_classes, empty_weight)
    else:
        loss_cls = F.cross_entropy(src_logits.transpose(1, 2), target_classes)

    loss_cls = torch.nan_to_num(loss_cls, nan=0.0, posinf=100.0, neginf=100.0)

    matched_count = src_idx[0].numel()
    if num_total_targets == 0 or matched_count == 0:
        return (cost_weight[0] / 10.0) * loss_cls

    src_masks = sanitize_logits(outputs["pred_masks"])[src_idx]
    target_masks = sanitize_targets(torch.cat([t["masks"] for t in targets], dim=0).to(src_masks))

    if src_masks.shape[0] == 0 or target_masks.shape[0] == 0:
        return (cost_weight[0] / 10.0) * loss_cls

    src_masks = src_masks[:, None]
    target_masks = target_masks[:, None]

    if point_rend:
        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks.float(),
                lambda logits: calculate_uncertainty(logits),
                num_points,
                oversample_ratio,
                importance_sample_ratio,
            )
            point_labels = point_sample_3d(
                target_masks.float(),
                point_coords.float(),
                align_corners=False,
                mode="bilinear",
            ).squeeze(1)

        point_logits = point_sample_3d(
            src_masks.float(),
            point_coords.float(),
            align_corners=False,
            mode="bilinear",
        ).squeeze(1)

        src_masks = sanitize_logits(point_logits)
        target_masks = sanitize_targets(point_labels)

    loss_mask_ce = matcher.compute_ce_loss(src_masks, target_masks)
    loss_mask_dice = matcher.compute_dice_loss(src_masks, target_masks)

    loss = (
        (cost_weight[0] / 10.0) * loss_cls
        + (cost_weight[1] / 10.0) * loss_mask_ce
        + (cost_weight[2] / 10.0) * loss_mask_dice
    )
    return torch.nan_to_num(loss, nan=0.0, posinf=100.0, neginf=100.0)


def point_sample_3d(input, point_coords, **kwargs):
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2).unsqueeze(2)  # (N, P, 1, 1, 3)

    input = torch.nan_to_num(input.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    point_coords = torch.nan_to_num(point_coords.float(), nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)

    if add_dim:
        output = output.squeeze(3).squeeze(3)
    return output


def calculate_uncertainty(logits):
    assert logits.shape[1] == 1
    logits = sanitize_logits(logits)
    return -torch.abs(logits)


def get_uncertain_point_coords_with_randomness(
    coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio
):
    assert oversample_ratio >= 1
    assert 0 <= importance_sample_ratio <= 1

    n_dim = 3
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)

    point_coords = torch.rand(num_boxes, num_sampled, n_dim, device=coarse_logits.device)
    point_logits = point_sample_3d(
        sanitize_logits(coarse_logits),
        point_coords,
        align_corners=False,
        mode="bilinear",
    )

    point_uncertainties = torch.nan_to_num(
        uncertainty_func(point_logits), nan=-1e6, posinf=-1e6, neginf=-1e6
    )

    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points

    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]

    point_coords = point_coords.view(-1, n_dim)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, n_dim
    )

    if num_random_points > 0:
        point_coords = torch.cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, n_dim, device=coarse_logits.device),
            ],
            dim=1,
        )

    return point_coords