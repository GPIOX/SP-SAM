# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial
from torch.nn import functional as F

from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer
from .tiny_vit_sam import TinyViT


def build_sam_vit_h(
        image_size=512, 
        num_classes=2, 
        pixel_mean=[123.675, 116.28, 103.53], 
        pixel_std=[58.395, 57.12, 57.375],    
        checkpoint=None
):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
        image_size=image_size,
        num_classes=num_classes,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        type='h'
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(
        image_size=512, 
        num_classes=2, 
        pixel_mean=[123.675, 116.28, 103.53], 
        pixel_std=[58.395, 57.12, 57.375],    
        checkpoint=None
    ):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
        num_classes=num_classes,
        image_size=image_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        type='l'
    )


def build_sam_vit_b(
        image_size=512, 
        num_classes=2, 
        pixel_mean=[123.675, 116.28, 103.53], 
        pixel_std=[58.395, 57.12, 57.375],    
        checkpoint=None
    ):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        num_classes=num_classes,
        image_size=image_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        type='b'
    )


def build_sam_vit_t(        
        image_size=512, 
        num_classes=2, 
        pixel_mean=[123.675, 116.28, 103.53], 
        pixel_std=[58.395, 57.12, 57.375],    
        checkpoint=None,
        type='t'
    ):
    prompt_embed_dim = 256
    # image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
            image_encoder=TinyViT(img_size=image_size, in_chans=3, num_classes=1000,
                embed_dims=[64, 128, 160, 320],
                depths=[2, 2, 6, 2],
                num_heads=[2, 4, 5, 10],
                window_sizes=[7, 7, 14, 7],
                mlp_ratio=4.,
                drop_rate=0.,
                drop_path_rate=0.0,
                use_checkpoint=False,
                mbconv_expand_ratio=4.0,
                local_conv_size=3,
                layer_lr_decay=0.8
            ),
            prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
            ),
            mask_decoder=MaskDecoder(
                    num_multimask_outputs=num_classes,
                    transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
            ),
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            type=type
        )

    sam.train()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        # try:
        # sam.load_state_dict(state_dict)
        # except:
        new_state_dict = mobile_sam_load_from(sam, state_dict, image_size, vit_patch_size, type)
        sam.load_state_dict(new_state_dict)
    return sam, image_embedding_size


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
    "vit_t": build_sam_vit_t,
}


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    num_classes,
    image_size,
    pixel_mean,
    pixel_std,
    checkpoint=None,
    type='b'
):
    prompt_embed_dim = 256
    image_size = image_size
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            # num_multimask_outputs=3,
            num_multimask_outputs = num_classes,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        type=type
    )
    sam.train()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        # try:
        #     sam.load_state_dict(state_dict)
        # except:
        new_state_dict = load_from(sam, state_dict, image_size, vit_patch_size, type)
        sam.load_state_dict(new_state_dict)
    return sam, image_embedding_size
    # return sam

def load_from(sam, state_dict, image_size, vit_patch_size, type):
    """
    从预训练模型中加载参数到SAM模型。
    
    此函数主要执行以下操作：
    1. 过滤掉不兼容或不需要的参数；
    2. 调整位置嵌入（pos_embed）以适应新的图像大小；
    3. 更新SAM模型的状态字典。

    参数:
    - sam: Segment Anything Model，需要加载参数的SAM模型实例。
    - state_dict: 预训练模型的状态字典。
    - image_size: 输入图像的目标大小。
    - vit_patch_size: Vision Transformer的patch大小。

    返回:
    - 更新后的SAM模型状态字典。
    """
    # 获取SAM模型当前的状态字典
    sam_dict = sam.state_dict()
    
    # 定义不需要的参数键列表
    except_keys = ['mask_tokens', 'output_hypernetworks_mlps', 'iou_prediction_head']
    
    # 筛选出需要的参数，排除except_keys中指定的键
    new_state_dict = {k: v for k, v in state_dict.items() if
                      k in sam_dict.keys() and except_keys[0] not in k and except_keys[1] not in k and except_keys[2] not in k}
    
    # 获取位置嵌入
    pos_embed = new_state_dict['image_encoder.pos_embed']
    
    # 计算目标位置嵌入的尺寸
    token_size = int(image_size // vit_patch_size)
    
    # 如果位置嵌入的尺寸不匹配，则进行插值调整
    if pos_embed.shape[1] != token_size:
        # 调整位置嵌入形状以便进行插值
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
        new_state_dict['image_encoder.pos_embed'] = pos_embed
        
        # 对全局相对位置参数进行插值调整
        rel_pos_keys = [k for k in sam_dict.keys() if 'rel_pos' in k]
        
        global_rel_pos_keys_dict = {
            'b': [k for k in rel_pos_keys if '2' in k or '5' in  k or '8' in k or '11' in k],
            'l': [k for k in rel_pos_keys if '.5' in k or '11' in  k or '17' in k or '23' in k],
            'h': [k for k in rel_pos_keys if '.7' in k or '15' in  k or '23' in k or '31' in k],
        }
        
        # global_rel_pos_keys = [k for k in rel_pos_keys if '2' in k or '5' in  k or '8' in k or '11' in k]
        global_rel_pos_keys = global_rel_pos_keys_dict[type]# [k for k in rel_pos_keys if '.5' in k or '11' in  k or '17' in k or '23' in k]
        for k in global_rel_pos_keys:
            rel_pos_params = new_state_dict[k]
            h, w = rel_pos_params.shape
            rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
            rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
            new_state_dict[k] = rel_pos_params[0, 0, ...]
    
    # 更新SAM模型的状态字典
    sam_dict.update(new_state_dict)
    
    # 返回更新后的状态字典
    return sam_dict

def mobile_sam_load_from(sam, state_dict, image_size, vit_patch_size, type):
    """
    从预训练模型中加载参数到SAM模型。
    
    此函数主要执行以下操作：
    1. 过滤掉不兼容或不需要的参数；
    2. 调整位置嵌入（pos_embed）以适应新的图像大小；
    3. 更新SAM模型的状态字典。

    参数:
    - sam: Segment Anything Model，需要加载参数的SAM模型实例。
    - state_dict: 预训练模型的状态字典。
    - image_size: 输入图像的目标大小。
    - vit_patch_size: Vision Transformer的patch大小。

    返回:
    - 更新后的SAM模型状态字典。
    """
    # 获取SAM模型当前的状态字典
    sam_dict = sam.state_dict()
    
    # 定义不需要的参数键列表
    except_keys = ['mask_tokens', 'output_hypernetworks_mlps', 'iou_prediction_head']
    
    # 筛选出需要的参数，排除except_keys中指定的键
    new_state_dict = {k: v for k, v in state_dict.items() if
                      k in sam_dict.keys() and except_keys[0] not in k and except_keys[1] not in k and except_keys[2] not in k}

    # 更新SAM模型的状态字典
    sam_dict.update(new_state_dict)
    
    # 返回更新后的状态字典
    return sam_dict
