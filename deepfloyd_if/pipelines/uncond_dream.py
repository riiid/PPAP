# -*- coding: utf-8 -*-
from datetime import datetime

import torch


def uncond_dream(
    if_I,
    if_II=None,
    if_III=None,
    *,
    seed=None,
    batch_size=1,
    aspect_ratio='1:1',
    if_I_kwargs=None,
    if_II_kwargs=None,
    if_III_kwargs=None,
    progress=True,
    return_tensors=False,
    disable_watermark=False,
):
    """
    Do unconditional image generation with deepfloyd-IF.

    :param optional dict if_I_kwargs:
        "dynamic_thresholding_p": 0.95, [0.5, 1.0] it controls color saturation on high cfg values
        "dynamic_thresholding_c": 1.5, [1.0, 15.0] clips the limiter to avoid greyish images on high limiter values
        "sample_timestep_respacing": "150", see available modes IFBaseModule.respacing_modes or use custom
    :param optional dict if_II_kwargs:
        "dynamic_thresholding_p": 0.95, [0.5, 1.0] it controls color saturation on high cfg values
        "dynamic_thresholding_c": 1.0, [1.0, 15.0] clips the limiter to avoid greyish images on high limiter values
        "aug_level": 0.25, [0.0, 1.0] adds additional augmentation to generate more realistic images
        "sample_timestep_respacing": "smart50", see available modes IFBaseModule.respacing_modes or use custom

    :param deepfloyd_if.modules.IFStageI if_I: obj
    :param deepfloyd_if.modules.IFStageII if_II: obj
    :param deepfloyd_if.modules.IFStageIII if_III: obj
    :param int seed: int, in case None will use random value
    :param aspect_ratio:
    :param progress:
    :return:
    """
    if seed is None:
        seed = int((datetime.utcnow().timestamp() * 10 ** 6) % (2 ** 32 - 1))
    # First stage generation
    if_I.seed_everything(seed)

    if_I_kwargs = if_I_kwargs or {}
    if_I_kwargs['seed'] = seed
    if_I_kwargs['aspect_ratio'] = aspect_ratio
    if_I_kwargs['progress'] = progress
    if_I_kwargs['batch_size'] = batch_size

    stageI_generations, _ = if_I.uncond_generation(**if_I_kwargs)
    pil_images_I = if_I.to_images(stageI_generations, disable_watermark=disable_watermark)

    result = {'I': pil_images_I}

    if if_II is not None:
        if_II_kwargs = if_II_kwargs or {}
        if_II_kwargs['low_res'] = stageI_generations
        if_II_kwargs['seed'] = seed
        if_II_kwargs['progress'] = progress
        if_II_kwargs['batch_size'] = batch_size

        stageII_generations, _meta = if_II.uncond_generation(**if_II_kwargs)
        pil_images_II = if_II.to_images(stageII_generations, disable_watermark=disable_watermark)
        result["II"] = pil_images_II
    else:
        stageII_generations = None

    if if_II is not None and if_III is not None:
        if_III_kwargs = if_III_kwargs or {}

        stageIII_generations = []
        for idx in range(len(stageII_generations)):
            if_III_kwargs['low_res'] = stageII_generations[idx:idx + 1]
            if_III_kwargs['seed'] = seed
            if_III_kwargs['progress'] = progress
            if_III_kwargs['batch_size'] = batch_size
            _stageIII_generations, _meta = if_III.uncond_generation(**if_III_kwargs)
            stageIII_generations.append(_stageIII_generations)

        stageIII_generations = torch.cat(stageIII_generations, 0)
        pil_images_III = if_III.to_images(stageIII_generations, disable_watermark=disable_watermark)
        result['III'] = pil_images_III
    else:
        stageIII_generations = None

    if return_tensors:
        return result, (stageI_generations, stageII_generations, stageIII_generations)
    else:
        return result
