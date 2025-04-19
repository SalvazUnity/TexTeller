"""
Custom augraphy pipeline for training

This file implements a custom augraphy data augmentation pipeline. We found that using augraphy's
default pipeline can cause significant degradation to formula images, potentially losing semantic
information. Therefore, we carefully selected several common augmentation effects,
adjusting their parameters and combination methods to preserve the original semantic information
of the images as much as possible.
"""

from augraphy import (
    InkColorSwap,
    LinesDegradation,
    OneOf,
    Dithering,
    InkBleed,
    InkShifter,
    NoiseTexturize,
    BrightnessTexturize,
    ColorShift,
    DirtyDrum,
    LightingGradient,
    Brightness,
    Gamma,
    SubtleNoise,
    Jpeg,
    AugraphyPipeline,
)
import random


def get_custom_augraphy():
    pre_phase = []

    ink_phase = [
        InkColorSwap(
            ink_swap_color="random",
            ink_swap_sequence_number_range=(5, 10),
            ink_swap_min_width_range=(2, 3),
            ink_swap_max_width_range=(100, 120),
            ink_swap_min_height_range=(2, 3),
            ink_swap_max_height_range=(100, 120),
            ink_swap_min_area_range=(10, 20),
            ink_swap_max_area_range=(400, 500),
            p=0.2,
        ),
        LinesDegradation(
            line_roi=(0.0, 0.0, 1.0, 1.0),
            line_gradient_range=(32, 255),
            line_gradient_direction=(0, 2),
            line_split_probability=(0.2, 0.4),
            line_replacement_value=(250, 255),
            line_min_length=(30, 40),
            line_long_to_short_ratio=(5, 7),
            line_replacement_probability=(0.4, 0.5),
            line_replacement_thickness=(1, 3),
            p=0.2,
        ),
        #  ============================
        OneOf(
            [
                Dithering(
                    dither="floyd-steinberg",
                    order=(3, 5),
                ),
                InkBleed(
                    intensity_range=(0.1, 0.2),
                    kernel_size=random.choice([(7, 7), (5, 5), (3, 3)]),
                    severity=(0.4, 0.6),
                ),
            ],
            p=0.2,
        ),
        #  ============================
        #  ============================
        InkShifter(
            text_shift_scale_range=(18, 27),
            text_shift_factor_range=(1, 4),
            text_fade_range=(0, 2),
            blur_kernel_size=(5, 5),
            blur_sigma=0,
            noise_type="perlin",
            p=0.2,
        ),
        #  ============================
    ]

    paper_phase = [
        NoiseTexturize(
            sigma_range=(3, 10),
            turbulence_range=(2, 5),
            texture_width_range=(300, 500),
            texture_height_range=(300, 500),
            p=0.2,
        ),
        BrightnessTexturize(texturize_range=(0.9, 0.99), deviation=0.03, p=0.2),
    ]

    post_phase = [
        ColorShift(
            color_shift_offset_x_range=(3, 5),
            color_shift_offset_y_range=(3, 5),
            color_shift_iterations=(2, 3),
            color_shift_brightness_range=(0.9, 1.1),
            color_shift_gaussian_kernel_range=(3, 3),
            p=0.2,
        ),
        DirtyDrum(
            line_width_range=(1, 6),
            line_concentration=random.uniform(0.05, 0.15),
            direction=random.randint(0, 2),
            noise_intensity=random.uniform(0.6, 0.95),
            noise_value=(64, 224),
            ksize=random.choice([(3, 3), (5, 5), (7, 7)]),
            sigmaX=0,
            p=0.2,
        ),
        # =====================================
        OneOf(
            [
                LightingGradient(
                    light_position=None,
                    direction=None,
                    max_brightness=255,
                    min_brightness=0,
                    mode="gaussian",
                    linear_decay_rate=None,
                    transparency=None,
                ),
                Brightness(
                    brightness_range=(0.9, 1.1),
                    min_brightness=0,
                    min_brightness_value=(120, 150),
                ),
                Gamma(
                    gamma_range=(0.9, 1.1),
                ),
            ],
            p=0.2,
        ),
        # =====================================
        # =====================================
        OneOf(
            [
                SubtleNoise(
                    subtle_range=random.randint(5, 10),
                ),
                Jpeg(
                    quality_range=(70, 95),
                ),
            ],
            p=0.2,
        ),
        # =====================================
    ]

    pipeline = AugraphyPipeline(
        ink_phase=ink_phase,
        paper_phase=paper_phase,
        post_phase=post_phase,
        pre_phase=pre_phase,
        log=False,
    )

    return pipeline
