"""
Provides functions for undo-ing decay correction and recalculating it.

"""

import math

import ants
import numpy as np

from ..utils import image_io
from ..utils.scan_timing import ScanTimingInfo


def undo_decay_correction(input_image: ants.ANTsImage,
                          metadata_dict: dict) -> (ants.ANTsImage, dict):
    """Uses decay factors from the metadata for an image to remove decay correction for each frame.

    This function expects to find decay factors in the metadata_dict. If there are
    no decay factors (under the BIDS-required key 'DecayCorrectionFactor') listed, it may result in unexpected behavior.

    Important:
        This function assumes metadata_dict is a BIDS-compliant dictionary for PET metadata.

    Args:
        input_image (ants.ANTSimage): Decay-corrected image
        metadata_dict (dict): Dictionary containing BIDS keys 'DecayCorrectionFactor' and 'ImageDecayCorrected'

    Returns:
        ants.ANTsImage: ANTsImage with decay correction reversed."""

    arr_to_uncorrect = input_image.numpy()
    metadata = metadata_dict
    scan_timing_obj = ScanTimingInfo.from_metadata(metadata)
    decay_factors = scan_timing_obj.decay.tolist()

    for frame_num, decay_factor in enumerate(decay_factors):
        arr_to_uncorrect[..., frame_num] /= decay_factor

    uncorrected_img = ants.from_numpy_like(data=arr_to_uncorrect,
                                           image=input_image)
    try:
        metadata['DecayCorrectionFactor'] = np.ones_like(decay_factors).tolist()
        metadata['ImageDecayCorrected'] = "false"
    except KeyError:
        raise KeyError('metadata_dict does not contain necessary BIDS keys "DecayCorrectionFactor" and '
                       '"ImageDecayCorrected". Ensure metadata is BIDS-compliant.')

    return uncorrected_img, metadata


def decay_correct(input_image: ants.ANTsImage,
                  metadata_dict: dict) -> ants.ANTsImage:
    r"""Recalculate decay_correction for nifti image based on frame reference times.

    This function will compute frame reference times based on frame time starts and frame durations (both of which
    are required by BIDS. These reference times are used in the following equation to determine the decay factor for
    each frame. For more information, refer to Turku Pet Centre's materials at
    https://www.turkupetcentre.net/petanalysis/decay.html

    .. math::
        decay\_factor = \exp(\lambda*t)

    where :math:`\lambda=\log(2)/T_{1/2}` is the decay constant of the radio isotope and depends on its half-life and
    `t` is the frame's reference time with respect to TimeZero (ideally, injection time).

    Note: BIDS 'DecayCorrectionTime' is set to 0 (seconds w.r.t. TimeZero) for the image. If this assumption doesn't
        hold, be wary of downstream effects.

    Args:
        input_image (str): Path to input (.nii.gz or .nii) image. A .json sidecar file should exist in the same
             directory as the input image.
        metadata_dict (str): Path to output (.nii.gz or .nii) output image.

    Returns:
        ants.ANTsImage: Decay-Corrected Image

    """
    try:
        half_life = metadata_dict['RadionuclideHalfLife']
    except KeyError as exc:
        raise KeyError("RadionuclideHalfLife not found in metadata dictionary") from exc

    half_life = float(half_life)
    uncorrected_img = input_image
    scan_timing_obj = ScanTimingInfo.from_metadata(metadata_dict)

    frame_reference_times = scan_timing_obj.center.tolist()
    original_decay_factors = scan_timing_obj.decay

    if np.any(original_decay_factors != 1):
        raise ValueError(f'Decay Factors other than 1 found in metadata for {input_image}. This likely means the '
                         f'image has not had its previous decay correction undone. Try running undo_decay_correction '
                         f'before running this function to avoid decay correcting an image more than once.')

    arr_to_correct = uncorrected_img.numpy()
    new_decay_factors = []
    for frame_num, frame_reference_time in enumerate(frame_reference_times):
        new_decay_factor = math.exp(((math.log(2) / half_life) * frame_reference_time))
        arr_to_correct[..., frame_num] *= new_decay_factor
        new_decay_factors.append(new_decay_factor)

    corrected_img = ants.from_numpy_like(data=arr_to_correct,
                                         image=uncorrected_img)

    metadata_dict['DecayCorrectionFactor'] = new_decay_factors
    metadata_dict['ImageDecayCorrected'] = "true"
    metadata_dict['ImageDecayCorrectionTime'] = 0

    return corrected_img, metadata_dict
