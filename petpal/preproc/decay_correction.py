"""
Provides functions for undo-ing decay correction and recalculating it.

"""

import math

import ants
import numpy as np

from ..utils import image_io
from ..utils.scan_timing import ScanTimingInfo
from ..utils.constants import HALF_LIVES


def undo_decay_correction(input_image: (ants.ANTsImage, dict)) -> (ants.ANTsImage, dict):
    """Uses decay factors from the metadata for an image to remove decay correction for each frame.

    This function expects to find decay factors in the .json sidecar file, or the metadata_dict, if given. If there are
    no decay factors (either under the key 'DecayFactor' or the BIDS-required 'DecayCorrectionFactor') listed, it may
    result in unexpected behavior. In addition to returning an ANTsImage containing the "decay uncorrected" data, the
    function writes an image to output_image_path, unless it is passed as 'None'.

    Args:
        input_image (str): Path to input (.nii.gz or .nii) image. A .json sidecar file should exist in the same
             directory as the input image.

    Returns:
        ants.ANTsImage: ANTsImage with decay correction reversed."""

    decay_corrected_img = input_image[0]
    json_data = input_image[1]

    frame_info = ScanTimingInfo.from_metadata(metadata_dict=json_data)
    decay_factors = frame_info.decay

    uncorrected_arr = decay_corrected_img.numpy()

    for frame_num, decay_factor in enumerate(decay_factors):
        uncorrected_arr[..., frame_num] /= decay_factor

    uncorrected_img = ants.from_numpy_like(data=uncorrected_arr,
                                           image=decay_corrected_img)

    json_data['DecayFactor'] = list(np.ones_like(decay_factors))
    json_data['ImageDecayCorrected'] = False

    return uncorrected_img, json_data


def decay_correct(input_image: (ants.ANTsImage, dict)) -> (ants.ANTsImage, dict):
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

    Returns:
        ants.ANTsImage: Decay-Corrected Image

    """

    json_data = input_image[1]

    # Do this more elegantly before merging PR.
    try:
        radionuclide = json_data['TracerRadionuclide'].lower().replace("-", "")
    except KeyError as exc:
        raise KeyError("Required BIDS metadata field 'TracerRadionuclide' not found.") from exc

    half_life = HALF_LIVES[radionuclide]
    ########################################

    uncorrected_img = input_image[0]

    frame_info = ScanTimingInfo.from_metadata(metadata_dict=json_data)
    frame_reference_times = np.asarray(frame_info.start + frame_info.duration / 2.0, float).tolist()

    if np.any(frame_info.decay != 1):
        raise ValueError(f'Decay Correction Factors other than 1 found in metadata. This likely '
                         f'means the image has already been decay-corrected. Try running undo_decay_correction '
                         f'before running this function to avoid decay correcting an image more than once.')

    corrected_arr = uncorrected_img.numpy()
    new_decay_factors = []
    for frame_num, frame_reference_time in enumerate(frame_reference_times):
        new_decay_factor = math.exp(((math.log(2) / half_life) * frame_reference_time))
        corrected_arr[..., frame_num] *= new_decay_factor
        new_decay_factors.append(new_decay_factor)

    corrected_img = ants.from_numpy_like(data=corrected_arr,
                                         image=uncorrected_img)

    json_data['DecayFactor'] = new_decay_factors
    json_data['ImageDecayCorrected'] = "true"
    json_data['ImageDecayCorrectionTime'] = 0
    json_data['FrameReferenceTime'] = frame_reference_times

    return corrected_img, json_data
