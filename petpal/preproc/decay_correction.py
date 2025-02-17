"""Decay Correction Module.

Provides functions for undo-ing decay correction and recalculating it."""

import warnings
import math

import numpy as np

from ..utils import image_io

def undo_decay_correction(input_image_path: str,
                          output_image_path: str,
                          metadata_dict: dict = None,
                          verbose: bool = False) -> np.ndarray:
    """Uses decay factors from the metadata for an image to remove decay correction for each frame.

    This function expects to find decay factors in the .json sidecar file, or the metadata_dict, if given. If there are
    no decay factors (either under the key 'DecayFactor' or the BIDS-required 'DecayCorrectionFactor') listed, it may
    result in unexpected behavior. In addition to returning a np.ndarray containing the "decay uncorrected" data, the
    function writes an image to output_image_path.

    Args:
        input_image_path (str): Path to input (.nii.gz or .nii) image. A .json sidecar file should exist in the same
             directory as the input image.
        output_image_path (str): Path to output (.nii.gz or .nii) output image. If None, no image will be written.
        metadata_dict (dict): Optional dictionary to use instead of corresponding .json sidecar. If not specified
             (default behavior), function will try to use sidecar .json in the same directory as input_image_path
        verbose (bool): If true, prints more information during processing. Default is False.

    Returns:
        np.ndarray: Image Data with decay correction reversed."""



    nifti_image = image_io.safe_load_4dpet_nifti(filename=input_image_path)
    if metadata_dict:
        json_data = metadata_dict
    else:
        json_data = image_io.load_metadata_for_nifti_with_same_filename(image_path=input_image_path)
    frame_info = image_io.get_frame_timing_info_for_nifti(image_path=input_image_path)
    decay_factors = frame_info['decay']

    image_data = nifti_image.get_fdata()

    for frame_num, decay_factor in  enumerate(decay_factors):
        image_data[..., frame_num] = image_data[..., frame_num] / decay_factor

    if output_image_path is not None:
        image_loader = image_io.ImageIO(verbose=verbose)
        output_image = image_loader.extract_np_to_nibabel(image_array=image_data,
                                                          header=nifti_image.header,
                                                          affine=nifti_image.affine)

        image_loader.save_nii(image=output_image,
                              out_file=output_image_path)

        # This guarantees the type is unchanged, unlike [1]*len(decay_factors)
        json_data['DecayFactor'] = list(np.ones_like(decay_factors))
        json_data['ImageDecayCorrected'] = "false"
        output_json_path = image_io._gen_meta_data_filepath_for_nifti(nifty_path=output_image_path)
        image_io.write_dict_to_json(meta_data_dict=json_data,
                                    out_path=output_json_path)

    return image_data

def decay_correct(input_image_path: str,
                  output_image_path: str,
                  verbose: bool = False) -> np.ndarray:
    r"""Recalculate decay_correction for nifti image based on frame reference times.

    This function will compute frame reference times based on frame time starts and frame durations (both of which
    are required by BIDS. These reference times are used in the following equation to determine the decay factor for
    each frame. For more information, refer to Turku Pet Centre's materials at
    https://www.turkupetcentre.net/petanalysis/decay.html

    .. math::
        decay\_factor = \exp(\lambda*t)

    where :math:`\lambda=\log(2)/T_{1/2}` is the decay constant of the radio isotope and depends on its half-life and
    `t` is the frame's reference time with respect to TimeZero (ideally, injection time).

    Args:
        input_image_path (str): Path to input (.nii.gz or .nii) image. A .json sidecar file should exist in the same
             directory as the input image.
        output_image_path (str): Path to output (.nii.gz or .nii) output image.
        verbose (bool): If true, prints more information during processing. Default is False.
    """
    half_life = image_io.get_half_life_from_nifti(image_path=input_image_path) # Note: this will need to be handled
    # intelligently if we change this function to take arrays (and provide decorators for reading from files).

    json_data = image_io.load_metadata_for_nifti_with_same_filename(image_path=input_image_path)

    nifti_image = image_io.safe_load_4dpet_nifti(filename=input_image_path)
    frame_info = image_io.get_frame_timing_info_for_nifti(image_path=input_image_path)
    frame_times_start = frame_info['start']
    frame_durations = frame_info['duration']
    frame_reference_times = [start+(duration/2) for start, duration in zip(frame_times_start, frame_durations)]

    original_decay_factors = np.asarray(frame_info['decay'])
    if np.any(original_decay_factors != 1):
        raise ValueError(f'Decay Factors other than 1 found in metadata for {input_image_path}. This likely means the '
                         f'image has not had its previous decay correction undone. Try running undo_decay_correction '
                         f'before running this function to avoid decay correcting an image more than once.')

    image_data = nifti_image.get_fdata()
    new_decay_factors = []
    for frame_num, frame_reference_time in enumerate(frame_reference_times):
        decay_factor = math.exp(((math.log(2) / half_life) * frame_reference_time))
        image_data[..., frame_num] = image_data[..., frame_num] * decay_factor
        new_decay_factors.append(decay_factor)

    if output_image_path is not None:
        image_loader = image_io.ImageIO(verbose=verbose)
        output_image = image_loader.extract_np_to_nibabel(image_array=image_data,
                                                          header=nifti_image.header,
                                                          affine=nifti_image.affine)

        image_loader.save_nii(image=output_image,
                              out_file=output_image_path)

        json_data['DecayFactor'] = new_decay_factors
        json_data['ImageDecayCorrected'] = "true"
        json_data['ImageDecayCorrectionTime'] = 0 # We always use BIDS TimeZero for decay correction, so 0 seconds w.r.t. it
        output_json_path = image_io._gen_meta_data_filepath_for_nifti(nifty_path=output_image_path)
        image_io.write_dict_to_json(meta_data_dict=json_data,
                                    out_path=output_json_path)

    return image_data

