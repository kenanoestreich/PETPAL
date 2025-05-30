"""
Provides tools to register PET images to anatomical or atlas space. Wrapper for
ANTs and FSL registration software.
"""
from typing import Union

import ants
import fsl.wrappers
import nibabel
import numpy as np
from nibabel.processing import resample_from_to

from .image_operations_4d import determine_motion_target
from ..utils import image_io
from ..utils.useful_functions import check_physical_space_for_ants_image_pair


def register_pet_to_pet(input_image_path: str,
                        output_image_path: str,
                        reference_pet_image_path: str) -> ants.ANTsImage:
    """
    Compute weighted series sum images for input and reference pet images, then register input to reference.

    Args:
        input_image_path (str): Path to input image (i.e. moving 4D PET Image)
        output_image_path (str): Path to write output image. If None is given, no image will be written.
        reference_pet_image_path (str): Path to reference image (i.e. fixed 4D PET Image).

    Returns:
        ants.ANTsImage: ANTsImage containing input image registered to reference image.
    """

    half_life = image_io.get_half_life_from_nifti(image_path=input_image_path)

    wss_input = determine_motion_target(motion_target_option='weighted_series_sum',
                                        input_image_4d_path=input_image_path,
                                        half_life=half_life)
    wss_reference = determine_motion_target(motion_target_option='weighted_series_sum',
                                            input_image_4d_path=reference_pet_image_path,
                                            half_life=half_life)

    wss_input_ants = ants.image_read(wss_input)
    wss_reference_ants = ants.image_read(wss_reference)
    input_ants = ants.image_read(input_image_path)

    registration_transform = ants.registration(fixed=wss_reference_ants,
                                               moving=wss_input_ants,
                                               type_of_transform='DenseRigid',
                                               write_composite_transform=True)
    registered_ants_image = ants.apply_transforms(moving=input_ants,
                                                  fixed=wss_reference_ants,
                                                  transformlist=registration_transform['fwdtransforms'],
                                                  interpolator='linear',
                                                  imagetype=3)

    if output_image_path is not None:
        ants.image_write(registered_ants_image, output_image_path)
        image_io.safe_copy_meta(input_image_path=input_image_path,
                                out_image_path=output_image_path)

    return registered_ants_image


def register_pet(input_reg_image_path: str,
                 out_image_path: str,
                 reference_image_path: str,
                 motion_target_option: Union[str, tuple],
                 verbose: bool,
                 type_of_transform: str = 'DenseRigid',
                 half_life: float = None,
                 **kwargs):
    """
    Computes and runs rigid registration of 4D PET image series to 3D anatomical image, typically
    a T1 MRI. Runs rigid registration module from Advanced Normalisation Tools (ANTs) with  default
    inputs. Will upsample PET image to the resolution of anatomical imaging.

    Args:
        input_reg_image_path (str): Path to a .nii or .nii.gz file containing a 4D
            PET image to be registered to anatomical space.
        reference_image_path (str): Path to a .nii or .nii.gz file containing a 3D
            anatomical image to which PET image is registered.
        motion_target_option (str | tuple): Target image for computing
            transformation. See :meth:`determine_motion_target`.
        type_of_transform (str): Type of transform to perform on the PET image, must be one of 
            antspy's transformation types, i.e. 'DenseRigid' or 'Translation'. Any transformation 
            type that uses >6 degrees of freedom is not recommended, use with caution. See 
            :py:func:`ants.registration`.
        out_image_path (str): Path to a .nii or .nii.gz file to which the registered PET series
            is written.
        verbose (bool): Set to ``True`` to output processing information.
        kwargs (keyword arguments): Additional arguments passed to :py:func:`ants.registration`.
    """
    motion_target = determine_motion_target(motion_target_option=motion_target_option,
                                            input_image_4d_path=input_reg_image_path,
                                            half_life=half_life)
    motion_target_image = ants.image_read(motion_target)
    mri_image = ants.image_read(reference_image_path)
    pet_image_ants = ants.image_read(input_reg_image_path)
    xfm_output = ants.registration(moving=motion_target_image,
                                   fixed=mri_image,
                                   type_of_transform=type_of_transform,
                                   write_composite_transform=True,
                                   **kwargs)
    if verbose:
        print(f'Registration computed transforming image {motion_target} to '
              f'{reference_image_path} space')

    if pet_image_ants.dimension == 4:
        dim = 3
    else:
        dim = 0

    xfm_apply = ants.apply_transforms(moving=pet_image_ants,
                                      fixed=mri_image,
                                      transformlist=xfm_output['fwdtransforms'],
                                      interpolator='linear',
                                      imagetype=dim)
    if verbose:
        print(f'Registration applied to {input_reg_image_path}')

    ants.image_write(xfm_apply, out_image_path)
    if verbose:
        print(f'Transformed image saved to {out_image_path}')

    image_io.safe_copy_meta(input_image_path=input_reg_image_path, out_image_path=out_image_path)


def warp_pet_to_atlas(input_image_path: str,
                      anat_image_path: str,
                      atlas_image_path: str,
                      type_of_transform: str = 'SyN',
                      **kwargs) -> ants.ANTsImage:
    """Warp a (3D or 4D) PET image (in anatomical space) to atlas space using an anatomical image.


    Args:
        input_image_path (str): Path to PET Image to be registered to atlas. Must be in
            anatomical space (i.e. same space as anat_image_path image). May be 3D or 4D.
        anat_image_path (str): Path to anatomical image used to compute registration to atlas space.
        atlas_image_path (str): Path to atlas to which input image is warped.
        type_of_transform (str): Type of non-linear transform applied to input 
            image using ants.registration. Default is 'SyN' (Symmetric Normalization).
        kwargs (keyword arguments): Additional arguments passed to ants.registration().
    
    Returns:
        ants.ANTsImage: Input image warped to atlas space.
    """
    input_img = ants.image_read(input_image_path)
    anat_img = ants.image_read(anat_image_path)
    atlas_img = ants.image_read(atlas_image_path)

    assert check_physical_space_for_ants_image_pair(input_img, anat_img), (
        "input image and anatomical image must occupy the same physical space")

    anat_atlas_xfm = ants.registration(fixed=atlas_img,
                                       moving=anat_img,
                                       type_of_transform=type_of_transform,
                                       write_composite_transform=True,
                                       **kwargs)

    if input_img.dimension == 4:
        dim = 3
    else:
        dim = 0

    warped_img = ants.apply_transforms(fixed=atlas_img,
                                       moving=input_img,
                                       transformlist=anat_atlas_xfm['fwdtransforms'],
                                       verbose=True,
                                       imagetype=dim)

    return warped_img


def apply_xfm_ants(input_image_path: str,
                   ref_image_path: str,
                   out_image_path: str,
                   xfm_paths: list[str],
                   copy_meta: bool = False,
                   **kwargs) -> ants.ANTsImage:
    """
    Applies existing transforms in ANTs or ITK format to an input image, onto
    a reference image. This is useful for applying the same transform on
    different images to atlas space, for example.

    Args:
        input_image_path (str): Path to image on which transform is applied.
        ref_image_path (str): Path to image to which transform is applied.
        out_image_path (str): Path to which the transformed image is saved.
        xfm_paths (list[str]): List of transforms to apply to image. Must be in
            ANTs or ITK format, and can be affine matrix or warp coefficients.
        copy_meta (bool): If True, copies metadata file read from input_image_path as the metadata
            for new image out_image_path.

    Returns:
        xfm_img (ants.ANTsImage): The input image transformed with an ANTs transform file.
    """
    pet_image_ants = ants.image_read(input_image_path)
    ref_image_ants = ants.image_read(ref_image_path)

    if pet_image_ants.dimension == 4:
        dim = 3
    else:
        dim = 0

    xfm_img = ants.apply_transforms(fixed=ref_image_ants,
                                    moving=pet_image_ants,
                                    transformlist=xfm_paths,
                                    imagetype=dim,
                                    **kwargs)

    ants.image_write(xfm_img, out_image_path)

    if copy_meta:
        image_io.safe_copy_meta(input_image_path=input_image_path,out_image_path=out_image_path)

    return xfm_img


def apply_xfm_fsl(input_image_path: str,
                  ref_image_path: str,
                  out_image_path: str,
                  warp_path: str = None,
                  premat_path: str = '',
                  postmat_path: str = '',
                  **kwargs):
    """
    Applies existing transforms in FSL format to an input image, onto a
    reference image. This is useful for applying the same transform on
    different images to atlas space, for example.

    .. important::
        Requires installation of ``FSL``, and environment variables ``FSLDIR`` and
        ``FSLOUTPUTTYPE`` set appropriately in the shell.

    Args:
        input_image_path (str): Path to image on which transform is applied.
        ref_image_path (str): Path to image to which transform is applied.
        out_image_path (str): Path to which the transformed image is saved.
        warp_path (str): Path to FSL warp file.
        premat_path (str): Path to FSL ``premat`` matrix file.
        postmat_path (str): Path to FSL ``postmat`` matrix file.
        kwargs (keyword arguments): Additional arguments passed to
            :py:func:`fsl.wrappers.applywarp`.
    """
    if premat_path == '' and postmat_path == '':
        fsl.wrappers.applywarp(src=input_image_path,
                               ref=ref_image_path,
                               out=out_image_path,
                               warp=warp_path,
                               **kwargs)
    elif premat_path == '' and postmat_path != '':
        fsl.wrappers.applywarp(src=input_image_path,
                               ref=ref_image_path,
                               out=out_image_path,
                               warp=warp_path,
                               postmat=postmat_path,
                               **kwargs)
    elif premat_path != '' and postmat_path == '':
        fsl.wrappers.applywarp(src=input_image_path,
                               ref=ref_image_path,
                               out=out_image_path,
                               warp=warp_path,
                               premat=premat_path,
                               **kwargs)
    else:
        fsl.wrappers.applywarp(src=input_image_path,
                               ref=ref_image_path,
                               out=out_image_path,
                               warp=warp_path,
                               premat=premat_path,
                               postmat=postmat_path,
                               **kwargs)

    image_io.safe_copy_meta(input_image_path=input_image_path, out_image_path=out_image_path)


def resample_nii_4dfp(input_image_path: str,
                      resampled_image_path: str,
                      mpr_image_path: str,
                      out_image_path: str):
    """
    Resample and rearrange a 3D PET image to mpr space. This function mimics the functionality
    of converting an image from Nifti to 4dfp and back into Nifti, without using 4dfp tools.

    The typical use case would be for multi-modal studies where getting PET data aligned using
    exactly the same transformation procedures as BOLD data is critical. If, say, transformations
    have been computed from MPRAGE to atlas space using 4dfp tools, but the MPRAGE has been
    modified by converting from Nifti to 4dfp and back into Nifti, then the same conversion must be
    applied to the PET data aligned to anatomical space. This function accurately replicates that
    conversion.

    Args:
        input_image_path (str): Path to PET image on which transform is applied.
        resampled_image (str): Path to image with sampling needed for output. Often `rawavg.mgz` in
            FreeSurfer directory.
        mpr_image_path (str): Path to mpr (MPRAGE) image the PET will be transformed to.
    """
    input_image = nibabel.load(input_image_path)
    resampled_image = nibabel.load(resampled_image_path)
    mpr_image = nibabel.load(mpr_image_path)
    input_image_resampled = resample_from_to(
        from_img=input_image,
        to_vox_map=(resampled_image.shape[:3], resampled_image.affine)
    )
    image_resampled_array = input_image_resampled.get_fdata()
    resampled_swapped = np.swapaxes(np.swapaxes(image_resampled_array, 0, 2), 1, 2)
    resampled_swapped_flipped = np.flip(np.flip(np.flip(resampled_swapped, 1), 2), 0)
    input_on_mpr = nibabel.nifti1.Nifti1Image(
        dataobj=resampled_swapped_flipped,
        affine=mpr_image.affine,
        header=mpr_image.header
    )
    nibabel.save(input_on_mpr, out_image_path)
    image_io.safe_copy_meta(input_image_path=input_image_path, out_image_path=out_image_path)
