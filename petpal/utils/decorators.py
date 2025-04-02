"""
A collection of decorators to handle boilerplate code. Most decorators will
extend the functionality of functions that only work with objects or arrays.
The extensions allow for the flexibility of passing in image
paths or the image objects themselves, and allow the users to pass an
optional path for saving the output of the decorated function.
"""

import functools
import ants

from .image_io import load_metadata_for_nifti_with_same_filename, write_dict_to_json, gen_meta_data_filepath_for_nifti

def ANTsImageToANTsImage(func):
    """
    A decorator for functions that process an ANTs image and output another ANTs image.
    Assumes that the argument of the passed in function is an ANTs image.

    This decorator is designed to extend functions that take an ANTs image as input
    and output another ANTs image. It supports seamless handling of input images
    provided as either file paths (str) or `ants.core.ANTsImage` objects. The resulting
    processed image can optionally be saved to a specified file path.

    Args:
        func (Callable): The function to be decorated. It should accept an ANTs image as
            the first argument and return a processed ANTs image.

    Returns:
        Callable: A wrapper function that:
            - Reads the input image if a file path (str) is provided.
            - Passes an `ants.core.ANTsImage` object to the decorated function.
            - Saves the output image to the specified file path if `out_path` is provided.

    Example:

        .. code-block:: python

            import ants
            from petpal.utils.decorators import ANTsImageToANTsImage
            from petpal.preproc.segmentaion_tools import calc_vesselness_measure_image

            # Defining the decorated function
            @ANTsImageToANTsImage
            def step_calc_vesselness_measure_image(input_image: ants.core.ANTsImage,
                                                   sigma_min: float = 2.0,
                                                   sigma_max: float = 8.0,
                                                   alpha: float = 0.5,
                                                   beta: float = 0.5,
                                                   gamma: float = 5.0,
                                                   morph_open_radius: int = 1,
                                                   **hessian_func_kwargs):
                return calc_vesselness_measure_image(input_image=input_image,
                                                     sigma_min=sigma_min,
                                                     sigma_max=sigma_max,
                                                     alpha=alpha,
                                                     beta=beta,
                                                     gamma=gamma,
                                                     morph_open_radius=morph_open_radius,
                                                     **hessian_func_kwargs)

            # Conventional use of calc_vesselness_measure_image

            input_img = ants.image_read('/path/to/3d/img/.nii.gz')
            vess_img = calc_vesselness_measure_image(input_img) # Using all default values
            ants.image_write(vess_img, '/path/to/out/img/.nii.gz')


            # Using the decorated version
            ## Using paths as inputs
            vess_img = step_calc_vesselness_measure_image('/path/to/3d/img/.nii.gz',
                                                          '/path/to/out/img/.nii.gz')

            ### Not saving output image
            vess_img = step_calc_vesselness_measure_image('/path/to/3d/img/.nii.gz',
                                                          None)

            ## Using images as inputs
            input_img = ants.image_read('/path/to/3d/img/.nii.gz')
            vess_img = step_calc_vesselness_measure_image(input_img,
                                                          '/path/to/out/img/.nii.gz')

            ## Ignoring the return value to just save image
            step_calc_vesselness_measure_image('/path/to/3d/img/.nii.gz',
                                               '/path/to/out/img/.nii.gz')


    Raises:
        TypeError: If `in_img` is not a string or `ants.core.ANTsImage`.

    Notes:
        - If `in_img` is provided as a file path, the image is read using `ants.image_read`.
        - The output image is written to the desired path using `ants.image_write` if
          `out_path` is specified.
    """

    @functools.wraps(func)
    def wrapper(in_img:ants.core.ANTsImage | str,
                out_path: str,
                *args, **kwargs):
        if isinstance(in_img, str):
            in_image = ants.image_read(in_img)
        elif isinstance(in_img, ants.core.ANTsImage):
            in_image = in_img
        else:
            raise TypeError('in_img must be str or ants.core.ANTsImage')
        out_img = func(in_image, *args, **kwargs)
        if out_path is not None:
            ants.image_write(out_img, out_path)
        return out_img
    return wrapper

def ANTsImageToANTsImageWithMetadata(func):
    """
    A decorator for functions that process a tuple (ANTs image, dict) and output another tuple of same type.
    Assumes that the argument of the passed in function is a tuple of type (ANTs image, dict).

    This decorator is a variation of `ANTsImageToANTsImage` to be used for functions operating on and returning tuples
    of type (ANTsImage, dict). It supports seamless handling of input images
    provided as either file paths (str) or tuple (ANTsImage, dict) objects. The resulting
    processed image and metadata can optionally be saved to a specified file path.

    Args:
        func (Callable): The function to be decorated. It should accept a tuple (ANTsImage, dict) as
            the first argument and return a processed tuple (ANTsImage, dict).

    Returns:
        Callable: A wrapper function that:
            - Reads the input image and sidecar metadata if a file path (str) is provided.
            - Passes a tuple (`ants.core.ANTsImage`, dict) object to the decorated function.
            - Saves the output image and metadata to the specified file path if `out_path` is provided.

    Raises:
        TypeError: If `in_img` is not a string or tuple (`ants.core.ANTsImage`, dict).

    Notes:
        - If `in_img` is provided as a file path, the image is read using `ants.image_read`.
        - The output image is written to the desired path using `ants.image_write` if
          `out_path` is specified.
    """

    @functools.wraps(func)
    def wrapper(in_img:(ants.core.ANTsImage, dict) | str,
                out_path: str,
                *args, **kwargs):
        if isinstance(in_img, str):
            in_tuple = (ants.image_read(in_img), load_metadata_for_nifti_with_same_filename(in_img))
        elif isinstance(in_img, tuple) and list(map(type, in_img)) == [ants.core.ANTsImage, dict]:
            in_tuple = in_img
        else:
            raise TypeError('in_img must be str or tuple of type (ants.core.ANTsImage, dict)')
        out_tuple = func(in_tuple, *args, **kwargs)
        if out_path is not None:
            ants.image_write(out_tuple[0], out_path)
            write_dict_to_json(out_tuple[1], gen_meta_data_filepath_for_nifti(out_path))
        return out_tuple
    return wrapper

def ANTsImagesToANTsImageWithMetadata(func):
    """
    A decorator for functions that process a list of tuples of type (ANTs image, dict) and output a single tuple.
    Assumes that the argument of the passed in function is a list of tuples of type (ANTs image, dict). The output tuple
    should also be of type (ANTsImage, dict).

    This decorator is a variation of `ANTsImageToANTsImageWithMetadata` to be used for functions operating on and
    returning lists of tuples of type (ANTsImage, dict). It supports seamless handling of input images
    provided as either file paths (list[str]) or list[(ANTsImage, dict)] objects. The resulting
    processed image and metadata can optionally be saved to a specified file path.

    Args:
        func (Callable): The function to be decorated. It should accept a list[(ANTsImage, dict)] as
            the first argument and return a processed tuple (ANTsImage, dict).

    Returns:
        Callable: A wrapper function that:
            - Reads the input images and sidecar metadata if filepaths (list[str]) are provided.
            - Passes a list[(`ants.core.ANTsImage`, dict)] object to the decorated function.
            - Saves the output image and metadata to the specified file path if `out_path` is provided.

    Raises:
        TypeError: If `in_images` is not a list[string] or list[(`ants.core.ANTsImage`, dict)].

    Notes:
        - If `in_images` is provided as list of filepaths, the images are read using `ants.image_read`.
        - The output image is written to the desired path using `ants.image_write` if
          `out_path` is specified.
    """

    @functools.wraps(func)
    def wrapper(in_images: list[(ants.core.ANTsImage, dict)] | list[str],
                out_path: str,
                *args, **kwargs):
        if isinstance(in_images, list) and all(isinstance(img, str) for img in in_images):
            in_tuples = [(ants.image_read(obj[0]), load_metadata_for_nifti_with_same_filename(obj[1]))
                        for obj in in_images]
        # This is horrendous; don't merge with this in.
        elif isinstance(in_images, list) and all(list(map(type, in_img)) == [ants.core.ANTsImage, dict] for in_img in in_images):
            in_tuples = in_images
        ####################################################################
        else:
            raise TypeError('in_img must be list[str] or list[tuple of type (ants.core.ANTsImage, dict)]')
        out_tuple = func(in_tuples, *args, **kwargs)
        if out_path is not None:
            ants.image_write(out_tuple[0], out_path)
            write_dict_to_json(out_tuple[1], gen_meta_data_filepath_for_nifti(out_path))
        return out_tuple
    return wrapper