"""
Module to handle timing information of PET scans.
"""
import numpy as np

@dataclass
class ScanTimingInfo:
    """
    A data structure to represent and streamline access to timing information for image scans.

    This class encapsulates details about a scan's timing, including:
    - Start and end times of each scan frame.
    - Duration and center times of the frames.
    - Decay values (if applicable).

    Additionally, the class provides properties for easy conversion of timing values to minutes
    if the times are given in seconds and exceed a threshold (assumed to be 200.0 seconds).

    Attributes:
        duration (np.ndarray[float]): Array of frame durations.
        end (np.ndarray[float]): Array of frame end times.
        start (np.ndarray[float]): Array of frame start times.
        center (np.ndarray[float]): Array of frame center times (midpoints).
        decay (np.ndarray[float]): Array of decay coefficients for the scan frames.

    Properties:
        duration_in_mins (np.ndarray[float]):
            Returns the frame durations converted to minutes if `end` is >= 200.0 seconds.
            Otherwise, returns the original durations.

        end_in_mins (np.ndarray[float]):
            Returns the frame end times converted to minutes if `end` is >= 200.0 seconds.
            Otherwise, returns the original end times.

        start_in_mins (np.ndarray[float]):
            Returns the frame start times converted to minutes if `end` is >= 200.0 seconds.
            Otherwise, returns the original start times.

        center_in_mins (np.ndarray[float]):
            Returns the frame center times converted to minutes if `end` is >= 200.0 seconds.
            Otherwise, returns the original center times.

    Examples:

        .. code-block:: python

            import numpy as np
            from petpal.utils.scan_timing import ScanTimingInfo, get_frame_timing_info_for_nifti

            # Explicitly setting the attributes
            ## Define scan timing information
            duration = np.array([60.0, 120.0, 180.0])  # seconds
            start = np.array([0.0, 60.0, 180.0])
            end = np.array([60.0, 180.0, 360.0])
            center = (start + end) / 2.0  # Calculate the midpoints
            decay = np.array([1.0, 0.9, 0.8])  # Example decay values

            ## Create an instance of ScanTimingInfo
            scan_timing_info = ScanTimingInfo(duration=duration, end=end, start=start, center=center, decay=decay)

            ## Access original timing information
            print(scan_timing_info.duration)  # [ 60. 120. 180.]
            print(scan_timing_info.center)    # [30.  120. 270.]

            ## Access timing as minutes (when times exceed 200.0 seconds)
            print(scan_timing_info.duration_in_mins)  # [ 60. 120. 180.] (Unchanged)
            print(scan_timing_info.center_in_mins)    # [30. 120. 270.] (Unchanged)

            ## Example when `end` is greater than 200.0:
            scan_timing_info.end = np.array([300.0, 400.0, 500.0])  # Update end times
            print(scan_timing_info.end_in_mins)  # [5. 6.66666667 8.33333333] (Converted to minutes)
            print(scan_timing_info.start_in_mins)  # [0. 1. 3.] (Converted to minutes)

            # Getting the object directly from a nifty image file (assuming the metadata shares the name)
            scan_timing_info = get_frame_timing_info_for_nifti("/path/to/image.nii.gz")

    """
    duration: np.ndarray[float]
    end: np.ndarray[float]
    start: np.ndarray[float]
    center: np.ndarray[float]
    decay: np.ndarray[float]

    @property
    def duration_in_mins(self) -> np.ndarray[float]:
        if self.end[-1] >= 200.0:
            return self.duration / 60.0
        else:
            return self.duration

    @property
    def end_in_mins(self) -> np.ndarray[float]:
        if self.end[-1] >= 200.0:
            return self.end / 60.0
        else:
            return self.end

    @property
    def start_in_mins(self) -> np.ndarray[float]:
        if self.end[-1] >= 200.0:
            return self.start / 60.0
        else:
            return self.start

    @property
    def center_in_mins(self) -> np.ndarray[float]:
        if self.end[-1] >= 200.0:
            return self.center / 60.0
        else:
            return self.center

def get_frame_timing_info_for_nifti(image_path: str) -> ScanTimingInfo:
    r"""
    Extracts frame timing information and decay factors from a NIfTI image metadata.
    Expects that the JSON metadata file has ``FrameDuration`` and ``DecayFactor`` or
    ``DecayCorrectionFactor`` keys.

    .. important::
        This function tries to infer `FrameTimesEnd` and `FrameTimesStart` from the frame durations
        if those keys are not present in the metadata file. If the scan is broken, this might generate
        incorrect results.


    Args:
        image_path (str): Path to the NIfTI image file.

    Returns:
        :class:`ScanTimingInfo`: Frame timing information with the following elements:
            - duration (np.ndarray): Frame durations in seconds.
            - start (np.ndarray): Frame start times in seconds.
            - end (np.ndarray): Frame end times in seconds.
            - center (np.ndarray): Frame center times in seconds.
            - decay (np.ndarray): Decay factors for each frame.
    """
    _meta_data = load_metadata_for_nifti_with_same_filename(image_path=image_path)
    frm_dur = np.asarray(_meta_data['FrameDuration'], float)
    try:
        frm_ends = np.asarray(_meta_data['FrameTimesEnd'], float)
    except KeyError:
        frm_ends = np.cumsum(frm_dur)
    try:
        frm_starts = np.asarray(_meta_data['FrameTimesStart'], float)
    except KeyError:
        frm_starts = np.diff(frm_ends)
    try:
        decay = np.asarray(_meta_data['DecayCorrectionFactor'], float)
    except KeyError:
        decay = np.asarray(_meta_data['DecayFactor'], float)
    try:
        frm_centers = np.asarray(_meta_data['FrameReferenceTime'], float)
    except KeyError:
        frm_centers = np.asarray(frm_starts + frm_dur / 2.0, float)

    return ScanTimingInfo(duration=frm_dur, start=frm_starts, end=frm_ends, center=frm_centers, decay=decay)
