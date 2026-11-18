"""
Module containing the optical elements that can be used in shadow_interface. 

Each optical element is represented as a class that inherits from the base 
class `OpticalElement`. Each class has its own specific parameters and 
methods for calculating the effect of the optical element on the beam.
"""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import Shadow

from .utils import save_image

# Module-level globals used by _caustic_step_worker under fork context
_worker_screen = None
_worker_beam = None


def _caustic_step_worker(distance):
    beam = _worker_beam.duplicate()
    beam.retrace(distance)
    img = save_image(_worker_screen, beam)
    sigma_x = img.sigma_h if img.sigma_h else np.nan
    sigma_y = img.sigma_v if img.sigma_v else np.nan
    # print(f"Distance: {distance:.2f}, FWHM X: {fwhm_x:.4f}, FWHM Y: {fwhm_y:.4f}")
    del beam
    return sigma_x, sigma_y


class OpticalElement:
    """
    A wrapper class for a Shadow optical element, 
    providing methods to load specifications and interact with the element.
    """

    # Shadow OE field groups corresponding to the cached state below.
    OFFSET_ATTRIBUTES = ['OFFX', 'OFFY', 'OFFZ']
    TILT_ATTRIBUTES   = ['X_ROT', 'Y_ROT', 'Z_ROT']

    # X_ROT/Y_ROT/Z_ROT are deliberately NOT persistent raw Shadow fields.
    # Shadow.traceOE() overwrites them in place with radians (they are supplied
    # in degrees). "Preserving" those values across traces feeds radians back
    # to Shadow as degrees, shrinking every tilt by pi/180 on each trace.
    # The user-facing state is instead cached on the wrapper (_offset_lab /
    # _tilt_mrad) and re-applied in degrees by reset() before every trace.
    PERSISTENT_ATTRIBUTES = []

    def __init__(self, name: str, specification_file: str):
        self.name = name
        self.shadow_oe = Shadow.OE()
        self.specification_file = specification_file
        self.load_specification()
        # Original properties to reset after simulation
        self.specification = self.shadow_oe.to_dictionary()
        # To be set when the element is added to a beamline
        self.frame = None  
        self.beamline = None
        # Image
        self.img = None
        self.pixel_size = None
        # Attributes kept verbatim across reset(). Offsets/tilts are excluded:
        # they are re-derived from the cached user state (see reset()).
        self.persistent_attributes = self.PERSISTENT_ATTRIBUTES
        self.up_to_date = False

        # Canonical user-facing offset and tilt, stored in the LAB frame.
        # shadow_oe cannot be the source of truth for tilts because Shadow
        # mutates X_ROT/Y_ROT/Z_ROT (deg -> rad) inside every traceOE() call,
        # so reading them back would also depend on whether a trace happened.
        # Initialised from the specification once the frame is known; see
        # cache_state_from_spec(), called by BeamLine.initialize_elements().
        self._offset_lab = None  # np.ndarray(3), user units
        self._tilt_mrad = None   # np.ndarray(3), milliradians

    # ------------------------------------------------------------------
    # State caching / synchronisation with Shadow
    # ------------------------------------------------------------------

    def _require_frame(self):
        """Raise a helpful error if the element is not attached to a beamline."""
        if self.frame is None:
            raise ValueError(f"{self.name} frame is not defined. "
                             f"Please add the element to a beamline first.")

    def cache_state_from_spec(self):
        """Snapshot the specification's offset/tilt into the cached lab state.

        Must be called after ``self.frame`` has been assigned while ``shadow_oe``
        still holds the pristine specification values: OFFX/Y/Z in the local
        frame and X/Y/Z_ROT in DEGREES. BeamLine.initialize_elements()
        calls it at that point, before any trace can replace the rotations with
        radians.
        """
        self._require_frame()

        offset_local = np.array([
            self.shadow_oe.OFFX,
            self.shadow_oe.OFFY,
            self.shadow_oe.OFFZ,
        ], dtype=float)

        # Spec rotations are Shadow input angles in degrees using Shadow's
        # (right-hand) sign convention. The wrapper stores lab-frame mrad with
        # the opposite sign, hence the negation (mirrors the tilt getter).
        tilt_local_deg = np.array([
            self.shadow_oe.X_ROT,
            self.shadow_oe.Y_ROT,
            self.shadow_oe.Z_ROT,
        ], dtype=float)

        self._offset_lab = self.frame.vector_to_lab(offset_local)
        self._tilt_mrad = 1000.0 * self.frame.vector_to_lab(
            -np.deg2rad(tilt_local_deg)
        )

    def _ensure_state_cache(self):
        """Initialise the cache lazily if the frame was assigned manually."""
        self._require_frame()
        if self._offset_lab is None or self._tilt_mrad is None:
            self.cache_state_from_spec()

    def apply_state(self, shadow_oe=None):
        """Write the cached user state into a Shadow OE in Shadow's format.

        Shadow expects OFFX/Y/Z in the local frame and X/Y/Z_ROT in
        DEGREES. This is called by reset() immediately before traceOE(),
        so Shadow always receives freshly converted degree values rather than
        its own post-trace radian values. That is what keeps a tilt stable over
        any number of traces.
        """
        self._require_frame()
        self._ensure_state_cache()
        oe = self.shadow_oe if shadow_oe is None else shadow_oe

        offset_local = self.frame.vector_from_lab(self._offset_lab)
        oe.OFFX = offset_local[0]
        oe.OFFY = offset_local[1]
        oe.OFFZ = offset_local[2]

        tilt_lab_rad = np.asarray(self._tilt_mrad, dtype=float) / 1000.0
        tilt_local_deg = np.rad2deg(self.frame.vector_from_lab(tilt_lab_rad))
        oe.X_ROT = -tilt_local_deg[0]
        oe.Y_ROT = -tilt_local_deg[1]
        oe.Z_ROT = -tilt_local_deg[2]

    def _mark_modified(self):
        """Synchronise shadow_oe for inspection and invalidate cached results."""
        # Keep the raw OE consistent for anyone inspecting shadow_oe between
        # traces; reset() writes the same values again before traceOE().
        self.apply_state()
        self.up_to_date = False
        if self.beamline is not None:
            self.beamline.up_to_date = False

    # General attributes

    @property
    def image(self):
        """
        Image instance used to characterize the element's image
        """

        if self.beamline is None:
            raise Warning(f"Element {self.name} has not been added to a beamline."
                          f"Add this element to a BeamLine instance to trace it.")

        if not self.beamline.up_to_date: 
            # Retrace the beamline to update the beam with the new element 
            self.beamline.trace()

        return self.img


    @property
    def orientation(self):
        """
        Optical element orientation angle (degrees) relative to the previous 
        element. Positive values indicate counter-clockwise rotation.
        """
        return self.shadow_oe.ALPHA
    

    @property
    def source_distance(self):
        """
        Distance from the previous element to the source point (user units).
        """
        return self.shadow_oe.T_SOURCE
    
    @property 
    def image_distance(self):
        """
        Distance from the previous element to the image point (user units).
        """
        return self.shadow_oe.T_IMAGE 

    @property
    def units(self):
        """
        User units for distances.
        """

        return self.shadow_oe.unit()
    
    @property
    def offset(self):
        """
        Get the current offset of the element's position in the lab frame.

        Returns:
            np.ndarray: A 3D vector representing the offset in the lab frame.

        Notes:
            Read from the cached user state, never from shadow_oe: Shadow does
            not modify OFFX/Y/Z during tracing, but keeping one source of truth
            for offsets and tilts avoids unit/frame mistakes.
        """
        self._require_frame()
        self._ensure_state_cache()
        return self._offset_lab.copy()

    @offset.setter
    def offset(self, offset_vector: np.ndarray):
        """
        Apply an offset to the element's position.

        Parameters:
            offset_vector (np.ndarray): A 3D vector in the LAB frame.
        """
        self._require_frame()
        self._ensure_state_cache()

        offset_lab = np.asarray(offset_vector, dtype=float)

        # Check if the new values are the same as the old ones
        if np.isclose(offset_lab, self._offset_lab).all():
            return  # No change, so do nothing

        self._offset_lab = offset_lab
        self._mark_modified()

    @property
    def tilt(self):
        """
        Get the current tilt angles of the element's orientation in the global frame.

        Returns:
            np.ndarray: A 3D vector representing the tilt angles (in mrad)
            around the lab frame axes.

        Notes:
            Reading shadow_oe.X_ROT/Y_ROT/Z_ROT here used to be ambiguous:
            Shadow stores them in DEGREES until traceOE() runs and in RADIANS
            afterwards. The cached lab-frame state is always in mrad and is
            therefore valid before and after any number of traces.
        """
        self._require_frame()
        self._ensure_state_cache()
        return self._tilt_mrad.copy()

    @tilt.setter
    def tilt(self, tilt_mrad: np.ndarray):
        """
        Apply tilts to the element's orientation.

        Parameters:
            tilt_mrad (np.ndarray): A 3D vector representing the tilt angles
            (in mrad) around the lab frame axes.

        Notes:
            Only the cached lab-frame value is authoritative here. Conversion
            to Shadow's local-frame degrees happens in apply_state(), which is
            invoked by reset() right before each trace.
        """
        self._require_frame()
        self._ensure_state_cache()
        self._tilt_mrad = np.asarray(tilt_mrad, dtype=float)
        self._mark_modified()

    @property
    def tx(self):
        """
        Global X offset
        """
        return self.offset[0]
    
    @tx.setter
    def tx(self, value):
        offset = self.offset
        offset[0] = value
        self.offset = offset

    @property
    def ty(self):
        """
        Global Y offset
        """
        return self.offset[1]
    
    @ty.setter
    def ty(self, value):
        offset = self.offset
        offset[1] = value
        self.offset = offset

    @property
    def tz(self):
        """
        Global Z offset
        """
        return self.offset[2]
    
    @tz.setter
    def tz(self, value):
        offset = self.offset
        offset[2] = value
        self.offset = offset

    @property
    def rx(self):
        """
        Global X tilt angle (mrad)
        """
        return self.tilt[0]

    @rx.setter
    def rx(self, value_mrad):
        tilt = self.tilt.copy()
        tilt[0] = value_mrad
        self.tilt = tilt

    @property
    def ry(self):
        """
        Global Y tilt angle (mrad)
        """
        return self.tilt[1]

    @ry.setter
    def ry(self, value_mrad):
        tilt = self.tilt.copy()
        tilt[1] = value_mrad
        self.tilt = tilt

    @property
    def rz(self):
        """
        Global Z tilt angle (mrad)
        """
        return self.tilt[2]

    @rz.setter
    def rz(self, value_mrad):
        tilt = self.tilt.copy()
        tilt[2] = value_mrad
        self.tilt = tilt

    def load_specification(self, specification_file: str = None):
        """
        Load the optical element specification from a file.
        
        Parameters:
            specification_file (str): The path to the specification file. 
            If None, uses the default specification file.
        """

        if specification_file is None:
            specification_file = self.specification_file
        self.shadow_oe.load(specification_file)
        # self.specification = self.shadow_oe.to_dictionary()

    # def update(self):
    #     """
    #     Update the optical element after changing its parameters. This method 
    #     should be called after modifying any parameters to ensure the changes 
    #     are applied to the beamline.
    #     """
    #     # Reset the element to apply the new tilt angles
    #     self.reset()

    def reset(self):
        """
        Restore the specification, then re-apply the cached user state.

        Shadow's traceOE() overwrites X/Y/Z_ROT with radians. Without
        re-applying the cached tilts in DEGREES here, the next trace would feed
        those radians back to Shadow as degrees, shrinking every tilt by a
        factor of pi/180 per trace. BeamLine.trace() calls reset() immediately
        before traceOE(), so Shadow always receives the intended angles.
        """
        for key, value in self.specification.items():
            if key not in self.persistent_attributes:
                setattr(self.shadow_oe, key, value)

        # Re-apply LAST so the specification restore cannot overwrite it.
        self.apply_state()

# ========================================================================
#                        OPTICAL ELEMENTS
# ========================================================================

class ToroidalMirror(OpticalElement):
    """
    A wrapper class for a Shadow toroidal mirror optical element.
    """

    def __init__(self, name: str, specification_file: str):
        super().__init__(name, specification_file)
        
        if self.shadow_oe.FMIRR != 3:
            raise ValueError(f"File {specification_file} is not"
                             f" configured for a toroidal mirror: "
                             f"FMIRR = {self.shadow_oe.FMIRR}")


class Screen(OpticalElement):
    """
    A wrapper class for a Shadow screen optical element.
    """

    def __init__(self, name: str, specification_file: str, 
                 screen_dimensions: tuple = (5., 5.)):
        super().__init__(name, specification_file)
        
        if self.shadow_oe.F_SCREEN != 1:
            raise ValueError(f"File {specification_file} is not"
                             f" configured for a screen: "
                             f"F_SCREEN = {self.shadow_oe.F_SCREEN}")
    
    def _caustic_step(self, beam: Shadow.Beam, distance: float):

        beam_copy = beam.duplicate()

        beam_copy.retrace(distance)

        ana = save_image(self, beam_copy)

        fwhm_x = ana.hprm_fitting['fwhmx']
        fwhm_y = ana.hprm_fitting['fwhmy']

        del beam_copy  # Explicitly delete the duplicate beam to free memory

        return fwhm_x, fwhm_y
# print(f"Distance: {distance:.2f}, FWHM X: {fwhm_x:.4f}, FWHM Y: {fwhm_y:.4f}")
        

    def parallel_caustic(self, beam: Shadow.Beam = None,
                         s_range: tuple = (-100, 100),
                         n_points: int = 10,
                         max_workers: int = None):

        global _worker_screen, _worker_beam
        _worker_screen = self
        _worker_beam = beam

        ctx = multiprocessing.get_context('fork')
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            print("Calculating caustic in parallel...")
            distances = np.linspace(s_range[0], s_range[1], n_points)
            results = list(executor.map(_caustic_step_worker, distances))
        print("Caustic calculation completed.")
        return distances, np.array([r[0] for r in results]), np.array([r[1] for r in results])

    def simple_caustic(self, beam: Shadow.Beam = None, 
                       s_range: tuple = (-100, 100), 
                       n_points: int = 10):
        """
        Calculate a simple caustic of the beam by tracing it through the screen 
        at different distances.

        Parameters:
            beam: The input beam to be traced.
            s_range (tuple): A tuple specifying the range of distances to trace the beam.
            n_points (int): The number of points in the distance range to trace.

        Returns:
            list: A list of beams traced at different distances.
        """
        distances = np.linspace(s_range[0], s_range[1], n_points)
        
        fwhm_x, fwhm_y, intensity = [], [], []

        for distance in distances:
            # Set the screen's position based on the current distance

            beam.retrace(distance)

            ana = save_image(self, beam)

            if ana.beam_visible:
                fwhm_x.append(ana.hprm_fitting['fwhmx'])
                fwhm_y.append(ana.hprm_fitting['fwhmy'])
            else:
                fwhm_x.append(np.nan)
                fwhm_y.append(np.nan)

            # intensity.append(ana.hprm_fitting['peak'])


        
        return np.array(distances), np.array(fwhm_x), np.array(fwhm_y)#, intensity

class Slits(Screen):
    """
    A wrapper class for a Shadow slits optical element.
    """

    def __init__(self, name: str, specification_file: str, 
                 slit_positions: tuple = (0., 0.)):
        super().__init__(name, specification_file)
        
        if any(self.shadow_oe.I_SLIT) != 1:
            raise ValueError(f"File {specification_file} is not"
                             f" configured for slits: "
                             f"I_SLIT = {self.shadow_oe.I_SLIT}")
