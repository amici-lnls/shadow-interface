from importlib.resources import files

import Shadow

from shadow_interface.beamline import BeamLine
from shadow_interface.optical_elements import Screen, Slits, ToroidalMirror
from shadow_interface.sources import BendingMagnet
from shadow_interface.utils import ReferenceFrame, rotation_matrix

# Beamline specification files ship as package data, so they are resolved
# relative to this package regardless of whether shadow_interface is installed
# (site-packages) or run from a source checkout (editable install).
SPECS_DIR = str(files("shadow_interface.beamline_config"))


# Utilize the cax-scripts repository for additional functionality
# sys.path.append(os.path.expanduser("~/repos/cax-scripts"))


class CAXSim(BeamLine):
    """
    Class to simulate the CARCARÁ-X (CAX) beamline using shadow_interface. 
    
    This class inherits from the Beamline class and provides methods to set 
    up and run simulations for the CAX beamline, perform scans and save 
    results.
    """

    CAX_SPECS = f"{SPECS_DIR}/CAX"

    sirius_frame = ReferenceFrame(name="sirius")

    R1 = rotation_matrix('x', 90)
    R2 = rotation_matrix('y', 180)
    shadow_R = R1 @ R2

    shadow_frame = sirius_frame.child_frame(relative_origin=[0, 0, 0], 
                                            relative_rotation=shadow_R, 
                                            name="shadow")

    def __init__(self, total_rays: int = 100000, image_size: tuple = (210, 280)):
        # Initialize the elements



        self.total_rays = total_rays        
        source = BendingMagnet("B1", f"{self.CAX_SPECS}/source B1.txt")

        self.mask_A1 = Slits(name="MASK_A1", 
                             specification_file=f"{self.CAX_SPECS}/mask_A1.txt")
        self.mirror = ToroidalMirror(name="M1", 
                                     specification_file=f"{self.CAX_SPECS}/mirror M1.txt")
        self.dvf_A1 = Slits(name="DVF_A1", 
                            specification_file=f"{self.CAX_SPECS}/slit_A1.txt")
        self.dvf_B1 = Screen(name="DVF_B1", 
                            specification_file=f"{self.CAX_SPECS}/dvf_B1.txt")
        # self.dvf_B1.pixel_size = 0.48/1000 # .48 μm in mm
        beam = Shadow.Beam()
        

        super().__init__(name="CAX", beam=beam, source=source, 
                 optical_elements=[self.mask_A1,
                                   self.mirror, 
                                   self.dvf_A1, 
                                   self.dvf_B1], 
                 beamline_frame=self.shadow_frame, 
                 lab_frame=self.sirius_frame,
                 image_size=image_size,
                 total_rays=self.total_rays)
        
        self.trace()