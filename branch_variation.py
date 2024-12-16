import math
from dataclasses import dataclass

@dataclass
class BranchVariation:
    """Stores variation parameters for individual branches"""
    length_variation: float = 0.2  # ±20% variation in length
    angle_variation: float = math.pi/6  # ±30° variation in angle
    radius_variation: float = 0.15  # ±15% variation in radius
    
    def copy(self):
        """Create a deep copy of the BranchVariation"""
        return BranchVariation(
            length_variation=self.length_variation,
            angle_variation=self.angle_variation,
            radius_variation=self.radius_variation
        )
