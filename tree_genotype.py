from dataclasses import dataclass, field
from branch_variation import BranchVariation

@dataclass
class TreeGenotype:
    # Basic tree parameters
    height: float          # Overall height of the tree
    trunk_radius: float    # Base radius of the trunk
    branch_angle: float    # Base angle between branches and trunk/parent branch
    branch_length_ratio: float  # Base ratio of branch length to parent branch
    num_branches_per_level: int # Number of branches at each level
    branching_levels: int  # How many levels of branching (recursion depth)
    variation: BranchVariation = field(default_factory=BranchVariation)  # Variation parameters
    
    def copy(self):
        """Create a deep copy of the TreeGenotype"""
        return TreeGenotype(
            height=self.height,
            trunk_radius=self.trunk_radius,
            branch_angle=self.branch_angle,
            branch_length_ratio=self.branch_length_ratio,
            num_branches_per_level=self.num_branches_per_level,
            branching_levels=self.branching_levels,
            variation=self.variation.copy()  # Make sure to copy the variation object too
        )