import random
import math
import bpy
import mathutils
from tree_genotype import TreeGenotype
from branch_variation import BranchVariation

tree_counter = 0

class Tree:
    def __init__(self, genotype: TreeGenotype = None):
        if genotype is None:
            self.genotype = self.generate_random_genotype()
        else:
            self.genotype = genotype
        self.branch_segments = []  # Store branch segments for leaf placement
        self.fitness = None
        # New tracking attributes
        self.creation_generation = 0  # Generation when tree was created
        self.best_fitness_generations = []  # Generations where this tree was the best
        self.crossover_generations = []  # Generations where this tree was used for crossover
    
    @staticmethod
    def generate_random_genotype() -> TreeGenotype:
        """Generate a random genotype with more variation while maintaining good size"""
        # Randomize branching strategy
        branching_style = random.random()  # Random value between 0 and 1
        
        if branching_style < 0.3:  # 30% chance of many small branches
            num_branches = random.randint(2, 3)  # Reduced by ~30% (was 3-5)
            levels = random.randint(3, 4)        # Reduced by ~20% (was 4-6)
        elif branching_style < 0.6:  # 30% chance of fewer, longer branches
            num_branches = random.randint(2, 3)  # Reduced by ~30% (was 2-4)
            levels = random.randint(3, 4)        # Reduced by ~20% (was 4-6)
        else:  # 40% chance of medium configuration
            num_branches = random.randint(2, 4)  # Reduced by ~30% (was 3-6)
            levels = random.randint(3, 4)        # Reduced by ~20% (was 4-6)

        # Increased height range for bigger trees
        height = random.uniform(6.0, 12.0)
        
        # Scale trunk radius with height for better proportions
        min_radius = max(0.15, height * 0.025)  # Minimum radius scales with height
        max_radius = max(0.3, height * 0.05)
        
        # Adjust branch length ratio for more natural look with increased size
        length_ratio = random.uniform(0.6, 0.85)
        
        # Randomize branch angle more
        branch_angle = random.uniform(math.pi/8, math.pi/2.5)
        
        return TreeGenotype(
            height=height,
            trunk_radius=random.uniform(min_radius, max_radius),
            branch_angle=branch_angle,
            branch_length_ratio=length_ratio,
            num_branches_per_level=num_branches,
            branching_levels=levels,
            variation=BranchVariation(
                length_variation=random.uniform(0.1, 0.25),
                angle_variation=random.uniform(math.pi/12, math.pi/4),
                radius_variation=random.uniform(0.1, 0.2)
            )
        )
    
    def mutate(self, mutation_rate: float = 0.1) -> 'Tree':
        """Create a mutated copy of the tree"""
        new_genotype = TreeGenotype(
            height=self._mutate_value(self.genotype.height, 0.5, 2.0, 10.0, mutation_rate),
            trunk_radius=self._mutate_value(self.genotype.trunk_radius, 0.05, 0.1, 0.5, mutation_rate),
            branch_angle=self._mutate_value(self.genotype.branch_angle, 0.1, math.pi/6, math.pi/2, mutation_rate),
            branch_length_ratio=self._mutate_value(self.genotype.branch_length_ratio, 0.1, 0.3, 0.9, mutation_rate),
            num_branches_per_level=int(self._mutate_value(self.genotype.num_branches_per_level, 1, 2, 5, mutation_rate)),
            branching_levels=int(self._mutate_value(self.genotype.branching_levels, 1, 2, 5, mutation_rate)),
            variation=self.genotype.variation
        )
        return Tree(new_genotype)
    
    @staticmethod
    def _mutate_value(value: float, mutation_std: float, min_val: float, max_val: float, mutation_rate: float) -> float:
        """Helper method to mutate a single value"""
        if random.random() < mutation_rate:
            value += random.gauss(0, mutation_std)
            return max(min_val, min(max_val, value))
        return value

    @staticmethod
    def crossover(parent1: 'Tree', parent2: 'Tree') -> tuple['Tree', 'Tree']:
        """Create two offspring by crossing over two parent trees"""
        # Simple averaging crossover
        child1_genotype = TreeGenotype(
            height=(parent1.genotype.height + parent2.genotype.height) / 2,
            trunk_radius=(parent1.genotype.trunk_radius + parent2.genotype.trunk_radius) / 2,
            branch_angle=(parent1.genotype.branch_angle + parent2.genotype.branch_angle) / 2,
            branch_length_ratio=(parent1.genotype.branch_length_ratio + parent2.genotype.branch_length_ratio) / 2,
            num_branches_per_level=int((parent1.genotype.num_branches_per_level + parent2.genotype.num_branches_per_level) / 2),
            branching_levels=random.choice([parent1.genotype.branching_levels, 
                                          parent2.genotype.branching_levels]),
            variation=BranchVariation(
                length_variation=(parent1.genotype.variation.length_variation + parent2.genotype.variation.length_variation) / 2,
                angle_variation=(parent1.genotype.variation.angle_variation + parent2.genotype.variation.angle_variation) / 2,
                radius_variation=(parent1.genotype.variation.radius_variation + parent2.genotype.variation.radius_variation) / 2
            )
        )
        
        # Create slight variation for second child
        child2_genotype = TreeGenotype(
            height=child1_genotype.height * random.uniform(0.9, 1.1),
            trunk_radius=child1_genotype.trunk_radius * random.uniform(0.9, 1.1),
            branch_angle=child1_genotype.branch_angle * random.uniform(0.9, 1.1),
            branch_length_ratio=child1_genotype.branch_length_ratio * random.uniform(0.9, 1.1),
            num_branches_per_level=child1_genotype.num_branches_per_level,
            branching_levels=child1_genotype.branching_levels,
            variation=BranchVariation(
                length_variation=child1_genotype.variation.length_variation * random.uniform(0.9, 1.1),
                angle_variation=child1_genotype.variation.angle_variation * random.uniform(0.9, 1.1),
                radius_variation=child1_genotype.variation.radius_variation * random.uniform(0.9, 1.1)
            )
        )
        
        return Tree(child1_genotype), Tree(child2_genotype)

    @staticmethod
    def _gaussian_fitness(value: float, optimal: float, std_dev: float) -> float:
        """Calculate fitness using a Gaussian function centered at the optimal value"""
        return math.exp(-((value - optimal) ** 2) / (2 * std_dev ** 2))

    @staticmethod
    def _linear_reward(value: float, min_value: float, max_value: float) -> float:
        """Calculate fitness that linearly increases with value, capped at max_value"""
        if value <= min_value:
            return 0.0
        if value >= max_value:
            return 1.0
        return (value - min_value) / (max_value - min_value)

    def calculate_fitness(self) -> float:
        """Calculate fitness with more tolerance for variation"""
        components = {}  # Store individual fitness components
        weights = {}     # Store weights for normalization
        
        # 1. Size fitness - prefer larger trees
        components['height'] = self._linear_reward(
            self.genotype.height,
            min_value=5.0,
            max_value=15.0
        )
        weights['height'] = 0.3
        
        # 2. Branch distribution fitness - more branches is better
        branch_count = self.genotype.num_branches_per_level * (2 ** (self.genotype.branching_levels - 1))
        # Heavily penalize trees with very few branches
        if branch_count < 10:  # Reduced minimum (was 15)
            components['branch_dist'] = 0.1
        else:
            components['branch_dist'] = self._linear_reward(
                branch_count,
                min_value=20,  # Reduced by ~30% (was 15)
                max_value=55   # Reduced by ~30% (was 80)
            )
        weights['branch_dist'] = 2.0  # Keeping high priority
        
        # 3. Branching levels - heavily penalize few levels
        if self.genotype.branching_levels < 3:  # Reduced minimum (was 4)
            components['branch_levels'] = 0.1
        else:
            components['branch_levels'] = self._linear_reward(
                self.genotype.branching_levels,
                min_value=4,  # Reduced by ~20% (was 4)
                max_value=6   # Reduced by ~20% (was 7)
            )
        weights['branch_levels'] = 1.5  # Very high priority
        
        # 4. Branch angle fitness
        components['angle'] = self._gaussian_fitness(
            self.genotype.branch_angle,
            optimal=math.pi/3,
            std_dev=math.pi/3
        )
        weights['angle'] = 0.2

        # 5. Branch length ratio - prefer significant reduction
        components['length_ratio'] = self._gaussian_fitness(
            self.genotype.branch_length_ratio,
            optimal=0.7,
            std_dev=0.15
        )
        weights['length_ratio'] = 0.3

        # 6. Length variation - more variation is better
        components['length_var'] = self._linear_reward(
            self.genotype.variation.length_variation,
            min_value=0.1,
            max_value=0.3
        )
        weights['length_var'] = 0.3

        # 7. Angle variation - more variation is better
        components['angle_var'] = self._linear_reward(
            self.genotype.variation.angle_variation,
            min_value=math.pi/12,
            max_value=math.pi/4
        )
        weights['angle_var'] = 0.3

        # 8. Radius variation - more variation is better
        components['radius_var'] = self._linear_reward(
            self.genotype.variation.radius_variation,
            min_value=0.2,
            max_value=0.4
        )
        weights['radius_var'] = 0.4
        
        # 9. Symmetry fitness
        components['symmetry'] = 1.0 if self.genotype.num_branches_per_level % 2 == 0 else 0.5
        weights['symmetry'] = 0.1

        # Calculate weighted sum
        weighted_sum = sum(components[k] * weights[k] for k in components)
        
        # Normalize by total weight
        total_weight = sum(weights.values())
        normalized_fitness = weighted_sum / total_weight
        
        return normalized_fitness

    def create_in_blender(self, position=(0, 0, 0)):
        """Create the tree geometry in Blender"""
        global tree_counter
        print(f"\n---Creating tree number {tree_counter}...")
        tree_counter += 1
        
        # Clear previous trees in the collection
        generation_name = f"Generation_{self.creation_generation}"
        if generation_name in bpy.data.collections:
            generation_collection = bpy.data.collections[generation_name]
            for obj in generation_collection.objects:
                bpy.data.objects.remove(obj, do_unlink=True)
        else:
            generation_collection = bpy.data.collections.new(generation_name)
            bpy.context.scene.collection.children.link(generation_collection)

        # Ensure we're in object mode
        if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        # Create a collection for this generation if it doesn't exist
        generation_name = f"Generation_{self.creation_generation}"
        if generation_name not in bpy.data.collections:
            generation_collection = bpy.data.collections.new(generation_name)
            bpy.context.scene.collection.children.link(generation_collection)
        else:
            generation_collection = bpy.data.collections[generation_name]
        
        # Create a unique collection for this tree
        tree_collection_name = f"Tree_{len(generation_collection.children)}"
        tree_collection = bpy.data.collections.new(tree_collection_name)
        generation_collection.children.link(tree_collection)

        def create_branch(start_pos, direction, length, radius, level=0):
            # print(f"Creating branch at level {level}")
            # print(f"Position: {start_pos}, Length: {length}, Radius: {radius}")
            
            if level >= self.genotype.branching_levels:
                return

            # Switch to object mode if needed
            if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')

            # Apply random variations to this branch
            varied_length = length * random.uniform(
                1 - self.genotype.variation.length_variation,
                1 + self.genotype.variation.length_variation
            )
            
            varied_radius = radius * random.uniform(
                1 - self.genotype.variation.radius_variation,
                1 + self.genotype.variation.radius_variation
            )

            # Create branch cylinder with varied dimensions
            try:
                bpy.ops.mesh.primitive_cylinder_add(
                    vertices=8,
                    radius=varied_radius,
                    depth=varied_length,
                    enter_editmode=False,
                    align='WORLD',
                    location=start_pos
                )
                #print(f"Created cylinder for branch level {level}")
            except Exception as e:
                print(f"Error creating cylinder: {e}")
                return
            
            # Get the created cylinder
            branch = bpy.context.active_object
            if not branch:
                print("Failed to get active object after cylinder creation")
                return
                
            # Move the branch to our tree collection
            for col in branch.users_collection:
                col.objects.unlink(branch)
            tree_collection.objects.link(branch)
                
            branch.name = f"{tree_collection_name}_Branch_L{level}"
            #print(f"Named branch: {branch.name}")
            
            # Calculate end position using varied length
            end_pos = mathutils.Vector(start_pos) + mathutils.Vector(direction) * varied_length
            
            # Store branch info if it's level 2 or higher
            if level >= 2:
                # Store multiple points along the branch for leaf placement
                num_segments = int(length * 4)  # Increased number of segments for denser leaf placement
                for i in range(num_segments + 1):
                    t = i / num_segments
                    pos = mathutils.Vector(start_pos) + mathutils.Vector(direction) * (length * t)
                    self.branch_segments.append((mathutils.Vector(pos), mathutils.Vector(direction), level, radius))
            
            # Point cylinder in right direction
            direction_vec = mathutils.Vector(direction)
            rot_quat = direction_vec.to_track_quat('-Z', 'Y')
            branch.rotation_euler = rot_quat.to_euler()
            
            # Move cylinder so its base is at start_pos
            branch.location = (
                start_pos[0] + direction[0] * varied_length / 2,
                start_pos[1] + direction[1] * varied_length / 2,
                start_pos[2] + direction[2] * varied_length / 2
            )
            
            # Assign material to branch
            branch_material = self.create_tree_material()
            if branch_material:
                if len(branch.data.materials) == 0:
                    branch.data.materials.append(branch_material)
                else:
                    branch.data.materials[0] = branch_material
            
            # Create child branches with variation
            if level < self.genotype.branching_levels:
                # Randomize number of branches at this level
                actual_branches = max(1, self.genotype.num_branches_per_level + random.randint(-1, 1))
                
                for i in range(actual_branches):
                    # Calculate new direction for child branch with variation
                    base_angle = (2 * math.pi * i) / actual_branches
                    varied_branch_angle = self.genotype.branch_angle + random.uniform(
                        -self.genotype.variation.angle_variation,
                        self.genotype.variation.angle_variation
                    )
                    
                    # Add some randomness to the distribution angle
                    distribution_variation = random.uniform(-math.pi/6, math.pi/6)
                    angle = base_angle + distribution_variation
                    
                    new_direction = (
                        direction[0] * math.cos(varied_branch_angle) + math.cos(angle) * math.sin(varied_branch_angle),
                        direction[1] * math.cos(varied_branch_angle) + math.sin(angle) * math.sin(varied_branch_angle),
                        direction[2] * math.cos(varied_branch_angle)
                    )
                    
                    # Create child branch
                    new_length = length * self.genotype.branch_length_ratio
                    new_radius = radius * self.genotype.branch_length_ratio
                    create_branch(end_pos, new_direction, new_length, new_radius, level + 1)
        
        # Create the trunk (first branch)
        trunk_direction = (0, 0, 1)  # Straight up
        create_branch(position, trunk_direction, self.genotype.height, self.genotype.trunk_radius)
        
        # Join all tree parts into a single object
        bpy.ops.object.select_all(action='DESELECT')
        tree_parts = [obj for obj in tree_collection.objects]
        #print(f"Found {len(tree_parts)} tree parts to join")
        
        if tree_parts:
            for obj in tree_parts:
                obj.select_set(True)
            bpy.context.view_layer.objects.active = tree_parts[0]
            bpy.ops.object.join()
            tree_obj = bpy.context.active_object
            tree_obj.name = f"{tree_collection_name}_Tree"
            #print("Tree parts joined successfully")
            
            # Add leaves to the tree
            self.add_leaves(tree_obj)
            
            # Now join leaves with the tree
            bpy.ops.object.select_all(action='DESELECT')
            tree_obj.select_set(True)
            leaves = next((obj for obj in tree_collection.objects if obj.name.endswith("_Leaves")), None)
            if leaves:
                leaves.select_set(True)
                bpy.context.view_layer.objects.active = tree_obj
                bpy.ops.object.join()
                print("Joined leaves with tree successfully")
            
            return tree_obj

    def create_tree_material(self):
        """Create a material for the tree branches"""
        mat_name = "TreeBarkMaterial"
        
        # Check if material already exists
        if (mat_name in bpy.data.materials):
            return bpy.data.materials[mat_name]
            
        material = bpy.data.materials.new(name=mat_name)
        material.use_nodes = True
        nodes = material.node_tree.nodes
        
        # Clear default nodes
        nodes.clear()
        
        # Create nodes
        principled = nodes.new('ShaderNodeBsdfPrincipled')
        output = nodes.new('ShaderNodeOutputMaterial')
        
        # Set material properties
        principled.inputs['Base Color'].default_value = (0.3, 0.2, 0.1, 1)  # Brown color
        principled.inputs['Roughness'].default_value = 0.7
        
        # Connect nodes
        material.node_tree.links.new(principled.outputs['BSDF'], output.inputs['Surface'])
        
        return material

    def create_leaf_material(self):
        """Create a material for leaves"""
        mat_name = "TreeLeafMaterial"
        
        # Check if material already exists
        if mat_name in bpy.data.materials:
            return bpy.data.materials[mat_name]
            
        material = bpy.data.materials.new(name=mat_name)
        material.use_nodes = True
        nodes = material.node_tree.nodes
        
        # Clear default nodes
        nodes.clear()
        
        # Create nodes
        principled = nodes.new('ShaderNodeBsdfPrincipled')
        output = nodes.new('ShaderNodeOutputMaterial')
        
        # Set material properties
        principled.inputs['Base Color'].default_value = (0.1, 0.4, 0.1, 1)  # Green color
        principled.inputs['Roughness'].default_value = 0.5
        
        # Connect nodes
        material.node_tree.links.new(principled.outputs['BSDF'], output.inputs['Surface'])
        
        return material

    def create_leaf_at_position(self, position, direction, size=0.3):
        """Create a single leaf mesh at the given position"""
        # Create vertices for a simple leaf shape (diamond-like)
        verts = [
            (-size/2, 0, -size/4),     # Left point
            (0, 0, size/2),            # Top point
            (size/2, 0, -size/4),      # Right point
            (0, 0, -size/2)            # Bottom point
        ]
        
        # Create faces
        faces = [(0, 1, 2, 3)]
        
        # Create the mesh
        leaf_mesh = bpy.data.meshes.new('leaf_mesh')
        leaf_mesh.from_pydata(verts, [], faces)
        leaf_mesh.update()
        
        # Create the object
        leaf_obj = bpy.data.objects.new("Leaf", leaf_mesh)
        
        # Link it to the scene
        bpy.context.scene.collection.objects.link(leaf_obj)
        
        # Set location
        leaf_obj.location = position
        
        # Calculate rotation to face the direction
        direction_vec = mathutils.Vector(direction)
        rot_quat = direction_vec.to_track_quat('-Z', 'Y')
        leaf_obj.rotation_euler = rot_quat.to_euler()
        
        # Add random rotation around local Z axis
        leaf_obj.rotation_euler.z += random.uniform(0, 2 * math.pi)
        
        # Add slight random tilt
        leaf_obj.rotation_euler.x += random.uniform(-0.2, 0.2)
        leaf_obj.rotation_euler.y += random.uniform(-0.2, 0.2)
        
        # Create and assign material
        leaf_material = self.create_leaf_material()
        if len(leaf_obj.data.materials) == 0:
            leaf_obj.data.materials.append(leaf_material)
        else:
            leaf_obj.data.materials[0] = leaf_material
            
        return leaf_obj

    def add_leaves(self, tree_obj):
        """Add leaves to the tree by creating actual meshes"""
        print("Adding leaves to the tree...")
        
        # Get the tree's collection
        tree_collection = tree_obj.users_collection[0]
        
        # Clean up any existing leaves in this tree's collection
        for obj in tree_collection.objects:
            if obj.name.endswith("_Leaves"):
                bpy.data.objects.remove(obj, do_unlink=True)
        
        print(f"Found {len(self.branch_segments)} branch segments for leaves")
        
        # Create leaf clusters along branches
        leaf_objects = []
        print(f"Creating leaf clusters...")
        for i, (pos, direction, level, radius) in enumerate(self.branch_segments):
            # Higher probability of leaves at higher levels, but generally higher chance
            if random.random() > (0.4 + (level * 0.1)):  # Increased base probability
                continue
                

            # Create more leaves per point
            num_leaves = random.randint(2, 3)  # Increased number of leaves
            
            # Calculate base positions around the branch
            for j in range(num_leaves):
                # Calculate angle around branch
                angle = (j / num_leaves) * 2 * math.pi + random.uniform(-0.5, 0.5)
                
                # Calculate offset from branch surface
                right = mathutils.Vector((-direction.y, direction.x, 0)).normalized()
                if right.length < 0.1:  # If branch is vertical, use a default right vector
                    right = mathutils.Vector((1, 0, 0))
                up = direction.cross(right)
                
                # Position leaf at branch surface
                offset = (right * math.cos(angle) + up * math.sin(angle)) * radius * 1.0  # Scaled by radius
                leaf_pos = mathutils.Vector(pos) + offset
                
                # Calculate leaf direction: pointing slightly outward from branch
                outward_dir = offset.normalized()  # Direction from branch center
                up_dir = mathutils.Vector((0, 0, 1))
                base_dir = (outward_dir * 0.6 + up_dir * 0.4).normalized()
                
                # Add slight randomization to direction
                leaf_dir = base_dir + mathutils.Vector((
                    random.uniform(-0.2, 0.2),
                    random.uniform(-0.2, 0.2),
                    random.uniform(-0.1, 0.3)
                ))
                leaf_dir.normalize()
                
                # Vary leaf size based on level but keep them a bit larger
                leaf_size = 0.5 - (level * 0.03)  # Larger base size, smaller reduction per level
                leaf = self.create_leaf_at_position(leaf_pos, leaf_dir, size=leaf_size)
                
                # Move leaf to tree's collection
                for col in leaf.users_collection:
                    col.objects.unlink(leaf)
                tree_collection.objects.link(leaf)
                
                leaf_objects.append(leaf)
        
        print(f"Created total of {len(leaf_objects)} individual leaves")
        
        # Join all leaves into a single object for better performance
        if leaf_objects:
            print("Joining leaf objects...")
            bpy.ops.object.select_all(action='DESELECT')
            for leaf in leaf_objects:
                leaf.select_set(True)
            bpy.context.view_layer.objects.active = leaf_objects[0]
            bpy.ops.object.join()
            
            # Name the combined leaf object using tree's collection name
            leaves_obj = bpy.context.active_object
            leaves_obj.name = f"{tree_collection.name}_Leaves"
            
            # Ensure leaves are in the tree's collection
            for col in leaves_obj.users_collection:
                col.objects.unlink(leaves_obj)
            tree_collection.objects.link(leaves_obj)
            
            print(f"Successfully created {leaves_obj.name} object")
        else:
            print("No leaves were created!")

    def mark_as_best(self, generation: int):
        """Mark this tree as the best in a given generation"""
        # Ensure generation is recorded correctly
        if generation not in self.best_fitness_generations:
            self.best_fitness_generations.append(generation)
            
    def mark_for_crossover(self, generation: int):
        """Mark this tree as used for crossover in a given generation"""
        # Ensure generation is recorded correctly
        if generation not in self.crossover_generations:
            self.crossover_generations.append(generation)
            
    def copy(self) -> 'Tree':
        """Create a deep copy of the tree"""
        new_tree = Tree(self.genotype.copy())
        new_tree.fitness = self.fitness
        new_tree.creation_generation = self.creation_generation
        new_tree.best_fitness_generations = self.best_fitness_generations.copy()
        new_tree.crossover_generations = self.crossover_generations.copy()
        return new_tree