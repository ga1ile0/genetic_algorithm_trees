from tree import Tree
import bpy
import random
import math

class TreePopulation:
    def __init__(self, size: int = 10):
        print("Initializing TreePopulation with size:", size)
        self.size = size
        self.trees = [Tree() for _ in range(size)]  # Initialize as self.trees
        self.population = self.trees  # Also set as self.population for compatibility
        self.generation = 0
        self.historical_trees = []

        # Calculate fitness for initial population
        for tree in self.population:
            tree.fitness = tree.calculate_fitness()
            print(f"Initial tree fitness: {tree.fitness}")

        # Store initial population in historical trees
        for tree in self.population:
            tree.creation_generation = self.generation
            self.historical_trees.append(tree.copy())
        print("TreePopulation initialized with", len(self.population), "trees")

    def evolve(self, num_generations: int):
        """Evolve the population for a specified number of generations"""
        print("Starting evolution with", len(self.population), "trees")
    
        for _ in range(num_generations-1):
            self.generation += 1  # Increment generation at the start
            print(f"Generation {self.generation}: Processing {len(self.population)} trees")
            
            # Calculate fitness for all trees
            for tree in self.population:
                if tree.fitness is None:
                    tree.fitness = tree.calculate_fitness()
                    
            # Sort population by fitness
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Mark the best tree
            best_tree = self.population[0]
            best_tree.mark_as_best(self.generation)

            # Ensure only the top tree is marked as best
            for tree in self.population[1:]:
                if self.generation in tree.best_fitness_generations:
                    tree.best_fitness_generations.remove(self.generation)
            
            # Create new population
            new_population = []
            
            # Keep the best tree (elitism)
            best_tree_copy = best_tree.copy()
            print(f"Copied best tree with fitness: {best_tree_copy.fitness}")
            best_tree_copy.creation_generation = self.generation  # Use current generation
            new_population.append(best_tree_copy)
            self.historical_trees.append(best_tree_copy)  # Add to historical trees
            print(f"Appended best tree to historical_trees. Total historical_trees: {len(self.historical_trees)}")

            
            # Fill rest of population with crossover
            while len(new_population) < self.size:
                # Select parents using tournament selection
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                
                # Mark parents as used for crossover
                parent1.mark_for_crossover(self.generation)
                parent2.mark_for_crossover(self.generation)
                
                # Create children through crossover
                child1, child2 = Tree.crossover(parent1, parent2)  # Call as static method
                child1.creation_generation = self.generation
                child2.creation_generation = self.generation
                
                # Possibly mutate children
                if random.random() < 0.2:  # 20% mutation chance
                    child1 = child1.mutate()
                    child1.creation_generation = self.generation  # Set correct generation
                if random.random() < 0.2:
                    child2 = child2.mutate()
                    child2.creation_generation = self.generation  # Set correct generation
                        
                # Collect children in a list
                children = [child1, child2]
                # Calculate remaining space
                space_left = self.size - len(new_population)
                            # Calculate fitness for children before copying
                for child in children[:space_left]:
                    if child.fitness is None:
                        child.fitness = child.calculate_fitness()
                        print(f"Calculated fitness for child created in generation {child.creation_generation}: {child.fitness}")

                    # Store copies in historical record
                    copied_child = child.copy()
                    self.historical_trees.append(copied_child)
                    print(f"Copied child to historical_trees with fitness: {copied_child.fitness}")
                # Add children without exceeding population size
                new_population.extend(children[:space_left])
                        
            self.population = new_population
            print(f"Generation {self.generation} complete. New population size: {len(self.population)}")
        print(f"Evolution complete. Total generations: {self.generation}")

    def _tournament_selection(self):
        """Select a parent using tournament selection"""
        # Tournament size should be smaller than population size
        tournament_size = min(3, len(self.population))
        tournament = random.sample(self.population, tournament_size)
        
        # Return the tree with the best fitness from the tournament
        return max(tournament, key=lambda tree: tree.fitness)

    def animate_evolution(self):
        """Create an animation showing the evolution process of trees"""
        import bpy
        
        # Clear existing trees before starting the animation
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.select_by_type(type='MESH')
        bpy.ops.object.delete()
        
        # Constants for animation
        TREE_SPACING = 25  # Space between trees
        FRAMES_PER_GENERATION = 80  # Frames per generation
        FADE_DURATION = 15          # Frames for fade animation
        INITIAL_DELAY = 5           # Initial empty frames
        EMPTY_SCENE_DURATION = 15   # Empty frames between generations

        # Clear any existing animation data
        for collection in bpy.data.collections:
            if collection.name.startswith("Generation_"):
                bpy.data.collections.remove(collection)

        # Corrected loop with enumerate
        for i, tree in enumerate(self.historical_trees):
            print(f'\n\n***Tree nr: {i} *****')
            print(f'Created in: {tree.creation_generation} generation')
            print(f'Was best in: {tree.best_fitness_generations}')
            print(f'Fitness: {tree.fitness}\n')
                
        # Get trees grouped by generation
        trees_by_generation = {}
        for tree in self.historical_trees:
            if tree.creation_generation not in trees_by_generation:
                trees_by_generation[tree.creation_generation] = []
            trees_by_generation[tree.creation_generation].append(tree)
            
        # Adjust generation numbering
        max_generation = self.generation
        bpy.context.scene.frame_end = (
            (max_generation + 1) * (FRAMES_PER_GENERATION + EMPTY_SCENE_DURATION)
            + FADE_DURATION
        )
        bpy.context.scene.frame_start = 0
        
        def set_tree_visibility(obj, frame, visible):
            """Helper function to set keyframes for tree visibility"""
            obj.hide_viewport = not visible
            obj.hide_render = not visible
            obj.keyframe_insert(data_path="hide_viewport", frame=frame)
            obj.keyframe_insert(data_path="hide_render", frame=frame)
            
            # Also animate material transparency
            if len(obj.material_slots) > 0:
                material = obj.material_slots[0].material
                if not material.use_nodes:
                    return
                
                # Find the principled BSDF node
                principled_bsdf = next((n for n in material.node_tree.nodes 
                                      if n.type == 'BSDF_PRINCIPLED'), None)
                if principled_bsdf:
                    alpha = 1.0 if visible else 0.0
                    principled_bsdf.inputs['Alpha'].default_value = alpha
                    principled_bsdf.inputs['Alpha'].keyframe_insert(
                        data_path="default_value", frame=frame)
        
        # Initialize last_gen_best_tree_objects
        last_gen_best_tree_objects = []

        # Process each generation
        for generation in range(0, max_generation + 1):
            print(f'\n\n#######Animating generation {generation} ...')
            start_frame = (
                generation * (FRAMES_PER_GENERATION + EMPTY_SCENE_DURATION)
            )
            end_frame = start_frame + FRAMES_PER_GENERATION
            trees = trees_by_generation.get(generation, [])
            
            # Update last_gen_best_tree_objects for final generation
            is_final_generation = (generation == max_generation)
            
            # Create trees for this generation
            for i, tree in enumerate(trees):
                position = (i * TREE_SPACING, 0, 0)  # Trees in a line along X axis
                tree.create_in_blender(position=position)
                
                # Find the tree's collection
                generation_collection = bpy.data.collections[f"Generation_{generation}"]
                # Look through all children to find the tree collection
                tree_objects = []
                for collection in generation_collection.children:
                    if collection.name.startswith("Tree_"):
                        tree_objects.extend(collection.objects)
                
                if not tree_objects:
                    print(f"Warning: No objects found for tree {i} in generation {generation}")
                    continue
                
                # Make tree appear
                for obj in tree_objects:
                    # Set initial visibility to False just before appearance
                    set_tree_visibility(obj, start_frame - 1, False)
                    # Make tree visible at start_frame
                    set_tree_visibility(obj, start_frame, True)
                    
                    # Handle best trees
                    if (generation in tree.best_fitness_generations):
                        if is_final_generation:
                            # Add to list to keep visible until the end
                            last_gen_best_tree_objects.append(obj)
                            set_tree_visibility(obj, end_frame, True)
                            set_tree_visibility(obj, bpy.context.scene.frame_end, True)
                        # Removed the else block to prevent best trees in non-final generations from reappearing
                    else:
                        # If tree wasn't selected for crossover or wasn't the best, make it fade out
                        if (generation not in tree.crossover_generations and 
                            generation not in tree.best_fitness_generations):
                            # Fade out non-selected trees midway
                            fade_start = start_frame + FRAMES_PER_GENERATION // 2
                            set_tree_visibility(obj, fade_start, True)
                            set_tree_visibility(obj, fade_start + FADE_DURATION, False)
                            # Ensure visibility remains off after fade-out
                            set_tree_visibility(obj, fade_start + FADE_DURATION + 1, False)
                    
                    print(f"Set animation for object {obj.name} in generation {generation}")
            print(f"Animated trees for generation {generation} \n\n")
            
        # Ensure the last generation's best tree stays visible
        if last_gen_best_tree_objects:
            for obj in last_gen_best_tree_objects:
                set_tree_visibility(obj, bpy.context.scene.frame_end, True)
        
        # Set the scene to the start frame
        bpy.context.scene.frame_set(0)
        
        # Set up animation settings
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.samples = 128
        bpy.context.scene.frame_set(0)
        bpy.context.scene.render.fps = 30