import bpy
import sys
import os
import importlib

num_generations = 5
population_size = 5

# Get the absolute path to the current directory
blend_dir = os.path.dirname(bpy.data.filepath)
if blend_dir not in sys.path:
    sys.path.append(blend_dir)

# Print for debugging
print(f"Python path: {sys.path}")
print(f"Current directory: {blend_dir}")
print(f"Files in directory: {os.listdir(blend_dir)}")

try:
    # Import and reload modules to ensure we have the latest version
    import tree_population
    import tree
    import tree_genotype
    import branch_variation
    
    importlib.reload(tree_population)
    importlib.reload(tree)
    importlib.reload(tree_genotype)
    importlib.reload(branch_variation)
    
    from tree_population import TreePopulation
    from tree import Tree
    from tree_genotype import TreeGenotype
    from branch_variation import BranchVariation
    print("Successfully imported all modules")
except ImportError as e:
    print(f"Import error: {e}")
    raise

def create_tree():
    print("Creating tree population...")
    # Clear existing objects to prevent trees from reappearing
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    population = TreePopulation(size=population_size)
    
    # Evolve for several generations
    population.evolve(num_generations)
    
    # Create animation of the evolution process
    population.animate_evolution()
    print("Evolution animation created. You can now play or render the animation.")

if __name__ == "__main__":
    create_tree()
    #sowef55sssdddsssssssssssssssdabcsssssssasdsdasdssssss