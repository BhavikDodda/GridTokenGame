import numpy as np
N=4
grid=[[0 for i in range(N)]for j in range(N)]


def print_grid(grid):
    ans="\n".join([" ".join([str(k) for k in i]) for i in grid])
    print(ans)

print_grid(grid)

from collections import deque

def dfs(grid, visited, i, j, component):
    rows, cols = len(grid), len(grid[0])
    if i < 0 or i >= rows or j < 0 or j >= cols or grid[i][j] == 0 or visited[i][j]:
        return
    visited[i][j] = True
    component.append((i, j))
    dfs(grid, visited, i + 1, j, component)
    dfs(grid, visited, i - 1, j, component)
    dfs(grid, visited, i, j + 1, component)
    dfs(grid, visited, i, j - 1, component)

def get_connected_components(grid):
    rows, cols = len(grid), len(grid[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    components = []
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 1 and not visited[i][j]:
                component = []
                dfs(grid, visited, i, j, component)
                components.append(component)
    return components


def rotate_component(component):
    """Rotate a component 90 degrees counterclockwise."""
    return [(-y, x) for x, y in component]

def flip_component(component, axis='horizontal'):
    """Flip a component horizontally or vertically."""
    if axis == 'horizontal':
        return [(x, -y) for x, y in component]
    elif axis == 'vertical':
        return [(-x, y) for x, y in component]
    else:
        raise ValueError("Invalid axis. Use 'horizontal' or 'vertical'.")

def generate_component_transformations(component):
    transformations = []
    current = component
    for _ in range(4):  # Rotate 0°, 90°, 180°, 270°
        transformations.append(current)
        transformations.append(flip_component(current, axis='horizontal'))  # Flip horizontally
        transformations.append(flip_component(current, axis='vertical'))  # Flip vertically
        current = rotate_component(current)  # Rotate 90°
    return transformations

def normalize_component(component):
    """Translate the component to the top-left corner of its bounding box."""
    min_x = min(x for x, y in component)
    min_y = min(y for x, y in component)
    return sorted([(x - min_x, y - min_y) for x, y in component])

def get_canonical_component(component):
    transformations = generate_component_transformations(component)
    normalized_transformations = [normalize_component(t) for t in transformations]
    return min(normalized_transformations)

def Canonical(grid):
    components=get_connected_components(grid)
    canonical=sorted([get_canonical_component(c) for c in components])
    return canonical

from itertools import combinations
from collections import defaultdict, deque
def dfs_pts(points, visited, point, component):
    """DFS to find connected points (4-connected)."""
    x, y = point
    if (x, y) not in points or (x, y) in visited:
        return
    visited.add((x, y))
    component.append((x, y))
    # Explore 4-connected neighbors (horizontal and vertical)
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        dfs_pts(points, visited, (x + dx, y + dy), component)
def get_connected_components_pts(points):
    """Group points into connected components using DFS."""
    visited = set()
    components = []
    for point in points:
        if point not in visited:
            component = []
            dfs_pts(points, visited, point, component)
            components.append(component)
    return components
#print("comps")
#print(get_connected_components_pts([(0, 0), (0, 1), (1, 0), (1, 2)]))

def CanonicalFromComps(comps):
    comps = [i for i in comps if len(i)>0]
    comps2 = []
    for comp in comps:
        comps2.extend(get_connected_components_pts(comp))
    canonical=sorted([get_canonical_component(c) for c in comps2])
    return canonical
#
def compare_configurations(grid1, grid2):    
    canonical1 = Canonical(grid1)
    canonical2 = Canonical(grid2)
    return canonical1 == canonical2
    

grid1 = np.array([
    [1, 1, 0, 0],
    [1, 1, 1, 0],
    [0, 0, 0, 0],
    [0, 0, 1, 1]
])

grid2 = np.array([
    [0, 0, 0, 0],
    [0, 1, 1, 1],
    [0, 1, 1, 0],
    [0, 0, 1, 0]
])

grid3 = np.array([
    [1, 1, 0, 0],
    [1, 1, 0, 1],
    [0, 1, 0, 1],
    [0, 0, 0, 0]
])

print(compare_configurations(grid1, grid2))
print(compare_configurations(grid1, grid3))
can=(Canonical(grid1))
print(can)
import numpy as np

def reconstruct_grid(canonical_components, grid_size=(7, 7)):
    """
    Reconstructs a grid from canonical components, ensuring components are disconnected.
    
    Args:
        canonical_components: List of components, where each component is a list of tuples.
        grid_size: Size of the grid to reconstruct (default is 7x7).
    
    Returns:
        grid: A numpy array representing the reconstructed grid.
        shifts: A list of (shift_x, shift_y) for each component.
    """
    grid = np.zeros(grid_size, dtype=int)
    shifts = []  # To store the shift applied to each component
    
    # Track occupied positions to avoid direct adjacency
    occupied = set()
    
    for component in canonical_components:
        if not component:  # Skip empty components
            shifts.append((0, 0))  # No shift for empty components
            continue
        
        # Find the bounding box of the component
        min_x = min(x for x, _ in component)
        min_y = min(y for _, y in component)
        max_x = max(x for x, _ in component)
        max_y = max(y for _, y in component)
        
        # Try to place the component in the grid
        placed = False
        for i in range(grid_size[0] - max_x):
            for j in range(grid_size[1] - max_y):
                # Check if the component can be placed at (i, j) without direct adjacency
                can_place = True
                for x, y in component:
                    if (i + x, j + y) in occupied:
                        can_place = False
                        break
                    # Check direct adjacency (up, down, left, right)
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        if (i + x + dx, j + y + dy) in occupied:
                            can_place = False
                            break
                    if not can_place:
                        break
                
                if can_place:
                    # Place the component
                    for x, y in component:
                        grid[i + x, j + y] = 1
                        occupied.add((i + x, j + y))
                    shifts.append((i, j))  # Store the shift applied to this component
                    placed = True
                    break
            if placed:
                break
        
        if not placed:
            raise ValueError("Cannot place component without violating disconnection constraint.")
    
    return grid, shifts

# Reconstruct the grid
reconstructed_grid,_ = reconstruct_grid(can, grid_size=(7,7))
print("Reconstructed Grid:")
print(reconstructed_grid)

NFset=[[]]
Fset=[]

def movesComb(w,h):
    seq=[]

    for Len_ in range(1,w+1):
        for j in range(h):
            for i in range(w-(Len_-1)):
                seq.append([(i+l,j) for l in range(Len_)])

    for Len_ in range(2,h+1):
        for j in range(h-(Len_-1)):
            for i in range(w):
                seq.append([(i,j+l) for l in range(Len_)])
    return seq


def remove_poss_from_component(canonical_components):
    afterMoves=[]
    for i, component in enumerate(canonical_components):
        component_set = set(component)
        min_x = min(x for x, y in component)
        min_y = min(y for x, y in component)
        max_x = max(x for x, y in component)
        max_y = max(y for x, y in component)
        
        possibilities = movesComb(max_x - min_x + 1, max_y - min_y + 1)
        
        
        for poss in possibilities:
            poss_set = set(poss)
            if poss_set.issubset(component_set):
                diff=component_set - poss_set
                ccano=canonical_components.copy()
                ccano[i] = list(diff)
                CCANO=CanonicalFromComps(ccano)
                afterMoves.append(CCANO)
                #print(reconstruct_grid(CCANO, grid_size=(7,7)))
    return afterMoves
F_strategies=[]
import sys
def F_or_NF(canonical_components):
    try:
        if(canonical_components in Fset):
            return "Fset"
        if(canonical_components in NFset):
            return "NFset"
        for i, component in enumerate(canonical_components):
            component_set = set(component)
            min_x = min(x for x, y in component)
            min_y = min(y for x, y in component)
            max_x = max(x for x, y in component)
            max_y = max(y for x, y in component)
            
            # Generate possibilities for the current component's size
            possibilities = movesComb(max_x - min_x + 1, max_y - min_y + 1)
            
            
            for poss in possibilities:
                poss_set = set(poss)
                if poss_set.issubset(component_set):
                    # Remove `poss_set` from the component
                    diff=component_set - poss_set
                    # Update the component in `canonical_components`
                    ccano=canonical_components.copy()
                    ccano[i] = list(diff)
                    CCANO=CanonicalFromComps(ccano)

                    if((CCANO not in Fset)):
                        if((CCANO not in NFset)):
                            F_or_NF(CCANO)
                    if(CCANO in NFset):
                        if(canonical_components not in Fset):
                            Fset.append(canonical_components)
                            F_strategies.append([i,poss_set])
                            return "Fset"
                    
                    
        if(canonical_components not in NFset):
            NFset.append(canonical_components)
            return "NFset"
    except KeyboardInterrupt:
        # Code to run when Ctrl+C is pressed
        print("\nKeyboard interrupt detected. Cleaning up...")
        # Add your cleanup code here
        visualize_configurations_to_pdf(NFset, grid_size=(7, 7), pdf_filename=os.path.join(script_dir, "NF_configurations.pdf"))
        visualize_configurations_with_strategies(Fset, F_strategies, grid_size=(7, 7), pdf_filename=os.path.join(script_dir, "F_configurations_with_strategies.pdf"), configs_per_row=4)
        print("Cleanup complete. Exiting program.")
        sys.exit(0)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def visualize_configurations_to_pdf(configurations, grid_size=(7, 7), pdf_filename="configurations.pdf", rows_per_page=6, configs_per_row=4):
    """
    Visualizes all configurations in a grid layout and saves them as a multi-page PDF.
    
    Args:
        configurations: List of configurations, where each configuration is a list of components.
        grid_size: Size of the grid (default is 7x7).
        pdf_filename: Name of the output PDF file.
        rows_per_page: Number of rows to display per page (default is 6).
        configs_per_row: Number of configurations to display per row (default is 4).
    """
    num_configs = len(configurations)
    configs_per_page = rows_per_page * configs_per_row  # Number of configurations per page
    
    # Create a PDF file
    with PdfPages(pdf_filename) as pdf:
        for page_num, start_idx in enumerate(range(0, num_configs, configs_per_page)):
            end_idx = min(start_idx + configs_per_page, num_configs)
            page_configs = configurations[start_idx:end_idx]
            
            # Calculate the number of rows for this page
            num_rows = (len(page_configs) + configs_per_row - 1) // configs_per_row
            
            # Create a figure for this page
            fig, axes = plt.subplots(num_rows, configs_per_row, figsize=(20, 5 * num_rows))
            axes = axes.flatten()  # Flatten the axes array for easy iteration
            
            for idx, config in enumerate(page_configs):
                ax = axes[idx]
                grid, _ = reconstruct_grid(config, grid_size=grid_size)
                
                # Plot the grid
                ax.imshow(grid, cmap='binary', vmin=0, vmax=1)
                ax.set_title(f"Config {start_idx + idx + 1}")
                ax.set_xticks(range(grid_size[1]))
                ax.set_yticks(range(grid_size[0]))
                ax.grid(which='both', color='black', linestyle='-', linewidth=1)
            
            # Hide unused subplots
            for j in range(idx + 1, len(axes)):
                axes[j].axis('off')
            
            plt.tight_layout()
            
            # Save the figure to the PDF
            pdf.savefig(fig)
            plt.close()

import os
script_dir = os.path.dirname(__file__)

#visualize_configurations_to_pdf(NFset, grid_size=(7, 7), pdf_filename=os.path.join(script_dir, "NF_configurations.pdf"))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def visualize_configurations_with_strategies(Fset, F_strategies, grid_size=(7, 7), pdf_filename="configurations_with_strategies.pdf", rows_per_page=6, configs_per_row=4):
    """
    Visualizes configurations with strategies and saves them as a multi-page PDF.
    
    Args:
        Fset: List of configurations, where each configuration is a list of components.
        F_strategies: List of strategies, where each strategy is [i, {positions to highlight}].
        grid_size: Size of the grid (default is 7x7).
        pdf_filename: Name of the output PDF file.
        rows_per_page: Number of rows to display per page (default is 6).
        configs_per_row: Number of configurations to display per row (default is 4).
    """
    num_configs = len(Fset)
    configs_per_page = rows_per_page * configs_per_row  # Number of configurations per page
    
    # Create a PDF file
    with PdfPages(pdf_filename) as pdf:
        for page_num, start_idx in enumerate(range(0, num_configs, configs_per_page)):
            end_idx = min(start_idx + configs_per_page, num_configs)
            page_configs = Fset[start_idx:end_idx]
            page_strategies = F_strategies[start_idx:end_idx]
            
            # Calculate the number of rows for this page
            num_rows = (len(page_configs) + configs_per_row - 1) // configs_per_row
            
            # Create a figure for this page
            fig, axes = plt.subplots(num_rows, configs_per_row, figsize=(20, 5 * num_rows))
            axes = axes.flatten()  # Flatten the axes array for easy iteration
            
            for idx, (config, strategy) in enumerate(zip(page_configs, page_strategies)):
                ax = axes[idx]
                grid, shifts = reconstruct_grid(config, grid_size=grid_size)
                
                # Plot the grid
                ax.imshow(grid, cmap='binary', vmin=0, vmax=1)
                ax.set_title(f"Config {start_idx + idx + 1}")
                ax.set_xticks(range(grid_size[1]))
                ax.set_yticks(range(grid_size[0]))
                ax.grid(which='both', color='black', linestyle='-', linewidth=1)
                
                # Extract the component index and positions to highlight
                component_index, positions_to_highlight = strategy
                shift_x, shift_y = shifts[component_index]  # Get the shift for this component
                
                # Highlight the strategy positions in green (mapped to actual grid positions)
                for x, y in positions_to_highlight:
                    actual_x = x + shift_x
                    actual_y = y + shift_y
                    if 0 <= actual_x < grid_size[0] and 0 <= actual_y < grid_size[1]:  # Ensure coordinates are within bounds
                        ax.add_patch(plt.Rectangle((actual_y - 0.5, actual_x - 0.5), 1, 1, fill=True, color='green', alpha=0.7))
            
            # Hide unused subplots
            for j in range(idx + 1, len(axes)):
                axes[j].axis('off')
            
            plt.tight_layout()
            
            # Save the figure to the PDF
            pdf.savefig(fig)
            plt.close()

#visualize_configurations_with_strategies(Fset, F_strategies, grid_size=(7, 7), pdf_filename=os.path.join(script_dir, "F_configurations_with_strategies.pdf"), configs_per_row=4)


#Backwards
GRID=[[1 for i in range(4)] for i in range(4)]

combss=Canonical(GRID)
can_combss=CanonicalFromComps(combss)
print(can_combss)
aorb=F_or_NF(can_combss)
print("Fset")
print(Fset)
print("F_strategies")
print(F_strategies)
print("NFset")
print(NFset)
print("F or NF:",aorb)
visualize_configurations_to_pdf(NFset, grid_size=(7, 7), pdf_filename=os.path.join(script_dir, "NF_configurations.pdf"))
visualize_configurations_with_strategies(Fset, F_strategies, grid_size=(7, 7), pdf_filename=os.path.join(script_dir, "F_configurations_with_strategies.pdf"), configs_per_row=4)