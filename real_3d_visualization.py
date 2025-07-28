# real_3d_visualization.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import pandas as pd
from tabulate import tabulate

class Real3DParamagneticVisualization:
    """3D visualization of paramagnetic materials as real objects with color-coded magnetization."""
    
    def __init__(self):
        self.materials = None
        self.current_field = 0
        self.animation_running = False
        self.visualization_data = {}
        
    def create_cube_vertices(self, center=(0, 0, 0), size=1):
        """Create vertices for a 3D cube - generates geometric faces for material representation."""
        x, y, z = center
        s = size / 2
        
        vertices = np.array([
            [x-s, y-s, z-s], [x+s, y-s, z-s], [x+s, y+s, z-s], [x-s, y+s, z-s],  # bottom
            [x-s, y-s, z+s], [x+s, y-s, z+s], [x+s, y+s, z+s], [x-s, y+s, z+s]   # top
        ])
        
        # Define faces of the cube
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
            [vertices[4], vertices[7], vertices[6], vertices[5]],  # top
            [vertices[0], vertices[4], vertices[5], vertices[1]],  # front
            [vertices[2], vertices[6], vertices[7], vertices[3]],  # back
            [vertices[1], vertices[5], vertices[6], vertices[2]],  # right
            [vertices[4], vertices[0], vertices[3], vertices[7]]   # left
        ]
        
        return faces
    
    def create_cylinder_vertices(self, center=(0, 0, 0), radius=0.5, height=1, num_segments=20):
        """Create vertices for a 3D cylinder - cylindrical geometry for materials."""
        x, y, z = center
        
        # Create circular cross-sections
        theta = np.linspace(0, 2*np.pi, num_segments)
        
        # Bottom circle
        bottom_x = x + radius * np.cos(theta)
        bottom_y = y + radius * np.sin(theta)
        bottom_z = np.full_like(bottom_x, z - height/2)
        
        # Top circle  
        top_x = x + radius * np.cos(theta)
        top_y = y + radius * np.sin(theta)
        top_z = np.full_like(top_x, z + height/2)
        
        faces = []
        
        # Side faces (rectangles)
        for i in range(num_segments):
            next_i = (i + 1) % num_segments
            face = [
                [bottom_x[i], bottom_y[i], bottom_z[i]],
                [bottom_x[next_i], bottom_y[next_i], bottom_z[next_i]],
                [top_x[next_i], top_y[next_i], top_z[next_i]],
                [top_x[i], top_y[i], top_z[i]]
            ]
            faces.append(face)
        
        # Bottom and top caps
        bottom_face = [[bottom_x[i], bottom_y[i], bottom_z[i]] for i in range(num_segments)]
        top_face = [[top_x[i], top_y[i], top_z[i]] for i in range(num_segments)]
        faces.append(bottom_face)
        faces.append(top_face)
        
        return faces
    
    def create_sphere_vertices(self, center=(0, 0, 0), radius=0.5, num_segments=20):
        """Create vertices for a 3D sphere using triangular faces."""
        x, y, z = center
        
        # Create sphere using spherical coordinates
        phi = np.linspace(0, np.pi, num_segments)  # polar angle
        theta = np.linspace(0, 2*np.pi, num_segments)  # azimuthal angle
        
        faces = []
        
        for i in range(num_segments-1):
            for j in range(num_segments-1):
                # Four corners of current segment
                p1 = [x + radius * np.sin(phi[i]) * np.cos(theta[j]),
                      y + radius * np.sin(phi[i]) * np.sin(theta[j]),
                      z + radius * np.cos(phi[i])]
                
                p2 = [x + radius * np.sin(phi[i+1]) * np.cos(theta[j]),
                      y + radius * np.sin(phi[i+1]) * np.sin(theta[j]),
                      z + radius * np.cos(phi[i+1])]
                
                p3 = [x + radius * np.sin(phi[i+1]) * np.cos(theta[j+1]),
                      y + radius * np.sin(phi[i+1]) * np.sin(theta[j+1]),
                      z + radius * np.cos(phi[i+1])]
                
                p4 = [x + radius * np.sin(phi[i]) * np.cos(theta[j+1]),
                      y + radius * np.sin(phi[i]) * np.sin(theta[j+1]),
                      z + radius * np.cos(phi[i])]
                
                # Create two triangular faces
                faces.append([p1, p2, p3])
                faces.append([p1, p3, p4])
        
        return faces
    
    def get_magnetization_color(self, magnetization, max_magnetization):
        """Convert magnetization value to color using colormap for visual representation."""
        if max_magnetization == 0:
            normalized = 0
        else:
            normalized = abs(magnetization) / max_magnetization
        
        # Use a colormap: blue (low) -> red (high magnetization)
        colormap = cm.get_cmap('plasma')  # plasma: dark purple -> bright yellow
        return colormap(normalized)
    
    def create_magnetic_field_lines(self, field_strength, num_lines=8):
        """Create magnetic field lines for visualization of field distribution."""
        lines = []
        
        # Create field lines around the material
        for i in range(num_lines):
            angle = 2 * np.pi * i / num_lines
            # Field lines go from left to right
            x_line = np.linspace(-3, 3, 50)
            y_line = 0.5 * np.sin(angle) * np.ones_like(x_line)
            z_line = 0.5 * np.cos(angle) * np.ones_like(x_line)
            
            # Adjust field line density based on field strength
            intensity = field_strength / 1e6  # Normalize
            lines.append((x_line, y_line, z_line, intensity))
        
        return lines
    
    def print_magnetization_data_table(self, materials, field_strengths, shape):
        """Print detailed magnetization data in table format to console."""
        print(f"\n=== MAGNETIZATION DATA TABLE - {shape.upper()} VISUALIZATION ===")
        
        # Create data for table
        table_data = []
        headers = ["Field Strength (A/m)", "Material", "Susceptibility", "Magnetization (A/m)", "Relative Response"]
        
        for field in field_strengths:
            for material in materials:
                magnetization = material.susceptibility * field
                relative_response = magnetization / (materials[0].susceptibility * field) if materials[0].susceptibility * field != 0 else 0
                
                table_data.append([
                    f"{field:.1e}",
                    material.name,
                    f"{material.susceptibility:.2e}",
                    f"{magnetization:.2e}",
                    f"{relative_response:.2f}"
                ])
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Summary statistics
        print(f"\n=== SUMMARY STATISTICS ===")
        summary_data = []
        for material in materials:
            max_field = max(field_strengths)
            min_field = min(field_strengths)
            max_mag = material.susceptibility * max_field
            min_mag = material.susceptibility * min_field
            
            summary_data.append([
                material.name,
                f"{material.susceptibility:.2e}",
                f"{min_mag:.2e}",
                f"{max_mag:.2e}",
                f"{max_mag/min_mag:.1f}x" if min_mag != 0 else "inf"
            ])
        
        summary_headers = ["Material", "Susceptibility", "Min Magnetization", "Max Magnetization", "Dynamic Range"]
        print(tabulate(summary_data, headers=summary_headers, tablefmt="grid"))
        
        # Store data for later analysis
        self.visualization_data[f"{shape}_magnetization"] = {
            'field_strengths': field_strengths,
            'materials': [(mat.name, mat.susceptibility) for mat in materials],
            'magnetizations': [[material.susceptibility * field for field in field_strengths] for material in materials]
        }
    
    def plot_real_3d_materials(self, materials, field_strengths, shape='cube'):
        """Create real 3D visualization of paramagnetic materials with specified geometry."""
        
        # Print data table first
        self.print_magnetization_data_table(materials, field_strengths, shape)
        
        # Calculate magnetizations for color coding
        max_field = max(field_strengths)
        max_magnetizations = {}
        
        for material in materials:
            max_mag = material.susceptibility * max_field
            max_magnetizations[material.name] = max_mag
        
        # Create material objects
        material_objects = {}
        positions = [(-1.5, 0, 0), (1.5, 0, 0)]  # Side by side
        
        for i, material in enumerate(materials):
            if shape == 'cube':
                faces = self.create_cube_vertices(positions[i], size=1.2)
            elif shape == 'cylinder':
                faces = self.create_cylinder_vertices(positions[i], radius=0.6, height=1.2)
            elif shape == 'sphere':
                faces = self.create_sphere_vertices(positions[i], radius=0.7)
            
            material_objects[material.name] = {
                'faces': faces,
                'position': positions[i],
                'material': material
            }
        
        # Create multiple figures for different field strengths
        figures = []
        for field_strength in field_strengths:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            ax.set_title(f'Real 3D Paramagnetic Materials - {shape.title()}\nField Strength: {field_strength:.1e} A/m', 
                        fontsize=14, fontweight='bold')
            
            # Draw magnetic field lines
            field_lines = self.create_magnetic_field_lines(field_strength)
            for x_line, y_line, z_line, intensity in field_lines:
                alpha = 0.3 + 0.7 * intensity  # More visible at higher fields
                ax.plot(x_line, y_line, z_line, 'b-', alpha=alpha, linewidth=0.5)
            
            # Draw materials with color-coded magnetization
            for material_name, obj_data in material_objects.items():
                material = obj_data['material']
                magnetization = material.susceptibility * field_strength
                
                # Get color based on magnetization
                color = self.get_magnetization_color(magnetization, max_magnetizations[material_name])
                
                # Create 3D collection
                poly3d = Poly3DCollection(obj_data['faces'], alpha=0.8, 
                                        facecolors=color, edgecolors='black', linewidths=0.5)
                ax.add_collection3d(poly3d)
                
                # Add material labels
                pos = obj_data['position']
                ax.text(pos[0], pos[1], pos[2]-1, f'{material_name}\nM = {magnetization:.2e}', 
                       ha='center', va='top', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # Add field strength indicator
            ax.text(0, 0, 2, f'Field: {field_strength:.1e} A/m', 
                   ha='center', va='bottom', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
            
            # Add colorbar legend
            legend_text = "Color Code:\nLow Magnetization -> High Magnetization\nBlue -> Yellow -> Red"
            ax.text(3, 0, 0, legend_text, ha='left', va='center', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
            
            # Set equal aspect ratio and limits
            ax.set_xlim([-4, 4])
            ax.set_ylim([-2, 2]) 
            ax.set_zlim([-2, 2])
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            
            # Set viewing angle for better visualization
            ax.view_init(elev=20, azim=45)
            
            figures.append(fig)
        
        return figures
    
    def create_animated_3d_visualization(self, materials, shape='cube'):
        """Create animated 3D visualization showing field changes over time."""
        
        print(f"\n=== ANIMATED 3D VISUALIZATION DATA - {shape.upper()} ===")
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Field strength animation parameters
        field_max = 1e6
        time_points = np.linspace(0, 4*np.pi, 200)  # 2 full cycles
        field_strengths = field_max * (0.5 + 0.5 * np.sin(time_points))  # Oscillating field
        
        # Print animation parameters table
        anim_data = [
            ["Parameter", "Value"],
            ["Max Field Strength", f"{field_max:.1e} A/m"],
            ["Animation Duration", f"{len(time_points)} frames"],
            ["Field Oscillation", "Sinusoidal (0.5 to 1.0 * max field)"],
            ["Shape", shape.title()],
            ["Materials", ", ".join([mat.name for mat in materials])]
        ]
        print(tabulate(anim_data, headers="firstrow", tablefmt="grid"))
        
        # Setup material objects
        positions = [(-1.5, 0, 0), (1.5, 0, 0)]
        material_objects = {}
        
        for i, material in enumerate(materials):
            if shape == 'cube':
                faces = self.create_cube_vertices(positions[i], size=1.2)
            elif shape == 'cylinder':
                faces = self.create_cylinder_vertices(positions[i], radius=0.6, height=1.2)
            elif shape == 'sphere':
                faces = self.create_sphere_vertices(positions[i], radius=0.7)
            
            material_objects[material.name] = {
                'faces': faces,
                'position': positions[i],
                'material': material
            }
        
        max_magnetizations = {mat.name: mat.susceptibility * field_max for mat in materials}
        
        def animate(frame):
            ax.clear()
            field_strength = field_strengths[frame]
            
            ax.set_title(f'Animated Paramagnetic Response - {shape.title()}\nField: {field_strength:.1e} A/m\nTime: {time_points[frame]/(2*np.pi):.1f} cycles', 
                        fontsize=12, fontweight='bold')
            
            # Draw materials
            for material_name, obj_data in material_objects.items():
                material = obj_data['material']
                magnetization = material.susceptibility * field_strength
                color = self.get_magnetization_color(magnetization, max_magnetizations[material_name])
                
                poly3d = Poly3DCollection(obj_data['faces'], alpha=0.8, 
                                        facecolors=color, edgecolors='black', linewidths=0.5)
                ax.add_collection3d(poly3d)
                
                # Add labels
                pos = obj_data['position']
                ax.text(pos[0], pos[1], pos[2]-1, f'{material_name}', 
                       ha='center', va='top', fontsize=10, fontweight='bold')
            
            ax.set_xlim([-3, 3])
            ax.set_ylim([-2, 2])
            ax.set_zlim([-2, 2])
            ax.view_init(elev=20, azim=45)
            
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(time_points), 
                                     interval=50, blit=False, repeat=True)
        
        return fig, anim
    
    def create_comparison_dashboard(self, materials):
        """Create comprehensive 2D + 3D dashboard with detailed analysis."""
        
        print(f"\n=== COMPREHENSIVE DASHBOARD DATA ===")
        
        # Create figure with mixed 2D and 3D subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 3D visualization (left side)
        ax_3d = fig.add_subplot(2, 3, (1, 4), projection='3d')
        
        # Draw 3D materials at medium field strength
        field_strength = 5e5  # Medium field
        positions = [(-1, 0, 0), (1, 0, 0)]
        
        dashboard_data = []
        
        for i, material in enumerate(materials):
            faces = self.create_cube_vertices(positions[i], size=0.8)
            magnetization = material.susceptibility * field_strength
            max_mag = material.susceptibility * 1e6
            color = self.get_magnetization_color(magnetization, max_mag)
            
            poly3d = Poly3DCollection(faces, alpha=0.8, facecolors=color, 
                                    edgecolors='black', linewidths=1)
            ax_3d.add_collection3d(poly3d)
            
            pos = positions[i]
            ax_3d.text(pos[0], pos[1], pos[2]-1, material.name, 
                      ha='center', va='top', fontsize=10, fontweight='bold')
            
            # Collect data for table
            dashboard_data.append([
                material.name,
                f"{material.susceptibility:.2e}",
                f"{magnetization:.2e}",
                f"{material.density:.0f}",
                f"{material.curie_constant:.2e}"
            ])
        
        ax_3d.set_title('3D Material Visualization\n(Color = Magnetization Level)')
        ax_3d.set_xlim([-2, 2])
        ax_3d.set_ylim([-1, 1])
        ax_3d.set_zlim([-1, 1])
        
        # Print dashboard data table
        dashboard_headers = ["Material", "Susceptibility", "Magnetization @ 5e5 A/m", "Density (kg/m³)", "Curie Constant"]
        print(tabulate(dashboard_data, headers=dashboard_headers, tablefmt="grid"))
        
        # 2D Analysis plots (right side)
        
        # 1. Magnetization vs Field
        ax1 = fig.add_subplot(2, 3, 2)
        H_range = np.linspace(0, 1e6, 100)
        mag_data = []
        
        for material in materials:
            M = material.susceptibility * H_range
            ax1.plot(H_range/1e5, M*1e6, 'o-', linewidth=2, label=material.name)
            mag_data.append([material.name, f"{np.max(M):.2e}", f"{np.mean(M):.2e}"])
        
        ax1.set_xlabel('Field Strength (x10^5 A/m)')
        ax1.set_ylabel('Magnetization (x10^-6 A/m)')
        ax1.set_title('Static Response')
        ax1.legend()
        ax1.grid(True)
        
        # Print magnetization analysis table
        print(f"\n=== STATIC RESPONSE ANALYSIS ===")
        mag_headers = ["Material", "Max Magnetization", "Mean Magnetization"]
        print(tabulate(mag_data, headers=mag_headers, tablefmt="grid"))
        
        # 2. Temperature dependence  
        ax2 = fig.add_subplot(2, 3, 3)
        T_range = np.linspace(250, 400, 100)
        temp_data = []
        
        for material in materials:
            chi_T = material.curie_constant / T_range
            ax2.plot(T_range, chi_T*1e6, 'o-', linewidth=2, label=material.name)
            temp_data.append([
                material.name, 
                f"{chi_T[0]*1e6:.2e}",  # at 250K
                f"{chi_T[-1]*1e6:.2e}",  # at 400K
                f"{(chi_T[0]/chi_T[-1]):.1f}x"
            ])
        
        ax2.set_xlabel('Temperature (K)')
        ax2.set_ylabel('Susceptibility (x10^-6)')
        ax2.set_title('Temperature Effect')
        ax2.legend()
        ax2.grid(True)
        
        # Print temperature analysis table
        print(f"\n=== TEMPERATURE DEPENDENCE ANALYSIS ===")
        temp_headers = ["Material", "Susceptibility @ 250K", "Susceptibility @ 400K", "Temperature Ratio"]
        print(tabulate(temp_data, headers=temp_headers, tablefmt="grid"))
        
        # 3. Dynamic response
        ax3 = fig.add_subplot(2, 3, 5)
        t = np.linspace(0, 1, 1000)
        H_dynamic = 5e5 * np.sin(2*np.pi*10*t)  # 10 Hz
        dynamic_data = []
        
        for material in materials:
            M_dynamic = material.susceptibility * H_dynamic
            ax3.plot(t[:200], M_dynamic[:200]*1e6, linewidth=2, label=material.name)
            dynamic_data.append([
                material.name,
                f"{np.max(M_dynamic)*1e6:.2e}",
                f"{np.min(M_dynamic)*1e6:.2e}",
                "10 Hz"
            ])
        
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Magnetization (x10^-6 A/m)')
        ax3.set_title('Dynamic Response (10 Hz)')
        ax3.legend()
        ax3.grid(True)
        
        # Print dynamic analysis table
        print(f"\n=== DYNAMIC RESPONSE ANALYSIS ===")
        dynamic_headers = ["Material", "Peak Positive", "Peak Negative", "Frequency"]
        print(tabulate(dynamic_data, headers=dynamic_headers, tablefmt="grid"))
        
        # 4. Material comparison
        ax4 = fig.add_subplot(2, 3, 6)
        properties = ['Susceptibility\n(x10^-6)', 'Density\n(kg/m³)', 'Curie Const.\n(x10^-6)']
        aluminum_vals = [materials[0].susceptibility*1e6, materials[0].density/1000, materials[0].curie_constant*1e6]
        magnesium_vals = [materials[1].susceptibility*1e6, materials[1].density/1000, materials[1].curie_constant*1e6]
        
        x = np.arange(len(properties))
        width = 0.35
        ax4.bar(x - width/2, aluminum_vals, width, label='Aluminum', alpha=0.8)
        ax4.bar(x + width/2, magnesium_vals, width, label='Magnesium', alpha=0.8)
        ax4.set_xticks(x)
        ax4.set_xticklabels(properties)
        ax4.set_title('Material Properties')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

def run_real_3d_visualization():
    """Main function to run the real 3D visualization with comprehensive data output."""
    from material import get_material
    
    print("Real 3D Paramagnetic Material Visualization")
    print("=" * 50)
    
    # Get materials
    aluminum = get_material('aluminum')
    magnesium = get_material('magnesium')
    materials = [aluminum, magnesium]
    
    # Print initial material properties table
    print("\n=== MATERIAL PROPERTIES ===")
    material_props = []
    for material in materials:
        material_props.append([
            material.name,
            f"{material.susceptibility:.2e}",
            f"{material.density:.0f}",
            f"{material.curie_constant:.2e}"
        ])
    
    headers = ["Material", "Susceptibility", "Density (kg/m³)", "Curie Constant"]
    print(tabulate(material_props, headers=headers, tablefmt="grid"))
    
    # Create visualization system
    viz = Real3DParamagneticVisualization()
    
    print("\nCreating visualizations...")
    
    # 1. Static 3D visualization with different shapes
    field_strengths = [1e4, 5e4, 1e5, 5e5, 1e6]
    
    # Create cubes
    for shape in ['cube', 'cylinder', 'sphere']:
        print(f"\n  -> Creating {shape} visualization")
        figures = viz.plot_real_3d_materials(materials, field_strengths, shape=shape)
        # Show first figure of each shape
        if figures:
            figures[0].show()
    
    # 2. Combined 2D + 3D dashboard
    print("\n  -> Creating combined 2D + 3D dashboard")
    fig_dashboard = viz.create_comparison_dashboard(materials)
    fig_dashboard.show()
    
    # 3. Animated visualization
    print("\n  -> Creating animated visualization")
    fig_anim, anim = viz.create_animated_3d_visualization(materials, shape='cube')
    fig_anim.show()
    
    plt.show()
    
    print("\n" + "="*50)
    print("VISUALIZATION COMPLETE")
    print("="*50)
    print("WHAT YOU'RE SEEING:")
    print("- Blue/Dark = Low magnetization")
    print("- Yellow = Medium magnetization") 
    print("- Red/Bright = High magnetization")
    print("\n3D Objects show REAL material shapes")
    print("Colors change based on magnetic field strength")
    print("2D plots show detailed numerical analysis")
    print("\nRESULT: Aluminum gets more colorful (more magnetic) than Magnesium!")

if __name__ == "__main__":
    # Import required libraries - check if tabulate is available
    try:
        from tabulate import tabulate
    except ImportError:
        print("Installing required package: tabulate")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tabulate"])
        from tabulate import tabulate
    
    run_real_3d_visualization()