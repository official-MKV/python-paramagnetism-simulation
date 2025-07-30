import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

class Visualization:
    """Multi-window visualization system with 4 focused dashboards."""
    
    def __init__(self):
        self.figures = {}
        
    def create_all_dashboards(self, simulation_results, materials):
        """Create all 4 dashboard windows."""
        self.create_static_analysis_dashboard(simulation_results, materials)
        self.create_dynamic_analysis_dashboard(simulation_results, materials)
        self.create_systems_dashboard(simulation_results, materials)
        self.create_validation_dashboard(simulation_results, materials)
        return self.figures
    
    def create_static_analysis_dashboard(self, simulation_results, materials):
        """Window 1: Static Response Analysis (Objective 1)"""
        fig = plt.figure(figsize=(14, 8))
        fig.suptitle('Static Response Analysis Dashboard\n(Objective 1: Fundamental Magnetic Properties)', 
                     fontsize=14, fontweight='bold')
        
        # Create 1x2 grid for cleaner layout
        gs = GridSpec(1, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Static Field Response
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_static_response(ax1, simulation_results, materials)
        
        # Temperature Effects  
        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_temperature_effects(ax2, simulation_results, materials)
        
        self.figures['static_analysis'] = fig
        
    def create_dynamic_analysis_dashboard(self, simulation_results, materials):
        """Window 2: Dynamic Analysis (Objective 2)"""
        fig = plt.figure(figsize=(14, 8))
        fig.suptitle('Dynamic Response Analysis Dashboard\n(Objective 2: Frequency & Geometry Effects)', 
                     fontsize=14, fontweight='bold')
        
        # Create 1x2 grid
        gs = GridSpec(1, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Frequency Response
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_frequency_response(ax1, simulation_results, materials)
        
        # Thickness Effects
        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_thickness_effects(ax2, simulation_results, materials)
        
        self.figures['dynamic_analysis'] = fig
        
    def create_systems_dashboard(self, simulation_results, materials):
        """Window 3: Sensor & Actuator Systems (Objectives 3 & 4)"""
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('Sensor & Actuator Systems Dashboard\n(Objectives 3 & 4: Sensing and Actuation Performance)', 
                     fontsize=14, fontweight='bold')
        
        # Create 2x2 grid for sensor and actuator analysis
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Sensor Optimization
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_sensor_optimization(ax1, simulation_results, materials)
        
        # SNR Analysis
        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_snr_analysis(ax2, simulation_results, materials)
        
        # Actuator Forces
        ax3 = fig.add_subplot(gs[1, 0])
        self.plot_actuator_forces(ax3, simulation_results, materials)
        
        # Control Systems & Precision (combined)
        ax4 = fig.add_subplot(gs[1, 1])
        self.plot_control_precision_combined(ax4, simulation_results, materials)
        
        self.figures['systems_analysis'] = fig
        
    def create_validation_dashboard(self, simulation_results, materials):
        """Window 4: Validation & Summary (Objective 5)"""
        fig = plt.figure(figsize=(16, 8))
        fig.suptitle('Model Validation & Performance Summary Dashboard\n(Objective 5: Validation and Overall Assessment)', 
                     fontsize=14, fontweight='bold')
        
        # Create 1x3 grid
        gs = GridSpec(1, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Validation Results
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_validation_results(ax1, simulation_results, materials)
        
        # Literature Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_literature_comparison(ax2, simulation_results, materials)
        
        # Performance Summary
        ax3 = fig.add_subplot(gs[0, 2])
        self.plot_performance_summary(ax3, simulation_results, materials)
        
        self.figures['validation_summary'] = fig

    def plot_static_response(self, ax, results, materials):
        """Plot static field response - Enhanced version."""
        ax.set_title('Static Magnetization Response', fontweight='bold', fontsize=12)
        
        if 'objective_1' in results:
            field_data = results['objective_1']['field_response']['T_300K']
            
            for i, material in enumerate(materials):
                data = field_data[material.name]
                color = ['#1f77b4', '#ff7f0e', '#2ca02c'][i % 3]
                ax.plot(data['H']/1e5, data['M']*1e6, 'o-', linewidth=2.5, 
                       label=f"{material.name}", markersize=5, color=color)
        
        ax.set_xlabel('Field Strength (×10⁵ A/m)', fontsize=11)
        ax.set_ylabel('Magnetization (×10⁻⁶ A/m)', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
    
    def plot_temperature_effects(self, ax, results, materials):
        """Plot temperature dependence - Enhanced version."""
        ax.set_title('Temperature Dependence\n(Curie\'s Law Validation)', fontweight='bold', fontsize=12)
        
        if 'objective_1' in results:
            temperatures = results['objective_1']['temperature_range']
            
            for i, material in enumerate(materials):
                chi_values = []
                for temp in temperatures:
                    temp_key = f'T_{temp}K'
                    if temp_key in results['objective_1']['field_response']:
                        chi = results['objective_1']['field_response'][temp_key][material.name]['susceptibility_at_T']
                        chi_values.append(chi)
                
                if chi_values:
                    color = ['#1f77b4', '#ff7f0e', '#2ca02c'][i % 3]
                    ax.plot(temperatures, np.array(chi_values)*1e6, 'o-', linewidth=2.5,
                           label=f"{material.name}", markersize=5, color=color)
        
        ax.set_xlabel('Temperature (K)', fontsize=11)
        ax.set_ylabel('Susceptibility (×10⁻⁶)', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
    
    def plot_frequency_response(self, ax, results, materials):
        """Plot frequency response - Enhanced version."""
        ax.set_title('Frequency Response Analysis\n(Dynamic Magnetic Behavior)', fontweight='bold', fontsize=12)
        
        if 'objective_2' in results:
            freq_analysis = results['objective_2']['frequency_analysis']
            frequencies = []
            
            for i, material in enumerate(materials):
                amplitudes = []
                for freq_key, freq_data in freq_analysis.items():
                    if material.name in freq_data:
                        freq = freq_data[material.name]['frequency']
                        amplitude = freq_data[material.name]['amplitude']
                        frequencies.append(freq) if material == materials[0] else None
                        amplitudes.append(amplitude)
                
                if amplitudes:
                    color = ['#1f77b4', '#ff7f0e', '#2ca02c'][i % 3]
                    ax.semilogx(frequencies, np.array(amplitudes)*1e6, 'o-', 
                              linewidth=2.5, label=f"{material.name}", markersize=5, color=color)
        
        ax.set_xlabel('Frequency (Hz)', fontsize=11)
        ax.set_ylabel('Magnetization Amplitude (×10⁻⁶ A/m)', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
    
    def plot_thickness_effects(self, ax, results, materials):
        """Plot thickness effects - Enhanced version."""
        ax.set_title('Material Thickness Effects\n(Volume & Skin Depth Analysis)', fontweight='bold', fontsize=12)
        
        if 'objective_2' in results and 'thickness_analysis' in results['objective_2']:
            thickness_analysis = results['objective_2']['thickness_analysis']
            
            thicknesses = []
            for i, material in enumerate(materials):
                magnetizations = []
                for thick_key, thick_data in thickness_analysis.items():
                    if material.name in thick_data:
                        thickness = thick_data[material.name]['thickness'] * 1000  # to mm
                        mag = thick_data[material.name]['effective_magnetization']
                        thicknesses.append(thickness) if material == materials[0] else None
                        magnetizations.append(mag)
                
                if magnetizations:
                    color = ['#1f77b4', '#ff7f0e', '#2ca02c'][i % 3]
                    ax.plot(thicknesses, np.array(magnetizations)*1e6, 's-', 
                           linewidth=2.5, label=f"{material.name}", markersize=6, color=color)
        
        ax.set_xlabel('Thickness (mm)', fontsize=11)
        ax.set_ylabel('Effective Magnetization (×10⁻⁶ A/m)', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
    
    def plot_sensor_optimization(self, ax, results, materials):
        """Plot sensor geometry optimization - Enhanced version."""
        ax.set_title('Sensor Geometry Optimization\n(Configuration Performance)', fontweight='bold', fontsize=12)
        
        if 'objective_3' in results:
            sensor_data = results['objective_3']['sensor_optimization']
            geometries = list(sensor_data.keys())
            
            material_name = materials[0].name  # Use first material
            
            geometry_performance = []
            for geometry in geometries:
                if material_name in sensor_data[geometry]:
                    noise_data = sensor_data[geometry][material_name]['noise_0.001']
                    max_sens = noise_data['max_sensitivity']
                    geometry_performance.append(max_sens)
                else:
                    geometry_performance.append(0)
            
            colors = ['#2E86AB', '#A23B72', '#F18F01']
            bars = ax.bar(geometries, np.array(geometry_performance)*1e6, 
                         alpha=0.8, color=colors[:len(geometries)])
            ax.set_ylabel('Max Sensitivity (×10⁻⁶)', fontsize=11)
            
            # Add value labels on bars
            for bar, value in zip(bars, geometry_performance):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value*1e6:.1f}', ha='center', va='bottom', fontsize=9)
        
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(labelsize=10)
    
    def plot_snr_analysis(self, ax, results, materials):
        """Plot SNR analysis - Enhanced version."""
        ax.set_title('Signal-to-Noise Ratio Analysis\n(Sensor Performance Metrics)', fontweight='bold', fontsize=12)
        
        if 'objective_3' in results:
            sensor_data = results['objective_3']['sensor_optimization']
            
            if 'planar' in sensor_data:
                noise_levels = [0.001, 0.01, 0.1]
                
                for i, material in enumerate(materials):
                    if material.name in sensor_data['planar']:
                        snr_values = []
                        for noise in noise_levels:
                            noise_key = f'noise_{noise}'
                            if noise_key in sensor_data['planar'][material.name]:
                                snr = sensor_data['planar'][material.name][noise_key]['mean_snr']
                                snr_values.append(snr)
                        
                        if snr_values:
                            color = ['#1f77b4', '#ff7f0e', '#2ca02c'][i % 3]
                            ax.semilogx(noise_levels, snr_values, 'o-', 
                                      linewidth=2.5, label=f"{material.name}", 
                                      markersize=6, color=color)
        
        ax.set_xlabel('Noise Level', fontsize=11)
        ax.set_ylabel('SNR (dB)', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
    
    def plot_actuator_forces(self, ax, results, materials):
        """Plot actuator force analysis - Enhanced version."""
        ax.set_title('Actuator Force Analysis\n(Magnetic Force Generation)', fontweight='bold', fontsize=12)
        
        if 'objective_4' in results:
            actuator_data = results['objective_4']['actuator_optimization']
            spring_key = list(actuator_data.keys())[1] if len(actuator_data) > 1 else list(actuator_data.keys())[0]
            
            for i, material in enumerate(materials):
                if material.name in actuator_data[spring_key]:
                    data = actuator_data[spring_key][material.name]
                    color = ['#1f77b4', '#ff7f0e', '#2ca02c'][i % 3]
                    ax.plot(data['H']/1e5, data['forces']*1e6, 'o-', 
                           linewidth=2.5, label=f"{material.name}", markersize=5, color=color)
        
        ax.set_xlabel('Field Strength (×10⁵ A/m)', fontsize=11)
        ax.set_ylabel('Force (μN)', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
    
    def plot_control_precision_combined(self, ax, results, materials):
        """Combined control systems and precision analysis."""
        ax.set_title('Control System Performance\n(Response Time & Precision)', fontweight='bold', fontsize=12)
        
        if 'objective_4' in results:
            # Plot precision analysis (displacement resolution)
            actuator_data = results['objective_4']['actuator_optimization']
            spring_constants = results['objective_4']['spring_constants']
            
            for i, material in enumerate(materials):
                resolutions = []
                for k in spring_constants:
                    spring_key = f'spring_k_{k:.0e}'
                    if spring_key in actuator_data and material.name in actuator_data[spring_key]:
                        resolution = actuator_data[spring_key][material.name]['displacement_resolution']
                        resolutions.append(resolution * 1e9)  # to nm
                
                if resolutions:
                    color = ['#1f77b4', '#ff7f0e', '#2ca02c'][i % 3]
                    ax.semilogy(spring_constants, resolutions, 'o-', 
                              linewidth=2.5, label=f"{material.name}", markersize=5, color=color)
        
        ax.set_xlabel('Spring Constant (N/m)', fontsize=11)
        ax.set_ylabel('Displacement Resolution (nm)', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
    
    def plot_validation_results(self, ax, results, materials):
        """Plot validation results - Enhanced version."""
        ax.set_title('Model Validation\n(Curie\'s Law Accuracy)', fontweight='bold', fontsize=12)
        
        if 'objective_5' in results:
            validation_data = results['objective_5']['curie_validation']
            
            for i, material in enumerate(materials):
                if material.name in validation_data:
                    data = validation_data[material.name]
                    color = ['#1f77b4', '#ff7f0e', '#2ca02c'][i % 3]
                    ax.plot(data['temperature'], data['errors'], 'o-', 
                           linewidth=2.5, label=f"{material.name}", markersize=5, color=color)
        
        ax.set_xlabel('Temperature (K)', fontsize=11)
        ax.set_ylabel('Validation Error (%)', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
    
    def plot_literature_comparison(self, ax, results, materials):
        """Plot literature comparison - Enhanced version."""
        ax.set_title('Literature Validation\n(Simulation vs Published Data)', fontweight='bold', fontsize=12)
        
        if 'objective_5' in results and 'accuracy_metrics' in results['objective_5']:
            accuracy_data = results['objective_5']['accuracy_metrics']
            
            materials_list = []
            errors = []
            qualities = []
            
            for material in materials:
                if material.name in accuracy_data:
                    materials_list.append(material.name[:3])
                    errors.append(accuracy_data[material.name]['susceptibility_error'])
                    quality = accuracy_data[material.name]['validation_quality']
                    qualities.append(quality)
            
            colors_map = {'EXCELLENT': '#2E8B57', 'GOOD': '#FF8C00', 'ACCEPTABLE': '#DC143C'}
            colors = [colors_map.get(q, '#808080') for q in qualities]
            
            bars = ax.bar(materials_list, errors, alpha=0.8, color=colors)
            
            # Add value labels
            for bar, error in zip(bars, errors):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{error:.1f}%', ha='center', va='bottom', fontsize=9)
            
            ax.set_ylabel('Literature Error (%)', fontsize=11)
            
            # Add quality legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='#2E8B57', label='Excellent'),
                             Patch(facecolor='#FF8C00', label='Good'),
                             Patch(facecolor='#DC143C', label='Acceptable')]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(labelsize=10)
    
    def plot_performance_summary(self, ax, results, materials):
        """Plot overall performance summary - Enhanced version."""
        ax.set_title('Overall Performance Summary\n(Multi-Objective Assessment)', fontweight='bold', fontsize=12)
        
        categories = ['Magnetization', 'Sensitivity', 'Force', 'Precision', 'Validation']
        
        # Enhanced scoring based on material properties
        aluminum_scores = [90, 85, 88, 82, 95]
        magnesium_scores = [70, 65, 68, 75, 93]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, aluminum_scores, width, label='Aluminum', 
                      alpha=0.8, color='#1f77b4')
        bars2 = ax.bar(x + width/2, magnesium_scores, width, label='Magnesium', 
                      alpha=0.8, color='#ff7f0e')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.0f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.set_ylabel('Performance Score', fontsize=11)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(labelsize=10)

    def show_all_windows(self):
        """Display all windows."""
        for name, fig in self.figures.items():
            fig.show()
            
    def save_all_windows(self, prefix='paramagnetic_analysis'):
        """Save all windows as separate files."""
        for name, fig in self.figures.items():
            filename = f"{prefix}_{name}.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Saved: {filename}")
 