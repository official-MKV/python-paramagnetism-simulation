"""
Main entry point for the Paramagnetic Materials Simulation System.
This script runs the complete simulation addressing all project objectives.
Updated to work with the new 4-window visualization system.
"""

import sys
import os

def main():
    """Main function to run the paramagnetic simulation."""
    print("=" * 60)
    print("PARAMAGNETIC MATERIALS SIMULATION SYSTEM")
    print("=" * 60)
    print("Entry Point: main.py")
    print("Starting simulation with 4-window visualization...")
    print()
    
    try:
        # Import the simulation function
        from simulation import run_paramagnetic_simulation
        
        # Run the simulation - now returns results and multiple figures
        results, figures = run_paramagnetic_simulation()
        
        if results is not None and figures is not None:
            print("\nSimulation completed successfully!")
            print(f"Generated {len(figures)} visualization windows:")
            for window_name in figures.keys():
                print(f"  - {window_name.replace('_', ' ').title()}")
            
            # Automatically display all windows
            print("\nDisplaying all visualization windows...")
            show_all_windows(figures)
            
            print("Check the visualization windows for results.")
                    
        else:
            print("Simulation failed. Check error messages above.")
            return 1
            
    except ImportError as e:
        print(f"Error: Could not import simulation module: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure simulation.py is in the same directory")
        print("2. Check that all required modules are available:")
        print("   - material.py")
        print("   - magnetization.py") 
        print("   - field.py")
        print("   - sensor.py")
        print("   - actuator.py")
        print("   - visualization.py")
        print("3. Install required packages:")
        print("   pip install numpy matplotlib")
        return 1
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        print("Please check the error message and try again.")
        return 1
    
    print("\nThank you for using the Paramagnetic Simulation System!")
    return 0

def show_all_windows(figures):
    """Display all visualization windows."""
    try:
        import matplotlib.pyplot as plt
        
        for name, fig in figures.items():
            fig.show()
        
        # Keep all windows open
        plt.show()
        
    except Exception as e:
        print(f"Error displaying windows: {e}")

if __name__ == "__main__":
    # Run main function and exit with appropriate code
    exit_code = main()
    sys.exit(exit_code)