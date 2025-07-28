# main.py - Entry point for paramagnetic simulation system
"""
Main entry point for the Paramagnetic Materials Simulation System.
This script runs the complete simulation addressing all project objectives.
"""

import sys
import os

def main():
    """Main function to run the paramagnetic simulation."""
    print("=" * 60)
    print("PARAMAGNETIC MATERIALS SIMULATION SYSTEM")
    print("=" * 60)
    print("Entry Point: main.py")
    print("Starting simulation...")
    print()
    
    try:
        # Import the simulation function
        from simulation import run_paramagnetic_simulation
        
        # Run the simulation
        results, fig = run_paramagnetic_simulation()
        
        if results is not None:
            print("\nSimulation completed successfully!")
            print("Check the visualization window for results.")
            
            # Optional: Save results
            save_option = input("\nWould you like to save the results? (y/n): ").lower().strip()
            if save_option in ['y', 'yes']:
                try:
                    import pickle
                    with open('simulation_results.pkl', 'wb') as f:
                        pickle.dump(results, f)
                    print("Results saved to 'simulation_results.pkl'")
                except Exception as e:
                    print(f"Could not save results: {e}")
            
            # Optional: Save figure
            save_fig = input("Would you like to save the visualization? (y/n): ").lower().strip()
            if save_fig in ['y', 'yes']:
                try:
                    fig.savefig('paramagnetic_simulation_results.png', dpi=300, bbox_inches='tight')
                    print("Visualization saved to 'paramagnetic_simulation_results.png'")
                except Exception as e:
                    print(f"Could not save visualization: {e}")
                    
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
        print("3. Install required packages:")
        print("   pip install numpy matplotlib")
        return 1
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        print("Please check the error message and try again.")
        return 1
    
    print("\nThank you for using the Paramagnetic Simulation System!")
    return 0

if __name__ == "__main__":
    # Run main function and exit with appropriate code
    exit_code = main()
    sys.exit(exit_code)