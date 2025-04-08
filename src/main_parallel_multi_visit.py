from multiview_analyzer.multiview_analyzer import MultiViewAnalyzer
import plotly.io as pio
from Patient import Patient
from itertools import product
import multiprocessing
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import numpy as np
import json
import sys
import psutil
import pickle
import tempfile

# Disable visualizations by patching matplotlib to use a non-interactive backend
# This must be done before any other imports that might use matplotlib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend that doesn't open windows

# Disable plotly from opening browser windows
pio.renderers.default = 'png'  # Use static renderer instead of opening browser

# Override plt.show to do nothing
original_show = matplotlib.pyplot.show


def no_show(*args, **kwargs):
    pass


matplotlib.pyplot.show = no_show

# Import MultiViewAnalyzer to patch its triangulation method

# Store the original triangulation method
original_triangulation = MultiViewAnalyzer.triangulation

# Create a patched version that forces debug=False


def triangulation_no_debug(self, *args, **kwargs):
    # Force debug to False to prevent browser tabs from opening
    kwargs['debug'] = False
    return original_triangulation(self, *args, **kwargs)


# Apply the patch
MultiViewAnalyzer.triangulation = triangulation_no_debug


def run_test_reconstruction(args):
    """
    Worker function to run a single test reconstruction with given parameters.
    This function will be executed in a separate process.

    Args:
        args: Tuple containing (config_path, visit_id, output_file, params, process_id)

    Returns:
        Dictionary with test results
    """
    config_path, visit_id, output_file, params, process_id = args

    # Set process affinity to improve CPU cache utilization
    try:
        proc = psutil.Process()
        # Assign this process to a specific CPU core based on process_id
        # This helps reduce context switching and improves cache locality
        num_cpus = psutil.cpu_count(logical=False)  # Physical cores only
        if num_cpus > 1:
            # Assign to physical cores in a round-robin fashion
            target_cpu = process_id % num_cpus
            proc.cpu_affinity([target_cpu])
    except Exception:
        # If setting CPU affinity fails, just continue without it
        pass

    # Create a new Patient instance for this process
    patient = Patient(config_path)

    # Initialize and add visits (required before triangulation)
    patient.process_all_visits([visit_id], ["initialize", "add_visits"])

    # Explicitly set debug=False in the test_reconstruction call
    try:
        # Create a copy of params and ensure debug is set to False
        params_copy = params.copy()

        # Force debug to False to prevent any visualizations
        # This is needed because the Patient.test_reconstruction method calls triangulation with debug=True
        refined_cameras, final_refined_teeth, alpha = patient.test_reconstruction(
            visit_id=visit_id,
            output_file=output_file,
            **params_copy
        )

        # Try to get the scaling factor from the results file to include in the return value
        scaling_factor = None
        try:
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                with open(output_file, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows and 'Scaling Factor (mm)' in rows[0]:
                        scaling_factor = rows[0]['Scaling Factor (mm)']
        except Exception:
            # If we can't read the scaling factor, just continue without it
            pass

        return {
            'status': 'success',
            'params': params,
            'alpha': alpha,
            'scaling_factor': scaling_factor,
            'visit_id': visit_id
        }
    except Exception as e:
        # Return error information if the test fails
        return {
            'status': 'error',
            'params': params,
            'error': str(e),
            'visit_id': visit_id
        }


def format_params_for_display(params):
    """Format parameters for display in console output"""
    # Create a copy to avoid modifying the original
    formatted = copy.deepcopy(params)

    # Format camera_rotations to be more readable
    if 'camera_rotations' in formatted and formatted['camera_rotations'] is not None:
        rotations = formatted['camera_rotations']
        formatted['camera_rotations'] = f"{len(rotations.keys())} views: {list(rotations.keys())}"

    return formatted


def chunk_parameters(params_list, num_chunks):
    """
    Split the parameters list into chunks for batch processing

    Args:
        params_list: List of parameter dictionaries
        num_chunks: Number of chunks to split into

    Returns:
        List of parameter list chunks
    """
    avg_chunk_size = max(1, len(params_list) // num_chunks)
    return [params_list[i:i + avg_chunk_size] for i in range(0, len(params_list), avg_chunk_size)]


def create_shared_patient_instance(config_path, visit_id):
    """
    Create a Patient instance and pickle it to a temporary file for sharing

    Args:
        config_path: Path to the configuration file
        visit_id: Visit ID to process

    Returns:
        Path to the temporary file containing the pickled Patient instance
    """
    # Create a Patient instance
    patient = Patient(config_path)

    # Initialize and add visits (required before triangulation)
    patient.process_all_visits([visit_id], ["initialize", "add_visits"])

    # Pickle the Patient instance to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
    with open(temp_file.name, 'wb') as f:
        pickle.dump(patient, f)

    return temp_file.name


if __name__ == "__main__":
    start_time = time.time()

    # Configuration
    # Edit with your config files
    config_files = [f"config/config_{i}.json" for i in range(1, 6)]
    visit_ids = [0, 1]  # Process both visit 0 and visit 1

    # Process each config file
    for config_idx, config_path in enumerate(config_files, 1):
        # Process each visit for this config
        for visit_id in visit_ids:
            config_start_time = time.time()

            # Create unique output files for this config and visit
            test_output_file = f"reconstruction_test_results_config_{config_idx}_visit_{visit_id}.csv"

            print(f"\n{'='*80}")
            print(
                f"PROCESSING CONFIG FILE {config_idx}/5: {config_path}, VISIT {visit_id}")
            print(f"{'='*80}\n")

            # Define parameter options
            param_options = {
                'upper_lower_weight': [1.5],
                'camera_translation': [True],
                'camera_rotations': [
                    {
                        'frontal': 1.0,  # assuming frontal is reasonably reliable
                        'upper': 0.5,    # lower weight because upper is very reliable
                        'lower': 0.5,    # lower weight because lower is very reliable
                        'left': 1.5,     # higher weight to allow more correction since left is not reliable
                        'right': 1.5     # higher weight to allow more correction since right is not reliable
                    }
                ],
                'use_arch_constraints': [True],
                'use_only_centroids': [False],
                'use_alpha': [True]
            }

            # Generate all combinations of parameters
            param_names = list(param_options.keys())
            combinations = list(product(*param_options.values()))
            params_list = [dict(zip(param_names, combination))
                           for combination in combinations]

            # Create a unique temporary output file for each process
            # This avoids file access conflicts when multiple processes try to write to the same file
            temp_output_files = [
                f"temp_results_config_{config_idx}_visit_{visit_id}_{i}.csv" for i in range(len(params_list))]

            # Determine optimal number of processes based on system resources
            physical_cores = psutil.cpu_count(logical=False) or 1
            logical_cores = psutil.cpu_count(logical=True) or 1

            # Check available memory to avoid overcommitting
            mem = psutil.virtual_memory()
            mem_per_process_gb = 1  # Estimated memory usage per process in GB
            max_processes_by_memory = max(
                1, int(mem.available / (mem_per_process_gb * 1024**3)))

            # Use the minimum of cores-1 and memory-based limit
            num_processes = min(max(1, logical_cores - 1),
                                max_processes_by_memory)

            print(
                f"System resources: {physical_cores} physical cores, {logical_cores} logical cores")
            print(f"Available memory: {mem.available / (1024**3):.2f} GB")
            print(
                f"Starting grid search with {len(params_list)} parameter combinations using {num_processes} processes")

            # Track progress
            total_combinations = len(params_list)
            completed = 0

            # Prepare arguments for the worker function with process IDs
            args_list = [(config_path, visit_id, temp_file, params, i % num_processes)
                         for i, (temp_file, params) in enumerate(zip(temp_output_files, params_list))]

            # Run tests in parallel using ProcessPoolExecutor with optimized batch submission
            results = []

            # Split the parameters into batches to reduce overhead
            batch_size = min(
                20, max(1, total_combinations // (num_processes * 2)))
            batched_args = [args_list[i:i + batch_size]
                            for i in range(0, len(args_list), batch_size)]

            with ProcessPoolExecutor(max_workers=num_processes,
                                     mp_context=multiprocessing.get_context('spawn')) as executor:
                # Process batches of tasks
                for batch_idx, batch in enumerate(batched_args):
                    print(
                        f"Submitting batch {batch_idx+1}/{len(batched_args)} with {len(batch)} tasks")

                    # Submit batch of tasks
                    future_to_params = {executor.submit(run_test_reconstruction, args): args[3]
                                        for args in batch}

                    # Process results as they complete
                    for future in as_completed(future_to_params):
                        params = future_to_params[future]
                        try:
                            result = future.result()
                            results.append(result)

                            # Update progress
                            completed += 1
                            progress = (completed / total_combinations) * 100
                            elapsed_time = time.time() - config_start_time
                            estimated_total = elapsed_time / completed * \
                                total_combinations if completed > 0 else 0
                            remaining_time = estimated_total - elapsed_time

                            # Print progress information
                            print(f"Config {config_idx}/5, Visit {visit_id} - [{completed}/{total_combinations}] {progress:.1f}% complete - "
                                  f"Elapsed: {elapsed_time/60:.1f}m, Remaining: {remaining_time/60:.1f}m")
                            print(
                                f"Completed: {format_params_for_display(params)}")
                            print(f"Status: {result['status']}")
                            if result['status'] == 'success':
                                print(f"Alpha: {result['alpha']:.4f}")
                                if 'scaling_factor' in result and result['scaling_factor']:
                                    print(
                                        f"Scaling Factor (mm): {result['scaling_factor']}")
                            else:
                                print(f"Error: {result['error']}")
                            print("-" * 80)

                        except Exception as exc:
                            print(
                                f"Task for {params} generated an exception: {exc}")

            # Merge all temporary result files into the final output file
            print(f"\nMerging results into {test_output_file}...")

            # Check if any temporary files exist
            existing_temp_files = [
                f for f in temp_output_files if os.path.exists(f)]

            if existing_temp_files:
                # Read all rows from all temporary files
                all_rows = []
                all_individual_rows = []

                for temp_file in existing_temp_files:
                    # Read main results
                    if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                        with open(temp_file, 'r', newline='') as f:
                            reader = csv.DictReader(f)
                            all_rows.extend(list(reader))

                    # Read individual results
                    individual_file = temp_file.replace(
                        '.csv', '_individual.csv')
                    if os.path.exists(individual_file) and os.path.getsize(individual_file) > 0:
                        with open(individual_file, 'r', newline='') as f:
                            reader = csv.DictReader(f)
                            all_individual_rows.extend(list(reader))

                # Write merged results to the final output file
                if all_rows:
                    with open(test_output_file, 'w', newline='') as f:
                        writer = csv.DictWriter(
                            f, fieldnames=all_rows[0].keys())
                        writer.writeheader()
                        writer.writerows(all_rows)

                # Write merged individual results to the final individual output file
                individual_output_file = test_output_file.replace(
                    '.csv', '_individual.csv')
                if all_individual_rows:
                    with open(individual_output_file, 'w', newline='') as f:
                        writer = csv.DictWriter(
                            f, fieldnames=all_individual_rows[0].keys())
                        writer.writeheader()
                        writer.writerows(all_individual_rows)

                # Clean up temporary files
                for temp_file in existing_temp_files:
                    try:
                        os.remove(temp_file)
                        individual_file = temp_file.replace(
                            '.csv', '_individual.csv')
                        if os.path.exists(individual_file):
                            os.remove(individual_file)
                    except Exception as e:
                        print(
                            f"Warning: Could not remove temporary file {temp_file}: {e}")

            # Print summary for this config and visit
            config_total_time = time.time() - config_start_time
            print(
                f"\nGrid search for config {config_idx}/5, visit {visit_id} completed in {config_total_time/60:.2f} minutes")
            print(f"Tested {total_combinations} parameter combinations")
            print(
                f"Results saved to {test_output_file} and {test_output_file.replace('.csv', '_individual.csv')}")

            # Find and print the best parameter combination based on overall mean centroid distance
            if os.path.exists(test_output_file) and os.path.getsize(test_output_file) > 0:
                with open(test_output_file, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)

                    if rows:
                        # Convert string values to appropriate types for sorting
                        for row in rows:
                            row['Overall Mean Centroid Distance'] = float(
                                row['Overall Mean Centroid Distance'])
                            # Also convert mm values if they exist
                            if 'Overall Mean Centroid Distance (mm)' in row and row['Overall Mean Centroid Distance (mm)']:
                                row['Overall Mean Centroid Distance (mm)'] = float(
                                    row['Overall Mean Centroid Distance (mm)'])

                        # Sort by overall mean centroid distance (lower is better)
                        sorted_rows = sorted(
                            rows, key=lambda x: x['Overall Mean Centroid Distance'])

                        # Print the top 3 best parameter combinations
                        print(
                            f"\nTop 3 best parameter combinations for config {config_idx}/5, visit {visit_id}:")
                        for i, row in enumerate(sorted_rows[:3]):
                            print(
                                f"{i+1}. Overall Mean Centroid Distance: {row['Overall Mean Centroid Distance']:.4f}")
                            # Print mm value if available
                            if 'Overall Mean Centroid Distance (mm)' in row and row['Overall Mean Centroid Distance (mm)']:
                                print(
                                    f"   Overall Mean Centroid Distance (mm): {float(row['Overall Mean Centroid Distance (mm)']):.4f}")
                            if 'Scaling Factor (mm)' in row and row['Scaling Factor (mm)']:
                                print(
                                    f"   Scaling Factor (mm): {row['Scaling Factor (mm)']}")
                            print(
                                f"   Upper-Lower Weight: {row['Upper-Lower Weight']}")
                            print(
                                f"   Camera Translation: {row['Camera Translation']}")
                            print(
                                f"   Camera Rotations: {row['Camera Rotations']}")
                            print(
                                f"   Use Arch Constraints: {row['Use Arch Constraints']}")
                            print(
                                f"   Use Only Centroids: {row['Use Only Centroids']}")
                            print(f"   Use Alpha: {row['Use Alpha']}")
                            print(f"   Alpha Value: {row['Alpha Value']}")
                            print()

    # Print overall summary
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"ALL CONFIGURATIONS COMPLETED")
    print(f"{'='*80}")
    print(f"Total execution time: {total_time/60:.2f} minutes")
    print(
        f"Processed {len(config_files)} configuration files with {len(visit_ids)} visits each")
    print(
        f"Results saved to reconstruction_test_results_config_[1-5]_visit_[0-1].csv files")
