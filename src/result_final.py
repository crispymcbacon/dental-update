import os
import pandas as pd

# Define patients and visits
patients = [1, 2, 3, 4, 5]
visits = [0, 1]

# Create lists to hold all summary and all individual DataFrames
all_summary_dfs = []
all_individual_dfs = []

for patient in patients:
    for visit in visits:
        # Construct filenames
        summary_filename = f"reconstruction_test_results_config_{patient}_visit_{visit}.csv"
        individual_filename = f"reconstruction_test_results_config_{patient}_visit_{visit}_individual.csv"

        # Safety check: make sure the file exists before reading
        if not os.path.exists(summary_filename):
            print(f"WARNING: File not found: {summary_filename}")
            continue
        if not os.path.exists(individual_filename):
            print(f"WARNING: File not found: {individual_filename}")
            continue
        
        # Read summary file
        df_summary = pd.read_csv(summary_filename)
        
        # Read individual file
        df_individual = pd.read_csv(individual_filename)

        # Print some information for tracking
        print(f"Processing patient {patient}, visit {visit} ...")
        print("Summary file head:")
        print(df_summary.head(), "\n")
        print("Individual file head:")
        print(df_individual.head(), "\n")

        # Optionally, add columns for Patient and Visit to keep track after concatenation
        df_summary["Patient"] = patient
        df_summary["Visit"] = visit

        df_individual["Patient"] = patient
        df_individual["Visit"] = visit

        # Collect data
        all_summary_dfs.append(df_summary)
        all_individual_dfs.append(df_individual)

# Concatenate all summary data into one DataFrame
if all_summary_dfs:
    summary_data = pd.concat(all_summary_dfs, ignore_index=True)
    print("\n=== Combined Summary Data (head) ===")
    print(summary_data.head())
else:
    summary_data = pd.DataFrame()
    print("No summary data to combine.")

# Concatenate all individual data into one DataFrame
if all_individual_dfs:
    individual_data = pd.concat(all_individual_dfs, ignore_index=True)
    print("\n=== Combined Individual Data (head) ===")
    print(individual_data.head())
else:
    individual_data = pd.DataFrame()
    print("No individual data to combine.")

if not summary_data.empty:
    # Example grouping by (Patient, Visit) and looking at "Overall Mean Centroid Distance" column
    # (Adjust the column name to match exactly what's in your CSV.)
    grouped_means = summary_data.groupby(["Patient", "Visit"])["Overall Mean Centroid Distance"].mean()

    print("\n=== Mean Overall Mean Centroid Distance per Patient/Visit ===")
    print(grouped_means)

# Similarly, from the individual data, you might do stats by point type, tooth number, etc.
if not individual_data.empty:
    grouped_individual = individual_data.groupby(["Patient", "Visit", "Point Type"])["Distance"].mean()
    print("\n=== Mean 'Distance' per (Patient, Visit, Point Type) from individual data ===")
    print(grouped_individual)

print("\nScript finished. You can now use 'summary_data' and 'individual_data' for further reporting.")
