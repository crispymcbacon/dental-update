import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

###############################################################################
# 1. DATA LOADING
###############################################################################

def load_config_data():
    """
    Loads all config_{patient}_visit_{visit}.csv files (excluding patient 3)
    into a single pandas DataFrame.
    Adds columns 'PatientID' and 'VisitID' based on the filename.
    """
    all_rows = []

    # We expect patient_id in 1..5, visit_id in 0..1.
    for patient_id in range(1, 5):
        for visit_id in range(2):  # 0 or 1
            filename = f"results/config_{patient_id}_visit_{visit_id}.csv"
            if os.path.exists(filename):
                df_temp = pd.read_csv(filename)
                df_temp['PatientID'] = patient_id
                df_temp['VisitID'] = visit_id
                all_rows.append(df_temp)
            else:
                print(f"WARNING: File not found: {filename}")

    if not all_rows:
        raise FileNotFoundError("No config_{patient}_visit_{visit}.csv files were loaded.")

    df_config = pd.concat(all_rows, ignore_index=True)
    return df_config


def load_individual_data():
    """
    Loads all config_{patient}_visit_{visit}_individual.csv files (excluding patient 3)
    into a single pandas DataFrame.
    Adds columns 'PatientID' and 'VisitID' based on the filename.
    """
    all_rows = []

    for patient_id in range(1, 5):
        for visit_id in range(2):  # 0 or 1
            filename = f"results/config_{patient_id}_visit_{visit_id}_individual.csv"
            if os.path.exists(filename):
                df_temp = pd.read_csv(filename)
                df_temp['PatientID'] = patient_id
                df_temp['VisitID'] = visit_id
                all_rows.append(df_temp)
            else:
                print(f"WARNING: File not found: {filename}")

    if not all_rows:
        raise FileNotFoundError("No config_{patient}_visit_{visit}_individual.csv files were loaded.")

    df_individual = pd.concat(all_rows, ignore_index=True)
    return df_individual


###############################################################################
# 2. HELPER FUNCTIONS
###############################################################################

def boolean_from_string(val):
    """
    Converts text like 'True'/'False' to actual bool if needed.
    """
    if str(val).lower() == 'true':
        return True
    elif str(val).lower() == 'false':
        return False
    return val

###############################################################################
# 3. MAIN ANALYSIS AND PLOTTING
###############################################################################

def main():
    ###########################################################################
    # 3.1 LOAD DATA
    ###########################################################################
    print("Loading config data (overall metrics and parameters)...")
    df_config = load_config_data()

    print("Loading individual data (tooth-level metrics)...")
    df_individual = load_individual_data()

    ###########################################################################
    # 3.2 DATA CLEANING / TYPE CONVERSION
    ###########################################################################
    # Boolean columns
    for col in ['Use Arch Constraints', 'Use Only Centroids', 'Use Alpha', 'Camera Translation', 'Camera Rotations']:
        if col in df_config.columns:
            df_config[col] = df_config[col].apply(boolean_from_string)

    # Numeric columns in the config DataFrame
    numeric_cols = [
        'Upper-Lower Weight',
        'Alpha Value',
        'Upper Mean Centroid Distance (mm)',
        'Upper Mean All Points Distance (mm)',
        'Lower Mean Centroid Distance (mm)',
        'Lower Mean All Points Distance (mm)',
        'Overall Mean Centroid Distance (mm)',
        'Overall Mean All Points Distance (mm)',
        'Scaling Factor (mm)',
    ]
    for col in numeric_cols:
        if col in df_config.columns:
            df_config[col] = pd.to_numeric(df_config[col], errors='coerce')

    # Numeric columns in the individual DataFrame
    if 'Distance' in df_individual.columns:
        df_individual['Distance'] = pd.to_numeric(df_individual['Distance'], errors='coerce')
    if 'Distance (mm)' in df_individual.columns:
        df_individual['Distance (mm)'] = pd.to_numeric(df_individual['Distance (mm)'], errors='coerce')

    # -------------------------------------------------------------------------
    # ADJUST FOR THE 2-DECIMAL SHIFT IN MM COLUMNS
    # -------------------------------------------------------------------------
    mm_cols_config = [c for c in numeric_cols if '(mm)' in c and c in df_config.columns]
    for col in mm_cols_config:
        df_config[col] = df_config[col] / 100.0

    if 'Distance (mm)' in df_individual.columns:
        df_individual['Distance (mm)'] = df_individual['Distance (mm)'] / 100.0

    ###########################################################################
    # 3.3 DESCRIPTIVE STATISTICS (CONFIG DATA)
    ###########################################################################
    print("\n========== DESCRIPTIVE STATISTICS: CONFIG DATA ==========")
    dist_cols = [
        'Overall Mean Centroid Distance (mm)',
        'Overall Mean All Points Distance (mm)',
        'Upper Mean Centroid Distance (mm)',
        'Lower Mean Centroid Distance (mm)',
        'Upper Mean All Points Distance (mm)',
        'Lower Mean All Points Distance (mm)'
    ]

    for col in dist_cols:
        if col in df_config.columns:
            print(f"\n--- {col} ---")
            # The value in df_config[col] is already a mean (from your CSV).
            # The following `.describe()` is therefore describing a column of means,
            # so it's effectively "the summary stats of the mean values."
            print("(NOTE: These values are themselves means. The stats below are 'mean of means', etc.)")
            print(df_config[col].describe())

    ###########################################################################
    # 3.4 VISIT 0 vs. VISIT 1 (PAIRED) FOR OVERALL ERRORS
    ###########################################################################
    print("\n========== PAIRED COMPARISON (VISIT 0 vs VISIT 1) ==========")
    metric = 'Overall Mean Centroid Distance (mm)'

    # We group by PatientID and VisitID, then take the mean of the metric again.
    # But recall that 'metric' is already an average from your CSV file.
    # So grouped[metric].mean() is effectively 'mean of the mean for that patient/visit'.
    # We clarify that below:
    if metric in df_config.columns:
        grouped = df_config.groupby(['PatientID', 'VisitID'])[metric].mean().unstack(level='VisitID')
        grouped = grouped.dropna(how='any')
        if 0 in grouped.columns and 1 in grouped.columns and len(grouped) > 1:
            print("(NOTE: The metric is already an average; here we're taking another mean per patient+visit, "
                  "so this is effectively a 'mean of means'.)")
            t_stat, p_val = stats.ttest_rel(grouped[0], grouped[1])
            print(f"Paired t-test for {metric} between visits:")
            print(f"    t-stat = {t_stat:.4f}, p-value = {p_val:.6f}")
            if p_val < 0.05:
                print("    => Significant difference (p < 0.05)\n")
            else:
                print("    => No significant difference (p >= 0.05)\n")
            print("Mean Visit0 =", grouped[0].mean(), "mm,", "Mean Visit1 =", grouped[1].mean(), "mm")
        else:
            print("Could not perform paired test (missing columns or insufficient data).")
    else:
        print(f"Column '{metric}' not in df_config. Skipping visit comparison.")

    ###########################################################################
    # 3.5 COMPARISON OF KEY PARAMETERS
    ###########################################################################
    print("\n========== COMPARISON OF RECONSTRUCTION PARAMETERS ==========")
    param_col = 'Use Arch Constraints'
    if param_col in df_config.columns and metric in df_config.columns:
        true_vals = df_config[df_config[param_col] == True][metric].dropna()
        false_vals = df_config[df_config[param_col] == False][metric].dropna()
        if len(true_vals) > 1 and len(false_vals) > 1:
            print("(NOTE: Each value here is already a mean from the CSV config file.)")
            print(f"Parameter: {param_col}")
            print(f"  Mean error (True):  {true_vals.mean():.3f} mm  (n={len(true_vals)})")
            print(f"  Mean error (False): {false_vals.mean():.3f} mm  (n={len(false_vals)})")
            t_stat, p_val = stats.ttest_ind(true_vals, false_vals, equal_var=False)
            print(f"  Two-sample t-test => t={t_stat:.4f}, p={p_val:.6f}")
            if p_val < 0.05:
                print("  => Significant difference.\n")
            else:
                print("  => No significant difference.\n")
        else:
            print(f"Not enough data to compare {param_col} True vs. False.")

    if 'Use Arch Constraints' in df_config.columns and 'Overall Mean Centroid Distance (mm)' in df_config.columns:
        arch_groups = df_config.groupby('Use Arch Constraints')['Overall Mean Centroid Distance (mm)']
        if True in arch_groups.groups and False in arch_groups.groups:
            arch_true = arch_groups.get_group(True).dropna()
            arch_false = arch_groups.get_group(False).dropna()
            print("\nUse Arch Constraints (True vs. False):")
            print(f"  True: mean={arch_true.mean():.3f} mm, n={len(arch_true)}")
            print(f"  False: mean={arch_false.mean():.3f} mm, n={len(arch_false)}")
            t_stat, p_val = stats.ttest_ind(arch_true, arch_false, equal_var=False)
            print(f"  t={t_stat:.4f}, p={p_val:.6g}")
            if p_val < 0.05:
                print("  => Significant difference\n")
            else:
                print("  => No significant difference\n")

    if 'Use Alpha' in df_config.columns and 'Alpha Value' in df_config.columns and metric in df_config.columns:
        df_alpha = df_config[df_config['Use Alpha'] == True].copy()
        df_alpha = df_alpha.dropna(subset=['Alpha Value', metric])
        if len(df_alpha) > 2:
            corr, p_val = stats.pearsonr(df_alpha['Alpha Value'], df_alpha[metric])
            print(f"Correlation between Alpha Value and {metric} (Use Alpha=True):")
            print(f"    r = {corr:.4f}, p = {p_val:.6f}")
            if p_val < 0.05:
                print("    => Significant correlation.\n")
            else:
                print("    => No significant correlation.\n")
        else:
            print("Not enough rows (Use Alpha=True) to correlate alpha value.")

    ###########################################################################
    # 3.6 JAW-LEVEL ERROR COMPARISON
    ###########################################################################
    print("\n========== JAW-LEVEL COMPARISON (UPPER vs LOWER) ==========")
    upper_col = 'Upper Mean Centroid Distance (mm)'
    lower_col = 'Lower Mean Centroid Distance (mm)'
    if upper_col in df_config.columns and lower_col in df_config.columns:
        # Again, these columns are already 'means'. 
        # We're grouping and then calling .mean() on them, so it's a 'mean of means'.
        grouped_jaw = df_config.groupby(['PatientID', 'VisitID'])[[upper_col, lower_col]].mean().dropna()
        if len(grouped_jaw) > 1:
            print("(NOTE: Values are already mean distances; grouping adds another averaging step.)")
            t_stat, p_val = stats.ttest_rel(grouped_jaw[upper_col], grouped_jaw[lower_col])
            print(f"Paired t-test for {upper_col} vs. {lower_col}:")
            print(f"    t = {t_stat:.4f}, p-value = {p_val:.6f}")
            if p_val < 0.05:
                print("    => Significant difference.\n")
            else:
                print("    => No significant difference.\n")
            print("Mean Upper =", grouped_jaw[upper_col].mean(), "mm,",
                  "Mean Lower =", grouped_jaw[lower_col].mean(), "mm")
        else:
            print("Insufficient data for jaw-level comparison.")
    else:
        print("Missing columns for jaw-level comparison. Skipping.")

    ###########################################################################
    # 3.7 INDIVIDUAL TOOTH-LEVEL ANALYSIS
    ###########################################################################
    print("\n========== INDIVIDUAL (TOOTH-LEVEL) ANALYSIS ==========")
    if all(x in df_individual.columns for x in ['Jaw', 'Tooth Number', 'Point Type', 'Distance (mm)']):
        # Here, each row in df_individual is presumably either a single tooth or single point measurement,
        # so these are not pre-averaged. This is more "raw" than the config-level data.
        
        group_cols = ['Jaw', 'Tooth Number', 'Point Type']
        grouped = df_individual.groupby(group_cols)['Distance (mm)']
        descriptive_stats = grouped.describe()
        print("\n=== Individual Tooth-Level Descriptive Statistics ===\n")
        print(descriptive_stats.head(25))

        # Compare point types (centroid vs apex vs base)
        point_type_stats = df_individual.groupby('Point Type')['Distance (mm)'].describe()
        print("\n=== Error by Point Type ===\n")
        print(point_type_stats)

        # Compare jaws (upper vs lower)
        jaw_stats = df_individual.groupby('Jaw')['Distance (mm)'].describe()
        print("\n=== Error by Jaw ===\n")
        print(jaw_stats)

        # Statistical test between point types
        point_types = df_individual['Point Type'].unique()
        if len(point_types) > 1:
            print("\n=== Statistical Comparison Between Point Types ===\n")
            for i, pt1 in enumerate(point_types):
                for pt2 in point_types[i+1:]:
                    group1 = df_individual[df_individual['Point Type'] == pt1]['Distance (mm)'].dropna()
                    group2 = df_individual[df_individual['Point Type'] == pt2]['Distance (mm)'].dropna()
                    if len(group1) > 1 and len(group2) > 1:
                        t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
                        print(f"{pt1} vs {pt2}: t={t_stat:.4f}, p={p_val:.6f}")
                        print(f"  Mean {pt1}: {group1.mean():.3f} mm (n={len(group1)})")
                        print(f"  Mean {pt2}: {group2.mean():.3f} mm (n={len(group2)})")
                        if p_val < 0.05:
                            print("  => Significant difference.\n")
                        else:
                            print("  => No significant difference.\n")

        # Analysis by tooth number
        tooth_stats = df_individual.groupby('Tooth Number')['Distance (mm)'].describe()
        print("\n=== Error by Tooth Number ===\n")
        print(tooth_stats.head(25))

        # Find teeth with highest and lowest errors
        tooth_means = df_individual.groupby('Tooth Number')['Distance (mm)'].mean()
        print("\n=== Teeth with Highest and Lowest Errors ===\n")
        print("Top 5 teeth with highest errors:")
        print(tooth_means.nlargest(5))
        print("\nTop 5 teeth with lowest errors:")
        print(tooth_means.nsmallest(5))
    else:
        print("Missing columns in df_individual. Skipping tooth-level analysis.")

    ###########################################################################
    # 3.8 PLOTTING AND SAVING FIGURES
    ###########################################################################
    print("\n========== GENERATING PLOTS AND SAVING TO DISK ==========")
    sns.set(style='whitegrid')

    # Figure 6.1: Histogram of Overall Mean Centroid Distance
    if metric in df_config.columns:
        plt.figure(figsize=(6, 4))
        plt.hist(df_config[metric].dropna(), bins=30, color='skyblue', edgecolor='black')
        plt.xlabel("Overall Mean Centroid Distance (mm)")
        plt.ylabel("Frequency")
        plt.title("Histogram of Overall Mean Centroid Distance\n('Mean of Means')")
        plt.savefig("6.1_Histogram_Overall_Mean_Centroid_Distance.png", dpi=150)
        plt.close()
        print("Saved: 6.1_Histogram_Overall_Mean_Centroid_Distance.png")

    # Figure 6.2: Boxplot of Overall Mean Centroid Distance vs. Overall Mean All Points Distance
    if all(col in df_config.columns for col in [
        'Overall Mean Centroid Distance (mm)',
        'Overall Mean All Points Distance (mm)'
    ]):
        df_melt = df_config[[
            'Overall Mean Centroid Distance (mm)',
            'Overall Mean All Points Distance (mm)'
        ]].melt(var_name='Error Metric', value_name='Error (mm)')
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df_melt, x='Error Metric', y='Error (mm)')
        plt.title("Overall Centroid vs. All Points Distance\n('Mean of Means')")
        plt.savefig("6.2_Boxplot_Overall_vs_All_Points.png", dpi=150)
        plt.close()
        print("Saved: 6.2_Boxplot_Overall_vs_All_Points.png")

    # Figure 6.3: Boxplot comparing Overall Mean Centroid Distance for Visit 0 vs. Visit 1
    if 'VisitID' in df_config.columns and metric in df_config.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df_config, x='VisitID', y=metric)
        plt.title("Overall Mean Centroid Distance by Visit\n('Mean of Means')")
        plt.savefig("6.3_Boxplot_Overall_Centroid_by_Visit.png", dpi=150)
        plt.close()
        print("Saved: 6.3_Boxplot_Overall_Centroid_by_Visit.png")

    # Figure 6.4: Boxplot comparing Overall Mean Centroid Distance with and without Arch Constraints
    if 'Use Arch Constraints' in df_config.columns and metric in df_config.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df_config, x='Use Arch Constraints', y=metric)
        plt.title("Overall Mean Centroid Distance by Arch Constraints\n('Mean of Means')")
        plt.savefig("6.4_Boxplot_Overall_Centroid_by_Arch_Constraints.png", dpi=150)
        plt.close()
        print("Saved: 6.4_Boxplot_Overall_Centroid_by_Arch_Constraints.png")

    # Figure 6.5: Scatter Plot of Alpha Value vs. Overall Mean Centroid Distance
    if 'Alpha Value' in df_config.columns and metric in df_config.columns:
        alpha_df = df_config.dropna(subset=['Alpha Value', metric])
        plt.figure(figsize=(6, 4))
        sns.scatterplot(data=alpha_df, x='Alpha Value', y=metric, hue='Use Alpha')
        plt.title("Alpha Value vs. Overall Mean Centroid Distance\n('Mean of Means')")
        plt.savefig("6.5_Scatter_Alpha_vs_Overall_Centroid.png", dpi=150)
        plt.close()
        print("Saved: 6.5_Scatter_Alpha_vs_Overall_Centroid.png")

        # Add regression line to better visualize the relationship
        plt.figure(figsize=(6, 4))
        sns.regplot(data=alpha_df, x='Alpha Value', y=metric, scatter_kws={'alpha':0.5})
        plt.title("Alpha Value vs. Overall Mean Centroid Distance (with Regression)\n('Mean of Means')")
        plt.savefig("6.5b_Regplot_Alpha_vs_Overall_Centroid.png", dpi=150)
        plt.close()
        print("Saved: 6.5b_Regplot_Alpha_vs_Overall_Centroid.png")

    # Figure 6.6: Boxplot comparing Upper vs. Lower Mean Centroid Distance
    if all(col in df_config.columns for col in ['Upper Mean Centroid Distance (mm)',
                                                'Lower Mean Centroid Distance (mm)']):
        df_melt_jaw = df_config[[
            'Upper Mean Centroid Distance (mm)',
            'Lower Mean Centroid Distance (mm)'
        ]].melt(var_name='Jaw', value_name='Centroid Distance (mm)')
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df_melt_jaw, x='Jaw', y='Centroid Distance (mm)')
        plt.title("Upper vs. Lower Mean Centroid Distance\n('Mean of Means')")
        plt.savefig("6.6_Boxplot_Upper_vs_Lower_Centroid.png", dpi=150)
        plt.close()
        print("Saved: 6.6_Boxplot_Upper_vs_Lower_Centroid.png")

        # Also create a direct comparison using individual data
        if 'Jaw' in df_individual.columns and 'Distance (mm)' in df_individual.columns:
            plt.figure(figsize=(6, 4))
            sns.boxplot(data=df_individual, x='Jaw', y='Distance (mm)')
            plt.title("Upper vs. Lower Jaw Error (Individual Points)\n(Raw Distances)")
            plt.savefig("6.6b_Boxplot_Upper_vs_Lower_Individual.png", dpi=150)
            plt.close()
            print("Saved: 6.6b_Boxplot_Upper_vs_Lower_Individual.png")

    # Figure 6.7: Heatmap of mean reconstruction error per tooth and landmark type
    if all(x in df_individual.columns for x in ['Tooth Number', 'Point Type', 'Distance (mm)']):
        # Create separate heatmaps for upper and lower jaw
        for jaw in df_individual['Jaw'].unique():
            jaw_data = df_individual[df_individual['Jaw'] == jaw]
            heatmap_data = jaw_data.groupby(['Point Type', 'Tooth Number'])['Distance (mm)'].mean().unstack()
            plt.figure(figsize=(12, 8))
            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis")
            plt.title(f"Mean Reconstruction Error per Tooth and Landmark Type - {jaw} Jaw")
            plt.xlabel("Tooth Number")
            plt.ylabel("Landmark Type")
            plt.savefig(f"6.7_{jaw}_Heatmap_Error_per_Tooth_Landmark.png", dpi=150)
            plt.close()
            print(f"Saved: 6.7_{jaw}_Heatmap_Error_per_Tooth_Landmark.png")
            
        # Combined heatmap
        heatmap_data = df_individual.groupby(['Point Type', 'Tooth Number'])['Distance (mm)'].mean().unstack()
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis")
        plt.title("Mean Reconstruction Error per Tooth and Landmark Type - All Jaws")
        plt.xlabel("Tooth Number")
        plt.ylabel("Landmark Type")
        plt.savefig("6.7_Heatmap_Error_per_Tooth_Landmark.png", dpi=150)
        plt.close()
        print("Saved: 6.7_Heatmap_Error_per_Tooth_Landmark.png")

    # Figure 6.8: Box plots comparing error distributions by point type
    if 'Point Type' in df_individual.columns and 'Distance (mm)' in df_individual.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_individual, x='Point Type', y='Distance (mm)')
        plt.title("Reconstruction Error by Landmark Type\n(Raw Distances)")
        plt.xlabel("Landmark Type")
        plt.ylabel("Error (mm)")
        plt.savefig("6.8_Boxplot_Error_by_Point_Type.png", dpi=150)
        plt.close()
        print("Saved: 6.8_Boxplot_Error_by_Point_Type.png")

    # Figure 6.9: Box plots comparing error distributions by jaw and point type
    if all(x in df_individual.columns for x in ['Jaw', 'Point Type', 'Distance (mm)']):
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df_individual, x='Point Type', y='Distance (mm)', hue='Jaw')
        plt.title("Reconstruction Error by Landmark Type and Jaw\n(Raw Distances)")
        plt.xlabel("Landmark Type")
        plt.ylabel("Error (mm)")
        plt.savefig("6.9_Boxplot_Error_by_Point_Type_and_Jaw.png", dpi=150)
        plt.close()
        print("Saved: 6.9_Boxplot_Error_by_Point_Type_and_Jaw.png")

    # Figure 6.10: Box plots comparing error distributions by tooth number
    if 'Tooth Number' in df_individual.columns and 'Distance (mm)' in df_individual.columns:
        # Limit to top 10 teeth by frequency to avoid overcrowding
        tooth_counts = df_individual['Tooth Number'].value_counts().head(10).index
        tooth_data = df_individual[df_individual['Tooth Number'].isin(tooth_counts)]

        plt.figure(figsize=(14, 6))
        sns.boxplot(data=tooth_data, x='Tooth Number', y='Distance (mm)')
        plt.title("Reconstruction Error by Tooth Number (Top 10 Most Frequent)\n(Raw Distances)")
        plt.xlabel("Tooth Number")
        plt.ylabel("Error (mm)")
        plt.xticks(rotation=45)
        plt.savefig("6.10_Boxplot_Error_by_Tooth_Number.png", dpi=150)
        plt.close()
        print("Saved: 6.10_Boxplot_Error_by_Tooth_Number.png")

    # Figure 6.11: Violin plots for a more detailed view of error distributions
    if 'Point Type' in df_individual.columns and 'Distance (mm)' in df_individual.columns:
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=df_individual, x='Point Type', y='Distance (mm)', inner='quartile')
        plt.title("Distribution of Reconstruction Error by Landmark Type\n(Raw Distances)")
        plt.xlabel("Landmark Type")
        plt.ylabel("Error (mm)")
        plt.savefig("6.11_Violin_Error_by_Point_Type.png", dpi=150)
        plt.close()
        print("Saved: 6.11_Violin_Error_by_Point_Type.png")

    # Figure 6.12: Cumulative distribution function of errors
    if 'Distance (mm)' in df_individual.columns:
        plt.figure(figsize=(8, 6))
        for pt in df_individual['Point Type'].unique():
            subset = df_individual[df_individual['Point Type'] == pt]['Distance (mm)'].dropna()
            x = np.sort(subset)
            y = np.arange(1, len(x)+1) / len(x)
            plt.plot(x, y, label=f"{pt} (n={len(x)})")

        plt.title("Cumulative Distribution of Reconstruction Errors by Point Type\n(Raw Distances)")
        plt.xlabel("Error (mm)")
        plt.ylabel("Cumulative Probability")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig("6.12_CDF_Error_by_Point_Type.png", dpi=150)
        plt.close()
        print("Saved: 6.12_CDF_Error_by_Point_Type.png")

    print("\n=== ALL ANALYSES AND PLOTTING COMPLETED SUCCESSFULLY ===")


###############################################################################
# SCRIPT ENTRY POINT
###############################################################################
if __name__ == "__main__":
    main()
