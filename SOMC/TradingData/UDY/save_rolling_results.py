import pandas as pd

def save_rolling_results_to_excel(rolling_results, output_filename='rolling_simulation_results.xlsx'):
    """
    Combines rolling results from a dictionary of pandas Series into a single
    DataFrame and saves it to an Excel file.

    Args:
        rolling_results (dict): A dictionary where keys are volatility labels and
                                values are dictionaries containing pandas Series
                                ('mean', 'p99_20', 'p99_250').
                                It's assumed the 'p99_20' and 'p99_250' series
                                have already been renamed.
        output_filename (str): The name of the output Excel file.
    """
    all_series_to_concat = []

    for vol_label, data_dict in rolling_results.items():
        # Rename the 'mean' series to avoid column name collisions
        mean_series = data_dict['mean'].rename(f'mean_{vol_label}')
        
        all_series_to_concat.append(mean_series)
        all_series_to_concat.append(data_dict['p99_20'])
        all_series_to_concat.append(data_dict['p99_250'])

    # Concatenate all series into a single DataFrame.
    # pd.concat aligns the data based on the index (dates).
    final_results_df = pd.concat(all_series_to_concat, axis=1)

    # Save the combined DataFrame to an Excel file
    try:
        final_results_df.to_excel(output_filename)
        print(f"Successfully saved all rolling results to '{output_filename}'")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")

# This is a placeholder for the user's 'rolling_results' dictionary.
# In the actual notebook, this variable would already be populated.
# rolling_results = { ... } 

# Example usage:
# save_rolling_results_to_excel(rolling_results)