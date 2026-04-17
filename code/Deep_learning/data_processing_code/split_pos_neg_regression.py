import pandas as pd
import numpy as np

# Read data
file_path = 'stat.ref.denoise.10.regression.txt'
data = pd.read_csv(file_path, sep='\t')

# Sort by Diff_R4_R2 column
data_sorted = data.sort_values(by='Diff_R4_R2', ascending=False)

# Select rows with positive Diff_R4_R2 as positive samples
positive_samples = data_sorted[data_sorted['Diff_R4_R2'] > 0]

# Select rows with negative Diff_R4_R2 as negative samples
negative_samples = data_sorted[data_sorted['Diff_R4_R2'] < 0]

# Calculate the number of positive samples
num_positive_samples = len(positive_samples)

# Generate evenly spaced indices using numpy.linspace
indices = np.linspace(0, len(negative_samples) - 1, num=num_positive_samples, dtype=int)
selected_negative_samples = negative_samples.iloc[indices]

# Select required columns and rename
positive_samples = positive_samples[['Sequence', 'Diff_R4_R2']].rename(columns={'Sequence': 'sequence', 'Diff_R4_R2': 'value'})
selected_negative_samples = selected_negative_samples[['Sequence', 'Diff_R4_R2']].rename(columns={'Sequence': 'sequence', 'Diff_R4_R2': 'value'})

# Output the number of positive samples and selected negative samples
print(f'Number of positive samples: {num_positive_samples}')
print(f'Selected negative samples: {selected_negative_samples["sequence"].tolist()}')

# Output to files
positive_samples.to_csv('positive_samples.txt', index=False, sep='\t')
selected_negative_samples.to_csv('negative_samples.txt', index=False, sep='\t')
