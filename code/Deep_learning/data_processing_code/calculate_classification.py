# Read file content
file_path = 'stat.ref.denoise.txt'
output_path = 'stat.ref.denoise.classification1.txt'

with open(file_path, 'r') as file:
    lines = file.readlines()

# Initialize sum of R2 and R4
sum_R2 = 0
sum_R4 = 0

# Extract R2 and R4 values from each line and accumulate
for line in lines:
    parts = line.split()
    if len(parts) >= 7:  # Ensure the line has enough data
        R2 = int(parts[3])
        R4 = int(parts[5])
        sum_R2 += R2
        sum_R4 += R4

# Calculate ratios and differences, generate final table
results = []
for line in lines:
    parts = line.split()
    if len(parts) >= 7:
        sequence = parts[0]
        R2 = int(parts[3])
        R4 = int(parts[5])
        ratio_R2 = R2 / sum_R2
        ratio_R4 = R4 / sum_R4
        if ratio_R2 != 0 or ratio_R4 != 0:  # Exclude cases where both R2 and R4 ratios are 0
            diff = ratio_R4 - ratio_R2
            flag = 1 if diff > 0 else 0
            results.append([sequence, f"{ratio_R2:.6f}", f"{ratio_R4:.6f}", f"{diff:.6f}", flag])

# Write results to file
with open(output_path, 'w') as file:
    # Write header
    file.write("Sequence\tRatio_R2\tRatio_R4\tDiff_R4_R2\tFlag\n")
    # Write data
    for result in results:
        file.write('\t'.join(map(str, result)) + '\n')
