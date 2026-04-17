# Read file content
file_path = 'stat.ref.denoise.10.txt'
output_path = 'stat.ref.denoise.10.regression.txt'

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
        #if R2 == 0 and R4 == 0:  # Skip this line if both R2 and R4 are 0
        #    continue
        #if R2 == 0 or R4 == 0:  # If R2 or R4 is 0
        #    R2 += 1
        #    R4 += 1
        R2_adjusted = R2 + 1 if R2 == 0 else R2  # Only add 1 when R2 is 0
        R4_adjusted = R4 + 1 if R4 == 0 else R4  # Only add 1 when R4 is 0
        ratio_R2 = R2_adjusted / sum_R2
        ratio_R4 = R4_adjusted / sum_R4
        if ratio_R2 != 0 or ratio_R4 != 0:  # Exclude cases where both R2 and R4 ratios are 0
            import math
            diff = math.log10(ratio_R4 / ratio_R2)   # Use log10 to calculate ratio, avoid division by zero
            flag = 1 if diff > 0 else 0
            results.append([sequence, R2, R4, f"{ratio_R2:.6f}", f"{ratio_R4:.6f}", f"{diff:.6f}", flag])
# Write results to file
with open(output_path, 'w') as file:
    # Write header
    file.write("Sequence\tR2\tR4\tRatio_R2\tRatio_R4\tRatio_R4_R2\tFlag\n")
    # Write data
    for result in results:
        file.write('\t'.join(map(str, result)) + '\n')
