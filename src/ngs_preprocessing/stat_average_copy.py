def count_sequences_and_copies(file_path):
    sequence_counts = {i: 0 for i in range(5)}
    copy_sums = {i: 0 for i in range(5)}
    
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            sequence = parts[0]
            values = list(map(int, parts[1:6]))  # Convert R0 to R4 values to integers
            
            # Check if sequence contains '_', 'q', or 'w'
            if any(char in sequence for char in ['_', 'q', 'w']):
                continue
            
            # Check if any of R0 to R4 is zero
            if any(value == 0 for value in values):
                # Allow zero if there is at least one value greater than 3
                if not any(value > 3 for value in values):
                    continue
            
            # Count sequences and sum copies for each R0 to R4
            for i in range(5):
                if values[i] > 0:
                    sequence_counts[i] += 1
                    copy_sums[i] += values[i]
    
    # Print results
    for i in range(5):
        print(f"R{i} - Sequence Types: {sequence_counts[i]}, Total Copies: {copy_sums[i]}")

# Supply the processed count table from the current working directory.
#count_sequences_and_copies('stat.ref.txt')
count_sequences_and_copies('stat.ref.denoise.txt')
