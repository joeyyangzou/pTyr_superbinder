def filter_sequences(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            sequence = parts[0]
            values = list(map(int, parts[1:5]))  # Convert R0 to R4 values to integers
            
            # Check if sequence contains '_', 'q', or 'w'
            if any(char in sequence for char in ['_', 'q', 'w']):
                continue
            
            # Check if any of R0 to R4 is zero
            if any(value == 0 for value in values):
                # Allow zero if there is at least one value greater than 3
                if not any(value > 3 for value in values):
                    continue
            
            print(line.strip())

# Example usage
filter_sequences('stat.ref.txt')
