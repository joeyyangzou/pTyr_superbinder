def linspace(start, stop, num):
    if num == 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [int(round(start + i * step)) for i in range(num)]

file_path = 'regression_input.txt'
output_path = 'regression_dataset.txt'

positive = []
negative = []

with open(file_path, 'r') as f:
    next(f)
    for line in f:
        parts = line.strip().split('\t')
        seq = parts[0]
        val = float(parts[1])
        if val > 0:
            positive.append((seq, val))
        elif val < 0:
            negative.append((seq, val))

negative_sorted = sorted(negative, key=lambda x: -x[1])

num_positive = len(positive)

if len(negative_sorted) > 0:
    indices = linspace(0, len(negative_sorted) - 1, num_positive)
    selected_negative = [negative_sorted[i] for i in indices]
else:
    selected_negative = []

combined = positive + selected_negative

print(f'正样本数: {len(positive)}')
print(f'负样本数: {len(selected_negative)}')
print(f'总样本数: {len(combined)}')

with open(output_path, 'w') as f:
    f.write('sequence\tvalue\n')
    for seq, val in combined:
        f.write(f'{seq}\t{val:.6f}\n')
