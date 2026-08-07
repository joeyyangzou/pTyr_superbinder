import math

# 可调节的阈值参数
# 用于计算 r2_val 和 r4_val 时的偏移量
# 可以修改为不同的值，如 1, 5, 10, 20 等
offset = 10  # 可以修改这个值

file_path = 'merged_sequences_all.txt'
output_path = 'regression_input.txt'

with open(file_path, 'r') as file:
    lines = file.readlines()

header = lines[0].split()
col_names = {name: idx for idx, name in enumerate(header)}

PYS2_CN_idx = col_names['PYS2_CN']
PYS4_CN_idx = col_names['PYS4_CN']

R2_number = 0
R4_number = 0
R2_dict = {}
R4_dict = {}

for line in lines[1:]:
    parts = line.split()
    if len(parts) >= max(PYS2_CN_idx, PYS4_CN_idx) + 1:
        pys2_val = int(parts[PYS2_CN_idx])
        pys4_val = int(parts[PYS4_CN_idx])
        if pys2_val == 0 and pys4_val == 0:
            continue
        seq = parts[0]
        r2_val = float(pys2_val) + offset
        r4_val = float(pys4_val) + offset
        R2_dict[seq] = r2_val
        R4_dict[seq] = r4_val
        R2_number += pys2_val
        R4_number += pys4_val

with open(output_path, 'w') as out_regression:
    out_regression.write('sequence\tvalue\n')
    for key in R2_dict:
        if key not in R4_dict:
            R4_dict[key] = offset
        if key not in R2_dict:
            R2_dict[key] = offset

        R4_frequency = R4_dict[key] / R4_number
        R2_frequency = R2_dict[key] / R2_number
        R4_R2_ratio = R4_frequency / R2_frequency
        label = math.log10(R4_R2_ratio)
        out_regression.write(f"{key}\t{label:.6f}\n")

