# 读取文件内容
file_path = 'merged_sequences_all.txt'
output_path = 'stat.ref.denoise.classification1.txt'

with open(file_path, 'r') as file:
    lines = file.readlines()

header = lines[0].split()
col_names = {name: idx for idx, name in enumerate(header)}

PYS2_CN_idx = col_names['PYS2_CN']
PYS4_CN_idx = col_names['PYS4_CN']

sum_R2 = 0
sum_R4 = 0

for line in lines[1:]:
    parts = line.split()
    if len(parts) >= max(PYS2_CN_idx, PYS4_CN_idx) + 1:
        R2 = int(parts[PYS2_CN_idx])
        R4 = int(parts[PYS4_CN_idx])
        sum_R2 += R2
        sum_R4 += R4

results = []
for line in lines[1:]:
    parts = line.split()
    if len(parts) >= max(PYS2_CN_idx, PYS4_CN_idx) + 1:
        sequence = parts[0]
        R2 = int(parts[PYS2_CN_idx])
        R4 = int(parts[PYS4_CN_idx])
        ratio_R2 = R2 / sum_R2
        ratio_R4 = R4 / sum_R4
        if ratio_R2 != 0 or ratio_R4 != 0:
            diff = ratio_R4 - ratio_R2
            flag = 1 if diff > 0 else 0
            results.append([sequence, f"{ratio_R2:.6f}", f"{ratio_R4:.6f}", f"{diff:.6f}", flag])

with open(output_path, 'w') as file:
    file.write("Sequence\tRatio_R2\tRatio_R4\tDiff_R4_R2\tFlag\n")
    for result in results:
        file.write('\t'.join(map(str, result)) + '\n')
