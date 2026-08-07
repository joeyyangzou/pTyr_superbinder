import pandas as pd
import math
from sklearn.model_selection import train_test_split
import sys
sequennce_infilepath = "regression_dataset.txt"#待分割的数据集
output_train = open("train_set.txt","w")#训练集的输出路径
output_test = open("test_set.txt","w")#独立测试集的输出路径


i=0


def process(sequennce_infilepath):
    line_number = 0
    sequencefile = pd.read_csv(sequennce_infilepath, sep='\t')#, header=0
    values=sequencefile['value']
    sequence=sequencefile['sequence']
    sequence_list=[]
    label_list=[]
    for seq in sequence:
        sequence_list.append(seq)
        line_number += 1
    for value in values:
        label_list.append(value)
    
    # 智能计算测试集大小（约20%）
    test_number = max(1, int(line_number * 0.2))
    print(f"自动计算测试集大小: {test_number}")
    
    split_span = int(line_number / test_number)
    print(split_span)
    ratio_remainder0 =1/split_span

    if (line_number - split_span * test_number)>=split_span:
        print('划分数据集异常')
        sys.exit(1)
    train_lnum = 0
    test_lnum = 0
    for i in range(0,line_number,split_span):
        if line_number % test_number ==0:
            x=sequence_list[i:i+split_span]
            y = label_list[i:i+split_span]
            x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=ratio_remainder0,random_state=1)#,shuffle=false)
            for (x_test_line,y_test_line) in zip(x_test,y_test):
                print(x_test_line,y_test_line)
                output_test.write(str(x_test_line) + '\t' + str(y_test_line) + '\n')
            for (x_train_line,y_train_line) in zip(x_train,y_train):
                output_train.write(str(x_train_line) + '\t' + str(y_train_line) + '\n')
        else:
            if i >= split_span*test_number and (line_number - split_span * test_number)<split_span:
                no_ratio_remainder0 = 1 / (line_number - i)
                if line_number - split_span * test_number > 1:
                    x = sequence_list[i:]
                    y = label_list[i:]
                    x_train,  x_test,y_train, y_test = train_test_split(x, y, test_size=no_ratio_remainder0,random_state=0)
                    for (x_test_line, y_test_line) in zip(x_test, y_test):
                        output_test.write(str(x_test_line) + '\t' + str(y_test_line) + '\n')
                        test_lnum+=1
                    for (x_train_line, y_train_line) in zip(x_train, y_train):
                        output_train.write(str(x_train_line) + '\t' + str(y_train_line) + '\n')
                        train_lnum+=1
                    print('!!!!!!!!!!!!!!!!!!!!!!抽样后的序列数量' + '(' + str(test_number) + '):' + '\t' + str(line_number - split_span * test_number))
                    print('!!!!!!!!!!!!!!!!!!!!!!剩余数量大于1')
                    print('!!!!!!!!!!!!!!!!!!!!!!test:' + str(test_lnum))
                    print('!!!!!!!!!!!!!!!!!!!!!!train:' + str(train_lnum))
                else:
                    x_test = sequence_list[i:]
                    y_test = label_list[i:]
                    for (x_test_line, y_test_line) in zip(x_test, y_test):
                        output_test.write(str(x_test_line) + '\t' + str(y_test_line) + '\n')
                        test_lnum += 1

                    print('!!!!!!!!!!!!!!!!!!!!!!剩余数量等于1')
                    print('!!!!!!!!!!!!!!!!!!!!!!test:' + str(test_lnum))
                    print('!!!!!!!!!!!!!!!!!!!!!!train:' + str(train_lnum))
            else:
                x = sequence_list[i:i + split_span]
                y = label_list[i:i + split_span]
                x_train,  x_test,y_train, y_test = train_test_split(x, y, test_size=ratio_remainder0,random_state=1)  # ,shuffle=false)
                for (x_test_line, y_test_line) in zip(x_test, y_test):
                    output_test.write(str(x_test_line) + '\t' + str(y_test_line) + '\n')
                    test_lnum+=1
                for (x_train_line, y_train_line) in zip(x_train, y_train):
                    output_train.write(str(x_train_line) + '\t' + str(y_train_line) + '\n')
                    train_lnum+=1
process(sequennce_infilepath)