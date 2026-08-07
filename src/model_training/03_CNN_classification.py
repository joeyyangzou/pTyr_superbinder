import numpy as np
import tensorflow as tf
import pandas as pd
import os
from tensorflow.keras import layers, Sequential, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import roc_curve, auc

# 配置GPU内存
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
config.gpu_options.allow_growth = True

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
model_save_path=os.path.join(current_dir, "models")#二分类模型保存路径
metrics_path=os.path.join(current_dir, "output", "confusion_matrix.csv")#混淆矩阵的路径
data_path=current_dir#训练集以及独立测试集的存储路径
result_path=os.path.join(current_dir, "output")#模型预测结果以及loss曲线图的存储路径

# 创建必要的目录
os.makedirs(os.path.join(current_dir, "models"), exist_ok=True)
os.makedirs(os.path.join(current_dir, "output"), exist_ok=True)

'''对原始序列进行分词操作，并生成能够被进行one-hot操作的列表 ↓ '''
def process(inputfile_path):
    data = pd.read_csv(inputfile_path, sep='\t', header=0)
    lable = data['label']
    inputfile = data['sequence']
    sequence_list = []
    line_number = 0
    for line in inputfile:
        line_number = line_number + 1
    return inputfile, line_number, lable
def One_Hot(sequence, line_number):
    AA = ['I', 'L', 'V', 'F', 'M', 'C', 'A', 'G', 'P', 'T', 'S', 'Y', 'W', 'Q', 'N', 'H', 'E', 'D', 'K', 'R']
    encodings = []
    seq_lengths = []
    
    # 确保sequence是列表格式
    if hasattr(sequence, 'iloc'):  # 如果是pandas的Series
        sequence_list = sequence.tolist()
    else:
        sequence_list = list(sequence)
    
    # 首先获取所有序列的长度，找出最常见的长度
    for seq_line in sequence_list:
        seq_line_list = list(str(seq_line))  # 确保转换为字符串再转列表
        seq_lengths.append(len(seq_line_list))
    
    # 找出最常见的序列长度作为标准长度
    if seq_lengths:
        from collections import Counter
        most_common = Counter(seq_lengths).most_common(1)
        if most_common:
            seq_length = most_common[0][0]
        else:
            seq_length = 8  # 默认长度，以防没有序列
        print(f"使用的标准序列长度: {seq_length}")
    else:
        seq_length = 8  # 默认长度，以防没有序列
        print(f"没有找到序列，使用默认长度: {seq_length}")
    
    # 处理每个序列，确保长度一致
    for seq_line in sequence_list:
        code = []
        # 确保seq_line是字符串并转换为列表
        seq_line_list = list(str(seq_line))
        
        # 如果序列长度不一致，进行截断或填充
        if len(seq_line_list) != seq_length:
            # 截断过长的序列
            if len(seq_line_list) > seq_length:
                seq_line_list = seq_line_list[:seq_length]
                print(f"警告: 序列 {seq_line[:10]}... 过长，已截断为 {seq_length} 个字符")
            # 填充过短的序列（用'X'作为未知氨基酸）
            else:
                padding = ['X'] * (seq_length - len(seq_line_list))
                seq_line_list.extend(padding)
                print(f"警告: 序列 {seq_line} 过短，已填充为 {seq_length} 个字符")
        
        # 进行one-hot编码
        for aa in seq_line_list:
            # 对每个氨基酸进行20种AA的编码
            for aa1 in AA:
                tag = 1 if aa == aa1 else 0
                code.append(tag)
        
        encodings.append(code)
    
    # 转换为numpy数组
    try:
        encodings = np.array(encodings)
        print(f"编码后的数组形状: {encodings.shape}")
        
        # 计算期望的形状
        expected_shape = (line_number, seq_length, 20)
        
        # 确保重塑操作安全
        try:
            encodings_reshaped = np.reshape(encodings, expected_shape)
            print(f"成功重塑为形状: {encodings_reshaped.shape}")
            return encodings_reshaped
        except ValueError as e:
            print(f"重塑错误: {e}")
            print(f"预期形状: {expected_shape}")
            print(f"实际数组大小: {encodings.size}")
            
            # 尝试重新计算正确的维度
            total_elements = encodings.size
            
            # 确保元素总数能被样本数和氨基酸类型数整除
            if total_elements % (line_number * 20) == 0:
                actual_seq_length = total_elements // (line_number * 20)
                print(f"根据元素总数计算的实际序列长度: {actual_seq_length}")
                encodings_reshaped = np.reshape(encodings, (line_number, actual_seq_length, 20))
                print(f"成功重塑为形状: {encodings_reshaped.shape}")
                return encodings_reshaped
            else:
                print("无法重塑数组: 元素总数不能被样本数和氨基酸类型数整除")
                # 尝试修复编码长度不一致的问题
                print("尝试修复编码长度不一致的问题...")
                
                # 找出最长的编码长度
                max_code_len = max(len(code) for code in encodings)
                
                # 确保所有编码长度一致
                fixed_encodings = []
                for code in encodings:
                    if len(code) < max_code_len:
                        # 填充短编码
                        code.extend([0] * (max_code_len - len(code)))
                    elif len(code) > max_code_len:
                        # 截断长编码
                        code = code[:max_code_len]
                    fixed_encodings.append(code)
                
                fixed_encodings = np.array(fixed_encodings)
                print(f"修复后的数组形状: {fixed_encodings.shape}")
                
                # 再次尝试重塑
                if max_code_len % 20 == 0:
                    actual_seq_length = max_code_len // 20
                    try:
                        result = np.reshape(fixed_encodings, (line_number, actual_seq_length, 20))
                        print(f"修复后成功重塑为形状: {result.shape}")
                        return result
                    except:
                        print("修复后仍然无法重塑")
                        return fixed_encodings
                else:
                    print("修复后编码长度仍然不能被20整除")
                    return fixed_encodings
    except Exception as e:
        print(f"处理编码时出错: {e}")
        # 返回原始编码列表作为后备
        return encodings
'''搭建网络结构，并且进行编译 ↓ '''


def calculate_metrics(labels, scores, cutoff=0.5, po_label=1):  # 计算阈值为0.5时的各性能指数
    my_metrics = {  # 先声明建立一个字典，对应KEY值
        'SN': 'NA',
        'SP': 'NA',
        'ACC': 'NA',
        'MCC': 'NA',
        'Recall': 'NA',
        'Precision': 'NA',
        'F1-score': 'NA',
        'Cutoff': cutoff,
    }

    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(scores)):
        if labels[i] == po_label:  # 如果为正样本
            if scores[i] >= cutoff:  # 阈值为0.5，如果打分大于0.5
                tp = tp + 1  # tp+1  预测为真，实际为真的
            else:
                fn = fn + 1  # 实际为真，预测为负
        else:  # 如果为负样本
            if scores[i] < cutoff:  # 打分小于阈值，说明实际为负，预测也为负
                tn = tn + 1  # tn+1
            else:
                fp = fp + 1  # 打分大于阈值，说明预测为正，实际为负

    my_metrics['SN'] = tp / (tp + fn) if (tp + fn) != 0 else 'NA'  # sn 灵敏度
    my_metrics['SP'] = tn / (fp + tn) if (fp + tn) != 0 else 'NA'  # sp 特异性
    my_metrics['ACC'] = (tp + tn) / (tp + fn + tn + fp)  # acc正确度
    my_metrics['MCC'] = (tp * tn - fp * fn) / np.math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if ( tp + fp) * ( tp + fn) * ( tn + fp) * ( tn + fn) != 0 else 'NA'
    my_metrics['Precision'] = tp / (tp + fp) if (tp + fp) != 0 else 'NA'  # 查准率
    my_metrics['Recall'] = my_metrics['SN']  # 召回率
    my_metrics['F1-score'] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 'NA'
    return my_metrics
def drow_roc(y_true, pred):
    # 移除matplotlib，只计算ROC数据并保存
    try:
        # 只导入必要的计算模块
        from sklearn.metrics import roc_curve, auc
        
        # 计算ROC曲线数据
        fpr, tpr, thresholds = roc_curve(y_true, pred)
        roc_auc = auc(fpr, tpr)
        
        # 保存ROC数据到文件
        with open('roc_data.txt', 'w') as f:
            f.write('# AUC: {:.4f}\n'.format(roc_auc))
            f.write('# FPR\tTPR\tThresholds\n')
            for i in range(len(fpr)):
                f.write('{:.6f}\t{:.6f}\t{:.6f}\n'.format(fpr[i], tpr[i], thresholds[i]))
        
        print(f"ROC数据已保存到roc_data.txt，AUC值: {roc_auc:.4f}")
        return roc_auc
        
    except Exception as e:
        print(f"计算ROC数据时出错: {e}")
        return None

def build_network():
    conv_layers = [
        layers.Conv1D(filters=128, kernel_size=1, padding='same', activation='relu'),
        layers.Dropout(0.5),
        layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'),
        layers.Dropout(0.5),
        layers.Conv1D(filters=128, kernel_size=9, padding='same', activation='relu'),
        layers.MaxPooling1D(2,1),
        layers.Dropout(0.5),
        layers.Conv1D(filters=128, kernel_size=10, padding='same', activation='relu'),
        layers.MaxPooling1D(pool_size=2, strides=1),
        layers.Dropout(0.7),
    ]

    fc_layers = [
        layers.Dense(64, activation='relu'),
        layers.MaxPooling1D(2,1),
        layers.Dense(32, activation='relu'),
        layers.MaxPooling1D(2,1),
        layers.Dense(8, activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(1, activation='sigmoid')
    ]

    conv_layers.extend(fc_layers)
    network = Sequential(conv_layers)
    network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return network


'''将数据类型进行转换 ↓ '''
def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.int32)
    return x, y

'''下面的evaluate自定义函数的作用是对训练完成的模型进行评估 ↓ '''
mean_fpr = np.linspace(0, 1, 100)
def evaluate(X, Y,X_vali, Y_vali, X_TEST, Y_TEST, batch_size=512, epochs=100,line_number_train=0,line_number_vali=0,line_number_test=0):
    classes = sorted([0,1])# #set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等。
    print(Y)
    X_train, y_train = X, Y
    '''将相应数据集转换为one-hot编码形式'''
    X_train = One_Hot(X_train,line_number_train) #训练集one-hot编码之后的得到的数据集
    X_vali = One_Hot(X_vali, line_number_vali) #验证集one-hot编码之后的得到的数据集
    X_test = One_Hot(X_TEST,line_number_test) #测试集one-hot编码之后的得到的数据集

    X_train_t = X_train
    X_vali_t = X_vali
    X_test_t = X_test
    X_test_t = tf.cast(X_test_t, dtype=tf.float32)
    '''下面六行命令之间两两类似，都是先将X与Y通过生成tf.data的形式绑定在一块，随后在下一行命令里面进行数据类型的变换，以及批量化处理 ↓ '''
    # 构建训练集对象，随机打乱，预处理，批量化
    train_db = tf.data.Dataset.from_tensor_slices((X_train_t, y_train)) #首先将pandas dataframe 数据格式转变为 tf.data 格式的数据集形式，这样是为了下一步数据进行打乱处理做准备，转换为tf.data格式之后，序列与label值则昂订到一块，我们就可以对其进行同步处理
    train_db = train_db.shuffle(len(X)).map(preprocess).batch(batch_size) #其中map函数是用来将序列做一键预处理用的。该行命令：先对数据进行随机化处理，随后使用map函数进行预处理，使用batch函数指定批次数量
    # 构建验证集对象，预处理，批量化
    vali_db = tf.data.Dataset.from_tensor_slices((X_vali, Y_vali))
    vali_db = vali_db.shuffle(len(X_vali_t)).map(preprocess).batch(batch_size)
    # 构建测试集对象，预处理，批量化
    test_db = tf.data.Dataset.from_tensor_slices((X_test_t, Y_TEST))
    test_db = test_db.shuffle(len(X_test_t)).map(preprocess).batch(batch_size)
    '''调用已经搭建好的神经网络，并进行训练，并将独立测试集的测试结果输出'''
    network = build_network()

    checkpoint = ModelCheckpoint( model_save_path,monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto', save_weights_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    callbacks_list = [checkpoint,early_stopping]
    history = network.fit(train_db,  validation_data=vali_db,epochs=epochs,verbose=1,callbacks = [callbacks_list])#,callbacks = [early_stopping]
    print("Independent test:", network.evaluate(test_db))
    tmp_result = np.zeros((len(Y_TEST), len(classes)))
    predict=network.predict(X_test_t, batch_size=batch_size)
    tmp_result[:, 0], tmp_result[:, 1] = Y_TEST, predict[:, 0]
    matrix=[]
    drow_roc(Y_TEST,predict)
    matrix.append(calculate_metrics(Y_TEST, predict))
    df = pd.DataFrame(matrix).to_csv(metrics_path)
    network.save(model_save_path, save_format='tf')
    return tmp_result, history, Y_TEST

'''下面的main自定义函数的作用主要为对以上的函数进行调用（主函数）'''
def main():
    os.chdir(data_path)
    epoch = 200
    '''0、将正样本与负样本中的信息先提取出来，包括sequence、label'''
    positive_sequence, positive_linenumber, positive_label = process('positive')
    negative_sequence, negative_linenumber, negative_label = process('negative')
    '''1、将正样本与负样本进行平衡,并将相应的数据合二为一'''
    # print('line_number:'+str(positive_sequence))
    # print('chang du:'+str(len(positive_sequence)))
    # positive_sequence_balance,positive_label_balance,p_sequence_balance,p_label_balance= random_dataset_number(positive_sequence, positive_label, sample_num=positive_linenumber) #sample_num是需要提取的正负样本的数量：6102
    # negative_sequence_balance,negative_label_balance,n_sequence_balance,n_label_balance= random_dataset_number(negative_sequence, negative_label, sample_num=positive_linenumber) #sample_num是需要提取的正负样本的数量：6102
    # sequence_balance = pd.concat([p_sequence_balance, n_sequence_balance])  # 将带有’sequence‘表头的正负样本中的sequence数据合二为一
    # label_balance = pd.concat([p_label_balance, n_label_balance])  # 将带有’label‘表头的正负样本中的sequence数据合二为一
    # balance_merge = pd.concat([sequence_balance, label_balance],axis=1,names=['sequence','label'])  # 将sequence数据与label数据合二为一
    # balance_merge.to_csv('balance.csv', encoding='gbk',sep='\t')  # 将测试集的数据保存到文件夹中去
    # balance_sequence, balance_linenumber, balance_label = process('balance.csv')
    # print('balance_sequence:'+str(balance_sequence))
    '''2、将独立测试集划分出来，划分之前首先判断文件夹中是否存在独立测试集'''
    if os.path.isfile('test_+10.csv'):
        print('测试集文件存在')
        independent_test_sequence, independent_test_linenumber, independent_test_label=process('test_+10.csv')
        train_vali_sequence, train_vali_linenumber, train_vali_label = process('train+vali_+10.csv')
    #'''在else中所进行的几个步骤：1、先按照比例将独立测试集的正负样本的sequence和label都提取出来。2、将提取出来的正负样本的sequence以及label都合成为一个文件。3、将sequence以及label合成为一个文件，并且保存'''
    else:
        print('未划分训练集，测试集！！！','\n')
        positive_sequence_test, positive_label_test,positive_sequence_train, positive_label_train,p_sequence_test,p_label_test,p_sequence_train_vali,p_label_test_train_vali = random_dataset_percent(positive_sequence, positive_label,percent=0.2)  # 独立测试集正样本取出比例为0.2,sequence_test,label_test是带表头的数据，另外的两个sequence、label不带表头
        negative_sequence_test, negative_label_test, negative_sequence_train, negative_label_train,n_sequence_test,n_label_test,n_sequence_train_vali,n_label_test_train_vali = random_dataset_percent(negative_sequence, negative_label,percent=0.2)  # 独立测试集正样本取出比例为0.2
        independent_test_sequence = pd.concat([p_sequence_test, n_sequence_test]) #将带有’sequence‘表头的正负样本中的sequence数据合二为一
        independent_test_label = pd.concat([p_label_test, n_label_test]) #将带有’label‘表头的正负样本中的sequence数据合二为一

        test_merge = pd.concat([independent_test_sequence, independent_test_label], axis=1,names=['sequence','label']) #将sequence数据与label数据合二为一
        test_merge.to_csv('test_+10.csv', encoding='gbk',sep='\t') #将测试集的数据保存到文件夹中去
        independent_test_sequence, independent_test_linenumber, independent_test_label = process('test_+10.csv')

        '''将已经抽出test后剩下的数据集用来拆分成训练集以及验证集，再次之前先做一些预处理'''
        sequence_train_vali = pd.concat([p_sequence_train_vali, n_sequence_train_vali])  # 将带有’sequence‘表头的正负样本中的sequence数据合二为一
        label_train_vali = pd.concat([p_label_test_train_vali, n_label_test_train_vali])  # 将带有’label‘表头的正负样本中的sequence数据合二为一
        train_vali_merge = pd.concat([sequence_train_vali, label_train_vali], axis=1,names=['sequence', 'label'])  # 将sequence数据与label数据合二为一
        train_vali_merge.to_csv('train+vali_+10.csv', encoding='gbk', sep='\t')  # 将测试集的数据保存到文件夹中去
        train_vali_sequence, train_vali_linenumber, train_vali_label = process('train+vali_+10.csv')
        print('##################################划分测试集成功，训练集测试集文件名称分别为：train+vali_+10.csv','\t','test_+10######################################','\n')
    '''3、接下来，拆分训练集以及验证集'''
    from sklearn.model_selection import train_test_split
    os.chdir(result_path)
    '''4、得到可用于输入到模型中的独立测试集'''
    X_TEST=independent_test_sequence
    Y_TEST=independent_test_label
    x_vali, y_vali=X_TEST,Y_TEST
    ind_res, history, y_test = evaluate(train_vali_sequence, train_vali_label,x_vali, y_vali, X_TEST,Y_TEST,epochs=epoch, batch_size=64,line_number_train=train_vali_linenumber,line_number_vali=independent_test_linenumber,line_number_test=independent_test_linenumber)
    save_predict_result(ind_res, 'dp_sh_221005_20%_test.txt')
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # 使用辅助函数绘制和保存图表，避免直接导入matplotlib
    save_training_plots(acc, val_acc, loss, val_loss)

#储存结果的自定义函数
def save_training_plots(acc, val_acc, loss, val_loss):
    # 移除matplotlib，只保存训练数据到文件
    try:
        # 保存训练数据到文件
        with open('training_metrics.txt', 'w') as f:
            f.write('# 训练指标数据\n')
            f.write('# Epoch\tTrain_Acc\tVal_Acc\tTrain_Loss\tVal_Loss\n')
            
            # 写入每个epoch的数据
            for i in range(len(acc)):
                train_acc_val = acc[i] if i < len(acc) else 'N/A'
                val_acc_val = val_acc[i] if i < len(val_acc) else 'N/A'
                train_loss_val = loss[i] if i < len(loss) else 'N/A'
                val_loss_val = val_loss[i] if i < len(val_loss) else 'N/A'
                
                f.write(f"{i+1}\t{train_acc_val}\t{val_acc_val}\t{train_loss_val}\t{val_loss_val}\n")
        
        # 输出最终指标摘要
        final_epoch = len(acc)
        final_train_acc = acc[-1] if acc else 'N/A'
        final_val_acc = val_acc[-1] if val_acc else 'N/A'
        final_train_loss = loss[-1] if loss else 'N/A'
        final_val_loss = val_loss[-1] if val_loss else 'N/A'
        
        print(f"训练指标已保存到training_metrics.txt")
        print(f"最终指标 (第{final_epoch}轮):")
        print(f"  训练准确率: {final_train_acc}")
        print(f"  验证准确率: {final_val_acc}")
        print(f"  训练损失: {final_train_loss}")
        print(f"  验证损失: {final_val_loss}")
        
    except Exception as e:
        print(f"保存训练指标时出错: {e}")
        
#储存结果的自定义函数
def save_predict_result(data, output):
    with open(output, 'w') as f:
        f.write('# result for true and predict\n')
        for i in data:
            f.write('%d\t%f\n' % (i[0], float(i[1])))
    return None

#运行main函数
def random_dataset_number(data,label,sample_num):
    '''随机产生一组数，加上random.seed(1)之后就可以保持每次循环所用的数是一样的，这里选择的是直接输入样本数量；还有另外一种方法，是输入取值所占百分比:1、平衡正负样本'''
    import random
    random.seed(1)  # 此行代码：括号里面是0，则每次产生的数组不一样，是1的话每次产生的数组是一样的
    sample_list = [i for i in range(len(data))]  # [0, 1, 2, 3, 4, 5, 6, 7]
    sample_list = random.sample(sample_list, sample_num)  # 随机选取出了 [3, 4, 2, 0]
    sample_data = [data[i] for i in sample_list]  # ['d', 'e', 'c', 'a']
    sample_label = [label[i] for i in sample_list]  # [3, 4, 2, 0]

    import pandas as pd
    '''保存list'''
    name1 = ['sequence']
    sequence = pd.DataFrame(columns=name1, data=sample_data)  # 定义数据的表头以及数据,此行命令保存的是数据那一列
    name2 = ['label']
    label = pd.DataFrame(columns=name2, data=sample_label)  # 定义数据的表头以及数据,此行命令保存的是label那一列
    return sample_data,sample_label,sequence,label
def random_dataset_percent(data,label,percent):
    '''随机产生一组数，加上random.seed(1)之后就可以保持每次循环所用的数是一样的，这里选择的是独立测试集所占百分比:2、提取独立测试集的自定函数'''
    import random
    sample_num = int(percent * len(data))  # 假设取20%的数据作为独立测试集
    random.seed(2)  # 此行代码：括号里面是0，则每次产生的数组不一样，是非0的话每次产生的数组是一样的，保证每次提取的独立测试集都相同
    sample_list_all = [i for i in range(len(data))]  # [0, 1, 2, 3, 4, 5, 6, 7]
    sample_list = random.sample(sample_list_all, sample_num)  # 随机选取出了 [3, 4, 2, 0]
    sample_difference = list(set(sample_list_all).difference(set(sample_list)))
    sample_test_data = [data[i] for i in sample_list]  # ['d', 'e', 'c', 'a']
    sample_test_label = [label[i] for i in sample_list]  # [3, 4, 2, 0]
    sample_train_data = [data[i] for i in sample_difference]  # ['d', 'e', 'c', 'a']
    sample_train_label = [label[i] for i in sample_difference]  # [3, 4, 2, 0]
    import pandas as pd
    '''保存list'''
    name = ['sequence']
    sequence = pd.DataFrame(columns=name, data=sample_test_data)  # 定义数据的表头以及数据,此行命令保存的是数据那一列
    name = ['label']
    label = pd.DataFrame(columns=name, data=sample_test_label)  # 定义数据的表头以及数据,此行命令保存的是label那一列
    '''保存train_vali的list'''
    name = ['sequence']
    sequence_train = pd.DataFrame(columns=name, data=sample_train_data)  # 定义数据的表头以及数据,此行命令保存的是数据那一列
    name = ['label']
    label_train = pd.DataFrame(columns=name, data=sample_train_label)  # 定义数据的表头以及数据,此行命令保存的是label那一列
    return sample_test_data,sample_test_label,sample_train_data,sample_train_label,sequence,label,sequence_train,label_train

if __name__ == '__main__':
    main()
'''1、平衡正负样本
   2、先把独立测试集按比例拆分出来（正负样本分别提取、然后合并），加上判断测试集文件是否存在的命令
   3、取出独立测试集之后的正样本负样本都随机按比例拆分成训练集、验证集，训练过程中保证每次所拆分的训练集与验证集保持不变
   4、按照原来代码中就有的函数将label与sequence合成一个文件（确保一一对应），然后随机化打乱处理
   5、训练集：验证集：测试集=6:2:2'''
