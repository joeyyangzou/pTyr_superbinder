# 基础库导入
import os
import numpy as np
import pandas as pd

# TensorFlow相关导入
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, optimizers, backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# 数据处理和评估相关导入
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score


# 配置GPU内存
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU配置成功")
    except RuntimeError as e:
        print(e)

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
standard_file_path=os.path.join(current_dir, "regression_input.txt")#用于数据标准化处理的文件路径
model_save_path=os.path.join(current_dir, "models", "regression_model")#模型保存路径
data_path=current_dir#训练数据集以及独立测试集的保存路径
result_save_path=os.path.join(current_dir, "output")#预测结果的保存路径

# 创建必要的目录
os.makedirs(os.path.join(current_dir, "models"), exist_ok=True)
os.makedirs(os.path.join(current_dir, "output"), exist_ok=True)

StandardScalerIN=pd.read_csv(standard_file_path,sep='\t')
StandardScaler_range=StandardScalerIN['value']
StandardScaler_range=StandardScaler_range.values.reshape(-1, 1)
mm = StandardScaler()
mm=mm.fit(StandardScaler_range)
value_transform=mm.transform(StandardScaler_range)

mm_minmax = MinMaxScaler(feature_range=(-1, 1))
mm_minmax=mm_minmax.fit(value_transform)
def process(inputfile_path):
    print(f"处理文件: {inputfile_path}")
    try:
        data = pd.read_csv(inputfile_path, sep='\t', header=None, names=['sequence', 'value'])
        print(f"成功读取文件，数据形状: {data.shape}")

        if 'value' not in data.columns:
            raise ValueError(f"文件 {inputfile_path} 中缺少 'value' 列")
        if 'sequence' not in data.columns:
            raise ValueError(f"文件 {inputfile_path} 中缺少 'sequence' 列")

        label = data['value']
        inputfile = data['sequence']

        line_number = len(inputfile)
        print(f"序列数量: {line_number}")

        if isinstance(label, pd.Series):
            label = label.values

        label_reshape = label.reshape(-1, 1)

        label_MinMaxScaler = mm.transform(label_reshape)
        label_MinMaxScaler = mm_minmax.transform(label_MinMaxScaler)

        return inputfile.tolist(), line_number, label, label_MinMaxScaler.flatten()
    except Exception as e:
        print(f"处理文件时出错: {e}")
        return [], 0, np.array([]), np.array([])

def One_Hot(sequence, line_number):
    AA = ['I', 'L', 'V', 'F', 'M', 'C', 'A', 'G', 'P', 'T', 'S', 'Y', 'W', 'Q', 'N', 'H', 'E', 'D', 'K', 'R']
    encodings = []
    seq_lengths = []

    print(f"One_Hot函数输入: sequence类型={type(sequence)}, line_number={line_number}")

    try:
        sequence_iter = iter(sequence)
    except TypeError:
        print(f"错误: sequence不是可迭代对象，类型为{type(sequence)}")
        return np.array([])

    all_sequences = []
    max_length = 0
    min_length = float('inf')

    try:
        for seq_line in sequence_iter:
            if isinstance(seq_line, str) or hasattr(seq_line, '__iter__'):
                seq_list = list(seq_line)
                seq_len = len(seq_list)
                max_length = max(max_length, seq_len)
                min_length = min(min_length, seq_len)
                all_sequences.append(seq_list)
            else:
                print(f"警告: 跳过非序列元素: {seq_line} (类型: {type(seq_line)})")
    except Exception as e:
        print(f"遍历序列时出错: {e}")
        return np.array([])

    if not all_sequences:
        print("错误: 没有找到有效序列")
        return np.array([])

    print(f"序列长度范围: {min_length} - {max_length}")
    standard_length = max_length

    for seq_line in all_sequences:
        code = []
        seq_len = len(seq_line)
        seq_lengths.append(seq_len)

        for aa in seq_line:
            if aa == 'X':
                code.extend([0.05 for _ in range(20)])
            else:
                for aa1 in AA:
                    tag = 1.0 if aa == aa1 else 0.0
                    code.append(tag)

        if seq_len < standard_length:
            padding_length = standard_length - seq_len
            code.extend([0.0 for _ in range(padding_length * 20)])

        flat_code = []
        for item in code:
            if isinstance(item, (int, float)):
                flat_code.append(float(item))
            else:
                try:
                    for sub_item in item:
                        flat_code.append(float(sub_item))
                except (TypeError, ValueError):
                    flat_code.append(0.0)

        encodings.append(flat_code)

    try:
        encodings = np.array(encodings, dtype=np.float32)
        print(f"编码后的形状: {encodings.shape}, 数据类型: {encodings.dtype}")
    except Exception as e:
        print(f"转换为numpy数组时出错: {e}")
        fixed_encodings = []
        for i, enc in enumerate(encodings):
            try:
                fixed_enc = np.array(enc, dtype=np.float32)
                fixed_encodings.append(fixed_enc)
            except Exception as inner_e:
                print(f"修复编码 {i} 时出错: {inner_e}")
                fixed_encodings.append(np.zeros(standard_length * 20, dtype=np.float32))

        max_enc_length = max(len(enc) for enc in fixed_encodings)
        for i in range(len(fixed_encodings)):
            if len(fixed_encodings[i]) < max_enc_length:
                padding = np.zeros(max_enc_length - len(fixed_encodings[i]), dtype=np.float32)
                fixed_encodings[i] = np.concatenate([fixed_encodings[i], padding])

        encodings = np.array(fixed_encodings, dtype=np.float32)
        print(f"修复后的编码形状: {encodings.shape}, 数据类型: {encodings.dtype}")

    actual_line_number = len(encodings)
    print(f"实际序列数量: {actual_line_number}, 预期数量: {line_number}")

    try:
        encoding_length = encodings.shape[1] if len(encodings.shape) > 1 else 0
        if encoding_length % 20 != 0:
            print(f"警告: 编码长度 {encoding_length} 不是20的倍数")
            new_length = (encoding_length // 20) * 20
            encodings = encodings[:, :new_length]
            encoding_length = new_length

        seq_len_in_encoding = encoding_length // 20
        print(f"计算出的序列长度: {seq_len_in_encoding}")

        encodings_reshaped = np.reshape(encodings, (actual_line_number, seq_len_in_encoding, 20))
        print(f"重塑后的形状: {encodings_reshaped.shape}, 数据类型: {encodings_reshaped.dtype}")
        return encodings_reshaped
    except Exception as e:
        print(f"重塑时出错: {e}")
        try:
            if len(encodings.shape) == 2:
                n_samples, n_features = encodings.shape
                possible_seq_len = n_features // 20
                if possible_seq_len * 20 == n_features:
                    reshaped = np.reshape(encodings, (n_samples, possible_seq_len, 20))
                    print(f"成功重塑为形状: {reshaped.shape}")
                    return reshaped
        except Exception as inner_e:
            print(f"备选重塑也失败: {inner_e}")

        if len(encodings.shape) == 1:
            encodings = encodings.reshape(-1, 1)
        print(f"返回未重塑的编码数组，形状: {encodings.shape}")
        return encodings

def build_network():
    conv_layers = [
        layers.Conv1D(filters=128, kernel_size=1, padding='same', activation='relu', input_shape=(None, 20)),
        layers.Dropout(0.5),
        layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'),
        layers.Dropout(0.5),
        layers.Conv1D(filters=128, kernel_size=9, padding='same', activation='relu'),
        layers.MaxPooling1D(2, 1),
        layers.Dropout(0.5),
        layers.Conv1D(filters=128, kernel_size=10, padding='same', activation='relu'),
        layers.MaxPooling1D(pool_size=2, strides=1),
        layers.Dropout(0.5),
    ]

    fc_layers = [
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(8, activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(1, activation='tanh')
    ]

    conv_layers.extend(fc_layers)
    network = Sequential(conv_layers)
    network.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    network.summary()
    print("模型已成功构建")
    return network


def coeff_determination(y_true, y_pred):
    SS_res =K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def preprocess(x, y):
    try:
        x = tf.cast(x, dtype=tf.float32)
    except TypeError as e:
        print(f"转换x到float32时出错: {e}")
        if isinstance(x, list):
            x = np.array(x, dtype=np.float32)
        x = tf.convert_to_tensor(x, dtype=tf.float32)

    try:
        if isinstance(y, list):
            y_flat = []
            for item in y:
                if isinstance(item, list):
                    y_flat.append(float(item[0]) if item else 0.0)
                else:
                    y_flat.append(float(item))
            y = np.array(y_flat, dtype=np.float32)

        y = tf.cast(y, dtype=tf.float32)
    except (TypeError, ValueError) as e:
        print(f"转换y到float32时出错: {e}")
        if not isinstance(y, (np.ndarray, tf.Tensor)):
            y = np.array([float(0.0)], dtype=np.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)

    return x, y

def evaluate(X, Y, Y_train_noNormalization, X_vali, Y_vali, Y_vali_noNormalization,
             X_TEST, Y_TEST, Y_test_noNormalization, batch_size=128, epochs=100,
             line_number_train=0, line_number_vali=0, line_number_test=0):
    print("开始评估模型...")
    X_train, y_train = X, Y

    if isinstance(X_train, list):
        print(f"X_train是列表类型，长度: {len(X_train)}")
        cleaned_X_train = []
        for i, item in enumerate(X_train):
            if isinstance(item, str):
                cleaned_X_train.append(item)
            elif hasattr(item, '__str__'):
                str_item = str(item)
                cleaned_X_train.append(str_item)
                print(f"将X_train[{i}]从 {type(item)} 转换为字符串")
            else:
                print(f"跳过无效的X_train元素[{i}]: {item} (类型: {type(item)})")
        X_train = cleaned_X_train
        line_number_train = len(X_train)
    elif isinstance(X_train, (pd.Series, pd.DataFrame)):
        print(f"X_train是pandas对象，转换为列表")
        X_train = X_train.tolist()
    elif isinstance(X_train, np.ndarray):
        print(f"X_train是numpy数组，转换为列表")
        try:
            if X_train.dtype == 'object':
                cleaned_X_train = []
                for i, item in enumerate(X_train):
                    try:
                        if isinstance(item, str):
                            cleaned_X_train.append(item)
                        elif hasattr(item, '__str__'):
                            str_item = str(item)
                            cleaned_X_train.append(str_item)
                        else:
                            print(f"跳过numpy数组中的无效元素[{i}]: {item} (类型: {type(item)})")
                    except Exception as e:
                        print(f"处理numpy数组中元素[{i}]时出错: {e}")
                X_train = cleaned_X_train
            else:
                X_train = [str(item) for item in X_train]
            line_number_train = len(X_train)
        except Exception as e:
            print(f"将numpy数组转换为字符串列表时出错: {e}")
            X_train = []
    elif X_train is None:
        print("警告: X_train为None，使用空列表")
        X_train = []
    else:
        print(f"X_train是未知类型 {type(X_train)}，尝试转换为列表")
        try:
            X_train = list(X_train)
        except Exception as e:
            print(f"转换X_train为列表失败: {e}")
            X_train = []

    if isinstance(y_train, list):
        print(f"将y_train从list转换为numpy数组，原始长度: {len(y_train)}")
        processed_y = []
        for i, item in enumerate(y_train):
            try:
                if isinstance(item, list):
                    processed_y.append(float(item[0]) if item else 0.0)
                elif isinstance(item, (int, float, np.number)):
                    processed_y.append(float(item))
                else:
                    processed_y.append(float(item))
            except (TypeError, ValueError):
                print(f"无法转换y_train[{i}]为浮点数: {item} (类型: {type(item)})")
                processed_y.append(0.0)
        y_train = np.array(processed_y, dtype=np.float32)
        print(f"转换后的y_train形状: {y_train.shape}")
    elif isinstance(y_train, (pd.Series, pd.DataFrame)):
        print(f"y_train是pandas对象，转换为numpy数组")
        y_train = y_train.values.astype(np.float32)
    elif isinstance(y_train, np.ndarray):
        print(f"y_train已经是numpy数组，确保类型为float32")
        y_train = y_train.astype(np.float32)
    elif y_train is None:
        print("警告: y_train为None，使用默认数组")
        y_train = np.array([0.0], dtype=np.float32)
    else:
        print(f"y_train是未知类型 {type(y_train)}，尝试转换为numpy数组")
        try:
            y_train = np.array(y_train, dtype=np.float32)
        except Exception as e:
            print(f"转换y_train时出错: {e}")
            y_train = np.array([0.0], dtype=np.float32)

    if isinstance(Y_vali, list):
        print(f"将Y_vali从list转换为numpy数组，原始长度: {len(Y_vali)}")
        processed_y = []
        for i, item in enumerate(Y_vali):
            try:
                if isinstance(item, list):
                    processed_y.append(float(item[0]) if item else 0.0)
                elif isinstance(item, (int, float, np.number)):
                    processed_y.append(float(item))
                else:
                    processed_y.append(float(item))
            except (TypeError, ValueError):
                print(f"无法转换Y_vali[{i}]为浮点数: {item} (类型: {type(item)})")
                processed_y.append(0.0)
        Y_vali = np.array(processed_y, dtype=np.float32)
        print(f"转换后的Y_vali形状: {Y_vali.shape}")
    elif isinstance(Y_vali, (pd.Series, pd.DataFrame)):
        Y_vali = Y_vali.values.astype(np.float32)
    elif isinstance(Y_vali, np.ndarray):
        Y_vali = Y_vali.astype(np.float32)
    elif Y_vali is None:
        print("警告: Y_vali为None，使用默认数组")
        Y_vali = np.array([0.0], dtype=np.float32)
    else:
        try:
            Y_vali = np.array(Y_vali, dtype=np.float32)
        except Exception as e:
            print(f"转换Y_vali时出错: {e}")
            Y_vali = np.array([0.0], dtype=np.float32)

    if isinstance(Y_TEST, list):
        print(f"将Y_TEST从list转换为numpy数组，原始长度: {len(Y_TEST)}")
        processed_y = []
        for i, item in enumerate(Y_TEST):
            try:
                if isinstance(item, list):
                    processed_y.append(float(item[0]) if item else 0.0)
                elif isinstance(item, (int, float, np.number)):
                    processed_y.append(float(item))
                else:
                    processed_y.append(float(item))
            except (TypeError, ValueError):
                print(f"无法转换Y_TEST[{i}]为浮点数: {item} (类型: {type(item)})")
                processed_y.append(0.0)
        Y_TEST = np.array(processed_y, dtype=np.float32)
        print(f"转换后的Y_TEST形状: {Y_TEST.shape}")
    elif isinstance(Y_TEST, (pd.Series, pd.DataFrame)):
        Y_TEST = Y_TEST.values.astype(np.float32)
    elif isinstance(Y_TEST, np.ndarray):
        Y_TEST = Y_TEST.astype(np.float32)
    elif Y_TEST is None:
        print("警告: Y_TEST为None，使用默认数组")
        Y_TEST = np.array([0.0], dtype=np.float32)
    else:
        try:
            Y_TEST = np.array(Y_TEST, dtype=np.float32)
        except Exception as e:
            print(f"转换Y_TEST时出错: {e}")
            Y_TEST = np.array([0.0], dtype=np.float32)

    for name, data in [('X_vali', X_vali), ('X_TEST', X_TEST)]:
        if isinstance(data, list):
            cleaned_data = []
            for i, item in enumerate(data):
                if isinstance(item, str):
                    cleaned_data.append(item)
                elif hasattr(item, '__str__'):
                    str_item = str(item)
                    cleaned_data.append(str_item)
                    print(f"将{name}[{i}]从 {type(item)} 转换为字符串")
                else:
                    print(f"跳过{name}中的无效元素[{i}]: {item} (类型: {type(item)})")
            if name == 'X_vali':
                X_vali = cleaned_data
                line_number_vali = len(X_vali)
            else:
                X_TEST = cleaned_data
                line_number_test = len(X_TEST)
        elif isinstance(data, (pd.Series, pd.DataFrame)):
            if name == 'X_vali':
                X_vali = data.tolist()
                line_number_vali = len(X_vali)
            else:
                X_TEST = data.tolist()
                line_number_test = len(X_TEST)
        elif isinstance(data, np.ndarray):
            if data.dtype == 'object':
                cleaned_data = []
                for i, item in enumerate(data):
                    if isinstance(item, str):
                        cleaned_data.append(item)
                    elif hasattr(item, '__str__'):
                        cleaned_data.append(str(item))
                if name == 'X_vali':
                    X_vali = cleaned_data
                    line_number_vali = len(X_vali)
                else:
                    X_TEST = cleaned_data
                    line_number_test = len(X_TEST)
            else:
                if name == 'X_vali':
                    X_vali = data.tolist()
                    line_number_vali = len(X_vali)
                else:
                    X_TEST = data.tolist()
                    line_number_test = len(X_TEST)
        else:
            if name == 'X_vali':
                X_vali = []
                line_number_vali = 0
            else:
                X_TEST = []
                line_number_test = 0

    print(f"数据处理完成: X_train={len(X_train)}, X_vali={len(X_vali)}, X_TEST={len(X_TEST)}")

    X_train = One_Hot(X_train, line_number_train)
    X_vali = One_Hot(X_vali, line_number_vali)
    X_test = One_Hot(X_TEST, line_number_test)

    X_train_t = X_train
    X_vali_t = X_vali
    X_test_t = X_test

    X_test_t = tf.cast(X_test_t, dtype=tf.float32)

    train_db = tf.data.Dataset.from_tensor_slices((X_train_t, y_train))
    train_db = train_db.shuffle(len(X_train)).map(preprocess).batch(batch_size)

    vali_db = tf.data.Dataset.from_tensor_slices((X_vali_t, Y_vali))
    vali_db = vali_db.shuffle(len(X_vali_t)).map(preprocess).batch(batch_size)

    test_db = tf.data.Dataset.from_tensor_slices((X_test_t, Y_TEST))
    test_db = test_db.shuffle(len(X_test_t)).map(preprocess).batch(batch_size)

    network = build_network()
    model_save_file_path=model_save_path
    checkpoint = ModelCheckpoint(model_save_file_path, monitor='val_mae', verbose=1, save_best_only=True, mode='auto', save_weights_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=50)
    callbacks_list = [early_stopping, checkpoint]

    history = network.fit(train_db, epochs=epochs, validation_data=vali_db,  verbose=1, callbacks=[callbacks_list])
    print("Independent test:", network.evaluate(test_db))

    predict = network.predict(X_test_t, batch_size=batch_size)
    print(predict)

    predict=mm_minmax.inverse_transform(predict)
    predict = mm.inverse_transform(predict)
    print()
    R_squire = r2_score(Y_test_noNormalization, predict)

    tmp_result = np.zeros((len(Y_test_noNormalization), 2))
    tmp_result[:, 0], tmp_result[:, 1] = Y_test_noNormalization, predict[:, 0]

    network.save(model_save_file_path, save_format='tf')
    return tmp_result, history, R_squire

def save_predict_result(data, output):
    with open(output, 'w') as f:
        f.write('value'+'\t'+'predict'+'\n')
        for i in data:
            f.write('%f\t%f\n' % (i[0], float(i[1])))
    return None

def save_training_metrics(history):
    try:
        with open('training_metrics.txt', 'w') as f:
            f.write('# 训练指标数据\n')
            f.write('# Epoch\tTrain_MAE\tVal_MAE\tTrain_Loss\tVal_Loss\n')

            acc = history.history['mae']
            val_acc = history.history['val_mae']
            loss = history.history['loss']
            val_loss = history.history['val_loss']

            for i in range(len(acc)):
                train_mae_val = acc[i] if i < len(acc) else 'N/A'
                val_mae_val = val_acc[i] if i < len(val_acc) else 'N/A'
                train_loss_val = loss[i] if i < len(loss) else 'N/A'
                val_loss_val = val_loss[i] if i < len(val_loss) else 'N/A'

                f.write(f"{i+1}\t{train_mae_val}\t{val_mae_val}\t{train_loss_val}\t{val_loss_val}\n")

        final_epoch = len(acc)
        final_train_mae = acc[-1] if acc else 'N/A'
        final_val_mae = val_acc[-1] if val_acc else 'N/A'
        final_train_loss = loss[-1] if loss else 'N/A'
        final_val_loss = val_loss[-1] if val_loss else 'N/A'

        print(f"训练指标已保存到training_metrics.txt")
        print(f"最终指标 (第{final_epoch}轮):")
        print(f"  训练MAE: {final_train_mae}")
        print(f"  验证MAE: {final_val_mae}")
        print(f"  训练损失: {final_train_loss}")
        print(f"  验证损失: {final_val_loss}")

    except Exception as e:
        print(f"保存训练指标时出错: {e}")

def main():
    os.chdir(data_path)
    epoch = 1000
    X_train,line_number_train,Y_train,Y_train_MinMaxScaler=process(os.path.join(data_path, "train_set.txt"))
    X_TEST, line_number_test,Y_TEST,Y_TEST_MinMaxScaler=process(os.path.join(data_path, "test_set.txt"))

    x_vali, y_vali = X_TEST, Y_TEST_MinMaxScaler
    line_number_vali=line_number_test
    os.chdir(result_save_path)
    ind_res_test, history,R_squire_test = evaluate(X_train, Y_train_MinMaxScaler,Y_train,x_vali, y_vali,y_vali, X_TEST,Y_TEST_MinMaxScaler,Y_TEST,epochs=epoch, batch_size=128,line_number_train=line_number_train,line_number_vali=line_number_vali,line_number_test=line_number_test)
    save_predict_result(ind_res_test, 'regression_predict.txt')
    save_training_metrics(history)
    print('R_squire_test:' + str(R_squire_test))

if __name__ == '__main__':
    main()
