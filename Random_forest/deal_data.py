import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

df1 = pd.read_csv("/Users/apple/Acoustic-Classification/Random_forest/IDMT_features.csv", usecols=['class'], index_col=None)
df2 = pd.read_csv("/Users/apple/Acoustic-Classification/Random_forest/IDMT_librosa_features.csv", usecols=['frequency_coefficients', 'average_zero_cross_rate'], index_col=None)
df3 = pd.read_csv("/Users/apple/Acoustic-Classification/Random_forest/IDMT_statistical_features.csv", usecols=['mode_var', 's', 'g', 'variance', 'gstd_var', 'ent'], index_col=None)

df1_numpy = df1.to_numpy()
df2_numpy = df2.to_numpy()
df3_numpy = df3.to_numpy()

IDMT_C_data = list()
IDMT_M_data = list()
IDMT_T_data = list()
IDMT_Other_data = list()

for i in range(df3_numpy.shape[0]):
    if df1_numpy[i][0] == "C":
        IDMT_C_data.append(df2_numpy[i].tolist() + df3_numpy[i].tolist() + [0])
    elif df1_numpy[i][0] == "M":
        IDMT_M_data.append(df2_numpy[i].tolist() + df3_numpy[i].tolist() + [1])
    elif df1_numpy[i][0] == "T":
        IDMT_T_data.append(df2_numpy[i].tolist() + df3_numpy[i].tolist() + [2])
    else:
        IDMT_Other_data.append(df2_numpy[i].tolist() + df3_numpy[i].tolist() + [3])
IDMT_C_data = np.array(IDMT_C_data)
IDMT_M_data = np.array(IDMT_M_data)
IDMT_T_data = np.array(IDMT_T_data)
IDMT_Other_data = np.array(IDMT_Other_data)
print(np.array(IDMT_C_data).shape)
print(np.array(IDMT_M_data).shape)
print(np.array(IDMT_T_data).shape)
print(np.array(IDMT_Other_data).shape)
# 归一化处理
IDMT_C_data_X = IDMT_C_data[:,:8]
IDMT_M_data_X = IDMT_M_data[:,:8]
IDMT_T_data_X = IDMT_T_data[:,:8]
IDMT_Other_data_X = IDMT_Other_data[:,:8]
IDMT_C_data_Y = IDMT_C_data[:,8:9]
IDMT_M_data_Y = IDMT_M_data[:,8:9]
IDMT_T_data_Y = IDMT_T_data[:,8:9]
IDMT_Other_data_Y = IDMT_Other_data[:,8:9]

# 对X数据的每一列进行归一化，归一化之后就变成了 0 到 1 之间
IDMT_C_data_X = IDMT_C_data_X / IDMT_C_data_X.max(axis=0)
IDMT_M_data_X = IDMT_M_data_X / IDMT_M_data_X.max(axis=0)
IDMT_T_data_X = IDMT_T_data_X / IDMT_T_data_X.max(axis=0)
IDMT_Other_data_X = IDMT_Other_data_X / IDMT_Other_data_X.max(axis=0)


df1 = pd.read_csv("/Users/apple/Acoustic-Classification/Random_forest/MVD_features.csv", usecols=['class'], index_col=None)
df2 = pd.read_csv("/Users/apple/Acoustic-Classification/Random_forest/MVD_librosa_features.csv", usecols=['frequency_coefficients', 'average_zero_cross_rate'], index_col=None)
df3 = pd.read_csv("/Users/apple/Acoustic-Classification/Random_forest/MVD_statistical_features.csv", usecols=['mode_var', 's', 'g', 'variance', 'gstd_var', 'ent'], index_col=None)

df1_numpy = df1.to_numpy()
df2_numpy = df2.to_numpy()
df3_numpy = df3.to_numpy()

MVD_C_data = list()
MVD_M_data = list()
MVD_T_data = list()
MVD_Other_data = list()

for i in range(df3_numpy.shape[0]):
    if df1_numpy[i][0] == "C":
        MVD_C_data.append(df2_numpy[i].tolist() + df3_numpy[i].tolist() + [0])
    elif df1_numpy[i][0] == "M":
        MVD_M_data.append(df2_numpy[i].tolist() + df3_numpy[i].tolist() + [1])
    elif df1_numpy[i][0] == "T":
        MVD_T_data.append(df2_numpy[i].tolist() + df3_numpy[i].tolist() + [2])
    else:
        MVD_Other_data.append(df2_numpy[i].tolist() + df3_numpy[i].tolist() + [3])

MVD_C_data = np.array(MVD_C_data)
MVD_M_data = np.array(MVD_M_data)
MVD_T_data = np.array(MVD_T_data)
MVD_Other_data = np.array(MVD_Other_data)
print(np.array(MVD_C_data).shape)
print(np.array(MVD_M_data).shape)
print(np.array(MVD_T_data).shape)
print(np.array(MVD_Other_data).shape)
# 归一化处理
MVD_C_data_X = MVD_C_data[:,:8]
MVD_M_data_X = MVD_M_data[:,:8]
MVD_T_data_X = MVD_T_data[:,:8]
MVD_Other_data_X = MVD_Other_data[:,:8]
MVD_C_data_Y = MVD_C_data[:,8:9]
MVD_M_data_Y = MVD_M_data[:,8:9]
MVD_T_data_Y = MVD_T_data[:,8:9]
MVD_Other_data_Y = MVD_Other_data[:,8:9]

# 对X数据的每一列进行归一化，归一化之后就变成了 0 到 1 之间
MVD_C_data_X = MVD_C_data_X / MVD_C_data_X.max(axis=0)
MVD_M_data_X = MVD_M_data_X / MVD_M_data_X.max(axis=0)
MVD_T_data_X = MVD_T_data_X / MVD_T_data_X.max(axis=0)
MVD_Other_data_X = MVD_Other_data_X / MVD_Other_data_X.max(axis=0)

IDMT_C_indexs = np.random.choice(IDMT_C_data_X.shape[0], size=100, replace=False)
MVD_C_indexs = np.random.choice(MVD_C_data_X.shape[0], size=100, replace=False)
C_data = np.concatenate((np.concatenate((IDMT_C_data_X[IDMT_C_indexs], IDMT_C_data_Y[IDMT_C_indexs]), axis=1), np.concatenate((MVD_C_data_X[MVD_C_indexs], MVD_C_data_Y[MVD_C_indexs]), axis=1)), axis=0)
IDMT_M_indexs = np.random.choice(IDMT_M_data_X.shape[0], size=100, replace=False)
MVD_M_indexs = np.random.choice(MVD_M_data_X.shape[0], size=100, replace=False)
M_data = np.concatenate((np.concatenate((IDMT_M_data_X[IDMT_M_indexs], IDMT_M_data_Y[IDMT_M_indexs]), axis=1), np.concatenate((MVD_M_data_X[MVD_M_indexs], MVD_M_data_Y[MVD_M_indexs]), axis=1)), axis=0)
IDMT_T_indexs = np.random.choice(IDMT_T_data_X.shape[0], size=100, replace=False)
MVD_T_indexs = np.random.choice(MVD_T_data_X.shape[0], size=100, replace=False)
T_data = np.concatenate((np.concatenate((IDMT_T_data_X[IDMT_T_indexs], IDMT_T_data_Y[IDMT_T_indexs]), axis=1), np.concatenate((MVD_T_data_X[MVD_T_indexs], MVD_T_data_Y[MVD_T_indexs]), axis=1)), axis=0)
MVD_Other_indexs = np.random.choice(MVD_Other_data_X.shape[0], size=147, replace=False)
Other_data = np.concatenate((np.concatenate((IDMT_Other_data_X, IDMT_Other_data_Y), axis=1), np.concatenate((MVD_Other_data_X[MVD_Other_indexs], MVD_Other_data_Y[MVD_Other_indexs]), axis=1)), axis=0)


print(C_data.shape)
print(M_data.shape)
print(T_data.shape)
print(Other_data.shape)

all_data = np.concatenate((C_data, M_data, T_data, Other_data), axis=0)
all_data_X = all_data[:,:8]
all_data_Y = all_data[:,8:9]
# 划分数据集，测试集占总数据的25%，随机状态设置为42以确保结果可复现
X_train, X_test, y_train, y_test = train_test_split(all_data_X, all_data_Y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=30, criterion='entropy', max_depth=13, random_state=43)

# 训练模型
rf.fit(X_train, y_train)

# 预测训练集
y_pred = rf.predict(X_train)
accuracy = accuracy_score(y_train, y_pred)
print(f"Train accuracy: {accuracy:.2f}")
# 使用confusion_matrix函数生成混淆矩阵
cm = confusion_matrix(y_train, y_pred)
# 打印混淆矩阵
print(cm)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm_normalized)

# 预测测试集
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy:.2f}")
# 使用confusion_matrix函数生成混淆矩阵
cm = confusion_matrix(y_test, y_pred)
# 打印混淆矩阵
print(cm)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm_normalized)