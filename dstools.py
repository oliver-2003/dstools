import math
from scipy.stats import ks_2samp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.impute import SimpleImputer


def missing(df):
    """
    计算每一列的缺失值及占比
    """
    missing_number = df.isnull().sum().sort_values(ascending=False)
    missing_percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False).apply(
        lambda x: round(x * 100, 4))
    missing_values = pd.concat([missing_number, missing_percent], axis=1, keys=["missing_number", 'missing_percent'])
    return missing_values


def cate_var_values(df_cate):
    res = {col: list(df_cate[col].unique()) for col in df_cate.columns}
    return res


def find_index(data_col, val_list):
    """
    查询某值在某列中第一次出现位置的索引，没有则返回-1
    :param data_col: 查询的列
    :param val: 具体取值
    """
    for var in val_list:
        if data_col.sum() == 0:
            index = -1
    else:
        index = data_col.isin(val_list).idxmax()  # 取值为0或1，返回第一次出现最大值的索引，即第一次出现1的索引
    return index


def value_index(df, val_list):
    # 创建一个空的数据框，只包含一个列 'index_list'
    result_df = pd.DataFrame(columns=['index_list', 'value_count'],
                             index=pd.MultiIndex.from_product([df.columns, val_list], names=['Feature', 'Value']))

    # 遍历数据框的列和列表 val_list 中的值
    for col in df.columns:
        for val in val_list:
            # 找到与 val_list 中的值相等的索引，并将索引存储在列表 B 中
            index_list = df.index[df[col].isin([val])].tolist()

            # 将索引列表存储在结果数据框中
            result_df.loc[(col, val), 'index_list'] = index_list
            result_df.loc[(col, val), 'value_count'] = len(index_list)

    return result_df


def detect_outlier(df, method='sigma'):
    result_df = pd.DataFrame(columns=['Outliers', 'Value_Count', 'Z-Score'], index=df.columns)
    if method == 'sigma':
        for col in df.columns:
            mean_value = df[col].mean()
            std = df[col].std()
            outliers = df[col][(df[col] >= mean_value + 3 * std) | (df[col] <= mean_value - 3 * std)].tolist()
            result_df.loc[col, "Outliers"] = outliers
            result_df.loc[col, "Value_Count"] = len(outliers)
            result_df.loc[col, "Z-Score"] = [round((outlier - mean_value) / std, 4) for outlier in outliers]
    elif method == 'IQR':
        for col in df.columns:
            mean_value = df[col].mean()
            std = df[col].std()
            q1 = df[col].describe()['25%']
            q3 = df[col].describe()['75%']
            iqr = q3 - q1
            outliers = df[col][(df[col] <= q1 - 1.5 * iqr) | (df[col] >= q3 + 1.5 * iqr)].tolist()
            result_df.loc[col, "Outliers"] = outliers
            result_df.loc[col, "Value_Count"] = len(outliers)
            result_df.loc[col, "Z-Score"] = [round((outlier - mean_value) / std, 4) for outlier in outliers]
    return result_df


def prop_bar_polt(df, var_list, target):
    height = math.floor(math.sqrt(len(var_list)))
    width = math.ceil(math.sqrt(len(var_list)))
    fig, axes = plt.subplots(nrows=height, ncols=width, figsize=(8 * height, 4 * width))
    for i, item in enumerate(var_list):
        plt.subplot(height, width, (i + 1))
        ax = sns.countplot(x=item, hue=target, data=df, palette="Blues", dodge=False)
        plt.xlabel(item)
        plt.title(f"{target} by " + item)


def kde_plot(df, target, feature, figsize=(16, 6), dpi=200):
    plt.figure(figsize=figsize, dpi=dpi)

    # Subset the data for the two target categories
    df_target_0 = df[df[target] == 0]
    df_target_1 = df[df[target] == 1]

    # Plot KDE for target == 0 in red
    ax = sns.kdeplot(data=df_target_0, x=feature, color="Red", fill=True, label=f'{target} = 0')

    # Plot KDE for target == 1 in blue
    ax = sns.kdeplot(data=df_target_1, x=feature, color="Blue", fill=True, label=f'{target} = 1')

    ax.set_ylabel('Density')
    ax.set_xlabel(feature)
    ax.set_title(f'Distribution of {feature} by {target}')

    plt.legend()
    plt.show()


def insert_missing(data):
    # 1、确定插⼊缺失值的⽐例
    missing_rate = 0.3
    # 2、计算插⼊缺失值的点数
    x_full, y_full = data.iloc[:, :-1], data.iloc[:, -1]
    n_samples = x_full.shape[0]
    n_features = x_full.shape[1]
    n_missing_samples = int(np.round(n_samples * n_features * missing_rate))
    # n_missing_samples = int(np.floor(n_samples * n_features * missing_rate))
    # np.floor向下取整，返回.0格式的浮点数
    # np.round向上取整，返回.0格式的浮点数
    # 我们希望放⼊的缺失数据的⽐例假设是30%，那总共就要有1185-1186个数据缺失

    # 3、获得插⼊缺失值的位置的索引
    rng = np.random.RandomState(0)
    # 缺失值插⼊位置的列索引 (0,8,1186)
    missing_col_index = rng.randint(0, n_features, n_missing_samples)
    # 缺失值插⼊位置的⾏索引(0,494,1186)
    missing_row_index = rng.randint(0, n_samples, n_missing_samples)

    # 4、按照对应的⾏索引和列索隐插⼊缺失值到指定位置：
    x_missing = x_full.copy()
    y_missing = y_full.copy()
    # 按照索引数据（DataFrame）插⼊缺失值
    for i in range(n_missing_samples):
        x_missing.iloc[missing_row_index[i], missing_col_index[i]] = np.nan
    # 把标签列拼接回来，使数据集和原始数据集维度⼀致
    data_missing = x_missing
    data_missing.iloc[:, -1] = y_missing


def fill_missing(data_missing, col):

    # ⽤0填补缺失值
    data_fill_0 = data_missing.copy()
    data_fill_0 = data_fill_0.fillna(value=0)

    # ⽤均值填补缺失值
    data_fill_mean = data_missing.copy()
    data_fill_mean[col] = data_fill_mean[col].fillna(data_fill_mean[col].mean())
    data_fill_mean.isnull().sum()

    # ⽤众数填补缺失值
    data_fill_mode = data_missing.copy()
    data_fill_mode[col] = data_fill_mode[col].fillna(data_fill_mode[col].mode()[0])
    data_fill_mode.isnull().sum()

    # 使⽤下⼀个值填补缺失值
    """
    method =
        ffill/pad:从缺失值前⾯的有效值来从前往后填充
        bfill/backfill:从缺失值后⾯的有效值来从后往前填补
    """
    data_fill_bf = data_missing.copy()
    data_fill_bf[col] = data_fill_bf[col].fillna(method='bfill')

    # 使⽤前⼀个值填补缺失值
    data_fill_pad = data_missing.copy()
    data_fill_pad[col] = data_fill_pad[col].fillna(method='pad')

    # 使⽤插值函数插补缺失值
    """
    method =
        nearest：最邻近插值法
        zero：阶梯插值
        slinear、linear：线性插值
        quadratic、cubic：2、3阶B样条曲线插值
    """
    data_fill_it_line = data_missing.copy()
    data_fill_it_line[col] = data_fill_it_line[col].interpolate(method="linear")

    data_fill_it_near = data_missing.copy()
    data_fill_it_near[col] = data_fill_it_near[col].interpolate(method="nearest")

    data_fill_it_zero = data_missing.copy()
    data_fill_it_zero[col] = data_fill_it_zero[col].interpolate(method="zero")

    data_fill_it_quad = data_missing.copy()
    data_fill_it_quad[col] = data_fill_it_quad[col].interpolate(method="quadratic")

    data_fill_it_cub = data_missing.copy()
    data_fill_it_cub[col] = data_fill_it_cub[col].interpolate(method="cubic")

    # 可视化数据拟合情况
    datas = [data_fill_0, data_fill_mean, data_fill_mode, data_fill_bf, data_fill_pad, data_fill_it_line, data_fill_it_near, data_fill_it_zero, data_fill_it_quad, data_fill_it_cub]
    mths = ["fill_0", "fill_mean", "fill_mode", "fill_bf", "fill_pad", "fill_it_line", 'data_fill_it_near', 'data_fill_it_zero', 'data_fill_it_quad', 'data_fill_it_cub', "data_missing"]
    plt.figure()
    for dat in datas:
        ax = sns.kdeplot(dat[col])
    ax = sns.kdeplot(data_missing[col], color="b", fill=True)
    ax = ax.legend(mths)
    plt.show()
    return datas


def stratified_fill_mean(data_missing, col_group, col_target, bin_num):
    # 选择age列进⾏数据分层，⾸先⽤均值填补age列缺失值
    data_fill_mean = data_missing.copy()
    data_fill_mean[col_group] = data_fill_mean[col_group].fillna(data_fill_mean[col_group].mean())
    data_fill_mean.isnull().sum()

    # 以age列的规律进⾏数据分层，并使⽤分层均值填补"2⼩时⾎清胰岛素浓度"列的缺失值
    # 1、对数据分组保留记录，⽤于后续分层均值填补验证
    data_range = data_fill_mean[col_group].max() - data_fill_mean[col_group].min()
    bin_width = data_range/(bin_num - 1)
    bins = [int(data_fill_mean[col_group].min() - 1) + bin_width * i for i in range(bin_num)]
    bins[-1] = int(data_fill_mean[col_group].max())+1
    labels = []
    for i in range(len(bins)-1):
        labels.append(str([bins[i], bins[i+1]]))
        i += 1
    # 将sale对应到指定的分组区间
    df = data_fill_mean.copy()
    df['age_catg'] = pd.cut(data_fill_mean.age, bins)
    # 统计每组频数
    aggResult = df.groupby(by=['age_catg']).agg({'age': 'count'})

    # 2、以age分层结果将“2⼩时⾎清胰岛素浓度”按照分层均值填补
    data_fill_avmean = data_fill_mean.copy()  # 使⽤填补过"age"列缺失值的数据为原始数据进⾏后续操作
    for i in range(len(bins)-1):
        ser = (data_fill_avmean[col_group] > bins[i]) & (data_fill_avmean[col_group] <= bins[i+1])
        df = data_fill_avmean[col_target][ser]
        avmean = df.mean()
        bool_index = df.index[np.where(np.isnan(df))]
        for k in range(len(bool_index)):
            data_fill_avmean.loc[bool_index[k], col_target] = avmean

    return data_fill_avmean


def linear_regression(feature, target):
    model1 = LinearRegression()
    model1.fit(feature, target)

    y_pred = model1.predict(feature)

    r2 = r2_score(target, y_pred)  # R-squared
    rmse = math.sqrt(mean_squared_error(target, y_pred))  # 均方误差

    print(f"R-squared: {r2:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")

    # 进行交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model1, feature, target, cv=kf)


    print(f"Cross-Validation R^2 Scores: {cv_scores}")
    print(f"Mean R^2: {np.mean(cv_scores):.4f}")


def best_model_selection(X, y):
    import math
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from itertools import combinations
    from sklearn.metrics import r2_score, make_scorer
    import numpy as np

    best_features = []

    for k in range(1, len(X.columns) + 1):
        best_score = -float('inf')
        for combo in combinations(X.columns, k):
            subset = X[list(combo)]  # 使用相同的训练数据
            model = LinearRegression()
            model.fit(subset, y)
            y_pred = model.predict(X[list(combo)])
            r2 = r2_score(y, y_pred)  # R-squared
            if r2 > best_score:
                best_score = r2
                n = X[list(combo)].shape[0]  # 样本数量
                p = X[list(combo)].shape[1]  # 特征数量
                adjusted_r2 = 1 - ((1 - best_score) * (n - 1) / (n - p - 1))
                best_combo = combo
        best_features.append({k: {best_combo: adjusted_r2}})
    print(best_features)

    max_r_squared = -float('inf')  # 初始设为负无穷大
    for item in best_features:
        for num_features, r_squared_dict in item.items():
            r_squared_value = list(r_squared_dict.values())[0]  # 获取R-squared值
            if r_squared_value > max_r_squared:
                max_r_squared = r_squared_value
                best_features_combo = list(r_squared_dict.keys())[0]  # 获取对应的特征组合

    print("Best R-squared:", max_r_squared)
    print("Best Features:", best_features_combo)

    return best_features_combo


def get_interactions(feature_matrix):
    data_interaction = feature_matrix.copy()
    subset = []
    for combo in combinations(feature_matrix.columns, 2):
        subset.append(combo)  # 使用相同的训练数据
    for pair in subset:
        first = pair[0]
        second = pair[1]
        data_interaction[f'{first}_{second}'] = data_interaction[first] * data_interaction[second]
        if data_interaction[f'{first}_{second}'].sum() == 0:
            data_interaction = data_interaction.drop(f'{first}_{second}', axis=1)

    return data_interaction


def forward_selection(df, feature_col, target_col):
    best_features = []
    selection_set = []
    tmp_set = []
    for k in range(1, len(feature_col) + 1):
        best_score = -float('inf')
        for feature in feature_col:
            subset = selection_set + [feature]
            model = LinearRegression()
            model.fit(df[subset], y)
            y_pred = model.predict(df[subset])
            r2 = r2_score(df[target_col], y_pred)
            if r2 > best_score:
                best_score = r2
                n = df[subset].shape[0]  # 样本数量
                p = df[subset].shape[1]  # 特征数量
                adjusted_r2 = 1 - ((1 - best_score) * (n - 1) / (n - p - 1))
                tmp_set = subset
        selection_set = tmp_set
        best_features.append({k: {tuple(selection_set): adjusted_r2}})

    max_r_squared = -float('inf')  # 初始设为负无穷大
    for item in best_features:
        for num_features, r_squared_dict in item.items():
            r_squared_value = list(r_squared_dict.values())[0]  # 获取R-squared值
            if r_squared_value > max_r_squared:
                max_r_squared = r_squared_value
                best_features_combo = list(r_squared_dict.keys())[0]  # 获取对应的特征组合

    print("Best R-squared:", max_r_squared)
    print("Best Features:", best_features_combo)
    return list(best_features_combo)


def backward_selection(data, feature_list, target_col):
    best_features = []
    selection_set = feature_list

    for k in range(len(feature_list), 1, -1):
        best_score = -float('inf')
        tmp_set = []

        for current_feature in selection_set:
            subset = [feature for feature in selection_set if feature != current_feature]
            model = LinearRegression()
            model.fit(data[subset], data[target_col])
            y_pred = model.predict(data[subset])
            r2 = r2_score(data[target_col], y_pred)

            n = data[subset].shape[0]  # 样本数量
            p = data[subset].shape[1]  # 特征数量
            adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

            if adjusted_r2 > best_score:
                best_score = adjusted_r2
                tmp_set = subset

        selection_set = tmp_set
        best_features.append({k: {tuple(selection_set): best_score}})

    max_r_squared = -float('inf')  # 初始设为负无穷大
    best_features_combo = None

    for item in best_features:
        for num_features, r_squared_dict in item.items():
            r_squared_value = list(r_squared_dict.values())[0]  # 获取R-squared值
            if r_squared_value > max_r_squared:
                max_r_squared = r_squared_value
                best_features_combo = list(r_squared_dict.keys())[0]  # 获取对应的特征组合

    print("Best R-squared:", max_r_squared)
    print("Best Features:", best_features_combo)

    return best_features_combo








