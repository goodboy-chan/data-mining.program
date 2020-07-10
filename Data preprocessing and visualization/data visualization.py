import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# 比较训练数据和测试数据不同特征值之间的数据分布


if __name__ == "__main__":
    train_data = pd.read_csv('../zhengqi_train.txt', sep='\t')
    test_data = pd.read_csv('../zhengqi_test.txt', sep='\t')
    '''
    for col in test_data.columns:
        seaborn.distplot(train_data[col])
        seaborn.distplot(test_data[col])
        plt.show()
    '''
    # 训练数据分布
    plt.figure(figsize=(18, 18))

    for column_index, column in enumerate(train_data.columns):
        plt.subplot(10, 4, column_index + 1)
        g = sns.kdeplot(train_data[column])
        g.set_xlabel(column)
        g.set_ylabel('Frequency')
    plt.show()
    # 测试数据分布
    for column_index, column in enumerate(test_data.columns):
        plt.subplot(10, 4, column_index + 1)
        g = sns.kdeplot(test_data[column])
        g.set_xlabel(column)
        g.set_ylabel('Frequency')
    plt.show()
    # 特征相关性分布
    plt.figure(figsize=(20, 16))
    colnm = train_data.columns.tolist()
    mcorr = train_data[colnm].corr(method="spearman")
    mask = np.zeros_like(mcorr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    g = sns.heatmap(mcorr, mask=mask, cmap=cmap, square=True, annot=True, fmt='0.2f')
    plt.show()
