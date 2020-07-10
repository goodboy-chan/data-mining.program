import pandas as pd

def loader(): #读取数据
    train = pd.read_csv("../zhengqi_train.txt", sep='\t')
    test = pd.read_csv("../zhengqi_test.txt", sep='\t')
    train = pd.DataFrame(train)
    test = pd.DataFrame(test)
    return train, test

if __name__ == "__main__":
    Train, Test = loader()
    # 统计是否有缺失值，结果无缺失值
    print("训练集各列存在缺失值情况：")
    print(Train.isnull().any())
    print("测试集各列存在缺失值情况：")
    print(Test.isnull().any())


