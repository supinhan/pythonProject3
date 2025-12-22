import pandas as pd

# 读取列头
all_columns = pd.read_csv('数据1.csv', nrows=0).columns.tolist()
# 去掉前6列，取第7列开始到最后的列名
desired_columns = all_columns[6:]
# 直接读取数据CSV文件
df0 = pd.read_csv('数据1.csv', usecols=desired_columns)

df1_t = pd.read_csv('../../溶剂标识符.csv', index_col=0)
df1 = df1_t.T

# 转numpy
arr0 = df0.values.astype(float)
arr1 = df1.values.astype(float)
result_arr = arr0 @ arr1
rows = range(result_arr.shape[0])
cols = range(result_arr.shape[1])

result = pd.DataFrame(result_arr, index=rows, columns=cols)
result.to_csv('output1.csv', index=False)
