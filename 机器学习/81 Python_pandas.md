# Pandas库

# the most popular Python library for data analysis.

教程文档：[Pandas Tutorial](https://www.w3schools.com/python/pandas/default.asp)

包含处理数据、进行数据分析的主要的库

参考文档：[https://pandas.pydata.org/](https://pandas.pydata.org/)

查看各种API的使用。

参考课程：[Learn Pandas Tutorials | Kaggle](https://www.kaggle.com/learn/pandas)

安装教程：

```shell
pip3 install pandas
```

### 数据类型

1. DataFrame：表示一个表格
2. Series： Pandas序列

### 使用用例

```python
import pandas as pd

# 数据集
pd.DataFrame({'Yes': [50, 21], 'No': [131, 2]}) // Key是表格的Column

pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 
              'Sue': ['Pretty good.', 'Bland.']},
             index=['Product A', 'Product B']) // Index


# Series
pd.Series([1,2,3,4,5], index=['2015', '2016 Sales'], name = 'Product A')


# Read File
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", ) // 包含很多参数，可以用来指定index索引等

wine_reviews.to_csv("file_path.csv")

# print a summary of the data in Melbourne data
wine_reviews.describe() // 用户查看数据的概述

wine_reviews.shape // 查看数据的行数、数据模型列等。

wine_reviews.head()

feature_names = ["LotArea",
    "YearBuilt",
    "1stFlrSF",
    "2ndFlrSF",
    "FullBath",
    "BedroomAbvGr",
    "TotRmsAbvGrd"]
```

# 选择多行数据

```python
X = home_data[feature_names]

// 查看前几行数据
X.head()

# 选择某几行
reviews.iloc[[0, 1, 2], 0] // 按行查找数据,只能选择数字行、列
reviews.iloc[[1,2,3,5,8],:]

# 按列选择数据
reviews.loc[:, ['taster_name', 'taster_twitter_handle', 'points']] // 按列查找数据
```

### Conditional selection
可以执行逻辑判断、比较（控制流）、filtering
```python
reviews.loc[reviews.country == 'Italy'] // 获取数据
reviews.loc[(reviews.country == 'Italy') | (reviews.points >= 90)]
reviews.loc[reviews.country.isin(['Italy', 'France'])]
reviews.loc[reviews.price.notnull()] // isnull
```

Both loc and iloc are row-first, column-second. This is the opposite of what we do in native Python, which is column-first, row-second.

iloc uses the Python stdlib indexing scheme, where the first element of the range is included and the last one excluded. So 0:10 will select entries 0,...,9. loc, meanwhile, indexes inclusively. So 0:10 will select entries 0,...,10.

reviews.points.describe() // This method generates a high-level summary of the attributes of the given column. It is type-aware, meaning that its output changes based on the data type of the input.

```python
reviews.taster_name.unique() //  查看唯一的值
reviews.taster_name.value_counts() // 查看各个值的count



review_points_mean = reviews.points.mean()
reviews.points.map(lambda p: p - review_points_mean) // 函数值转换、

def remean_points(row):
    row.points = row.points - review_points_mean
    return row

reviews.apply(remean_points, axis='columns')


countries_reviewed.sort_values(by='len', ascending=False)


reviews.points.dtype // 获取类型
reviews.points.astype('float64') // 类型转换

// isnull\notnull
reviews.region_2.fillna("Unknown") // 填充值

replace

reviews.rename(index={0: 'firstEntry', 1: 'secondEntry'})


// 数据合并
canadian_youtube = pd.read_csv("../input/youtube-new/CAvideos.csv")
british_youtube = pd.read_csv("../input/youtube-new/GBvideos.csv")

pd.concat([canadian_youtube, british_youtube])
```

## RESAMPLING PANDAS TIME SERIES 改变数据采样粒度
Resampling: statistical method over different time intervals
Needs string to specify frequency like "M" = month or "A" = year
Downsampling: reduce date time rows to slower frequency like from daily to weekly
Upsampling: increase date time rows to faster frequency like from daily to hourly
Interpolate: Interpolate values according to different methods like ‘linear’, ‘time’ or index’
https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.interpolate.html