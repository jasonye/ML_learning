# Python学习笔记

[toc]

官方文档：[The Python Language Reference &mdash; Python 3.4.10 documentation](https://docs.python.org/3.4/reference/index.html)

安装命令

```bash
brew install python
```

## 虚拟环境的使用

查看虚拟环境命令：

```bash
python3 -m venv -h


python3 -m venv python_env     // 创建虚拟目录

// 激活开发环境很重要
source ~/python_env/bin/activate // 激活开发环境，多个不同环境都可以激活


// 这里做很多事情

deactivate // 注销环境、

// do other things
pip3 install flask // 安装相关包manman

# 查看当前安装的包：
python -m pip freeze
```

## 数据结构

### 变量的基本操作

```
a // b   Floor division    Quotient of a and b, removing fractional parts
a ** b     Exponentiation    a raised to the power of b
```

### 列表list、元组、和字典

shoplist = ['apple', 'mango', 'carrot', 'banana']

* list 的方法append
  域 mylist.field
  元组：不能修改元组。zoo = ("wolf",'elephant','penguin');
  元组可以嵌套： new_zoo = ('monkey','dolphin',zoo)
  注意：此时new_zoo =('monkey','dolphin','("wolf",'elephant','penguin')')

new_zoo[2][2] = 'penguin'
new = (1 , ) 单个元素的元组。
注意：列表中的列表、列表中的元组、元组中的元组不会打散。永远是以一个整体作为对象存储。

打印：
print '%s is years old'% age
注意：print 字符串后面没有 ， 号，age前必须加%号

只能用不可变的对象作为字典的键，但可以用可变的对象作为字典的值。
d = {key1:value1, key2:value2 }
has_key

索引操作符、切片操作符 [1:] 注意，数是可选的，冒号是必须有的。

注意：列表的赋值语句不创建拷贝。你得使用切片操作符来创见序列的拷贝。

字符串只是 str 类的对象，具有方法。
startwith 方法 find方法。join方法

* time 和 os 模块
  time.strftime

## strings

```python
isdigit 判断是否数字

len 字符串长度
```

## List

**List comprehensions** are one of Python's most beloved and unique features. 

```python
squares = [n**2 for n in range(10)]
squares
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

还有：

# Conditionals on iterable
num1 = [5,10,15]
num2 = [i**2 if i == 10 else i-5 if i < 7 else i+5 for i in num1]
print(num2)

这个表达式是一个条件表达式，它实际上可以分解为：
如果i等于10，则取i**2
否则，如果i<7，则取i-5
否则，取i+5

但是注意，这个条件表达式是嵌套的，可以写为：
条件1: i == 10 -> i**2
条件2: i < 7 -> i-5
否则: i+5
```


```python
def elementwise_greater_than(L, thresh):
    """Return a list with the same length as L, where the value at index i is 
    True if L[i] is greater than thresh, and False otherwise.

    >>> elementwise_greater_than([1, 2, 3, 4], 2)
    [False, False, True, True]
    """
    result = []
    for i in L:
        if i > thresh:
            result.append(True)
        else:
            result.append(False)
    return result


for i, fruit in enumerate(fruits):
    print(f"Index: {i}, Fruit: {fruit}")
```

### 循环+if条件

```python
short_planets = [planet for planet in planets if len(planet) < 6]
short_planets
['Venus', 'Earth', 'Mars']


## str.upper() returns an all-caps version of a string
loud_short_planets = [planet.upper() + '!' for planet in planets if len(planet) < 6]
loud_short_planets
```

```python
multiplicands = (2, 2, 2, 3, 3, 5)
product = 1
for mult in multiplicands:
    product = product * mult
product

def elementwise_greater_than(L, thresh):
    """Return a list with the same length as L, where the value at index i is 
    True if L[i] is greater than thresh, and False otherwise.

    >>> elementwise_greater_than([1, 2, 3, 4], 2)
    [False, False, True, True]
    """
    result = []
    for i in L:
        if i > thresh:
            result.append(True)
        else:
            result.append(False)
    return result

def elementwise_greater_than(L, thresh):
    return [ele > thresh for ele in L]    
```

## Operator overloading

参考文档：https://www.kaggle.com/code/colinmorris/working-with-external-libraries

numpy的Array可以重载运算符。

```python
[3, 4, 1, 2, 2, 1] + 10 // 这是失败的，但是下面：


import numpy
# Roll 10 dice
rolls = numpy.random.randint(low=1, high=6, size=10)
rolls

rolls + 10  // numpy重载了操作符号
array([13, 14, 13, 14, 15, 15, 12, 11, 13, 13])
```

## 函数

1. help函数：用于查看build-in函数怎么使用，比如：

```python
help(round)

// round函数的不同参数的用法，分别为小数点前后取整数
print(round(2342345.8988545435435, 3))
print(round(2342345.8988545435435, 2))
print(round(2342345.8988545435435, 1))
print(round(2342345.8988545435435, -1))
print(round(2342345.8988545435435, -2))
print(round(2342345.8988545435435, -3))

输出：
2342345.899
2342345.9
2342345.9
2342350.0
2342300.0
2342000.0
```

2. Functions Applied to Functions¶

Here's something that's powerful, though it can feel very abstract at first. You can supply functions as arguments to other functions. Some example may make this clearer:px

```python
def mult_by_five(x):
    return 5 * x

def call(fn, arg):
    """Call fn on arg"""
    return fn(arg)

def squared_call(fn, arg):
    """Call fn on the result of calling fn on arg"""
    return fn(fn(arg))

print(
    call(mult_by_five, 1),
    squared_call(mult_by_five, 1), 
    sep='\n', # '\n' is the newline character - it starts a new line
)
```

输出：
5
25

```python
// 函数和条件表达式
def sign(arg):
    if arg < 0:
        return -1 
    elif arg>0:
        return 1
    else: 
        return 0 
```

// 条件表达式的使用 
print("Splitting", total_candies, "candy" if total_candies == 1 else "candies")   

2. 嵌套函数

```python
#nested function
def square():
    """ return square of value """
    def add():
        """ add two local variable """
        x = 2
        y = 3
        z = x + y
        return z
    return add()**2
print(square())   
```

3. 函数参数

DEFAULT and FLEXIBLE ARGUMENTS¶

Default argument example:
```python 
def f(a, b=1):
  """ b = 1 is default argument"""
```

Flexible argument example:
```python
def f(*args):
 """ *args can be one or more"""

def f(** kwargs)
 """ **kwargs is a dictionary"""
```

### 实践

```python

# Get number of unique entries in each column with categorical data
object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))
d = dict(zip(object_cols, object_nunique))

# Print number of unique entries by column, in ascending order
sorted(d.items(), key=lambda x: x[1])

```
输出：

```python

[('Street', 2),
 ('Utilities', 2),
 ('CentralAir', 2),
 ('LandSlope', 3),
 ('PavedDrive', 3),
 ('LotShape', 4),
 ('LandContour', 4),
 ('ExterQual', 4),
 ('KitchenQual', 4),
 ('MSZoning', 5)]
```

5. Lambda函数

```python
# lambda function
square = lambda x: x**2     # where x is name of argument
print(square(4))
tot = lambda x,y,z: x+y+z   # where x,y,z are names of arguments
print(tot(1,2,3))
```

6. 匿名函数
ANONYMOUS FUNCTİON¶
Like lambda function but it can take more than one arguments.

map(func,seq) : applies a function to all the items in a list

```python
number_list = [1,2,3]
y = map(lambda x:x**2,number_list)
print(list(y))
[1, 4, 9]
```

7. zip 和 unzip的使用
```python 
# zip example
list1 = [1,2,3,4]
list2 = [5,6,7,8]
z = zip(list1,list2)
print(z)
z_list = list(z)
print(z_list)
```

<zip object at 0x7f6f768ece08>
[(1, 5), (2, 6), (3, 7), (4, 8)]

```python 
un_zip = zip(*z_list)
un_list1,un_list2 = list(un_zip) # unzip returns tuple
print(un_list1)
print(un_list2)
print(type(un_list2))
```

8. ITERATORS
iterable is an object that can return an iterator
iterable: an object with an associated iter() method
example: list, strings and dictionaries
iterator: produces next value with next() method

```python
# iteration example
name = "ronaldo"
it = iter(name)
print(next(it))    # print next iteration
print(*it)         # print remaining iteration
```

## 类

域 ---类的变量 和 对象的变量 
类的变量： 相当于C++的静态成员
对象的变量： 每个对象都所拥有。

```python
class Person：
    population = 0;
    def __init__(self, name):
        self.name = name;                   //这里是对象变量
        Person.population +=1;                 //这里是类变量
```

如果某个变量只想在类或者对象中使用，就应该以单下划线前缀。

### 继承

```python
class SchoolMember：

    def __init__(self, name, age):
        self.name = name;
        self.age = age;
        print '(Initialized SchoolMember:%s)' %self.name;

    def tell(self):
        print "Name:%s" Age:"%s" %(self.name, self.age)


class Teacher(SchoolMember):
    def __init__(self, name, age, salary):
        SchoolMember.__init__(self, name, age)
        self.salary = salary
        print '(Initialized teacher:%s)'%self.name

    def tell(self):
        SchoolMember.tell(self)
        print 'Salary:"%d"' % self.salary
```

# 异步IO
Python3.4之后引入asyncio标准库，直接内置对异步IO的支持。

```python 
# coding: utf-8

import asyncio
import threading
import sys

# 检查Python版本
if sys.version_info < (3, 5):
    print("错误: 此代码需要Python 3.5或更高版本。")
    sys.exit(1)

async def hello(name):
    # 打印name和当前线程:
    print(f"Hello {name}! ({threading.current_thread().name})")
    # 异步调用asyncio.sleep(1):
    await asyncio.sleep(1)
    print(f"Hello {name} again! ({threading.current_thread().name})")
    return name


async def main():
    L = await asyncio.gather(hello("Bob"), hello("Alice"))
    print(L)

if __name__ == "__main__":
    asyncio.run(main())
```

执行结果：
Hello Bob! (<function current_thread at 0x10387d260>)
Hello Alice! (<function current_thread at 0x10387d260>)
(等待约1秒)
Hello Bob again! (<function current_thread at 0x10387d260>)
Hello Alice again! (<function current_thread at 0x10387d260>)
['Bob', 'Alice']


asyncio可以实现单线程并发IO操作。如果仅用在客户端，发挥的威力不大。如果把asyncio用在服务器端，例如Web服务器，由于HTTP连接就是IO操作，因此可以用单线程+async函数实现多用户的高并发支持。

asyncio实现了TCP、UDP、SSL等协议，aiohttp则是基于asyncio实现的HTTP框架。



# 机器学习库/工具的使用

## Numpy  --- 科学计算数据处理

NumPy is the fundamental package for scientific computing in Python. It is a Python library that provides a multidimensional array object, various derived objects (such as masked arrays and matrices), and an assortment of routines for fast operations on arrays, including mathematical, logical, shape manipulation, sorting, selecting, I/O, discrete Fourier transforms, basic linear algebra, basic statistical operations, random simulation and much more.

多维数组操作、

参考文档： https://numpy.org/doc/1.26/user/basics.rec.html

### 常见错误：

1. 执行错误ModuleNotFoundError: No module named 'numpy'

## scikit-learn---包含各种类型的机器学习库

### 构建和使用模型步骤

- Define: What type of model will it be? A decision tree? Some other type of model? Some other parameters of the model type are specified too.
- Fit: Capture patterns from provided data. This is the heart of modeling.
- Predict: Just what it sounds like
- Evaluate: Determine how accurate the model's predictions are.

基本步骤：

1. 加载数据：进行数据处理和清晰、查看数据是否符合要求、
2. 模型训练：
   选择合适的模型（分类、回归 or 决策树等）、把数据拆分为训练数据和评估数据
3. 模型预测：
4. 评估模型预测的效果是否符合需求
   常见的评估模型好坏的指标：Mean Absolute Error (also called MAE)、

```
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))
```

注意事项:

1. It’s crucial to randomize the split and ensure the model doesn’t overfit to a specific data subset, which can be achieved by setting shuffle=True.

## Matplotlib  图形显示

Visualization with Python

介绍文档：[Matplotlib Getting Started](https://www.w3schools.com/python/matplotlib_getting_started.asp)

参考：https://matplotlib.org/

Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python. Matplotlib makes easy things easy and hard things possible.

```python
# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.Speed.plot(kind = 'line', color = 'g',label = 'Speed',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.Defense.plot(color = 'r',label = 'Defense',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot 把线标签显示在右上角
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
```

配置显示的颜色条、线条等各种显示。

[Matplotlib Scatter](https://www.w3schools.com/python/matplotlib_scatter.asp)

3. Histogram: 直方图的设置等
bins: number of bins
range(tuble): min and max values of bins
normed(boolean): normalize or not
cumulative(boolean): compute cumulative distribution

## seaborn库

https://seaborn.pydata.org/

Python图形化展示所使用的库。

## TensorFlow

[TensorFlow](https://www.tensorflow.org/) is a library for developing and training machine learning models.

## [Keras](https://keras.io/)

Keras is an API built on top of TensorFlow designed for neural networks and deep learning.
