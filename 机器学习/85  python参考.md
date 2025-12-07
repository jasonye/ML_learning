# Zip函数

`zip()` 函数是 Python 中一个强大的内置函数，用于将多个可迭代对象（如列表、元组、字符串等）的元素**按索引位置配对**，生成一个元组迭代器。以下是详细说明和示例：

---

### **作用**

1. **并行迭代**：同时遍历多个序列，每次取相同索引位置的元素组合成元组。
2. **长度处理**：当输入的可迭代对象长度不同时，以**最短**的序列为准截断结果。
3. **解压数据**：结合 `*` 操作符可将已压缩的数据解压为独立序列。

---

### **基本语法**

```python
zip(iterable1, iterable2, ..., iterableN)
```

---

### **示例代码**

#### 1. 基础配对（列表+元组）

```python
names = ["Alice", "Bob", "Charlie"]
ages = (25, 30, 28)

# 将两个序列配对
paired = list(zip(names, ages))
print(paired)
```

**输出**：

```
[('Alice', 25), ('Bob', 30), ('Charlie', 28)]
```

#### 2. 不同长度序列（自动截断）

```python
letters = ['a', 'b', 'c']
numbers = [1, 2]
symbols = ['!', '@', '#', '$']

# 以最短序列(numbers)为准截断
result = list(zip(letters, numbers, symbols))
print(result)
```

**输出**：

```
[('a', 1, '!'), ('b', 2, '@')]  # 只保留前2个元素
```

#### 3. 遍历配对结果（高效循环）

```python
for name, score in zip(["Tom", "Jerry"], [85, 92]):
    print(f"{name}: {score}分")
```

**输出**：

```
Tom: 85分
Jerry: 92分
```

#### 4. 数据转置（矩阵行列互换）

```python
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# 使用 zip(*matrix) 实现转置
transposed = list(zip(*matrix))
print(transposed)
```

**输出**：

```
[(1, 4, 7), (2, 5, 8), (3, 6, 9)]
```

#### 5. 解压数据（还原独立序列）

```python
zipped = [('Apple', 3), ('Banana', 6), ('Cherry', 9)]
fruits, counts = zip(*zipped)  # 解压为两个独立元组

print(f"水果列表: {fruits}")
print(f"数量列表: {counts}")
```

**输出**：

```
水果列表: ('Apple', 'Banana', 'Cherry')
数量列表: (3, 6, 9)
```

---

### **注意事项**

- **返回值类型**：Python 3 中 `zip()` 返回迭代器（节省内存），如需列表需显式转换 `list(zip(...))`。
- **空输入处理**：无参数时返回空迭代器，如 `list(zip()) → []`。
- **长度一致要求**：若需保留最长序列的尾部元素，可使用 `itertools.zip_longest()`。

通过灵活运用 `zip()`，可简化多序列操作、数据转换等任务！
