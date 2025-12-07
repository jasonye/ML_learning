# 5天Gen AI课程

https://www.kaggle.com/whitepaper-foundational-llm-and-text-generation

https://drive.google.com/file/d/1rYu-mIcsTrAeCuH-xHPofrI1i1qNVzqO/view?pli=1

课程内容

1. Foundational models & Prompt Engineering：
    研究LLMs的发展史、介绍transformers、fine-tuning和inference acceleration技术，同时介绍prompt engineering 

2. Embeddings and Vector Stores/Databases
    介绍embedding方法、vector查找算法、LLM的应用

3. Generative AI Agents
    介绍AI Agents的原理和核心组件、以及组件间的交互过程

4. Domain-Specific LLMs
    特殊领域LLM的介绍，比如SecLM、MedPaLM

5. MLOps for Generative AI

# Foundational models & Prompt Engineering

应用领域：语言翻译、代码生成和不全、文本生成、文本分类、问答和其他。

核心问题：

* 大预言模型的原理是什么？怎么工作的？
  
  主要讲解transformer架构、和发展历史、《Attention is all you need》论文、最近的模型：Gemini、Google的大预言模型能力、training、fine-tuning技术、提高生成速度的方法等。

### LLM模型

通过深度神经网络实现和大量的文本数据训练。介绍Transformer架构和最初的《Attention is all you need》论文。

1. Transformer
   
   Transformer架构之前，使用RNNs架构，有个缺点：The sequential nature of RNNs makes them compute-intensive and hard to parallelize during training (though  recent work in state space modeling is attempting to overcome these challenges).
* Transformer和RNNs机制的优缺点比较：    

            优势： Transformers,on the other hand, are a type of neural network that can process sequences of tokens in parallel thanks to the self-attention mechanism.

            Transformer所使用的self-attention机制支持并行计算，限制了文本的长度。而RNN不限制文本长度，但是会出现梯度下降的问题。

2017年 transformer架构在Google开发出来，用于翻译模型。 最早用在语言翻译中，包含encoder和decoder两个组件，encoder用来从一个向量空间转化为representation、然后用decoder翻译为输出。

1.1 Input preparation和embedding

* Normalization(Optional)：去除掉冗余的空格、accents等
* Tokenization：文本token化、mapping to integer token IDs from vocabulary
* Embedding：转化token ID到对应的高纬度vector、
* Positional Encoding：添加token的位置信息等 

1.2 Multi-head attention
1.2.1 理解self-attention技术

1. Creating queries、keys、and values
    每个input embedding乘以3个学习的权重矩阵Wq、Wk、Wv，产生query、key和value向量。 

2. calculating scores

3. Normalization

4. Weighted values

1.2.2 Multi-head attention: power in diversity

2. 理解self-attention

### 

### Transfer的发展历史

* 

* BERT

* Gemini

* LLaMa

* OpenAI O1

* DeepSeek

### Fine-tuning LLMs

### 使用LLM

1. Prompt engineering

### 大预言模型的应用

* 代码生存和数学、
* 机器翻译
* 文本Summarization
* 机器人
* 文本分类和分析等

## Embeddings and Vector Stores/Databases

## Generative AI Agents

## Domain-Specific LLMs

## MLOps for Generative AI