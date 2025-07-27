# 人工智能

Updated 2253 GMT+8 Jul 27 2025

2025 summer, Complied by Hongfei Yan



https://github.com/GMyhf/2025fall-cs201/





# 第一部分 人工智能原理概述

## 1.1 人工智能的概念与发展历史

### 什么是人工智能？

人工智能（AI）是使计算机系统执行通常需要人类智能的任务的技术。常见的人工智能应用包括语音识别、图像识别、推荐系统等。

### 图灵测试

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20250727133958507.png" alt="image-20250727133958507" style="zoom:50%;" />

艾伦·图灵于1950年提出了“图灵测试”作为判断机器是否具有智能的方法。在图灵测试中，测试者通过与机器和另一个人进行对话，如果无法区分机器和人类的回答，机器则被认为具有人类般的智能。

```text
图灵测试原理：如果机器与人类对话无法区分，则机器具有智能。
```

### 达特茅斯会议与人工智能的诞生

1956年，达特茅斯会议正式提出并确定了“人工智能”这一术语，并明确了AI研究的目标：开发能够模仿人类学习、推理和解决问题能力的计算机程序。这标志着人工智能学科的诞生。

**发展历程**

1. **1950s-1960s**：人工智能的早期探索，重点是基于规则的智能系统。
2. **1970s-1980s**：符号主义流派的兴起，专家系统和推理方法得到应用。
3. **1990s至今**：连接主义流派的崛起，深度学习和神经网络的突破。



## 1.2 人工智能的三大学派

人工智能的研究有多个流派，其中三大学派包括符号主义（Symbolism）、连接主义（Connectionism） 和 行为主义（Behaviorism）。



> https://zhuanlan.zhihu.com/p/443257981

### 1. 符号主义（Symbolism / Symbolic AI）

- **核心观点**：智能源于对知识的符号表示和操作（规则推理）。

- **代表方法**：专家系统、搜索算法、逻辑推理（如一阶逻辑）、规划系统。

- **代表人物**：艾伦·纽厄尔（Allen Newell）、赫伯特·西蒙（Herbert Simon）、约翰·麦卡锡（John McCarthy）等。

  > 约翰·麦卡锡：提出了LISP编程语言（后来成为人工智能领域的主要语言）并倡导符号主义AI。

  >  符号主义也称为“强人工智能”的早期代表，强调形式化知识和推理。

**优点**：

- 高可解释性：符号主义系统有明确的规则，易于理解。
- 专用性：适用于数学推理、棋类游戏等明确规则的任务。

**缺点**：

- 学习能力弱：难以处理模糊信息，且需要大量手动编写规则。

**代表性成果**

- **专家系统**：如XCON系统，帮助DEC公司配置计算机系统。

  > 20世纪80年代卡内基梅隆大学为数字设备公司(DEC)设计的名为XCON的专家系统用于协助配置DEC公司复杂的VAX计算机系统。它能够根据客户订单中的需求（如处理器、存储、外围设备等），自动生成硬件组装和连接方案，并确保所有部件的兼容性。在1986年之前，它每年为公司省下四千万美元，XCON的成功标志着专家系统在企业实际应用中的突破，对后来的商业AI系统开发有重要影响。

- **深蓝（Deep Blue）**：IBM开发的国际象棋AI，1997年战胜卡斯帕罗夫。



### 2. 连接主义（Connectionism）

连接主义强调<mark>通过神经网络模仿人脑的结构与功能</mark>。智能来源于神经元之间的连接与信号传递。

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20250727141054606.png" alt="image-20250727141054606" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20250727141109558.png" alt="image-20250727141109558" style="zoom:50%;" />

- **核心观点**：智能是神经元之间连接和激活的结果，强调从数据中学习。

- **代表方法**：人工神经网络（ANN）、深度学习（CNN、RNN、Transformer等）。

- **代表人物**：杰弗里·辛顿（Geoffrey Hinton）、约翰·霍普菲尔德、伊利亚·苏茨基弗等。Yann LeCun、David Rumelhart。

  > 约翰·霍普菲尔德（John Hopfield），2024年与Hinton共获诺贝尔物理学奖。1982年和1984年发表两篇重要论文，提出提出Hopfield网络，为后来深度学习的崛起铺平了道路。Hopfield网络，被广泛应用于图像修复、模式识别等领域。

- **应用**：图像识别、自然语言处理、自动驾驶等。

- **代表系统**：AlphaGo、GPT、深度视觉系统等。

  > 是近年来人工智能突破（尤其是深度学习）的主导力量，强调“弱规则，强数据”。

#### 代表性成果

- **感知机（1958年）**：一个简单的神经网络模型，能进行二分类任务。

- **反向传播算法（1986年）**：使多层神经网络得以有效训练，成为深度学习的基石。

  > Geoffrey Hinton、David Rumelhart 和 Ronald Williams 提出了反向传播算法（Backpropagation）。使得多层神经网络得以有效训练。成为深度学习的基石之一，极大推动了神经网络的研究与应用。

#### 示例：简单神经网络分类 Iris 数据集（PyTorch）

```python
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 数据准备
X, y = load_iris(return_X_y=True)
X = StandardScaler().fit_transform(X)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)

# 构建神经网络
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 10),
            nn.ReLU(),
            nn.Linear(10, 3)
        )
    def forward(self, x):
        return self.fc(x)

model = Net()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练
for epoch in range(100):
    out = model(train_x)
    loss = loss_fn(out, train_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 测试
acc = (model(test_x).argmax(1) == test_y).float().mean()
print("Test Accuracy:", acc.item())
```



> /Users/hfyan/miniconda3/bin/python /Users/hfyan/Library/Application Support/JetBrains/PyCharm2024.1/scratches/scratch_1894.py 
> Test Accuracy: 0.9666666388511658



### 3. 行为主义（Behaviorism / Actionism / Cybernetics）

行为主义的核心思想是<mark>智能体的行为是对外界刺激的反应</mark>。智能体通过与环境的交互，并根据反馈调整行为，从而实现自我学习。

- **核心观点**：智能体无需具备内部知识结构，只需通过与环境交互来学习行为。

- **代表人物**：罗德尼·布鲁克斯（Rodney Brooks）、戴密斯·哈萨比斯等。B.F. Skinner（心理学）、Norbert Wiener（控制论）

  > 罗德尼·布鲁克斯(Rodney A. Brooks)，提出了智能行为的“感知-动作”模式，发明六足行走机器人。英国人戴密斯·哈萨比斯（Demis Hassabis）的AlphaGo击败人类。

- **应用**：强化学习（例如AlphaGo、自动驾驶等）。

  > 强调“行动优于思考”，对机器人学、智能体研究有重要影响。

#### 强化学习示例

强化学习模型的核心是通过奖励和惩罚来引导智能体的行为。伪代码如下：

```python
# 更新Q值函数
Q[state][action] = Q[state][action] + alpha * (reward + gamma * max(Q[next_state]) - Q[state][action])
```



## 1.3 人工智能三要素：算法、算力、数据

### 1. 算法：人工智能的“灵魂”

AI算法根据任务类型和数据特点分为以下几类：

- **监督学习**（Supervised Learning）通过带标签的数据训练模型，进行分类或回归。
- **无监督学习**（Unsupervised Learning）从无标签数据中挖掘数据结构，常用于聚类。
- **强化学习**：通过奖惩机制优化行为策略，常用于游戏AI、自动驾驶等。

#### 监督学习 vs 无监督学习

```python
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

# 监督学习：分类
clf = LogisticRegression().fit(train_x, train_y)

# 无监督学习：聚类
kmeans = KMeans(n_clusters=3).fit(X)
```



完整示例：监督学习 + 无监督学习（Iris 数据集）

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. 加载数据
iris = load_iris()
X = iris.data
y = iris.target

# 2. 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 划分训练集和测试集（用于监督学习）
train_x, test_x, train_y, test_y = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 4. 监督学习：逻辑回归分类
clf = LogisticRegression(max_iter=200)
clf.fit(train_x, train_y)
pred = clf.predict(test_x)
print("Logistic Regression Accuracy:", accuracy_score(test_y, pred))

# 5. 无监督学习：KMeans 聚类（聚成3类）
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# 6. 可视化聚类（降维到二维）
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 5))

# 聚类结果
plt.subplot(1, 2, 1)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=clusters, cmap='viridis', s=50)
plt.title("KMeans Clustering (unsupervised)")

# 原始标签
plt.subplot(1, 2, 2)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='Set1', s=50)
plt.title("Ground Truth Labels (supervised)")

plt.show()

```

> 使用 `LogisticRegression` 训练一个有监督分类器，并输出测试集准确率；
>
> 使用 `KMeans` 进行无监督聚类；
>
> 使用 PCA 将 4 维数据降维为 2 维，以便可视化聚类结果和真实标签；
>
> 可清晰看到聚类与真实分类的对比效果。



/Users/hfyan/miniconda3/bin/python /Users/hfyan/Library/Application Support/JetBrains/PyCharm2024.1/scratches/scratch_1895.py 
Logistic Regression Accuracy: 1.0



<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20250727143806445.png" alt="image-20250727143806445" style="zoom: 67%;" />



### 2. 算力：人工智能的“引擎”

算力是AI发展的关键因素。现代深度学习模型需要强大的计算资源支持，尤其是GPU和TPU的应用大幅加速了AI训练过程。

#### 测试 GPU 和 MPS（适用于 Mac 用户）

```python
import torch
print("CUDA:", torch.cuda.is_available())  # 检测GPU是否可用
print("MPS:", torch.backends.mps.is_available())  # 检测MPS是否可用
```

### 3. 数据：人工智能的“燃料”

数据是AI模型学习的基础。足够大量且高质量的数据可以推动AI算法的成功应用。

- **数据标注**：包括图像分类、语音识别等领域，标注数据用于监督学习。

  > 2006年李飞飞提出ImageNet计划：创建涵盖数千个物体类别，并且每个类别包含足够数量的图片的数据集，以便训练和验证复杂的计算机视觉算法。
  > 数据来源：通过互联网抓取数千万张图片。
  > 标签标注：利用众包平台，雇佣数千名标注员对数据进行标注。
  > 分类标准：基于 WordNet 的词汇层次结构，设计了数据集的类别层级。
  > 最终成果：2009发布，包含1400万张标注的图片。

- **无标签数据**：无监督学习依赖于无标签数据进行模式识别。

  > 无需人工干预，自然产生：通过传感器、摄像头等各种设备记录，各种网站和应用程序的用户生成……..



## 1.4 人工智能应用前沿

### 1. 智能博弈：AlphaGo 和 AlphaZero

- **AlphaGo**：利用深度卷积神经网络、强化学习和蒙特卡洛树搜索技术，成功战胜了围棋顶尖选手李世石。

  > DeepMind公司的AlphaGo 2016年4:1战胜李世石，2017年3：0战胜柯洁
  >
  > AlphaGo以深度卷积神经网络（CNN）为基础，使用深度学习、强化学习、蒙特卡洛树搜索技术，以人类棋谱进行监督学习。蒙特卡洛树搜索: 前几步路要仔细选择好的，剩下的就随便快速多次乱试，再反馈结果。参数个数百万级别。
  >

- **AlphaZero**：继AlphaGo后，进一步发展，AlphaZero可以从零开始学习，自动掌握不同棋类的策略，且不依赖人类棋谱。

  > 由AlphaGo改进后得到更强的AlphaGo Zero ，Alpha Zero
  >
  > AlphaGo Zero参数2000万
  >
  > AlphaGo Zero从完全随机的棋局开始，通过反复的自对弈进行学习。不需要人类棋谱，自己造数据自己学。经过40多天学习战胜AlphaGo。
  >
  > 2018年的Alpha Zero 并不依赖具体棋类的规则，可以从新给的规则中，通过自我对弈学习出最优策略。通过强化学习，能够适应不同棋类。
  >
  > Alpha Zero通过8小时自对弈学习，战胜AlphaGo Zero ,经过2h学习，战胜顶尖国际象棋AI  Stockfish, 用4h学习，战胜顶尖日本将棋AI。
  >
  > Alpha Zero揭示也许在有些领域放弃人类经验会有更好结果

#### AlphaZero 示例：

```text
通过自对弈学习，AlphaZero超越了传统的棋类程序，掌握了自我优化和策略调整能力。
```

### 2. 自然语言处理：Transformer 与 GPT-3

- **Transformer模型**：基于自注意力机制，能够更高效地处理长序列数据。
- **GPT-3**：通过1750亿个参数，GPT-3能够生成流畅的自然语言文本，应用广泛于文本生成、翻译等领域。

#### GPT 示例：Hugging Face调用



| 模型                                     | 适用语言 | 用途     |
| ---------------------------------------- | -------- | -------- |
| `distilbert-base-cased-distilled-squad`  | 英文     | 英文问答 |
| `uer/roberta-base-chinese-extractive-qa` | 中文     | 中文问答 |

Q: 如何用**浏览器手动下载** `uer/roberta-base-chinese-extractive-qa` 模型，做到完全 **离线部署** 的步骤：

> 🔗 1. 打开模型页面
>
> 浏览器访问：https://huggingface.co/uer/roberta-base-chinese-extractive-qa
>
> ------
>
> 📁 2. 进入 “Files and versions” 页面，手动下载以下几个关键文件：
>
> | 文件名                    | 说明                           |
> | ------------------------- | ------------------------------ |
> | `config.json`             | 模型结构配置                   |
> | `pytorch_model.bin`       | 模型权重（很大，400MB 左右）   |
> | `tokenizer_config.json`   | tokenizer 配置                 |
> | `vocab.txt`               | 中文词表（必需）               |
> | `special_tokens_map.json` | 特殊符号定义（可选但推荐）     |
> | `tokenizer.json`          | tokenizer 的二进制形式（可选） |
>
> 你可以在网页中依次点击这些文件，然后点击右上角 “Download”。
>
> ------
>
> 🗂️ 3. 下载后，把它们放入一个本地文件夹，例如：
>
> ```
> ./models/roberta-chinese-qa/
> ├── config.json
> ├── pytorch_model.bin
> ├── tokenizer_config.json
> ├── vocab.txt
> ├── special_tokens_map.json
> ├── tokenizer.json
> ```





```python
from transformers import pipeline

qa = pipeline(
    "question-answering",
    #model="./models/distilbert-base-cased-distilled-squad",
    model="./models/roberta-chinese-qa",
    tokenizer="./models/roberta-chinese-qa",
    framework="pt"  # 显式要求使用 PyTorch
)

result = qa(
    question="谁是人工智能之父？",
    context="艾伦·图灵是人工智能之父，被誉为计算机科学的奠基人。"
)

print(result)       # 看完整结果
print(result["answer"])  # 输出应为：艾伦·图灵
```

/Users/hfyan/miniconda3/bin/python /Users/hfyan/Code/tmp/script2.py 
{'score': 0.3183244466781616, 'start': 0, 'end': 5, 'answer': '艾伦·图灵'}
艾伦·图灵



> “用一个本地的中文问答模型，在一段文本里提取问题的答案”。
>
> 
>
> ✅ 1. 引入 Hugging Face 的 `pipeline`
>
> ```python
> from transformers import pipeline
> ```
>
> 这是 Hugging Face `transformers` 提供的简洁 API，用于快速构建 NLP 任务管道，例如文本分类、问答、翻译等。
>
> 
>
> ✅ 2. 构造问答任务的 pipeline（使用本地模型）
>
> ```python
> qa = pipeline(
>     "question-answering",
>     #model="./distilbert-base-cased-distilled-squad",
>     model="./roberta-chinese-qa",      # 使用本地中文模型目录
>     framework="pt"                     # 强制使用 PyTorch
> )
> ```
>
> 参数说明：
>
> | 参数                           | 含义                                                         |
> | ------------------------------ | ------------------------------------------------------------ |
> | `"question-answering"`         | 指定任务类型是抽取式问答（extractive QA）                    |
> | `model="./roberta-chinese-qa"` | 指向本地下载的模型文件夹，里面包含 `pytorch_model.bin`、`config.json` 等 |
> | `framework="pt"`               | 显式要求使用 PyTorch，而不是 TensorFlow，防止意外加载 TF 模型引发错误 |
>
> 📝 `tokenizer` 会自动从模型目录中加载，无需单独指定。
>
> ------
>
> ✅ 3. 运行问答推理
>
> ```python
> result = qa(
>     question="谁是人工智能之父？",
>     context="艾伦·图灵是人工智能之父，被誉为计算机科学的奠基人。"
> )
> ```
>
> 说明：
>
> - 输入的 `question` 是用户要问的问题；
> - `context` 是包含答案的上下文文本（模型会在里面查找答案）；
> - 返回的是一个包含 **预测答案位置、内容、置信度** 的字典结构。
>
> 
>
> 🛠 常见补充建议：
>
> - **更复杂 context**：你可以给它更多段落，它会找最有可能的答案；
>
> - **中文精度提高**：你可以尝试 `hfl/chinese-roberta-wwm-ext-large` 之类的模型；
>
>   



# 第二部分 人工智能应用编程

阅读：PyTorch 教程，https://www.runoob.com/pytorch/pytorch-tutorial.html

使用PyTorch实现5个从基础模型到较复杂模型的训练与应用。相关代码及说明文档已整理于 Markdown 文件中，详见项目仓库：https://github.com/GMyhf/2025spring-cs201/tree/main/LLM

1. `0_xor_bp_neural_net_manual`：手动实现反向传播的简单神经网络，用于异或问题。
2. `1_iris_neural_network`：构建并训练用于鸢尾花分类的数据驱动神经网络。
3. `2_mnist_resnet18`：使用 ResNet18 模型对 MNIST 手写数字进行分类。
4. `3_cifar10_resnet18`：将 ResNet18 应用于 CIFAR-10 图像分类任务。
5. `4_tiny_imagenet_resnet50`：基于 ResNet50 模型处理 Tiny ImageNet 图像分类任务。



## 2.1 神经网络中的反向传播

https://www.geeksforgeeks.org/machine-learning/backpropagation-in-neural-network/



反向传播（Back Propagation），又称为“误差的反向传播”，是一种用于训练神经网络的方法。其目标是通过调整网络中的权重（weights）和偏置（biases），来减小模型预测输出与实际输出之间的差异。

它通过迭代方式更新权重和偏置，以最小化损失函数（cost function）。在每一个训练周期（epoch）中，模型会根据误差梯度（error gradient）更新参数，常用的优化算法包括梯度下降（Gradient Descent）或随机梯度下降（SGD）。该算法使用微积分中的链式法则来计算梯度，从而能够有效地穿越复杂的神经网络结构，优化损失函数。

> Back Propagation is also known as "Backward Propagation of Errors" is a method used to train neural network . Its goal is to reduce the difference between the model’s predicted output and the actual output by adjusting the weights and biases in the network.
>
> It works iteratively to adjust weights and bias to minimize the cost function. In each epoch the model adapts these parameters by reducing loss by following the error gradient. It often uses optimization algorithms like **gradient descent** or **stochastic gradient descent**. The algorithm computes the gradient using the chain rule from calculus allowing it to effectively navigate complex layers in the neural network to minimize the cost function.

![Backpropagation-in-Neural-Network-1](https://media.geeksforgeeks.org/wp-content/uploads/20250701163824448467/Backpropagation-in-Neural-Network-1.webp)

<center>Fig(a) A simple illustration of how the backpropagation works by adjustments of weights</center>

<center>图(a)：通过权重调整，简单展示反向传播的工作方式</center>



**反向传播的重要性：**

- **高效的权重更新**：利用链式法则计算损失函数对每个权重的梯度，从而高效地更新参数。
- **良好的扩展性**：适用于多层结构和复杂架构，是深度学习可行的核心算法。
- **自动学习能力**：训练过程自动进行，模型会不断调整自身来优化性能。

> **Back Propagation** plays a critical role in how neural networks improve over time. Here's why:
>
> 1. **Efficient Weight Update**: It computes the gradient of the loss function with respect to each weight using the chain rule making it possible to update weights efficiently.
> 2. **Scalability**: The Back Propagation algorithm scales well to networks with multiple layers and complex architectures making deep learning feasible.
> 3. **Automated Learning**: With Back Propagation the learning process becomes automated and the model can adjust itself to optimize its performance.



### 2.1.1 反向传播算法的工作流程

反向传播算法包括两个主要步骤：**前向传播（Forward Pass）** 和 **反向传播（Backward Pass）**

#### 1. Forward Pass Work前向传播

输入数据从输入层开始，经过带权重的连接传递到隐藏层。例如，一个有两个隐藏层 h1 和 h2 的网络中，h1 的输出作为 h2 的输入。在应用激活函数前，还会加上偏置项。

每一层都会计算输入的加权和（记作 `a`），再通过如 ReLU 等激活函数得到输出 `o`。最终，输出层通常会使用 softmax 激活函数将结果转换为分类概率。

> ### Working of Back Propagation Algorithm
>
> The Back Propagation algorithm involves two main steps: the **Forward Pass** and the **Backward Pass**.
>
> ### 1. Forward Pass Work
>
> In **forward pass** the input data is fed into the input layer. These inputs combined with their respective weights are passed to hidden layers. For example in a network with two hidden layers (h1 and h2) the output from h1 serves as the input to h2. Before applying an activation function, a bias is added to the weighted inputs.
>
> Each hidden layer computes the weighted sum (`a`) of the inputs then applies an activation function like [**ReLU (Rectified Linear Unit)**](https://www.geeksforgeeks.org/deep-learning/relu-activation-function-in-deep-learning/) to obtain the output (`o`). The output is passed to the next layer where an activation function such as [**softmax**](https://www.geeksforgeeks.org/deep-learning/the-role-of-softmax-in-neural-networks-detailed-explanation-and-applications/) converts the weighted outputs into probabilities for classification.

![Backpropagation-in-Neural-Network-2](https://media.geeksforgeeks.org/wp-content/uploads/20250701163954688803/Backpropagation-in-Neural-Network-2.webp)

<center>The forward pass using weights and biases</center>

> h1,h2，表示隐藏层的两个神经元



#### 2. Backward Pass反向传播

反向传播阶段会将预测输出与实际输出的误差向后传递，并调整每一层的权重和偏置。常见的误差计算方法是**均方误差（MSE）**：

$MSE = (\text{Predicted Output} − \text{Actual Output})^2$

在误差计算之后，通过链式法则计算梯度，这些梯度用于指导权重和偏置的更新方向和幅度。反向传播过程是逐层执行的，<mark>激活函数的导数在梯度计算中起着关键作用</mark>。



**反向传播的示例：机器学习中的案例**

假设我们使用 sigmoid 激活函数，目标输出为 0.5，学习率为 1。

> ### 2. Backward Pass
>
> In the backward pass the error (the difference between the predicted and actual output) is propagated back through the network to adjust the weights and biases. One common method for error calculation is the [**Mean Squared Error (MSE)**](https://www.geeksforgeeks.org/maths/mean-squared-error/) given by:
>
> $MSE = (\text{Predicted Output} − \text{Actual Output})^2$
>
> Once the error is calculated the network adjusts weights using **gradients** which are computed with the chain rule. These gradients indicate how much each weight and bias should be adjusted to minimize the error in the next iteration. The backward pass continues layer by layer ensuring that the network learns and improves its performance. The activation function through its derivative plays a crucial role in computing these gradients during Back Propagation.
>
> 
>
> ## Example of Back Propagation in Machine Learning
>
> Let’s walk through an example of Back Propagation in machine learning. Assume the neurons use the sigmoid activation function for the forward and backward pass. The target output is 0.5 and the learning rate is 1.

![Backpropagation-in-Neural-Network-3](https://media.geeksforgeeks.org/wp-content/uploads/20250701164029130520/Backpropagation-in-Neural-Network-3.webp)

<center>Example (1) of backpropagation sum</center>



### 2.1.2 前向传播Forward Propagation

#### 1. Initial Calculation初始计算

The weighted sum at each node is calculated using:

> $a_j=\sum(w_{i,j}∗x_i)$

Where,

- $a_j$ is the weighted sum of all the inputs and weights at each node
- $w_{i,j}$ represents the weights between the $i^{th}$ input and the $j^{th}$ neuron
- $x_i$ represents the value of the $i^{th}$ input

`O (output):`After applying the activation function to `a`, we get the output of the neuron:

> $o_j = \text{activation function}(a_j)$

#### 2. Sigmoid Function

The sigmoid function returns a value between 0 and 1, introducing non-linearity into the model.

> $y_j = \frac{1}{1+e^{−a_j}}$ 

![Backpropagation-in-Neural-Network-4](https://media.geeksforgeeks.org/wp-content/uploads/20250701164114106895/Backpropagation-in-Neural-Network-4.webp)

<center>To find the outputs of y3, y4 and y5</center>



#### 3. Computing Outputs输出计算

h1 节点：
$$
a_1 = (w_{1,1} \times x_1) + (w_{2,1} \times x_2)
$$
$$
a_1 = (0.2 \times 0.35) + (0.2 \times 0.7) = 0.21
$$

计算完 $ a_1 $ 后，我们可以继续计算 $ y_3 $ 的值：

$$
y_j = F(a_j) = \frac{1}{1 + e^{-a_1}}
$$
$$
y_3 = F(0.21) = \frac{1}{1 + e^{-0.21}} = 0.56
$$



h2 节点：
$$
a_2 = (w_{1,2} \times x_1) + (w_{2,2} \times x_2) = (0.3 \times 0.35) + (0.3 \times 0.7) = 0.315
$$
$$
y_4 = F(0.315) = \frac{1}{1 + e^{-0.315}} = 0.578
$$



输出节点 O3：
$$
a_3 = (w_{1,3} \times y_3) + (w_{2,3} \times y_4) = (0.3 \times 0.56) + (0.9 \times 0.58) = 0.702
$$
$$
y_5 = F(0.702) = \frac{1}{1 + e^{-0.702}} = 0.67
$$



> At h1 node
>
> Once we calculated the a1 value, we can now proceed to find the y3 value:
>
> Similarly find the values of y4 at h2 and y5 at O3



![Backpropagation-in-Neural-Network-5](https://media.geeksforgeeks.org/wp-content/uploads/20250701164956768059/Backpropagation-in-Neural-Network-5.webp)

<center>Values of y3, y4 and y5</center>



#### 4. Error Calculation误差计算

Our actual output is 0.5 but we obtained 0.67**.** To calculate the error we can use the below formula:

> $Error_j=y_{target}−y_5$ 

=> 0.5−0.67=−0.17

Using this error value we will be backpropagating.



### 2.1.3 反向传播Back Propagation

#### 1. Calculating Gradients计算梯度

The change in each weight is calculated as:

> $Δw_{ij}=η×δ_j×O_j$

Where:

- $δ_j$ is the error term for each unit,
- $η$ is the learning rate.

#### 2. Output Unit Error输出层误差

For O3:

> $δ_5=y_5(1−y_5)(y_{target}−y_5)$

=0.67(1−0.67)(−0.17)=−0.0376

#### 3. Hidden Unit Error隐藏层误差

For h1:

> $δ_3=y_3(1−y_3)(w_{1,3}×δ_5)$

=0.56(1−0.56)(0.3×−0.0376)=−0.0027



For h2:

> $δ_4=y_4(1−y_4)(w_{2,3}×δ_5)$

=0.59(1−0.59)(0.9×−0.0376)=−0.0819



#### 4. Weight Updates权重更新

For the weights from hidden to output layer:

> $Δw_{2,3}=1×(−0.0376)×0.59=−0.022184$

New weight:

> $w_{2,3}(new)=−0.022184+0.9=0.877816$

For weights from input to hidden layer:

> $Δw_{1,1}=1×(−0.0027)×0.35=0.000945$

New weight:

> $w_{1,1}(new)=0.000945+0.2=0.200945$

Similarly other weights are updated:

- $w_{1,2}(new)=0.273225$
- $w_{1,3}(new)=0.086615$
- $w_{2,1}(new)=0.269445$
- $w_{2,2}(new)=0.18534$

The updated weights are illustrated below

![Backpropagation-in-Neural-Network-5](https://media.geeksforgeeks.org/wp-content/uploads/20250701165041328619/Backpropagation-in-Neural-Network-5.webp)

<center>Through backward pass the weights are updated</center>

> 上图权重没有更新，例如：$w_{2,2}$应该更新为0.18534



After updating the weights the forward pass is repeated yielding:

- y3=0.57
- y4=0.56
- y5=0.61

仍未达到目标值 0.5，因此继续进行反向传播，直到收敛。

> Since y5=0.61 is still not the target output the process of calculating the error and backpropagating continues until the desired output is reached.



This process demonstrates how Back Propagation iteratively updates weights by minimizing errors until the network accurately predicts the output.

> $Error=y_{target}−y_5 $

=0.5−0.61=−0.11=0.5−0.61=−0.11

This process is said to be continued until the actual output is gained by the neural network.



### 2.1.4 Back Propagation Implementation in Python for XOR Problem

This code demonstrates how Back Propagation is used in a neural network to solve the XOR problem. The neural network consists of:

#### 1. Defining Neural Network定义神经网络结构

输入层：2个节点，隐藏层：4个神经元，输出层：1个神经元，激活函数：Sigmoid

> We define a neural network as Input layer with 2 inputs, Hidden layer with 4 neurons, Output layer with 1 output neuron and use **Sigmoid** function as activation function.

- **self.input_size = input_size**: stores the size of the input layer
- **self.hidden_size = hidden_size:** stores the size of the hidden layer
- **self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)**: initializes weights for input to hidden layer
- **self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)**: initializes weights for hidden to output layer
- **self.bias_hidden = np.zeros((1, self.hidden_size)):** initializes bias for hidden layer
- **self.bias_output = np.zeros((1, self.output_size)):** initializes bias for output layer



```python3

import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weights_input_hidden = np.random.randn(
            self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(
            self.hidden_size, self.output_size)

        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)
```



#### 2. Defining Feed Forward Network定义前向传播

In Forward pass inputs are passed through the network activating the hidden and output layers using the sigmoid function.

- **self.hidden_activation = np.dot(X, self.weights_input_hidden) + self.bias_hidden**: calculates activation for hidden layer
- **self.hidden_output= self.sigmoid(self.hidden_activation)**: applies activation function to hidden layer
- **self.output_activation= np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output:** calculates activation for output layer
- **self.predicted_output = self.sigmoid(self.output_activation):** applies activation function to output layer





```python3

def feedforward(self, X):
    self.hidden_activation = np.dot(
        X, self.weights_input_hidden) + self.bias_hidden
    self.hidden_output = self.sigmoid(self.hidden_activation)

    self.output_activation = np.dot(
        self.hidden_output, self.weights_hidden_output) + self.bias_output
    self.predicted_output = self.sigmoid(self.output_activation)

    return self.predicted_output
```



#### 3. Defining Backward Network定义反向传播

In Backward pass or Back Propagation the errors between the predicted and actual outputs are computed. The gradients are calculated using the derivative of the sigmoid function and weights and biases are updated accordingly.

- **output_error = y - self.predicted_output:** calculates the error at the output layer
- **output_delta = output_error * self.sigmoid_derivative(self.predicted_output):** calculates the delta for the output layer
- **hidden_error = np.dot(output_delta, self.weights_hidden_output.T):** calculates the error at the hidden layer
- **hidden_delta = hidden_error \* self.sigmoid_derivative(self.hidden_output):** calculates the delta for the hidden layer
- **self.weights_hidden_output += np.dot(self.hidden_output.T, output_delta) * learning_rate:** updates weights between hidden and output layers
- **self.weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate:** updates weights between input and hidden layers



```python3

def backward(self, X, y, learning_rate):
    output_error = y - self.predicted_output
    output_delta = output_error * \
        self.sigmoid_derivative(self.predicted_output)

    hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
    hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

    self.weights_hidden_output += np.dot(self.hidden_output.T,
                                         output_delta) * learning_rate
    self.bias_output += np.sum(output_delta, axis=0,
                               keepdims=True) * learning_rate
    self.weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate
    self.bias_hidden += np.sum(hidden_delta, axis=0,
                               keepdims=True) * learning_rate
```



#### 4. Training Network训练网络

The network is trained over 10,000 epochs using the Back Propagation algorithm with a learning rate of 0.1 progressively reducing the error.

- **output = self.feedforward(X):** computes the output for the current inputs
- **self.backward(X, y, learning_rate):** updates weights and biases using Back Propagation
- **loss = np.mean(np.square(y - output)):** calculates the mean squared error (MSE) loss



```python3

def train(self, X, y, epochs, learning_rate):
    for epoch in range(epochs):
        output = self.feedforward(X)
        self.backward(X, y, learning_rate)
        if epoch % 4000 == 0:
            loss = np.mean(np.square(y - output))
            print(f"Epoch {epoch}, Loss:{loss}")
```

#### 5. Testing Neural Network测试神经网络

- **X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]):** defines the input data
- **y = np.array([[0], [1], [1], [0]]):** defines the target values
- **nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1):** initializes the neural network
- **nn.train(X, y, epochs=10000, learning_rate=0.1):** trains the network
- **output = nn.feedforward(X):** gets the final predictions after training





```python3

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
nn.train(X, y, epochs=10000, learning_rate=0.1)

output = nn.feedforward(X)
print("Predictions after training:")
print(output)
```

**Output:**

![Screenshot-2025-03-07-130223](https://media.geeksforgeeks.org/wp-content/uploads/20250307130234877256/Screenshot-2025-03-07-130223.png)

<center>Trained Model</center>



训练初期损失为 0.2713，逐步下降到 0.0066（第8000轮）。最终模型可以很好地逼近 XOR 函数的输出，即：

- 对于输入 [0,0] 和 [1,1]，输出接近 0

- 对于输入 [0,1] 和 [1,0]，输出接近 1

  

> - The output shows the training progress of a neural network over 10,000 epochs. Initially the loss was high (0.2713) but it gradually decreased as the network learned reaching a low value of 0.0066 by epoch 8000.
> - The final predictions are close to the expected XOR outputs: approximately 0 for [0, 0] and [1, 1] and approximately 1 for [0, 1] and [1, 0] indicating that the network successfully learned to approximate the XOR function.



### 2.1.5 反向传播的优点

**易于实现**：适合初学者，无需太多神经网络背景

**结构简单，灵活应用**：从简单前馈到复杂卷积/循环网络都可使用

**高效**：直接根据误差更新权重，学习速度快

**良好的泛化能力**：有助于模型在新数据上表现更好

**可扩展性好**：适用于大型数据集和深层模型

> **Advantages of Back Propagation for Neural Network Training**
>
> The key benefits of using the Back Propagation algorithm are:
>
> 1. **Ease of Implementation:** Back Propagation is beginner-friendly requiring no prior neural network knowledge and simplifies programming by adjusting weights with error derivatives.
> 2. **Simplicity and Flexibility:** Its straightforward design suits a range of tasks from basic feedforward to complex convolutional or recurrent networks.
> 3. **Efficiency**: Back Propagation accelerates learning by directly updating weights based on error especially in deep networks.
> 4. **Generalization:** It helps models generalize well to new data improving prediction accuracy on unseen examples.
> 5. **Scalability:** The algorithm scales efficiently with larger datasets and more complex networks making it ideal for large-scale tasks.



### 2.1.6 反向传播面临的挑战

**梯度消失**：在深层网络中梯度可能过小，导致学习困难（特别是在使用 sigmoid/tanh 时）

**梯度爆炸**：梯度可能变得过大，使训练不稳定

**过拟合**：模型结构过于复杂时，可能记住训练集而非学习一般性规律

> **Challenges with Back Propagation**
>
> While Back Propagation is useful it does face some challenges:
>
> 1. **Vanishing Gradient Problem**: In deep networks the gradients can become very small during Back Propagation making it difficult for the network to learn. This is common when using activation functions like sigmoid or tanh.
> 2. **Exploding Gradients**: The gradients can also become excessively large causing the network to diverge during training.
> 3. **Overfitting:** If the network is too complex it might memorize the training data instead of learning general patterns.



### 2.1.7 完整xor_nn代码

```python
# 对于XOR问题（输入为[0,0], [0,1], [1,0], [1,1]），期望输出为[0,1,1,0]
# 手动实现反向传播，没有使用深度学习框架，这有助于理解底层原理
# https://www.geeksforgeeks.org/backpropagation-in-neural-network/
import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size  # 输入特征维度
        self.hidden_size = hidden_size  # 隐藏层神经元数量
        self.output_size = output_size  # 输出层神经元数量

        # 输入层到隐藏层的权重，形状为 (输入维度, 隐藏层维度)
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        # 隐藏层到输出层的权重，形状为 (隐藏层维度, 输出层维度)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)

        # 隐藏层的偏置，形状为 (1, 隐藏层维度)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        # 输出层的偏置，形状为 (1, 输出层维度)
        self.bias_output = np.zeros((1, self.output_size))

    def sigmoid(self, x):  # 激活函数，将输入压缩到(0,1)区间
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)  # Sigmoid的导数，用于反向传播中的梯度计算

    def feedforward(self, X):
        # 隐藏层计算
        self.hidden_activation = np.dot(X, self.weights_input_hidden) + self.bias_hidden  # 线性变换
        self.hidden_output = self.sigmoid(self.hidden_activation)  # 激活函数

        # 输出层计算
        self.output_activation = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.predicted_output = self.sigmoid(self.output_activation)

        return self.predicted_output

    def backward(self, X, y, learning_rate):
        # 计算输出层误差
        output_error = y - self.predicted_output  # 误差 = 真实值 - 预测值
        # 计算输出层的delta（梯度的一部分，损失对激活输入的梯度）
        output_delta = output_error * self.sigmoid_derivative(self.predicted_output)  # Delta = 误差 × 激活函数导数
        # output_delta = (y - ŷ) * σ'(z_output)

        # 计算隐藏层误差（反向传播）
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)  # 将误差从输出层反向传播到隐藏层
        # hidden_error = output_delta @ W_hidden_output^T
        # 计算隐藏层的delta（损失对隐藏层激活输入的梯度）
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)  # Delta = 误差 × 激活函数导数
        # hidden_delta = (hidden_error) * σ'(z_hidden)

        # 更新权重和偏置（使用梯度下降法）
        # 计算并更新隐藏层到输出层的权重
        self.weights_hidden_output += np.dot(self.hidden_output.T,
                                             output_delta) * learning_rate  # 权重更新量 = 学习率 × (隐藏层输出转置 × 输出层delta)
        # W_hidden_output += learning_rate * (hidden_output^T @ output_delta)

        # 更新输出层偏置，基于所有样本的输出层delta沿列求和
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate  # 偏置更新量 = 学习率 × (沿列求和输出层delta)
        # b_output += learning_rate * sum(output_delta)

        # 计算并更新从输入层到隐藏层的权重的梯度
        self.weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate  # 权重更新量 = 学习率 × (输入数据转置 × 隐藏层delta)
        # W_input_hidden += learning_rate * (X^T @ hidden_delta)

        # 更新隐藏层偏置，基于所有样本的隐藏层delta沿列求和
        # axis=0：沿列求和，聚合所有样本的梯度
        # keepdims=True：保持原矩阵的行数维度，确保偏置更新的形状兼容性
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate  # 偏置更新量 = 学习率 × (沿列求和隐藏层delta)
        # b_hidden += learning_rate * sum(hidden_delta)

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.feedforward(X)  # 前向传播
            self.backward(X, y, learning_rate)  # 反向传播与参数更新
            if epoch % 4000 == 0:
                loss = np.mean(np.square(y - output))  # 计算均方误差
                print(f"Epoch {epoch}, Loss:{loss}")


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# 输入维度 2（二维二进制特征），隐藏层4个神经元，输出层1个神经元（二分类问题）
nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
# 训练总轮次, 学习率
nn.train(X, y, epochs=10000, learning_rate=0.1)

output = nn.feedforward(X)
print("Predictions after training:")
print(output)
"""
Epoch 0, Loss:0.2653166263520884
Epoch 4000, Loss:0.007000926683956338
Epoch 8000, Loss:0.001973630232951721
Predictions after training:
[[0.03613239]
 [0.96431351]
 [0.96058291]
 [0.03919372]]
"""
```



## 2.2 Neural Network实现鸢尾花卉数据分类

Updated 2013 GMT+8 Jul 26 2025

2025 spring, Complied by Hongfei Yan



鸢（yuān）尾花数据集包含三个类别，每个类别有50个样本，每个样本有四个特征：花萼长度、花萼宽度、花瓣长度、花瓣宽度。目标是根据这四个特征预测花的种类，属于多分类问题。

### 1.准备数据

鸢尾花数据集通常是通过scikit-learn的datasets模块获取的，所以可能需要结合scikit-learn来加载数据，然后转换成PyTorch的Tensor。

数据预处理方面，需要将特征数据和标签分开。特征数据需要标准化或归一化，因为不同特征的量纲可能不同，这对神经网络的训练有帮助。标签需要转换成数值形式，比如0、1、2，然后可能还需要转换为长整型张量，因为交叉熵损失函数在PyTorch中通常要求这样的格式。

### 2.训练模型和验证

然后，构建神经网络模型。考虑到鸢尾花数据集相对简单，可能不需要很深的网络。一个简单的全连接网络可能就足够了。比如，输入层4个节点，隐藏层可以选择10个节点，输出层3个节点对应三个类别。激活函数可以用ReLU，输出层用Softmax，不过更常见的做法是使用CrossEntropyLoss，它内部已经结合了Softmax，所以输出层不需要显式应用Softmax。

接下来是数据集的划分，通常分为训练集和测试集。这里需要注意的是，鸢尾花数据集样本较少，可能需要进行分层抽样，确保每个类别的样本在训练集和测试集中的比例一致。或者使用交叉验证，但由于用户可能希望一个简单的示例，可能直接采用80-20的划分。

将数据转换为PyTorch的DataLoader，方便批量训练。数据集进行训练可以分小批量处理。

训练过程中，定义损失函数和优化器。交叉熵损失函数适用于多分类问题，优化器可以选择Adam或SGD。学习率需要适当设置，比如0.01或0.001。

训练循环部分，需要迭代多个epoch，在每个epoch中进行前向传播、计算损失、反向传播和参数更新。同时，可以监控训练过程中的损失和准确率的变化。

验证阶段，使用测试集评估模型的准确率，确保模型没有过拟合，并且泛化能力良好。

可能需要注意的问题包括数据标准化应该在训练集上计算均值和标准差，然后应用到测试集，避免数据泄漏。另外，标签是否需要转换为one-hot编码？在PyTorch中，CrossEntropyLoss不需要，因为标签是类别的索引。

现在，将这些步骤整合成代码：

1. 导入必要的库：torch, sklearn.datasets中的load_iris，sklearn.model_selection的train_test_split，以及相关的PyTorch模块。
2. 加载数据，划分训练集和测试集，并进行标准化处理。使用StandardScaler对特征进行标准化，处理时注意仅拟合训练数据。
3. 转换数据为PyTorch的Tensor，并创建DataLoader。可能由于数据量小，直接使用整个数据集作为批量。
4. 定义神经网络模型的结构，使用全连接层和ReLU激活函数。
5. 初始化模型、损失函数和优化器。
6. 编写训练循环，迭代epoch，计算损失，反向传播，记录准确率。
7. 在测试集上评估模型性能。

另外，由于鸢尾花数据集样本较少，模型可能会很快过拟合，所以可能需要早停或者调整网络结构，比如减少隐藏层的大小，或者添加正则化，比如Dropout层。不过，作为示例，可能保持简单即可。

这样应该能完成一个基本的分类任务。不过可能需要调整超参数，比如学习率、epoch次数，或者网络结构，以达到更好的效果。例如，如果准确率不够高，可以尝试增加隐藏层的大小，或者增加更多的层。但鸢尾花数据集相对简单，可能简单的模型就足够。

另外，可能需要注意数据划分时的随机种子，确保结果可复现。此外，标准化处理是否正确应用，避免数据泄漏。在示例中，已经正确使用fit_transform在训练集，transform在测试集。



以下是使用PyTorch实现Fisher鸢尾花数据集分类的Python程序`iris_neural_network.py`，强调使用了神经网络进行分类。

如果缺少module，安装即可，例如：pip install torch，pip install sklearn等。

> 使用 pip install sklearn 可能不会得到预期的结果，因为PyPI（Python Package Index）上的官方包名称是 scikit-learn。但我确实用的 pip install sklearn。这是因为 sklearn 是 scikit-learn 的一个别名，某些情况下通过 pip 安装时能够识别这个别名并正确安装软件包。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 定义模型结构
class IrisNet(nn.Module):
    def __init__(self, input_size=4, hidden_size=10, num_classes=3):
        super(IrisNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# 训练函数
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_X.size(0)

    return running_loss / len(dataloader.dataset)


# 测试函数
def evaluate(model, X, y, device):
    model.eval()
    with torch.no_grad():
        X, y = X.to(device), y.to(device)
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y).float().mean().item()
    return accuracy, predicted


# 主程序
def main():
    # 设置设备
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 加载数据
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    """
    random_state=42
    设定随机数种子，从而确保每次运行代码时数据划分的结果都是相同的。这样做可以使实验具有可重复性，
    有利于调试和结果对比。

    stratify=y
    这个参数表示按照 y 中的标签进行分层抽样，也就是说，训练集和测试集中各类别的
    比例会与原始数据中的类别比例保持一致。这对于类别不平衡的数据集尤为重要，可以
    避免某一类别在划分时被严重低估或过采样。
    """

    # 标准化：只在训练集上计算均值和标准差，再将相同的变换应用到测试集上
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 转换为 Tensor
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # 构造 DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # 模型、损失函数、优化器
    model = IrisNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 训练
    num_epochs = 100
    for epoch in range(1, num_epochs + 1):
        loss = train(model, train_loader, criterion, optimizer, device)
        if epoch % 10 == 0:
            print(f"Epoch [{epoch:3d}/{num_epochs}], Loss: {loss:.4f}")

    # 评估
    test_acc, test_pred = evaluate(model, X_test, y_test, device)
    print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}%")

    # 示例预测
    sample = X_test[0].unsqueeze(0)
    sample_pred = model(sample.to(device))
    pred_class = torch.argmax(sample_pred, dim=1).item()
    print(f"🔍 Sample Prediction: True = {y_test[0].item()}, Predicted = {pred_class}")


if __name__ == "__main__":
    main()

```

> 云虚拟机运行结果：
>
> <img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20250223151816482.png" alt="image-20250223151816482" style="zoom:50%;" />



**代码说明：**

1. **数据准备**：
   - 使用scikit-learn加载鸢尾花数据集
   - 将数据划分为训练集（80%）和测试集（20%）
   - 使用标准化处理（StandardScaler）对特征进行归一化

2. **神经网络结构**：
   - 输入层：4个神经元（对应4个特征）
   - 隐藏层：10个神经元（使用ReLU激活函数）
   - 输出层：3个神经元（对应3个类别）

3. **训练配置**：
   - 使用交叉熵损失函数（CrossEntropyLoss）
   - 使用Adam优化器（学习率0.01）
   - 训练100个epoch

4. **训练过程**：
   - 每个epoch记录损失和准确率
   - 每10个epoch打印训练进度

5. **评估与预测**：
   - 最终在测试集上评估模型准确率
   - 包含一个预测示例展示

**输出示例：**

```
$ python iris_neural_network.py 
Epoch [ 10/100], Loss: 0.2363
Epoch [ 20/100], Loss: 0.0899
Epoch [ 30/100], Loss: 0.0614
Epoch [ 40/100], Loss: 0.0634
Epoch [ 50/100], Loss: 0.0498
Epoch [ 60/100], Loss: 0.0492
Epoch [ 70/100], Loss: 0.0492
Epoch [ 80/100], Loss: 0.0451
Epoch [ 90/100], Loss: 0.0479
Epoch [100/100], Loss: 0.0436

✅ Test Accuracy: 100.00%
🔍 Sample Prediction: True = 0, Predicted = 0
```

**注意事项：**

1. 由于数据集较小，模型可能很快达到100%训练准确率
2. 可以调整以下参数优化性能：
   - 隐藏层大小（10）
   - 学习率（0.01）
   - epoch数量（100）
   - 优化器（尝试SGD等）
3. 添加正则化（如Dropout层）可以防止过拟合
4. 可以使用GPU加速（将数据和模型移动到`cuda`设备）



## 2.3 MNIST 图像分类项目

通过手写数字识别的实例，我们可以看到神经网络的强大，也可以更好地理解它是如何运行的。

MNIST是一个著名的手写体数字识别数据集，训练集包含60,000个样本和标签，测试集包含10,000 个样本和标签。其中样本为代表0~9中的一个数字的灰度图片，对应一个所代表数字的标签，图片大小28*28，且数字出现在图片正中间。

**数据集**: 数据集 MNIST。

**数据预处理**: 因为 MNIST 图像是单通道（灰度图），而 ResNet18 模型是为 RGB 图像设计的，期望输入的通道数为 3，所以调整数据预处理的部分。

**输出层**: 将模型输出层为 10 类，因为 MNIST 有 10 个类别（0 到 9）。

**模型调整**: 使用的模型仍然是 `ResNet18`，但需要考虑到 MNIST 是单通道图像，调整输入层。

```python
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import time

def main():
    # 1. 数据增强 + 预处理
    transform_train = transforms.Compose([
        transforms.RandomRotation(10),  # 随机旋转
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # MNIST 是单通道，使用 (0.5,) 来规范化
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 加载 MNIST 数据集
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2, pin_memory=True)

    classes = [str(i) for i in range(10)]  # MNIST 类别是 0 到 9

    # 2. 设置设备和模型
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    # 加载预定义的 ResNet18 并修改输入层和输出层
    net = models.resnet18(weights=None)
    # 修改输入层的第一个卷积层，使其接受单通道（1通道灰度图像）
    net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    net.fc = nn.Linear(net.fc.in_features, 10)  # MNIST 10 类
    net.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # 3. 训练过程
    best_loss = float('inf')
    patience = 10  # 提高耐心
    patience_counter = 0

    start_time = time.time()
    print("Starting training with early stopping...")
    for epoch in range(800):  # 可适当增大 epoch
        net.train()
        epoch_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if i % 100 == 99:
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {loss.item():.3f}")

        avg_loss = epoch_loss / len(trainloader)
        print(f"[{epoch+1}] Avg Loss: {avg_loss:.3f}")

        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"✅ Training completed in {execution_time_minutes:.2f} minutes.")

    # 保存模型
    torch.save(net.state_dict(), './resnet18_mnist.pth')

    # 4. 测试准确率
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on test images: {100 * correct / total:.2f}%")

    # 每类准确率
    class_correct = list(0. for _ in range(10))
    class_total = list(0. for _ in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                class_correct[labels[i]] += c[i].item()
                class_total[labels[i]] += 1

    for i in range(10):
        print(f'Accuracy of {classes[i]:5s}: {100 * class_correct[i] / class_total[i]:.2f}%')

    # --- 可视化预测 ---

    def imshow_grid(images, labels, preds=None, classes=None, rows=8, cols=8):
        images = images.cpu() / 2 + 0.5  # unnormalize
        npimg = images.numpy()
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
        for i in range(rows * cols):
            r, c = divmod(i, cols)
            ax = axes[r, c]
            img = np.transpose(npimg[i], (1, 2, 0))
            ax.imshow(img.squeeze(), cmap="gray")
            title = f'{classes[labels[i]]}'
            if preds is not None:
                title += f'\n→ {classes[preds[i]]}'
            ax.set_title(title, fontsize=8)
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    # 获取一批图像用于显示
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    while images.size(0) < 64:
        more_images, more_labels = next(dataiter)
        images = torch.cat([images, more_images], dim=0)
        labels = torch.cat([labels, more_labels], dim=0)
    images = images[:64]
    labels = labels[:64]

    # 预测
    net.eval()
    with torch.no_grad():
        outputs = net(images.to(device))
        _, predicted = torch.max(outputs, 1)

    # 显示图像网格
    imshow_grid(images, labels, predicted.cpu(), classes=classes, rows=8, cols=8)

if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()


```



运行机器

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/202507261935350.png" alt="b452b39cfb47eb8bf5b640c828b6b71b" style="zoom:50%;" />



运行结果

```
/Users/hfyan/miniconda3/bin/python /Users/hfyan/git/2025spring-cs201/LLM/mnist_resnet18.py 
100%|██████████| 9.91M/9.91M [02:52<00:00, 57.6kB/s]
100%|██████████| 28.9k/28.9k [00:00<00:00, 97.2kB/s]
100%|██████████| 1.65M/1.65M [00:04<00:00, 374kB/s]
100%|██████████| 4.54k/4.54k [00:00<00:00, 6.74kB/s]
Using device: mps
Starting training with early stopping...
/Users/hfyan/miniconda3/lib/python3.10/site-packages/torch/utils/data/dataloader.py:683: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, then device pinned memory won't be used.
  warnings.warn(warn_msg)
[1,   100] loss: 0.136
[1,   200] loss: 0.132
[1,   300] loss: 0.035
[1,   400] loss: 0.098
[1] Avg Loss: 0.150
[2,   100] loss: 0.137
[2,   200] loss: 0.030
[2,   300] loss: 0.030
[2,   400] loss: 0.015
[2] Avg Loss: 0.052
[3,   100] loss: 0.018
[3,   200] loss: 0.105
[3,   300] loss: 0.078
[3,   400] loss: 0.026
[3] Avg Loss: 0.039
[4,   100] loss: 0.032
[4,   200] loss: 0.056
[4,   300] loss: 0.008
[4,   400] loss: 0.013
[4] Avg Loss: 0.031
[5,   100] loss: 0.003
[5,   200] loss: 0.025
[5,   300] loss: 0.029
[5,   400] loss: 0.022
[5] Avg Loss: 0.027
[6,   100] loss: 0.041
[6,   200] loss: 0.022
[6,   300] loss: 0.047
[6,   400] loss: 0.005
[6] Avg Loss: 0.023
[7,   100] loss: 0.039
[7,   200] loss: 0.000
[7,   300] loss: 0.022
[7,   400] loss: 0.014
[7] Avg Loss: 0.018
[8,   100] loss: 0.001
[8,   200] loss: 0.044
[8,   300] loss: 0.021
[8,   400] loss: 0.002
[8] Avg Loss: 0.019
No improvement. Patience: 1/10
[9,   100] loss: 0.002
[9,   200] loss: 0.020
[9,   300] loss: 0.002
[9,   400] loss: 0.007
[9] Avg Loss: 0.017
[10,   100] loss: 0.027
[10,   200] loss: 0.034
[10,   300] loss: 0.031
[10,   400] loss: 0.004
[10] Avg Loss: 0.016
[11,   100] loss: 0.003
[11,   200] loss: 0.004
[11,   300] loss: 0.005
[11,   400] loss: 0.003
[11] Avg Loss: 0.015
[12,   100] loss: 0.011
[12,   200] loss: 0.000
[12,   300] loss: 0.031
[12,   400] loss: 0.003
[12] Avg Loss: 0.015
No improvement. Patience: 1/10
[13,   100] loss: 0.002
[13,   200] loss: 0.002
[13,   300] loss: 0.002
[13,   400] loss: 0.019
[13] Avg Loss: 0.013
[14,   100] loss: 0.019
[14,   200] loss: 0.004
[14,   300] loss: 0.025
[14,   400] loss: 0.003
[14] Avg Loss: 0.013
No improvement. Patience: 1/10
[15,   100] loss: 0.003
[15,   200] loss: 0.001
[15,   300] loss: 0.011
[15,   400] loss: 0.056
[15] Avg Loss: 0.013
[16,   100] loss: 0.034
[16,   200] loss: 0.008
[16,   300] loss: 0.001
[16,   400] loss: 0.003
[16] Avg Loss: 0.011
[17,   100] loss: 0.008
[17,   200] loss: 0.001
[17,   300] loss: 0.001
[17,   400] loss: 0.001
[17] Avg Loss: 0.011
[18,   100] loss: 0.009
[18,   200] loss: 0.015
[18,   300] loss: 0.002
[18,   400] loss: 0.036
[18] Avg Loss: 0.013
No improvement. Patience: 1/10
[19,   100] loss: 0.019
[19,   200] loss: 0.001
[19,   300] loss: 0.023
[19,   400] loss: 0.005
[19] Avg Loss: 0.011
[20,   100] loss: 0.002
[20,   200] loss: 0.007
[20,   300] loss: 0.007
[20,   400] loss: 0.005
[20] Avg Loss: 0.011
No improvement. Patience: 1/10
[21,   100] loss: 0.001
[21,   200] loss: 0.008
[21,   300] loss: 0.012
[21,   400] loss: 0.005
[21] Avg Loss: 0.011
[22,   100] loss: 0.007
[22,   200] loss: 0.001
[22,   300] loss: 0.001
[22,   400] loss: 0.002
[22] Avg Loss: 0.011
No improvement. Patience: 1/10
[23,   100] loss: 0.003
[23,   200] loss: 0.002
[23,   300] loss: 0.001
[23,   400] loss: 0.014
[23] Avg Loss: 0.011
No improvement. Patience: 2/10
[24,   100] loss: 0.003
[24,   200] loss: 0.001
[24,   300] loss: 0.003
[24,   400] loss: 0.002
[24] Avg Loss: 0.010
[25,   100] loss: 0.016
[25,   200] loss: 0.002
[25,   300] loss: 0.010
[25,   400] loss: 0.000
[25] Avg Loss: 0.009
[26,   100] loss: 0.001
[26,   200] loss: 0.002
[26,   300] loss: 0.006
[26,   400] loss: 0.021
[26] Avg Loss: 0.008
[27,   100] loss: 0.002
[27,   200] loss: 0.002
[27,   300] loss: 0.017
[27,   400] loss: 0.000
[27] Avg Loss: 0.010
No improvement. Patience: 1/10
[28,   100] loss: 0.001
[28,   200] loss: 0.012
[28,   300] loss: 0.009
[28,   400] loss: 0.000
[28] Avg Loss: 0.008
No improvement. Patience: 2/10
[29,   100] loss: 0.001
[29,   200] loss: 0.008
[29,   300] loss: 0.009
[29,   400] loss: 0.031
[29] Avg Loss: 0.010
No improvement. Patience: 3/10
[30,   100] loss: 0.038
[30,   200] loss: 0.001
[30,   300] loss: 0.031
[30,   400] loss: 0.001
[30] Avg Loss: 0.011
No improvement. Patience: 4/10
[31,   100] loss: 0.017
[31,   200] loss: 0.013
[31,   300] loss: 0.029
[31,   400] loss: 0.032
[31] Avg Loss: 0.010
No improvement. Patience: 5/10
[32,   100] loss: 0.002
[32,   200] loss: 0.000
[32,   300] loss: 0.003
[32,   400] loss: 0.001
[32] Avg Loss: 0.009
No improvement. Patience: 6/10
[33,   100] loss: 0.009
[33,   200] loss: 0.018
[33,   300] loss: 0.001
[33,   400] loss: 0.007
[33] Avg Loss: 0.010
No improvement. Patience: 7/10
[34,   100] loss: 0.001
[34,   200] loss: 0.001
[34,   300] loss: 0.001
[34,   400] loss: 0.011
[34] Avg Loss: 0.010
No improvement. Patience: 8/10
[35,   100] loss: 0.004
[35,   200] loss: 0.005
[35,   300] loss: 0.009
[35,   400] loss: 0.010
[35] Avg Loss: 0.011
No improvement. Patience: 9/10
[36,   100] loss: 0.001
[36,   200] loss: 0.004
[36,   300] loss: 0.013
[36,   400] loss: 0.007
[36] Avg Loss: 0.008
[37,   100] loss: 0.008
[37,   200] loss: 0.003
[37,   300] loss: 0.007
[37,   400] loss: 0.002
[37] Avg Loss: 0.010
No improvement. Patience: 1/10
[38,   100] loss: 0.002
[38,   200] loss: 0.002
[38,   300] loss: 0.011
[38,   400] loss: 0.004
[38] Avg Loss: 0.009
No improvement. Patience: 2/10
[39,   100] loss: 0.006
[39,   200] loss: 0.003
[39,   300] loss: 0.002
[39,   400] loss: 0.001
[39] Avg Loss: 0.008
No improvement. Patience: 3/10
[40,   100] loss: 0.000
[40,   200] loss: 0.012
[40,   300] loss: 0.011
[40,   400] loss: 0.001
[40] Avg Loss: 0.009
No improvement. Patience: 4/10
[41,   100] loss: 0.010
[41,   200] loss: 0.008
[41,   300] loss: 0.006
[41,   400] loss: 0.002
[41] Avg Loss: 0.008
No improvement. Patience: 5/10
[42,   100] loss: 0.005
[42,   200] loss: 0.003
[42,   300] loss: 0.014
[42,   400] loss: 0.005
[42] Avg Loss: 0.010
No improvement. Patience: 6/10
[43,   100] loss: 0.010
[43,   200] loss: 0.000
[43,   300] loss: 0.012
[43,   400] loss: 0.002
[43] Avg Loss: 0.008
No improvement. Patience: 7/10
[44,   100] loss: 0.001
[44,   200] loss: 0.004
[44,   300] loss: 0.035
[44,   400] loss: 0.000
[44] Avg Loss: 0.011
No improvement. Patience: 8/10
[45,   100] loss: 0.002
[45,   200] loss: 0.014
[45,   300] loss: 0.010
[45,   400] loss: 0.014
[45] Avg Loss: 0.010
No improvement. Patience: 9/10
[46,   100] loss: 0.001
[46,   200] loss: 0.076
[46,   300] loss: 0.001
[46,   400] loss: 0.004
[46] Avg Loss: 0.009
No improvement. Patience: 10/10
Early stopping triggered.
✅ Training completed in 27.72 minutes.
Accuracy on test images: 99.57%
Accuracy of 0    : 99.59%
Accuracy of 1    : 99.91%
Accuracy of 2    : 99.71%
Accuracy of 3    : 99.80%
Accuracy of 4    : 99.49%
Accuracy of 5    : 99.33%
Accuracy of 6    : 99.37%
Accuracy of 7    : 99.22%
Accuracy of 8    : 99.90%
Accuracy of 9    : 99.31%

Process finished with exit code 0

```



<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/202507261936331.png" alt="22485e1e277b7dfea954fe0cd8a1af4f" style="zoom:50%;" />



**Q: net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) ?**

> 是的，`net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)`这一行代码是用于修改 ResNet18 的输入层，使其适应 MNIST 数据集。
>
> **解释：**
>
> - **MNIST 图像** 是灰度图（单通道），所以输入图像的通道数是 1，而原始的 ResNet18 模型是为 RGB 图像设计的，期望输入的通道数为 3。
> - **`nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)`** 这一行就是修改原来 `ResNet18` 中第一个卷积层（`conv1`）的定义，以使其能接收单通道的灰度图像（`1` 通道）。
>   - **`1`**: 输入图像的通道数（即 MNIST 图像的灰度通道数）。
>   - **`64`**: 输出的卷积通道数（ResNet18 中通常是 64）。
>   - **`kernel_size=(7, 7)`**: 卷积核的大小为 7x7。这个值与原始 ResNet18 中的设置一致。
>   - **`stride=(2, 2)`**: 卷积的步长为 2，这意味着每次卷积后，图像尺寸会减少一半。
>   - **`padding=(3, 3)`**: 填充为 3，保持输入图像的尺寸在卷积后不至于变化太大（确保卷积后输出的空间维度适当）。
>   - **`bias=False`**: 通常在深度网络中，如果使用了批量归一化（BatchNorm）等层，卷积层可以去掉偏置项。
>
> **为什么要做这一步：**
>
> - **输入层适配**: MNIST 图像是单通道的（灰度图），所以需要将 `ResNet18` 的输入层 `conv1` 的输入通道数由 3 调整为 1。否则，如果保持原样，网络无法正确处理单通道输入图像。
> - **卷积核和步长选择**: `kernel_size`, `stride`, `padding` 都是保持与原始 ResNet18 模型一致的超参数，目的是尽量保持模型结构不变，从而确保预期的效果。
>
> **总结：**
>
> 这一修改是必须的，因为 MNIST 是灰度图像，而 ResNet18 是为 RGB 图像设计的，必须修改输入通道数才能正确处理 MNIST 数据集。如果你用其他数据集（比如 CIFAR-10 或更大尺寸的图像），这部分的修改就不需要。



## 2.4 CIFAR-10 图像分类项目

用到的数据集是 The CIFAR-10 dataset，https://www.cs.toronto.edu/~kriz/cifar.html

> The CIFAR-10 and CIFAR-100 datasets are labeled subsets of the [80 million tiny images](http://people.csail.mit.edu/torralba/tinyimages/) dataset. CIFAR-10 and CIFAR-100 were created by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.
>
> **The CIFAR-10 dataset**
>
> The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
>
> The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.
>
> Here are the classes in the dataset, as well as 10 random images from each:
>
> | airplane   | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane1.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane2.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane3.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane4.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane5.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane6.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane7.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane8.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane9.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane10.png) |
> | ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
> | automobile | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile1.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile2.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile3.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile4.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile5.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile6.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile7.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile8.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile9.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile10.png) |
> | bird       | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/bird1.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/bird2.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/bird3.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/bird4.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/bird5.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/bird6.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/bird7.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/bird8.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/bird9.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/bird10.png) |
> | cat        | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/cat1.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/cat2.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/cat3.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/cat4.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/cat5.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/cat6.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/cat7.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/cat8.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/cat9.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/cat10.png) |
> | deer       | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/deer1.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/deer2.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/deer3.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/deer4.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/deer5.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/deer6.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/deer7.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/deer8.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/deer9.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/deer10.png) |
> | dog        | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog1.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog2.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog3.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog4.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog5.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog6.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog7.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog8.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog9.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog10.png) |
> | frog       | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/frog1.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/frog2.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/frog3.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/frog4.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/frog5.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/frog6.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/frog7.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/frog8.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/frog9.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/frog10.png) |
> | horse      | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/horse1.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/horse2.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/horse3.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/horse4.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/horse5.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/horse6.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/horse7.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/horse8.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/horse9.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/horse10.png) |
> | ship       | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/ship1.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/ship2.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/ship3.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/ship4.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/ship5.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/ship6.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/ship7.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/ship8.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/ship9.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/ship10.png) |
> | truck      | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/truck1.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/truck2.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/truck3.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/truck4.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/truck5.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/truck6.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/truck7.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/truck8.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/truck9.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/truck10.png) |
>
> The classes are completely mutually exclusive. There is no overlap between automobiles and trucks. "Automobile" includes sedans, SUVs, things of that sort. "Truck" includes only big trucks. Neither includes pickup trucks.
>
> ### Download
>
> If you're going to use this dataset, please cite the tech report at the bottom of this page. 
>
> | Version                                                      | Size   | md5sum                           |
> | ------------------------------------------------------------ | ------ | -------------------------------- |
> | [CIFAR-10 python version](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) | 163 MB | c58f30108f718f92721af3b95e74349a |
> | [CIFAR-10 Matlab version](https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz) | 175 MB | 70270af85842c9e89bb428ec9976c926 |
> | [CIFAR-10 binary version (suitable for C programs)](https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz) | 162 MB | c32a1d4ab5d03f1284b67883e8d87530 |
>
> ### Baseline results
>
> You can find some baseline replicable results on this dataset [on the project page for cuda-convnet](http://code.google.com/p/cuda-convnet/). These results were obtained with a convolutional neural network. Briefly, they are 18% test error without data augmentation and 11% with. Additionally, [Jasper Snoek](http://www.cs.toronto.edu/~jasper/) has a [new paper](http://hips.seas.harvard.edu/content/practical-bayesian-optimization-machine-learning-algorithms) in which he used Bayesian hyperparameter optimization to find nice settings of the weight decay and other hyperparameters, which allowed him to obtain a test error rate of 15% (without data augmentation) using the architecture of the net that got 18%.



本意是运行 https://www.runoob.com/pytorch/pytorch-image-classification.html

但是准确率没有达到baseline的82%，所以做了改进。

### 用ResNet18结构，重新训练，做了数据增强

```python
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import time

def main():
    # 1. 数据增强 + 预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),  # 随机旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 色彩调整
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2, pin_memory=True)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # 2. 设置设备和模型
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    # 加载预定义的 ResNet18 并修改输出层
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, 10)  # CIFAR10 10 类
    net.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # 3. 训练过程
    best_loss = float('inf')
    patience = 10 # 提高耐心
    patience_counter = 0

    start_time = time.time()
    print("Starting training with early stopping...")
    for epoch in range(800):  # 可适当增大 epoch
        net.train()
        epoch_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if i % 100 == 99:
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {loss.item():.3f}")

        avg_loss = epoch_loss / len(trainloader)
        print(f"[{epoch+1}] Avg Loss: {avg_loss:.3f}")

        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break



    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"✅ Training completed in {execution_time_minutes:.2f} minutes.")


    # 保存模型
    torch.save(net.state_dict(), './resnet18_cifar10_data_augument.pth')

    # 4. 测试准确率
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on test images: {100 * correct / total:.2f}%")

    # 每类准确率
    class_correct = list(0. for _ in range(10))
    class_total = list(0. for _ in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                class_correct[labels[i]] += c[i].item()
                class_total[labels[i]] += 1

    for i in range(10):
        print(f'Accuracy of {classes[i]:5s}: {100 * class_correct[i] / class_total[i]:.2f}%')

    # --- 可视化预测 ---

    def imshow_grid(images, labels, preds=None, classes=None, rows=8, cols=8):
        images = images.cpu() / 2 + 0.5  # unnormalize
        npimg = images.numpy()
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
        for i in range(rows * cols):
            r, c = divmod(i, cols)
            ax = axes[r, c]
            img = np.transpose(npimg[i], (1, 2, 0))
            ax.imshow(img)
            title = f'{classes[labels[i]]}'
            if preds is not None:
                title += f'\n→ {classes[preds[i]]}'
            ax.set_title(title, fontsize=8)
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    # 获取一批图像用于显示
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    while images.size(0) < 64:
        more_images, more_labels = next(dataiter)
        images = torch.cat([images, more_images], dim=0)
        labels = torch.cat([labels, more_labels], dim=0)
    images = images[:64]
    labels = labels[:64]

    # 预测
    net.eval()
    with torch.no_grad():
        outputs = net(images.to(device))
        _, predicted = torch.max(outputs, 1)

    # 显示图像网格
    imshow_grid(images, labels, predicted.cpu(), classes=classes, rows=8, cols=8)

if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()

```



运行结果如下：

```
/Users/hfyan/miniconda3/bin/python /Users/hfyan/Desktop/LLMs-from-scratch-main/runoob/pytorch-image-classification/image_classification-ResNet18-RandomCropFlipLR_Cosine.py 
Using device: mps
Starting training with early stopping...
[1,   100] loss: 1.752
[1,   200] loss: 1.675
[1,   300] loss: 1.654
[1] Avg Loss: 1.806
[2,   100] loss: 1.497
[2,   200] loss: 1.459
[2,   300] loss: 1.453
[2] Avg Loss: 1.520
[3,   100] loss: 1.534
[3,   200] loss: 1.383
[3,   300] loss: 1.167
[3] Avg Loss: 1.372
[4,   100] loss: 1.390
[4,   200] loss: 1.221
[4,   300] loss: 1.238
[4] Avg Loss: 1.244
[5,   100] loss: 1.089
[5,   200] loss: 1.020
[5,   300] loss: 1.133
[5] Avg Loss: 1.159
......
No improvement. Patience: 1/10
[192,   100] loss: 0.187
[192,   200] loss: 0.293
[192,   300] loss: 0.356
[192] Avg Loss: 0.302
[193,   100] loss: 0.223
[193,   200] loss: 0.348
[193,   300] loss: 0.309
[193] Avg Loss: 0.301
[194,   100] loss: 0.303
[194,   200] loss: 0.219
[194,   300] loss: 0.280
[194] Avg Loss: 0.304
No improvement. Patience: 1/10
[195,   100] loss: 0.279
[195,   200] loss: 0.296
[195,   300] loss: 0.313
[195] Avg Loss: 0.296
[196,   100] loss: 0.254
[196,   200] loss: 0.385
[196,   300] loss: 0.280
[196] Avg Loss: 0.300
No improvement. Patience: 1/10
[197,   100] loss: 0.216
[197,   200] loss: 0.298
[197,   300] loss: 0.290
[197] Avg Loss: 0.298
No improvement. Patience: 2/10
[198,   100] loss: 0.267
[198,   200] loss: 0.218
[198,   300] loss: 0.367
[198] Avg Loss: 0.290
[199,   100] loss: 0.270
[199,   200] loss: 0.240
[199,   300] loss: 0.351
[199] Avg Loss: 0.301
No improvement. Patience: 1/10
[200,   100] loss: 0.251
[200,   200] loss: 0.227
[200,   300] loss: 0.302
[200] Avg Loss: 0.299
No improvement. Patience: 2/10
[201,   100] loss: 0.348
[201,   200] loss: 0.301
[201,   300] loss: 0.193
[201] Avg Loss: 0.299
No improvement. Patience: 3/10
[202,   100] loss: 0.313
[202,   200] loss: 0.329
[202,   300] loss: 0.305
[202] Avg Loss: 0.295
No improvement. Patience: 4/10
[203,   100] loss: 0.266
[203,   200] loss: 0.254
[203,   300] loss: 0.307
[203] Avg Loss: 0.294
No improvement. Patience: 5/10
[204,   100] loss: 0.372
[204,   200] loss: 0.295
[204,   300] loss: 0.348
[204] Avg Loss: 0.300
No improvement. Patience: 6/10
[205,   100] loss: 0.392
[205,   200] loss: 0.353
[205,   300] loss: 0.306
[205] Avg Loss: 0.296
No improvement. Patience: 7/10
[206,   100] loss: 0.262
[206,   200] loss: 0.213
[206,   300] loss: 0.396
[206] Avg Loss: 0.293
No improvement. Patience: 8/10
[207,   100] loss: 0.293
[207,   200] loss: 0.204
[207,   300] loss: 0.337
[207] Avg Loss: 0.291
No improvement. Patience: 9/10
[208,   100] loss: 0.413
[208,   200] loss: 0.294
[208,   300] loss: 0.315
[208] Avg Loss: 0.295
No improvement. Patience: 10/10
Early stopping triggered.
✅ Training completed in 79.91 minutes.
Accuracy on test images: 83.57%
Accuracy of plane: 83.70%
Accuracy of car  : 92.20%
Accuracy of bird : 78.70%
Accuracy of cat  : 60.40%
Accuracy of deer : 79.30%
Accuracy of dog  : 77.40%
Accuracy of frog : 90.30%
Accuracy of horse: 92.50%
Accuracy of ship : 88.50%
Accuracy of truck: 92.70%

Process finished with exit code 0
```



### 本部分QA

Q: 之前版本 `image_classification-ResNet18.py` 没有baseline好？

> 你现在的结果是：
>
> - **Test Accuracy:** 76.77%
> - **Test Error Rate:** 100% - 76.77% = **23.23%**
>
> 而你引用的 **baseline（来自 cuda-convnet 项目）**提供的是：
>
> | 方法                       | Test Error (%) | Test Accuracy (%) |
> | -------------------------- | -------------- | ----------------- |
> | Baseline CNN（无数据增强） | **18.0%**      | **82.0%**         |
> | Bayesian 优化后（无增强）  | **15.0%**      | **85.0%**         |
> | CNN（带数据增强）          | **11.0%**      | **89.0%**         |
>
> ------
>
> 🆚 对比分析：
>
> | 指标              | 你的模型（ResNet18） | baseline 最好  | 差距                |
> | ----------------- | -------------------- | -------------- | ------------------- |
> | Test Accuracy (%) | **76.77%**           | **89.0%**      | ↓ ~12.2%            |
> | 网络结构          | ResNet18             | 自定义 CNN     | ResNet18 是更强模型 |
> | 数据增强          | 否                   | 是             | ✅ 可提升            |
> | 超参调优          | 默认                 | 有（Bayesian） | ✅ 可提升            |
>
> ------
>
> 🧠 结论：
>
> 你当前的结果是 **很不错的起点**，考虑到：
>
> - 使用的是标准 `ResNet18`，且没有做任何 **数据增强**；
> - 没有特别调 `lr`、`weight_decay`、`batch_size` 等超参数；
> - 没有用额外 trick，如 CosineLR、Mixup、Label Smoothing 等；
> - 用的是 **MPS 后端**（速度快，但可能稳定性和 cuda 还差点）；
>
> **因此，你的结果基本和“未调优的 ResNet18 基线”持平。**
>
> ------
>
> ✅ 如何追平甚至超越 baseline？
>
> 你的 ResNet18 很强，但你可以继续提升准确率：
>
> | 方法                    | 说明                                        |
> | ----------------------- | ------------------------------------------- |
> | ✅ 加数据增强            | `transforms.RandomCrop + HorizontalFlip`    |
> | ✅ 调优学习率/调度器     | 使用 `CosineAnnealingLR`、warmup 等         |
> | ✅ 引入正则化技巧        | 加 weight_decay，dropout 或 label smoothing |
> | ✅ 换更高效模型（可选）  | MobileNetV2 或 EfficientNet                 |
> | ✅ 加入验证集 early stop | 防止过拟合，提升泛化                        |
> | ✅ 多轮训练+冻结微调     | 微调 ImageNet 权重部分                      |
>
> ------
>
> 是否需要我帮你**加数据增强版本**？几行代码就可以在 `transform` 中添加，让 ResNet18 发挥更大潜力。
>
> ```python
> transform_train = transforms.Compose([
>     transforms.RandomCrop(32, padding=4),
>     transforms.RandomHorizontalFlip(),
>     transforms.ToTensor(),
>     transforms.Normalize(mean, std),
> ])
> ```
>
> 是否需要我直接给出对比完整训练脚本？可以一键替换你当前版本。



Q: transforms.Resize(224), # ResNet 输入要求至少224x224，现在不需要了？

> 你问得非常好，这涉及到 **ResNet18 与 CIFAR-10 输入图像尺寸的兼容性问题**：
>
> ------
>
> 🔍 ResNet18 的原始设计输入尺寸
>
> - 原始 ImageNet 预训练的 ResNet18 输入尺寸是 **224×224**。
>
> - 因此常见用法中通常包含：
>
>   ```python
>   transforms.Resize(256),
>   transforms.CenterCrop(224),
>   ```
>
> ------
>
> ✅ 在 CIFAR-10 上使用 ResNet18 不需要 `Resize(224)` 的情况
>
> 因为你在训练中使用的是：
>
> ```python
> models.resnet18(weights=None)
> ```
>
> 这表示：
>
> - **没有加载预训练权重**
> - 网络从头开始训练
> - 所以你可以**自由地替换第一层卷积**或直接用默认的，**不需要放大到 224×224**
>
> 而且你的数据（CIFAR-10）是 `32x32`，所以 torchvision 中常见的处理方式是：
>
> - **保留原始尺寸**
> - 只使用：`RandomCrop(32, padding=4)` + `RandomHorizontalFlip()` 就足够了
>
> ------
>
> 🧠 如果你想使用预训练权重
>
> 如果你打算使用：
>
> ```python
> models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
> ```
>
> 则你必须：
>
> 1. **调整输入图像大小到 ≥224x224**：
>
>    ```python
>    transforms.Resize(256),
>    transforms.CenterCrop(224),
>    ```
>
> 2. 或者修改模型第一层：
>
>    将：
>
>    ```python
>    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
>    model.maxpool = nn.Identity()
>    ```
>
> ------
>
> ✅ 结论
>
> | 用法                                           | 是否需要 Resize(224) | 说明                |
> | ---------------------------------------------- | -------------------- | ------------------- |
> | `weights=None`（从头训练）                     | ❌ 不需要             | 可直接用 32×32 图像 |
> | `weights=ResNet18_Weights.DEFAULT`（用预训练） | ✅ 需要               | 需Resize到224x224   |
>
> ------
>
> 是否需要我帮你写一份使用 **预训练ResNet18** + 自动调整尺寸的完整代码版本？



Q:  **精度 (Accuracy)** 仍然比 **baseline (82%)** 稍低，可能的原因和改进建议：

> 你的训练结果已经相当不错，但 **精度 (Accuracy)** 仍然比 **baseline (82%)** 稍低，以下是一些可能的原因和改进建议：
>
> ------
>
> ✅ 可能的原因：
>
> 1. **没有数据增强**：
>    - 你的当前数据增强设置中仅包含了 `RandomCrop` 和 `RandomHorizontalFlip`。这对 ResNet18 来说可能不够，尤其是在 CIFAR-10 这种小规模数据集上。Baseline 的 82% 准确率是使用了 **更多的数据增强**。
> 2. **学习率（Learning Rate）设置问题**：
>    - 你的学习率设置是 `lr=0.1`，这可能稍微有些高，导致网络在训练初期就发生了震荡，无法有效收敛。
>    - 你可以尝试 **逐步降低学习率**，比如使用学习率调度器（`lr_scheduler`）或者 **更低的初始学习率**（如 0.01 或 0.001）。
> 3. **训练轮数（Epochs）不足**：
>    - 虽然你设置了 **100 个 epoch**，但根据训练过程来看，模型在 15 分钟内只训练了大约 **24 个 epoch**。这意味着可能还需要更多的训练时间来进一步降低损失。
> 4. **早期停止（Early Stopping）**：
>    - 早期停止可能在损失尚未完全降低时就停止了训练。你可能会希望 **放宽早期停止的条件**，让训练继续更多 epoch，直到精度达到更高水平。
>
> ------
>
> ✅ 改进建议：
>
> 1. **增加数据增强**：
>
> - 为了提高模型的泛化能力，建议添加更多的数据增强操作。可以加入：
>   - `RandomRotation(10)`：随机旋转图像
>   - `ColorJitter`：调整图像的亮度、对比度等
>   - `RandomAffine`：随机仿射变换
>
> 修改后的数据增强代码：
>
> ```python
> transform_train = transforms.Compose([
>     transforms.RandomCrop(32, padding=4),
>     transforms.RandomHorizontalFlip(),
>     transforms.RandomRotation(10),  # 随机旋转
>     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 色彩调整
>     transforms.ToTensor(),
>     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
> ])
> ```
>
> 2. **调整学习率（Learning Rate）**：
>
> - **逐步降低学习率**：建议在训练时 **使用动态学习率**，可以尝试更低的初始学习率（如 0.01），并设置一个学习率调度器（`lr_scheduler`）来逐步降低学习率。
>
> ```python
> optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
> scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)  # 每20个epoch将学习率降低50%
> ```
>
> 3. **放宽 Early Stopping 的条件**：
>
> - 提高 `patience`（例如从 5 改为 10），或者将停止条件设为更宽松的标准，比如 **增加验证集准确率的持续提升**。
>
> ```python
> patience = 10  # 提高耐心
> ```
>
> 4. **增加训练轮数（Epochs）**：
>
> - 尝试增加训练轮数（例如 200-300），即使使用早期停止，增加训练时间仍然有助于模型的收敛。
>
> 5. **检查数据集的划分**：
>
> - 确保你在训练和验证集上没有数据泄漏，并且确保 **数据加载时没有随机性** 影响模型评估（比如使用固定的随机种子）。
>
> 6. **使用数据增强（如 MixUp 或 Cutout）**：
>
> - 你可以进一步尝试 **MixUp**（通过线性插值进行数据增强）或 **Cutout**（随机遮挡一部分图像）等方法，已知在 CIFAR-10 上这些方法有助于提高准确率。
>
> ------
>
> ✅ 小结：
>
> 你的模型训练已经非常接近 baseline 的结果，经过数据增强、适当的学习率调节、更多训练轮次等优化后，应该能进一步提升准确率。你还可以尝试 **使用预训练模型**，或者 **迁移学习**，这些方法能够让模型更快收敛，特别是在较小数据集上。
>
> ------
>
> 是否需要我进一步帮助你集成 **学习率调度器**、**更多数据增强**、**验证集评估**等功能，或者直接给出一个全新的版本进行改进？



> 🔧 新改进点：
>
> 1. **数据增强**：
>    - 增加了 `RandomRotation` 和 `ColorJitter`，可以进一步增强模型的泛化能力。
> 2. **学习率调度器**：
>    - 使用 `CosineAnnealingLR` 来动态调整学习率，帮助模型在训练后期更平滑地收敛。
> 3. **早期停止**：
>    - 调整了 `patience` 为 10 轮，允许模型有更多时间来收敛，防止过早停止。
> 4. **训练时间统计**：
>    - 输出了总训练时间，方便了解训练消耗。
> 5. **模型保存**：
>    - 保存了训练好的模型，方便以后加载和推理。
>
> ------
>
> 下一步：
>
> 你可以运行这个脚本，看是否能 **超越 baseline 的 82%** 准确率。如果结果不理想，可以进一步调整超参数、添加更多的技巧（如 MixUp、Cutout）。



**Q: 解读代码**

```
解读：    # 每类准确率
    class_correct = list(0. for _ in range(10))
    class_total = list(0. for _ in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                class_correct[labels[i]] += c[i].item()
                class_total[labels[i]] += 1
```

> 这段代码用于计算 **每一类的准确率**，具体步骤如下：
>
> ------
>
> 🧩 代码解读
>
> ```python
> # 创建两个列表，分别用于存储每一类的正确预测数和总样本数
> class_correct = list(0. for _ in range(10))  # 存储每个类别正确预测的次数
> class_total = list(0. for _ in range(10))    # 存储每个类别的样本数
> ```
>
> - `class_correct[i]` 用于记录类别 `i` 的正确预测次数。
> - `class_total[i]` 用于记录类别 `i` 的总样本数。
>
> `10` 是 CIFAR-10 数据集中的类别数量（10 类）。
>
> ------
>
> ```python
> # 不计算梯度，加速推理
> with torch.no_grad():
>     for data in testloader:
>         images, labels = data
>         images, labels = images.to(device), labels.to(device)
>         outputs = net(images)
>         _, predicted = torch.max(outputs.data, 1)
> ```
>
> - 使用 `torch.no_grad()` 来 **关闭梯度计算**，这样在推理时节省内存和计算。
> - `testloader` 是加载测试集的数据迭代器，`images` 是输入图像，`labels` 是对应的标签。
> - `outputs = net(images)`：用训练好的模型 `net` 对图像进行推理，得到每个类别的预测概率。
> - `_, predicted = torch.max(outputs.data, 1)`：通过 `torch.max` 获取 **每张图像的最大概率类别**（即模型的预测标签）。
>
> ------
>
> ```python
> # 计算每个类别的正确预测数量和总样本数量
> c = (predicted == labels).squeeze()
> for i in range(len(labels)):
>     class_correct[labels[i]] += c[i].item()  # 记录正确预测的数量
>     class_total[labels[i]] += 1              # 记录每个类别的样本数量
> ```
>
> - `(predicted == labels)`：检查预测标签与实际标签是否一致，返回布尔值（True 或 False）。`True` 代表预测正确，`False` 代表预测错误。
> - `c[i].item()`：转换布尔值为 1 或 0，表示该样本是否被正确分类。
> - `class_correct[labels[i]] += c[i].item()`：将每个正确预测的样本数量累加到对应类别的 `class_correct` 中。
> - `class_total[labels[i]] += 1`：每遇到一个样本，就累加到该类别的总样本数 `class_total`。
>
> ------
>
> 🧠 总结：
>
> 这段代码的目的是通过遍历测试集，计算每个类别的**正确预测数**和**总样本数**。最后，你可以通过以下方式计算 **每个类别的准确率**：
>
> ```python
> for i in range(10):
>     accuracy = 100 * class_correct[i] / class_total[i]
>     print(f'Accuracy of {classes[i]:5s}: {accuracy:.2f}%')
> ```
>
> `classes[i]` 是 CIFAR-10 中的类别名称（如 `plane`, `car` 等）。
>
> ------
>
> 🔄 关键点：
>
> 1. **`torch.max(outputs.data, 1)`**: 获取每个样本的预测类别。
> 2. **`class_correct[labels[i]] += c[i].item()`**: 累加正确预测的数量。
> 3. **`class_total[labels[i]] += 1`**: 累加每个类别的样本数量。
>
> 通过这个过程，你能得到每个类别的准确率，有助于了解模型在哪些类别上表现较好或较差。



**Q: 具体示例呢？是一个一个图像检测吧**

> 是的，**这段代码确实是在一个个图像上检测预测是否正确**，并按类别统计每类的预测准确率。下面我们结合一个**具体示例**来演示这段代码在实际运行中做了什么。
>
> ------
>
> 🎯 假设场景
>
> 我们假设测试集中某个 batch（小批量）里有 5 张图像，真实标签为：
>
> ```python
> labels = [1, 3, 3, 5, 1]  # 实际类别编号
> ```
>
> 模型预测结果为：
>
> ```python
> predicted = [1, 2, 3, 5, 0]  # 模型预测的类别编号
> ```
>
> 那么对比：
>
> | 图片编号 | 实际标签（labels[i]） | 模型预测（predicted[i]） | 是否正确 (==) |
> | -------- | --------------------- | ------------------------ | ------------- |
> | 0        | 1                     | 1                        | ✅ True        |
> | 1        | 3                     | 2                        | ❌ False       |
> | 2        | 3                     | 3                        | ✅ True        |
> | 3        | 5                     | 5                        | ✅ True        |
> | 4        | 1                     | 0                        | ❌ False       |
>
> 生成布尔比较：
>
> ```python
> c = [True, False, True, True, False]
> ```
>
> ------
>
> 📊 累加到 `class_correct` 和 `class_total`
>
> 初始化：
>
> ```python
> class_correct = [0.0] * 10
> class_total = [0.0] * 10
> ```
>
> 遍历每个样本：
>
> ```python
> for i in range(5):
>  class_correct[labels[i]] += c[i].item()
>  class_total[labels[i]] += 1
> ```
>
> 每一步的变化如下：
>
> | i    | labels[i] | c[i]  | class_correct         | class_total           |
> | ---- | --------- | ----- | --------------------- | --------------------- |
> | 0    | 1         | True  | [0, 1, 0, 0, 0…]      | [0, 1, 0, 0, 0…]      |
> | 1    | 3         | False | [0, 1, 0, 0, 0…]      | [0, 1, 0, 1, 0…]      |
> | 2    | 3         | True  | [0, 1, 0, 1, 0…]      | [0, 1, 0, 2, 0…]      |
> | 3    | 5         | True  | [0, 1, 0, 1, 0, 1, …] | [0, 1, 0, 2, 0, 1, …] |
> | 4    | 1         | False | [0, 1, 0, 1, 0, 1, …] | [0, 2, 0, 2, 0, 1, …] |
>
> ------
>
> ✅ 准确率结果（最终）
>
> ```python
> Accuracy of class 1: 1 / 2 = 50.00%
> Accuracy of class 3: 1 / 2 = 50.00%
> Accuracy of class 5: 1 / 1 = 100.00%
> ```
>
> 其他类别没有出现，准确率为 0%。
>
> ------
>
> 🔁 在真实代码中：
>
> 这段逻辑在 **整个 testloader（测试集）** 上运行，每次处理一个 batch（如 100 张图像），最后统计出完整的 per-class accuracy。
>
> ------
>
> ✅ 总结：
>
> - 是一个图像一个图像在比对 `预测是否正确`。
> - 累加每个类别的正确预测次数和样本总数。
> - 最后计算 `每类准确率 = 正确预测数 / 样本总数`。
>
> 你可以在训练完后非常直观地看到：**模型在哪些类别上表现好（准确率高），哪些类别不好（比如猫通常较难分类）**。这对分析模型瓶颈很有帮助。



**Q: _, predicted = torch.max(outputs.data, 1) ？**

> 这行代码：
>
> ```python
> _, predicted = torch.max(outputs.data, 1)
> ```
>
> 是 PyTorch 中的常见写法，用于从模型的输出中**获取每张图片预测的类别编号**（也就是最大得分对应的下标）。下面是详细解读。
>
> ------
>
> 📘 背景：模型输出是什么？
>
> 在分类任务中，比如 CIFAR-10，模型对每张图像的输出是一个 **长度为10的向量**，表示每个类别的得分（logits）。例如对一张图片，输出可能是：
>
> ```python
> [2.1, 1.5, -0.3, 4.2, 0.8, 3.3, 1.1, 0.0, -1.2, 2.0]
> ```
>
> 这表示对类别 0～9 的“置信度”或“匹配程度”，越大越可能是正确类别。
>
> ------
>
> 🧠 `torch.max(outputs.data, 1)` 解释
>
> ```python
> torch.max(tensor, dim)
> ```
>
> 这个函数会在指定的维度 `dim` 上寻找最大值，并返回两个结果：
>
> 1. **最大值的数值**
> 2. **最大值的下标（索引）**
>
> 例如：
>
> ```python
> outputs.data = tensor([
>   [2.1, 1.5, -0.3, 4.2, 0.8, 3.3, 1.1, 0.0, -1.2, 2.0],
>   [0.5, 2.8, 1.0, 0.7, 0.1, 0.0, 3.9, 0.2, 0.3, 0.5]
> ])
> ```
>
> 表示 2 张图片的分类输出。
>
> 执行：
>
> ```python
> _, predicted = torch.max(outputs.data, 1)
> ```
>
> 就会得到：
>
> ```python
> predicted = tensor([3, 6])
> ```
>
> 表示第 1 张图像预测为类别 3，第 2 张为类别 6。
>
> ------
>
> 📌 为什么用 `_` 接收第一个返回值？
>
> 因为你只关心“哪个类别是最大”，不关心“最大得分是多少”，所以第一个返回值（得分）就用 `_` 忽略掉了。
>
> ------
>
> ✅ 总结
>
> ```python
> _, predicted = torch.max(outputs.data, 1)
> ```
>
> 含义是：
>
> > 在每张图像的输出向量中，找出最大值所在的下标 `predicted`，作为该图像的预测类别编号。
>
> 这个 `predicted` 就可以和 `labels` 做比较来判断是否预测正确。





### 附录

**ImageNet 图像分类任务中的 Top-5 错误率**

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20250726164756658.png" alt="image-20250726164756658" style="zoom: 67%;" />



**x轴：模型和年份**

从左到右列出了不同年份的代表性模型：

| 年份 | 模型         | 所属机构/作者            |
| ---- | ------------ | ------------------------ |
| 2011 | XRCE         | 非深度学习方法           |
| 2012 | AlexNet      | Alex Krizhevsky 等人     |
| 2013 | ZFNet        | Zeiler & Fergus          |
| 2014 | VGG          | Visual Geometry Group    |
| 2014 | GoogLeNet    | Google（Inception）      |
| 2015 | ResNet       | Microsoft Research       |
| 2016 | GoogLeNet-v4 | 更深更复杂的 GoogLeNet   |
| ---  | **Human**    | 人类平均表现（对比基线） |

y轴：**Top-5 错误率（%）**

表示模型在 ImageNet 上错误预测的概率（Top-5）。错误率越低，模型性能越好。

从图中可以看到：

- **2011 XRCE：Top-5错误率为 26.0%**
- **2012 AlexNet** 将错误率降低到了 **16.4%**，开启深度学习时代
- **2015 ResNet** 将错误率降至 **3.6%**，首次超过了 **人类**（人类误差为 5.0%）
- **2016 GoogLeNet-v4** 降到仅 **3.1%**

------

🧠 什么是 Top-5 错误率？

在 ImageNet 图像分类中，**每张图像的标签属于 1000 个类别之一**。

- **Top-1 错误率**：预测第一名是否正确。
- **Top-5 错误率**：只要模型预测出的前 5 个类别中包含正确答案，就算 **预测成功**。

👉 例如：
如果模型对某张图片的输出是：

```text
1. dog
2. cat
3. horse
4. frog
5. deer
```

而真实类别是 **frog**，则：

- **Top-1** 是错的（第一名是 dog）
- **Top-5** 是对的（frog 在前五名中）

------

✅ 总结

这张图展示了深度学习模型在 ImageNet 上的飞跃式进展，尤其是：

- AlexNet（2012）首次使用深度 CNN，大幅改进结果；
- ResNet（2015）提出残差网络，首次超过人类；
- GoogLeNet 和 VGG 也都是代表性的经典 CNN 架构。

**Top-5 错误率低，意味着模型更稳定、鲁棒性更强**。在工业界、比赛中经常作为重要评价指标。

如果你想，我可以进一步解释每个模型架构的改进要点。



## 2.5 Tiny ImageNet图像分类项目

现在用的chatgpt架构都是transformer模型, Transformer包括编码器和解码器两大部分，其中编码器负责将输入序列映射为隐藏表示，而解码器则利用这些隐藏表示来生成输出序列。<mark>transformer看起来是否与ASCII编码相似</mark>。

![image-20250224172229492](https://raw.githubusercontent.com/GMyhf/img/main/img/image-20250224172229492.png)



基于 PyTorch 和 torchvision 库的示例代码，我们演示如何加载Tiny ImageNet 数据集、构建基于预训练 ResNet50 模型的神经网络，并进行微调训练实现图像分类。ImageNet 数据集体积较大，我们使用Tiny ImageNet，即是这样，训练时还需要充足的存储和计算资源，实际使用时需要正确设置数据路径、batch size 和其他超参数。

起因是书中提到

![image-20250224172024784](https://raw.githubusercontent.com/GMyhf/img/main/img/image-20250224172024784.png)



![image-20250224172055496](https://raw.githubusercontent.com/GMyhf/img/main/img/image-20250224172055496.png)

**Q1. ResNet50?**

> ResNet-50是一种深度卷积神经网络，属于ResNet（Residual Network）系列模型之一。它由微软研究院的Kaiming He等人在2015年提出，旨在解决训练非常深的神经网络时遇到的梯度消失或梯度爆炸问题。ResNet通过引入“残差学习单元”（residual learning units 或者 shortcut connections/跳跃连接），使得网络可以更深且更容易训练。
>
> **ResNet-50结构特点**
>
> - **深度**: ResNet-50是一个包含50层的深层网络（包括卷积层、激活函数层、池化层等），因此得名ResNet-50。
> - **残差块(Residual Block)**: 核心概念是残差学习，通过添加直连边（shortcut connections），让输入直接跳过一层或多层传到后面层，这有助于解决深层网络中的退化问题（即随着网络深度增加，训练错误率反而上升的问题）。公式表达为：\[y = F(x, \{W_i\}) + x\]，其中\(x\)和\(y\)分别是层的输入和输出，\(F(x, \{W_i\})\)表示学习到的残差映射。
> - **具体架构**:
>   - 输入图像尺寸通常是224x224。
>   - 网络开始于一个7x7的卷积层，接着是最大池化层。
>   - 接下来是4个主要的残差阶段，每个阶段分别包含了3、4、6和3个残差块（对于ResNet-50），每个残差块内部有两个3x3的卷积层。
>   - 在最后一个残差阶段之后，使用全局平均池化层和一个全连接层进行分类任务，默认设置下用于ImageNet数据集的分类，有1000个输出节点。
>
> **应用场景**
>
> 由于其优秀的性能和相对合理的计算需求，ResNet-50广泛应用于各种计算机视觉任务中，如图像分类、目标检测、人脸识别等。此外，它也常被用作迁移学习的基础模型，在特定领域的小型数据集上进行微调以达到较好的效果。
>
> ResNet的成功推动了深度学习特别是计算机视觉领域的快速发展，后续还出现了更深的版本如ResNet-101和ResNet-152，以及针对效率优化的变体如ResNeXt。



### 1.准备Tiny ImageNet数据集

Tiny ImageNet。它包含 200 个类别，每个类别 500 张训练图片，总数据量大约 500MB，非常适合实验和调试。

下载 `wget http://cs231n.stanford.edu/tiny-imagenet-200.zip`，记237MB。

验证集通常解压后所有图片会在同一个文件夹中，而 ImageFolder 要求每个类别有独立子文件夹。你需要根据官方提供的 验证集标签文件，如 val_annotations.txt，对图片进行分类整理。常见的做法是编写一个脚本，根据文件中的类别信息将图片移动到对应的子文件夹中。

`tinyimagenet.sh`

```sh
#!/bin/bash

# download and unzip dataset
#wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip

current="$(pwd)/tiny-imagenet-200"

# training data
cd $current/train
for DIR in $(ls); do
   cd $DIR
   rm *.txt
   mv images/* .
   rm -r images
   cd ..
done

# validation data
cd $current/val
annotate_file="val_annotations.txt"
length=$(cat $annotate_file | wc -l)
for i in $(seq 1 $length); do
    # fetch i th line
    line=$(sed -n ${i}p $annotate_file)
    # get file name and directory name
    file=$(echo $line | cut -f1 -d" " )
    directory=$(echo $line | cut -f2 -d" ")
    mkdir -p $directory
    mv images/$file $directory
done
rm -r images
echo "done"

```



运行`sh tinyimagenet.sh`，数据解压并分类准备好，记472MB。

```
% ls -l
total 5200
drwxrwxr-x    3 hfyan  staff       96 Dec 12  2014 test
drwxrwxr-x  202 hfyan  staff     6464 Dec 12  2014 train
drwxrwxr-x  203 hfyan  staff     6496 Feb 24 11:09 val
-rw-rw-r--    1 hfyan  staff     2000 Feb  9  2015 wnids.txt
-rw-------    1 hfyan  staff  2655750 Feb  9  2015 words.txt
(base) hfyan@HongfeideMac-Studio tiny-imagenet-200 % pwd
/Users/hfyan/data/tiny-imagenet-200

```



### 2.查看本地机器Mac Studio的配置

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/202502160845440.png" alt="image-20250216084507879" style="zoom:50%;" />

Apple M1 Ultra 是 Apple 芯片系列中的一员，专为高性能需求设计，特别是在 Mac Studio 等设备中使用。M1 Ultra 的配置包括了中央处理器（CPU）、图形处理器（GPU）以及统一内存架构（Unified Memory Architecture, UMA），其中统一内存可供 CPU、GPU 以及其他组件共享。

**关于 GPU 内存**

在 M1 Ultra 中， 64GB 内存实际上是整个系统共享的统一内存容量，这意味着这64GB内存是由CPU、GPU及其他组件共同使用的，而不是专门分配给GPU的独立内存。 这种设计极大地提高了灵活性和性能表现，尤其是在处理复杂图形任务或多任务处理场景下。 

- **统一内存架构**：Apple 的设计理念是通过统一内存架构来提升性能和效率。这种架构允许 GPU 和 CPU 访问相同的内存池，减少了数据复制的需求，并且可以更灵活地根据需要分配内存资源。

- **M1 Ultra 的 GPU 资源**：M1 Ultra 配备了一个强大的 48 核心 GPU。尽管没有“专用”的 GPU 显存，但其可以从整个 64GB 统一内存中获取所需的工作内存。这对于许多图形密集型应用来说是非常有利的，因为它避免了传统显存与主存之间可能存在的瓶颈。

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/202502160845741.png" alt="image-20250216084526735" style="zoom:50%;" />



### 3. 训练模型

基于 PyTorch 和 torchvision 库的示例代码，该代码演示了如何加载 ImageNet 数据集、构建基于预训练 ResNet 模型的神经网络，并进行微调训练实现图像分类。

在clab.pku云端虚拟机，内存只有4GB，无GPU，跑不起来。以下是在我本地mac机器运行的。

> “Killed” 通常是操作系统（Linux 内核）出于内存不足（OOM）的原因终止了进程。这可能是由于以下原因引起的：
>
> - **内存不足**：程序在训练过程中占用了过多内存，超出了系统可用内存或交换空间（swap）的限制。
> - **GPU 内存不足**：如果使用 GPU 训练，可能也会出现 GPU 内存溢出的问题。

`tiny_imagenet_resnet50_epoch25.py`

```python
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms

# 训练和验证函数
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25, device='cpu'):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # 每个 epoch 分为训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置为训练模式
            else:
                model.eval()   # 设置为评估模式

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()  # 梯度清零

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 仅在训练阶段反向传播与参数更新
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            # MPS 后端不支持 float64 运算。解决方法是使用 float32，即调用 .float()。
            #epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 保存最佳模型参数
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    print('Best val Acc: {:.4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model

def main():
    # 1. 数据预处理与加载
    # 注意：此处假定ImageNet数据集按照train/val文件夹分别存放各类别图片，
    # 且每个类别作为一个子文件夹存在
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),           # 随机裁剪为224×224
            transforms.RandomHorizontalFlip(),           # 随机水平翻转
            transforms.ToTensor(),                         # 转为Tensor
            transforms.Normalize([0.485, 0.456, 0.406],    # ImageNet均值
                                 [0.229, 0.224, 0.225])      # ImageNet标准差
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),                           # 中心裁剪
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }

    # Tiny ImageNet 数据路径
    data_dir = '/Users/hfyan/data/tiny-imagenet-200'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                data_transforms[x])
                      for x in ['train', 'val']}

    # 设置 num_workers 为 4 以利用多进程数据加载
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                    batch_size=128,    # 可根据实际情况调整
                                                    shuffle=True,
                                                    num_workers=8)
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 使用 MPS 作为 GPU 后端（适用于 Apple Silicon）
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device for GPU acceleration")
    else:
        device = torch.device("cpu")
        print("MPS device not available, using CPU")

    #2. 构建模型（使用预训练 ResNet50）
    # 这里我们加载预训练的 ResNet50 模型，并修改最后的全连接层以适应Tiny ImageNet的类别数（200类）
    model_ft = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # 学习率调整策略，每7个epoch降低一次学习率
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    #3. 训练模型
    num_epochs = 25  # 可根据需要调整epoch数量
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           dataloaders, dataset_sizes, num_epochs=num_epochs, device=device)

    #4. 保存模型，文件名建议为 tiny_imagenet_resnet50_epoch25.pth
    torch.save(model_ft.state_dict(), 'tiny_imagenet_resnet50_epoch25.pth')
    print("Model saved as tiny_imagenet_resnet50_epoch25.pth")

if __name__ == '__main__':
    main()

```

> **说明**
>
> - **数据预处理**
>   使用了 `transforms` 对数据进行了数据增强（如随机裁剪、水平翻转）以及归一化（ImageNet常用的均值和标准差）。数据文件夹需要符合 `ImageFolder` 的要求，每个类别存放在独立的子文件夹中。
> - **模型构建**
>   本示例中采用预训练的 ResNet50 模型，并修改了最后一层全连接层以输出与类别数匹配的概率分布。
> - **训练过程**
>   代码中定义了 `train_model` 函数，对模型进行训练和验证，并在验证集上选取准确率最高的模型参数。学习率调度器用于逐步降低学习率以便更好地收敛。
> - **注意事项**
>   - ImageNet 数据集较大，建议在使用时注意数据加载、内存管理和训练时长。
>   - 如需更深入的模型调优或使用分布式训练，请参考 PyTorch 官方文档和相关资料。
>
> 该示例代码为入门级示例，实际项目中可能需要更多的优化和配置。
>
> 
>
> **主入口保护**：所有涉及多进程或多线程的代码都封装在 `if __name__ == '__main__':` 下，避免 macOS 下的启动问题。



**Q. 为什么每次运行都重新下载resnet50?**

为什么每次运行都重新下载 Downloading: "https://download.pytorch.org/models/resnet50-11ad3fa6.pth" to /Users/hfyan/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth ，下载过了一次了。

> PyTorch 会将预训练模型的权重下载到默认缓存目录（通常是 `~/.cache/torch/hub/checkpoints`），如果每次运行都重新下载，可能有以下几个原因：
>
> 1. **缓存目录权限或环境变量问题**  
>    请检查 `/Users/hfyan/.cache/torch/hub/checkpoints/` 目录是否存在、是否有写入权限以及文件是否完整。如果该目录不可写或文件被删除，PyTorch 就无法使用缓存文件。你可以尝试设置环境变量 `TORCH_HOME`，将缓存目录指定到一个你确定有权限并且持久的目录：
>    ```bash
>    export TORCH_HOME=/Users/hfyan/torch_cache
>    ```
>    然后再运行程序。
>
> 2. **运行环境重置**  
>    如果你每次运行程序时使用的是临时环境（例如某些 IDE 的沙盒模式、容器或虚拟环境在退出后清除缓存），那么每次启动时缓存文件可能都会被清空，从而导致重新下载。
>
> 3. **下载文件校验问题**  
>    PyTorch 在加载权重时会检查文件的完整性。如果文件损坏或校验失败，也会重新下载。
>
> 建议先确认缓存目录中是否已经存在 `resnet50-11ad3fa6.pth` 文件，且该文件大小和校验值是否正确。如果有权限问题或缓存目录被重置，可以通过设置 `TORCH_HOME` 环境变量来指定一个稳定的缓存目录，从而避免每次运行时重新下载。



#### 2025/2/24 11:30开始运行，16:00结束

>  File "/Users/hfyan/data/tiny_imagenet_resnet50_epoch25.py", line 47, in train_model
>
>  epoch_acc = running_corrects.double() / dataset_sizes[phase]
>
>  TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.
>
>  (base) hfyan@HongfeideMac-Studio data % python tiny_imagenet_resnet50_epoch25.py 

```
(base) hfyan@HongfeideMac-Studio data % python tiny_imagenet_resnet50_epoch25.py 
Using MPS device for GPU acceleration
Epoch 1/25
----------
train Loss: 5.0366 Acc: 0.0720
val Loss: 4.1348 Acc: 0.2819

Epoch 2/25
----------
train Loss: 3.2563 Acc: 0.3406
val Loss: 1.7006 Acc: 0.6197

Epoch 3/25
----------
train Loss: 2.1834 Acc: 0.5065
val Loss: 1.2068 Acc: 0.7062

Epoch 4/25
----------
train Loss: 1.8635 Acc: 0.5663
val Loss: 1.0010 Acc: 0.7498

Epoch 5/25
----------
train Loss: 1.6788 Acc: 0.6029
val Loss: 0.8927 Acc: 0.7702

Epoch 6/25
----------
train Loss: 1.5723 Acc: 0.6268
val Loss: 0.8407 Acc: 0.7808

Epoch 7/25
----------
train Loss: 1.5044 Acc: 0.6390
val Loss: 0.7990 Acc: 0.7907

Epoch 8/25
----------
train Loss: 1.4324 Acc: 0.6567
val Loss: 0.7788 Acc: 0.7939

Epoch 9/25
----------
train Loss: 1.4212 Acc: 0.6571
val Loss: 0.7701 Acc: 0.7981

Epoch 10/25
----------
train Loss: 1.4054 Acc: 0.6614
val Loss: 0.7669 Acc: 0.7966

Epoch 11/25
----------
train Loss: 1.4035 Acc: 0.6615
val Loss: 0.7634 Acc: 0.7980

Epoch 12/25
----------
train Loss: 1.3995 Acc: 0.6626
val Loss: 0.7595 Acc: 0.7990

Epoch 13/25
----------
train Loss: 1.3882 Acc: 0.6647
val Loss: 0.7558 Acc: 0.7988

Epoch 14/25
----------
train Loss: 1.3747 Acc: 0.6680
val Loss: 0.7517 Acc: 0.7997

Epoch 15/25
----------
train Loss: 1.3754 Acc: 0.6683
val Loss: 0.7490 Acc: 0.8006

Epoch 16/25
----------
train Loss: 1.3685 Acc: 0.6689
val Loss: 0.7592 Acc: 0.7970

Epoch 17/25
----------
train Loss: 1.3771 Acc: 0.6681
val Loss: 0.7567 Acc: 0.8009

Epoch 18/25
----------
train Loss: 1.3690 Acc: 0.6688
val Loss: 0.7508 Acc: 0.8011

Epoch 19/25
----------
train Loss: 1.3716 Acc: 0.6694
val Loss: 0.7521 Acc: 0.8008

Epoch 20/25
----------
train Loss: 1.3729 Acc: 0.6687
val Loss: 0.7527 Acc: 0.8002

Epoch 21/25
----------
train Loss: 1.3709 Acc: 0.6689
val Loss: 0.7501 Acc: 0.8014

Epoch 22/25
----------
train Loss: 1.3706 Acc: 0.6708
val Loss: 0.7516 Acc: 0.8008

Epoch 23/25
----------
train Loss: 1.3681 Acc: 0.6696
val Loss: 0.7502 Acc: 0.8002

Epoch 24/25
----------
train Loss: 1.3725 Acc: 0.6698
val Loss: 0.7508 Acc: 0.8003

Epoch 25/25
----------
train Loss: 1.3708 Acc: 0.6696
val Loss: 0.7480 Acc: 0.8004

Best val Acc: 0.8014
Model saved as tiny_imagenet_resnet50_epoch25.pth

```

跑了4小时30分钟。

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20250224171542842.png" alt="image-20250224171542842" style="zoom:50%;" />



```
% ls -lh *.pth
-rw-r--r--  1 hfyan  staff    92M Feb 24 16:02 tiny_imagenet_resnet50_epoch25.pth
```





**Q.这个程序保存为什么文件名字好？**

通常建议文件名能够反映出数据集、模型结构以及训练轮数等关键信息，这样便于后续管理和调试。比如你可以保存为：

```python
tiny_imagenet_resnet50_epoch25.pth
```

这里 "tiny_imagenet" 表示数据集，"resnet50" 表示模型结构，"epoch25" 则说明训练了 25 个轮次。



### 4.加载训练好的的模型并进行验证

前面已经保存了模型权重，可以通过如下步骤加载模型并在验证集上进行评估：

1. **加载模型结构和权重**  
   请确保你定义的模型结构与训练时保持一致。使用 `torch.load` 加载权重，并用 `model.load_state_dict` 导入。

2. **切换到评估模式**  
   调用 `model.eval()` 确保模型关闭 BatchNorm、Dropout 等训练时特有的行为。

3. **遍历验证数据并计算准确率**  
   使用 `torch.no_grad()` 关闭梯度计算，加快验证速度，并防止内存浪费。

下面是一个完整的示例代码，`eval_tiny_imagenet_resnet50_epoch25_pth.py `：

```python
import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

# 数据预处理
data_transforms = {
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

# 确保子进程安全启动
if __name__ == '__main__':
    data_dir = '/Users/hfyan/data/tiny-imagenet-200'
    val_dir = os.path.join(data_dir, 'val')
    val_dataset = datasets.ImageFolder(val_dir, data_transforms['val'])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64,
                                             shuffle=False, num_workers=4)

    # 选择设备
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device for GPU acceleration")
    else:
        device = torch.device("cpu")
        print("MPS device not available, using CPU")

    # 加载模型
    model_ft = models.resnet50(pretrained=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(val_dataset.classes))
    model_ft = model_ft.to(device)

    # 加载模型权重
    model_path = 'tiny_imagenet_resnet50_epoch25.pth'
    model_ft.load_state_dict(torch.load(model_path, map_location=device))

    # 评估模式
    model_ft.eval()

    # 模型评估
    running_corrects = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

    val_acc = running_corrects.float() / total_samples
    print('Validation Accuracy: {:.4f}'.format(val_acc))

```

---

**总结**

- **模型保存文件名建议**：  
  使用描述性文件名，例如 `tiny_imagenet_resnet50_epoch25.pth`，这样便于识别数据集、模型及训练轮次。

- **验证步骤**：  
  - 加载与你训练时一致的模型结构。  
  - 使用 `model.load_state_dict()` 加载权重。  
  - 调用 `model.eval()` 进入验证模式。  
  - 遍历验证数据集，计算准确率或其他指标。

这样，你就可以加载已保存的模型并对验证集数据进行测试。

![image-20250224171857564](https://raw.githubusercontent.com/GMyhf/img/main/img/image-20250224171857564.png)



```python
(base) hfyan@HongfeideMac-Studio data % python eval_tiny_imagenet_resnet50_epoch25_pth.py 
Using MPS device for GPU acceleration
/Users/hfyan/miniconda3/lib/python3.10/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/Users/hfyan/miniconda3/lib/python3.10/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=None`.
  warnings.warn(msg)
Validation Accuracy: 0.8014
(base) hfyan@HongfeideMac-Studio data % 
```



### 本部分QA

**Q.resnet50 图像分类，精度达到多少？**

ResNet50 的精度根据不同的训练设置和数据集有所变化。以下是几个不同情况下的准确率：

1. 原始的 ResNet-50 模型在 ImageNet 数据集上的 Top-1 准确率大约为 76.5%（根据原始论文中的结果）。

2. 在一些优化和改进后，比如使用一系列技巧（tricks），可以将 ResNet-50 的 Top-1 准确率提升到大约 79.1%。

3. 使用 SSLD（Simple Semi-supervised Label Distillation）知识蒸馏策略，并结合 ImageNet22k 数据集进行训练，可以进一步提高模型性能，使得 ResNet-50_vd 的 Top-1 准确率达到 82.4%。

4. 更进一步的改进，如采用特定的数据增强、分辨率调整以及基于 Fix 策略的微调等方法，可以在 ImageNet 验证集上达到更高的精度。例如，PaddleClas 文档中提到，通过这些优化措施，ResNet50_vd 的精度达到了 83%，而在 320x320 尺度下测试时精度可达 83.7%。最终，在采用了特定的 Fix 策略之后，精度甚至可以达到 84.0%。

5. 另外，有文献报道了通过知识蒸馏等方法，可以让 ResNet-50 达到 82.8% 的 Top-1 准确率。

请注意，上述数值反映了在标准 ImageNet 数据集上的表现，实际应用中针对特定任务或数据集的精度可能会有所不同。如果你要针对自己的数据集进行训练，可能需要根据具体情况调整网络结构、训练参数或使用迁移学习等技术来提高分类精度。



**Q.ImageNet和Tiny ImageNet的数据规模**

**ImageNet**  

- **类别数**：ImageNet的常规版本（如ILSVRC挑战使用的数据）包含 **1000个类别**。  
- **每类图片数**：  
  - **训练集**：大约 **1200张图片/类**。  
  - **验证集**：每类 **50张图片**。  
  - **测试集**：每类 **100张图片**。  
- **总图片数**：  
  - 训练集约 **128万张图片**。  
  - 验证集 **50,000张图片**。  
  - 测试集 **100,000张图片**。  
- **硬盘空间**：原始ImageNet的完整数据集大约 **150GB**。解压后可能更大。  

---

**Tiny ImageNet**  

- **类别数**：**200个类别**。  
- **每类图片数**：  
  - **训练集**：每类 **500张图片**。  
  - **验证集**：每类 **50张图片**。  
  - **测试集**：每类 **50张图片**。  
- **总图片数**：  
  - 训练集 **100,000张图片**。  
  - 验证集 **10,000张图片**。  
  - 测试集 **10,000张图片**。  
- **图片尺寸**：所有图片均为 **64x64像素**。  
- **硬盘空间**：大约 **250MB**。  

---

**总结**：

- **ImageNet** 是大规模数据集（150GB+），适合训练深层模型。  
- **Tiny ImageNet** 是简化版，250MB，非常适合快速测试模型原型或进行学术实验。  



### 附录

#### 多进程 vs. 多线程

**多进程**：`multiprocessing` 模块允许你创建多个独立的进程。每个进程都有自己的Python解释器实例和内存空间，这意味着它们不会受到GIL（Global Interpreter Lock，全局解释器锁）的影响，并且可以充分利用多核处理器的能力进行真正的并行计算。然而，进程间的通信和数据交换（如共享状态或传递消息）相比线程会更加复杂和消耗资源。

**多线程**：另一方面，Python 的 `threading` 模块提供了多线程的支持。在同一个程序内，你可以启动多个线程，这些线程共享相同的内存空间。尽管这使得线程间通信更加简单直接，但由于GIL的存在，对于CPU密集型任务，多线程并不能带来真正的并行执行，它更适合于I/O密集型的任务（例如网络请求、文件读写等），在这种情况下，线程可以在等待I/O操作完成的同时让出执行权给其他线程。

因此，如果你正在处理需要大量CPU计算的任务，并希望利用多核处理器提高性能，那么使用 `multiprocessing` 模块是更合适的选择。而对于涉及大量等待外部资源（如数据库访问、网络请求等）的应用场景，`threading` 可能更为适用。





#### 示例18161: 矩阵运算

matrices, http://cs101.openjudge.cn/practice/18161



##### 多进程multiprocessing

使用 multiprocessing 模块以及 dot_product 函数来实现矩阵运算 A·B + C。将利用多进程并行计算矩阵乘法部分，然后将结果与矩阵 C 相加。

内存: 29964，时间: 520ms

```python
import multiprocessing

def dot_product(row, col):
    """
    计算两个向量的点积。
    :param row: 第一个矩阵的一行
    :param col: 第二个矩阵的一列
    :return: 点积结果
    """
    return sum(a * b for a, b in zip(row, col))

def matrix_multiply_parallel(A, B, num_processes=None):
    """
    使用多进程并行计算两个矩阵的乘积。
    :param A: 第一个矩阵，作为列表的列表
    :param B: 第二个矩阵，作为列表的列表
    :param num_processes: 要使用的进程数，默认为 None (自动决定)
    :return: 矩阵乘积的结果
    """
    if len(A[0]) != len(B):
        raise ValueError("Matrix dimensions do not match for multiplication")

    result = [[None for _ in range(len(B[0]))] for _ in range(len(A))]
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        tasks = []
        for i in range(len(A)):
            for j in range(len(B[0])):
                row = A[i]
                col = [B_element[j] for B_element in B]
                # 异步调用dot_product函数
                tasks.append(pool.apply_async(dot_product, (row, col)))
        
        index = 0
        for i in range(len(A)):
            for j in range(len(B[0])):
                #任务收集逻辑：通过直接使用 tasks[index].get() 来获取每个异步任务的结果
                result[i][j] = tasks[index].get()
                index += 1
    
    return result

def matrix_add(X, Y):
    if len(X) != len(Y) or len(X[0]) != len(Y[0]):
        raise ValueError("Matrices must have the same dimensions for addition")
    return [[X[i][j] + Y[i][j] for j in range(len(X[0]))] for i in range(len(X))]

def read_matrix():
    import sys
    input = sys.stdin.read
    data = input().strip().split('\n')
    matrices = []
    idx = 0
    while idx < len(data):
        row, col = map(int, data[idx].split())
        matrix = []
        for r in range(row):
            matrix.append(list(map(int, data[idx + 1 + r].split())))
        matrices.append(matrix)
        idx += row + 1
    return matrices

def main():
    matrices = read_matrix()
    A, B, C = matrices
    
    try:
        AB = matrix_multiply_parallel(A, B, 4)
        result = matrix_add(AB, C)
        for row in result:
            print(' '.join(map(str, row)))
    except ValueError:
        print("Error!")

if __name__ == "__main__":
    main()
```



##### 单进程

内存: 4392kB，时间: 71ms

```python
def read_matrix():
    """读取矩阵输入"""
    row, col = map(int, input().split())
    matrix = [list(map(int, input().split())) for _ in range(row)]
    return matrix

def matrix_multiply(A, B):
    """计算两个矩阵的乘积"""
    # 确保A的列数等于B的行数
    if len(A[0]) != len(B):
        raise ValueError("Matrix dimensions do not match for multiplication")
    
    result = []
    for i in range(len(A)):
        row = []
        for j in range(len(B[0])):
            sum = 0
            for k in range(len(A[0])):
                sum += A[i][k] * B[k][j]
            row.append(sum)
        result.append(row)
    return result

def matrix_add(X, Y):
    """计算两个矩阵的和"""
    # 确保X和Y的维度相同
    if len(X) != len(Y) or len(X[0]) != len(Y[0]):
        raise ValueError("Matrices must have the same dimensions for addition")
    
    result = [[X[i][j] + Y[i][j] for j in range(len(X[0]))] for i in range(len(X))]
    return result

def main():
    # 读取矩阵A、B、C
    A = read_matrix()
    B = read_matrix()
    C = read_matrix()

    try:
        # 计算矩阵乘法A·B
        AB = matrix_multiply(A, B)

        # 检查矩阵AB和C的维度是否适合相加
        if len(AB) != len(C) or len(AB[0]) != len(C[0]):
            raise ValueError("Resultant matrix AB and C must have the same dimensions for addition")

        # 计算矩阵加法AB+C
        result = matrix_add(AB, C)

        # 输出结果
        for row in result:
            print(' '.join(map(str, row)))
    except ValueError as e:
        print("Error!")

if __name__ == "__main__":
    main()
```



> Q. 开 multiprocessing，4线程，运行时间比普通方法计算反倒慢了一个数量级。是测试数据太少？
>
> 使用 `multiprocessing` 模块进行并行计算时，确实有可能在某些情况下（特别是处理小规模数据或简单任务）导致性能下降。这主要是因为：
>
> 1. **进程启动和通信开销**：创建新进程以及跨进程边界传递数据（如将任务分配给不同的工作进程）需要时间。如果任务本身非常轻量级（比如简单的矩阵乘法元素计算），这些额外的开销可能会超过并行化带来的性能增益。
>
> 2. **数据规模不足**：对于较小的数据集或简单的计算任务，CPU可能很快就能完成计算，此时多线程或多进程的优势无法体现出来。实际上，由于上述提到的进程管理和通信成本，反而可能导致总体执行时间增加。
>
> 3. **Python的全局解释器锁（GIL）**：尽管使用了 `multiprocessing` 可以绕过 GIL，但对于I/O密集型而非计算密集型任务，其优势并不明显。此外，在计算密集型任务中，如果任务粒度太细，频繁的上下文切换也可能影响性能。
>
> **测试数据大小的影响**
>
> 如果你使用的测试数据很小（例如样例输入中的3x1、1x2等小矩阵），那么并行化的开销会显得尤为突出。这是因为：
>
> - 并行化带来的加速比依赖于任务可以被有效分割的程度以及每个子任务的计算量。
> - 对于非常小的矩阵，单个点积操作的计算量不足以抵消进程间通信和管理的开销。
>
> **改进建议**
>
> 如果你想验证并行化是否能在更大规模的数据上带来性能提升，建议尝试以下方法：
>
> 1. **增加数据规模**：使用更大的矩阵进行测试，这样每个子任务（即两个向量的点积）会有更多的计算量，从而更有可能从并行化中受益。
>
> 2. **调整线程/进程数**：根据你的CPU核心数合理设置进程池的大小。过多的进程会导致不必要的上下文切换，而过少则不能充分利用多核处理器的能力。
>
> 3. **优化任务粒度**：确保你划分的任务足够大，以便能够掩盖进程间的通信和调度开销。例如，可以考虑将整个行的计算作为一个任务，而不是单独计算每个元素。
>
> 4. **测量性能**：使用计时工具（如 `time` 模块）来精确测量不同实现方式下的运行时间，并对比分析结果。
>
> 总之，对于小规模数据或简单计算任务，直接使用单线程实现通常更为高效。当处理大规模数据或复杂计算时，适当利用并行化技术才能显著提高性能。





#### Torch 的多进程计算不受 GIL 限制

https://github.com/GMyhf/2025spring-cs201/blob/main/LLM/nn_4_tiny_imagenet_classify.md

```python
...
    # 设置 num_workers 为 4 以利用多线程数据加载
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                    batch_size=128,    
                                                    shuffle=True,
                                                    num_workers=8)
...
```

GIL 的全称是 **Global Interpreter Lock**（全局解释器锁）。这是 Python 解释器（CPython）中的一个机制，用于确保同一时间只有一个线程执行 Python 字节码，即便在多核 CPU 上，也无法实现真正的并行计算。

不过，**Torch 的多近程计算** 通常 **不受 GIL 限制**，原因如下：

1. **计算密集型任务在 C/C++ 后端执行**  
   PyTorch 的底层计算是基于 C++ 和 CUDA 实现的。一旦计算任务进入底层库（如调用 `torch.matmul()`、`torch.nn` 等操作），它就脱离了 Python 的解释层，因此不会被 GIL 锁住。计算主要在 C++ 层面并行执行，或者交给 GPU 加速。

2. **DataLoader 使用多进程并行加载数据**  
   你设置的 `num_workers=8` 实际上会启动 8 个 **子进程** 来加载数据，因为 PyTorch 的 `DataLoader` 默认使用 **multiprocessing** 库，而不是线程池。这种方式是跨进程的，不会触发 GIL。

3. **异步计算（CUDA）**  
   如果你在使用 GPU 计算（如 `.to('cuda')`），大部分操作是异步的。Python 线程触发计算任务后，CUDA 内核在 GPU 上执行，线程会立刻返回，继续处理其他任务。因此不会卡在 GIL 上。

**总结：**

- **CPU-bound任务**（纯 Python 代码计算）会受到 GIL 限制。
- **I/O-bound任务**（如数据加载、文件读取）在 PyTorch 中主要靠多进程，不受 GIL 影响。
- **GPU-bound任务** 完全在 CUDA 上执行，与 GIL 无关。

所以 `num_workers=8` 是在开 8 个进程来并行加载数据，计算部分走的底层 C++/CUDA，不会因为 GIL 产生性能瓶颈！





# 第三部分 深度学习与大语言模型

阅读《从零构建大模型 (Sebastian Raschka, 塞巴斯蒂安·拉施卡, 覃立波, 冯晓骋, 刘乾)》。

代码可以本地或者云端运行，https://github.com/rasbt/LLMs-from-scratch

目前2025/7/21，github该项目获59.5k星。https://github.com/rasbt/LLMs-from-scratch

笔试内容《Test_Yourself_On_Build_a_Large_Language》



## 3.1 深度学习基础

- 数字角度看世界：一切皆数字，一切智能皆**函数**
  - 如何数值化：文字 → 词表编码，图片 → 像素 RGB，视频 → 连续的帧图片...
  - 图像检测：输入一张图片，输出其中是否包含某个物体、物体的边缘轮廓...
  - 文字、图像生成：输入一段文字，输出这段文字的续写/回答/文字描述的图片...
- 流程：输入（文/图/视频）→ 编码为数值化表示 → 函数进行一系列运算 → 得到数值输出 → 解码为所需格式
- **机器学习/深度学习（Machine Leaning/Deep Learning）**：使用（深层神经网络）模型来实现这些函数
- 早期的机器学习模型：线性分类器
  - 公式：$y=wx+b$
  - 寻找划分不同类别数据点的最佳超平面（$w, b$ 参数取值）
  - 能解决简单的分类任务

- 深度学习模型：神经网络（Neural Network）
  - 一个人工定义的计算图，每个节点设定 $w, b$ 参数，节点输入与这些参数做乘法，加法和非线性运算得到输出（公式：$Activation(wx+ b)$），按图结构向前传递直至得到最终输出。
  - 与生物学上大脑的神经元连接结构相似
  - 节点更多、层数更深（参数量更大）的神经网络能够拟合更复杂的函数（表达能力更强）
  - 数学上可证明神经网络可以拟合任何函数
    - https://www.zhihu.com/question/268384579
- 拟合的方法：训练（Training）
  - 用大量要拟合的函数产生的输入输出数据，使用优化算法，逐步调整模型的参数，直至对任意输入能得到足够接近真实输出的结果
  - 损失函数：用于计算模型输出与真实输出的差距
  - 优化算法：随机梯度下降法（SGD），Adams等
  - 训练不足/过度时会出现模型欠拟合/过拟合
  - 评测：在一组没有用来训练的数据上，测试模型输出的准确性

## 3.2 从 Transformer 到生成式大模型

- 自然语言处理（Natural Language Processing, NLP）：计算机科学中重要的研究领域
  - 多种任务：情感分类，机器翻译，文本摘要，...
  - 两大类：判别式 和 生成式
- 2017 年，Transformer 模型结构被提出
  - Encoder-Decoder 结构，可独立/组合使用
  - Encoder：token 序列（文本）进，向量出，适合判别式任务，代表工作：BERT（2018）
  - Decoder：token 序列（文本）进，每次出一个 token，和之前序列拼接算下一个token，循环直至生成了特殊的 <eos> token 结束，适合生成式任务，代表工作：GPT（2018）

- 自然语言可以表示为一种**概率分布**：
  - $P(<bos>北京<eos>)=P(北│<bos>)∗P(京│<bos>北)∗P(<eos>|<bos>北京)$​
  - 词元（Token）：自然语言按 词/子词/字 粒度切分后的单元
  - 一切文本生成任务（无论是创作诗歌、解决数学题、写代码...）都是 **Next Token Prediction** 任务
  - Transformer Decoder 就是为了解决这种任务的模型
- 语言模型的训练
  - 2018 之前：针对每个任务，构造有监督（Supervised）数据集，从零专门训练一个模型
    - 生成模型希望有多任务能力
  - 2018 年，BERT 和 GPT 论文提出 Pre-Training + Fine-Tuning 模式：
  - **预训练**：先在大规模的无监督语料上训练模型的通用语言建模能力
    - 方法：从互联网爬取海量文本，每个句子挖空填词 or 给前半句预测后半句
  - **微调**：然后在特定任务的监督数据集上继续训练模型，提升模型在任务上的性能
  - 在多个任务上的分数大幅超过 SOTA（state-of-the-art）结果

- “大”模型的体现
  - 参数量：BERT (2018): 334M（3.34亿），GPT-3 (2020): 175B（1750亿）, DeepSeek-V3 (2025): 671B（6710亿）
  - 预训练数据量：BERT (2018)：3.7B tokens，GPT-3 (2020)：500B tokens，Llama (2023)：1.4T tokens，DeepSeek-V3 (2025): 14.8T tokens
  - 从 2018 至今，模型的参数规模和训练数据量增长了数千倍
- 为什么需要大模型
  - 随着参数量和训练数据量的增大，模型在语言能力、知识记忆等方面的表现没有出现停滞或倒退，而是不断有提升
  - 涌现能力（Emergent Abilities）：在模型尺寸足够大/训练数据量足够多时，某些方面的能力有突然提升

- 计算硬件：GPU
  - 最初为图形计算设计，拥有强大矩阵算力，与神经的网络计算互通
  - 英伟达（NVIDIA）：行业龙头，最强硬件性能 + 沉淀多年的 CUDA 编程生态
    - RTX 系列：消费级GPU
    - A100/H100 系列：专业GPU
  - Intel Gaudi，华为昇腾等产品正在追赶

- 大模型的“量化”
  - 大部分模型的参数是 float16/bfloat16 格式，每个参数 2 字节（16 bit）
    - DeepSeek-V3/R1 671B 满血版除外，纯 fp8 训练

  - 模型推理需要的显存约为 2.5x~3x 参数量
    - 例：1.5B 模型 ≈ 3.7G - 4.5G，7B 模型 ≈ 17.5 - 21G

  - 模型量化（Quantization）：将参数近似存储为更低字节
    - 小幅牺牲效果，大幅节省显存
    - INT4 = 4 bit 量化，INT8 = 8 bit 量化
    - 量化后模型需要的显存相应除字节压缩倍数

- 当前的主流大模型（2025.02）
  - 商业化模型
    - OpenAI：ChatGPT，GPT-4o，GPT-o1/o3
    - Google：Gemini 2.0 Pro
    - Anthropic：Claude 3.5 Sonnet
    - 百度：文心一言4.0；
    - 月之暗面：Kimi v1, k1.5
  - 开源模型
    - Meta AI (Facebook)：LLaMA-3, 3.1, 3.2, 3.3
    - 阿里：通义千问 Qwen-2.5, 2.5 Coder
    - 深度求索：DeepSeek V3/R1, Distill-Qwen2.5, Distill-Llama3
  - 竞争点：评测指标（数学，代码等能力），长上下文，多模态，长思考，价格

## 3.3 大模型本地部署、API 调用

- 本地部署大模型
  - LM Studio：图形化界面，一键下载运行模型
  - Ollama (+ Open WebUI)
    - 注意 tag，ollama 默认下载的是量化模型
  - HuggingFace：最流行的深度学习开源社区，AI 圈的 Github

- 通过 API 调用大模型
  - 为什么需要 API：批量调用，基于大模型调用构建上层应用等
  - OpenAI API 规范：在 ChatGPT 爆火后迅速成为大模型调用的事实标准，所有云平台/本地部署工具都会提供 OpenAI API 格式的调用服务
  - 当前主流文本生成模型的调用参考 Chat Completion 接口的输入输出即可
  - 两种具体调用方法：
    - 使用 OpenAI 库，client.chat.completions 对象
    - 本质是 HTTP 接口，通过 requests 库或 curl 自行组装 json 直接发请求也可以
  - 文档：https://platform.openai.com/docs/api-reference/chat/create