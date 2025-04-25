📄 中文 | [English](./README.md)

<div align="center">
    <h1>
        <img src="./fire.svg" width=30, height=30>
        𝚃𝚎𝚡𝚃𝚎𝚕𝚕𝚎𝚛
        <img src="./fire.svg" width=30, height=30>
    </h1>

  [![](https://img.shields.io/badge/API-文档-orange.svg?logo=read-the-docs)](https://oleehyo.github.io/TexTeller/)
  [![](https://img.shields.io/badge/docker-镜像-green.svg?logo=docker)](https://hub.docker.com/r/oleehyo/texteller)
  [![](https://img.shields.io/badge/数据-Texteller1.0-brightgreen.svg?logo=huggingface)](https://huggingface.co/datasets/OleehyO/latex-formulas)
  [![](https://img.shields.io/badge/权重-Texteller3.0-yellow.svg?logo=huggingface)](https://huggingface.co/OleehyO/TexTeller)
  [![](https://img.shields.io/badge/协议-Apache_2.0-blue.svg?logo=github)](https://opensource.org/licenses/Apache-2.0)

</div>

https://github.com/OleehyO/TexTeller/assets/56267907/532d1471-a72e-4960-9677-ec6c19db289f

TexTeller 是一个端到端的公式识别模型，能够将图像转换为对应的 LaTeX 公式。

TexTeller 使用 **8千万图像-公式对** 进行训练（前代数据集可在此[获取](https://huggingface.co/datasets/OleehyO/latex-formulas)），相较 [LaTeX-OCR](https://github.com/lukas-blecher/LaTeX-OCR) 使用的 10 万量级数据集，TexTeller 具有**更强的泛化能力**和**更高的准确率**，覆盖绝大多数使用场景。

>[!NOTE]
> 如果您想对本项目提出反馈或建议，欢迎前往 [讨论区](https://github.com/OleehyO/TexTeller/discussions) 发起讨论。

---

<table>
<tr>
<td>

## 🔖 目录
- [快速开始](#-快速开始)
- [网页演示](#-网页演示)
- [服务部署](#-服务部署)
- [Python接口](#-python接口)
- [公式检测](#-公式检测)
- [模型训练](#️️-模型训练)

</td>
<td>

<div align="center">
  <figure>
    <img src="cover.png" width="800">
    <figcaption>
      <p>TexTeller 可识别的图像示例</p>
    </figcaption>
  </figure>
  <div>
  </div>
</div>

</td>
</tr>
</table>

## 📮 更新日志

- [2024-06-06] **TexTeller3.0 发布！** 训练数据增至 **8千万**（是 TexTeller2.0 的 **10倍** 并提升了数据多样性）。TexTeller3.0 新特性：

  - 支持扫描件、手写公式、中英文混合公式识别

  - 支持印刷体中英文混排公式的OCR识别

- [2024-05-02] 支持**段落识别**功能

- [2024-04-12] **公式检测模型**发布！

- [2024-03-25] TexTeller2.0 发布！TexTeller2.0 的训练数据增至750万（是前代的15倍并提升了数据质量）。训练后的 TexTeller2.0 在测试集中展现了**更优性能**，特别是在识别罕见符号、复杂多行公式和矩阵方面表现突出。

  > [此处](./assets/test.pdf) 展示了更多测试图像及各类识别模型的横向对比。

## 🚀 快速开始

1. 安装uv：

   ```bash
   pip install uv
   ```

2. 安装项目依赖：

   ```bash
   uv pip install texteller
   ```

3. 若使用 CUDA 后端，可能需要安装 `onnxruntime-gpu`：

   ```bash
   uv pip install texteller[onnxruntime-gpu]
   ```

4. 运行以下命令开始推理：

   ```bash
   texteller inference "/path/to/image.{jpg,png}"
   ```

   > 更多参数请查看 `texteller inference --help`

## 🌐 网页演示

命令行运行：

```bash
texteller web
```

在浏览器中输入 `http://localhost:8501` 查看网页演示。

> [!NOTE]
> 段落识别无法还原文档结构，仅能识别其内容。

## 🖥️ 服务部署

我们使用 [ray serve](https://github.com/ray-project/ray) 为 TexTeller 提供 API 服务。启动服务：

```bash
texteller launch
```

| 参数 | 说明 |
| --------- | -------- |
| `-ckpt` | 权重文件路径，*默认为 TexTeller 预训练权重* |
| `-tknz` | 分词器路径，*默认为 TexTeller 分词器* |
| `-p` | 服务端口，*默认 8000* |
| `--num-replicas` | 服务副本数，*默认 1*。可使用更多副本来提升吞吐量 |
| `--ncpu-per-replica` | 单个副本使用的CPU核数，*默认 1* |
| `--ngpu-per-replica` | 单个副本使用的GPU数，*默认 1*。可设置为0~1之间的值来在单卡上运行多个服务副本共享GPU，提升GPU利用率（注意，若--num_replicas为2，--ngpu_per_replica为0.7，则需有2块可用GPU） |
| `--num-beams` | beam search的束宽，*默认 1* |
| `--use-onnx` | 使用Onnx Runtime进行推理，*默认关闭* |

向服务发送请求：

```python
# client_demo.py

import requests

server_url = "http://127.0.0.1:8000/predict"

img_path = "/path/to/your/image"
with open(img_path, 'rb') as img:
    files = {'img': img}
    response = requests.post(server_url, files=files)

print(response.text)
```

## 🐍 Python接口

我们为公式OCR场景提供了多个易用的Python API接口，请参考[接口文档](https://oleehyo.github.io/TexTeller/)了解对应的API接口及使用方法。

## 🔍 公式检测

TexTeller的公式检测模型在3415张中文资料图像和8272张[IBEM数据集](https://zenodo.org/records/4757865)图像上训练。

<div align="center">
    <img src="./det_rec.png" width=250>
</div>

我们在Python接口中提供了公式检测接口，详见[接口文档](https://oleehyo.github.io/TexTeller/)。

## 🏋️‍♂️ 模型训练

请按以下步骤配置训练环境：

1. 安装训练依赖：

   ```bash
   uv pip install texteller[train]
   ```

2. 克隆仓库：

   ```bash
   git clone https://github.com/OleehyO/TexTeller.git
   ```

### 数据集准备

我们在`examples/train_texteller/dataset/train`目录中提供了示例数据集，您可按照示例数据集的格式放置自己的训练数据。

### 开始训练

在`examples/train_texteller/`目录下运行：

   ```bash
   accelerate launch train.py
   ```

训练参数可通过[`train_config.yaml`](./examples/train_texteller/train_config.yaml)调整。

## 📅 计划列表

- [X] ~~使用更大规模数据集训练模型~~
- [X] ~~扫描件识别支持~~
- [X] ~~中英文场景支持~~
- [X] ~~手写公式支持~~
- [ ] PDF文档识别
- [ ] 推理加速

## ⭐️ 项目星标

[![Star增长曲线](https://starchart.cc/OleehyO/TexTeller.svg?variant=adaptive)](https://starchart.cc/OleehyO/TexTeller)

## 👥 贡献者

<a href="https://github.com/OleehyO/TexTeller/graphs/contributors">
   <a href="https://github.com/OleehyO/TexTeller/graphs/contributors">
      <img src="https://contrib.rocks/image?repo=OleehyO/TexTeller" />
   </a>
</a>
