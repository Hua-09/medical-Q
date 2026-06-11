# 医疗问答系统

一个基于自然语言处理技术实现的医疗问答课程大作业。项目从命令行检索式问答开始，逐步扩展为带聊天式可视化页面的医疗问答系统，并支持打包为 Windows `.exe` 程序运行。

系统包含两种核心能力：

- **本地检索问答**：基于医疗问答数据集，使用中文分词、停用词过滤、TF-IDF、文本向量化、余弦相似度和 Top-K 召回完成检索式问答。
- **在线/混合问答**：调用千问 API 生成更自然的医疗咨询回复；混合模式会先进行本地检索，再把检索证据交给千问整理回答。

> 注意：本系统仅用于自然语言处理课程学习和技术展示，不能替代医生诊断、处方或急救判断。

## 项目特点

- 支持命令行交互版，便于展示 NLP 检索流程。
- 支持聊天式 Web UI，界面类似常见 AI 聊天应用。
- 支持打包为 Windows `.exe` 程序。
- 默认使用内科 + 外科数据，也可在 UI 中切换其他科室或全部科室。
- 本地索引后台加载，页面顶部显示加载进度条。
- 支持本地检索模式、在线问答模式、混合模式。
- 支持语音输入。
- 支持对回答进行手动语音播报。
- 支持最近对话历史记录。
- 在线模式使用严格医疗安全 prompt，并对模型输出进行格式清洗。

## 数据来源

本项目使用公开医疗问答数据集：

- 数据集名称：Chinese-medical-dialogue-data
- 数据规模：原始数据约 79 万条医患问答
- 数据格式：CSV
- 科室范围：内科、外科、妇产科、儿科、男科、肿瘤科
- github下载：<https://github.com/Toyhom/Chinese-medical-dialogue-data>
- GitCode 镜像：<https://gitcode.com/gh_mirrors/ch/Chinese-medical-dialogue-data>
- HyperAI 下载：<https://go.hyper.ai/lM5sd>

当前项目数据位于：

```text
Data_数据/
```

目录结构：

```text
Data_数据/
  IM_内科/
    内科5000-33000.csv
    内科.txt
    数据处理.py
  Surgical_外科/
    外科5-14000.csv
  OAGD_妇产科/
    妇产科6-28000.csv
  Pediatric_儿科/
    儿科5-14000.csv
  Andriatria_男科/
    男科5-13000.csv
  Oncology_肿瘤科/
    肿瘤科5-10000.csv
```

由于数据量太大，系统默认加载：

```text
IM_内科
Surgical_外科
```

即 **内科 + 外科**。

## 技术路线

### 本地检索流程

本地检索问答使用经典检索式问答方案：

1. 读取医疗问答 CSV 数据。
2. 提取患者问题、问题标题和医生回复。
3. 使用 `jieba` 进行中文分词。
4. 使用停用词表过滤无意义词。
5. 加入常见医疗词汇，提高医学短语分词效果。
6. 使用 `TfidfVectorizer` 提取 TF-IDF 特征。
7. 将问答库文本和用户问题向量化。
8. 使用余弦相似度计算用户问题与历史问题的相似度。
9. 按相似度排序，返回 Top-K 候选结果。
10. 根据最高相似问题的医生回复生成本地参考回答。

### 在线与混合问答

在线模式调用千问 OpenAI 兼容接口。

混合模式流程：

1. 先执行本地 Top-K 召回。
2. 将召回的问题、患者描述、医生回复和相似度整理为检索上下文。
3. 将检索上下文交给千问模型。
4. 由模型生成更自然、结构更清晰的医疗咨询回答。
5. 如果千问 API 超时或失败，系统会自动返回本地检索结果作为兜底回答。

## 功能模块

### 第一阶段：命令行交互版

入口文件：

```text
medical_qa_cli.py
```

主要作用：

- 展示 NLP 检索式问答完整流程。
- 支持命令行连续提问。
- 支持单次查询。
- 支持 Top-K、相似度阈值、加载数据量等参数配置。

### 第二阶段：可视化问答系统与 exe

入口文件：

```text
medical_qa_app.py
```

页面文件：

```text
ui_preview.html
```

打包输出：

```text
dist/MedicalQA/MedicalQA.exe
```

第二阶段重点是将第一阶段的检索式问答系统封装成可运行、可交互、可展示的医疗问答应用。用户打开 exe 后，会进入聊天式问答页面，可以选择本地检索、在线问答或混合模式进行咨询。

第二阶段主要功能：

- 聊天式问答页面。
- 本地检索回答。
- 千问 API 在线回答。
- 本地检索 + 千问整理的混合回答。
- 本地索引后台加载与进度显示。
- 科室范围选择。
- 语音输入。
- 回答手动语音播报。
- 历史对话保存。
- API 设置状态展示。

## 项目结构

```text
medical_qa_cli.py        第一阶段命令行问答入口
medical_qa_runtime.py    问答系统运行时逻辑
medical_qa_app.py        Web 服务和 exe 程序入口
medical_prompt.py        千问在线模式医疗安全 prompt
qwen_client.py           千问 API 调用、重试、超时和输出清洗
qwen_config.py           千问 API Key、模型、超时等配置
ui_preview.html          聊天式前端页面
build_exe.ps1            exe 打包脚本
medical_qa_app.spec      PyInstaller 打包配置
Data_数据/               医疗问答 CSV 数据
tests/                   单元测试
dist/MedicalQA/          本地打包后生成的 exe 应用目录
README.md                项目说明文档
LICENSE                  许可证
```

## 环境要求

建议环境：

- Windows 10/11
- Python 3.10 或以上

主要依赖：

```powershell
pip install jieba scikit-learn joblib pyinstaller
```

依赖说明：

- `jieba`：中文分词。
- `scikit-learn`：TF-IDF、向量化、余弦相似度。
- `joblib`：本地索引缓存。
- `pyinstaller`：打包 Windows exe。

在线模式使用 Python 标准库访问千问 OpenAI 兼容接口，不额外依赖 `requests`。

## 第一阶段-python代码交互版使用方式

在项目根目录运行：

```powershell
python medical_qa_cli.py
```

进入命令行交互后，直接输入医疗问题即可，也可以进入pycharm,vs code里面进行运行，需要python环境。

示例：

```text
请输入您的医疗问题：
> 最近拉肚子并伴随腹痛，应该怎么办？
```

退出方式：

```text
退出
```

或：

```text
exit
```

单次查询：

```powershell
python medical_qa_cli.py --once "最近拉肚子并伴随腹痛，应该怎么办？"
```

调整 Top-K：

```powershell
python medical_qa_cli.py --top-k 5
```

调整相似度阈值：

```powershell
python medical_qa_cli.py --threshold 0.05
```

限制加载记录数，便于调试：

```powershell
python medical_qa_cli.py --max-records 5000
```

指定加载科室：

```powershell
python medical_qa_cli.py --departments IM_内科 Surgical_外科
```

## 第二阶段-UI界面交互和exe文件使用方式

### 方式一：运行源码版

在项目根目录运行：

```powershell
python medical_qa_app.py
```

默认会打开浏览器页面：

```text
http://127.0.0.1:7860
```

只启动服务，不自动打开浏览器：

```powershell
python medical_qa_app.py --no-browser --port 7860
```

限制加载记录数，便于调试：

```powershell
python medical_qa_app.py --max-records 5000
```

启动后，本地数据索引会在后台加载。页面顶部会显示加载进度，加载完成后即可使用本地检索模式和混合模式。在线问答模式不依赖本地索引，可以先使用。

### 方式二：运行 exe 版

打包完成后运行：

```powershell
.\dist\MedicalQA\MedicalQA.exe
```

也可以双击：

```text
dist/MedicalQA/MedicalQA.exe
```

注意不要只复制 `MedicalQA.exe` 单文件，必须保留整个目录：

```text
dist/MedicalQA/
```

因为 `_internal/` 中包含程序依赖、页面文件和数据文件。

## 可视化页面使用说明

打开第二阶段页面后，可以看到聊天式问答界面。

### 回答模式

页面底部提供三种模式：

- **本地检索模式**：只使用本地医疗问答数据，不调用千问 API。
- **在线问答**：直接调用千问 API。
- **混合模式**：先本地召回相似问答，再调用千问整理回答。

### 数据范围

点击“设置”，可以选择数据范围：

- 内科 + 外科
- 仅内科
- 仅外科
- 儿科
- 妇产科
- 男科
- 肿瘤科
- 全部科室

默认是：

```text
内科 + 外科
```

切换数据范围后，系统会在后台重新加载对应科室并构建索引。

### 本地索引进度

页面顶部有本地索引加载进度条，会显示类似阶段：

```text
准备加载本地索引
读取本地 CSV 数据
构建 TF-IDF 检索索引
保存索引缓存
本地索引已就绪
```

### 语音功能

语音输入：

- 点击输入框左侧麦克风按钮。
- 浏览器会请求麦克风权限。
- 识别结果会先进入输入框，确认后再发送。

语音播报：

- 回答生成后，回答下方会出现“播放回答”按钮。
- 点击后开始播报。
- 再次点击可以停止。

语音能力依赖浏览器的 Web Speech API，不同浏览器支持情况可能不同。

## 千问 API 配置

在线模式和混合模式中的在线生成部分需要配置千问 API。

配置文件：

```text
qwen_config.py
```

示例：

```python
QWEN_API_KEY = "请在这里填写你的千问API Key"
QWEN_MODEL = "qwen-plus"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_TEMPERATURE = 0.2
QWEN_TIMEOUT_SECONDS = 60
QWEN_MAX_RETRIES = 2
QWEN_MAX_TOKENS = 800
```

可选模型示例：

```python
QWEN_MODEL = "qwen-turbo"
QWEN_MODEL = "qwen-plus"
QWEN_MODEL = "qwen-max"
```

如果没有配置 API Key：

- 本地检索模式仍然可以正常使用。
- 在线问答和混合模式中的在线生成部分不可用或会自动降级。

不要公开真实 API Key。公开分享项目时，应将 `QWEN_API_KEY` 改为占位符。

## 打包方式

项目使用 PyInstaller 打包。

在 PowerShell 中执行：

```powershell
.\build_exe.ps1
```

如果 PowerShell 脚本执行受限，可以运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\build_exe.ps1
```

打包输出：

```text
dist/MedicalQA/MedicalQA.exe
```

修改以下文件后，如果要更新 exe，需要重新打包：

```text
medical_qa_app.py
medical_qa_runtime.py
medical_qa_cli.py
qwen_client.py
qwen_config.py
medical_prompt.py
ui_preview.html
Data_数据/
```

## 数据准备

为方便课程检查和下载后直接运行，本项目仓库中可以一并保留 `Data_数据/` 目录。下载或克隆项目后，请确认数据目录结构如下：

```text
Data_数据/
  IM_内科/
  Surgical_外科/
  OAGD_妇产科/
  Pediatric_儿科/
  Andriatria_男科/
  Oncology_肿瘤科/
```

系统默认加载：

```text
IM_内科
Surgical_外科
```

即内科 + 外科。第二阶段可视化页面中可以在设置里切换其他科室范围。

如果下载后的项目缺少 `Data_数据/`，请根据上文“数据来源”中的链接重新下载数据，并按上述目录结构放回项目根目录。

## 测试

项目包含单元测试：

```text
tests/
```

运行全部测试：

```powershell
python -m unittest discover -s tests
```

测试覆盖：

- 命令行检索流程。
- 本地问答运行时。
- 千问请求构造与超时重试。
- 混合模式兜底。
- Web 服务接口。
- UI 页面结构。
- 打包配置。

## 常见问题

### 1. 为什么启动后本地检索不能马上用？

本地数据较大，系统需要读取 CSV 并构建 TF-IDF 索引。第二阶段已经改为后台加载，页面顶部进度条显示加载状态。加载完成后，本地检索模式和混合模式即可使用。

### 2. 为什么在线问答超时？

可能原因包括：

- 网络连接不稳定。
- 千问 API Key 错误。
- 模型名配置错误。
- API 服务响应较慢。

系统已设置 60 秒超时和 2 次重试。在线失败时，可以切换到本地检索模式。

### 3. 为什么混合模式有时返回本地答案？

混合模式会先做本地检索，再调用千问。如果千问调用失败，系统会自动返回本地检索结果，保证系统仍然可用。

### 4. 可以只保留和运行 exe 吗？

运行打包后的完整medicalQA.exe程序，需要完整的输出文件夹：

```text
dist/MedicalQA/
```

不要只用 `MedicalQA.exe` 单文件，需要在完整文件内运行。

## 医疗安全声明

本系统用于自然语言处理课程展示，回答仅供健康咨询参考，不能替代医生诊断、处方、急救判断或线下治疗。

如出现胸痛、呼吸困难、意识障碍、大出血、高热不退、严重腹痛、严重脱水等急症表现，应立即线下就医或拨打急救电话。

## 项目总结

本项目完成了从检索式问答到可视化问答系统的完整实现：

- 第一阶段突出 NLP 检索核心流程。
- 第二阶段突出系统展示、交互体验和 exe 打包运行。
- 本地检索保证离线可用。
- 在线与混合模式提升回答表达能力。
- 测试和打包脚本保证项目可验证、可运行、可提交。
