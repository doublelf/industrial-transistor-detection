# 文件校验报告（无下载，仅本地校验）

> 执行时间：2026-11-25
> 校验方式：纯本地读写 + 哈希计算 + 官方代码路径加载，**未发生任何网络下载**

---

## 0. 结论速览

| 项目 | 结果 |
|---|---|
| 骨干权重文件存在 | ✅ 833,625,910 字节（≈795 MB） |
| 骨干结构匹配 `ViT-B-16-plus-240` | ✅ 完美匹配（0 缺失 / 0 多余 / 0 形状不符） |
| 权重能否加载 | ✅ 通过 `open_clip.create_model_and_transforms` 实际加载成功 |
| 权重种子（e31 还是 e32） | ⚠️ **不是官方 e31，见第 2 节，需你知悉** |
| transistor test 图片 | ✅ 100 张（good 60 + 缺陷 40），符合官方划分 |
| ground_truth 掩码 | ✅ 40 张（额外获得，本项目图级指标不需要，但已留存） |

---

## 1. 文件清点

### 1.1 权重

| 项 | 值 |
|---|---|
| 你提供的原始文件 | `Temporary\open_clip_pytorch_model.bin` |
| 已放置到 | `weights\vit_b_16_plus_240-laion400m_e31-8fb26589.pt` |
| 大小 | 833,625,910 字节 |
| SHA256（实测） | `F777E0E06CCBB33ADF0C11B7B2B732FF2063A6BB98D7AC64FA80568674A54556` |
| 结构 | `OrderedDict`，302 个张量键 |
| 视觉编码器 | 12 层 Transformer，width=896，patch 16×16 → grid 15×15（即 240×240 输入） |
| 视觉位置编码 | `(226, 896)` = 1(cls) + 225 patch |
| 文本编码器 | 12 层，`text_projection (640, 640)`，位置编码 `(77, 640)` |

### 1.2 数据集

解压到的位置：`datasets\mvtec_transistor\transistor\`

```
transistor/
├── test/
│   ├── good/          60 张
│   ├── bent_lead/     10 张
│   ├── cut_lead/      10 张
│   ├── damaged_case/  10 张
│   └── misplaced/     10 张      —— test 合计 100 张
├── ground_truth/      bent_lead / cut_lead / damaged_case / misplaced 各 10 张掩码，共 40 张
├── readme.txt         → MVTec Software GmbH, CC BY-NC-SA 4.0
└── license.txt        → 完整 CC BY-NC-SA 4.0 文本
```

- 类别：`transistor`（单类，符合范围约束）
- 缺陷子类目：`bent_lead`、`cut_lead`、`damaged_case`、`misplaced` —— 与官方 MVTec AD 一致
- 许可：**CC BY-NC-SA 4.0，禁止商用**（已在 README 注明）

---

## 2. ⚠️ 权重种子归属：与官方 anomalib 默认值不同

这是本次校验最重要的发现，单独说明。

### 2.1 事实

1. 你附带的 `Temporary\README.txt` 第 4 行明确写着：
   > `weights_part_00 到 weights_part_09：模型权重 OpenCLIP ViT-B-16-plus-240（laion400m_e32）的分卷`
2. 而 anomalib 代码请求的是 **`laion400m_e31`**（`winclip/torch_model.py:55`）。
3. 该文件的 SHA256 = `F777E0E0...`，而 OpenCLIP 对官方权重内建的前缀校验是：
   - `e31` → `8fb26589`
   - `e32` → `699c4b84`
   **两者都对不上 `F777E0E0`**。

### 2.2 如何判断它到底是 e31 还是 e32

仅从哈希无法判定。可以确定的是：

- ✅ **骨干架构 100% 是 `ViT-B-16-plus-240`**：该文件对 `ViT-B-16-plus-240` 的 `state_dict` 是 **0 缺失、0 多余、0 形状不符**的完美匹配；而对普通 `ViT-B-16` 有 **301 处形状不符**（如 `visual.pos_emb (226,896)` vs `(197,768)`），可完全排除。
- ⚠️ 它**不是** OpenCLIP release 直链下的那个原始 `.pt` 文件（哈希前缀不匹配 e31/e32 任一官方值），说明它可能是重打包/转换过的副本（例如从 HuggingFace 镜像或 `safetensors` 导回 `.bin`、或经过再序列化），**内容等价与否无法仅靠哈希确认**。

### 2.3 对你的影响（如实说明）

- **技术上可跑通**：它属于同一个骨干家族，加载无误，模型能正常出分数和热力图。本项目后续所有数字都基于**这一个权重**，内部自洽。
- **但与“官方 e31 基线”不可直接对照**：按 README 所述它是 **e32**。WinCLIP 论文与 anomalib 默认结果基于 **e31**。所以：
  - 本项目与论文数字（MVTec AD 零样本图级 AUROC ≈ 91.8）的对比，会因骨干种子不同而存在**额外偏差**，我在 `run_report.md` 里会明确标注这一条，不会归因为算法本身。
  - `results/environment.md` 与所有结果 JSON 都会记录实际权重文件的路径与 SHA256 前缀 `F777E0E0`，便于追溯。

### 2.4 我的处理

**不改动你给的文件、不再要求你下载第二个权重**（那会违反“不要传输/不要下载”的分工），按现有文件跑通全流程，并在报告中把“实测数字”与“论文数字”严格分开、注明骨干种子差异。

> 若你后续希望严格对齐官方 e31 基线，只需把 `vit_b_16_plus_240-laion400m_e31-8fb26589.pt`（SHA256 前缀 `8fb26589`）放到 `weights\`，把配置里 `backbone_weights` 指过去重跑即可，代码无需改动。

---

## 3. 加载验证（真实运行输出）

调用与 anomalib 完全相同的官方代码路径 `open_clip.create_model_and_transforms(..., pretrained=<本地路径>)`：

```
load OK
grid_size   = (15, 15)
visual width= 896
image feat shape: (1, 640)
text  feat shape: (1, 640)
logit_scale: 100.00000762939453
```

- 图像特征 `(1, 640)`、文本特征 `(1, 640)` 维度一致 → WinCLIP 的图文相似度计算可以正常进行。
- `grid_size=(15,15)` → 2×2、3×3 两档滑窗掩码可正常生成（`make_masks` 要求 grid ≥ kernel，15 ≥ 3 满足）。
- `create_model_and_transforms` 全程未联网：因为传入的是已存在的本地路径，`get_pretrained_cfg` 查不到该 tag，直接走 `os.path.exists` 分支（`open_clip/factory.py:285`）。

---

## 4. 零下载证明

| 检查项 | 结果 |
|---|---|
| `%USERPROFILE%\.cache\clip` | 不存在（OpenCLIP 自动下载的落盘目录，未创建） |
| `%USERPROFILE%\.cache\huggingface` 中新增权重 | 无（本次只读取本地文件） |
| 本会话执行的网络动作 | `pip install anomalib[clip]`、GitHub codeload 取源码压缩包 |
| 权重/数据集下载命令 | **0 次** |

---

## 5. 下一步

1. 记录环境与版本 → `results/environment.md`
2. 写 `predict.py`（单图推理：分数 / OK-NG / 热力图 / JSON）
3. 在 `test/good` 上定阈值
4. 批量跑 100 张 → `results/batch_results.csv` + 图级 AUROC
5. 抽查 6 张、写 `run_report.md`
6. Web 产线演示界面
7. 一致性抽查 → `results/ui_demo/check.md`
