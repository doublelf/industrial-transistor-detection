# 环境记录

> 生成时间：2026-11-25
> 完整依赖快照：`logs/pip_freeze.txt`（82 个包）

## 1. 硬件与系统

| 项 | 值 |
|---|---|
| 操作系统 | Windows 11 (10.0.26200) |
| CPU | 16 逻辑核（`os.cpu_count()=16`，torch 使用 12 线程） |
| GPU | **无 NVIDIA GPU**。仅有 `Intel(R) Iris(R) Xe Graphics` 核显 |
| CUDA | `torch.cuda.is_available() = False` |
| **推理设备** | **CPU**（`device=cpu`） |

> 说明：`torch` 安装的是 `2.15.0+cpu` 纯 CPU wheel（PyPI 默认），因此不存在 CUDA 路径。
> 本项目所有耗时数据均为 **CPU 推理**结果，与 GPU 环境不可直接比较。

## 2. 运行环境

| 项 | 值 |
|---|---|
| 隔离方式 | Python `venv`，路径 `.venv\` |
| Python | 3.12.14 |
| 解释器绝对路径 | `D:\Desktop\chat\opencode\project\industrial-transistor-detection\.venv\Scripts\python.exe` |
| 环境来源 | 由本机 DSH 自带 Python 3.12.14 创建（系统 `python` 命令为 Microsoft Store 占位符，不可用） |

## 3. 关键版本

| 包 | 版本 | 说明 |
|---|---|---|
| `anomalib` | **2.7.1** | 官方 PyPI 发行版，Apache-2.0 |
| `open-clip-torch` | **2.24.0** | 满足 anomalib 的 `<2.26.1` 约束 |
| `torch` | 2.15.0+cpu | |
| `torchvision` | 0.29.2+cpu | |
| `lightning` | 2.6.6 | |
| `torchmetrics` | 1.9.0 | 用于 AUROC |
| `timm` | 1.0.31 | open_clip 依赖 |
| `numpy` | 2.5.5 | |

## 4. 安装命令（已执行）

```powershell
# 1) 建独立环境
<DSH_PY>\python.exe -m venv .venv

# 2) 安装 anomalib 及 clip 扩展
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install "anomalib[clip]"
```

- 安装日志：`logs/pip_install_anomalib.log`
- **注意**：`pip install anomalib[clip]` 只安装 Python 包（代码依赖），**不会**下载任何模型权重或数据集。

## 5. 官方源码

| 项 | 值 |
|---|---|
| 仓库 | https://github.com/open-edge-platform/anomalib （Apache-2.0） |
| 获取方式 | `git clone` 卡住后改用 GitHub codeload 官方压缩包（`main` 分支 tarball，22.6 MB） |
| 本地路径 | `third_party\anomalib\anomalib-main` |
| 源码版本 | `__version__ = "2.7.2.dev0"` |
| 用途 | 只读，用于核对实现与提示词；实际运行使用 PyPI 安装的 2.7.1 |

> 说明：安装版 2.7.1 与源码 main（2.7.2.dev0）的 WinCLIP 关键常量已比对一致：
> `BACKBONE="ViT-B-16-plus-240"`、`PRETRAINED="laion400m_e31"`、`TEMPERATURE=0.07`、`scales=(2,3)`、输入 240×240。

## 6. 权重与数据

| 项 | 值 |
|---|---|
| 权重路径 | `weights\vit_b_16_plus_240-laion400m_e31-8fb26589.pt` |
| 权重大小 | 833,625,910 字节 |
| 权重 SHA256 | `F777E0E06CCBB33ADF0C11B7B2B732FF2063A6BB98D7AC64FA80568674A54556` |
| 骨干结构 | `ViT-B-16-plus-240`（视觉 width 896 / grid 15×15） |
| 种子归属 | ⚠️ 附带 README 标注为 `laion400m_e32`，非 anomalib 默认的 `e31`；详见 `results/files_check.md` 第 2 节 |
| 数据集 | `datasets\mvtec_transistor\transistor\` |
| 数据集许可 | CC BY-NC-SA 4.0，**禁止商用**，仅用于研发验证 |

## 7. 设备确认结论

- ✅ 已确认设备为 **CPU**，无 GPU。
- 影响：单张 240×240 推理需经 1 次整图 + 2×2 与 3×3 两档滑窗（共 225+100 次窗口编码），CPU 下预计**单张数秒**级别；Web 演示节拍默认 2 秒/件，若单件推理超过节拍会在界面如实显示耗时，不做虚假预测。
