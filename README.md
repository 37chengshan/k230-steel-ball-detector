# 电赛 · 庐山派 K230 钢珠识别开源模型

![宣传图](assets/k230-steel-ball-detector.svg)

面向 **标准庐山派 K230 + CanMV v1.6** 的钢珠检测训练与部署工程。模型会识别画面中的每颗钢珠；K230 端脚本绘制检测框，并经 UART2 输出钢珠中心坐标。

> 当前发布的是训练完成的 YOLO11n 权重和可复现脚本。KModel 的量化与 K230 实机长时间验收仍在进行，因此本仓库不会把未经实机验证的旧 KModel 冒充可用成品。

## 本次发布

| 文件 | 说明 |
| --- | --- |
| `models/yolo11n.pt` | Ultralytics YOLO11n 原始预训练权重 |
| `models/steel_ball_reference_yolo11n_1024_best.pt` | 1024×1024 钢珠检测训练最佳权重 |
| `canmv/steel_ball_yolo11_uart.py` | CanMV K230 运行、框选、平滑追踪与 UART 上报脚本 |
| `training/scripts/` | 数据清单构建、训练、ONNX 导出、训练看板脚本 |
| `training/data/k230_hard_examples/` | K230 实拍的孔板、暗场两个空标签负样本 |

本次高清训练在完成第 40 轮后手动早停：验证集最佳 **mAP50-95 = 0.96108**，mAP50 约 **0.995**。这是验证集成绩，不等同于 K230 实机效果；最终仍需针对实际赛场、光照和钢珠密度验收。

## 训练数据来源

本训练不上传原始大数据集。数据来源、数量与用途均在此明确列出：

| 来源 | 内容 | 用途 |
| --- | --- | --- |
| [flashy_dreams/steel_ball_](https://gitee.com/flashy_dreams/steel_ball_) | `real_field_sample`：237 张真实现场图及 YOLO 标签 | 训练、验证、真实现场测试 |
| 同一来源 | `synth_field`：3000 张 copy-paste 合成现场图及 YOLO 标签 | 训练、验证 |
| 本项目实拍 | K230 孔板、暗场共 2 张空标签负样本 | 抑制孔板/暗场误报 |

参考数据集和其图片、标签仍受原仓库的许可与使用规则约束；请先阅读其说明，再下载、使用或再发布原始数据。

## 训练

准备 Python、PyTorch 和 Ultralytics 后，在仓库根目录执行：

```powershell
git clone https://gitee.com/flashy_dreams/steel_ball_.git reference/maixcam2-steel-ball-v5
python training/scripts/build_reference_dataset.py
python training/scripts/train_reference.py
```

默认训练为 1024×1024、80 轮、提前停止耐心值 20。若只需快速比较，可使用：

```powershell
python training/scripts/train_reference.py --imgsz 640 --batch 16 --epochs 60 --name steel_ball_reference_yolo11n_640_fast
```

本地训练看板：

```powershell
python training/scripts/training_dashboard.py
```

浏览器打开 `http://127.0.0.1:8765`；左右两侧分别展示高清和快速训练，含曲线与实时终端输出。

## 导出 K230 模型

先导出 nncase 2.11 兼容的 ONNX：

```powershell
python training/scripts/export_k230_onnx.py `
  --weights models/steel_ball_reference_yolo11n_1024_best.pt `
  --output models/steel_ball_reference_yolo11n_1024_legacy.onnx `
  --imgsz 1024
```

再用与你的 K230 CanMV 固件兼容的 nncase 2.11 工具链完成 uint8 PTQ，得到：

```text
steel_ball_reference_yolo11n_1024_uint8.kmodel
```

量化样本应来自上面下载的 `reference/maixcam2-steel-ball-v5/data/synth_field/images`。量化成功后，把 KModel 放到：

```text
/sdcard/models/steel_ball_reference_yolo11n_1024_uint8.kmodel
```

## K230 部署与 UART

把 `canmv/steel_ball_yolo11_uart.py` 放到 TF 卡。脚本默认使用 CanMV IDE 虚拟显示，摄像头 AI 画面为 1024×1024。

| 信号 | 标准庐山派 K230 | 说明 |
| --- | --- | --- |
| UART2 TX | GPIO11 / UART2_TXD | 接收端 RX |
| UART2 RX | GPIO12 / UART2_RXD | 接收端 TX，可选 |
| GND | GND | 必须共地 |

UART 为 **115200、3.3 V TTL**，不要接 RS-232 电平。输出格式示例：

```text
BALL,N=3;120,88;251,104;382,301\r\n
```

程序通过两帧确认、短暂滑行保持和坐标平滑，减少检测框与串口坐标闪烁。

## 验收清单

- 孔板、暗场负样本：应输出 `balls=0`。
- 正样本：应打印 `stage=KPU_OUTPUT_READY`，并显示稳定检测框。
- 长时间运行：观察 FPS、内存与 UART 是否持续稳定。
- 本项目尚未完成上述 K230 实机验收；完成后会补充已验证的 `.kmodel` 与结果。

## 许可与致谢

本仓库新增代码采用 [MIT License](LICENSE)。`yolo11n.pt`、Ultralytics、参考数据集及其衍生内容分别遵循其原始许可；使用前请确认适用条款。

数据参考：[@flashy_dreams](https://gitee.com/flashy_dreams/steel_ball_) 的 MaixCAM2 钢珠检测开源分享。本项目仅复用其标注数据进行 K230 训练；其 MaixCAM2 模型不能直接用于 K230。
