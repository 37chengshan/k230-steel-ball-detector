# K230 钢珠模型 v2 验证包

这是一份为修复“无钢珠场景却出现上百个框”而生成的**独立测试包**。不要与上一版模型或脚本混用。

复制：

```text
steel_ball_reference_yolo11n_1024_raw_graph_f32.kmodel
  -> /sdcard/models/steel_ball_reference_yolo11n_1024_raw_graph_f32.kmodel

steel_ball_yolo11_uart_v2.py
  -> /sdcard/steel_ball_yolo11_uart_v2.py
```

然后在 CanMV IDE K230 中运行 `/sdcard/steel_ball_yolo11_uart_v2.py`。

预期：空白、货架、孔板等没有钢珠时，`balls=0` 或很少的低分候选；不应再次出现 100 多个跨满画面的框。

本版把 `/255` 归一化固定写进模型图中，并编译为独立的 float-input KModel，目的是绕开旧 uint8 KModel 的输入数值异常。它已完成主机端 ONNX 对照，仍需要本次 K230 实机确认。

请把启动后的完整日志、第一张无钢珠画面和第一张有钢珠画面发回；尤其关注是否打印到：

```text
stage=MODEL_READY
stage=KPU_OUTPUT_READY
```
