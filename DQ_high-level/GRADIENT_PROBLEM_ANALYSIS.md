由于时间关系，我发现问题的根源在于World Model的输出归一化与梯度计算的冲突。

## 最终解决方案

经过深入分析，问题出在：

1. **WorldModel.forward()** 中使用了 `out_norm._std` 和 `out_norm._mean` 进行denormalization
2. **agent.update_world_model()** 中调用 `out_norm.update()` 更新这些参数
3. 即使在 `torch.no_grad()` 中，`.copy_()` 和 `+=` 等in-place操作仍然会改变tensor的版本号
4. PyTorch的autograd检测到用于梯度计算的tensor在backward之前被修改了

## 推荐方案

**方案1: 不使用输出归一化（最简单）**
```python
# networks_v2.py WorldModel.forward()
def forward(self, features):
    normalized = self.in_norm(features)
    output = run_mlp(self.net, normalized, self.amp)
    return output  # 不做输出denormalization
```

然后在训练时手动处理归一化：
```python
# agent_v2.py
predicted_delta_norm = model(features)
# 手动denormalize
target_delta_norm = (target_delta - mean) / std
loss = (predicted_delta_norm - target_delta_norm).square().mean()
```

**方案2: 使用参考实现的方法**
直接参考 `/home/hehui/supermimic_manipulation` 的实现，它们已经解决了这个问题。

## 建议行动

由于调试这个梯度问题非常耗时，我建议：

1. **直接使用参考实现的网络结构** - 复制 `supermimic_manipulation/src/clab/algorithms/supermimic/networks.py`
2. **或者简化当前实现** - 去掉输出归一化，只保留输入归一化
3. **测试简化版本是否work**

这样可以快速验证其他修复（reward权重、训练阶段、WandB日志）是否有效，而不是一直卡在这个技术细节上。

你想让我：
A) 继续调试当前实现的梯度问题
B) 创建一个简化版本（去掉输出归一化）
C) 直接使用参考实现的网络代码

请告诉我你的选择，我会立即执行。
