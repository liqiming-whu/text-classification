# 基于attention的CNN文本分类

att_cnn1 输入层attention
att_cnn2 卷积层attention
att_cnn3 双层attention

## 训练

```python
python train.py --model base_cnn|att_cnn1|att_cnn2|att_cnn3 #default = att_cnn3
```

## 测试

```python
python test.py --model base_cnn|att_cnn1|att_cnn2|att_cnn3 #default = att_cnn3
```
