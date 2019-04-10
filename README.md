# 基于attention的CNN文本分类

svm 基于tf-idf特征的支持向量机分类

att_cnn1 输入层attention

att_cnn2 卷积层attention

att_cnn3 双层attention

## 支持向量机

```python
python svm.py
```

## 训练

```python
python train.py --model base_cnn|att_cnn1|att_cnn2|att_cnn3 #default = att_cnn3
```

## 测试

```python
python test.py --model base_cnn|att_cnn1|att_cnn2|att_cnn3 #default = att_cnn3
```
