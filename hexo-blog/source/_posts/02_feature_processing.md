---
title: Qlib 特征处理机制
date: 2025-03-06 17:07:00
tags:
  - Qlib
  - 量化投资
  - 特征工程
categories:
  - 技术分享
---

# Qlib 特征处理机制

## 1. 简介

特征处理是量化投资中的关键环节，Qlib 提供了强大而灵活的特征处理机制，支持从原始金融数据生成、转换和处理各种特征。本文档将详细介绍 Qlib 的特征处理流程和相关组件。

## 2. 特征体系架构

Qlib 的特征处理体系采用分层设计，主要包括以下几个层次：

### 2.1 特征表达式层

- **表达式引擎**：解析和计算特征表达式
- **运算符（Operators）**：提供各种特征计算操作符
- **特征（Feature）**：表示基本特征和复合特征

### 2.2 特征处理层

- **处理器（Processors）**：对特征进行标准化、填充等处理
- **处理管道（Pipeline）**：组织多个处理器形成处理流程

### 2.3 特征组织层

- **数据处理器（DataHandler）**：组织特征和标签
- **Alpha158**：预定义的 158 个 Alpha 因子集合

## 3. 特征表达式系统

Qlib 实现了一套强大的特征表达式系统，允许用户通过类似 SQL 的语法定义复杂特征。

### 3.1 基本语法

特征表达式使用 `$` 符号引用原始特征，例如：

- `$close`：收盘价
- `$open`：开盘价
- `$high`：最高价
- `$low`：最低价
- `$volume`：成交量

### 3.2 运算符

Qlib 提供了丰富的运算符，用于构建复杂特征：

- **算术运算符**：`+`, `-`, `*`, `/`
- **比较运算符**：`>`, `<`, `==`, `!=`
- **逻辑运算符**：`&`, `|`, `~`
- **时序运算符**：`Ref`, `Mean`, `Std`, `Max`, `Min`
- **截面运算符**：`Rank`, `Quantile`

### 3.3 表达式示例

```python
# 计算收益率
"Ref($close, -1)/$close - 1"

# 计算 5 日均线
"Mean($close, 5)/$close"

# 计算波动率
"Std($close, 10)/$close"

# 计算动量因子
"Ref($close, -5)/$close - 1"

# 计算 MACD
"Mean($close, 12) - Mean($close, 26)"
```

## 4. 特征计算流程

### 4.1 表达式解析

特征表达式首先被解析为抽象语法树（AST）：

```python
# qlib/data/base.py
class ExpressionD(Cal):
    def __init__(self, expression: str, data_loader: CalendarProvider = None):
        self.expression = expression
        self.data_loader = data_loader if data_loader is not None else D
        
    def __call__(self, instruments=None, start_time=None, end_time=None, freq="day"):
        # 解析表达式
        expression = parse_field(self.expression)
        
        # 计算表达式
        # ...
```

### 4.2 特征计算

表达式解析后，通过递归计算得到特征值：

```python
# qlib/data/ops.py
class Feature(Expression):
    """特征基类"""
    
    def __call__(self, instrument=None, start_time=None, end_time=None, freq="day"):
        # 计算特征值
        # ...

class PFeature(Feature):
    """原始特征"""
    
    def __call__(self, instrument=None, start_time=None, end_time=None, freq="day"):
        # 从数据源获取原始特征
        # ...

class EFeature(Feature):
    """表达式特征"""
    
    def __call__(self, instrument=None, start_time=None, end_time=None, freq="day"):
        # 计算表达式特征
        # ...
```

### 4.3 特征缓存

为了提高性能，计算结果会被缓存：

```python
# qlib/data/cache.py
class ExpressionCache(BaseProviderCache):
    """表达式缓存"""
    
    def _expression(self, instrument, field, start_time, end_time, freq):
        # 检查缓存
        # 如果缓存存在，直接返回
        # 否则计算并缓存结果
        # ...
```

## 5. Alpha158 特征集

Qlib 预定义了 Alpha158 特征集，包含 158 个常用的 Alpha 因子。

### 5.1 特征分类

Alpha158 特征集包含以下几类特征：

- **K线特征**：基于开盘价、收盘价、最高价、最低价的特征
- **价格特征**：基于价格的特征
- **交易量特征**：基于交易量的特征
- **技术指标**：各种技术分析指标

### 5.2 特征定义

Alpha158 特征的定义在 `Alpha158DL` 类中：

```python
# qlib/contrib/data/loader.py
class Alpha158DL(QlibDataLoader):
    @staticmethod
    def get_feature_config(config=...):
        fields = []
        names = []
        
        # K线特征
        if "kbar" in config:
            fields += [
                "($close-$open)/$open",
                "($high-$low)/$open",
                # 更多K线特征...
            ]
            names += ["KMID", "KLEN", ...]
        
        # 价格特征
        if "price" in config:
            # ...
            
        # 交易量特征
        if "volume" in config:
            # ...
            
        # 滚动窗口特征
        if "rolling" in config:
            windows = config["rolling"].get("windows", [5, 10, 20, 30, 60])
            include = config["rolling"].get("include", None)
            exclude = config["rolling"].get("exclude", [])
            
            def use(x):
                return x not in exclude and (include is None or x in include)
            
            # 价格变化率
            if use("ROC"):
                fields += ["Ref($close, %d)/$close" % d for d in windows]
                names += ["ROC%d" % d for d in windows]
            
            # 简单移动平均
            if use("MA"):
                fields += ["Mean($close, %d)/$close" % d for d in windows]
                names += ["MA%d" % d for d in windows]
            
            # 标准差
            if use("STD"):
                fields += ["Std($close, %d)/$close" % d for d in windows]
                names += ["STD%d" % d for d in windows]
            
            # 更多技术指标...
```

### 5.3 特征使用

在 YAML 配置中使用 Alpha158 特征集：

```yaml
dataset:
    class: DatasetH
    module_path: qlib.data.dataset
    kwargs:
        handler:
            class: Alpha158
            module_path: qlib.contrib.data.handler
            kwargs:
                start_time: 2008-01-01
                end_time: 2020-08-01
                fit_start_time: 2008-01-01
                fit_end_time: 2014-12-31
                instruments: csi500
```

## 6. 特征处理器

Qlib 提供了多种特征处理器，用于对特征进行标准化、填充等处理。

### 6.1 处理器类型

- **RobustZScoreNorm**：稳健的 Z 分数标准化
- **Fillna**：填充缺失值
- **DropnaLabel**：删除含有缺失标签的样本
- **CSRankNorm**：截面排序标准化

### 6.2 处理器配置

在 YAML 配置中配置处理器：

```yaml
data_handler_config:
    infer_processors:
        - class: RobustZScoreNorm
          kwargs:
              fields_group: feature
              clip_outlier: true
        - class: Fillna
          kwargs:
              fields_group: feature
    learn_processors:
        - class: DropnaLabel
        - class: CSRankNorm
          kwargs:
              fields_group: label
```

### 6.3 处理器实现

处理器的实现基于 `Processor` 基类：

```python
# qlib/data/dataset/processor.py
class Processor:
    def __init__(self):
        self.fitted = False
    
    def fit(self, data):
        """拟合处理器参数"""
        self.fitted = True
        return self
    
    def __call__(self, data):
        """处理数据"""
        return self.transform(data)
    
    def transform(self, data):
        """转换数据"""
        raise NotImplementedError("transform not implemented!")
    
    def get_fitted(self):
        """返回拟合状态"""
        return self.fitted
```

具体处理器的实现示例：

```python
# qlib/data/dataset/processor.py
class RobustZScoreNorm(Processor):
    """稳健的 Z 分数标准化"""
    
    def __init__(self, fields_group="feature", clip_outlier=True):
        super().__init__()
        self.fields_group = fields_group
        self.clip_outlier = clip_outlier
        
    def fit(self, data):
        # 计算中位数和绝对偏差中位数
        self.median = data.median()
        self.mad = (data - self.median).abs().median()
        return super().fit(data)
    
    def transform(self, data):
        # 应用稳健的 Z 分数标准化
        if not self.fitted:
            return data
            
        data = data - self.median
        data = data / (self.mad * 1.4826)
        
        if self.clip_outlier:
            data = data.clip(-3, 3)
            
        return data
```

## 7. 特征处理流程

### 7.1 处理流程概述

Qlib 的特征处理流程主要包括以下步骤：

1. **特征加载**：从数据源加载原始特征
2. **特征计算**：计算复合特征
3. **特征处理**：应用处理器进行标准化、填充等处理
4. **特征组织**：将处理后的特征组织为模型所需的格式

### 7.2 处理流程实现

特征处理流程在 `DataHandlerLP` 类中实现：

```python
# qlib/data/dataset/handler.py
class DataHandlerLP(DataHandler):
    def __init__(
        self,
        instruments=None,
        start_time=None,
        end_time=None,
        data_loader=None,
        infer_processors=[],
        learn_processors=[],
        process_type=PTYPE_A,
        **kwargs,
    ):
        # 初始化
        # ...
        
    def fetch(self, selector: Union[pd.Timestamp, slice, str] = slice(None, None), level: Union[str, int] = "both"):
        # 获取原始数据
        # ...
        
        # 应用 infer_processors
        if self.infer_processors:
            if self.process_type == DataHandlerLP.PTYPE_A:
                # 一次性处理所有数据
                data = pd.concat([data], keys=["all"], names=[self.instruments_list])
                for proc in self.infer_processors:
                    if proc.get_fitted():
                        data = proc.transform(data)
                    else:
                        data = proc.fit(data).transform(data)
                data = data.loc["all"]
            elif self.process_type == DataHandlerLP.PTYPE_I:
                # 单独处理每个股票
                for inst in instruments:
                    if inst in data:
                        for proc in self.infer_processors:
                            if proc.get_fitted():
                                data[inst] = proc.transform(data[inst])
                            else:
                                data[inst] = proc.fit(data[inst]).transform(data[inst])
        
        return data

    def fetch_learn(self, data=None):
        """应用 learn_processors"""
        if data is None:
            data = self.fetch()
        
        # 应用 learn_processors
        if self.learn_processors:
            for proc in self.learn_processors:
                if proc.get_fitted():
                    data = proc.transform(data)
                else:
                    data = proc.fit(data).transform(data)
        
        return data
```

## 8. 自定义特征

Qlib 支持用户自定义特征和处理器，以满足特定需求。

### 8.1 自定义特征表达式

用户可以定义自己的特征表达式：

```python
# 自定义动量因子
momentum_10d = "Ref($close, -10)/$close - 1"

# 自定义波动率因子
volatility_20d = "Std($close/Ref($close, -1) - 1, 20)"

# 自定义交易量变化率
volume_change = "Log($volume/Ref($volume, -1))"
```

### 8.2 自定义处理器

用户可以实现自己的处理器：

```python
from qlib.data.dataset.processor import Processor

class MyNormalizer(Processor):
    def __init__(self, fields_group="feature"):
        super().__init__()
        self.fields_group = fields_group
        
    def fit(self, data):
        # 计算自定义参数
        self.min = data.min()
        self.max = data.max()
        return super().fit(data)
    
    def transform(self, data):
        # 应用自定义标准化
        if not self.fitted:
            return data
            
        return (data - self.min) / (self.max - self.min)
```

### 8.3 自定义数据处理器

用户可以实现自己的数据处理器：

```python
from qlib.contrib.data.handler import Alpha158

class MyAlpha(Alpha158):
    def get_feature_config(self):
        # 自定义特征配置
        conf = {
            "kbar": {},
            "price": {
                "windows": [0, 1, 2, 3, 4],
                "feature": ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"],
            },
            "rolling": {
                "windows": [5, 10, 20, 30, 60],
                "include": ["ROC", "MA", "STD", "BETA", "RSQR", "RESI", "MAX", "MIN"],
            },
        }
        return Alpha158DL.get_feature_config(conf)
    
    def get_label_config(self):
        # 自定义标签配置
        return ["Ref($close, -5)/Ref($close, -1) - 1"], ["LABEL0"]
```

## 9. 总结

Qlib 的特征处理机制提供了强大而灵活的工具，用于从原始金融数据生成、转换和处理各种特征。其核心优势包括：

1. **表达式引擎**：支持通过简洁的表达式定义复杂特征
2. **丰富的运算符**：提供各种时序和截面运算符
3. **预定义特征集**：提供 Alpha158 等预定义特征集
4. **灵活的处理器**：支持各种特征处理操作
5. **可扩展性**：支持用户自定义特征和处理器

通过这些工具，用户可以轻松实现复杂的特征工程，为量化投资模型提供高质量的输入数据。 