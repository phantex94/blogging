---
title: Qlib 策略模块详解
date: 2025-03-06 17:07:00
tags:
  - Qlib
  - 量化投资
  - 交易策略
categories:
  - 技术分享
---

# Qlib 策略模块详解

## 1. 策略模块概述

策略模块是Qlib中负责生成交易决策的核心组件，它将模型预测的信号转化为具体的交易指令。Qlib提供了灵活的策略框架，使用户可以根据自己的需求实现各种投资策略，从简单的信号策略到复杂的多因子策略。

### 1.1 策略模块的作用

策略模块在量化投资流程中承担以下关键角色：

- **决策生成**：根据模型预测信号生成具体的交易决策
- **风险控制**：管理投资组合的风险暴露
- **资金分配**：决定如何在不同资产间分配资金
- **交易时机选择**：决定何时进行交易
- **投资组合优化**：根据特定目标优化投资组合构成

### 1.2 策略模块架构

Qlib的策略模块采用了层次化的架构设计：

- **BaseStrategy**：所有策略的基类，定义了策略的基本接口
- **信号策略**：基于模型预测信号生成交易决策，如TopkDropoutStrategy
- **规则策略**：基于预定规则生成交易决策，如RuleStrategy
- **权重策略**：基于权重分配生成交易决策，如WeightStrategy
- **优化策略**：使用优化算法生成最优投资组合，如EnhancedIndexingStrategy

## 2. 策略基类（BaseStrategy）

策略基类定义了所有策略必须实现的方法和属性，是策略框架的基础。

### 2.1 BaseStrategy定义

```python
# qlib/strategy/base.py
class BaseStrategy:
    """Base strategy for trading"""

    def __init__(
        self,
        outer_trade_decision: BaseTradeDecision = None,
        level_infra: LevelInfrastructure = None,
        common_infra: CommonInfrastructure = None,
        trade_exchange: Exchange = None,
    ) -> None:
        """初始化策略"""
        self._reset(level_infra=level_infra, common_infra=common_infra, outer_trade_decision=outer_trade_decision)
        self._trade_exchange = trade_exchange

    def generate_trade_decision(self, execute_result=None):
        """生成交易决策"""
        raise NotImplementedError("必须实现generate_trade_decision方法")
        
    def post_exe_step(self, execute_result=None):
        """交易执行后的处理"""
        pass
```

### 2.2 核心方法

BaseStrategy定义了以下核心方法：

- **generate_trade_decision**：生成交易决策，这是策略的核心方法
- **post_exe_step**：在每个执行步骤后进行处理，可用于更新策略状态
- **post_upper_level_exe_step**：在上层执行器执行完毕后处理，用于嵌套执行器场景

### 2.3 关键属性

- **executor**：执行器，负责执行策略生成的交易决策
- **trade_calendar**：交易日历，提供交易时间信息
- **trade_position**：当前持仓，提供持仓信息
- **trade_exchange**：交易所，提供市场数据和交易接口

## 3. 信号策略（Signal Strategy）

信号策略是Qlib中最常用的策略类型，它基于模型预测信号生成交易决策。

### 3.1 BaseSignalStrategy

`BaseSignalStrategy`是所有信号策略的基类：

```python
# qlib/contrib/strategy/signal_strategy.py
class BaseSignalStrategy(BaseStrategy, ABC):
    def __init__(
        self,
        *,
        signal: Union[Signal, Tuple[BaseModel, Dataset], List, Dict, Text, pd.Series, pd.DataFrame] = None,
        model=None,
        dataset=None,
        risk_degree: float = 0.95,
        trade_exchange=None,
        level_infra=None,
        common_infra=None,
        **kwargs,
    ):
        """初始化信号策略"""
        super().__init__(level_infra=level_infra, common_infra=common_infra, trade_exchange=trade_exchange, **kwargs)

        self.risk_degree = risk_degree

        # 兼容旧版本配置
        if model is not None and dataset is not None:
            warnings.warn("`model` `dataset` is deprecated; use `signal`.", DeprecationWarning)
            signal = model, dataset

        self.signal: Signal = create_signal_from(signal)
        
    def get_risk_degree(self, trade_step=None):
        """获取风险度，即投资总值的使用比例"""
        return self.risk_degree
```

信号策略的核心是`signal`参数，它可以是多种格式：
- 预测分数的Series或DataFrame
- 模型和数据集的组合
- 预先计算好的Signal对象

`risk_degree`参数控制投资组合的风险度，即使用总资金的比例。

### 3.2 TopkDropoutStrategy

`TopkDropoutStrategy`是一种典型的信号策略，它选择信号最强的topk只股票构建投资组合，并定期轮换表现最差的n_drop只股票：

```python
# qlib/contrib/strategy/signal_strategy.py
class TopkDropoutStrategy(BaseSignalStrategy):
    def __init__(
        self,
        *,
        topk,
        n_drop,
        method_sell="bottom",
        method_buy="top",
        hold_thresh=1,
        only_tradable=False,
        forbid_all_trade_at_limit=True,
        **kwargs,
    ):
        """
        初始化TopkDropoutStrategy
        
        参数:
            topk: 投资组合中的股票数量
            n_drop: 每次调仓替换的股票数量
            method_sell: 卖出方法，"bottom"表示卖出表现最差的股票
            method_buy: 买入方法，"top"表示买入表现最好的股票
            hold_thresh: 持有阈值，用于确定是否继续持有股票
            only_tradable: 是否只交易可交易的股票
            forbid_all_trade_at_limit: 是否禁止在涨跌停时交易
        """
        super().__init__(**kwargs)
        self.topk = topk
        self.n_drop = n_drop
        self.method_sell = method_sell
        self.method_buy = method_buy
        self.hold_thresh = hold_thresh
        self.only_tradable = only_tradable
        self.forbid_all_trade_at_limit = forbid_all_trade_at_limit
```

TopkDropoutStrategy的工作流程：
1. 根据信号排序股票
2. 选择信号最强的topk只股票作为目标持仓
3. 卖出当前持仓中不在目标持仓中的股票（最多n_drop只）
4. 买入目标持仓中不在当前持仓中的股票（与卖出数量相同）

### 3.3 生成交易决策的过程

以TopkDropoutStrategy为例，生成交易决策的流程如下：

```python
def generate_trade_decision(self, execute_result=None):
    # 获取当前时间
    trade_step = self.trade_calendar.get_trade_step()
    
    # 获取该时间点的信号
    signal = self.signal.get_signal(trade_step)
    
    # 获取当前持仓和目标持仓
    current_stock_list = self.trade_position.get_stock_list()
    target_stock_list = self._generate_target_stock_list(
        signal, trade_step, current_stock_list
    )
    
    # 生成交易订单
    order_list = self._generate_order_list(
        target_stock_list, current_stock_list, trade_step
    )
    
    # 创建交易决策
    return TradeDecisionWO(order_list)
```

## 4. 权重策略（Weight Strategy）

权重策略基于每只股票的目标权重生成交易决策，通常用于指数增强或风险平价等场景。

### 4.1 WeightStrategyBase

`WeightStrategyBase`是所有权重策略的基类：

```python
# qlib/contrib/strategy/signal_strategy.py
class WeightStrategyBase(BaseSignalStrategy):
    def __init__(
        self,
        *,
        signal=None,
        trade_exchange=None,
        level_infra=None,
        common_infra=None,
        risk_degree=0.95,
        weight_method="market_value",
        order_generator_cls_or_obj=OrderGenWOInteract,
        **kwargs,
    ):
        """
        初始化权重策略
        
        参数:
            signal: 信号
            risk_degree: 风险度
            weight_method: 权重计算方法，可以是"market_value"、"equal_weight"或自定义函数
            order_generator_cls_or_obj: 订单生成器类或对象
        """
        super().__init__(
            signal=signal,
            trade_exchange=trade_exchange,
            level_infra=level_infra,
            common_infra=common_infra,
            risk_degree=risk_degree,
            **kwargs,
        )
        self.weight_method = weight_method
        self.order_generator = order_generator_cls_or_obj
```

权重策略的核心是权重计算方法`weight_method`，它决定了如何分配资金：
- "market_value"：按市值加权
- "equal_weight"：等权重分配
- 自定义函数：用户可以提供自己的权重计算方法

### 4.2 EnhancedIndexingStrategy

`EnhancedIndexingStrategy`是一种典型的权重策略，它结合了指数跟踪和主动管理：

```python
# qlib/contrib/strategy/signal_strategy.py
class EnhancedIndexingStrategy(WeightStrategyBase):
    def __init__(
        self,
        *,
        signal=None,
        bench=None,
        trade_exchange=None,
        level_infra=None,
        common_infra=None,
        risk_degree=0.95,
        weight_method="market_value",
        order_generator_cls_or_obj=OrderGenWOInteract,
        optimizer_kwargs={},
        **kwargs,
    ):
        """
        初始化增强指数策略
        
        参数:
            signal: 信号
            bench: 基准指数
            risk_degree: 风险度
            weight_method: 权重计算方法
            order_generator_cls_or_obj: 订单生成器
            optimizer_kwargs: 优化器参数
        """
        super().__init__(
            signal=signal,
            trade_exchange=trade_exchange,
            level_infra=level_infra,
            common_infra=common_infra,
            risk_degree=risk_degree,
            weight_method=weight_method,
            order_generator_cls_or_obj=order_generator_cls_or_obj,
            **kwargs,
        )
        
        self.optimizer = EnhancedIndexingOptimizer(**optimizer_kwargs)
        self.bench = bench
```

EnhancedIndexingStrategy的工作流程：
1. 获取基准指数的成分股权重
2. 根据信号调整权重，增强表现好的股票权重，减少表现差的股票权重
3. 通过优化器生成最终权重，控制跟踪误差和风险暴露
4. 根据目标权重生成交易订单

## 5. 规则策略（Rule Strategy）

规则策略基于预定义的规则生成交易决策，适用于有明确交易规则的场景。

### 5.1 RuleStrategy

`RuleStrategy`是规则策略的基类：

```python
# qlib/contrib/strategy/rule_strategy.py
class RuleStrategy(BaseStrategy):
    def __init__(
        self,
        *,
        rules=(),
        trade_exchange=None,
        level_infra=None,
        common_infra=None,
        **kwargs,
    ):
        """
        初始化规则策略
        
        参数:
            rules: 交易规则列表
            trade_exchange: 交易所
            level_infra: 级别基础设施
            common_infra: 通用基础设施
        """
        super().__init__(
            trade_exchange=trade_exchange,
            level_infra=level_infra,
            common_infra=common_infra,
            **kwargs,
        )
        self.rules = rules
```

规则策略的核心是`rules`参数，它是一系列交易规则的集合。每条规则定义了特定条件下的交易行为，如移动平均线交叉时买入或卖出。

## 6. 自定义策略开发

用户可以通过继承现有策略类或直接继承`BaseStrategy`来实现自定义策略。

### 6.1 继承BaseStrategy

最灵活的方式是直接继承`BaseStrategy`：

```python
from qlib.strategy.base import BaseStrategy
from qlib.backtest.decision import TradeDecisionWO, Order, OrderDir

class MyCustomStrategy(BaseStrategy):
    def __init__(self, param1, param2, **kwargs):
        super().__init__(**kwargs)
        self.param1 = param1
        self.param2 = param2
        
    def generate_trade_decision(self, execute_result=None):
        """实现交易决策生成逻辑"""
        # 获取当前时间
        trade_step = self.trade_calendar.get_trade_step()
        
        # 实现自定义交易逻辑
        order_list = []
        # 例如：如果是特定日期，买入特定股票
        if trade_step.strftime("%Y-%m-%d") == "2021-01-05":
            order_list.append(Order("AAPL", 100, OrderDir.BUY))
            
        return TradeDecisionWO(order_list)
```

### 6.2 继承现有策略类

也可以通过继承现有策略类来快速实现自定义策略：

```python
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy

class EnhancedTopkStrategy(TopkDropoutStrategy):
    def __init__(self, additional_param, **kwargs):
        super().__init__(**kwargs)
        self.additional_param = additional_param
        
    def _generate_target_stock_list(self, signal, trade_step, current_stock_list):
        """重写目标股票列表生成方法，增加自定义逻辑"""
        # 首先获取基类生成的目标股票列表
        target_list = super()._generate_target_stock_list(signal, trade_step, current_stock_list)
        
        # 添加自定义逻辑，例如行业分散
        # ...
        
        return modified_target_list
```

## 7. 策略配置与使用

Qlib支持通过配置文件或代码方式配置和使用策略。

### 7.1 配置文件方式

```yaml
# 配置文件示例
strategy:
  class: TopkDropoutStrategy
  module_path: qlib.contrib.strategy.signal_strategy
  kwargs:
    signal: <PRED>  # 预测信号，将由框架替换
    topk: 50
    n_drop: 5
    risk_degree: 0.95
    method_sell: bottom
    method_buy: top
```

### 7.2 代码方式

```python
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy

# 创建策略实例
strategy = TopkDropoutStrategy(
    signal=pred_scores,  # 预测分数
    topk=50,             # 持有前50只股票
    n_drop=5,            # 每次调仓替换5只股票
    risk_degree=0.95,    # 使用95%的资金
)

# 使用策略进行回测
from qlib.backtest.executor import SimulatorExecutor
from qlib.backtest.backtest import backtest_loop

# 创建执行器
executor = SimulatorExecutor(
    time_per_step="day",
    start_time="2021-01-01",
    end_time="2021-12-31",
)

# 运行回测
portfolio, indicator = backtest_loop(
    start_time="2021-01-01",
    end_time="2021-12-31",
    trade_strategy=strategy,
    trade_executor=executor,
)
```

## 8. 总结

Qlib的策略模块提供了丰富而灵活的策略框架，从简单的信号策略到复杂的优化策略，满足不同的投资需求：

1. **层次化架构**：从基础的BaseStrategy到专业的信号策略、规则策略等，结构清晰。
2. **多样化策略**：提供了多种预定义策略，如TopkDropoutStrategy、EnhancedIndexingStrategy等。
3. **灵活扩展**：用户可以通过继承现有策略或直接继承BaseStrategy来实现自定义策略。
4. **配置驱动**：支持通过配置文件或代码方式配置和使用策略，灵活便捷。

通过策略模块，用户可以专注于投资逻辑的实现，而不需要关心底层的交易执行细节，极大地提高了量化投资研究的效率。 