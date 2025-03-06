---
title: Qlib 回测引擎解析
date: 2025-03-06 17:07:00
tags:
  - Qlib
  - 量化投资
  - 回测引擎
categories:
  - 技术分享
---
# Qlib 回测引擎原理

## 1. 回测引擎概述

Qlib的回测引擎是一个完整的系统，负责将模型预测转化为交易决策、执行交易、计算收益并评估策略表现。它通过模拟真实市场环境，提供策略回测和评估的全流程支持。本文将详细介绍回测引擎的工作原理，特别是模型预测如何转换为持仓，以及持仓如何转化为成交回报。

### 1.1 回测引擎的组成

Qlib的回测引擎主要由以下组件组成：

- **策略层（Strategy）**：将模型预测转换为交易决策
- **执行层（Executor）**：执行交易决策，模拟交易过程
- **交易所（Exchange）**：提供市场数据和执行订单
- **账户层（Account）**：记录持仓和现金状态
- **评估层（Evaluation）**：计算各种性能指标

### 1.2 回测引擎工作流程

回测引擎的基本工作流程如下：

1. **模型预测**：模型生成预测信号，如股票的预期收益率
2. **策略决策**：策略根据预测信号生成交易决策
3. **订单执行**：执行器接收交易决策并执行订单
4. **持仓更新**：账户根据执行结果更新持仓
5. **收益计算**：计算策略的收益和风险指标
6. **结果评估**：评估策略的整体表现

## 2. 模型预测到持仓的转换

### 2.1 预测信号的生成

回测引擎的起点是模型预测生成的信号，这些信号通常表示为每只股票的预期收益率或评分。信号可以通过多种方式获取：

```python
# 方式1：直接使用模型预测
signal = model.predict(dataset)

# 方式2：加载预先计算好的预测结果
signal = pd.read_csv("prediction.csv").set_index("datetime")

# 方式3：从实验记录中获取
signal = R.get_recorder(experiment=1, recorder_id="abcd1234").load_object("pred.pkl")
```

### 2.2 预测信号到交易决策

策略层负责将预测信号转换为具体的交易决策。以TopkDropoutStrategy为例，这一转换过程如下：

```python
# qlib/contrib/strategy/signal_strategy.py
def generate_trade_decision(self, execute_result=None):
    # 获取当前时间点
    trade_step = self.trade_calendar.get_trade_step()
    
    # 1. 获取当前时间点的预测信号
    signal = self.signal.get_signal(trade_step)
    
    # 2. 获取当前持仓
    current_stock_list = self.trade_position.get_stock_list()
    
    # 3. 根据预测信号生成目标持仓
    target_stock_list = self._generate_target_stock_list(signal, trade_step, current_stock_list)
    
    # 4. 根据当前持仓和目标持仓生成订单列表
    order_list = self._generate_order_list(target_stock_list, current_stock_list, trade_step)
    
    # 5. 创建交易决策
    return TradeDecisionWO(order_list)
```

这个过程的关键步骤是：

#### 2.2.1 生成目标持仓列表

根据预测信号和当前持仓，生成目标持仓列表，确定应该持有哪些股票：

```python
def _generate_target_stock_list(self, signal, trade_step, current_stock_list):
    # 排除不可交易的股票
    if self.only_tradable:
        exclude_mask = self.trade_exchange.get_exclude_mask(trade_step)
        signal = signal.loc[~exclude_mask]
    
    # 排序股票（根据信号值）
    ranked_stocks = signal.sort_values(ascending=False)
    
    # 根据topk生成目标持仓
    if self.method_buy == "top":
        # 选择信号最高的topk只股票
        target_list = ranked_stocks.index[:self.topk].tolist()
    else:
        # 其他买入策略
        # ...
        
    return target_list
```

#### 2.2.2 生成订单列表

根据当前持仓和目标持仓的差异，生成具体的买入和卖出订单：

```python
def _generate_order_list(self, target_stock_list, current_stock_list, trade_step):
    # 计算需要卖出的股票
    stock_to_sell = [s for s in current_stock_list if s not in target_stock_list]
    # 限制每次卖出的数量
    if len(stock_to_sell) > self.n_drop:
        if self.method_sell == "bottom":
            signal = self.signal.get_signal(trade_step)
            # 按信号值排序，卖出信号最低的n_drop只股票
            stock_to_sell = signal.reindex(stock_to_sell).sort_values().index[:self.n_drop].tolist()
        # 其他卖出策略
        # ...
    
    # 计算需要买入的股票
    stock_to_buy = [s for s in target_stock_list if s not in current_stock_list]
    # 限制每次买入的数量
    if len(stock_to_buy) > self.n_drop:
        if self.method_buy == "top":
            signal = self.signal.get_signal(trade_step)
            # 按信号值排序，买入信号最高的n_drop只股票
            stock_to_buy = signal.reindex(stock_to_buy).sort_values(ascending=False).index[:self.n_drop].tolist()
        # 其他买入策略
        # ...
    
    # 生成订单列表
    order_list = []
    
    # 添加卖出订单
    for stock_id in stock_to_sell:
        # 卖出当前持有的全部股票
        amount = self.trade_position[stock_id]
        order_list.append(Order(stock_id=stock_id, amount=amount, direction=OrderDir.SELL))
    
    # 获取总资金
    cash = self.trade_exchange.get_cash_position(self, trade_step)
    trade_val = self.trade_exchange.get_trade_proportion_value(self, trade_step)
    
    # 添加买入订单
    for stock_id in stock_to_buy:
        # 计算买入金额（平均分配资金）
        buy_amount = trade_val / len(stock_to_buy)
        # 计算买入数量
        buy_price = self.trade_exchange.get_deal_price(stock_id, trade_step)
        amount = buy_amount / buy_price
        # 创建买入订单
        order_list.append(Order(stock_id=stock_id, amount=amount, direction=OrderDir.BUY))
    
    return order_list
```

### 2.3 权重策略中的持仓转换

除了TopkDropoutStrategy外，Qlib还提供了基于权重的策略，如WeightStrategyBase。在权重策略中，持仓转换过程如下：

```python
# qlib/contrib/strategy/signal_strategy.py
class WeightStrategyBase(BaseSignalStrategy):
    def generate_trade_decision(self, execute_result=None):
        # 获取当前时间
        trade_step = self.trade_calendar.get_trade_step()
        
        # 1. 获取股票权重
        weight_dict = self.get_stock_weight(trade_step)
        
        # 2. 获取可用资金
        risk_degree = self.get_risk_degree(trade_step)
        total_value = self.trade_exchange.get_total_value(self, trade_step)
        trade_val = risk_degree * total_value
        
        # 3. 通过订单生成器生成订单
        order_list = self.order_generator.generate(
            weight_dict=weight_dict,
            current_position=self.trade_position,
            trade_val=trade_val,
            trade_step=trade_step,
            trade_exchange=self.trade_exchange,
        )
        
        # 4. 创建交易决策
        return TradeDecisionWO(order_list)
```

在权重策略中，关键是权重计算和订单生成：

#### 2.3.1 权重计算

```python
def get_stock_weight(self, trade_step):
    # 获取预测信号
    signal = self.signal.get_signal(trade_step)
    
    # 计算权重
    if callable(self.weight_method):
        # 使用自定义权重计算函数
        weight_dict = self.weight_method(signal, trade_step)
    elif self.weight_method == "equal_weight":
        # 等权重分配
        weight_dict = {stock_id: 1.0 / len(signal) for stock_id in signal.index}
    elif self.weight_method == "market_value":
        # 市值加权
        market_value = self.trade_exchange.get_stock_value(signal.index, trade_step)
        total_value = market_value.sum()
        weight_dict = {stock_id: market_value[stock_id] / total_value for stock_id in signal.index}
    # 其他权重计算方法
    # ...
    
    return weight_dict
```

#### 2.3.2 订单生成

```python
# qlib/contrib/strategy/order_generator.py
class OrderGenWOInteract:
    def generate(self, weight_dict, current_position, trade_val, trade_step, trade_exchange):
        # 计算目标持仓市值
        target_pos_value = {}
        for stock_id, weight in weight_dict.items():
            target_pos_value[stock_id] = trade_val * weight
        
        # 计算当前持仓市值
        current_pos_value = {}
        for stock_id in current_position:
            current_amount = current_position[stock_id]
            deal_price = trade_exchange.get_deal_price(stock_id, trade_step)
            current_pos_value[stock_id] = current_amount * deal_price
        
        # 生成订单
        order_list = []
        
        # 卖出不在目标持仓中的股票
        for stock_id in current_position:
            if stock_id not in target_pos_value or target_pos_value[stock_id] == 0:
                amount = current_position[stock_id]
                order_list.append(Order(stock_id=stock_id, amount=amount, direction=OrderDir.SELL))
        
        # 买入或调整目标持仓中的股票
        for stock_id, target_value in target_pos_value.items():
            if target_value > 0:
                current_value = current_pos_value.get(stock_id, 0)
                if target_value != current_value:
                    # 计算需要调整的金额
                    diff_value = target_value - current_value
                    # 计算调整的数量
                    deal_price = trade_exchange.get_deal_price(stock_id, trade_step)
                    diff_amount = diff_value / deal_price
                    # 创建订单
                    if diff_amount > 0:
                        order_list.append(Order(stock_id=stock_id, amount=diff_amount, direction=OrderDir.BUY))
                    elif diff_amount < 0:
                        order_list.append(Order(stock_id=stock_id, amount=-diff_amount, direction=OrderDir.SELL))
        
        return order_list
```

## 3. 持仓到成交回报的转换

### 3.1 订单执行过程

一旦策略生成交易决策，执行器就会接收这些决策并执行订单。执行过程中会考虑市场条件、交易规则和成本等因素。

#### 3.1.1 执行器处理订单

SimulatorExecutor是Qlib中最常用的执行器，它模拟真实市场环境下的订单执行：

```python
# qlib/backtest/executor.py
class SimulatorExecutor(BaseExecutor):
    def execute(self, trade_decision: BaseTradeDecision):
        # 获取交易决策中的订单
        order_list = trade_decision.get_orders()
        
        # 处理每个订单
        for order in order_list:
            # 处理订单
            if order.direction == OrderDir.BUY:
                # 买入订单
                deal_amount, trade_val, cost = self._buy(order)
            elif order.direction == OrderDir.SELL:
                # 卖出订单
                deal_amount, trade_val, cost = self._sell(order)
            else:
                raise ValueError(f"订单方向无效: {order.direction}")
                
            # 更新订单状态
            order.deal_amount = deal_amount
            order.trade_val = trade_val
            order.cost = cost
        
        # 返回执行结果
        return trade_decision.generate_result_with_details(executed_orders=order_list)
```

#### 3.1.2 交易所执行订单

执行器依赖交易所来执行具体的订单操作：

```python
# qlib/backtest/exchange.py
class Exchange:
    def deal_order(self, order, position, position_dict=None):
        # 获取成交价格
        deal_price = self.get_deal_price(order.stock_id, order.start_time)
        
        # 考虑涨跌停限制
        if self.limit_threshold is not None:
            # 检查是否涨跌停
            if self.is_stock_limit(order.stock_id, order.start_time):
                # 涨跌停时，订单无法成交
                return deal_price, 0, 0, 0
        
        # 计算交易量和成本
        trade_val = order.amount * deal_price
        if order.direction == OrderDir.BUY:
            # 买入成本
            cost = max(trade_val * self.open_cost, self.min_cost)
        else:
            # 卖出成本
            cost = max(trade_val * self.close_cost, self.min_cost)
            
        # 考虑冲击成本
        cost += trade_val * self.impact_cost
        
        return deal_price, order.amount, trade_val, cost
```

### 3.2 账户更新过程

订单执行后，账户会根据执行结果更新持仓和现金状态：

```python
# qlib/backtest/account.py
class Account:
    def update_order(self, order, trade_val, cost, trade_price):
        # 更新现金
        if order.direction == OrderDir.BUY:
            # 买入：减少现金
            self.cash -= (trade_val + cost)
        else:
            # 卖出：增加现金
            self.cash += (trade_val - cost)
        
        # 更新持仓
        self.current_position.update_order(order, trade_val, cost, trade_price)
        
        # 更新累计信息
        if order.direction == OrderDir.BUY:
            self._accumulated_info.add_turnover(trade_val)
        else:
            self._accumulated_info.add_turnover(trade_val)
        self._accumulated_info.add_cost(cost)
```

### 3.3 持仓更新过程

账户中的持仓对象负责管理具体的股票持仓：

```python
# qlib/backtest/position.py
class Position(BasePosition):
    def update_order(self, order, trade_val, cost, trade_price):
        stock_id = order.stock_id
        # 更新持仓
        if order.direction == OrderDir.BUY:
            # 买入：增加持仓
            if stock_id not in self:
                # 新建持仓
                self[stock_id] = order.deal_amount
            else:
                # 增加现有持仓
                self[stock_id] += order.deal_amount
        else:
            # 卖出：减少持仓
            if stock_id not in self:
                raise KeyError(f"卖出不存在的股票: {stock_id}")
            # 减少持仓
            self[stock_id] -= order.deal_amount
            # 如果持仓变为0，删除该股票
            if self[stock_id] == 0:
                del self[stock_id]
```

### 3.4 PNL计算过程

在每个交易日结束时，账户会计算当日收益：

```python
# qlib/backtest/account.py
class Account:
    def update_bar_end(self, trade_start_time, trade_end_time, trade_exchange):
        # 更新持仓价值
        self.update_current_position(trade_start_time, trade_end_time, trade_exchange)
        
        # 计算当前总资产
        portfolio_value = self.current_position.calculate_value(trade_end_time) + self.cash
        
        # 计算收益率
        if self.portfolio_metrics.last_assets > 0:
            returns = portfolio_value / self.portfolio_metrics.last_assets - 1
        else:
            returns = 0
            
        # 更新投资组合指标
        self.portfolio_metrics.accounts[trade_end_time] = portfolio_value
        self.portfolio_metrics.returns[trade_end_time] = returns
        
        # 更新累计信息
        if returns != 0:
            self._accumulated_info.add_return_value(returns)
            
        # 更新上一次资产值
        self.portfolio_metrics.last_assets = portfolio_value
```

## 4. 回测引擎调用流程

### 4.1 回测初始化

回测开始前，需要初始化回测环境，包括执行器、交易所和账户等：

```python
# 初始化回测环境
def init_backtest_env(pred_scores, start_time, end_time):
    # 创建交易所
    exchange = Exchange(
        freq="day",
        start_time=start_time,
        end_time=end_time,
        limit_threshold=0.095,
        deal_price="close",
        open_cost=0.0005,
        close_cost=0.0015,
        min_cost=5,
    )
    
    # 创建账户
    account = Account(init_cash=100000000, freq="day")
    
    # 创建策略
    strategy = TopkDropoutStrategy(
        signal=pred_scores,
        topk=50,
        n_drop=5,
        risk_degree=0.95,
        trade_exchange=exchange,
    )
    
    # 创建执行器
    executor = SimulatorExecutor(
        time_per_step="day",
        start_time=start_time,
        end_time=end_time,
        trade_exchange=exchange,
        trade_account=account,
    )
    
    return strategy, executor
```

### 4.2 回测执行

初始化完成后，可以执行回测循环：

```python
# 执行回测
def run_backtest(strategy, executor, start_time, end_time):
    # 执行回测循环
    portfolio_dict, indicator_dict = backtest_loop(
        start_time=start_time,
        end_time=end_time,
        trade_strategy=strategy,
        trade_executor=executor,
    )
    
    return portfolio_dict, indicator_dict
```

回测循环的内部实现如下：

```python
# qlib/backtest/backtest.py
def backtest_loop(start_time, end_time, trade_strategy, trade_executor):
    trade_executor.reset(start_time=start_time, end_time=end_time)
    trade_strategy.reset(level_infra=trade_executor.get_level_infra())
    
    with tqdm(total=trade_executor.trade_calendar.get_trade_len(), desc="backtest loop") as bar:
        _execute_result = None
        while not trade_executor.finished():
            # 1. 策略生成交易决策
            _trade_decision = trade_strategy.generate_trade_decision(_execute_result)
            
            # 2. 执行器执行交易决策
            _execute_result = trade_executor.collect_data(_trade_decision, level=0)
            
            # 3. 策略处理执行结果
            trade_strategy.post_exe_step(_execute_result)
            
            bar.update(1)
        trade_strategy.post_upper_level_exe_step()
    
    # 4. 获取回测结果
    all_executors = trade_executor.get_all_executors()
    portfolio_dict = {}
    indicator_dict = {}
    
    for exec_path, executor in all_executors.items():
        # 获取投资组合指标
        portfolio_metrics = executor.trade_account.portfolio_metrics
        portfolio_dict[exec_path] = (portfolio_metrics.generate_portfolio_dataframe(), portfolio_metrics)
        # 获取性能指标
        indicator = Indicator(portfolio_metrics)
        indicator_dict[exec_path] = (indicator.generate_indicator_dataframe(), indicator)
    
    return portfolio_dict, indicator_dict
```

### 4.3 结果分析

回测完成后，可以分析回测结果：

```python
# 分析回测结果
def analyze_backtest_result(portfolio_dict, indicator_dict):
    # 获取投资组合指标
    portfolio_df, portfolio_metrics = portfolio_dict["portfolio"]
    
    # 获取性能指标
    indicator_df, indicator = indicator_dict["indicator"]
    
    # 打印关键指标
    print(f"年化收益率: {indicator_df.loc['annualized_return'].values[0]:.4f}")
    print(f"夏普比率: {indicator_df.loc['sharpe'].values[0]:.4f}")
    print(f"最大回撤: {indicator_df.loc['max_drawdown'].values[0]:.4f}")
    print(f"胜率: {indicator_df.loc['win_rate'].values[0]:.4f}")
    
    # 绘制收益曲线
    plot_portfolio_performance(portfolio_metrics)
    
    return portfolio_df, indicator_df
```

## 5. 实际应用示例

下面是一个完整的回测示例，展示了从模型预测到回测结果的全流程：

```python
import qlib
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest.executor import SimulatorExecutor
from qlib.backtest.backtest import backtest_loop
from qlib.contrib.report import analysis_position, analysis_model
from qlib.contrib.evaluate import risk_analysis, backtest

# 1. 初始化Qlib
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")

# 2. 准备数据
data_handler = Alpha158(
    start_time="2008-01-01",
    end_time="2020-12-31",
    fit_start_time="2008-01-01",
    fit_end_time="2014-12-31",
    instruments="csi300",
)

# 3. 训练模型
model = LGBModel()
model.fit(data_handler)

# 4. 生成预测信号
pred_scores = model.predict(data_handler)

# 5. 创建策略
strategy = TopkDropoutStrategy(
    signal=pred_scores,
    topk=50,
    n_drop=5,
    risk_degree=0.95,
)

# 6. 创建执行器
executor = SimulatorExecutor(
    time_per_step="day",
    start_time="2015-01-01",
    end_time="2020-12-31",
)

# 7. 执行回测
portfolio_dict, indicator_dict = backtest_loop(
    start_time="2015-01-01",
    end_time="2020-12-31",
    trade_strategy=strategy,
    trade_executor=executor,
)

# 8. 分析回测结果
portfolio_df, portfolio_metrics = portfolio_dict["portfolio"]
indicator_df, indicator = indicator_dict["indicator"]

# 9. 打印关键指标
print(f"年化收益率: {indicator_df.loc['annualized_return'].values[0]:.4f}")
print(f"夏普比率: {indicator_df.loc['sharpe'].values[0]:.4f}")
print(f"最大回撤: {indicator_df.loc['max_drawdown'].values[0]:.4f}")

# 10. 可视化分析
analysis_position.report_graph(
    executor.trade_account.get_hist_positions(),
    portfolio_metrics,
)
```

## 6. 总结

Qlib的回测引擎提供了一套完整的机制，将模型预测转化为交易决策、执行交易并计算收益。整个过程可以概括为：

1. **模型预测转化为信号**：模型生成预测分数，表示每只股票的预期收益
2. **信号转化为交易决策**：策略根据预测信号生成目标持仓，并与当前持仓比较生成交易订单
3. **交易决策转化为订单执行**：执行器接收交易决策并执行订单，考虑市场条件、交易规则和成本
4. **订单执行转化为持仓更新**：根据订单执行结果更新账户持仓和现金
5. **持仓更新转化为收益计算**：根据持仓价值变化计算策略收益和风险指标

通过这一流程，Qlib的回测引擎能够全面模拟投资策略的表现，帮助用户评估和优化策略，为实盘交易提供决策依据。 