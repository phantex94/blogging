# Qlib 模型实验记录 (MLRUNS)

## 1. 实验记录概述

Qlib 提供了强大的实验记录功能，使用户能够跟踪、管理和分析量化投资研究中的实验结果。这一功能主要基于 `mlflow` 实现，但 Qlib 对其进行了封装和扩展，提供了更适合量化投资场景的接口和功能。

### 1.1 实验记录的意义

在量化投资研究中，实验记录具有重要意义：

- **可重复性**：记录实验参数和结果，确保研究可以被重复验证
- **可比较性**：比较不同模型、策略的性能
- **可追溯性**：追踪模型和策略的演变历程
- **协作共享**：团队成员之间共享研究结果

### 1.2 实验记录体系结构

Qlib 的实验记录体系主要包括以下组件：

- **实验管理器 (ExperimentManager)**：管理实验的创建、查询和删除
- **记录器 (Recorder)**：记录单个实验的参数、指标和产物
- **记录模板 (Record)**：预定义的记录生成器，用于生成特定类型的记录

## 2. 实验管理器

### 2.1 实验管理器架构

实验管理器是实验记录的核心组件，负责管理所有实验：

```python
# qlib/workflow/expm.py
class ExperimentManager:
    """实验管理器基类"""
    
    def __init__(self, uri=None, default_exp_name="Experiment"):
        self.uri = uri
        self.default_exp_name = default_exp_name
        
    def start_exp(self, *args, **kwargs):
        """开始新实验"""
        raise NotImplementedError("必须实现start_exp方法")
        
    def end_exp(self, *args, **kwargs):
        """结束实验"""
        raise NotImplementedError("必须实现end_exp方法")
        
    def list_experiments(self):
        """列出所有实验"""
        raise NotImplementedError("必须实现list_experiments方法")
        
    def list_recorders(self, experiment_id, **kwargs):
        """列出实验中的记录器"""
        raise NotImplementedError("必须实现list_recorders方法")
        
    def search_records(self, experiment_ids, **kwargs):
        """搜索记录"""
        raise NotImplementedError("必须实现search_records方法")
```

### 2.2 MLflow 实验管理器

Qlib 默认使用 MLflow 作为实验管理器的后端：

```python
# qlib/workflow/expm.py
class MLflowExpManager(ExperimentManager):
    """基于 MLflow 的实验管理器"""
    
    def __init__(self, uri=None, default_exp_name="Experiment"):
        super().__init__(uri, default_exp_name)
        
        import mlflow
        self.mlflow = mlflow
        if uri is not None:
            self.mlflow.set_tracking_uri(uri)
        self.client = self.mlflow.tracking.MlflowClient()
        
    def start_exp(self, experiment_name=None, recorder_name=None, recorder_id=None, resume=False):
        """开始新实验"""
        # ...

    def end_exp(self, recorder_status=None):
        """结束实验"""
        # ...

    def list_experiments(self):
        """列出所有实验"""
        exps = self.client.search_experiments(view_type=ViewType.ACTIVE_ONLY)
        return [(exp.experiment_id, exp.name) for exp in exps]
        
    def list_recorders(self, experiment_id, **kwargs):
        """列出实验中的记录器"""
        # ...

    def search_records(self, experiment_ids=None, **kwargs):
        """搜索记录"""
        # ...
```

### 2.3 实验管理器配置

在 Qlib 初始化时可以配置实验管理器：

```python
# 配置 MLflow 实验管理器
qlib.init(
    provider_uri="~/.qlib/qlib_data/cn_data",
    region="cn",
    exp_manager={
        "class": "MLflowExpManager",
        "module_path": "qlib.workflow.expm",
        "kwargs": {
            "uri": "file:///path/to/mlruns",
            "default_exp_name": "my_experiment",
        }
    }
)
```

## 3. 记录器

### 3.1 记录器架构

记录器是记录单个实验运行的组件，负责存储实验的参数、指标和产物：

```python
# qlib/workflow/recorder.py
class Recorder:
    """记录器基类"""
    
    def __init__(self, experiment_id=None, name=None):
        self.id = None
        self.name = name
        self.experiment_id = experiment_id
        
    def log_params(self, **params):
        """记录参数"""
        raise NotImplementedError("必须实现log_params方法")
        
    def log_metrics(self, **metrics):
        """记录指标"""
        raise NotImplementedError("必须实现log_metrics方法")
        
    def log_artifact(self, local_path, artifact_path=None):
        """记录产物"""
        raise NotImplementedError("必须实现log_artifact方法")
        
    def save_objects(self, **objects):
        """保存对象"""
        raise NotImplementedError("必须实现save_objects方法")
        
    def load_object(self, name):
        """加载对象"""
        raise NotImplementedError("必须实现load_object方法")
```

### 3.2 MLflow 记录器

Qlib 默认使用基于 MLflow 的记录器：

```python
# qlib/workflow/recorder.py
class MLflowRecorder(Recorder):
    """基于 MLflow 的记录器"""
    
    def __init__(self, experiment_id=None, name=None):
        super().__init__(experiment_id, name)
        self.client = MlflowClient()
        self.uri = mlflow.get_tracking_uri()
        
    def log_params(self, **params):
        """记录参数"""
        for k, v in params.items():
            self.client.log_param(self.id, k, v)
            
    def log_metrics(self, **metrics):
        """记录指标"""
        for k, v in metrics.items():
            if isinstance(v, pd.Series):
                for i, sv in v.items():
                    self.client.log_metric(self.id, k, sv, step=i)
            else:
                self.client.log_metric(self.id, k, v)
                
    def log_artifact(self, local_path, artifact_path=None):
        """记录产物"""
        self.client.log_artifact(self.id, local_path, artifact_path)
        
    def save_objects(self, **objects):
        """保存对象"""
        with tempfile.TemporaryDirectory() as d:
            for name, obj in objects.items():
                path = os.path.join(d, name)
                if isinstance(obj, (pd.DataFrame, pd.Series)):
                    obj.to_pickle(path)
                else:
                    with open(path, "wb") as f:
                        pickle.dump(obj, f)
                self.client.log_artifact(self.id, path)
                
    def load_object(self, name):
        """加载对象"""
        path = os.path.join(self.client.download_artifacts(self.id, ""), name)
        with open(path, "rb") as f:
            return pickle.load(f)
```

### 3.3 记录器使用示例

记录器的使用方式如下：

```python
# 获取记录器
with R.start(experiment_name="my_experiment"):
    # 记录参数
    R.log_params(learning_rate=0.01, n_estimators=100)
    
    # 训练模型
    model = LGBModel(learning_rate=0.01, n_estimators=100)
    model.fit(dataset)
    
    # 记录指标
    pred = model.predict(dataset)
    ic = calculate_ic(pred, label)
    R.log_metrics(ic=ic)
    
    # 保存对象
    R.save_objects(model=model, prediction=pred)
```

## 4. 记录模板

### 4.1 记录模板架构

记录模板是预定义的记录生成器，用于生成特定类型的记录：

```python
# qlib/workflow/record_temp.py
class Record:
    """记录模板基类"""
    
    def __init__(self, recorder=None):
        self.recorder = recorder
        
    def generate(self, **kwargs):
        """生成记录"""
        raise NotImplementedError("必须实现generate方法")
```

### 4.2 预定义记录模板

Qlib 提供了多种预定义的记录模板：

#### 4.2.1 SignalRecord

用于记录模型的预测信号：

```python
# qlib/workflow/record_temp.py
class SignalRecord(Record):
    def __init__(self, model=None, dataset=None, recorder=None):
        super().__init__(recorder)
        self.model = model
        self.dataset = dataset
        
    def generate(self):
        # 生成预测信号
        pred = self.model.predict(self.dataset)
        self.recorder.save_objects(**{"pred.pkl": pred})
        return pred
```

#### 4.2.2 SigAnaRecord

用于分析预测信号的性能：

```python
# qlib/workflow/record_temp.py
class SigAnaRecord(Record):
    def __init__(self, recorder=None, ana_long_short=False, ann_scaler=252):
        super().__init__(recorder)
        self.ana_long_short = ana_long_short
        self.ann_scaler = ann_scaler
        
    def generate(self):
        # 加载预测结果
        pred = self.recorder.load_object("pred.pkl")
        
        # 计算评估指标
        analysis = dict()
        analysis["analysis"] = self._calculate_metrics(pred)
        
        # 保存评估结果
        self.recorder.save_objects(**analysis)
        return analysis
```

#### 4.2.3 PortAnaRecord

用于分析投资组合的性能：

```python
# qlib/workflow/record_temp.py
class PortAnaRecord(Record):
    def __init__(self, recorder=None, config=None, freq="day"):
        super().__init__(recorder)
        self.config = config
        self.freq = freq
        
    def generate(self):
        # 加载预测结果
        pred = self.recorder.load_object("pred.pkl")
        
        # 构建回测对象
        strategy_config = self.config["strategy"]
        strategy_config["kwargs"]["signal"] = pred
        strategy = init_instance_by_config(strategy_config)
        
        # 执行回测
        executor_config = self.config["executor"]
        executor = init_instance_by_config(executor_config)
        portfolio = executor.run(strategy)
        
        # 保存回测结果
        report_df = portfolio.report_df
        self.recorder.save_objects(**{"portfolio_analysis": report_df})
        return report_df
```

### 4.3 记录模板使用示例

记录模板的使用方式如下：

```python
# 获取记录器
with R.start(experiment_name="my_experiment"):
    model = init_instance_by_config(model_config)
    dataset = init_instance_by_config(dataset_config)
    
    # 训练模型
    model.fit(dataset)
    
    # 记录预测信号
    sr = SignalRecord(model, dataset, R.get_recorder())
    sr.generate()
    
    # 分析预测信号
    sar = SigAnaRecord(recorder=R.get_recorder())
    sar.generate()
    
    # 分析投资组合
    par = PortAnaRecord(recorder=R.get_recorder(), config=backtest_config)
    par.generate()
```

## 5. 实验记录存储

### 5.1 存储结构

Qlib 的实验记录默认存储在 MLflow 的目录结构中：

```
mlruns/
  ├── 0/                               # 实验 ID
  │   ├── meta.yaml                    # 实验元信息
  │   ├── abcd1234/                    # 记录器 ID (Run ID)
  │   │   ├── meta.yaml                # 记录器元信息
  │   │   ├── params/                  # 参数
  │   │   │   ├── learning_rate        # 参数名
  │   │   │   └── n_estimators         # 参数名
  │   │   ├── metrics/                 # 指标
  │   │   │   └── ic                   # 指标名
  │   │   └── artifacts/               # 产物
  │   │       ├── model.pkl            # 模型文件
  │   │       ├── pred.pkl             # 预测结果
  │   │       └── portfolio_analysis.pkl # 回测结果
  │   └── ...
  └── ...
```

### 5.2 存储配置

可以通过配置 MLflow 的 URI 来指定存储位置：

```python
# 本地文件系统
qlib.init(exp_manager={"kwargs": {"uri": "file:///path/to/mlruns"}})

# 远程服务器
qlib.init(exp_manager={"kwargs": {"uri": "http://localhost:5000"}})

# SQLite 数据库
qlib.init(exp_manager={"kwargs": {"uri": "sqlite:///path/to/mlruns.db"}})
```

### 5.3 产物存储

实验产物（如模型、预测结果、回测结果）通常以 pickle 格式存储：

```python
# 保存对象
def save_objects(self, **objects):
    with tempfile.TemporaryDirectory() as d:
        for name, obj in objects.items():
            path = os.path.join(d, name)
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                obj.to_pickle(path)
            else:
                with open(path, "wb") as f:
                    pickle.dump(obj, f)
            self.client.log_artifact(self.id, path)
            
# 加载对象
def load_object(self, name):
    path = os.path.join(self.client.download_artifacts(self.id, ""), name)
    with open(path, "rb") as f:
        return pickle.load(f)
```

## 6. 实验记录查询和分析

### 6.1 查询实验

可以通过 `R` 对象查询实验：

```python
# 列出所有实验
experiments = R.list_experiments()
print(experiments)  # [(0, "Default"), (1, "my_experiment"), ...]

# 列出实验中的记录器
recorders = R.list_recorders(experiment_id=1)
print(recorders)
```

### 6.2 搜索记录

可以根据条件搜索记录：

```python
# 搜索记录
records = R.search_records(
    experiment_ids=[1],
    filter_string="params.learning_rate='0.01'",
    order_by=["metrics.ic DESC"]
)
print(records)
```

### 6.3 加载模型和结果

可以加载指定记录器中的对象：

```python
# 获取记录器
recorder = R.get_recorder(experiment=1, recorder_id="abcd1234")

# 加载模型
model = recorder.load_object("model.pkl")

# 加载预测结果
pred = recorder.load_object("pred.pkl")

# 加载回测结果
portfolio = recorder.load_object("portfolio_analysis.pkl")
```

### 6.4 对比分析

可以对比不同实验的结果：

```python
# 获取多个记录器
recorder1 = R.get_recorder(experiment=1, recorder_id="abcd1234")
recorder2 = R.get_recorder(experiment=1, recorder_id="efgh5678")

# 加载预测结果
pred1 = recorder1.load_object("pred.pkl")
pred2 = recorder2.load_object("pred.pkl")

# 对比预测结果
comparison = pd.DataFrame({
    "model1": pred1,
    "model2": pred2
})
correlation = comparison.corr()
print(correlation)

# 对比回测结果
port1 = recorder1.load_object("portfolio_analysis.pkl")
port2 = recorder2.load_object("portfolio_analysis.pkl")
print(f"Model 1 Sharpe: {port1.loc['mean'].loc['sharp']}")
print(f"Model 2 Sharpe: {port2.loc['mean'].loc['sharp']}")
```

## 7. Web UI

MLflow 提供了 Web UI 界面，可以方便地浏览和分析实验结果：

```bash
# 启动 MLflow UI
mlflow ui --backend-store-uri file:///path/to/mlruns
```

Web UI 提供了以下功能：

- **实验列表**：查看所有实验
- **记录器列表**：查看实验中的所有记录器
- **参数对比**：对比不同记录器的参数
- **指标对比**：对比不同记录器的指标
- **产物查看**：查看和下载记录器的产物

## 8. 最佳实践

### 8.1 命名规范

- **实验名称**：使用有意义的名称，如 `lgb_alpha158`, `nn_model_v2` 等
- **记录器名称**：使用描述性名称，如 `lr_0.01_ne_100`, `dropout_0.5` 等

### 8.2 参数记录

记录所有重要参数，包括：

- **模型参数**：如 learning_rate, n_estimators 等
- **数据集参数**：如 features, labels, segments 等
- **训练参数**：如 batch_size, epochs 等
- **环境参数**：如 seed, device 等

### 8.3 指标记录

记录所有重要指标，包括：

- **训练指标**：如 train_loss, valid_loss 等
- **评估指标**：如 IC, ICIR, Rank IC 等
- **回测指标**：如 annualized_return, sharpe, max_drawdown 等

### 8.4 产物记录

记录所有重要产物，包括：

- **模型**：保存训练好的模型
- **预测结果**：保存模型的预测结果
- **回测结果**：保存策略的回测结果
- **可视化结果**：保存图表和分析报告

## 9. 总结

Qlib 的实验记录功能通过 MLflow 提供了强大的实验管理能力，使量化投资研究更加系统化和规范化：

1. **实验管理器**：管理实验的创建、查询和删除
2. **记录器**：记录单个实验的参数、指标和产物
3. **记录模板**：预定义的记录生成器，用于生成特定类型的记录
4. **存储机制**：灵活的存储配置，支持本地和远程存储
5. **查询分析**：丰富的查询和分析功能，支持对比不同实验
6. **Web UI**：直观的界面，方便浏览和分析实验结果

通过这些功能，研究人员可以更好地跟踪、管理和分析量化投资研究中的实验结果，提高研究效率和质量。 