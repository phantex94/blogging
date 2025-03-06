# Qlib 数据缓存机制

## 1. 简介

Qlib 处理大量金融数据时，高效的缓存机制是提升性能的关键。本文档详细介绍 Qlib 的数据缓存机制，包括缓存架构、实现细节和优化策略。

## 2. 缓存架构概览

Qlib 实现了多级缓存架构，主要包括以下几个层次：

### 2.1 缓存层次

- **内存缓存（Memory Cache）**：存储在内存中的缓存，访问速度最快
- **磁盘缓存（Disk Cache）**：存储在磁盘上的缓存，容量大但访问较慢
- **分布式缓存**：通过 Redis 实现的分布式缓存，支持多进程/多机器共享

### 2.2 缓存类型

- **表达式缓存（Expression Cache）**：缓存表达式计算结果
- **数据集缓存（Dataset Cache）**：缓存处理后的数据集
- **日历缓存（Calendar Cache）**：缓存交易日历
- **股票列表缓存（Instrument Cache）**：缓存股票列表

### 2.3 缓存策略

- **LRU（最近最少使用）**：优先淘汰最久未使用的缓存项
- **过期时间**：设置缓存项的过期时间，自动清理过期项
- **大小限制**：限制缓存大小，防止内存溢出

## 3. 内存缓存实现

### 3.1 内存缓存单元

Qlib 的内存缓存基于 `MemCacheUnit` 类实现：

```python
# qlib/data/cache.py
class MemCacheUnit(abc.ABC):
    """内存缓存单元"""
    
    def __init__(self, *args, **kwargs):
        self.size_limit = kwargs.pop("size_limit", 0)
        self._size = 0
        self.od = OrderedDict()  # 使用 OrderedDict 实现 LRU
        
    def __setitem__(self, key, value):
        # 如果键已存在，先移除旧值
        if key in self.od:
            self._size -= self._get_value_size(self.od[key])
            
        # 添加新值
        self.od[key] = value
        self._size += self._get_value_size(value)
        
        # 如果超过大小限制，移除最旧的项
        while self.limited and self.od:
            self.popitem(last=False)
            
    def __getitem__(self, key):
        # 获取值并更新访问顺序
        value = self.od[key]
        self.od.move_to_end(key)
        return value
        
    def popitem(self, last=True):
        # 移除最旧/最新的项
        key, value = self.od.popitem(last=last)
        self._size -= self._get_value_size(value)
        return key, value
        
    @abc.abstractmethod
    def _get_value_size(self, value):
        # 获取值的大小
        raise NotImplementedError("_get_value_size not implemented!")
```

Qlib 提供了两种内存缓存单元实现：

```python
class MemCacheLengthUnit(MemCacheUnit):
    """基于项目数量限制缓存大小"""
    
    def _get_value_size(self, value):
        return 1
        
class MemCacheSizeofUnit(MemCacheUnit):
    """基于内存占用大小限制缓存大小"""
    
    def _get_value_size(self, value):
        return sys.getsizeof(value)
```

### 3.2 内存缓存管理器

`MemCache` 类管理不同类型的内存缓存：

```python
# qlib/data/cache.py
class MemCache:
    """内存缓存管理器"""
    
    def __init__(self, mem_cache_size_limit=None, limit_type="length"):
        size_limit = C.mem_cache_size_limit if mem_cache_size_limit is None else mem_cache_size_limit
        limit_type = C.mem_cache_limit_type if limit_type is None else limit_type
        
        if limit_type == "length":
            klass = MemCacheLengthUnit
        elif limit_type == "sizeof":
            klass = MemCacheSizeofUnit
        else:
            raise ValueError(f"limit_type must be length or sizeof, your limit_type is {limit_type}")
            
        self.__calendar_mem_cache = klass(size_limit)  # 日历缓存
        self.__instrument_mem_cache = klass(size_limit)  # 股票列表缓存
        self.__feature_mem_cache = klass(size_limit)  # 特征缓存
        
    def __getitem__(self, key):
        if key == "c":
            return self.__calendar_mem_cache
        elif key == "i":
            return self.__instrument_mem_cache
        elif key == "f":
            return self.__feature_mem_cache
        else:
            raise KeyError("Unknown memcache unit")
            
    def clear(self):
        self.__calendar_mem_cache.clear()
        self.__instrument_mem_cache.clear()
        self.__feature_mem_cache.clear()
```

### 3.3 带过期时间的内存缓存

`MemCacheExpire` 类为内存缓存添加了过期时间功能：

```python
# qlib/data/cache.py
class MemCacheExpire:
    """带过期时间的内存缓存"""
    
    CACHE_EXPIRE = C.mem_cache_expire  # 默认过期时间
    
    @staticmethod
    def set_cache(mem_cache, key, value):
        # 存储值和时间戳
        mem_cache[key] = (value, time.time())
        
    @staticmethod
    def get_cache(mem_cache, key):
        if key in mem_cache:
            value, set_time = mem_cache[key]
            # 检查是否过期
            if time.time() - set_time < MemCacheExpire.CACHE_EXPIRE:
                return value
        return None
```

## 4. 磁盘缓存实现

### 4.1 表达式磁盘缓存

`DiskExpressionCache` 类实现了表达式计算结果的磁盘缓存：

```python
# qlib/data/cache.py
class DiskExpressionCache(ExpressionCache):
    """表达式磁盘缓存"""
    
    def __init__(self, provider, **kwargs):
        super(DiskExpressionCache, self).__init__(provider)
        self.r = get_redis_connection()
        self.remote = kwargs.get("remote", False)
        
    def get_cache_dir(self, freq: str = None) -> Path:
        return super(DiskExpressionCache, self).get_cache_dir(C.features_cache_dir_name, freq)
        
    def _uri(self, instrument, field, start_time, end_time, freq):
        # 生成缓存文件的 URI
        field = remove_fields_space(field)
        instrument = str(instrument).lower()
        return hash_args(instrument, field, freq)
        
    def _expression(self, instrument, field, start_time=None, end_time=None, freq="day"):
        # 获取缓存路径
        _cache_uri = self._uri(instrument=instrument, field=field, start_time=None, end_time=None, freq=freq)
        _instrument_dir = self.get_cache_dir(freq).joinpath(instrument.lower())
        cache_path = _instrument_dir.joinpath(_cache_uri)
        
        # 获取日历和索引
        from .data import Cal
        _calendar = Cal.calendar(freq=freq)
        _, _, start_index, end_index = Cal.locate_index(start_time, end_time, freq, future=False)
        
        if self.check_cache_exists(cache_path, suffix_list=[".meta"]):
            # 缓存存在，直接读取
            try:
                if not self.remote:
                    CacheUtils.visit(cache_path)
                series = read_bin(cache_path, start_index, end_index)
                return series
            except Exception:
                # 读取失败处理
                self.logger.error("reading %s file error : %s" % (cache_path, traceback.format_exc()))
                series = None
            return series
        else:
            # 缓存不存在，生成缓存
            field = remove_fields_space(field)
            _instrument_dir.mkdir(parents=True, exist_ok=True)
            
            if not isinstance(eval(parse_field(field)), Feature):
                # 非原始特征，需要计算
                series = self.provider.expression(instrument, field, _calendar[0], _calendar[-1], freq)
                if not series.empty:
                    # 使用分布式锁确保写入安全
                    with CacheUtils.writer_lock(self.r, f"{str(C.dpm.get_data_uri(freq))}:expression-{_cache_uri}"):
                        self.gen_expression_cache(cache_path, series, instrument, field, freq)
                return series[start_index:end_index]
            else:
                # 原始特征，直接从提供者获取
                return self.provider.expression(instrument, field, start_time, end_time, freq)
```

### 4.2 数据集磁盘缓存

`DiskDatasetCache` 类实现了数据集的磁盘缓存：

```python
# qlib/data/cache.py
class DiskDatasetCache(DatasetCache):
    """数据集磁盘缓存"""
    
    def __init__(self, provider, **kwargs):
        super(DiskDatasetCache, self).__init__(provider)
        self.r = get_redis_connection()
        self.remote = kwargs.get("remote", False)
        
    @staticmethod
    def _uri(instruments, fields, start_time, end_time, freq, disk_cache=1, inst_processors=[], **kwargs):
        # 生成缓存文件的 URI
        return hash_args(*DatasetCache.normalize_uri_args(instruments, fields, freq), disk_cache, inst_processors)
        
    def get_cache_dir(self, freq: str = None) -> Path:
        return super(DiskDatasetCache, self).get_cache_dir(C.dataset_cache_dir_name, freq)
        
    @classmethod
    def read_data_from_cache(cls, cache_path: Union[str, Path], start_time, end_time, fields):
        """从缓存读取数据"""
        # 使用索引管理器
        im = DiskDatasetCache.IndexManager(cache_path)
        index_data = im.get_index(start_time, end_time)
        
        # 确定读取范围
        if index_data.shape[0] > 0:
            start, stop = (
                index_data["start"].iloc[0].item(),
                index_data["end"].iloc[-1].item(),
            )
        else:
            start = stop = 0
            
        # 从 HDF 文件读取数据
        with pd.HDFStore(cache_path, mode="r") as store:
            if "/{}".format(im.KEY) in store.keys():
                df = store.select(key=im.KEY, start=start, stop=stop)
                df = df.swaplevel("datetime", "instrument").sort_index()
                # 将缓存数据转换为原始数据格式
                df = cls.cache_to_origin_data(df, fields)
            else:
                df = pd.DataFrame(columns=fields)
        return df
        
    def _dataset(self, instruments, fields, start_time=None, end_time=None, freq="day", disk_cache=0, inst_processors=[]):
        """获取数据集，使用缓存"""
        if disk_cache == 0:
            # 跳过缓存
            return self.provider.dataset(instruments, fields, start_time, end_time, freq, inst_processors=inst_processors)
            
        # 获取缓存路径
        _cache_uri = self._uri(instruments, fields, start_time, end_time, freq, disk_cache, inst_processors)
        cache_path = self.get_cache_dir(freq).joinpath(_cache_uri)
        
        if self.check_cache_exists(cache_path):
            # 缓存存在，直接读取
            try:
                return self.read_data_from_cache(cache_path, start_time, end_time, fields)
            except Exception:
                self.logger.error("reading %s file error : %s" % (cache_path, traceback.format_exc()))
                
        # 缓存不存在或读取失败，从提供者获取数据
        data = self.provider.dataset(instruments, fields, start_time, end_time, freq, inst_processors=inst_processors)
        
        if disk_cache == 1:
            # 使用缓存但不替换
            if not self.check_cache_exists(cache_path):
                # 缓存不存在，创建缓存
                self.dump_dataset_cache(cache_path, data, start_time, end_time)
        elif disk_cache == 2:
            # 使用缓存并替换
            self.dump_dataset_cache(cache_path, data, start_time, end_time)
            
        return data
```

### 4.3 索引管理器

`IndexManager` 是 `DiskDatasetCache` 的内部类，用于管理数据集的索引：

```python
# qlib/data/cache.py
class IndexManager:
    """索引管理器"""
    
    KEY = "df"
    INDEX_KEY = "index"
    
    def __init__(self, cache_path: Union[str, Path]):
        self.cache_path = Path(cache_path)
        self.index = None
        self.sync_from_disk()
        
    def get_index(self, start_time=None, end_time=None):
        """获取指定时间范围的索引"""
        if self.index is None or self.index.empty:
            return pd.DataFrame(columns=["start", "end", "start_time", "end_time"])
            
        if start_time is None and end_time is None:
            return self.index
            
        # 过滤索引
        mask = np.ones(len(self.index), dtype=bool)
        if start_time is not None:
            mask &= self.index["end_time"] >= pd.Timestamp(start_time)
        if end_time is not None:
            mask &= self.index["start_time"] <= pd.Timestamp(end_time)
            
        return self.index[mask]
        
    def sync_to_disk(self):
        """将索引同步到磁盘"""
        if self.index is not None:
            with pd.HDFStore(self.cache_path, mode="a") as store:
                store.put(self.INDEX_KEY, self.index)
                
    def sync_from_disk(self):
        """从磁盘读取索引"""
        if self.cache_path.exists():
            with pd.HDFStore(self.cache_path, mode="r") as store:
                if f"/{self.INDEX_KEY}" in store.keys():
                    self.index = store[self.INDEX_KEY]
                    
    def update(self, data, sync=True):
        """更新索引"""
        # ...
        
    def append_index(self, data, to_disk=True):
        """追加索引"""
        # ...
        
    @staticmethod
    def build_index_from_data(data, start_index=0):
        """从数据构建索引"""
        # ...
```

## 5. 缓存锁机制

为了处理多进程/多线程环境下的并发访问，Qlib 实现了基于 Redis 的分布式锁机制：

```python
# qlib/data/cache.py
class CacheUtils:
    """缓存工具类"""
    
    LOCK_ID = "QLIB"
    
    @staticmethod
    @contextlib.contextmanager
    def reader_lock(redis_t, lock_name: str):
        """读锁，允许多个读取者同时访问"""
        lock_name = f"{lock_name}:RL"
        with redis_lock.Lock(redis_t, lock_name, id=CacheUtils.LOCK_ID):
            try:
                yield
            finally:
                pass
                
    @staticmethod
    @contextlib.contextmanager
    def writer_lock(redis_t, lock_name):
        """写锁，独占访问"""
        lock_name = f"{lock_name}:WL"
        with redis_lock.Lock(redis_t, lock_name, id=CacheUtils.LOCK_ID):
            try:
                yield
            finally:
                pass
                
    @staticmethod
    def visit(path: Union[str, Path]):
        """更新缓存文件的访问时间"""
        path = Path(path)
        if path.exists():
            meta_path = path.with_suffix(path.suffix + ".meta")
            if meta_path.exists():
                meta = pickle.load(meta_path.open("rb"))
                meta["last_visit"] = time.time()
                pickle.dump(meta, meta_path.open("wb"))
```

## 6. 缓存更新机制

Qlib 提供了缓存更新机制，用于更新过期的缓存：

```python
# qlib/data/cache.py
class DiskExpressionCache(ExpressionCache):
    def update(self, cache_uri: Union[str, Path], freq: str = "day"):
        """更新表达式缓存"""
        cache_path = Path(cache_uri)
        if not cache_path.exists():
            return 2  # 更新失败
            
        # 读取元数据
        meta_path = cache_path.with_suffix(cache_path.suffix + ".meta")
        if not meta_path.exists():
            return 2  # 更新失败
            
        with meta_path.open("rb") as f:
            meta = pickle.load(f)
            
        # 获取最新日历
        from .data import Cal
        _calendar = Cal.calendar(freq=freq)
        latest_calendar_index = len(_calendar) - 1
        
        if meta["last_calendar_index"] >= latest_calendar_index:
            return 1  # 无需更新
            
        # 获取表达式和股票代码
        instrument = meta["instrument"]
        field = meta["field"]
        
        # 计算新数据
        series = self.provider.expression(
            instrument, field, _calendar[meta["last_calendar_index"] + 1], _calendar[-1], freq
        )
        
        if series.empty:
            return 1  # 无需更新
            
        # 更新缓存
        with CacheUtils.writer_lock(self.r, f"{str(C.dpm.get_data_uri(freq))}:expression-{cache_path.name}"):
            # 读取旧数据
            with cache_path.open("rb") as fp:
                old_data = np.fromfile(fp, dtype="<f")
                
            # 合并新旧数据
            new_data = np.hstack([old_data, series.values.astype("<f")])
            
            # 写入更新后的数据
            with cache_path.open("wb") as fp:
                new_data.tofile(fp)
                
            # 更新元数据
            meta["last_calendar_index"] = latest_calendar_index
            meta["last_update"] = time.time()
            with meta_path.open("wb") as f:
                pickle.dump(meta, f)
                
        return 0  # 更新成功
```

## 7. 缓存配置

Qlib 的缓存行为可以通过配置文件进行配置：

```python
# qlib/config.py
class Config:
    # 内存缓存配置
    mem_cache_size_limit = 500  # 内存缓存大小限制
    mem_cache_limit_type = "length"  # 内存缓存限制类型（length 或 sizeof）
    mem_cache_expire = 60 * 60  # 内存缓存过期时间（秒）
    
    # 磁盘缓存配置
    features_cache_dir_name = "features"  # 特征缓存目录名
    dataset_cache_dir_name = "dataset"  # 数据集缓存目录名
    
    # 并行配置
    kernels = 16  # 并行处理的核心数
    maxtasksperchild = 1  # 每个进程处理的最大任务数
    joblib_backend = "loky"  # joblib 后端
    
    # Redis 配置
    redis_host = "127.0.0.1"  # Redis 主机
    redis_port = 6379  # Redis 端口
    redis_task_db = 1  # Redis 任务数据库
```

## 8. 缓存使用流程

### 8.1 表达式缓存使用流程

1. 计算表达式的哈希值作为缓存键
2. 检查缓存是否存在
3. 如果存在，直接读取缓存
4. 如果不存在，计算表达式结果并写入缓存
5. 返回表达式结果

```python
# 使用表达式缓存的示例
def expression(self, instrument, field, start_time=None, end_time=None, freq="day"):
    try:
        # 尝试使用缓存
        return self._expression(instrument, field, start_time, end_time, freq)
    except NotImplementedError:
        # 缓存不可用，直接从提供者获取
        return self.provider.expression(instrument, field, start_time, end_time, freq)
```

### 8.2 数据集缓存使用流程

1. 计算数据集参数的哈希值作为缓存键
2. 检查缓存是否存在
3. 如果存在，使用 `IndexManager` 确定读取范围，然后从 HDF5 文件读取数据
4. 如果不存在，从原始数据源获取数据，处理后写入缓存
5. 返回数据集

```python
# 使用数据集缓存的示例
def dataset(self, instruments, fields, start_time=None, end_time=None, freq="day", disk_cache=1, inst_processors=[]):
    if disk_cache == 0:
        # 跳过缓存
        return self.provider.dataset(instruments, fields, start_time, end_time, freq, inst_processors=inst_processors)
    else:
        # 使用缓存
        try:
            return self._dataset(instruments, fields, start_time, end_time, freq, disk_cache, inst_processors=inst_processors)
        except NotImplementedError:
            return self.provider.dataset(instruments, fields, start_time, end_time, freq, inst_processors=inst_processors)
```

## 9. 缓存性能优化

Qlib 的缓存系统采用了多种性能优化技术：

### 9.1 二进制存储

表达式结果使用二进制格式存储，减少磁盘空间占用，提高读写效率：

```python
# qlib/utils/__init__.py
def read_bin(file_path: Union[str, Path], start_index, end_index):
    """读取二进制文件"""
    file_path = Path(file_path.expanduser().resolve())
    with file_path.open("rb") as f:
        # 读取起始索引
        ref_start_index = int(np.frombuffer(f.read(4), dtype="<f")[0])
        si = max(ref_start_index, start_index)
        if si > end_index:
            return pd.Series(dtype=np.float32)
        # 计算偏移量
        f.seek(4 * (si - ref_start_index) + 4)
        # 读取数据
        count = end_index - si + 1
        data = np.frombuffer(f.read(4 * count), dtype="<f")
        series = pd.Series(data, index=pd.RangeIndex(si, si + len(data)))
    return series
```

### 9.2 HDF5 格式

数据集使用 HDF5 格式存储，支持高效的部分读取：

```python
# qlib/data/cache.py
def read_data_from_cache(cls, cache_path: Union[str, Path], start_time, end_time, fields):
    """从缓存读取数据"""
    # ...
    with pd.HDFStore(cache_path, mode="r") as store:
        if "/{}".format(im.KEY) in store.keys():
            # 只读取需要的部分
            df = store.select(key=im.KEY, start=start, stop=stop)
            # ...
    return df
```

### 9.3 索引管理

使用索引管理器快速定位数据位置，避免全量读取：

```python
# qlib/data/cache.py
def get_index(self, start_time=None, end_time=None):
    """获取指定时间范围的索引"""
    # ...
    # 过滤索引
    mask = np.ones(len(self.index), dtype=bool)
    if start_time is not None:
        mask &= self.index["end_time"] >= pd.Timestamp(start_time)
    if end_time is not None:
        mask &= self.index["start_time"] <= pd.Timestamp(end_time)
    return self.index[mask]
```

### 9.4 LRU 策略

内存缓存使用 LRU 策略，优先保留最近使用的数据：

```python
# qlib/data/cache.py
def __getitem__(self, key):
    # 获取值并更新访问顺序
    value = self.od[key]
    self.od.move_to_end(key)
    return value
```

### 9.5 并行处理

支持多进程并行读取缓存，提高读取效率：

```python
# qlib/data/data.py
def dataset_processor(instruments_d, column_names, start_time, end_time, freq, inst_processors=[]):
    # ...
    # 并行执行任务
    data = dict(
        zip(
            inst_l,
            ParallelExt(n_jobs=workers, backend=C.joblib_backend, maxtasksperchild=C.maxtasksperchild)(task_l),
        )
    )
    # ...
```

### 9.6 读写分离

使用读锁和写锁分离读写操作，提高并发性能：

```python
# qlib/data/cache.py
@staticmethod
@contextlib.contextmanager
def reader_lock(redis_t, lock_name: str):
    """读锁，允许多个读取者同时访问"""
    # ...
    
@staticmethod
@contextlib.contextmanager
def writer_lock(redis_t, lock_name):
    """写锁，独占访问"""
    # ...
```

## 10. 总结

Qlib 的缓存实现机制是一个多层次、高性能的缓存系统，包括内存缓存和磁盘缓存两个主要层次。它通过以下关键技术提高数据加载和处理效率：

1. **多级缓存**：内存缓存用于频繁访问的数据，磁盘缓存用于持久化存储
2. **LRU 策略**：自动淘汰最少使用的缓存项，优化内存使用
3. **二进制存储**：使用高效的二进制格式存储数据，提高读写效率
4. **部分读取**：支持只读取需要的时间段数据，避免全量读取
5. **并发控制**：使用分布式锁机制处理并发访问，确保数据一致性
6. **增量更新**：支持缓存的增量更新，避免全量重建

这种设计使得 Qlib 能够高效地处理大量股票数据，同时保持代码的可维护性和扩展性。用户可以根据自己的需求配置缓存行为，或者实现自定义的缓存机制。 