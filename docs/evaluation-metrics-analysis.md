# GraphRAG 评价指标构建分析

## 一、指标体系概览

GraphRAG 构建了多层次的评价指标体系，涵盖从图结构到查询质量的各个维度：

```
┌─────────────────────────────────────────────────────────────────┐
│                     GraphRAG 评价指标体系                        │
├─────────────────────────────────────────────────────────────────┤
│  图结构指标          查询质量指标          性能监控指标           │
│  ├─ 节点度数         ├─ 相关性评分         ├─ LLM调用次数         │
│  ├─ 边权重           ├─ 重要性评分         ├─ Token消耗           │
│  ├─ 模块度           ├─ 社区评级           ├─ 响应时间            │
│  └─ PMI/RRF权重      └─ 答案得分           └─ 向量相似度          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、节点级指标 (Node-Level Metrics)

### 1. 节点度数 (Node Degree)

**文件位置**: `graphrag/index/operations/compute_degree.py:10-15`

度数表示节点在图中的连接数量，用于衡量实体的重要性：

```python
def compute_degree(graph: nx.Graph) -> pd.DataFrame:
    """Create a new DataFrame with the degree of each node in the graph."""
    return pd.DataFrame([
        {"title": node, "degree": int(degree)}
        for node, degree in graph.degree
    ])
```

**用途**:
- 作为实体排名的基础指标
- 度数越高的实体通常在知识图谱中越重要
- 用于 `finalize_entities` 步骤中的 `rank` 字段

### 2. 节点频率 (Node Frequency)

**定义**: 节点在多少个文本单元(text_units)中出现

```python
# 在实体合并时统计
frequency=("source_id", "count")  # 统计出现频次
```

**用途**:
- 衡量实体在语料中的普遍性
- 用于图剪枝时的过滤阈值

---

## 三、边级指标 (Edge-Level Metrics)

### 1. 边权重 (Edge Weight)

**来源**: LLM 抽取时的 `relationship_strength` 字段

```python
# 从 LLM 输出解析
weight = float(record_attributes[-1])  # 关系强度

# 跨文档合并时累加
weight=("weight", "sum")
```

### 2. 组合度数 (Combined Degree)

**文件位置**: `graphrag/index/operations/compute_edge_combined_degree.py:11-39`

组合度数 = 源节点度数 + 目标节点度数

```python
def compute_edge_combined_degree(
    edge_df: pd.DataFrame,
    node_degree_df: pd.DataFrame,
    ...
) -> pd.Series:
    """Compute the combined degree for each edge in a graph."""
    output_df["combined_degree"] = (
        output_df[_degree_colname(edge_source_column)]
        + output_df[_degree_colname(edge_target_column)]
    )
    return output_df["combined_degree"]
```

**用途**: 衡量关系的重要性，连接重要节点的边更重要

### 3. PMI 权重 (Pointwise Mutual Information)

**文件位置**: `graphrag/index/utils/graphs.py:155-201`

PMI 用于衡量两个实体共现的统计显著性：

```python
def calculate_pmi_edge_weights(nodes_df, edges_df, ...):
    """
    PMI公式: pmi(x,y) = p(x,y) * log2(p(x,y) / (p(x)*p(y)))

    其中:
    - p(x,y) = edge_weight(x,y) / total_edge_weights  # 边的联合概率
    - p(x) = freq_occurrence(x) / total_freq_occurrences  # 节点的边际概率
    """
    total_edge_weights = edges_df[edge_weight_col].sum()
    total_freq_occurrences = nodes_df[node_freq_col].sum()

    # 计算各节点的出现概率
    copied_nodes_df["prop_occurrence"] = copied_nodes_df[node_freq_col] / total_freq_occurrences

    # 计算边的联合概率
    edges_df["prop_weight"] = edges_df[edge_weight_col] / total_edge_weights

    # 计算 PMI
    edges_df[edge_weight_col] = edges_df["prop_weight"] * np.log2(
        edges_df["prop_weight"] / (edges_df["source_prop"] * edges_df["target_prop"])
    )
```

**特点**: 能够消除高频节点的偏差，突出真正有意义的共现关系

### 4. RRF 权重 (Reciprocal Rank Fusion)

**文件位置**: `graphrag/index/utils/graphs.py:204-235`

RRF 融合 PMI 排名和原始权重排名：

```python
def calculate_rrf_edge_weights(nodes_df, edges_df, rrf_smoothing_factor=60):
    """
    RRF公式: weight = 1/(k + pmi_rank) + 1/(k + raw_weight_rank)

    其中 k 是平滑因子（默认60）
    """
    edges_df = calculate_pmi_edge_weights(...)

    edges_df["pmi_rank"] = edges_df[edge_weight_col].rank(method="min", ascending=False)
    edges_df["raw_weight_rank"] = edges_df[edge_weight_col].rank(method="min", ascending=False)

    edges_df[edge_weight_col] = edges_df.apply(
        lambda x: (1 / (rrf_smoothing_factor + x["pmi_rank"]))
                + (1 / (rrf_smoothing_factor + x["raw_weight_rank"])),
        axis=1,
    )
```

**用途**: 综合多个排名信号，提供更鲁棒的边权重

---

## 四、图质量指标 (Graph Quality Metrics)

### 1. 模块度 (Modularity)

**文件位置**: `graphrag/index/utils/graphs.py:20-152`

模块度衡量图的社区结构质量，值越高表示社区划分越好：

```python
from graspologic.partition import hierarchical_leiden, modularity

def calculate_root_modularity(graph, max_cluster_size=10, random_seed=0xDEADBEEF):
    """计算根层级社区的模块度"""
    hcs = hierarchical_leiden(graph, max_cluster_size=max_cluster_size, random_seed=random_seed)
    root_clusters = hcs.first_level_hierarchical_clustering()
    return modularity(graph, root_clusters)

def calculate_leaf_modularity(graph, max_cluster_size=10, random_seed=0xDEADBEEF):
    """计算叶层级社区的模块度"""
    hcs = hierarchical_leiden(graph, max_cluster_size=max_cluster_size, random_seed=random_seed)
    leaf_clusters = hcs.final_level_hierarchical_clustering()
    return modularity(graph, leaf_clusters)
```

### 2. 模块度类型

**文件位置**: `graphrag/index/utils/graphs.py:117-152`

支持三种模块度计算方式：

```python
def calculate_modularity(graph, modularity_metric: ModularityMetric):
    match modularity_metric:
        case ModularityMetric.Graph:
            # 整图模块度
            return calculate_graph_modularity(graph, ...)

        case ModularityMetric.LCC:
            # 最大连通分量模块度
            return calculate_lcc_modularity(graph, ...)

        case ModularityMetric.WeightedComponents:
            # 加权组件模块度
            return calculate_weighted_modularity(graph, ...)
```

### 3. 加权模块度

**文件位置**: `graphrag/index/utils/graphs.py:79-114`

对多个连通分量加权求和：

```python
def calculate_weighted_modularity(graph, min_connected_component_size=10, ...):
    """
    加权模块度 = Σ(component_modularity × component_size) / total_nodes
    """
    connected_components = list(nx.connected_components(graph))
    filtered_components = [c for c in connected_components if len(c) > min_connected_component_size]

    total_nodes = sum(len(c) for c in filtered_components)
    total_modularity = 0

    for component in filtered_components:
        subgraph = graph.subgraph(component)
        modularity = calculate_root_modularity(subgraph, ...)
        total_modularity += modularity * len(component) / total_nodes

    return total_modularity
```

---

## 五、图剪枝阈值 (Graph Pruning Thresholds)

**文件位置**: `graphrag/index/operations/prune_graph.py:18-83`

使用多种统计阈值来过滤低质量节点和边：

```python
def prune_graph(
    graph: nx.Graph,
    min_node_freq: int = 1,           # 最小节点频率
    max_node_freq_std: float = None,   # 最大频率标准差倍数
    min_node_degree: int = 1,          # 最小节点度数
    max_node_degree_std: float = None, # 最大度数标准差倍数
    min_edge_weight_pct: float = 40,   # 最小边权重百分位数
    remove_ego_nodes: bool = False,    # 是否移除中心节点
    lcc_only: bool = False,            # 是否只保留最大连通分量
):
    # 1. 移除度数过低的节点
    graph.remove_nodes_from([
        node for node, degree in degrees if degree < min_node_degree
    ])

    # 2. 基于标准差移除度数过高的节点
    if max_node_degree_std is not None:
        upper_threshold = mean + std_trim * std
        graph.remove_nodes_from([
            node for node, degree in degrees if degree > upper_threshold
        ])

    # 3. 移除频率过低的节点
    graph.remove_nodes_from([
        node for node, data in graph.nodes(data=True)
        if data[schemas.NODE_FREQUENCY] < min_node_freq
    ])

    # 4. 基于百分位数移除低权重边
    if min_edge_weight_pct > 0:
        min_edge_weight = np.percentile(edge_weights, min_edge_weight_pct)
        graph.remove_edges_from([
            (s, t) for s, t, data in graph.edges(data=True)
            if data[schemas.EDGE_WEIGHT] < min_edge_weight
        ])
```

**标准差阈值计算**:

```python
def _get_upper_threshold_by_std(data, std_trim):
    """上界阈值 = 均值 + std_trim × 标准差"""
    mean = np.mean(data)
    std = np.std(data)
    return mean + std_trim * std
```

---

## 六、社区报告评级 (Community Report Rating)

**文件位置**: `graphrag/index/operations/summarize_communities/community_reports_extractor.py:23-39`

LLM 生成的社区报告包含结构化评级：

```python
class CommunityReportResponse(BaseModel):
    """社区报告响应结构"""
    title: str                          # 报告标题
    summary: str                        # 报告摘要
    findings: list[FindingModel]        # 发现列表
    rating: float                       # 评级 (0-10)
    rating_explanation: str             # 评级解释

class FindingModel(BaseModel):
    """发现结构"""
    summary: str       # 发现摘要
    explanation: str   # 发现解释
```

**评级用途**:
- 衡量社区的重要性和相关性
- 用于查询时的社区筛选和排序

---

## 七、查询相关性评分 (Query Relevancy Rating)

**文件位置**: `graphrag/query/context_builder/rate_relevancy.py:21-77`

使用 LLM 评估查询与社区的相关性：

```python
async def rate_relevancy(
    query: str,
    description: str,
    model: ChatModel,
    num_repeats: int = 1,    # 重复评分次数
    ...
) -> dict[str, Any]:
    """
    评估查询与描述的相关性 (0-5 分制)

    Returns:
        rating: 最终评分（投票选出）
        ratings: 所有评分列表
        llm_calls: LLM调用次数
        prompt_tokens: 提示词token数
        output_tokens: 输出token数
    """
    ratings = []
    for _ in range(num_repeats):
        model_response = await model.achat(
            prompt=query,
            history=messages,
            json=True
        )
        parsed_response = try_parse_json_object(response)
        ratings.append(parsed_response["rating"])

    # 投票选择最终评分
    options, counts = np.unique(ratings, return_counts=True)
    rating = int(options[np.argmax(counts)])

    return {
        "rating": rating,
        "ratings": ratings,
        "llm_calls": llm_calls,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
    }
```

---

## 八、全局搜索评分 (Global Search Scoring)

**文件位置**: `graphrag/query/structured_search/global_search/search.py:266-294`

Map-Reduce 架构中的答案评分：

### Map 阶段评分

```python
def _parse_search_response(self, search_response: str) -> list[dict[str, Any]]:
    """解析搜索响应，提取关键点和评分"""
    parsed_elements = json.loads(search_response).get("points")

    return [
        {
            "answer": element["description"],  # 答案内容
            "score": int(element["score"]),    # 重要性评分 (整数)
        }
        for element in parsed_elements
        if "description" in element and "score" in element
    ]
```

### Reduce 阶段过滤和排序

**文件位置**: `graphrag/query/structured_search/global_search/search.py:296-370`

```python
async def _reduce_response(self, map_responses, query, ...):
    # 1. 收集所有关键点
    key_points = []
    for index, response in enumerate(map_responses):
        for element in response.response:
            key_points.append({
                "analyst": index,
                "answer": element["answer"],
                "score": element["score"],
            })

    # 2. 过滤评分为0的响应（无关信息）
    filtered_key_points = [
        point for point in key_points
        if point["score"] > 0
    ]

    # 3. 按评分降序排序
    filtered_key_points = sorted(
        filtered_key_points,
        key=lambda x: x["score"],
        reverse=True,
    )

    # 4. 按token预算截断
    data = []
    total_tokens = 0
    for point in filtered_key_points:
        formatted_text = f"Importance Score: {point['score']}\n{point['answer']}"
        if total_tokens + len(tokenizer.encode(formatted_text)) > self.max_data_tokens:
            break
        data.append(formatted_text)
        total_tokens += len(tokenizer.encode(formatted_text))
```

---

## 九、性能监控指标 (Performance Metrics)

**文件位置**: `graphrag/query/structured_search/base.py`

SearchResult 数据类追踪多维度性能指标：

```python
@dataclass
class SearchResult:
    response: str | list[dict[str, Any]]
    context_data: str | list[pd.DataFrame] | dict[str, pd.DataFrame]
    context_text: str | list[str] | dict[str, str]

    # 性能指标
    completion_time: float = 0            # 执行时间（秒）
    llm_calls: int = 0                    # LLM调用总次数
    prompt_tokens: int = 0                # 提示词token总数
    output_tokens: int = 0                # 输出token总数

    # 分类统计
    llm_calls_categories: dict = {}       # 按操作类型的调用次数
    prompt_tokens_categories: dict = {}    # 按操作类型的提示词token
    output_tokens_categories: dict = {}    # 按操作类型的输出token
```

**分类示例**:

```python
llm_calls = {
    "build_context": context_result.llm_calls,  # 构建上下文
    "map": sum(r.llm_calls for r in map_responses),  # Map阶段
    "reduce": reduce_response.llm_calls,  # Reduce阶段
}
```

---

## 十、向量相似度指标 (Vector Similarity)

**用于实体检索的语义相似度搜索**:

```python
# 基于嵌入向量的相似度搜索
def similarity_search_by_vector(query_embedding, k=10):
    """返回与查询向量最相似的k个文档"""
    ...

def similarity_search_by_text(query_text, k=10):
    """将文本转为向量后搜索"""
    query_embedding = embed(query_text)
    return similarity_search_by_vector(query_embedding, k)
```

**支持的向量存储后端**:
- LanceDB
- Azure AI Search
- CosmosDB (余弦相似度)

---

## 十一、指标汇总表

| 指标类别 | 指标名称 | 计算方式 | 取值范围 | 用途 |
|---------|---------|---------|---------|------|
| **节点** | degree | 连接边数量 | ≥0 整数 | 实体重要性排名 |
| **节点** | frequency | 出现的文档数 | ≥1 整数 | 实体普遍性 |
| **边** | weight | LLM评分累加 | >0 浮点数 | 关系强度 |
| **边** | combined_degree | 源度+目标度 | ≥0 整数 | 关系重要性 |
| **边** | PMI | 点互信息 | 可正可负 | 共现显著性 |
| **边** | RRF | 排名倒数融合 | >0 浮点数 | 综合权重 |
| **图** | modularity | Leiden聚类质量 | [-0.5, 1] | 社区结构质量 |
| **社区** | rating | LLM评级 | 0-10 | 社区重要性 |
| **查询** | relevancy | LLM相关性评分 | 0-5 | 社区筛选 |
| **查询** | importance_score | LLM重要性评分 | 0-10 整数 | 答案排序 |
| **性能** | completion_time | 执行时间 | >0 秒 | 性能监控 |
| **性能** | llm_calls | API调用次数 | ≥0 整数 | 成本监控 |
| **性能** | tokens | token消耗 | ≥0 整数 | 成本监控 |

---

## 十二、总结

GraphRAG 的评价指标体系具有以下特点：

1. **多层次**: 从节点、边、社区到查询响应，全链路指标覆盖
2. **统计驱动**: 使用度数、频率、PMI等统计方法量化结构质量
3. **LLM增强**: 社区评级、相关性评分等由 LLM 生成
4. **可配置**: 阈值参数可调，支持不同场景需求
5. **性能感知**: 完整的 token 和调用次数统计，便于成本优化

这套指标体系为知识图谱的质量评估和查询优化提供了坚实基础。
