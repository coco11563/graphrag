# GraphRAG 知识图谱构建与实体抽取分析

## 一、整体架构流程

```
文本文档 → 文本分块(text_units) → LLM实体抽取 → 图谱合并 → 描述摘要 → 社区检测 → 最终知识图谱
```

### 核心文件结构

| 文件路径 | 功能描述 |
|---------|---------|
| `graphrag/index/operations/extract_graph/graph_extractor.py` | 核心抽取器，处理 LLM 输出并构建图 |
| `graphrag/index/operations/extract_graph/extract_graph.py` | 抽取编排，合并多文档结果 |
| `graphrag/index/operations/extract_graph/graph_intelligence_strategy.py` | LLM 策略实现 |
| `graphrag/prompts/index/extract_graph.py` | 实体抽取 Prompt 模板 |
| `graphrag/index/workflows/extract_graph.py` | 工作流编排 |
| `graphrag/index/operations/cluster_graph.py` | 社区检测 |
| `graphrag/index/operations/create_graph.py` | 最终图结构创建 |

---

## 二、实体抽取核心机制

### 1. 基于 LLM 的 Prompt 抽取

**文件位置**: `graphrag/prompts/index/extract_graph.py:6-126`

GraphRAG 使用 **few-shot prompting** 让 LLM 从文本中抽取实体和关系：

#### Prompt 结构

```
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types,
identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
   - entity_name: Name of the entity, capitalized
   - entity_type: One of the following types: [{entity_types}]
   - entity_description: Comprehensive description of the entity's attributes and activities
   Format: ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity)
   that are *clearly related* to each other.
   - source_entity: name of the source entity
   - target_entity: name of the target entity
   - relationship_description: explanation as to why the entities are related
   - relationship_strength: a numeric score indicating strength of the relationship
   Format: ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. Return output as a single list using **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}
```

#### 默认分隔符

**文件位置**: `graphrag/index/operations/extract_graph/graph_extractor.py:25-28`

```python
DEFAULT_TUPLE_DELIMITER = "<|>"      # 元组内字段分隔符
DEFAULT_RECORD_DELIMITER = "##"       # 记录间分隔符
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"  # 完成标记
DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event"]  # 默认实体类型
```

#### LLM 输出示例

```
("entity"<|>CENTRAL INSTITUTION<|>ORGANIZATION<|>The Central Institution is the Federal Reserve of Verdantis)
##
("entity"<|>MARTIN SMITH<|>PERSON<|>Martin Smith is the chair of the Central Institution)
##
("relationship"<|>MARTIN SMITH<|>CENTRAL INSTITUTION<|>Martin Smith is the Chair of the Central Institution<|>9)
<|COMPLETE|>
```

### 2. Gleaning 机制（迭代抽取）

**文件位置**: `graphrag/index/operations/extract_graph/graph_extractor.py:143-177`

Gleaning 是一种**迭代抽取机制**，用于提高实体和关系的召回率：

```python
async def _process_document(self, text: str, prompt_variables: dict[str, str]) -> str:
    # 第一次抽取
    response = await self._model.achat(
        self._extraction_prompt.format(**{**prompt_variables, self._input_text_key: text})
    )
    results = response.output.content or ""

    # Gleaning 迭代抽取
    if self._max_gleanings > 0:
        for i in range(self._max_gleanings):
            # 继续抽取提示
            response = await self._model.achat(
                CONTINUE_PROMPT,  # "MANY entities and relationships were missed..."
                history=response.history,
            )
            results += response.output.content or ""

            if i >= self._max_gleanings - 1:
                break

            # 检查是否需要继续
            response = await self._model.achat(
                LOOP_PROMPT,  # "Answer Y if there are still entities..."
                history=response.history,
            )
            if response.output.content != "Y":
                break

    return results
```

**相关 Prompt**:

```python
CONTINUE_PROMPT = "MANY entities and relationships were missed in the last extraction. Remember to ONLY emit entities that match any of the previously extracted types. Add them below using the same format:\n"

LOOP_PROMPT = "It appears some entities and relationships may have still been missed. Answer Y if there are still entities or relationships that need to be added, or N if there are none. Please answer with a single letter Y or N.\n"
```

---

## 三、图谱构建详细流程

### 1. 解析 LLM 输出并构建 NetworkX 图

**文件位置**: `graphrag/index/operations/extract_graph/graph_extractor.py:179-290`

```python
async def _process_results(
    self,
    results: dict[int, str],
    tuple_delimiter: str,
    record_delimiter: str,
) -> nx.Graph:
    """Parse the result string to create an undirected unipartite graph."""
    graph = nx.Graph()

    for source_doc_id, extracted_data in results.items():
        # 按记录分隔符分割
        records = [r.strip() for r in extracted_data.split(record_delimiter)]

        for record in records:
            record = re.sub(r"^\(|\)$", "", record.strip())
            record_attributes = record.split(tuple_delimiter)

            # 解析实体
            if record_attributes[0] == '"entity"' and len(record_attributes) >= 4:
                entity_name = clean_str(record_attributes[1].upper())
                entity_type = clean_str(record_attributes[2].upper())
                entity_description = clean_str(record_attributes[3])

                if entity_name in graph.nodes():
                    # 合并已存在实体的描述
                    node = graph.nodes[entity_name]
                    if self._join_descriptions:
                        node["description"] = "\n".join(
                            list({*_unpack_descriptions(node), entity_description})
                        )
                    node["source_id"] = ", ".join(
                        list({*_unpack_source_ids(node), str(source_doc_id)})
                    )
                else:
                    graph.add_node(
                        entity_name,
                        type=entity_type,
                        description=entity_description,
                        source_id=str(source_doc_id),
                    )

            # 解析关系
            if record_attributes[0] == '"relationship"' and len(record_attributes) >= 5:
                source = clean_str(record_attributes[1].upper())
                target = clean_str(record_attributes[2].upper())
                edge_description = clean_str(record_attributes[3])
                try:
                    weight = float(record_attributes[-1])
                except ValueError:
                    weight = 1.0

                # 确保节点存在
                if source not in graph.nodes():
                    graph.add_node(source, type="", description="", source_id=str(source_doc_id))
                if target not in graph.nodes():
                    graph.add_node(target, type="", description="", source_id=str(source_doc_id))

                # 合并边的权重和描述
                if graph.has_edge(source, target):
                    edge_data = graph.get_edge_data(source, target)
                    if edge_data is not None:
                        weight += edge_data["weight"]
                        if self._join_descriptions:
                            edge_description = "\n".join(
                                list({*_unpack_descriptions(edge_data), edge_description})
                            )

                graph.add_edge(source, target, weight=weight, description=edge_description, source_id=str(source_doc_id))

    return graph
```

### 2. 跨文档实体和关系合并

**文件位置**: `graphrag/index/operations/extract_graph/extract_graph.py:100-123`

当同一实体出现在多个文档中时，执行合并操作：

```python
def _merge_entities(entity_dfs) -> pd.DataFrame:
    """合并来自多个文档的实体"""
    all_entities = pd.concat(entity_dfs, ignore_index=True)
    return (
        all_entities.groupby(["title", "type"], sort=False)
        .agg(
            description=("description", list),      # 合并描述为列表
            text_unit_ids=("source_id", list),      # 记录来源文档
            frequency=("source_id", "count"),       # 统计出现频次
        )
        .reset_index()
    )


def _merge_relationships(relationship_dfs) -> pd.DataFrame:
    """合并来自多个文档的关系"""
    all_relationships = pd.concat(relationship_dfs, ignore_index=False)
    return (
        all_relationships.groupby(["source", "target"], sort=False)
        .agg(
            description=("description", list),      # 合并描述为列表
            text_unit_ids=("source_id", list),      # 记录来源文档
            weight=("weight", "sum"),               # 权重累加
        )
        .reset_index()
    )
```

### 3. 策略执行流程

**文件位置**: `graphrag/index/operations/extract_graph/graph_intelligence_strategy.py:26-102`

```python
async def run_graph_intelligence(
    docs: list[Document],
    entity_types: EntityTypes,
    cache: PipelineCache,
    args: StrategyConfig,
) -> EntityExtractionResult:
    """Run the graph intelligence entity extraction strategy."""
    llm_config = LanguageModelConfig(**args["llm"])

    llm = ModelManager().get_or_create_chat_model(
        name="extract_graph",
        model_type=llm_config.type,
        config=llm_config,
        cache=cache,
    )

    return await run_extract_graph(llm, docs, entity_types, args)


async def run_extract_graph(
    model: ChatModel,
    docs: list[Document],
    entity_types: EntityTypes,
    args: StrategyConfig,
) -> EntityExtractionResult:
    """Run the entity extraction chain."""
    extractor = GraphExtractor(
        model_invoker=model,
        prompt=args.get("extraction_prompt", None),
        max_gleanings=args.get("max_gleanings", 1),
    )

    text_list = [doc.text.strip() for doc in docs]
    results = await extractor(list(text_list), {"entity_types": entity_types, ...})

    graph = results.output

    # 转换为实体列表和关系 DataFrame
    entities = [{"title": item[0], **(item[1] or {})} for item in graph.nodes(data=True)]
    relationships = nx.to_pandas_edgelist(graph)

    return EntityExtractionResult(entities, relationships, graph)
```

---

## 四、后处理阶段

### 1. 描述摘要

**文件位置**: `graphrag/index/workflows/extract_graph.py:135-159`

使用 LLM 将多个描述片段合并为统一描述：

```python
async def get_summarized_entities_relationships(
    extracted_entities: pd.DataFrame,
    extracted_relationships: pd.DataFrame,
    callbacks: WorkflowCallbacks,
    cache: PipelineCache,
    summarization_strategy: dict[str, Any] | None = None,
    summarization_num_threads: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize the entities and relationships."""
    entity_summaries, relationship_summaries = await summarize_descriptions(
        entities_df=extracted_entities,
        relationships_df=extracted_relationships,
        callbacks=callbacks,
        cache=cache,
        strategy=summarization_strategy,
        num_threads=summarization_num_threads,
    )

    # 合并摘要到原始数据
    relationships = extracted_relationships.drop(columns=["description"]).merge(
        relationship_summaries, on=["source", "target"], how="left"
    )
    entities = extracted_entities.drop(columns=["description"]).merge(
        entity_summaries, on="title", how="left"
    )

    return entities, relationships
```

### 2. 社区检测（Hierarchical Leiden）

**文件位置**: `graphrag/index/operations/cluster_graph.py:19-80`

使用 **graspologic** 库的 Hierarchical Leiden 算法进行层次化社区检测：

```python
from graspologic.partition import hierarchical_leiden

def cluster_graph(
    graph: nx.Graph,
    max_cluster_size: int,
    use_lcc: bool,
    seed: int | None = None,
) -> Communities:
    """Apply a hierarchical clustering algorithm to a graph."""
    if len(graph.nodes) == 0:
        return []

    # 计算 Leiden 社区
    node_id_to_community_map, parent_mapping = _compute_leiden_communities(
        graph=graph,
        max_cluster_size=max_cluster_size,
        use_lcc=use_lcc,
        seed=seed,
    )

    # 构建层次化社区结构
    levels = sorted(node_id_to_community_map.keys())
    clusters: dict[int, dict[int, list[str]]] = {}

    for level in levels:
        result = {}
        clusters[level] = result
        for node_id, raw_community_id in node_id_to_community_map[level].items():
            if raw_community_id not in result:
                result[raw_community_id] = []
            result[raw_community_id].append(node_id)

    # 返回 (level, cluster_id, parent_id, nodes) 列表
    results: Communities = []
    for level in clusters:
        for cluster_id, nodes in clusters[level].items():
            results.append((level, cluster_id, parent_mapping[cluster_id], nodes))
    return results


def _compute_leiden_communities(
    graph: nx.Graph | nx.DiGraph,
    max_cluster_size: int,
    use_lcc: bool,
    seed: int | None = None,
) -> tuple[dict[int, dict[str, int]], dict[int, int]]:
    """Return Leiden root communities and their hierarchy mapping."""
    if use_lcc:
        graph = stable_largest_connected_component(graph)

    community_mapping = hierarchical_leiden(
        graph, max_cluster_size=max_cluster_size, random_seed=seed
    )

    results: dict[int, dict[str, int]] = {}
    hierarchy: dict[int, int] = {}

    for partition in community_mapping:
        results[partition.level] = results.get(partition.level, {})
        results[partition.level][partition.node] = partition.cluster
        hierarchy[partition.cluster] = (
            partition.parent_cluster if partition.parent_cluster is not None else -1
        )

    return results, hierarchy
```

### 3. 创建最终图结构

**文件位置**: `graphrag/index/operations/create_graph.py:10-23`

```python
def create_graph(
    edges: pd.DataFrame,
    edge_attr: list[str | int] | None = None,
    nodes: pd.DataFrame | None = None,
    node_id: str = "title",
) -> nx.Graph:
    """Create a networkx graph from nodes and edges dataframes."""
    graph = nx.from_pandas_edgelist(edges, edge_attr=edge_attr)

    if nodes is not None:
        nodes.set_index(node_id, inplace=True)
        graph.add_nodes_from((n, dict(d)) for n, d in nodes.iterrows())

    return graph
```

---

## 五、工作流编排

**文件位置**: `graphrag/index/workflows/extract_graph.py:28-79`

```python
async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """All the steps to create the base entity graph."""
    # 1. 加载文本单元
    text_units = await load_table_from_storage("text_units", context.output_storage)

    # 2. 配置抽取策略
    extract_graph_llm_settings = config.get_language_model_config(config.extract_graph.model_id)
    extraction_strategy = config.extract_graph.resolved_strategy(config.root_dir, extract_graph_llm_settings)

    # 3. 配置摘要策略
    summarization_llm_settings = config.get_language_model_config(config.summarize_descriptions.model_id)
    summarization_strategy = config.summarize_descriptions.resolved_strategy(config.root_dir, summarization_llm_settings)

    # 4. 执行抽取
    entities, relationships, raw_entities, raw_relationships = await extract_graph(
        text_units=text_units,
        callbacks=context.callbacks,
        cache=context.cache,
        extraction_strategy=extraction_strategy,
        extraction_num_threads=extract_graph_llm_settings.concurrent_requests,
        entity_types=config.extract_graph.entity_types,
        summarization_strategy=summarization_strategy,
        summarization_num_threads=summarization_llm_settings.concurrent_requests,
    )

    # 5. 保存结果
    await write_table_to_storage(entities, "entities", context.output_storage)
    await write_table_to_storage(relationships, "relationships", context.output_storage)

    return WorkflowFunctionOutput(result={"entities": entities, "relationships": relationships})
```

---

## 六、关键数据结构

### 实体 (Entity)

| 字段 | 类型 | 描述 |
|------|------|------|
| `title` | string | 实体名称（大写） |
| `type` | string | 实体类型 (organization, person, geo, event) |
| `description` | string/list | 实体描述（合并后为列表） |
| `text_unit_ids` | list | 来源文档 ID 列表 |
| `frequency` | int | 出现频次 |
| `source_id` | string | 原始来源标识 |

### 关系 (Relationship)

| 字段 | 类型 | 描述 |
|------|------|------|
| `source` | string | 源实体名称 |
| `target` | string | 目标实体名称 |
| `description` | string/list | 关系描述 |
| `weight` | float | 关系强度（累加） |
| `text_unit_ids` | list | 来源文档 ID 列表 |
| `source_id` | string | 原始来源标识 |

### 社区 (Community)

| 字段 | 类型 | 描述 |
|------|------|------|
| `level` | int | 层次级别 |
| `cluster_id` | int | 社区 ID |
| `parent_cluster` | int | 父社区 ID (-1 表示根) |
| `nodes` | list[str] | 社区内实体列表 |

---

## 七、策略模式

**文件位置**: `graphrag/index/operations/extract_graph/extract_graph.py:85-97`

系统支持可扩展的抽取策略：

```python
def _load_strategy(strategy_type: ExtractEntityStrategyType) -> EntityExtractStrategy:
    """Load strategy method definition."""
    match strategy_type:
        case ExtractEntityStrategyType.graph_intelligence:
            # LLM-based 抽取（主要方式）
            from graphrag.index.operations.extract_graph.graph_intelligence_strategy import (
                run_graph_intelligence,
            )
            return run_graph_intelligence

        case _:
            msg = f"Unknown strategy: {strategy_type}"
            raise ValueError(msg)
```

此外还有 NLP-based 方式（使用名词短语抽取）在 `graphrag/index/operations/build_noun_graph/` 目录下实现。

---

## 八、总结

GraphRAG 的知识图谱构建采用 **LLM-first** 的设计理念：

1. **LLM 驱动抽取**：使用精心设计的 few-shot prompt 让 LLM 识别实体和关系
2. **Gleaning 迭代**：通过多轮对话提高实体召回率，避免遗漏重要信息
3. **结构化解析**：将 LLM 输出按固定分隔符解析为 NetworkX 图结构
4. **跨文档合并**：聚合来自不同文档的同一实体信息，累加关系权重
5. **描述摘要**：用 LLM 合并多个描述片段为连贯描述
6. **社区检测**：使用 Hierarchical Leiden 算法发现实体社区
7. **最终输出**：生成带有层次化社区结构的知识图谱

这种设计使得 GraphRAG 能够从非结构化文本中自动构建丰富的知识图谱，为后续的图增强检索（Graph-RAG）提供基础。
