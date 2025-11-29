# GraphRAG 查询处理流程深度解析

本文档深度分析 GraphRAG 系统中查询从输入到最终响应的完整处理流程。

## 一、整体架构概览

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            GraphRAG 查询处理流程                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   User Query                                                                    │
│       │                                                                         │
│       ▼                                                                         │
│   ┌─────────────────┐                                                           │
│   │  CLI / API      │  入口点: cli/query.py, api/query.py                       │
│   └────────┬────────┘                                                           │
│            │                                                                    │
│            ▼                                                                    │
│   ┌─────────────────┐                                                           │
│   │  Data Loading   │  加载索引数据: entities, relationships, communities...    │
│   └────────┬────────┘                                                           │
│            │                                                                    │
│            ▼                                                                    │
│   ┌─────────────────┐                                                           │
│   │  Search Factory │  创建搜索引擎实例: factory.py                              │
│   └────────┬────────┘                                                           │
│            │                                                                    │
│            ├──────────────┬──────────────┬──────────────┐                       │
│            ▼              ▼              ▼              ▼                       │
│   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐           │
│   │ LocalSearch  │ │ GlobalSearch │ │ DRIFTSearch  │ │ BasicSearch  │           │
│   └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘           │
│          │                │                │                │                   │
│          ▼                ▼                ▼                ▼                   │
│   ┌─────────────────────────────────────────────────────────────────┐           │
│   │                    Context Builder                              │           │
│   │  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐          │           │
│   │  │Entity Retrieval│ │Community     │ │Text Unit      │          │           │
│   │  │(向量相似度)    │ │Selection     │ │Retrieval      │          │           │
│   │  └───────────────┘ └───────────────┘ └───────────────┘          │           │
│   └────────────────────────────┬────────────────────────────────────┘           │
│                                │                                                │
│                                ▼                                                │
│   ┌─────────────────────────────────────────────────────────────────┐           │
│   │                     LLM Response Generation                     │           │
│   │                   (Streaming / Non-Streaming)                   │           │
│   └────────────────────────────┬────────────────────────────────────┘           │
│                                │                                                │
│                                ▼                                                │
│                        Final Response                                           │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、入口层 (Entry Point Layer)

### 1. CLI 入口

**文件位置**: `graphrag/cli/query.py`

用户通过命令行调用查询：

```bash
graphrag query --method local --query "What is the main topic?"
graphrag query --method global --query "Summarize the key themes"
```

### 2. API 入口

**文件位置**: `graphrag/api/query.py`

提供程序化访问接口：

```python
# 同步搜索
response, context_data = await local_search(
    config=config,
    entities=entities,
    communities=communities,
    community_reports=community_reports,
    text_units=text_units,
    relationships=relationships,
    covariates=covariates,
    community_level=2,
    response_type="multiple paragraphs",
    query="What are the main themes?",
)

# 流式搜索
async for chunk in local_search_streaming(...):
    print(chunk, end="")
```

### 3. 支持的搜索模式

| 搜索模式 | API 函数 | 适用场景 |
|---------|---------|---------|
| **Local Search** | `local_search()` | 针对具体实体的细节查询 |
| **Global Search** | `global_search()` | 全局主题/摘要类查询 |
| **DRIFT Search** | `drift_search()` | 需要迭代探索的复杂查询 |
| **Basic Search** | `basic_search()` | 简单向量检索 |

---

## 三、数据加载层 (Data Loading Layer)

**文件位置**: `graphrag/api/query.py:161-172` (以 global_search_streaming 为例)

```python
# 加载社区数据
communities_ = read_indexer_communities(communities, community_reports)

# 加载社区报告
reports = read_indexer_reports(
    community_reports,
    communities,
    community_level=community_level,
    dynamic_community_selection=dynamic_community_selection,
)

# 加载实体
entities_ = read_indexer_entities(entities, communities, community_level)

# 加载 Prompt
map_prompt = load_search_prompt(config.root_dir, config.global_search.map_prompt)
reduce_prompt = load_search_prompt(config.root_dir, config.global_search.reduce_prompt)
```

### 数据适配器函数

**文件位置**: `graphrag/query/indexer_adapters.py`

| 函数 | 用途 |
|-----|-----|
| `read_indexer_entities()` | 加载实体列表，包含嵌入向量 |
| `read_indexer_relationships()` | 加载关系图数据 |
| `read_indexer_communities()` | 加载社区层级结构 |
| `read_indexer_reports()` | 加载社区报告 |
| `read_indexer_text_units()` | 加载原始文本块 |
| `read_indexer_covariates()` | 加载协变量/声明 |

---

## 四、搜索引擎工厂层 (Search Engine Factory Layer)

**文件位置**: `graphrag/query/factory.py`

工厂函数创建完整配置的搜索引擎实例：

### Local Search 引擎创建

```python
def get_local_search_engine(config, reports, text_units, entities, relationships,
                            covariates, response_type, description_embedding_store, ...):
    # 1. 创建 Chat 模型
    chat_model = ModelManager().get_or_create_chat_model(
        name="local_search_chat",
        model_type=model_settings.type,
        config=model_settings,
    )

    # 2. 创建 Embedding 模型
    embedding_model = ModelManager().get_or_create_embedding_model(
        name="local_search_embedding",
        model_type=embedding_settings.type,
        config=embedding_settings,
    )

    # 3. 创建 Tokenizer
    tokenizer = get_tokenizer(model_config=model_settings)

    # 4. 构建 LocalSearch 实例
    return LocalSearch(
        model=chat_model,
        system_prompt=system_prompt,
        context_builder=LocalSearchMixedContext(
            community_reports=reports,
            text_units=text_units,
            entities=entities,
            relationships=relationships,
            covariates=covariates,
            entity_text_embeddings=description_embedding_store,
            text_embedder=embedding_model,
            tokenizer=tokenizer,
        ),
        tokenizer=tokenizer,
        context_builder_params={
            "text_unit_prop": 0.5,        # 文本单元占比
            "community_prop": 0.25,       # 社区报告占比
            "top_k_mapped_entities": 10,  # 检索实体数量
            "top_k_relationships": 10,    # 检索关系数量
            "max_context_tokens": 12000,  # 最大上下文 token
        },
        response_type=response_type,
    )
```

---

## 五、Local Search 详细流程

### 整体流程图

```
Query
  │
  ▼
┌──────────────────────────────────────────────────────────────┐
│                    LocalSearch.search()                      │
├──────────────────────────────────────────────────────────────┤
│  1. 构建上下文 (context_builder.build_context)               │
│     ├─ 实体检索 (map_query_to_entities)                      │
│     ├─ 社区上下文 (_build_community_context)                  │
│     ├─ 本地上下文 (_build_local_context)                      │
│     │   ├─ 实体上下文                                        │
│     │   ├─ 关系上下文                                        │
│     │   └─ 协变量上下文                                      │
│     └─ 文本单元上下文 (_build_text_unit_context)              │
│                                                              │
│  2. 格式化 Prompt                                            │
│     └─ system_prompt.format(context_data=..., response_type=...)│
│                                                              │
│  3. LLM 流式响应                                             │
│     └─ model.achat_stream(prompt=query, history=messages)    │
│                                                              │
│  4. 返回 SearchResult                                        │
└──────────────────────────────────────────────────────────────┘
```

### Step 1: 实体检索 (Entity Retrieval)

**文件位置**: `graphrag/query/context_builder/entity_extraction.py:37-92`

```python
def map_query_to_entities(
    query: str,
    text_embedding_vectorstore: BaseVectorStore,
    text_embedder: EmbeddingModel,
    all_entities_dict: dict[str, Entity],
    k: int = 10,
    oversample_scaler: int = 2,
) -> list[Entity]:
    """通过语义相似度将查询映射到实体"""
    matched_entities = []

    if query != "":
        # 向量相似度搜索（过采样以处理排除项）
        search_results = text_embedding_vectorstore.similarity_search_by_text(
            text=query,
            text_embedder=lambda t: text_embedder.embed(t),
            k=k * oversample_scaler,  # 检索 2x 数量以备过滤
        )

        # 将搜索结果映射回实体对象
        for result in search_results:
            matched = get_entity_by_id(all_entities_dict, result.document.id)
            if matched:
                matched_entities.append(matched)
    else:
        # 无查询时按 rank 排序返回 top-k
        all_entities.sort(key=lambda x: x.rank if x.rank else 0, reverse=True)
        matched_entities = all_entities[:k]

    # 过滤排除项，添加包含项
    return included_entities + matched_entities
```

### Step 2: 上下文构建 (Context Building)

**文件位置**: `graphrag/query/structured_search/local_search/mixed_context.py:91-222`

```python
def build_context(self, query, max_context_tokens=8000, text_unit_prop=0.5,
                  community_prop=0.25, top_k_mapped_entities=10, ...):
    """构建混合上下文"""

    # 1. 实体检索
    selected_entities = map_query_to_entities(
        query=query,
        text_embedding_vectorstore=self.entity_text_embeddings,
        text_embedder=self.text_embedder,
        all_entities_dict=self.entities,
        k=top_k_mapped_entities,
    )

    # 2. 社区上下文 (25% token 预算)
    community_tokens = int(max_context_tokens * community_prop)
    community_context, community_data = self._build_community_context(
        selected_entities=selected_entities,
        max_context_tokens=community_tokens,
    )

    # 3. 本地上下文：实体 + 关系 + 协变量 (25% token 预算)
    local_prop = 1 - community_prop - text_unit_prop
    local_tokens = int(max_context_tokens * local_prop)
    local_context, local_data = self._build_local_context(
        selected_entities=selected_entities,
        max_context_tokens=local_tokens,
    )

    # 4. 文本单元上下文 (50% token 预算)
    text_unit_tokens = int(max_context_tokens * text_unit_prop)
    text_unit_context, text_unit_data = self._build_text_unit_context(
        selected_entities=selected_entities,
        max_context_tokens=text_unit_tokens,
    )

    # 5. 组合所有上下文
    return ContextBuilderResult(
        context_chunks="\n\n".join([community_context, local_context, text_unit_context]),
        context_records={**community_data, **local_data, **text_unit_data},
    )
```

### Step 3: LLM 调用

**文件位置**: `graphrag/query/structured_search/local_search/search.py:51-130`

```python
async def search(self, query: str, conversation_history=None, **kwargs):
    # 1. 构建上下文
    context_result = self.context_builder.build_context(
        query=query,
        conversation_history=conversation_history,
        **self.context_builder_params,
    )

    # 2. 格式化 System Prompt
    search_prompt = self.system_prompt.format(
        context_data=context_result.context_chunks,
        response_type=self.response_type,
    )

    history_messages = [{"role": "system", "content": search_prompt}]

    # 3. 流式调用 LLM
    full_response = ""
    async for response in self.model.achat_stream(
        prompt=query,
        history=history_messages,
        model_parameters=self.model_params,
    ):
        full_response += response
        # 触发回调
        for callback in self.callbacks:
            callback.on_llm_new_token(response)

    # 4. 返回结果
    return SearchResult(
        response=full_response,
        context_data=context_result.context_records,
        context_text=context_result.context_chunks,
        completion_time=time.time() - start_time,
        llm_calls=1,
        prompt_tokens=len(self.tokenizer.encode(search_prompt)),
        output_tokens=len(self.tokenizer.encode(full_response)),
    )
```

---

## 六、Global Search 详细流程 (Map-Reduce 架构)

### 整体流程图

```
Query
  │
  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         GlobalSearch.search()                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │ Stage 1: Context Building                                           │     │
│  │   ├─ (Optional) Dynamic Community Selection                         │     │
│  │   └─ Build community report batches                                 │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │ Stage 2: Map Phase (并行)                                           │     │
│  │                                                                     │     │
│  │   ┌─────────┐   ┌─────────┐   ┌─────────┐                           │     │
│  │   │ Batch 1 │   │ Batch 2 │   │ Batch N │                           │     │
│  │   └────┬────┘   └────┬────┘   └────┬────┘                           │     │
│  │        │             │             │                                │     │
│  │        ▼             ▼             ▼                                │     │
│  │   ┌─────────┐   ┌─────────┐   ┌─────────┐                           │     │
│  │   │  LLM    │   │  LLM    │   │  LLM    │    并行 LLM 调用           │     │
│  │   └────┬────┘   └────┬────┘   └────┬────┘                           │     │
│  │        │             │             │                                │     │
│  │        ▼             ▼             ▼                                │     │
│  │   ┌─────────┐   ┌─────────┐   ┌─────────┐                           │     │
│  │   │Key Points│   │Key Points│   │Key Points│  每批生成关键点+评分    │     │
│  │   │+ Scores │   │+ Scores │   │+ Scores │                           │     │
│  │   └─────────┘   └─────────┘   └─────────┘                           │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │ Stage 3: Reduce Phase                                               │     │
│  │   1. 收集所有 Key Points                                            │     │
│  │   2. 过滤 score=0 的点                                              │     │
│  │   3. 按 score 降序排序                                              │     │
│  │   4. 截断到 max_data_tokens                                         │     │
│  │   5. 调用 LLM 生成最终答案                                          │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                              │                                               │
│                              ▼                                               │
│                       Final Response                                         │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Map 阶段代码

**文件位置**: `graphrag/query/structured_search/global_search/search.py:209-264`

```python
async def _map_response_single_batch(self, context_data, query, max_length, **llm_kwargs):
    """为单个社区报告批次生成答案"""

    # 格式化 Map Prompt
    search_prompt = self.map_system_prompt.format(
        context_data=context_data,
        max_length=max_length,
    )
    search_messages = [{"role": "system", "content": search_prompt}]

    # 调用 LLM (JSON 模式)
    async with self.semaphore:  # 并发控制
        model_response = await self.model.achat(
            prompt=query,
            history=search_messages,
            model_parameters=llm_kwargs,
            json=True,
        )

    # 解析响应为 Key Points 列表
    processed_response = self._parse_search_response(search_response)
    # 返回格式: [{"answer": "...", "score": 8}, ...]

    return SearchResult(response=processed_response, ...)


def _parse_search_response(self, search_response):
    """解析 Map 响应的 JSON"""
    parsed_elements = json.loads(search_response).get("points")
    return [
        {
            "answer": element["description"],
            "score": int(element["score"]),  # 重要性评分
        }
        for element in parsed_elements
    ]
```

### Reduce 阶段代码

**文件位置**: `graphrag/query/structured_search/global_search/search.py:296-413`

```python
async def _reduce_response(self, map_responses, query, **llm_kwargs):
    """合并所有 Map 结果生成最终答案"""

    # 1. 收集所有 Key Points
    key_points = []
    for index, response in enumerate(map_responses):
        for element in response.response:
            key_points.append({
                "analyst": index,
                "answer": element["answer"],
                "score": element["score"],
            })

    # 2. 过滤 score=0 的点（无关信息）
    filtered_key_points = [p for p in key_points if p["score"] > 0]

    if len(filtered_key_points) == 0:
        return SearchResult(response=NO_DATA_ANSWER, ...)

    # 3. 按 score 降序排序
    filtered_key_points.sort(key=lambda x: x["score"], reverse=True)

    # 4. 截断到 token 预算
    data = []
    total_tokens = 0
    for point in filtered_key_points:
        formatted_text = f"----Analyst {point['analyst'] + 1}----\n"
        formatted_text += f"Importance Score: {point['score']}\n"
        formatted_text += point["answer"]

        if total_tokens + len(tokenizer.encode(formatted_text)) > self.max_data_tokens:
            break
        data.append(formatted_text)
        total_tokens += len(tokenizer.encode(formatted_text))

    text_data = "\n\n".join(data)

    # 5. 调用 LLM 生成最终答案
    search_prompt = self.reduce_system_prompt.format(
        report_data=text_data,
        response_type=self.response_type,
    )

    async for chunk in self.model.achat_stream(prompt=query, history=messages):
        search_response += chunk

    return SearchResult(response=search_response, ...)
```

---

## 七、DRIFT Search 详细流程 (迭代探索架构)

### 整体流程图

```
Query
  │
  ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                           DRIFTSearch.search()                                │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │ Step 1: Query Priming (DRIFTPrimer)                                     │  │
│  │   ├─ 使用社区报告模板扩展查询                                            │  │
│  │   └─ 生成初始 follow-up 问题                                            │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                              │                                                │
│                              ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │ Step 2: Initialize QueryState (Action Graph)                            │  │
│  │   └─ NetworkX MultiDiGraph 存储动作节点和关系                            │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                              │                                                │
│                              ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │ Step 3: Iterative Exploration Loop                                      │  │
│  │   ┌────────────────────────────────────────────────────────────────┐    │  │
│  │   │  while has_unanswered_actions():                               │    │  │
│  │   │      1. 获取最高评分的未完成动作                                │    │  │
│  │   │      2. 执行 LocalSearch 获取答案                               │    │  │
│  │   │      3. LLM 评分答案质量                                        │    │  │
│  │   │      4. 生成新的 follow-up 动作                                 │    │  │
│  │   │      5. 更新 Action Graph                                       │    │  │
│  │   └────────────────────────────────────────────────────────────────┘    │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                              │                                                │
│                              ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │ Step 4: Final Reduction                                                 │  │
│  │   └─ 聚合所有中间答案，生成最终响应                                     │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                              │                                                │
│                              ▼                                                │
│                        Final Response                                         │
└───────────────────────────────────────────────────────────────────────────────┘
```

### 核心组件

| 组件 | 文件位置 | 功能 |
|-----|---------|------|
| `DRIFTSearch` | `drift_search/search.py` | 搜索编排器 |
| `DRIFTPrimer` | `drift_search/primer.py` | 查询扩展器 |
| `QueryState` | `drift_search/state.py` | 动作图状态管理 |
| `DriftAction` | `drift_search/action.py` | 单个搜索动作 |

---

## 八、Basic Search 详细流程

### 流程图

```
Query
  │
  ▼
┌──────────────────────────────────────────────────────────────┐
│                    BasicSearch.search()                      │
├──────────────────────────────────────────────────────────────┤
│  1. 向量嵌入查询                                             │
│     └─ text_embedder.embed(query)                            │
│                                                              │
│  2. 相似度搜索 Text Units                                    │
│     └─ text_unit_embeddings.similarity_search_by_text(k=...)│
│                                                              │
│  3. 构建上下文                                               │
│     └─ 将 top-k 文本块拼接为上下文                           │
│                                                              │
│  4. LLM 生成答案                                             │
│     └─ model.achat_stream(prompt=query, context=...)         │
└──────────────────────────────────────────────────────────────┘
```

---

## 九、上下文组成结构

### Local Search 上下文构成

```
┌──────────────────────────────────────────────────────────────────┐
│                     Local Search Context                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Conversation History (if exists)                           │  │
│  │   Previous Q&A turns for context continuity                │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Community Reports (25% token budget)                       │  │
│  │   - Sorted by: matched_entities × rank                     │  │
│  │   - Format: title, summary, rank                           │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Local Context (25% token budget)                           │  │
│  │   ┌──────────────────────────────────────────────────────┐ │  │
│  │   │ Entities                                             │ │  │
│  │   │   id | entity | description | rank                   │ │  │
│  │   └──────────────────────────────────────────────────────┘ │  │
│  │   ┌──────────────────────────────────────────────────────┐ │  │
│  │   │ Relationships                                        │ │  │
│  │   │   id | source | target | description | weight        │ │  │
│  │   └──────────────────────────────────────────────────────┘ │  │
│  │   ┌──────────────────────────────────────────────────────┐ │  │
│  │   │ Covariates/Claims (if available)                     │ │  │
│  │   │   id | subject | type | description | status         │ │  │
│  │   └──────────────────────────────────────────────────────┘ │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Text Units / Sources (50% token budget)                    │  │
│  │   - Original text chunks from source documents             │  │
│  │   - Sorted by: entity_order × relationship_count           │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Global Search 上下文构成

```
┌──────────────────────────────────────────────────────────────────┐
│                     Global Search Context                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Community Reports (batched)                                │  │
│  │                                                            │  │
│  │   Batch 1: [Report_1, Report_2, ..., Report_n]             │  │
│  │   Batch 2: [Report_n+1, ..., Report_m]                     │  │
│  │   ...                                                      │  │
│  │                                                            │  │
│  │   Each report contains:                                    │  │
│  │   - community_id                                           │  │
│  │   - title                                                  │  │
│  │   - summary / full_content                                 │  │
│  │   - rank                                                   │  │
│  │   - occurrence_weight (normalized)                         │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Dynamic Community Selection (optional):                         │
│   - LLM rates relevancy of each community to query              │
│   - Only includes communities with rating >= threshold          │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 十、回调机制 (Callback System)

**文件位置**: `graphrag/callbacks/query_callbacks.py`

```python
class QueryCallbacks:
    """查询回调接口"""

    def on_map_response_start(self, context_chunks: list[str]) -> None:
        """Map 阶段开始"""
        pass

    def on_map_response_end(self, map_responses: list[SearchResult]) -> None:
        """Map 阶段结束"""
        pass

    def on_context(self, context: Any) -> None:
        """上下文构建完成"""
        pass

    def on_llm_new_token(self, token: str) -> None:
        """LLM 生成新 token（流式）"""
        pass
```

---

## 十一、返回结果结构

**文件位置**: `graphrag/query/structured_search/base.py`

```python
@dataclass
class SearchResult:
    """搜索结果数据类"""

    # 核心响应
    response: str | list[dict[str, Any]]
    context_data: str | list[pd.DataFrame] | dict[str, pd.DataFrame]
    context_text: str | list[str] | dict[str, str]

    # 性能指标
    completion_time: float = 0        # 执行时间
    llm_calls: int = 0                # LLM 调用次数
    prompt_tokens: int = 0            # 输入 token 数
    output_tokens: int = 0            # 输出 token 数

    # 分类统计
    llm_calls_categories: dict = {}
    prompt_tokens_categories: dict = {}
    output_tokens_categories: dict = {}
```

---

## 十二、完整调用链路示例

### Local Search 调用链

```
1. CLI: graphrag query --method local --query "..."
   └─ cli/query.py: run_local_search()

2. API 调用:
   └─ api/query.py: local_search_streaming()
       ├─ read_indexer_entities()        # 加载实体
       ├─ read_indexer_relationships()   # 加载关系
       ├─ read_indexer_reports()         # 加载社区报告
       ├─ read_indexer_text_units()      # 加载文本块
       └─ get_local_search_engine()      # 创建搜索引擎

3. Factory 创建:
   └─ factory.py: get_local_search_engine()
       ├─ ModelManager().get_or_create_chat_model()
       ├─ ModelManager().get_or_create_embedding_model()
       └─ LocalSearch(context_builder=LocalSearchMixedContext(...))

4. 执行搜索:
   └─ local_search/search.py: LocalSearch.stream_search()
       ├─ context_builder.build_context()
       │   ├─ map_query_to_entities()        # 向量检索
       │   ├─ _build_community_context()     # 社区上下文
       │   ├─ _build_local_context()         # 实体/关系上下文
       │   └─ _build_text_unit_context()     # 文本上下文
       │
       ├─ system_prompt.format(context_data=...)  # 格式化 Prompt
       │
       └─ model.achat_stream()               # LLM 流式响应
           └─ callback.on_llm_new_token()    # 每个 token 回调

5. 返回结果:
   └─ SearchResult(response=..., context_data=..., metrics=...)
```

---

## 十三、总结

GraphRAG 的查询处理流程具有以下特点：

1. **多模式支持**: Local/Global/DRIFT/Basic 四种搜索模式适应不同场景
2. **分层架构**: 入口层 → 数据加载层 → 工厂层 → 搜索层 → 上下文层
3. **智能检索**: 结合向量相似度和图结构进行实体检索
4. **Token 预算管理**: 精确控制各部分上下文的 token 分配
5. **流式响应**: 支持逐 token 流式返回，提升用户体验
6. **性能监控**: 完整的调用次数、token 消耗、执行时间统计
7. **回调机制**: 可扩展的回调系统支持自定义处理逻辑
