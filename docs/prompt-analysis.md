# GraphRAG 各阶段 Prompt 深度解析

本文档对 GraphRAG 系统中所有 Prompt 进行深度分析和索引。

---

## 一、Prompt 全景索引

### 索引阶段 (Indexing)

| Prompt 名称 | 文件位置 | 用途 |
|------------|---------|------|
| `GRAPH_EXTRACTION_PROMPT` | `prompts/index/extract_graph.py` | 从文本中抽取实体和关系 |
| `CONTINUE_PROMPT` | `prompts/index/extract_graph.py` | Gleaning 机制：继续抽取遗漏实体 |
| `LOOP_PROMPT` | `prompts/index/extract_graph.py` | Gleaning 机制：判断是否继续 |
| `SUMMARIZE_PROMPT` | `prompts/index/summarize_descriptions.py` | 合并多个实体描述 |
| `COMMUNITY_REPORT_PROMPT` | `prompts/index/community_report.py` | 生成社区报告 |
| `COMMUNITY_REPORT_TEXT_PROMPT` | `prompts/index/community_report_text_units.py` | 基于文本单元生成社区报告 |
| `EXTRACT_CLAIMS_PROMPT` | `prompts/index/extract_claims.py` | 抽取声明/协变量 |

### 查询阶段 (Query)

| Prompt 名称 | 文件位置 | 用途 |
|------------|---------|------|
| `LOCAL_SEARCH_SYSTEM_PROMPT` | `prompts/query/local_search_system_prompt.py` | Local Search 生成答案 |
| `MAP_SYSTEM_PROMPT` | `prompts/query/global_search_map_system_prompt.py` | Global Search Map 阶段 |
| `REDUCE_SYSTEM_PROMPT` | `prompts/query/global_search_reduce_system_prompt.py` | Global Search Reduce 阶段 |
| `DRIFT_LOCAL_SYSTEM_PROMPT` | `prompts/query/drift_search_system_prompt.py` | DRIFT Search 本地搜索 |
| `DRIFT_REDUCE_PROMPT` | `prompts/query/drift_search_system_prompt.py` | DRIFT Search 最终合并 |
| `DRIFT_PRIMER_PROMPT` | `prompts/query/drift_search_system_prompt.py` | DRIFT Search 查询扩展 |
| `BASIC_SEARCH_SYSTEM_PROMPT` | `prompts/query/basic_search_system_prompt.py` | Basic Search 生成答案 |
| `QUESTION_SYSTEM_PROMPT` | `prompts/query/question_gen_system_prompt.py` | 生成候选问题 |
| `RATE_QUERY` | `query/context_builder/rate_prompt.py` | 动态社区相关性评分 |
| `GENERAL_KNOWLEDGE_INSTRUCTION` | `prompts/query/global_search_knowledge_system_prompt.py` | 通用知识标注指令 |

---

## 二、索引阶段 Prompt 详解

### 2.1 实体关系抽取 Prompt

**文件**: `graphrag/prompts/index/extract_graph.py`

#### GRAPH_EXTRACTION_PROMPT

**目标**: 从文本中识别指定类型的实体及其关系

**输入变量**:
- `{entity_types}`: 要抽取的实体类型列表，如 `["organization", "person", "geo", "event"]`
- `{tuple_delimiter}`: 元组分隔符，默认 `<|>`
- `{record_delimiter}`: 记录分隔符，默认 `##`
- `{completion_delimiter}`: 完成标记，默认 `<|COMPLETE|>`
- `{input_text}`: 待处理的文本

**Prompt 结构**:

```markdown
-Goal-
给定一个文本文档和实体类型列表，识别所有这些类型的实体以及实体间的关系。

-Steps-
1. 识别实体，提取：
   - entity_name: 实体名称（大写）
   - entity_type: 实体类型
   - entity_description: 实体描述
   格式: ("entity"<|>NAME<|>TYPE<|>DESCRIPTION)

2. 识别关系，提取：
   - source_entity: 源实体名称
   - target_entity: 目标实体名称
   - relationship_description: 关系描述
   - relationship_strength: 关系强度（数值）
   格式: ("relationship"<|>SOURCE<|>TARGET<|>DESCRIPTION<|>STRENGTH)

3. 以 {record_delimiter} 分隔输出

4. 完成后输出 {completion_delimiter}

######################
-Examples-  (3个 few-shot 示例)
######################

-Real Data-
Entity_types: {entity_types}
Text: {input_text}
```

**输出示例**:
```
("entity"<|>CENTRAL INSTITUTION<|>ORGANIZATION<|>The Central Institution is the Federal Reserve of Verdantis)
##
("entity"<|>MARTIN SMITH<|>PERSON<|>Martin Smith is the chair of the Central Institution)
##
("relationship"<|>MARTIN SMITH<|>CENTRAL INSTITUTION<|>Martin Smith is the Chair of the Central Institution<|>9)
<|COMPLETE|>
```

---

#### CONTINUE_PROMPT (Gleaning)

**用途**: 在首次抽取后，提示 LLM 继续抽取遗漏的实体

```
MANY entities and relationships were missed in the last extraction.
Remember to ONLY emit entities that match any of the previously extracted types.
Add them below using the same format:
```

---

#### LOOP_PROMPT (Gleaning)

**用途**: 询问 LLM 是否还有遗漏的实体需要抽取

```
It appears some entities and relationships may have still been missed.
Answer Y if there are still entities or relationships that need to be added,
or N if there are none. Please answer with a single letter Y or N.
```

---

### 2.2 描述摘要 Prompt

**文件**: `graphrag/prompts/index/summarize_descriptions.py`

#### SUMMARIZE_PROMPT

**目标**: 将来自多个文档的实体描述合并为一个连贯的综合描述

**输入变量**:
- `{entity_name}`: 实体名称
- `{description_list}`: 描述列表
- `{max_length}`: 最大字数限制

**Prompt**:

```markdown
You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.

Given one or more entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description.
Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we have the full context.
Limit the final description length to {max_length} words.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
```

---

### 2.3 社区报告生成 Prompt

**文件**: `graphrag/prompts/index/community_report.py`

#### COMMUNITY_REPORT_PROMPT

**目标**: 为每个社区生成结构化的分析报告

**输入变量**:
- `{input_text}`: 社区数据（实体、关系、声明）
- `{max_report_length}`: 最大报告字数

**Prompt 结构**:

```markdown
# Goal
Write a comprehensive report of a community, given a list of entities that belong to the community
as well as their relationships and optional associated claims.
The report will be used to inform decision-makers about information associated with the community and their potential impact.

# Report Structure
报告应包含以下部分：
- TITLE: 代表社区关键实体的名称（简短但具体）
- SUMMARY: 社区整体结构的执行摘要
- IMPACT SEVERITY RATING: 0-10 分的影响严重程度评分
- RATING EXPLANATION: 评分解释（一句话）
- DETAILED FINDINGS: 5-10 个关键发现列表

# Output Format (JSON)
{
    "title": <report_title>,
    "summary": <executive_summary>,
    "rating": <impact_severity_rating>,
    "rating_explanation": <rating_explanation>,
    "findings": [
        {"summary": <insight_1_summary>, "explanation": <insight_1_explanation>},
        {"summary": <insight_2_summary>, "explanation": <insight_2_explanation>}
    ]
}

# Grounding Rules
数据引用格式：
"This is an example sentence [Data: Reports (1), Entities (5, 7); Relationships (23); Claims (7, 2, 34, 64, 46, +more)]."
- 单个引用不超过 5 个 record id
- 无证据支持的信息不应包含

# Example Input/Output
(包含完整示例)

# Real Data
Text: {input_text}
```

---

### 2.4 声明/协变量抽取 Prompt

**文件**: `graphrag/prompts/index/extract_claims.py`

#### EXTRACT_CLAIMS_PROMPT

**目标**: 从文本中抽取与实体相关的声明（claims/covariates）

**输入变量**:
- `{entity_specs}`: 实体规格（名称列表或类型列表）
- `{claim_description}`: 声明描述类型
- `{input_text}`: 待处理文本
- `{tuple_delimiter}`, `{record_delimiter}`, `{completion_delimiter}`: 分隔符

**抽取字段**:
- **Subject**: 声明主体（大写）
- **Object**: 声明客体（若无则为 NONE）
- **Claim Type**: 声明类型（大写，可跨文档复用）
- **Claim Status**: TRUE / FALSE / SUSPECTED
- **Claim Description**: 详细描述及证据
- **Claim Date**: 日期范围 (ISO-8601)
- **Claim Source Text**: 原文引用

**输出格式**:
```
(<subject>{tuple_delimiter}<object>{tuple_delimiter}<claim_type>{tuple_delimiter}<status>{tuple_delimiter}<start_date>{tuple_delimiter}<end_date>{tuple_delimiter}<description>{tuple_delimiter}<source>)
```

---

## 三、查询阶段 Prompt 详解

### 3.1 Local Search Prompt

**文件**: `graphrag/prompts/query/local_search_system_prompt.py`

#### LOCAL_SEARCH_SYSTEM_PROMPT

**目标**: 基于混合上下文（实体、关系、社区、文本）生成答案

**输入变量**:
- `{response_type}`: 响应类型（如 "multiple paragraphs"）
- `{context_data}`: 上下文数据表

**Prompt 核心结构**:

```markdown
---Role---
You are a helpful assistant responding to questions about data in the tables provided.

---Goal---
Generate a response of the target length and format that responds to the user's question,
summarizing all information in the input data tables appropriate for the response length and format,
and incorporating any relevant general knowledge.

If you don't know the answer, just say so. Do not make anything up.

---Data Reference Format---
"This is an example sentence [Data: <dataset name> (record ids); <dataset name> (record ids)]."
- 单个引用不超过 5 个 record id，超出用 "+more" 表示

---Target response length and format---
{response_type}

---Data tables---
{context_data}

Add sections and commentary to the response as appropriate. Style the response in markdown.
```

---

### 3.2 Global Search Prompts

**文件**: `graphrag/prompts/query/global_search_map_system_prompt.py` 和 `global_search_reduce_system_prompt.py`

#### MAP_SYSTEM_PROMPT

**目标**: 从单个社区报告批次中提取关键点并评分

**输入变量**:
- `{context_data}`: 社区报告批次
- `{max_length}`: 最大响应字数

**输出格式 (JSON)**:
```json
{
    "points": [
        {"description": "Description of point 1 [Data: Reports (report ids)]", "score": 85},
        {"description": "Description of point 2 [Data: Reports (report ids)]", "score": 72}
    ]
}
```

**关键规则**:
- 每个关键点包含 description 和 importance score (0-100)
- "I don't know" 类型的响应应得分 0
- 引用格式: `[Data: Reports (report ids)]`

---

#### REDUCE_SYSTEM_PROMPT

**目标**: 综合多个分析师的报告生成最终答案

**输入变量**:
- `{response_type}`: 响应类型
- `{report_data}`: 来自 Map 阶段的聚合报告（按重要性降序）
- `{max_length}`: 最大响应字数

**核心指令**:
```markdown
---Role---
You are a helpful assistant responding to questions about a dataset by synthesizing perspectives from multiple analysts.

---Goal---
Generate a response that summarizes all the reports from multiple analysts who focused on different parts of the dataset.

Note that the analysts' reports provided below are ranked in the **descending order of importance**.

The final response should:
- Remove all irrelevant information from the analysts' reports
- Merge the cleaned information into a comprehensive answer
- Preserve all data references
- Do not mention the roles of multiple analysts in the analysis process
```

---

### 3.3 DRIFT Search Prompts

**文件**: `graphrag/prompts/query/drift_search_system_prompt.py`

#### DRIFT_PRIMER_PROMPT

**目标**: 扩展初始查询，生成中间答案和后续问题

**输入变量**:
- `{query}`: 用户查询
- `{community_reports}`: 顶级社区摘要

**输出格式 (JSON)**:
```json
{
    "intermediate_answer": "2000字符的中间答案（markdown格式）",
    "score": 75,
    "follow_up_queries": ["后续问题1", "后续问题2", "后续问题3", "后续问题4", "后续问题5"]
}
```

**关键特点**:
- intermediate_answer 必须正好 2000 字符
- 至少生成 5 个后续问题
- 不要问复合问题（如 "What is the market cap of Apple and Microsoft?"）

---

#### DRIFT_LOCAL_SYSTEM_PROMPT

**目标**: DRIFT Search 中的本地搜索步骤

**独特功能**:
- 包含 `{global_query}` 参数（全局研究问题）
- 要求对答案评分 (0-100)
- 生成后续问题建议

**输出格式 (JSON)**:
```json
{
    "response": "markdown格式的答案",
    "score": 85,
    "follow_up_queries": ["问题1", "问题2", "问题3"]
}
```

---

#### DRIFT_REDUCE_PROMPT

**目标**: 聚合所有中间答案生成最终响应

**独特功能**:
- 支持通用知识标注: `[Data: General Knowledge (href)]`
- 强调准确性和简洁性

---

### 3.4 Basic Search Prompt

**文件**: `graphrag/prompts/query/basic_search_system_prompt.py`

#### BASIC_SEARCH_SYSTEM_PROMPT

**目标**: 基于纯向量检索的文本块生成答案

与 Local Search 类似，但:
- 上下文仅包含文本单元（Sources）
- 引用格式: `[Data: Sources (record ids)]`

---

### 3.5 辅助 Prompts

#### RATE_QUERY (动态社区选择)

**文件**: `graphrag/query/context_builder/rate_prompt.py`

**目标**: 评估社区信息与查询的相关性

**输出格式 (JSON)**:
```json
{
    "reason": "评分理由",
    "rating": 4
}
```
- 评分范围: 0-5

---

#### QUESTION_SYSTEM_PROMPT

**文件**: `graphrag/prompts/query/question_gen_system_prompt.py`

**目标**: 基于数据表生成候选问题

**输入变量**:
- `{question_count}`: 要生成的问题数量
- `{context_data}`: 上下文数据

---

#### GENERAL_KNOWLEDGE_INSTRUCTION

**文件**: `graphrag/prompts/query/global_search_knowledge_system_prompt.py`

**用途**: 标注来自 LLM 通用知识的信息

```markdown
The response may also include relevant real-world knowledge outside the dataset,
but it must be explicitly annotated with a verification tag [LLM: verify].
```

---

## 四、Prompt 设计模式分析

### 4.1 Few-Shot 示例模式

**使用场景**: 实体抽取、声明抽取

**结构**:
```
-Goal-
...

-Steps-
...

######################
-Examples-
Example 1: ...
Example 2: ...
Example 3: ...
######################

-Real Data-
{input}
```

**优点**: 通过具体示例指导 LLM 输出格式

---

### 4.2 结构化 JSON 输出模式

**使用场景**: 社区报告、Global Search Map、DRIFT Search

**特点**:
- 明确定义 JSON schema
- 包含字段说明和示例
- 便于程序解析

---

### 4.3 数据引用规范

**统一格式**:
```
[Data: <dataset name> (record ids); <dataset name> (record ids)]
```

**规则**:
- 单个引用不超过 5 个 ID
- 超出用 `+more` 表示
- 无证据不得包含

**目的**: 确保答案可溯源、可验证

---

### 4.4 Gleaning 迭代模式

**使用场景**: 实体抽取、声明抽取

**流程**:
```
初始抽取 → CONTINUE_PROMPT → 补充抽取 → LOOP_PROMPT (Y/N?) → 循环或结束
```

**目的**: 提高召回率，减少遗漏

---

## 五、Prompt 变量映射表

### 索引阶段

| Prompt | 变量 | 类型 | 描述 |
|--------|------|------|------|
| `GRAPH_EXTRACTION` | `entity_types` | list[str] | 实体类型列表 |
| | `tuple_delimiter` | str | 默认 `<\|>` |
| | `record_delimiter` | str | 默认 `##` |
| | `completion_delimiter` | str | 默认 `<\|COMPLETE\|>` |
| | `input_text` | str | 待处理文本 |
| `SUMMARIZE` | `entity_name` | str | 实体名称 |
| | `description_list` | list[str] | 描述列表 |
| | `max_length` | int | 最大字数 |
| `COMMUNITY_REPORT` | `input_text` | str | 社区数据 |
| | `max_report_length` | int | 报告最大字数 |
| `EXTRACT_CLAIMS` | `entity_specs` | str | 实体规格 |
| | `claim_description` | str | 声明类型描述 |
| | `input_text` | str | 待处理文本 |

### 查询阶段

| Prompt | 变量 | 类型 | 描述 |
|--------|------|------|------|
| `LOCAL_SEARCH` | `response_type` | str | 如 "multiple paragraphs" |
| | `context_data` | str | 上下文数据表 |
| `MAP` | `context_data` | str | 社区报告批次 |
| | `max_length` | int | 最大字数 |
| `REDUCE` | `response_type` | str | 响应类型 |
| | `report_data` | str | Map 阶段聚合数据 |
| | `max_length` | int | 最大字数 |
| `DRIFT_PRIMER` | `query` | str | 用户查询 |
| | `community_reports` | str | 社区摘要 |
| `DRIFT_LOCAL` | `response_type` | str | 响应类型 |
| | `context_data` | str | 上下文数据 |
| | `global_query` | str | 全局研究问题 |
| `RATE_QUERY` | `description` | str | 社区描述 |
| | `question` | str | 用户问题 |

---

## 六、总结

GraphRAG 的 Prompt 体系具有以下特点：

1. **阶段分离**: 索引阶段专注于信息抽取和结构化，查询阶段专注于检索和生成
2. **格式规范**: 统一的数据引用格式确保可溯源性
3. **迭代优化**: Gleaning 机制提高召回率
4. **结构化输出**: JSON 格式便于程序解析和后续处理
5. **多模式支持**: 不同搜索模式有针对性的 Prompt 设计
6. **质量控制**: 评分机制、重要性排序确保输出质量
