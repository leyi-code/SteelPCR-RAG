# -*- coding: utf-8 -*-
"""
Definitions, condition combinations and scoring anchors for the four case-validation scenarios

How each field is used (verifiable line by line)
  . Only two things take part in retrieval and generation: the id/title/scenario/conditions/question
    fields of CASES and build_complex_question(). The prompts live with the code that sends them:
    SYSTEM_PROMPT and EXPAND_SYSTEM in rag_core/pipeline.py, DECOMPOSE_SYSTEM and DECOMPOSE_V2 plus
    SUBANSWER_SYSTEM and INTEGRATE_SYSTEM in rag_core/multirag.py.
  . kb_notes and rubric carry the anchor set of the case-validation protocol (Anchor / Condition / Gap /
    Forbid); these items are scored against the generated answer, while the four case scripts here only
    generate answers and never pass either field to a model

Where the rubric comes from:
  * the retrieval task and the assessment focus written for each case in the task brief
  * the core source papers actually retrievable from the knowledge base (checked by hand)
Anchor types:
  anchor   - a value or direction that the evidence must support, together with its conditions
  cond     - a conclusion must be tied to the operating conditions (feedstock x parameters x boundary), never generalised
  conflict - a directional conflict or trade-off must be exposed
  boundary - the system boundary must be stated explicitly
  gap      - an evidence gap must be admitted, never invented
  forbid   - an out-of-scope or wrong claim that must not appear (penalised)
"""

import os


CASES = [
    # ============================================================ case 1
    {
        "id": 1,
        "title": "烧结配加生物质炭——源头减排 + 过程节能减排 + 三废治理",
        "scenario": "烧结工序以生物质炭部分替代焦粉",
        "conditions": [
            ("基准", "全焦粉（生物质炭替代比例 0%）"),
            ("工况A", "生物质炭替代焦粉 20%"),
            ("工况B", "生物质炭替代焦粉 40%（文献报告的适宜上限）"),
            ("工况C", "生物质炭替代焦粉 60%（超出适宜窗口）"),
        ],
        "question": (
            "请检索上述生物质炭配比下烧结烟气SO₂、NOx排放规律的证据，标注每条证据的来源期刊、适用配比范围，"
            "并明确指出：文献中关于NOx随配比变化的趋势（线性下降 vs 先降后升 vs 存在阈值）是否存在方向性冲突？"
            "冲突是否与生物质炭类型或烧结工况（料层、燃料分布）有关？"
        ),
        "kb_notes": (
            "库内核心文献：'Effect of biochar substitution on iron ore sintering characteristics based on "
            "optimization of fuel distribution through the bed'（56块）。已核实要点：①生物质炭 N、S 含量远低于焦粉，"
            "随替代比例升高，烟气 NO 与 SO₂ 浓度逐渐降低；②不同生物质适宜替代比例不同：木炭40%、碳化秸秆20%、成型木屑15%；"
            "③替代比例过高时燃烧速率过快、燃烧效率下降，烧结矿质量（DTMT/MQI/TI）显著恶化；"
            "④分层配碳（DFA）可把替代比例由40%提到50%。库内未见“先降后升”或方向性冲突的直接证据。"
        ),
        "rubric": [
            {"id": "1-1", "type": "anchor", "w": 3,
             "item": "给出“随生物质炭替代比例升高，烧结烟气 NO 与 SO₂ 浓度下降”的方向性结论，并绑定机理（生物质炭 N、S 含量远低于焦粉）"},
            {"id": "1-2", "type": "cond", "w": 3,
             "item": "把“40% 适宜替代上限”归属到**特定生物质类型（木炭/charcoal）**，而不是泛化为“所有生物质炭 40%”（碳化秸秆20%、成型木屑15%）"},
            {"id": "1-3", "type": "anchor", "w": 3,
             "item": "指出超出适宜比例后是**烧结矿质量与燃烧效率恶化**（DTMT/MQI/TI 下降），而不是“排放继续下降、无代价”"},
            {"id": "1-4", "type": "gap", "w": 2,
             "item": "对“先降后升”这一选项，明确说明库内未见该方向性冲突的直接证据（或说明现有证据均指向单调下降），而不是虚构存在分歧"},
            {"id": "1-5", "type": "cond", "w": 2,
             "item": "对 20% / 40% / 60% 三个工况分别给出结论或明确指出哪个工况库内无直接证据；60% 超窗口不得直接套用 40% 结论"},
            {"id": "1-6", "type": "anchor", "w": 2,
             "item": "提到底层可调手段（分层配碳/燃料分布优化）可提升可替代比例（40%→50%），或指出 SO₂ 还受铁矿石硫限制、不能只按燃料硫外推"},
            {"id": "1-7", "type": "forbid", "w": -3,
             "item": "不得凭空给出库外数值（如 28.2%、25.7%）或其他文献未提供的数据"},
        ],
    },
    # ============================================================ case 2
    {
        "id": 2,
        "title": "高炉喷吹生物质炭——源头减排 + 过程节能减排",
        "scenario": "高炉风口喷吹生物质炭替代喷吹煤（PCI）",
        "conditions": [
            ("基准", "全煤粉喷吹"),
            ("工况A", "高挥发分、低灰生物质炭，替代 25%"),
            ("工况B", "高灰、高碱金属生物质炭，替代 50%"),
            ("工况C", "替代 75%（接近文献评估的高替代边界）"),
        ],
        "question": (
            "请检索上述生物质炭类型与替代比例下高炉SO₂、NOx排放趋势及关键约束条件（碱金属负荷、灰分、燃料比）的证据，"
            "标注来源与适用生物炭特征，并指出：文献中是否存在“某类生物质炭有利于降碳但可能增加特定污染物负荷”的权衡证据？"
        ),
        "kb_notes": (
            "库内核心文献：'Towards Carbon-Neutral Ironmaking: Stepwise Integration of Biocarbon in PCI'（84块）——"
            "已核实：①torrefied pine/kenaf 增加 fuel-N → 提高 NOx；②ashless treatment 去除 AAEM，"
            "kenaf 钾去除率99.5%、UBC降75%、NOx降43%；③高灰、高碱金属 → slagging/fouling 与碱金属循环约束；"
            "④生物质热值低、HGI 差、灰分高 → 直接替代不可行，需预处理。另有 'Modelling and optimisation of biomass "
            "injection in ironmaking blast furnaces'（414块）与 'Combustibility of biochar injected into the raceway'（122块）。"
        ),
        "rubric": [
            {"id": "2-1", "type": "conflict", "w": 3,
             "item": "明确指出“降碳明确但污排方向不统一，取决于生物质炭的 N 含量/挥发分/灰分/碱金属组分”，不给出无条件统一方向"},
            {"id": "2-2", "type": "anchor", "w": 3,
             "item": "给出“提高燃料氮（如烘焙后 fuel-N 升高）→ NOx 升高”的权衡证据，并与“低氮/低硫生物质→排放下降”并列"},
            {"id": "2-3", "type": "anchor", "w": 3,
             "item": "给出碱金属/灰分带来的炉况约束（碱金属循环、K/Na/Zn 对焦炭反应性、渣系、铁水质量、slagging/fouling）"},
            {"id": "2-4", "type": "cond", "w": 3,
             "item": "对 25% / 50% / 75% 三个替代比例分别绑定其适用前提；高替代比例（尤其75%）须标注为库内无直接工业验证或需预处理支撑"},
            {"id": "2-5", "type": "anchor", "w": 2,
             "item": "指出生物质炭硫含量显著低于煤（或低硫有利于 SO₂ 下降），同时能指出脱除 AAEM 可同时降低 NOx（如 K 去除→NOx 降低）"},
            {"id": "2-6", "type": "gap", "w": 2,
             "item": "说明库内缺乏的定量支撑（如高炉全流程 SO₂/NOx 实测排放因子、75% 替代的工业数据），而不是编造"},
            {"id": "2-7", "type": "forbid", "w": -3,
             "item": "不得凭空给出库外数值（如 27% GHG 减排）或其他文献未提供的数据"},
        ],
    },
    # ============================================================ case 3
    {
        "id": 3,
        "title": "氢基竖炉——跨维度综合协同 + 末端捕集",
        "scenario": "高炉长流程改为不同气源的气基竖炉直接还原",
        "conditions": [
            ("基准", "传统高炉长流程"),
            ("工况A", "天然气基竖炉直接还原"),
            ("工况B", "焦炉煤气基竖炉直接还原"),
            ("工况C", "绿氢基竖炉直接还原（系统边界扩展至上游能源）"),
        ],
        "question": (
            "请检索上述不同气源氢基竖炉路线下NOx、SO₂排放特征的证据，标注每条证据的系统边界"
            "（仅直接排放 vs 全生命周期/上游能源），并指出：文献中是否存在“直接排放显著降低，但上游或间接排放增加”的权衡证据？"
            "该权衡与气源类型如何关联？"
        ),
        "kb_notes": (
            "库内相关文献：'Dynamic modeling of a direct reduced iron shaft furnace'（196块）、"
            "'Toward decarbonization in steelmaking: A review on direct reduced iron using biomass-based syngas'（204块）、"
            "'Hydrogen direct reduction ironmaking process for zero CO2 emission'（174块）、"
            "'Towards carbon neutrality: The ammonia approach to green steel'（164块），以及 276 篇含 LCA/全生命周期内容的文档。"
            "注意：任务书中的“NOx降低43.2%”在库内未检索到原文出处。"
        ),
        "rubric": [
            {"id": "3-1", "type": "boundary", "w": 3,
             "item": "对每条排放结论显式标注系统边界（仅直接排放 / 含上游能源或全生命周期），并说明结论不能跨边界套用"},
            {"id": "3-2", "type": "conflict", "w": 3,
             "item": "指出“直接排放降低但上游/间接排放（发电、制氢、燃料制备）可能增加”的权衡，或明确说明库内该方向证据不足"},
            {"id": "3-3", "type": "cond", "w": 3,
             "item": "把 NOx/SO₂ 特征与气源类型（天然气/焦炉煤气/绿氢）关联，说明不同气源的杂质（H₂S、含氮组分）与燃烧条件导致趋势不同"},
            {"id": "3-4", "type": "cond", "w": 2,
             "item": "四个工况分别给出结论；对无直接证据的工况（如焦炉煤气基、绿氢基的实测 NOx/SO₂）明确标注证据缺失"},
            {"id": "3-5", "type": "anchor", "w": 2,
             "item": "给出库内可查的定量支撑（如竖炉 CO₂ 减排幅度、氢还原的能耗/温度或还原度约束），且带适用边界"},
            {"id": "3-6", "type": "forbid", "w": -3,
             "item": "不得把“NOx降低43.2%”这类无来源数值当作库内证据无条件泛化，也不得编造其他具体数值"},
        ],
    },
    # ============================================================ case 4
    {
        "id": 4,
        "title": "钢渣催化污泥热解——三废治理 + 资源化利用 + 产业耦合",
        "scenario": "以钢渣粉（SSP）为添加剂催化污泥热解（以废治废）",
        "conditions": [
            ("基准", "污泥单独热解"),
            ("工况A", "钢渣添加 10%，500℃"),
            ("工况B", "钢渣添加 20%，500℃（文献核心工况附近）"),
            ("工况C", "钢渣添加 20%，600℃"),
        ],
        "question": (
            "请检索上述钢渣添加比例与温度条件下污泥热解含氮/含硫气体释放规律的证据，标注来源、适用添加比例与温度窗口，"
            "并指出：文献中关于“钢渣抑制效果随温度或添加比例如何变化”是否给出了明确的边界条件？是否存在抑制效果减弱或反向的工况证据？"
        ),
        "kb_notes": (
            "库内核心文献：'Steel slag catalyzed municipal sludge pyrolysis to produce low-pollution gas, biochar, "
            "and bio-oil products'（89块，即任务书所指 Yang 2025 J. Environ. Manage.）。已核实原始条件："
            "**钢渣粉添加 10%，管式炉 600℃**，N₂ 气氛。原始数据：热解油中含氮化合物减少 79.09%；羧酸类由 25.54% 降至 13.69%（减少 46.40%）；"
            "热解碳中 N 元素减少 29.44%、S 元素减少 28.11%；三相产物油 -45.06%、气 +13.83%、碳 +4.89%；"
            "SSP 主成分 CaO 35.35%、Fe₂O₃ 27.24%。注意：库内**没有** 20% 添加比例与 500℃ 的直接实验证据，也没有温度序列边界。"
        ),
        "rubric": [
            {"id": "4-1", "type": "cond", "w": 4,
             "item": "把“含氮化合物减少 79.09%”严格锚定到 **钢渣粉添加 10% + 600℃（N₂/惰性气氛）**，而非题目假设的 20%/500℃"},
            {"id": "4-2", "type": "gap", "w": 4,
             "item": "明确指出题目给出的 20% 添加比例与 500℃ 工况在库内**无直接实验证据**，不能把 10%/600℃ 结果直接外推到该工况"},
            {"id": "4-3", "type": "anchor", "w": 3,
             "item": "区分不同指标口径：油中含氮化合物 79.09%、羧酸类 46.40%（25.54%→13.69%）、热解碳 N 29.44%/S 28.11%，不得混用"},
            {"id": "4-4", "type": "anchor", "w": 3,
             "item": "给出含硫侧证据（S 元素 28.11% 或含硫气体逸出受抑），并说明与含氮侧的差异"},
            {"id": "4-5", "type": "gap", "w": 2,
             "item": "说明“抑制效果随温度/添加比例如何变化”在库内只有单一温度点（600℃）与单一致剂量（10%），没有边界序列，因此该问题方面证据不足"},
            {"id": "4-6", "type": "anchor", "w": 2,
             "item": "给出机理解释（SSP 中 CaO/Fe₂O₃ 等金属氧化物催化二次裂解、碱性氧化物影响 NH₃/HCN 释放）"},
            {"id": "4-7", "type": "forbid", "w": -3,
             "item": "不得把 79.09% 泛化为“钢渣总是有效”或直接套到 20%/500℃ 上；不得编造温度序列或添加比例序列数据"},
        ],
    },
]


PROBE_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(PROBE_ROOT, "rerun")   # a run writes here


def build_complex_question(case) -> str:
    """Join the condition list and the original retrieval task into the single-round input (shared by the S and D arms so the input is identical)."""
    lines = ["【工况组合】"]
    for tag, desc in case["conditions"]:
        lines.append(f"- {tag}：{desc}")
    lines.append("")
    lines.append("【检索任务】")
    lines.append(case["question"])
    return "\n".join(lines)


def get_case(cid: int):
    for c in CASES:
        if c["id"] == cid:
            return c
    raise KeyError(f"unknown case id={cid}")
