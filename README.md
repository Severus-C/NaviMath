<div align="center">

<a href="https://github.com/Severus-C/NaviMath"><img src="https://placehold.co/128x128/111827/67e8f9?text=NaviMath" width="112" alt="NaviMath logo placeholder" /></a>

# NaviMath

### 面向竞赛数学的可验证、多候选推理智能体
### A verifiable, multi-candidate reasoning agent for competition mathematics

将路由、题型 Skill、并行候选、对抗校验、答案归一化与评测诊断组合成一条可迭代的数学推理流水线。  
Route the problem, solve it several ways, attack the answers, normalize the output, and learn from every miss.

<p>
<a href="https://github.com/Severus-C/NaviMath/stargazers"><img src="https://img.shields.io/github/stars/Severus-C/NaviMath?style=for-the-badge&logo=github&label=Stars" alt="GitHub stars" /></a>
<a href="https://github.com/Severus-C/NaviMath/actions"><img src="https://img.shields.io/github/actions/workflow/status/Severus-C/NaviMath/ci.yml?style=for-the-badge&logo=github-actions&label=CI" alt="CI status" /></a>
<a href="#license"><img src="https://img.shields.io/badge/license-TBD-f59e0b?style=for-the-badge&label=License" alt="License pending" /></a>
<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+" /></a>
<a href="https://github.com/Severus-C/NaviMath/issues"><img src="https://img.shields.io/github/issues/Severus-C/NaviMath?style=for-the-badge&label=Issues" alt="Open issues" /></a>
</p>

<p>快速开始 · Quick start　|　架构 · Architecture　|　基准评测 · Benchmark　|　Roadmap　|　Contributing</p>
</div>

<div align="center">
<img src="https://placehold.co/1280x640/0b1220/67e8f9?text=NaviMath+Animated+Demo+%28placeholder%29" alt="Animated demo placeholder" width="92%" />
<br /><sub>动画演示占位图 · Animated demo placeholder — replace with docs/demo.gif when ready.</sub>
</div>

## 为什么是 NaviMath？ · Why NaviMath?

| 传统单次调用 · One-shot LLM | NaviMath |
|---|---|
| 一个答案、一次机会 | 多角色候选 + 独立攻击校验 |
| 输出字符串难以评分 | schema-aware AnswerNormalizer |
| 失败只能看 accuracy | 自动错因报告与证据链 |
| 所有题目使用同一策略 | 题型路由 + Skill 策略 |
| 调参依赖感觉 | JSONL trace、可复现实验、持续迭代 |

> **核心理念 · Core loop:** 每一道错题都是下一轮路由、校验和格式规范化的训练信号。

## 功能卡片 · Feature cards

<table>
<tr>
<td width="33%"><h3>🧭 智能路由</h3>按代数、几何、数论、概率、复分析等题型选择 Skill、难度和动作预算。</td>
<td width="33%"><h3>🧠 多候选推理</h3>direct solver、domain solver、checker 并行提出独立解法，降低单一路径失误。</td>
<td width="33%"><h3>🛡️ 对抗式验证</h3>攻击候选的假设、边界、计算和最终答案，必要时触发 refine。</td>
</tr>
<tr>
<td><h3>🧾 答案规范化</h3>分数、小数、集合、区间、向量、矩阵、多答案与 AIME 前导零策略。</td>
<td><h3>📊 可解释评测</h3>输出 accuracy、按题型统计、trace、错误根因和 Markdown 报告。</td>
<td><h3>⚙️ 竞赛友好接口</h3>根目录 <code>user_agent.py</code> 暴露稳定的 <code>ReasoningAgent(client=...)</code> 入口。</td>
</tr>
</table>

## 架构 · Architecture

~~~mermaid
flowchart LR
    P[Problem + metadata] --> R[SkillRegistry<br/>route & difficulty]
    R --> S[Solver ensemble<br/>multiple roles]
    S --> N[AnswerNormalizer<br/>extract & canonicalize]
    N --> V[Attack verifier<br/>verdict + evidence]
    V -->|reject| F[Refine candidate]
    F --> N
    V --> C{Consensus?}
    C -->|unique cluster| L[Consensus lock]
    C -->|ambiguous| J[Final judge]
    L --> O[Final response]
    J --> O
    O --> E[Evaluator]
    E --> A[Error analysis<br/>JSON + Markdown]
    A -. feedback .-> R
~~~

## 一题的工作流 · One-problem workflow

~~~mermaid
sequenceDiagram
    participant U as Runner
    participant A as ReasoningAgent
    participant S as Skill Router
    participant M as Solver Ensemble
    participant V as Verifier
    participant N as Normalizer
    participant E as Evaluator
    U->>A: solve(problem, metadata)
    A->>S: detect subject / difficulty / answer schema
    S-->>A: Route + action plan
    A->>M: generate independent candidates
    M-->>A: answers + reasoning traces
    A->>V: attack top candidates
    V-->>A: ACCEPT / REJECT + report
    A->>N: extract, canonicalize, compare
    N-->>A: stable answer key
    A-->>U: final_response + trace
    U->>E: evaluate JSONL outputs
    E-->>U: accuracy + error report
~~~

## 截图 · Screenshots

<table>
<tr><td><img src="https://placehold.co/720x420/111827/67e8f9?text=Trace+Viewer+placeholder" alt="Trace viewer placeholder" /></td><td><img src="https://placehold.co/720x420/111827/a7f3d0?text=Error+Report+placeholder" alt="Error report placeholder" /></td></tr>
<tr><td align="center"><sub>Trace viewer · 占位图</sub></td><td align="center"><sub>Error report · 占位图</sub></td></tr>
</table>

## 安装 · Installation

- Python **3.10+**
- 一个 OpenAI-compatible Chat Completions endpoint
- <code>INTERN_API_KEY</code>（本地运行时必需）

~~~bash
git clone https://github.com/Severus-C/NaviMath.git
cd navimath
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
~~~

## 快速开始 · Quick start

~~~python
from user_agent import ReasoningAgent
agent = ReasoningAgent(client=official_client)
result = agent.solve(problem="Find the largest integer n such that ...", metadata={"contest": "AIME", "answer_type": "integer"})
print(result["final_response"])
~~~

### 本地 JSONL runner

~~~powershell
$env:INTERN_API_KEY = "<your-key>"
python scripts\run_local.py --input_file data\dev.jsonl --output_dir outputs\dev
~~~

<code>LOCAL_MAX_CONCURRENCY</code> 默认为 <code>8</code>，可按限流策略调整。

### 离线评测与错因报告 · Evaluation

~~~powershell
python scripts\eval_agent_on_dataset.py --dataset data\public_math_aime_500.jsonl --limit 10 --keep-trace
~~~

评测会生成 prediction JSONL、summary JSON、<code>*_error_report.json</code> 和 <code>*_error_report.md</code>。

## 示例 · Examples

<details><summary>标准返回值 · Standard result</summary>

~~~json
{
  "final_response": "038",
  "trace": [
    {"step": "route", "content": {"subject": "number_theory"}},
    {"step": "consensus_lock", "content": {"answer": "38"}},
    {"step": "select_final_response", "content": {"normalized": "38"}}
  ]
}
~~~
</details>

<details><summary>答案规范化 · AnswerNormalizer</summary>

~~~python
from agent.answer_normalizer import AnswerContext, AnswerNormalizer
n = AnswerNormalizer()
assert n.equivalent(r"\frac{2}{4}", "0.5")
assert n.equivalent(r"\{3, 1, 2\}", "{1,2,3}", AnswerContext(answer_type="set"))
assert n.canonicalize(r"\left[0,\infty\right)", AnswerContext(answer_type="interval")) == "[0,infinity)"
assert n.format_aime("38") == "038"
~~~
</details>

## 项目结构 · Project structure

~~~text
NaviMath/
├── user_agent.py                    # Stable competition entrypoint
├── agent/
│   ├── reasoning_agent.py            # Route → solve → verify → judge
│   ├── agent_utils.py                # Compatibility API
│   ├── answer_normalizer.py          # Schema-aware canonicalization
│   ├── error_analysis.py             # Root-cause diagnostics
│   └── llm_client.py                 # OpenAI-compatible client
├── scripts/
│   ├── run_local.py                  # Concurrent, resumable runner
│   ├── eval_agent_on_dataset.py      # Accuracy + reports
│   └── build_public_math_dataset.py  # Dataset preparation
├── data/                             # Local benchmark JSONL files
├── eval_outputs/                     # Evaluation artifacts
├── tests/                            # Unit and behavior tests
└── docs/                             # Research notes
~~~

## 配置 · Configuration

| 变量 · Variable | 默认值 · Default | 用途 · Purpose |
|---|---:|---|
| <code>INTERN_API_KEY</code> | — | API key；未设置时客户端会明确报错 |
| <code>INTERN_API_BASE</code> | <code>https://chat.intern-ai.org.cn/api/v1/chat/completions</code> | Chat Completions endpoint |
| <code>INTERN_MODEL</code> | <code>intern-s2-preview</code> | 模型名称 · model name |
| <code>LOCAL_MAX_CONCURRENCY</code> | <code>8</code> | 本地 runner 并发上限 |

## 基准评测 · Benchmark

评测脚本支持 AIME-style JSONL，并输出按题型、答案类型和 contest 的统计。当前小样本结果用于展示报告格式，不代表最终排行榜成绩。

| Run | Samples | Accuracy | Diagnostic coverage | Top root cause |
|---|---:|---:|---:|---|
| <code>aime_wrong4_v6_analyzed</code> | 4 | 50.0% | 100.0% | verifier misjudgment (2) |

~~~text
prediction JSONL → summary JSON → error_report.json + error_report.md
~~~

~~~powershell
python scripts\eval_agent_on_dataset.py --predictions eval_outputs\aime_wrong4_v6.jsonl --output eval_outputs\reproduced.jsonl --summary eval_outputs\reproduced_summary.json
~~~

## Roadmap

| 状态 | Milestone | 目标 |
|---|---|---|
| ✅ | Error analysis | 格式、路由、共识、verifier、judge 和 LaTeX 失败归因 |
| ✅ | AnswerNormalizer v1 | 集合、区间、矩阵、向量、多答案与 AIME 格式 |
| 🚧 | ToolVerify | SymPy、随机代入、方程根、导数/积分/极限、留数校验 |
| 🚧 | Skill distillation | 题型模板、陷阱和 verifier checklist |
| 📅 | Adaptive budget | 依据难度、候选一致性和 verifier 信号动态早停 |
| 📅 | Learned navigator | 从日志学习题型到动作预算的 bandit/RLoT 策略 |
| 📅 | Final-round memory | MCP、persistent memory、proof graph store |

## FAQ

<details><summary><b>必须使用 Intern-S API 吗？</b></summary>
不必须。<code>ReasoningAgent</code> 接收 <code>client</code> 对象；实现兼容的 <code>chat(messages, temperature, max_tokens)</code> 即可接入其他 provider。
</details>

<details><summary><b>AnswerNormalizer 会做通用符号推理吗？</b></summary>
不会。它负责确定性的语法规范化；通用代数等价性属于后续 ToolVerify。
</details>

## Contributing

欢迎提交 issue、benchmark、Skill 和 verifier 改进：
1. Fork 并创建分支：<code>feat/your-improvement</code>。
2. 为行为变化增加测试或最小 JSONL 样例。
3. 运行 <code>python -m unittest discover -s tests -v</code>。
4. 附上一小段评测 summary 与 error report。
5. PR 中说明准确率、调用预算和已知回归。

请不要提交 API key、真实用户数据或敏感 trace。

## Citation

~~~bibtex
@software{navimath,
  title  = {NaviMath: Verifiable Multi-Candidate Reasoning for Competition Mathematics},
  author = {NaviMath Contributors},
  year   = {2026},
  url    = {https://github.com/Severus-C/NaviMath}
}
~~~

## Acknowledgements

- [InternLM](https://github.com/InternLM) 与 Intern-S API 生态
- [Lagent](https://github.com/InternLM/lagent) agent tooling
- AIME、HARP 及公开竞赛数学数据社区
- 所有提交题目、评测 trace 和错误报告的贡献者

## License

当前仓库尚未包含正式的 `LICENSE` 文件；发布前请在此处明确选择并提交许可证。数据集与第三方服务可能拥有各自的许可和使用条款，请分别核对。

<hr />
<div align="center"><sub>Built for careful reasoning, measurable progress, and fewer “会做但输错格式”的答案.</sub><br /><sub>Made with ☕, exact arithmetic, and a healthy suspicion of confident wrong answers.</sub></div>

---

# English version

NaviMath is a competition-math reasoning agent built around a measurable loop: route the problem, generate independent candidates, attack them, normalize the answer, and turn failures into actionable diagnostics.

## What makes it different

- **Structured routing:** subject Skills, difficulty estimation, proof mode, and action plans.
- **Multi-candidate solving:** independent domain roles reduce single-path failures.
- **Adversarial verification:** attack reports, verdicts, refinement, and consensus locking.
- **Schema-aware answers:** exact numeric forms, sets, intervals, vectors, matrices, multiple answers, and AIME zero-padding.
- **Evaluation as a product feature:** JSONL traces, accuracy breakdowns, root-cause evidence, and Markdown reports.
- **Stable integration surface:** import <code>ReasoningAgent</code> from <code>user_agent.py</code>.

## English quick start

~~~bash
git clone https://github.com/Severus-C/NaviMath.git
cd navimath
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export INTERN_API_KEY="<your-key>"
python scripts/run_local.py --input_file data/dev.jsonl --output_dir outputs/dev
~~~

~~~python
from user_agent import ReasoningAgent
agent = ReasoningAgent(client=official_client)
result = agent.solve(problem="Find the largest integer n such that ...", metadata={"contest": "AIME", "answer_type": "integer"})
assert result["final_response"]
~~~

The Chinese documentation above contains the architecture, workflow, screenshots, feature matrix, configuration reference, examples, benchmark protocol, roadmap, FAQ, contribution guide, citation, acknowledgements, and license notes.

## English contribution checklist

Run the unit suite, include a minimal regression case, attach a benchmark summary, and never commit API keys or sensitive traces. For architectural changes, update the Mermaid diagram and roadmap in this README.

<div align="center"><sub>Questions, ideas, or a new Skill? Open an issue and bring evidence.</sub></div>
