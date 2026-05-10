# Skill Retrieval Augmentation for Agentic AI

Weihang $\mathbf { S u } ^ { * 1 }$ , Jianming Long1, Qingyao $\mathbf { A } \mathbf { i } ^ { \dagger 1 }$ , Yichen Tang1, Changyue Wang1, Yiteng $\mathbf { T } \mathbf { u } ^ { 1 }$ , Yiqun Liu1

1Department of Computer Science and Technology, Tsinghua University

§ GitHub: https://github.com/oneal2000/SR-Agents

Hugging Face: https://huggingface.co/datasets/WeihangSu/SRA-Bench

# Abstract

As large language models (LLMs) evolve into agentic problem solvers, they increasingly rely on external, reusable skills to handle tasks beyond their native parametric capabilities. In existing agent systems, the dominant strategy for incorporating skills is to explicitly enumerate available skills within the context window. However, this strategy fails to scale: as skill corpora expand, context budgets are consumed rapidly, and the agent becomes markedly less accurate in identifying the right skill. To this end, this paper formulates Skill Retrieval Augmentation (SRA), a new paradigm in which agents dynamically retrieve, incorporate, and apply relevant skills from large external skill corpora on demand. To make this problem measurable, we construct a large-scale skill corpus and introduce SRA-Bench, the first benchmark for decomposed evaluation of the full SRA pipeline, covering skill retrieval, skill incorporation, and end-task execution. SRA-Bench contains 5,400 capability-intensive test instances and 636 manually constructed gold skills, which are mixed with web-collected distractor skills to form a large-scale corpus of 26,262 skills. Extensive experiments show that retrieval-based skill augmentation can substantially improve agent performance, validating the promise of the paradigm. At the same time, we uncover a fundamental gap in skill incorporation: current LLM agents tend to load skills at similar rates, regardless of whether a gold skill is retrieved or whether the task actually requires external capabilities. This shows that the bottleneck in skill augmentation lies not only in retrieval but also in the base model’s ability to determine which skill to load and when external loading is actually needed. These findings position SRA as a distinct research problem and establish a foundation for the scalable augmentation of capabilities in future agent systems. All the code and data are released at the following GitHub link: https://github.com/oneal2000/SR-Agents.

# 1 Introduction

Recent advancements in Large Language Models (LLMs) have catalyzed a paradigm shift toward Agentic AI, where models are evolving from passive text generators into active problem solvers capable of reasoning, planning, tool calling, and interacting with external environments [55, 58, 28, 29]. As these agents are expected to solve broader and more complex tasks, their internal parametric knowledge is no longer sufficient to support robust performance [54, 12]. Instead, tackling such diverse and open-ended tasks increasingly relies on externalized, reusable capabilities that extend an agent’s competence beyond the base model’s intrinsic capacity. In many emerging systems (e.g., OpenClaw [32]), these capabilities are packaged as skills: modular capability packages that help

an agent solve recurring classes of problems, often encompassing natural-language instructions, invocation conditions, tool-usage procedures, executable code, and auxiliary resources [15, 16].

However, the prevailing paradigm for equipping agents with such capabilities relies on a fundamentally unscalable mechanism: explicit in-context skill injection [32, 20, 2]. Current frameworks typically construct prompts that enumerate available skills, often compressing them into compact metadata or instructional summaries (e.g., the SKILL.md files used in OpenClaw [32]), and then require the LLM to evaluate numerous candidates in context to identify and load relevant skills on demand based on user instructions. While this design is intuitive and aligns naturally with existing agentic workflows, it becomes increasingly impractical in real-world deployments as the scale of skill libraries continues to grow [30]. In the past few months, the ecosystem of available skills has been expanding explosively: as of April 26, 2026, platforms such as SkillsMP [30] host more than one million distinct skills, and agents are expected to maintain large and lifelong skill libraries that continue to grow [1, 50]. Under this setting, the conventional practice of exposing all candidate skills in context begins to collapse under two compounding bottlenecks: the hard architectural limits of context windows, and the sharp degradation in reasoning and selection accuracy when the model is confronted with a massive volume of skills. Consequently, dynamically retrieving and injecting the right capability from a vast, out-of-context skill collection has emerged as a critical research problem.

To address this scalability failure, we formulate Skill Retrieval Augmentation (SRA), a new paradigm for augmenting agents with external capabilities at scale. Rather than relying on a small, fixed set of visible skills in the prompt, SRA treats skills as entries in a large external capability corpus and requires agents to retrieve, load, and apply relevant skills on demand. Under this paradigm, we propose Skill Retrieval Augmented Agents (SR-Agents), which dynamically retrieve and use relevant skills from large-scale skill corpora to expand their problem-solving capabilities. SRA is closely related to Retrieval-Augmented Generation (RAG), but it is not simply RAG with a different retrieval target. In classical knowledge-centric RAG, retrieved items are primarily declarative evidence used to ground generation. In contrast, SRA retrieves executable capabilities that augment the agent’s functional competence, rather than merely providing declarative knowledge to support generation. Accordingly, retrieval in SRA should be evaluated not only by semantic relevance, but also by downstream utility: whether the retrieved candidate set contains the skills relevant to the current task, whether those skills can be correctly incorporated into the agent’s problem-solving process, and whether their application ultimately improves end-task performance. This difference fundamentally changes the problem structure. Beyond skill retrieval itself, SRA introduces a multistage pipeline with three tightly coupled components: skill retrieval, which asks whether the system can identify relevant skills for a given user request from a large external corpus; skill incorporation, which asks whether the agent can correctly recognize, organize, and invoke the truly useful skills among retrieved candidates without being distracted or overwhelmed by irrelevant ones; and skill application, which asks whether incorporated skills are actually leveraged during task solving in ways that improve agent behavior and end-task success. Together, these components define a broader research agenda for scalable capability augmentation in agent systems. More broadly, this paradigm opens up new directions in skill indexing and organization, quality control over heterogeneous skills, and feedback-driven skill debugging, refinement, and lifelong accumulation.

To make this paradigm scientifically tractable, we build a large-scale skill corpus and introduce SRA-Bench, the first benchmark specifically designed for studying the SRA paradigm. Each instance in SRA-Bench is annotated with a user query, a ground-truth answer, and the corresponding gold skill(s), enabling fine-grained, decomposed evaluation across the full SRA pipeline. Rather than asking whether an agent ultimately succeeds, this benchmark allows us to diagnose three distinct questions: whether the system can retrieve the right skills from a large external corpus, whether the agent can correctly identify and incorporate the truly useful skills among retrieved candidates, and whether such retrieved skills can be translated into improved behavior and end-task performance. In this sense, the contribution of SRA-Bench is not merely to introduce a new benchmark, but to formulate skill retrieval augmentation as a concrete research problem that can be clearly defined, rigorously evaluated, and systematically analyzed.

Through extensive experiments on SRA-Bench, we uncover several key empirical findings about scalable skill augmentation. First, even a simple SRA pipeline that retrieves a single skill using a general-purpose retriever and injects it into the context can already improve strong LLM agents over their skill-free counterparts, establishing the practical promise of our proposed paradigm. Beyond this positive result, however, our experiments reveal that scalable skill augmentation is bottlenecked not

![](images/e8fa4c3b660f630a94d465a12df2add7eee1e5590e8893858516618bd2f5c39e.jpg)  
Figure 1: An illustration of the Skill Retrieval Augmentation (SRA) paradigm. The agent retrieves candidate skills from a large external skill corpus, selectively incorporates useful skills into context, and applies them for downstream reasoning and acting. Black arrows denote the standard SRA workflow, while blue arrows represent iterative skill retrieval during reasoning and acting.

only by retrieval quality, but by a previously underappreciated variable: whether the agent chooses to load external skills at all. We find that skill-loading behavior is highly model-dependent: using the same skill corpus and retriever, different models exhibit dramatically different loading rates, with no monotonic trend that larger models behave more rationally or robustly. Moreover, agents attempt to load skills at nearly identical rates regardless of whether the gold skill is actually present among the retrieved candidates, revealing a profound disconnect between successful retrieval and effective utilization. More importantly, agents are no more likely to invoke a skill on tasks that genuinely require external capability than on tasks they can already solve natively, suggesting a fundamental absence of need-aware skill invocation. Taken together, these findings show that scalable skill augmentation is not merely a retrieval problem, but a broader challenge of controlled skill exposure, need-aware incorporation, and reliable application, thereby motivating SRA as a distinct research agenda for agent systems.

To summarize, our contributions are as follows:

• We propose Skill Retrieval Augmentation (SRA), a new paradigm in which LLM-based agents retrieve and operationalize reusable external skills from large-scale skill corpora rather than relying on small, fixed skill sets directly placed in context.   
• We build a large-scale skill corpus and introduce SRA-Bench, a benchmark resource with query, answer, and gold-skill annotations that supports decomposed evaluation of skill retrieval, skill incorporation, and end-to-end task solving.   
• We establish SR-Agents as a baseline family for studying the SRA pipeline and conduct a systematic empirical analysis across models, retrievers, and task domains.   
• We provide empirical findings that clarify both the promise and the bottlenecks of scalable skill use, showing that while external skills can substantially improve agent performance, effective access to capabilities requires advances not only in retrieval but also in skill incorporation and application.

# 2 Problem Formulation

Building upon the conceptual foundation above, we now formalize the core objects and operational pipeline of Skill Retrieval Augmentation (SRA). Our formulation characterizes scalable skill augmentation as a multi-stage process that separates whether an agent can retrieve relevant skills, correctly incorporate them into its active problem-solving state, and ultimately translate them into improved task performance.

# 2.1 Agent Skills and Skill Corpus

We begin by defining the basic unit of SRA: the agent skill. Unlike ordinary retrieved documents, which primarily provide declarative evidence for grounded generation, or standalone tool APIs, which expose only isolated callable interfaces, a skill is a reusable capability package that enables an agent to solve a recurring class of problems. A standard skill typically contains a name, a short natural-language description, detailed usage instructions, invocation conditions, procedural guidance, executable code, and auxiliary resources, thereby exposing not only what capability is available, but also when and how it should be applied.

Formally, let $\mathcal { C } = \{ s _ { 1 } , s _ { 2 } , . . . , s _ { N } \}$ denote a skill corpus containing $N$ skills. Each skill $s _ { i } \in \mathcal { C }$ is represented as

$$
s _ {i} = \left(n _ {i}, r _ {i}, c _ {i}, \pi_ {i}\right), \tag {1}
$$

where $n _ { i }$ is the name of the skill, serving as its compact semantic identifier, $r _ { i }$ is a short description summarizing the capability and intended use of the skill, and $c _ { i }$ is the main content, which may include natural-language instructions, usage constraints, procedural guidance, and other languagevisible artifacts exposed to the agent. The executable payload $\pi _ { i }$ includes code, tools, or other operational resources that realize the capability in an external environment.

In practical systems, these components may correspond to artifacts such as a skill title, a one-line summary, SKILL.md, scripts, and static assets. More generally, a skill is any modular unit that couples a natural-language interface with an executable capability.

# 2.2 Skill Retrieval Augmentation

Given a user query $q$ and a large external skill corpus $\mathcal { C }$ , the objective of SRA is to augment an agent with relevant external capabilities retrieved from a large skill corpus on demand, rather than relying on a pre-exposed, fixed set of skills enumerated directly in the context. We formalize this process as a three-stage pipeline.

Skill Retrieval. A retriever $R$ first maps the user query $q$ and the skill corpus $\mathcal { C }$ to a ranked list of candidate skills:

$$
\mathcal {L} _ {k} = R (q, \mathcal {C}) = \left[ s ^ {(1)}, s ^ {(2)}, \dots , s ^ {(k)} \right], \quad s ^ {(j)} \in \mathcal {C}, k \ll N. \tag {2}
$$

The retrieved candidates are ranked by their estimated relevance to the current query, with earlier positions indicating higher relevance. The role of this stage is to reduce a massive external capability space into a manageable ranked list of candidate skills that may be useful for the current task.

Skill Incorporation. Given the retrieved candidates $\mathcal { L } _ { k }$ , the agent must then determine whether any external skill should be used for the current task, and if so, which retrieved skills should be incorporated into the active problem-solving state and in what form. We denote this stage as

$$
\widetilde {\mathcal {S}} = G (q, \mathcal {L} _ {k}; \mathcal {M}), \tag {3}
$$

where $\mathcal { M }$ is the underlying base language model and $\widetilde { s }$ denotes the skill representations that are actually prepared and made available for downstream task solving. These representations may correspond to a selected subset of retrieved skills, or to transformed variants derived from them, such as rewritten, compressed, restructured, or model-adapted forms. Accordingly, skill incorporation is broader than merely selecting a subset from the retrieved list: it concerns whether and how external capabilities are converted into a form that the agent can subsequently use. Importantly, $\widetilde { s }$ may be empty, either because the agent determines that its parametric capability is sufficient for the current task, or because none of the retrieved skills is deemed sufficiently relevant or useful to incorporate.

Skill Application. Finally, conditioned on the incorporated skills $\tilde { s }$ , the agent applies the corresponding capabilities during task solving and produces a final response:

$$
\hat {A} = F (q, \tilde {\mathcal {S}}; \mathcal {M}). \tag {4}
$$

Table 1: Overview of the six source datasets used to construct SRA-Bench. Skill Mapping indicates whether each benchmark instance is associated with one gold skill or multiple gold skills.   

<table><tr><td>Dataset</td><td>Capability Type</td><td>#Inst.</td><td>#Skills</td><td>Skill Mapping</td><td>Evaluation</td></tr><tr><td>THEOREMQA [4]</td><td>Theorem Application</td><td>747</td><td>320</td><td>Single</td><td>Rule-Based</td></tr><tr><td>LOGICBENCH [22]</td><td>Logical Reasoning Patterns</td><td>760</td><td>19</td><td>Single</td><td>Rule-Based</td></tr><tr><td>TOOLQA [60]</td><td>Tool-Use Workflows</td><td>1,430</td><td>14</td><td>Single</td><td>Rule-Based</td></tr><tr><td>MEDCALC-BENCH [13]</td><td>Medical Calculators</td><td>1,100</td><td>55</td><td>Single</td><td>Rule-Based</td></tr><tr><td>CHAMP [17]</td><td>Mathematical Concepts</td><td>223</td><td>89</td><td>Multi</td><td>Rule-Based</td></tr><tr><td>BIGCODEBENCH [61]</td><td>Software Libraries</td><td>1,140</td><td>139</td><td>Multi</td><td>Execution</td></tr></table>

Importantly, successful incorporation does not guarantee successful application: the relevant skill may be present in the agent’s active problem-solving state yet still fail to improve performance if the model cannot properly operationalize it during downstream task solving. This stage, therefore, assesses whether the incorporated skills are leveraged to improve downstream behavior, including whether the agent follows them correctly, invokes them at the right time, integrates them into its reasoning process, and adapts its behavior accordingly.

# 3 Benchmark Construction

SRA-Bench is built from three components: (1) a set of capability-intensive test instances, (2) a set of manually constructed gold skills and corresponding instance-level annotations that associate each benchmark instance with one or more relevant skills, and (3) a large external skill corpus formed by inserting these gold skills into a noisy web-collected skill collection containing realistic distractors. This design makes the SRA setting empirically tractable by transforming skill augmentation into a decomposable evaluation problem: a system must retrieve the correct capability from a large, noisy skill corpus, correctly incorporate it, and ultimately use it to improve downstream task performance. We introduce the construction process in three subsections: source dataset selection and test instance curation (§3.1), gold skill construction (§3.2), and large-scale skill corpus collection (§3.3).

# 3.1 Source Dataset Selection and Test Instance Curation

We begin the construction of SRA-Bench from the instance side: a collection of test problems that genuinely require reusable external capabilities rather than mere factual question answering (QA). To this end, we curate test instances from six existing benchmarks: THEOREMQA [4], LOG-ICBENCH [22], TOOLQA [60], MEDCALC-BENCH [13], CHAMP [17], and BIGCODEBENCH [61]. Together, these datasets cover diverse forms of capability use, including theorem-based reasoning, formal logic, tool-interleaved question answering, medical calculation, competition mathematics, and code generation. The resulting benchmark contains 5,400 test instances associated with 636 unique gold skills, as summarized in Table 1. Rather than starting from a pre-existing skill repository and searching for compatible tasks, we construct SRA-Bench in reverse: we first identify capabilityintensive problem instances and then manually annotate the reusable skill(s) associated with them using structured signals from the source datasets.

Since existing benchmarks do not directly provide reusable skill annotations, the selection of source datasets is critical. We therefore select candidate datasets according to three criteria. First, the dataset should provide structured signals that reveal shared problem-solving patterns across multiple instances, enabling human annotators to reliably infer reusable capabilities. Second, the associated capability should encode how to solve a class of problems, rather than directly revealing the answer to a particular instance. Third, the final task evaluation should remain objective and reproducible, so that downstream performance can be measured reliably. Together, these criteria ensure that the source datasets not only contain capability-intensive problems but also provide the annotation basis needed for constructing meaningful gold-skill supervision for SRA.

Under this design, each selected source instance is transformed into an SRA-Bench example consisting of a user query, a ground-truth answer, and one or more gold skills. Since reusable skill annotations are not available in existing benchmarks, we construct them manually using structured signals provided by the source datasets. For example, theorem names in THEOREMQA, logic patterns in LOGICBENCH, calculator types in MEDCALC-BENCH, and library names in BIGCODEBENCH

expose recurring capability structure across instances. These dataset-specific signals do not constitute gold skills themselves; instead, they guide annotators in abstracting instance-level problem-solving patterns into standardized reusable skills. Section 3.2 details this annotation process.

# 3.2 Gold Skill Construction

The source datasets provide structured annotations that indicate the capabilities involved in each instance, but they do not directly provide reusable skills. For example, annotations such as theorem names, logic patterns, calculator names, concept IDs, or library names identify what capability is relevant, but they do not by themselves tell an agent when to use that capability or how to apply it for downstream tasks. Our goal in this stage is therefore to transform these source-side signals into explicit gold skills: reusable skill artifacts that can be retrieved and used under the SRA setting. Concretely, we construct one gold skill for each annotation category in the source datasets. Depending on the dataset, a gold skill may correspond to a theorem, a reasoning pattern, a tool-use workflow, a medical calculator, a mathematical concept, or a software library. Although these capability types differ in form, they can all be represented in the same way: as standalone skill artifacts that describe the applicability conditions and problem-solving procedure for a recurring class of tasks. Each benchmark instance is then associated with one or more gold skills according to its source annotation(s), forming the ground-truth skill supervision used in SRA-Bench. Since the available supervision signals differ markedly across source benchmarks, we defer full dataset-specific construction details and examples to Appendix A.

We build each gold skill through a two-stage process: LLM drafting followed by expert revision. For each annotation category, we first collect the available source materials, including the source dataset definition or description, representative instances annotated with this category, and external references when needed. We then provide these materials to an LLM to generate an initial draft skill. This draft serves only as a starting point. It is then manually revised into the final gold skill, since draft skills often remain too tied to specific examples, miss important conditions or edge cases, or contain factual and procedural errors. Across datasets, this revision is guided by three shared principles. At the same time, different source benchmarks exhibit different recurring weaknesses in LLM drafts and therefore require additional dataset-specific refinements, which we detail in Appendix A. These shared principles are as follows. First, generality: the final skill should describe a reusable method rather than a benchmark-specific template or paraphrase of example instances. Second, correctness: formulas, reasoning procedures, tool usage, and executable components must be checked and revised against reliable references. Third, leakage control: the skill should not reveal benchmark answers or encode shortcuts that trivialize evaluation. To enforce this, overlapping examples are replaced with newly constructed ones, benchmark-specific constants are removed, and the final skill is written to preserve a clear application gap: even after retrieving the correct skill, the agent must still interpret the query, identify the appropriate case, extract instance-specific variables, and correctly execute the procedure.

The finalized gold skills are stored as standardized Markdown artifacts with a skill name, a short description, and procedural content that describes what the capability is, when it applies, how it should be used, and which common pitfalls to avoid. For capability types that inherently require execution, we additionally attach runnable resources such as Python implementations of medical calculators. This representation makes gold skills not merely annotation labels, but realistic skill artifacts that can be mixed into a large external corpus.

# 3.3 Skill Corpus Collection

To evaluate SRA under realistic large-scale conditions, gold skills must be retrieved from a noisy external corpus rather than presented in isolation. We therefore mix the 636 gold skills into a larger collection of 25,626 publicly available skill documents collected from the web, covering domains such as programming, data science, system administration, and general productivity. This yields a final skill corpus of 26,262 skills, of which only $2 . 4 \%$ are gold. To construct this background corpus, we follow a public-ecosystem crawling pipeline. Specifically, we collect publicly available skills from open web sources and community skill repositories, such as GitHub, Skills.sh, and the Hugging Face Hub. For each entry, we retrieve its associated skill documents or repository contents, and retain only those that expose sufficiently self-contained skill descriptions for standalone use. We further remove inaccessible, malformed, or duplicate entries and normalize the remaining skills into a

unified document format for indexing and retrieval. The resulting corpus is intended to approximate a realistic skill ecosystem in which high-value skills are sparse, heterogeneous, and mixed with many irrelevant or weakly related candidates.

# 4 Study Design and Experimental Setup

In this section, we present the study design of our systematic empirical study of scalable skill augmentation. Rather than treating SRA as merely a new benchmark setting, we aim to understand the empirical factors that govern whether retrieved skills can actually improve agent performance. We first present our study design and research questions, and then describe the experimental setup, including benchmarks, baselines, metrics, and implementation details.

# 4.1 Study Design

The Skill Retrieval Augmentation (SRA) paradigm introduces a multi-stage pipeline in which an agent must retrieve, incorporate, and apply external skills to solve tasks that may exceed its native parametric capabilities. While the formulation is conceptually clean, each stage introduces distinct challenges and potential failure modes, and the interactions among stages further compound the difficulty of building effective SR-Agents. Therefore, before presenting experimental results, we first analyze the structure of the SRA problem and identify the key questions that must be answered to understand where the pipeline succeeds or fails.

At the highest level, the first question is whether SRA is useful at all. If retrieving and injecting external skills does not improve performance over skill-free baselines, then the practical value of the paradigm would be fundamentally limited. At the same time, even if SRA is beneficial in principle, real retrieval is inevitably noisy: retrieved candidate sets will contain distractors, partially relevant skills, and misleading entries in addition to useful ones. This means that the viability of SRA depends not only on whether skill augmentation helps, but also on whether SR-Agents remain effective under noisy retrieval conditions. Therefore, we first focus on the following two questions:

RQ1: Does the SRA paradigm improve agent performance over skill-free baselines, and how do different SR-Agents configurations compare in terms of end-task effectiveness?

RQ2: How robust are current SR-Agents to retrieval noise? Specifically, can they still identify and effectively utilize relevant skills when candidate sets contain irrelevant distractors, and how does robustness vary across different SR-Agents designs?

The next question concerns the retrieval stage itself. Since skills are heterogeneous capability packages rather than text passages that primarily convey factual world knowledge, it is unclear whether existing retrieval methods transfer effectively to this setting, or whether lexical and dense retrievers exhibit different strengths for skill retrieval. Moreover, retrieval quality alone does not fully determine end-task success: SRA is a coupled pipeline, and gains from better retrieval may be amplified by downstream incorporation and application. Therefore, beyond evaluating retrieval in isolation, we must also ask how strongly retrieval quality actually affects end-to-end performance. Accordingly, we study the following two questions:

RQ3: How effective are existing retrieval methods at identifying relevant skills from large-scale skill corpus, and how do classical lexical matching approaches compare with dense retrieval methods in the skill retrieval setting?

RQ4: To what extent does retrieval quality influence end-to-end SR-Agents performance? Does better retrieval consistently lead to better task outcomes, or are the gains mediated or attenuated by downstream incorporation and application stages?

Finally, even strong retrieval does not guarantee effective skill use, because the language model must properly incorporate the retrieved candidates. This raises two closely related questions about skill-loading behavior. First, a rational agent should be relevance-aware: it should be more likely to load skills when the retrieved candidate set actually contains a relevant skill. Second, a rational agent should be need-aware: it should preferentially load external skills for tasks that exceed its native capability, while refraining from unnecessary loading for tasks it can already solve on its own. These

two properties determine whether retrieval success translates into effective utilization. Therefore, we finally focus on the following two questions:

RQ5: During skill incorporation, can current LLMs distinguish between retrieved candidate sets that contain a relevant (gold) skill and those that do not? Does the presence of a gold skill among retrieved candidates meaningfully influence the agent’s skill-loading behavior?

RQ6: Do current LLMs exhibit need-aware skill-loading behavior during incorporation? Specifically, are agents more inclined to load external skills for tasks that exceed their native capabilities than for tasks they can already solve without external augmentation?

Taken together, these six questions cover the major uncertainty sources of the SRA pipeline: whether the paradigm is beneficial at all (RQ1), whether it remains effective under noisy retrieval conditions (RQ2), how well skill retrieval itself can be solved (RQ3), whether retrieval improvements translate into downstream gains (RQ4), and whether current LLMs incorporate skills in a relevance-aware and need-aware manner (RQ5–RQ6). By answering these questions systematically, we aim to provide a more diagnostic understanding of scalable skill augmentation and to identify the key bottlenecks that currently limit SR-Agents’ performance.

# 4.2 Experimental Setup

To provide a consistent basis for answering the research questions in $\ S 4 . 1$ , we standardize the main experimental components shared across our study, including benchmarks, evaluated models, skill-use strategies, and evaluation metrics. Since this paper presents a paradigm and systematic study rather than a single method under a single protocol, experiment-specific variations are deferred to the corresponding subsections.

Benchmarks and Skill Corpus. We conduct experiments on SRA-Bench, which comprises six capability-intensive benchmarks: TheoremQA [4], LogicBench [22], ToolQA [60], CHAMP [17], MedCalc-Bench [13], and BigCodeBench [61]. These datasets cover diverse task settings, including mathematical reasoning, formal logic, tool use, and code generation. Each instance is paired with annotated gold skill(s) and evaluated against a shared external skill corpus, enabling decomposed analysis of retrieval, incorporation, and end-task execution.

Evaluation Metrics. We evaluate the SRA pipeline from both retrieval and end-task perspectives. For skill retrieval, we report Recall $@ K$ and nDCG@K. For end-task performance, each benchmark follows its standard evaluation protocol: rule-based exact match or accuracy for reasoning and QA datasets, and pass $@ 1$ based on unit-test execution for BigCodeBench.

Selected Models. We evaluate six LLMs from three open-weight families: Qwen3-4B, Qwen3- 32B, and Qwen3-235B-A22B [57]; Llama-3.1-8B-Instruct [8] and Llama-3.3-70B-Instruct [18]; and Mistral-Small-3.1-24B-Instruct-2503 [19]. For behavioral analyses of skill-loading dynamics (RQ5 and RQ6), we also include proprietary frontier models GLM-5.1 [59] and GPT-5.4 [21]. All models are served with a 128K-token context window and sampling temperature 0.7.

Skill-Use Strategies. We study both non-augmented and skill-augmented inference settings. The non-augmented baseline is LLM Direct, where the model solves the task using only its parametric knowledge. The upper-bound setting is Oracle Skill, where the annotated gold skill(s) are directly provided to the model. For skill-retrieval-augmented inference, we consider three representative strategies. Full-Skill Injection injects the complete content of the top- $k$ retrieved skills into the task context. This represents the most direct use of retrieved skills, but also exposes the model to all retrieved content, including potentially irrelevant skills. LLM Selection instead exposes only metadata for the retrieved candidates, asks the model to select the single most relevant skill, and then injects the full content of the selected skill. Progressive Disclosure follows an OpenClaw-style design, where the model is given a compact skill catalog and can selectively load full skill content on demand during inference. The exact prompt templates used for these skill-use baselines are provided in Appendix B.

Implementation Details. When a skill is injected, its full content is prepended before the original task prompt. In selection-based settings, the model is given a list of candidates, each with a skill name

Table 2: End-task performance $( \% )$ of different models and skill-use methods on SRA-Bench. Oracle is included for comparison. The average is computed over all instances. For retrieval-based methods, Full-Skill Injection uses the top-1 skill retrieved by BM25, while LLM Selection and Progressive Disclosure select skills from the top-50 BM25 candidates. The best results are in bold, and the second-best results are underlined. Statistical significance is assessed using a paired t-test, where ∗ indicates performance significantly worse than the bold method at $p < 0 . 0 5$ .   

<table><tr><td>Model</td><td>Method</td><td>TheoremQA</td><td>LogicBench</td><td>ToolQA</td><td>CHAMP</td><td>MedCalc</td><td>BigIntCodeBench</td><td>Average</td></tr><tr><td rowspan="5">Llama-3.1-8B</td><td>LLM Direct</td><td>32.4*</td><td>54.6*</td><td>16.7*</td><td>22.4*</td><td>26.9*</td><td>32.3</td><td>29.8*</td></tr><tr><td>Oracle Skill</td><td>49.4</td><td>69.5</td><td>23.3</td><td>40.8</td><td>62.0</td><td>35.2</td><td>44.5</td></tr><tr><td>Full-Skill Injection</td><td>36.5*</td><td>58.3*</td><td>13.6*</td><td>27.8*</td><td>36.7*</td><td>34.2</td><td>32.7*</td></tr><tr><td>LLM Selection</td><td>35.1*</td><td>53.8*</td><td>19.4*</td><td>24.7*</td><td>57.0*</td><td>32.1*</td><td>37.0*</td></tr><tr><td>Progressive Disclosure</td><td>36.9*</td><td>50.0*</td><td>16.4*</td><td>25.1*</td><td>59.6</td><td>31.4*</td><td>36.3*</td></tr><tr><td rowspan="5">Llama-3.3-70B</td><td>LLM Direct</td><td>59.6*</td><td>60.5*</td><td>30.9*</td><td>56.5*</td><td>53.9*</td><td>45.0*</td><td>47.8*</td></tr><tr><td>Oracle Skill</td><td>68.5</td><td>81.1</td><td>48.7</td><td>68.6</td><td>79.7</td><td>54.7</td><td>64.4</td></tr><tr><td>Full-Skill Injection</td><td>60.1*</td><td>66.1*</td><td>35.0*</td><td>61.9*</td><td>59.5*</td><td>52.9</td><td>52.7*</td></tr><tr><td>LLM Selection</td><td>62.7*</td><td>69.7*</td><td>43.3*</td><td>58.7*</td><td>79.4</td><td>50.4*</td><td>59.2*</td></tr><tr><td>Progressive Disclosure</td><td>53.3*</td><td>65.3*</td><td>29.9*</td><td>54.3*</td><td>59.5*</td><td>45.7*</td><td>48.5*</td></tr><tr><td rowspan="5">Mistral3.1-24B</td><td>LLM Direct</td><td>49.3*</td><td>56.1*</td><td>29.2*</td><td>52.5*</td><td>49.6*</td><td>41.1*</td><td>43.4*</td></tr><tr><td>Oracle Skill</td><td>66.3</td><td>78.6</td><td>48.2</td><td>66.8</td><td>78.6</td><td>54.0</td><td>63.2</td></tr><tr><td>Full-Skill Injection</td><td>58.8*</td><td>57.2*</td><td>34.4*</td><td>59.6*</td><td>54.6*</td><td>45.6*</td><td>48.5*</td></tr><tr><td>LLM Selection</td><td>59.0*</td><td>66.7*</td><td>42.6*</td><td>55.6*</td><td>76.3*</td><td>46.8*</td><td>56.6*</td></tr><tr><td>Progressive Disclosure</td><td>54.8*</td><td>55.8*</td><td>27.1*</td><td>54.7*</td><td>62.6*</td><td>42.8*</td><td>46.7*</td></tr><tr><td rowspan="5">Qwen3-235B</td><td>LLM Direct</td><td>61.6*</td><td>76.1*</td><td>36.4*</td><td>66.4*</td><td>58.2*</td><td>46.7*</td><td>53.3*</td></tr><tr><td>Oracle Skill</td><td>65.9</td><td>86.6</td><td>52.3</td><td>75.8</td><td>84.5</td><td>57.0</td><td>67.5</td></tr><tr><td>Full-Skill Injection</td><td>64.4*</td><td>72.8*</td><td>38.0*</td><td>68.6*</td><td>66.2*</td><td>50.4*</td><td>56.2*</td></tr><tr><td>LLM Selection</td><td>66.8</td><td>79.5*</td><td>45.4*</td><td>73.5</td><td>82.5</td><td>49.7*</td><td>62.8*</td></tr><tr><td>Progressive Disclosure</td><td>68.1</td><td>81.1*</td><td>36.9*</td><td>72.2</td><td>77.1*</td><td>50.3*</td><td>59.9*</td></tr><tr><td rowspan="5">Qwen3-32B</td><td>LLM Direct</td><td>57.4*</td><td>75.9*</td><td>35.0*</td><td>65.5</td><td>53.9*</td><td>43.9*</td><td>50.8*</td></tr><tr><td>Oracle Skill</td><td>71.6</td><td>86.6</td><td>51.2</td><td>70.9</td><td>83.5</td><td>55.2</td><td>67.2</td></tr><tr><td>Full-Skill Injection</td><td>68.1*</td><td>70.5*</td><td>36.2*</td><td>70.4</td><td>59.5*</td><td>49.0*</td><td>54.3*</td></tr><tr><td>LLM Selection</td><td>69.3</td><td>81.1*</td><td>44.1*</td><td>65.9</td><td>82.5</td><td>48.1*</td><td>62.4*</td></tr><tr><td>Progressive Disclosure</td><td>64.9*</td><td>74.7*</td><td>35.5*</td><td>58.7*</td><td>71.1*</td><td>44.7*</td><td>55.3*</td></tr><tr><td rowspan="5">Qwen3-4B</td><td>LLM Direct</td><td>50.7*</td><td>75.0*</td><td>25.6*</td><td>56.1*</td><td>22.0*</td><td>36.4*</td><td>38.8*</td></tr><tr><td>Oracle Skill</td><td>69.7</td><td>85.1</td><td>47.1</td><td>70.9</td><td>73.5</td><td>45.5</td><td>61.6</td></tr><tr><td>Full-Skill Injection</td><td>64.0*</td><td>69.1*</td><td>30.2*</td><td>68.6</td><td>36.1*</td><td>41.5*</td><td>45.5*</td></tr><tr><td>LLM Selection</td><td>63.1*</td><td>70.0*</td><td>39.0*</td><td>62.3*</td><td>65.7*</td><td>39.7*</td><td>53.3*</td></tr><tr><td>Progressive Disclosure</td><td>52.7*</td><td>65.7*</td><td>26.7*</td><td>60.1*</td><td>45.0*</td><td>37.7*</td><td>43.2*</td></tr></table>

and description, and must select the most relevant one before generating an answer. In the Progressive Disclosure setting, the model interacts with a compact skill catalog and may explicitly issue skillloading actions to inspect a skill’s full content during reasoning. In retrieval-based experiments, candidate skills are retrieved from the shared external skill corpus in response to the input query. BM25 serves as the default retriever in the main experiments, while additional sparse, dense, hybrid, and reranking-based retrieval variants are introduced in $\ S 5 . 3$ .

# 5 Systematic Empirical Study

# 5.1 RQ1: Does Skill Retrieval Augmentation Improve Agent Performance?

We begin with the most fundamental question for the SRA paradigm: does retrieving and injecting external skills actually improve an agent’s performance compared to solving tasks directly from parametric knowledge alone? Table 2 reports end-task results across six benchmarks, six LLMs, and five skill-use settings, which differ in how external skills are accessed and incorporated during inference. At the two extremes, LLM Direct measures the model’s native task-solving ability without any external skill support. In contrast, Oracle Skill directly provides the annotated gold skill(s), serving as an upper bound on the potential value of correct skill access. Between these two extremes, we evaluate three practical SRA settings. Full-Skill Injection directly injects the top-1 skill retrieved by BM25 into the model context. LLM Selection first retrieves a larger candidate pool (top-50) and then requires the model to choose one skill to load. Progressive Disclosure follows an OpenClaw-style

protocol, where the agent is shown only a compact catalog of retrieved candidates and may decide for itself whether to reveal and load a skill on demand. Importantly, in both LLM Selection and Progressive Disclosure, the initial context contains only the name and description of each candidate among the retrieved top-50 skills, rather than the full skill content; the full skill is exposed only after a skill is explicitly selected. Conceptually, Progressive Disclosure is the most rational setting for an agentic system, as an ideal agent should only invoke external capabilities when its native parametric knowledge falls short, rather than injecting skills regardless of necessity.

Several conclusions emerge clearly from the results. First, the answer to RQ1 is unambiguously positive: external skills can substantially improve agent performance. When the correct skill is provided, Oracle Skill consistently outperforms LLM Direct across essentially all model-benchmark combinations, often by a large margin. This confirms a central premise of SRA-Bench: many tasks in our benchmark genuinely benefit from externalized procedural knowledge, and such skills are not redundant auxiliary context. When correctly accessed and incorporated, they materially extend what agents can accomplish beyond their native parametric capability.

Second, practical retrieval-based SRA methods can already yield meaningful gains, but these gains are far from uniform or guaranteed. In many model-benchmark combinations, Full-Skill Injection and especially LLM Selection improve over LLM Direct, showing that the value of skill augmentation is not confined to idealized oracle settings. However, these improvements are highly uneven across models, tasks, and exposure strategies, and retrieval-based methods remain substantially below the Oracle Skill upper bound in most cases. In other words, our results clearly establish the potential of skill retrieval augmentation, but also show that realizing this potential robustly under practical inference settings remains an open challenge.

Third, and most importantly, the effectiveness of SRA depends not only on whether the right skill is retrieved, but also on how the retrieved skill is exposed to the agent. Among the practical methods, LLM Selection provides the strongest overall trade-off between effectiveness and stability, consistently translating retrieval results into downstream gains and, in many cases, substantially narrowing the gap to the oracle upper bound. This suggests that, under current models, selection-based skill exposure is a more reliable way to operationalize external capabilities than simply injecting the top-ranked skill or leaving the loading decision entirely to the agent. By contrast, Progressive Disclosure exhibits a much less stable pattern: while it can improve performance in some settings, its gains are inconsistent, and it rarely matches the strongest selection-based configurations. The implication is that the bottleneck in SRA is not retrieval alone but controlled skill utilization: scalable skill augmentation requires not only access to a large skill corpus but also a reliable mechanism for deciding when and how retrieved skills should enter the reasoning process.

Overall, RQ1 supports a positive but qualified conclusion: Skill Retrieval Augmentation can indeed improve agent performance across a diverse range of models and tasks. Yet these benefits are not guaranteed by retrieval alone: they also depend critically on how retrieved skills are presented, selected, and integrated into the agent’s reasoning process. This observation directly motivates RQ2: once retrieval becomes noisy and candidate sets contain irrelevant distractors, can current SR-Agents still identify and utilize the right skills effectively?

# 5.2 RQ2: How Robust Are SR-Agents to Retrieval Noise?

The practical effectiveness of SRA depends not only on whether the correct skill can be retrieved, but also on whether the agent can still identify and use it once the candidate set becomes noisy. In realistic deployments, retrieved results inevitably include irrelevant yet plausible distractors, which may interfere with skill selection, loading, and downstream task solving. To isolate this incorporation challenge from retrieval itself, we construct a controlled setting in which the gold skill is always included in the candidate set, together with $N$ hard-negative distractor skills, where $N \in \{ 0 , 2 , 4 , \dot { 8 } \}$ . The distractors are constructed by alternately selecting non-gold, non-duplicate candidates from the BM25 and BGE retrieval lists in rank order, starting at rank 1, to capture both lexical and semantic distractor patterns. The final candidate order is then shuffled to avoid positional bias. We evaluate two representative skill-exposure strategies: Full Skill Injection, which injects the full content of all candidate skills into the prompt, and Progressive Disclosure, which presents only the name and description, and injects the full skill content only after an explicit selection. Figure 2 reports the end-task performance under increasing distractor levels.

![](images/eafb366ed1f1b0ede6df83dd0eb1094892e9a15051bd84fc6921cc3c5d0b9aad.jpg)

![](images/f1410924a96e13c62d805ccc65cb6c811075e8c87ee151751c1c0b689f21d153.jpg)

![](images/2d556639e59d7d478f53dc0b10413052c1f3f99d2061f9c422a3b002c287be06.jpg)

![](images/f884eaaddace801220f72349b507a104cfcfc1de691fe22f154f5bf5ca9b8660.jpg)

![](images/b968de71a184c3ec38424c9f407d8e6bc09790f6149464126d2ec62c9cef9cf5.jpg)

![](images/0362e0e712f9989e2f2a6352a397952844f54370471fc8c55b478a2ca5cf12f0.jpg)  
Full Skill Injection Progressive Disclosure   
Figure 2: End-task accuracy as the number of hard-negative distractor skills increases, with the gold skill always included. Full Skill Injection is consistently more brittle to noise, while Progressive Disclosure remains more robust across models.

Figure 2 shows that current SR-Agents are highly brittle to retrieval noise, even when the correct skill is already present in the candidate set. Across models, adding hard-negative distractors consistently degrades end-task performance, indicating that recall alone is insufficient for effective skill augmentation: once multiple plausible but irrelevant candidates coexist with the correct skill, the primary failure mode shifts to incorporation, i.e., whether the agent can reliably identify, preserve, and utilize the truly useful skill under distraction. Notably, this brittleness varies substantially across skill-exposure strategies. Under Full Skill Injection setting, performance drops sharply as distractors accumulate, suggesting that directly injecting multiple full skills creates substantial interference from both prompt overload and procedural confusion. By contrast, Progressive Disclosure is more stable: by withholding full skill content until an explicit reveal decision is made, it reduces irrelevant exposure and often matches or even surpasses Full Skill Injection under heavier noise. Notably, this robustness gap does not disappear with model scale. Larger models are not consistently better at suppressing distractors, and both small and strong models can fail once the candidate set becomes noisy. Taken together, these results show that scalable skill augmentation is not bottlenecked by retrieval alone, but by the agent’s ability to selectively expose and incorporate retrieved skills under interference, making controlled skill exposure a core requirement for robust SR-Agents.

# 5.3 RQ3: How Effective Are Existing Retrievers for Skill Retrieval?

We next examine whether existing retrieval methods can reliably identify relevant skills from a large external skill corpus. We compare a spectrum of first-stage retrieval methods, including lexical matching (BM25 [27] and TF-IDF [31]), dense retrieval (BGE [56] and Contriever [9]), and a simple hybrid method [3] that interleaves the ranked lists from BM25 and BGE to combine lexical and semantic signals. We further study LLM-based reranking [45] as a second-stage ranking mechanism: given a top-50 candidate pool retrieved by BM25, the target LLM reranks the candidates based on their relevance to the current request.

Tables 3 and 4 report Recall $@ K$ and nDCG@K across all benchmarks. Several conclusions emerge clearly from the results. First, skill retrieval is already feasible, but far from solved. Across datasets, existing retrievers can achieve high Recall $@ 1 0$ , confirming that skill descriptions do contain usable retrieval signals. However, retrieval quality varies sharply across benchmarks, and some settings remain difficult even for the strongest methods. This suggests that the difficulty of skill retrieval is highly task-dependent: in some domains, the relevant skill is expressed in ways that align naturally with skill text, while in others, the mapping from user request to useful skill is substantially more indirect. Second, there is no universal winner among conventional retrievers. Sparse methods remain surprisingly competitive, especially when the required capability is exposed through distinctive terminology, formulas, or code-like surface patterns. Dense retrieval is stronger

Table 3: Skill retrieval performance across different retrieval methods. We report Recall $@ 1$ and Recall $@ 1 0$ for each dataset. The upper block shows first-stage retrievers, while the lower block shows LLM-based rerankers applied to the top-50 candidates retrieved by BM25.   

<table><tr><td rowspan="2">Method</td><td colspan="2">TheoremQA</td><td colspan="2">LogicBench</td><td colspan="2">ToolQA</td><td colspan="2">CHAMP</td><td colspan="2">MedCalc-Bench</td><td colspan="2">BigIntCodeBench</td></tr><tr><td>R@1</td><td>R@10</td><td>R@1</td><td>R@10</td><td>R@1</td><td>R@10</td><td>R@1</td><td>R@10</td><td>R@1</td><td>R@10</td><td>R@1</td><td>R@10</td></tr><tr><td>BM25</td><td>57.2</td><td>80.7</td><td>12.0</td><td>36.1</td><td>7.0</td><td>55.1</td><td>13.2</td><td>36.1</td><td>29.3</td><td>69.2</td><td>23.6</td><td>61.1</td></tr><tr><td>TF-IDF</td><td>41.4</td><td>68.8</td><td>1.8</td><td>18.2</td><td>7.0</td><td>35.0</td><td>7.2</td><td>25.7</td><td>37.5</td><td>71.4</td><td>20.9</td><td>60.2</td></tr><tr><td>BGE</td><td>66.8</td><td>86.1</td><td>4.1</td><td>20.5</td><td>32.2</td><td>83.4</td><td>9.8</td><td>34.0</td><td>41.4</td><td>70.1</td><td>20.7</td><td>62.1</td></tr><tr><td>Contriever</td><td>52.1</td><td>75.6</td><td>5.5</td><td>18.4</td><td>21.2</td><td>42.7</td><td>3.7</td><td>29.3</td><td>34.9</td><td>66.9</td><td>19.0</td><td>54.1</td></tr><tr><td>Hybrid</td><td>57.2</td><td>90.0</td><td>12.0</td><td>33.6</td><td>7.0</td><td>83.5</td><td>13.2</td><td>41.4</td><td>29.3</td><td>67.8</td><td>23.6</td><td>68.4</td></tr><tr><td>Llama-3.1-8B</td><td>58.8</td><td>83.7</td><td>15.9</td><td>42.0</td><td>25.7</td><td>66.4</td><td>15.8</td><td>41.6</td><td>86.0</td><td>91.5</td><td>22.9</td><td>68.7</td></tr><tr><td>Llama-3.3-70B</td><td>76.0</td><td>88.6</td><td>27.4</td><td>55.4</td><td>51.0</td><td>76.9</td><td>22.5</td><td>47.8</td><td>89.5</td><td>92.5</td><td>27.4</td><td>78.6</td></tr><tr><td>Mistral3.1-24B</td><td>74.2</td><td>89.0</td><td>26.7</td><td>53.2</td><td>53.9</td><td>76.6</td><td>18.0</td><td>49.3</td><td>91.1</td><td>92.5</td><td>27.7</td><td>79.4</td></tr><tr><td>Qwen3-235B</td><td>75.4</td><td>88.8</td><td>30.9</td><td>56.4</td><td>56.4</td><td>76.2</td><td>22.1</td><td>50.2</td><td>92.3</td><td>92.5</td><td>27.2</td><td>80.0</td></tr><tr><td>Qwen3-32B</td><td>77.4</td><td>88.8</td><td>31.4</td><td>55.3</td><td>43.7</td><td>74.8</td><td>22.3</td><td>49.1</td><td>91.2</td><td>92.4</td><td>28.2</td><td>79.7</td></tr><tr><td>Qwen3-4B</td><td>69.5</td><td>87.1</td><td>21.8</td><td>43.3</td><td>39.9</td><td>70.8</td><td>18.5</td><td>44.0</td><td>85.9</td><td>91.2</td><td>26.3</td><td>72.5</td></tr></table>

Table 4: Skill retrieval performance across different retrieval methods. We report $\mathrm { N } @ 1$ (nDCG@1) and $\mathrm { N } @ 1 0 ( \mathrm { n D C G } @ 1 0 )$ for each dataset. The upper block shows first-stage retrievers, while the lower block shows LLM-based rerankers applied to the top-50 candidates retrieved by BM25.   

<table><tr><td rowspan="2">Method</td><td colspan="2">TheoremQA</td><td colspan="2">LogicBench</td><td colspan="2">ToolQA</td><td colspan="2">CHAMP</td><td colspan="2">MedCalc-Bench</td><td colspan="2">BigIntCodeBench</td></tr><tr><td>N@1</td><td>N@10</td><td>N@1</td><td>N@10</td><td>N@1</td><td>N@10</td><td>N@1</td><td>N@10</td><td>N@1</td><td>N@10</td><td>N@1</td><td>N@10</td></tr><tr><td>BM25</td><td>57.2</td><td>69.2</td><td>12.0</td><td>22.6</td><td>7.0</td><td>27.0</td><td>20.6</td><td>27.2</td><td>29.3</td><td>44.7</td><td>61.7</td><td>55.4</td></tr><tr><td>TF-IDF</td><td>41.4</td><td>54.4</td><td>1.8</td><td>8.3</td><td>7.0</td><td>18.9</td><td>12.1</td><td>17.1</td><td>37.5</td><td>52.0</td><td>55.2</td><td>52.0</td></tr><tr><td>BGE</td><td>66.8</td><td>75.9</td><td>4.1</td><td>11.1</td><td>32.2</td><td>58.6</td><td>13.5</td><td>22.6</td><td>41.4</td><td>54.5</td><td>54.0</td><td>53.6</td></tr><tr><td>Contriever</td><td>52.1</td><td>63.8</td><td>5.5</td><td>10.9</td><td>21.2</td><td>31.9</td><td>5.4</td><td>16.4</td><td>34.9</td><td>49.4</td><td>49.2</td><td>47.9</td></tr><tr><td>Hybrid</td><td>57.2</td><td>75.3</td><td>12.0</td><td>21.0</td><td>7.0</td><td>44.5</td><td>20.6</td><td>28.6</td><td>29.3</td><td>47.4</td><td>61.7</td><td>59.9</td></tr><tr><td>Llama-3.1-8B</td><td>58.8</td><td>71.5</td><td>15.9</td><td>28.8</td><td>25.7</td><td>47.5</td><td>25.6</td><td>32.1</td><td>86.0</td><td>88.9</td><td>59.2</td><td>61.6</td></tr><tr><td>Llama-3.3-70B</td><td>76.0</td><td>82.8</td><td>27.4</td><td>41.8</td><td>51.0</td><td>66.2</td><td>35.0</td><td>39.1</td><td>89.5</td><td>91.3</td><td>71.1</td><td>72.1</td></tr><tr><td>Mistral3.1-24B</td><td>74.2</td><td>81.9</td><td>26.7</td><td>40.3</td><td>53.9</td><td>67.3</td><td>27.4</td><td>37.0</td><td>91.1</td><td>91.9</td><td>72.0</td><td>72.9</td></tr><tr><td>Qwen3-235B</td><td>75.4</td><td>83.1</td><td>30.9</td><td>44.4</td><td>56.4</td><td>68.1</td><td>34.5</td><td>41.5</td><td>92.3</td><td>92.4</td><td>70.7</td><td>73.1</td></tr><tr><td>Qwen3-32B</td><td>77.4</td><td>83.5</td><td>31.4</td><td>44.2</td><td>43.7</td><td>62.7</td><td>35.4</td><td>41.0</td><td>91.2</td><td>91.9</td><td>73.2</td><td>74.1</td></tr><tr><td>Qwen3-4B</td><td>69.5</td><td>79.0</td><td>21.8</td><td>32.3</td><td>39.9</td><td>57.5</td><td>30.0</td><td>35.2</td><td>85.9</td><td>88.9</td><td>68.4</td><td>66.6</td></tr></table>

when relevance is expressed more semantically and less through direct lexical overlap, with BGE generally outperforming Contriever in this setting. Overall, these results suggest that skill retrieval is not well captured by either pure keyword matching or pure embedding similarity alone: the relevant signal is partly lexical, partly semantic, and often intertwined with procedural intent. Third, combining lexical and semantic signals mainly improves candidate coverage rather than consistently identifying the best skill at the first rank. The hybrid retriever often strengthens top-10 recall, indicating that sparse and dense retrieval provide complementary candidates. Finally, LLM-based reranking is the strongest overall retrieval strategy. Once a candidate pool is available, an LLM can substantially improve the ranking quality. This suggests that skill retrieval depends not only on topical relevance, but also on recognizing whether a candidate actually constitutes an actionable capability for the current task. At the same time, reranking quality is not strictly monotonic with model scale, indicating that better skill retrieval depends on more than parameter count alone.

Overall, RQ3 yields a clear but qualified answer: existing retrievers can make skill retrieval feasible, but they are not reliable enough to make it a solved problem. Sparse and dense methods exhibit complementary strengths, and LLM reranking provides the strongest overall performance. These findings establish retrieval as a meaningful bottleneck in SRA and motivate the next question: how does retrieval quality translate into end-to-end performance gains for SR-Agents?

Table 5: End-to-end performance $( \% )$ under different retrievers across six SRA-Bench datasets. The best results are in bold, and the second-best results are underlined.   

<table><tr><td>Model</td><td>Method</td><td>TheoremQA</td><td>LogicBench</td><td>ToolQA</td><td>CHAMP</td><td>MedCalc</td><td>BigIntCodeBench</td><td>Average</td></tr><tr><td rowspan="5">Llama-3.1-8B</td><td>BM25</td><td>36.5</td><td>58.3</td><td>13.6</td><td>27.8</td><td>36.7</td><td>34.2</td><td>32.7</td></tr><tr><td>TF-IDF</td><td>37.8</td><td>51.8</td><td>12.3</td><td>27.8</td><td>40.0</td><td>34.9</td><td>32.4</td></tr><tr><td>BGE</td><td>39.6</td><td>55.9</td><td>17.8</td><td>24.7</td><td>42.3</td><td>34.1</td><td>34.9</td></tr><tr><td>Contriever</td><td>37.8</td><td>54.7</td><td>19.7</td><td>24.2</td><td>37.5</td><td>31.3</td><td>33.4</td></tr><tr><td>BM25 + Rerank</td><td>40.8</td><td>60.4</td><td>18.3</td><td>28.3</td><td>56.8</td><td>36.1</td><td>39.4</td></tr><tr><td rowspan="5">Llama-3.3-70B</td><td>BM25</td><td>60.1</td><td>66.1</td><td>35.0</td><td>61.9</td><td>59.5</td><td>52.9</td><td>52.7</td></tr><tr><td>TF-IDF</td><td>56.4</td><td>56.3</td><td>30.7</td><td>59.6</td><td>60.8</td><td>51.4</td><td>49.6</td></tr><tr><td>BGE</td><td>61.7</td><td>58.6</td><td>38.7</td><td>64.1</td><td>60.9</td><td>49.9</td><td>52.6</td></tr><tr><td>Contriever</td><td>59.7</td><td>58.9</td><td>38.1</td><td>61.4</td><td>58.6</td><td>50.1</td><td>51.7</td></tr><tr><td>BM25 + Rerank</td><td>64.3</td><td>69.2</td><td>40.3</td><td>58.3</td><td>77.2</td><td>51.4</td><td>58.3</td></tr><tr><td rowspan="5">Mistral3.1-24B</td><td>BM25</td><td>58.8</td><td>57.2</td><td>34.4</td><td>59.6</td><td>54.6</td><td>45.6</td><td>48.5</td></tr><tr><td>TF-IDF</td><td>57.7</td><td>45.7</td><td>29.9</td><td>56.1</td><td>57.8</td><td>45.4</td><td>46.0</td></tr><tr><td>BGE</td><td>59.0</td><td>50.1</td><td>36.3</td><td>55.6</td><td>56.9</td><td>45.3</td><td>48.3</td></tr><tr><td>Contriever</td><td>55.3</td><td>46.7</td><td>35.0</td><td>57.0</td><td>48.4</td><td>43.0</td><td>44.8</td></tr><tr><td>BM25 + Rerank</td><td>62.0</td><td>64.1</td><td>38.7</td><td>52.0</td><td>74.6</td><td>48.4</td><td>55.4</td></tr><tr><td rowspan="5">Qwen3-235B</td><td>BM25</td><td>64.4</td><td>72.8</td><td>38.0</td><td>68.6</td><td>66.2</td><td>50.4</td><td>56.2</td></tr><tr><td>TF-IDF</td><td>60.9</td><td>69.1</td><td>34.6</td><td>70.4</td><td>67.8</td><td>50.1</td><td>54.6</td></tr><tr><td>BGE</td><td>62.5</td><td>68.4</td><td>42.7</td><td>72.2</td><td>68.8</td><td>48.1</td><td>56.7</td></tr><tr><td>Contriever</td><td>62.5</td><td>68.7</td><td>42.9</td><td>71.7</td><td>65.5</td><td>50.0</td><td>56.5</td></tr><tr><td>BM25 + Rerank</td><td>66.3</td><td>80.1</td><td>45.2</td><td>74.0</td><td>82.2</td><td>49.4</td><td>62.6</td></tr><tr><td rowspan="5">Qwen3-32B</td><td>BM25</td><td>68.1</td><td>70.5</td><td>36.2</td><td>70.4</td><td>59.5</td><td>49.0</td><td>54.3</td></tr><tr><td>TF-IDF</td><td>63.9</td><td>70.3</td><td>33.1</td><td>65.0</td><td>60.5</td><td>48.3</td><td>52.7</td></tr><tr><td>BGE</td><td>67.1</td><td>70.0</td><td>40.0</td><td>65.5</td><td>62.5</td><td>48.9</td><td>55.5</td></tr><tr><td>Contriever</td><td>65.6</td><td>69.1</td><td>40.6</td><td>57.8</td><td>58.6</td><td>48.7</td><td>54.1</td></tr><tr><td>BM25 + Rerank</td><td>69.3</td><td>78.3</td><td>43.0</td><td>65.9</td><td>82.2</td><td>49.0</td><td>61.8</td></tr><tr><td rowspan="5">Qwen3-4B</td><td>BM25</td><td>64.0</td><td>69.1</td><td>30.2</td><td>68.6</td><td>36.1</td><td>41.5</td><td>45.5</td></tr><tr><td>TF-IDF</td><td>59.7</td><td>65.7</td><td>24.3</td><td>47.1</td><td>43.8</td><td>39.6</td><td>43.1</td></tr><tr><td>BGE</td><td>62.5</td><td>66.1</td><td>38.6</td><td>61.0</td><td>43.1</td><td>38.9</td><td>47.7</td></tr><tr><td>Contriever</td><td>57.7</td><td>65.1</td><td>36.2</td><td>59.2</td><td>38.9</td><td>39.4</td><td>45.4</td></tr><tr><td>BM25 + Rerank</td><td>65.5</td><td>72.9</td><td>36.7</td><td>65.0</td><td>66.2</td><td>40.9</td><td>53.8</td></tr></table>

# 5.4 RQ4: How Does Retrieval Quality Affect End-Task Performance?

To investigate how retrieval quality affects downstream task performance, we evaluate SR-Agents with five retrieval strategies: BM25 [27], TF-IDF [31], BGE [56], Contriever [9], and ${ \bf B } { \bf M } 2 5 + { \bf \Delta }$ Rerank [45]. Given an input query, each method retrieves candidate skills, from which only the top-1 skill is selected and injected into the LLM for final answer generation. For ${ \bf B } { \bf M } 2 5 + { \bf \Delta }$ Rerank, we first retrieve the top 50 skills using BM25, then use the corresponding base LLM to rerank these candidates, and finally select the top-1 reranked skill for injection.

Table 5 compares end-to-end SR-Agents’ performance under different retrievers across six SRA-Bench datasets and six LLMs. Overall, RQ4 does not admit a simple answer: stronger retrieval generally improves end-task performance, but the relationship is neither direct nor strictly monotonic. Across models and datasets, retrieval pipelines that achieve better ranking quality, especially higher precision at top ranks, tend to deliver stronger downstream performance, indicating that retrieval remains an important bottleneck for SR-Agents. In particular, reranking BM25-retrieved candidates often produces the most consistent end-to-end gains, suggesting that downstream success depends not only on whether relevant skills are retrieved, but also on whether they are reliably selected and utilized by the agent.

At the same time, the impact of retrieval quality varies substantially across tasks, and no single retrieval paradigm consistently dominates across benchmarks. This variability highlights a key distinction between skill retrieval and conventional document retrieval: in Skill Retrieval Augmentation, the retrieved units are executable capability packages rather than plain textual evidence, so end-task success depends not only on retrieval itself, but also on whether the agent can correctly interpret, incorporate, and execute the retrieved skills. As a result, improvements in standalone retrieval metrics

Table 6: Skill loading rate $( \% )$ across models and datasets. Skill loading rate measures whether the agent successfully incorporates at least one valid skill before producing the final answer. Overall is computed over all 5,400 instances, rather than by averaging dataset-level rates.   

<table><tr><td>Model</td><td>TheoremQA</td><td>LogicBench</td><td>ToolQA</td><td>CHAMP</td><td>MedCalc</td><td>BigIntCodeBench</td><td>Overall</td></tr><tr><td>Qwen3-4B</td><td>15.5%</td><td>55.0%</td><td>2.4%</td><td>13.5%</td><td>33.4%</td><td>14.8%</td><td>21.0%</td></tr><tr><td>Llama-8B</td><td>97.9%</td><td>99.7%</td><td>2.9%</td><td>86.5%</td><td>97.8%</td><td>96.0%</td><td>72.1%</td></tr><tr><td>Mistral-24B</td><td>45.1%</td><td>38.0%</td><td>29.6%</td><td>22.9%</td><td>32.1%</td><td>6.4%</td><td>28.3%</td></tr><tr><td>Qwen3-32B</td><td>24.9%</td><td>45.9%</td><td>0.6%</td><td>1.8%</td><td>41.3%</td><td>7.2%</td><td>20.1%</td></tr><tr><td>Llama-70B</td><td>31.7%</td><td>32.6%</td><td>6.9%</td><td>1.8%</td><td>0.9%</td><td>0.0%</td><td>11.1%</td></tr><tr><td>Qwen3-235B</td><td>77.2%</td><td>87.5%</td><td>1.5%</td><td>15.7%</td><td>56.4%</td><td>88.2%</td><td>54.1%</td></tr><tr><td>GLM-5.1</td><td>55.3%</td><td>25.7%</td><td>7.8%</td><td>25.6%</td><td>86.5%</td><td>27.5%</td><td>37.8%</td></tr><tr><td>GPT-5.4</td><td>6.8%</td><td>0.4%</td><td>38.7%</td><td>15.7%</td><td>74.7%</td><td>19.5%</td><td>31.2%</td></tr></table>

Table 7: Comparison of overall skill-loading rates between instances whose gold skill is covered by BM25 top-50 and those whose gold skill is not. Instances are grouped by whether at least one gold skill appears in BM25 top-50 for multi-label datasets.   

<table><tr><td>Model</td><td>Gold in Top-50 Load Rate</td><td>Gold not in Top-50 Load Rate</td><td>Diff.</td></tr><tr><td>Qwen3-4B</td><td>21.8%</td><td>16.9%</td><td>+4.9pp</td></tr><tr><td>Llama-8B</td><td>73.7%</td><td>64.1%</td><td>+9.6pp</td></tr><tr><td>Mistral-24B</td><td>30.4%</td><td>17.3%</td><td>+13.1pp</td></tr><tr><td>Qwen3-32B</td><td>21.2%</td><td>14.3%</td><td>+6.9pp</td></tr><tr><td>Llama-70B</td><td>10.4%</td><td>14.5%</td><td>-4.2pp</td></tr><tr><td>Qwen3-235B</td><td>57.8%</td><td>35.3%</td><td>+22.4pp</td></tr><tr><td>GLM-5.1</td><td>43.5%</td><td>8.2%</td><td>+35.4pp</td></tr><tr><td>GPT-5.4</td><td>36.6%</td><td>3.6%</td><td>+33.0pp</td></tr></table>

do not always translate proportionally into gains on the final task. Better retrieval should therefore be understood as increasing the likelihood of successful skill use, rather than guaranteeing it.

Therefore, RQ4 suggests that retrieval quality is a necessary but insufficient condition for strong SRA performance. Improving retrieval is clearly beneficial, particularly when it enhances top-rank precision and reduces exposure to distracting or misleading skills. Still, its ultimate effect is mediated by downstream skill incorporation and execution. This further supports our broader claim that Skill Retrieval Augmentation should be studied not as a retrieval problem in isolation, but as a coupled pipeline in which retrieval and subsequent skill use jointly determine end-task outcomes.

# 5.5 RQ5: Are Current SR-Agents Relevance-Aware in Skill Loading?

Table 6 and Table 7 examine whether current LLMs load skills in a relevance-aware manner, namely, whether they are more likely to load a skill when the retrieved candidate set actually contains a gold skill. Overall, the answer is largely negative. Skill-loading behavior is highly model-dependent rather than a stable capability that improves monotonically with scale. Under the same skill corpus and retriever, different models exhibit sharply different loading propensities across datasets: some tend to load skills almost indiscriminately, while others are overly conservative and rarely load at all. This already suggests that current SR-Agents do not follow a consistent or rational loading policy, and that skill incorporation is heavily shaped by model-specific tendencies.

More importantly, for most open-source models, the presence of a gold skill in the retrieved candidate pool has only limited influence on whether the agent decides to load a skill. Although gold-covered cases sometimes lead to slightly higher loading rates, many models still load skills at substantial rates even when no gold skill is present in the BM25 top-50. This reveals a clear form of skill-loading hallucination: rather than first determining whether the retrieved candidates contain a genuinely relevant capability, the model often proceeds to load some retrieved skill anyway. As a result, successful gold-skill loading is often accompanied by a comparable tendency to load irrelevant skills, exposing a substantial disconnect between retrieval success and effective utilization. A clearer form

Table 8: Need-aware skill-loading analysis. We partition instances according to whether the same model can solve them without external skills (Skill-Free Correct) or not (Skill-Free Wrong). A need-aware SR-Agent should load skills more often when the task cannot be solved natively. $\Delta$ is computed as Skill-Free Wrong − Skill-Free Correct, in percentage points (pp).   

<table><tr><td rowspan="2">Model</td><td colspan="3">Skill-Free Correct</td><td colspan="3">Skill-Free Wrong</td><td colspan="2">Difference</td></tr><tr><td>n</td><td>Load Rate</td><td>Gold Load Rate</td><td>n</td><td>Load Rate</td><td>Gold Load Rate</td><td>Δ Load</td><td>Δ Gold</td></tr><tr><td>Qwen3-4B</td><td>1618</td><td>23.6</td><td>13.1</td><td>2902</td><td>20.8</td><td>17.2</td><td>-2.8</td><td>+4.1</td></tr><tr><td>Llama-8B</td><td>1273</td><td>84.5</td><td>59.7</td><td>3247</td><td>69.4</td><td>54.0</td><td>-15.1</td><td>-5.7</td></tr><tr><td>Mistral-24B</td><td>1893</td><td>28.0</td><td>22.0</td><td>2627</td><td>32.1</td><td>25.0</td><td>+4.1</td><td>+3.0</td></tr><tr><td>Qwen3-32B</td><td>2201</td><td>21.4</td><td>17.5</td><td>2319</td><td>21.0</td><td>18.8</td><td>-0.4</td><td>+1.3</td></tr><tr><td>Llama-70B</td><td>2105</td><td>10.7</td><td>7.6</td><td>2415</td><td>10.1</td><td>7.1</td><td>-0.6</td><td>-0.6</td></tr><tr><td>Qwen3-235B</td><td>2335</td><td>59.5</td><td>46.3</td><td>2185</td><td>56.0</td><td>46.6</td><td>-3.5</td><td>+0.4</td></tr><tr><td>GLM-5.1</td><td>2616</td><td>45.0</td><td>42.5</td><td>1904</td><td>41.5</td><td>38.0</td><td>-3.4</td><td>-4.4</td></tr><tr><td>GPT-5.4</td><td>2790</td><td>34.4</td><td>33.2</td><td>1730</td><td>40.2</td><td>37.0</td><td>+5.8</td><td>+3.8</td></tr><tr><td>Overall</td><td>16831</td><td>36.9</td><td>30.0</td><td>19329</td><td>36.9</td><td>30.5</td><td>+0.1</td><td>+0.5</td></tr></table>

of relevance awareness emerges only in the strongest models. Among the open-source models, this pattern becomes apparent only at the 235B scale. By contrast, frontier models such as GLM-5.1 and GPT-5.4 show much sharper separation between gold-covered and gold-absent cases, indicating a substantially stronger ability to condition skill loading on actual candidate relevance. Importantly, their advantage does not come from indiscriminately loading more skills overall, but from being more selective: they are more willing to load when a relevant skill is available, and much less willing to do so when it is not.

Taken together, these results show that most current LLMs are generally not reliably relevance-aware when incorporating skills. Although frontier models such as GLM-5.1 and GPT-5.4 exhibit stronger separation between gold-covered and gold-absent cases than open-source models, their behavior remains far from ideal: skill loading is still not fully governed by whether a genuinely relevant skill has been retrieved. For most models, whether a suitable skill has actually been retrieved exerts surprisingly weak control over loading behavior, leaving a major gap between successful retrieval and effective skill utilization.

# 5.6 RQ6: Are Current SR-Agents Need-Aware in Skill Loading?

We next examine whether current LLMs exhibit need-aware skill-loading behavior during incorporation. Intuitively, a rational SR-Agent should be more inclined to load external skills when a task exceeds its native parametric capability, and should remain more conservative when the same task can already be solved without external skill augmentation. To test this, we partition instances based on whether the same model answers them correctly in the skill-free setting, and then compare skill-loading behavior between natively solvable and natively unsolvable cases. To isolate the effect of need awareness from retrieval availability, this analysis is restricted to instances where a gold skill appears among the top-50 retrieved candidates, ensuring that both groups are evaluated under the same condition in which relevant external skills are available for loading. The resulting loading rates are shown in Table 8.

The results reveal a striking absence of need awareness. Across models, skill-loading rates remain remarkably similar between instances that the model can already solve on its own and those it fails to solve without external skills, including the frontier models that showed comparatively stronger relevance awareness in RQ5. In other words, current LLMs do not meaningfully condition skillloading decisions on whether a genuine capability gap is present. Even when a task demonstrably exceeds the model’s native competence, the agent is generally no more likely to load a skill than when the task is already within reach of its parametric knowledge. This pattern indicates that skill loading is not functioning as a targeted compensatory mechanism for missing capability, but is instead triggered in a largely indiscriminate manner.

Taken together, these results show that current LLMs are generally not need-aware when incorporating skills. Combined with the weak relevance awareness observed in RQ5, this points to a deeper limitation of today’s SR-Agents: they still lack a reliable mechanism for deciding what external skill

to load and when external loading is actually needed. This substantially weakens the practical value of large-scale skill retrieval augmentation and suggests that scalable skill augmentation is fundamentally a problem of controlled, selective, and need-aware skill utilization, rather than retrieval alone.

# 6 Related Work

# 6.1 Retrieval Augmented Generation and Tool Use

Retrieval-Augmented Generation (RAG) extends large language models (LLMs) by grounding them with task-relevant information retrieved from external knowledge sources [14, 5, 49, 35, 48]. A growing body of work has shown that RAG is effective for reducing hallucinations [51, 38, 42, 43], supporting knowledge updating [53, 52], and adapting LLMs to new domains without costly fullmodel retraining [44, 37]. Most existing RAG systems follow a retrieval-then-read pipeline, in which the input query is first used to retrieve relevant documents from a large external corpus [26, 33, 7, 34], and the retrieved evidence is then incorporated into the model’s context to assist with generation. Beyond this standard formulation, recent extensions have explored dynamic RAG [10, 40, 39], graph RAG [6], parametric RAG [41, 46], and agentic RAG [11].

In parallel, a rich line of work explores tool use and function calling, in which models or agents learn to invoke external APIs or functions. These studies typically focus on tool/function use rather than the retrieval of reusable skills [47, 28]. This paradigm is well exemplified by models such as Toolformer [28], Gorilla [24], and ToolLLM [25], and is evaluated on benchmarks such as BFCL [23].

While these paradigms have dramatically expanded model capabilities, SRA operates at their intersection while systematically addressing their respective limitations. Unlike RAG, which retrieves static knowledge, SRA retrieves executable and operational capabilities. Unlike conventional tool use, which assumes a visible, limited set of APIs, SRA retrieves from an open-ended corpus of modular packages that encompass instructions, invocation conditions, and procedural guidance. Consequently, SRA introduces a more complex pipeline that requires evaluating retrieval relevance alongside downstream skill incorporation and task application.

# 6.2 Agent Skills

Beyond atomic tools, recent research increasingly treats modular skills as a core abstraction for agent design. Pioneering systems like Voyager [54] demonstrate the value of maintaining an ever-growing skill corpus of executable programs for embodied lifelong learning. Furthermore, recent analyses [16] have begun to map the emerging ecosystem of agent skills as plug-and-play extensions for LLMs. While these works underscore the utility of reusable skill abstractions, their primary focus remains on skill acquisition, code synthesis, or ecosystem analysis. SRA departs from this focus by addressing the critical bottleneck of scalability: rather than investigating how to create or store skills, we tackle the challenge of retrieval-time access, investigating how agents can dynamically identify and load the right capabilities from a massive, out-of-context library.

# 6.3 Benchmarking Skill Effectiveness

As skill ecosystems continue to expand, rigorous benchmarks are increasingly needed to evaluate how effectively external skills support agent performance. However, existing benchmarks primarily evaluate tool/function invocation, and even when retrieval is involved, they do not decompose the full pipeline of standalone skill retrieval, skill incorporation, and downstream task application. More recently, SkillsBench [15] was introduced to study whether curated skills genuinely assist agents in solving diverse tasks. While highly complementary, our work addresses a distinct and fundamentally different problem. By formulating skill retrieval augmentation as a standalone research problem, we introduce SRA-Bench, which provides a fine-grained, decomposed evaluation pipeline. Instead of merely assessing whether skills are broadly beneficial, SRA-Bench diagnoses the specific bottlenecks of scalable augmentation: whether the correct skills can be retrieved from a vast corpus, whether the agent can selectively incorporate them without being overwhelmed, and whether this multi-stage process ultimately translates into measurable downstream success.

# 7 Toward a Research Agenda for Skill Retrieval Augmentation

The results of this paper suggest that Skill Retrieval Augmentation (SRA) should not be understood merely as a new retrieval problem over skills, but as a broader research agenda for scalable capability augmentation in agent systems. On the positive side, our experiments show that retrieving external skills from a large skill corpus can substantially improve downstream performance, confirming the practical promise of augmenting agents with large-scale reusable capabilities beyond their native parametric knowledge. At the same time, our findings also make clear that retrieval alone is far from sufficient. Even when relevant skills are retrieved, current agents often struggle to identify which skills are actually useful, decide whether external help is needed, and reliably convert retrieved skills into better task execution. Taken together, these observations suggest that the next stage of progress in SRA will require advances not only in retrieval methods but also in how skills are organized, maintained, incorporated, internalized, and continually improved over time. In this section, we outline several research directions that, in our view, define the emerging agenda of SRA.

From unstructured skill collections to structured skill libraries. The first research direction concerns how a large-scale skill corpus should be represented and organized. In this paper, we instantiate SRA with a large, noisy skill collection, enabling us to study retrieval and skill use under realistic open-world conditions. However, simply treating a collection of skills as an unstructured list of independently indexed skills may become increasingly inadequate as the number of skills grows. Unlike ordinary documents, skills are reusable capability units with rich relationships among themselves. Some skills serve as prerequisites for others, some provide alternative implementations of the same capability, some specialize broader skills into narrower domains, and some are naturally composed into multi-step workflows. As a result, future SRA systems may need more structured forms of organization, such as graphs, hierarchies, clusters, or explicit dependency structures over skills. These structures could help retrieval systems narrow the search space, reduce interference from superficially similar yet functionally irrelevant skills, and support more effective reasoning across the available capabilities. Importantly, our proposed large-scale skill corpus and gold-skill annotations introduced in this paper provide a concrete testbed for studying these questions under decomposed evaluation.

Skill quality control, offline refinement, and skill evolution. The second direction concerns the quality of the skill corpus itself. Open skill ecosystems are inherently heterogeneous: many skills may be incomplete, poorly written, outdated, redundant, or even incorrect. This challenge is qualitatively different from conventional document retrieval, because the retrieved unit in SRA is not merely textual evidence but an actionable capability package whose errors may directly mislead downstream reasoning and execution. As a result, scalable SRA will likely require dedicated pipelines for skill validation, debugging, and refinement. One promising direction is offline skill optimization: given abundant offline compute, a system may iteratively inspect skills, detect missing preconditions, repair procedural flaws, rewrite ambiguous descriptions, attach executable resources, or revise invocation conditions to make the skill more retrievable and more usable. Such optimization could be driven by static analysis of the skill artifact itself, by execution-based testing on synthesized tasks, or by agentic self-improvement loops in which agents attempt to use a skill, reflect on failure cases, and revise the skill accordingly. In this view, the skill corpus is no longer a passive repository, but a continually maintained capability resource whose quality can be improved over time.

From semantic matching to utility-aware skill retrieval. The third direction concerns retrieval itself. Our formulation already emphasizes that retrieval in SRA should be evaluated not only by semantic relevance, but by downstream utility: whether the retrieved candidates actually help solve the task. This suggests that future skill retrieval methods should be optimized for expected usefulness rather than topical similarity alone. In contrast to classical document retrieval, where the target is often factual evidence, skill retrieval requires identifying actionable capability packages whose value depends on whether they can be effectively incorporated and operationalized by the agent. This opens up several opportunities. Retrieval models could be trained directly from end-task feedback, using success or failure on downstream tasks as a supervision signal. Dense retrievers could be optimized with utility-aware objectives rather than purely relevance-based contrastive signals. Rerankers could prioritize skills that are not only semantically related to the query but also likely to close the agent’s current capability gap. More ambitiously, retrieval may need to move beyond single-skill ranking to retrieve sets of complementary skills, or to be conditioned on the agent’s intermediate reasoning

state rather than only on the initial user request. Taken together, these challenges suggest that skill retrieval in SRA should be studied as a distinct IR problem, rather than as a straightforward extension of classical document retrieval.

Toward Parametric Skill Augmentation. Finally, a particularly promising direction is Parametric Skill Augmentation, where external skills are not only retrieved as text, but also transformed offline into plug-in parameters that can be loaded into the model when needed. This direction is motivated by recent advances in Parametric RAG [41, 36], which suggest that external information need not be injected only through the input context, but can instead be integrated into the model’s parameters, thereby reducing online context overhead while enabling deeper interaction with the model’s internal computation. For skills, this idea may be even more natural than for documents. Unlike open-ended world knowledge, skills are reusable capability modules: many are invoked repeatedly across tasks, and their role is not merely to provide declarative information, but to induce reliable procedures, decision patterns, and action policies. As a result, repeatedly exposing the same skill through textual descriptions can be an inefficient use of context budget, especially for frequently used skills. Parameterizing such head skills could amortize this cost by replacing repeated token-level injection with compact plug-in modules, while retaining rare, long-tail, or rapidly evolving skills in an external retrieval corpus. Beyond efficiency, parameterization may also improve effectiveness. As these plug-in parameters are constructed offline, they can be explicitly optimized for downstream use, for example, through task-aware initialization, refinement, or feedback-driven updates, rather than being presented to the model only as surface-form prompt text. More importantly, in-context skill injection primarily affects the model at the input level. In contrast, parametric skill augmentation offers a potential pathway for agents to internalize external capabilities more deeply into the LLM’s parameters that drive reasoning and acting. In this sense, parametric skill augmentation is not merely a compression strategy for SRA but a potentially stronger mechanism for turning retrieved skills into durable, operational competence. A hybrid architecture may therefore be especially attractive: frequently used core skills could be maintained as parametric modules for efficient and reliable access, while the open-ended long tail of skills remains retrievable from an external corpus.

Taken together, these directions suggest that SRA is not a narrowly scoped extension of retrievalaugmented generation, but an emerging research paradigm for scalable capability augmentation, for which the resources and findings introduced in this paper provide a concrete starting point.

# 8 Conclusion

In this paper, we formulate Skill Retrieval Augmentation (SRA) as a new paradigm for augmenting agents with external capabilities at scale. Rather than assuming that all potentially useful skills can be explicitly exposed in context, SRA studies how agents should retrieve, incorporate, and apply reusable skills from large external skill corpora on demand. To make this problem concrete and measurable, we construct a large-scale skill corpus, introduce SRA-Bench as the first benchmark for decomposed evaluation of the SRA pipeline, and establish SR-Agents as a baseline family for systematic empirical study. Together, these contributions provide a concrete foundation for studying scalable skill augmentation as a distinct problem in agent systems.

Our experiments show both the promise and the central challenge of this paradigm. On the one hand, external skills can substantially improve downstream performance when they are correctly accessed and used. On the other hand, our results reveal that scalable skill augmentation cannot be solved by retrieval alone. Even when relevant skills are successfully retrieved, current LLM agents often still fail to recognize whether external help is needed, which retrieved skills are truly worth loading, and how to translate those skills into reliable gains in reasoning and action. These findings suggest that the core bottleneck of SRA lies not only in finding relevant capabilities but in enabling agents to access them selectively, incorporate them appropriately, and operationalize them robustly.

We therefore view this work not merely as introducing a benchmark, but as helping define a broader research agenda. By releasing the skill corpus, SRA-Bench, and the analyses in this work, we aim to provide a shared foundation for future research on how external skills should be organized, maintained, retrieved, selected, refined, internalized, and accumulated over time. More broadly, if Retrieval-Augmented Generation made external knowledge a central object of study for language models, we believe Skill Retrieval Augmentation can help make external capabilities a central object

of study for agent systems. The next generation of agents will be shaped not only by what they know but also by how effectively they can access and leverage capabilities beyond their weights.

# References

[1] Qingyao Ai, Yichen Tang, Changyue Wang, Jianming Long, Weihang Su, and Yiqun Liu. Memorybench: A benchmark for memory and continual learning in llm systems. arXiv preprint arXiv:2510.17281, 2025.   
[2] Anthropic. Claude code. https://www.anthropic.com/product/claude-code, 2026. Official product page. Accessed: 2026-04-20.   
[3] Sebastian Bruch, Siyu Gai, and Amir Ingber. An analysis of fusion functions for hybrid retrieval. ACM Transactions on Information Systems, 42(1):1–35, 2023.   
[4] Wenhu Chen, Ming Yin, Max Ku, Pan Lu, Yixin Wan, Xueguang Ma, Jianyu Xu, Xinyi Wang, and Tony Xia. Theoremqa: A theorem-driven question answering dataset. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 7889–7901, 2023.   
[5] Qian Dong, Qingyao Ai, Hongning Wang, Yiding Liu, Haitao Li, Weihang Su, Yiqun Liu, Tat-Seng Chua, and Shaoping Ma. Decoupling knowledge and context: An efficient and effective retrieval augmented generation framework via cross attention. In Proceedings of the ACM on Web Conference 2025, 2025.   
[6] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, and Jonathan Larson. From local to global: A graph rag approach to query-focused summarization. arXiv preprint arXiv:2404.16130, 2024.   
[7] Yan Fang, Jingtao Zhan, Qingyao Ai, Jiaxin Mao, Weihang Su, Jia Chen, and Yiqun Liu. Scaling laws for dense retrieval. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 1339–1349, 2024.   
[8] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.   
[9] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. Unsupervised dense information retrieval with contrastive learning. arXiv preprint arXiv:2112.09118, 2021.   
[10] Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. Active retrieval augmented generation. arXiv preprint arXiv:2305.06983, 2023.   
[11] Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516, 2025.   
[12] Ehud Karpas, Omri Abend, Yonatan Belinkov, Barak Lenz, Opher Lieber, Nir Ratner, Yoav Shoham, Hofit Bata, Yoav Levine, Kevin Leyton-Brown, et al. Mrkl systems: A modular, neuro-symbolic architecture that combines large language models, external knowledge sources and discrete reasoning. arXiv preprint arXiv:2205.00445, 2022.   
[13] Nikhil Khandekar, Qiao Jin, Guangzhi Xiong, Soren Dunn, Serina S Applebaum, Zain Anwar, Maame Sarfo-Gyamfi, Conrad W Safranek, Abid A Anwar, Andrew Zhang, et al. Medcalcbench: Evaluating large language models for medical calculations. Advances in Neural Information Processing Systems, 37:84730–84745, 2024.   
[14] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in Neural Information Processing Systems, 33:9459–9474, 2020.

[15] Xiangyi Li, Wenbo Chen, Yimin Liu, Shenghan Zheng, Xiaokun Chen, Yifeng He, Yubo Li, Bingran You, Haotian Shen, Jiankai Sun, et al. Skillsbench: Benchmarking how well agent skills work across diverse tasks. arXiv preprint arXiv:2602.12670, 2026.   
[16] George Ling, Shanshan Zhong, and Richard Huang. Agent skills: A data-driven analysis of claude skills for extending large language model functionality. arXiv preprint arXiv:2602.08004, 2026.   
[17] Yujun Mao, Yoon Kim, and Yilun Zhou. Champ: A competition-level dataset for fine-grained analyses of llms’ mathematical reasoning capabilities. In Findings of the Association for Computational Linguistics: ACL 2024, pages 13256–13274, 2024.   
[18] Meta. Llama 3.3 model card. https://www.llama.com/docs/ model-cards-and-prompt-formats/llama3_3/, 2024. Accessed: 2026-04-20.   
[19] Mistral AI. Mistral small 3.1. https://mistral.ai/news/mistral-small-3-1, March 2025. Official release announcement. Accessed: 2026-04-20.   
[20] OpenAI. Codex. https://openai.com/codex/, 2026. Official product page. Accessed: 2026-04-20.   
[21] OpenAI. Introducing gpt-5.4. https://openai.com/index/introducing-gpt-5-4/, March 2026. Accessed: 2026-04-20.   
[22] Mihir Parmar, Nisarg Patel, Neeraj Varshney, Mutsumi Nakamura, Man Luo, Santosh Mashetty, Arindam Mitra, and Chitta Baral. Logicbench: Towards systematic evaluation of logical reasoning ability of large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13679–13707, 2024.   
[23] Shishir G Patil, Huanzhi Mao, Fanjia Yan, Charlie Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, and Joseph E Gonzalez. The berkeley function calling leaderboard (bfcl): From tool use to agentic evaluation of large language models. In Forty-second International Conference on Machine Learning, 2025.   
[24] Shishir G Patil, Tianjun Zhang, Xin Wang, and Joseph E Gonzalez. Gorilla: Large language model connected with massive apis. Advances in Neural Information Processing Systems, 37:126544–126565, 2024.   
[25] Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, et al. Toolllm: Facilitating large language models to master $1 6 0 0 0 +$ real-world apis. The Twelfth International Conference on Learning Representations, 2023.   
[26] Stephen Robertson, Hugo Zaragoza, et al. The probabilistic relevance framework: Bm25 and beyond. Foundations and Trends® in Information Retrieval, 3(4):333–389, 2009.   
[27] Stephen Edward Robertson, Steve Walker, Susan Jones, Micheline M Hancock-Beaulieu, Mike Gatford, et al. Okapi at trec. 1994.   
[28] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. Advances in neural information processing systems, 36:68539– 68551, 2023.   
[29] Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. Hugginggpt: Solving ai tasks with chatgpt and its friends in hugging face. Advances in Neural Information Processing Systems, 36:38154–38180, 2023.   
[30] SkillsMP. Skillsmp: Agent skills marketplace. https://skillsmp.com/, 2026. Accessed: 2026-04-26.   
[31] Karen Sparck Jones. A statistical interpretation of term specificity and its application in retrieval. Journal of documentation, 28(1):11–21, 1972.   
[32] Peter Steinberger and OpenClaw Contributors. Openclaw, 2026. Open-source personal AI assistant. Accessed: 2026-04-20.

[33] Weihang Su, Qingyao Ai, Xiangsheng Li, Jia Chen, Yiqun Liu, Xiaolong Wu, and Shengluan Hou. Wikiformer: Pre-training with structured information of wikipedia for ad-hoc retrieval. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 19026–19034, 2024.   
[34] Weihang Su, Qingyao Ai, Yueyue Wu, Anzhe Xie, Changyue Wang, Yixiao Ma, Haitao Li, Zhijing Wu, Yiqun Liu, and Min Zhang. Pre-training for legal case retrieval based on inter-case distinctions. ACM Transactions on Information Systems, 43(5):1–27, 2025.   
[35] Weihang Su, Qingyao Ai, Jingtao Zhan, Qian Dong, and Yiqun Liu. Dynamic and parametric retrieval-augmented generation, 2025.   
[36] Weihang Su, Qian Dong, Qingyao Ai, and Yiqun Liu. Sigir-ap 2025 tutorial proposal: Dynamic and parametric retrieval-augmented generation. In 3rd International ACM SIGIR Conference on Information Retrieval in the Asia Pacific, 2025.   
[37] Weihang Su, Yiran Hu, Anzhe Xie, Qingyao Ai, Quezi Bing, Ning Zheng, Yun Liu, Weixing Shen, and Yiqun Liu. STARD: A Chinese statute retrieval dataset derived from real-life queries by non-professionals. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors, Findings of the Association for Computational Linguistics: EMNLP 2024, pages 10658–10671, Miami, Florida, USA, November 2024. Association for Computational Linguistics.   
[38] Weihang Su, Jianming Long, Changyue Wang, Shiyu Lin, Jingyan Xu, Ziyi Ye, Qingyao Ai, and Yiqun Liu. Towards unification of hallucination detection and fact verification for large language models. arXiv preprint arXiv:2512.02772, 2025.   
[39] Weihang Su, Yichen Tang, Qingyao Ai, Changyue Wang, Zhijing Wu, and Yiqun Liu. Mitigating entity-level hallucination in large language models. In Proceedings of the 2024 Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region, pages 23–31, 2024.   
[40] Weihang Su, Yichen Tang, Qingyao Ai, Zhijing Wu, and Yiqun Liu. DRAGIN: Dynamic retrieval augmented generation based on the real-time information needs of large language models. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12991–13013, Bangkok, Thailand, August 2024. Association for Computational Linguistics.   
[41] Weihang Su, Yichen Tang, Qingyao Ai, Junxi Yan, Changyue Wang, Hongning Wang, Ziyi Ye, Yujia Zhou, and Yiqun Liu. Parametric retrieval augmented generation. In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 1240–1250, 2025.   
[42] Weihang Su, Changyue Wang, Qingyao Ai, Yiran Hu, Zhijing Wu, Yujia Zhou, and Yiqun Liu. Unsupervised real-time hallucination detection based on the internal states of large language models. arXiv preprint arXiv:2403.06448, 2024.   
[43] Weihang Su, Anzhe Xie, Qingyao Ai, Jianming Long, Xuanyi Chen, Jiaxin Mao, Ziyi Ye, and Yiqun Liu. Surge: A benchmark and evaluation framework for scientific survey generation. arXiv preprint arXiv:2508.15658, 2025.   
[44] Weihang Su, Baoqing Yue, Qingyao Ai, Yiran Hu, Jiaqi Li, Changyue Wang, Kaiyuan Zhang, Yueyue Wu, and Yiqun Liu. Judge: Benchmarking judgment document generation for chinese legal system. In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’25), July 13–18, 2025, Padua, Italy, 2025.   
[45] Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang Wang, Pengjie Ren, Zhumin Chen, Dawei Yin, and Zhaochun Ren. Is chatgpt good at search? investigating large language models as re-ranking agents. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 14918–14937, 2023.   
[46] Yuqiao Tan, Shizhu He, Huanxuan Liao, Jun Zhao, and Kang Liu. Dynamic parametric retrieval augmented generation for test-time knowledge enhancement. arXiv preprint arXiv:2503.23895, 2025.

[47] Yichen Tang, Weihang Su, Yiqun Liu, and Qingyao Ai. Multi-field tool retrieval. arXiv preprint arXiv:2602.05366, 2026.   
[48] Yiteng Tu, Shuo Miao, Weihang Su, Yiqun Liu, and Qingyao Ai. Analytical search. arXiv preprint arXiv:2602.11581, 2026.   
[49] Yiteng Tu, Weihang Su, Yujia Zhou, Yiqun Liu, and Qingyao Ai. Robust fine-tuning for retrieval augmented generation against retrieval defects. In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 1272–1282, 2025.   
[50] Changyue Wang, Weihang Su, Qingyao Ai, and Yiqun Liu. Improve large language model systems with user logs. arXiv preprint arXiv:2602.06470, 2026.   
[51] Changyue Wang, Weihang Su, Qingyao Ai, and Yiqun Liu. Joint evaluation of answer and reasoning consistency for hallucination detection in large reasoning models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 33377–33385, 2026.   
[52] Changyue Wang, Weihang Su, Qingyao Ai, Yichen Tang, and Yiqun Liu. Knowledge editing through chain-of-thought. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 10684–10704, 2025.   
[53] Changyue Wang, Weihang Su, Qingyao Ai, Yujia Zhou, and Yiqun Liu. Decoupling reasoning and knowledge injection for in-context knowledge editing. In Findings of the Association for Computational Linguistics: ACL 2025, pages 24543–24562, 2025.   
[54] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023.   
[55] Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al. A survey on large language model based autonomous agents. Frontiers of Computer Science, 18(6):186345, 2024.   
[56] Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff, Defu Lian, and Jian-Yun Nie. Cpack: Packed resources for general chinese embeddings. In Proceedings of the 47th international ACM SIGIR conference on research and development in information retrieval, pages 641–649, 2024.   
[57] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.   
[58] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In The eleventh international conference on learning representations, 2022.   
[59] Z.ai. Glm-5.1: Towards long-horizon tasks. https://z.ai/blog/glm-5.1, April 2026. Accessed: 2026-04-20.   
[60] Yuchen Zhuang, Yue Yu, Kuan Wang, Haotian Sun, and Chao Zhang. Toolqa: A dataset for llm question answering with external tools. Advances in Neural Information Processing Systems, 36:50117–50143, 2023.   
[61] Terry Yue Zhuo, Minh Chien Vu, Jenny Chim, Han Hu, Wenhao Yu, Ratnadira Widyasari, Imam Nur Bani Yusuf, Haolan Zhan, Junda He, Indraneil Paul, et al. Bigcodebench: Benchmarking code generation with diverse function calls and complex instructions. arXiv preprint arXiv:2406.15877, 2024.

# A Dataset-Specific Construction Details

This appendix provides the dataset-specific construction details underlying SRA-Bench. While Section 3.2 presents the benchmark construction pipeline at a unified level, the actual supervision signals differ markedly across source datasets. For example, reusable capabilities may be anchored by theorem names, logic patterns, tool workflows, medical calculator types, mathematical concepts, or software libraries. To make the construction process fully transparent, we therefore unpack the pipeline separately for each dataset. For every source benchmark, we describe the full path from raw source-side annotations to finalized gold skills. This includes the available source materials, the dataset-specific inputs used to construct an initial LLM draft, the expert revision process that converts the draft into a reusable, leakage-controlled skill artifact, and concrete construction examples that illustrate the annotation results. These details are important for understanding how heterogeneous instance-level signals are abstracted into a unified form of gold-skill that is suitable for retrieval, incorporation, and end-to-end evaluation in the SRA setting.

Beyond documentation, this appendix also serves as evidence for a central design choice in SRA-Bench: gold skills are not directly inherited from the source datasets, but manually constructed from structured supervision signals into standalone capability artifacts. By presenting the construction procedure at the per-dataset level, we aim to make this process easier to inspect, verify, and reproduce.

# A.1 TheoremQA

Each instance in TheoremQA contains a question, a theorem annotation identifying the relevant theorem, and evaluation metadata including the ground-truth answer and its answer type (float, integer, bool, list, or option). The original dataset contains approximately 800 instances spanning 334 theorems. We remove image-dependent questions and retain the text-only subset, yielding 747 instances associated with 320 theorems.

The retained theorems span four STEM domains: mathematics $( 5 7 \% )$ , EECS $( 1 7 \% )$ , physics $( 1 5 \% )$ , and finance $( 1 1 \% )$ . Each theorem is further tagged with a field and subfield (e.g., Mathematics / Combinatorics), which provides useful annotation-level context during skill construction. Each instance is associated with exactly one theorem, and each theorem is mapped to one gold skill.

# A.1.1 LLM Draft Generation

For each theorem, the LLM receives the theorem name, its field and subfield, all associated questions and answers, and domain-appropriate reference pages (e.g., Wikipedia and MathWorld1 for mathematics topics) as input. Since the dataset does not provide solution steps, the draft must infer a preliminary application procedure from the theorem label, the distribution of associated questions, and external reference materials. The questions and answers are used only to help the LLM characterize the scope of application during drafting; any answer-revealing content, benchmark-specific shortcuts, or copied examples are removed during expert revision under the leakage-control principle described in Section 3.2. The following prompt template is used:

# Prompt Template for TheoremQA

You are a STEM education expert. Given a theorem and a set of problems that apply it, write a reusable skill document that teaches how to apply this theorem to solve new problems.

```txt
Theorem: {theorem_name}  
Field: {field} Subfield: {subfield}  
Reference material:  
{reference_pages}  
Problems applying this theorem (each with its ground truth answer):  
{questions_and Answers}  
Based on the above materials, write a skill document with the following structure: 
```

1. name: Canonical theorem name.   
2. description: One sentence summarizing what the theorem computes or states.   
3. Content (Markdown):

• Core principle or theorem statement.   
• When to recognize that this theorem applies.   
• Step-by-step application procedure.   
• Common errors or pitfalls.   
• One worked example using a new problem (not from the problems above).

# A.1.2 Expert Revision

Beyond the general revision principles described in Section 3.2, TheoremQA requires special attention to two recurring failure modes.

• Procedural under-specification. As the source data do not provide solution steps, LLM drafts often explain what a theorem states but not how to apply it in problem solving. As a result, many drafts resemble encyclopedia-style summaries, listing formulas and variable definitions without a clear procedure for application. During expert revision, such drafts are restructured into stepby-step skills, with explicit identification of inputs, case distinctions, and computation templates where needed.   
• Confusion with closely related concepts. Many STEM theorems have neighboring variants or closely related formulas that are easy to confuse in practice (e.g., Lah numbers vs. Stirling numbers, population vs. sample covariance). LLM drafts may mention such related concepts without clearly distinguishing the conditions under which they apply. Expert revision, therefore, adds explicit disambiguation cues, comparison rules, and recognition patterns to reduce misapplication.

# A.1.3 Construction Examples

We present two complete construction examples from different STEM domains.

Example 1: Dividend Discount Model (DDM). DDM has three common variants (zero growth, constant growth, and multi-stage); correctly selecting and applying the right variant is a challenge.

# Source Material

Theorem: Dividend Discount Model (DDM)

Field: Finance Subfield: Asset pricing

2 of 5 associated instances:

Q: If a stock pays a $\$ 5$ dividend this year, and the dividend has been growing $6 \%$ annually, what will be the stock’s intrinsic value, assuming a required rate of return of $12 \%$ ?

A: 88.33 (float)

Q: CheckMate forecasts that its dividend will grow at $20 \%$ per year for the next four years before settling down at a constant $8 \%$ forever. Dividend (current year, $2 0 1 6 ) = \$ 12$ ; expected rate of return $= 1 5 \%$ . What is the fair value of the stock now?

A: 273.0 (float)

# LLM Drafted Skill

name: Dividend Discount Model (DDM) description: Value stocks using dividend discount models including the Gordon Growth Model.

# Dividend Discount Model (DDM)

The DDM values a stock as the present value of all future dividends: $\begin{array} { r } { P _ { 0 } = \sum D _ { t } / ( 1 + r ) ^ { t } } \end{array}$ .

Zero Growth (Preferred Stock): $P _ { 0 } = D / r$

Constant Growth (Gordon Growth Model):

$$
P _ {0} = \frac {D _ {1}}{r - g} = \frac {D _ {0} \times (1 + g)}{r - g}
$$

where $D _ { 1 }$ is next period’s dividend. Requires $r > g$

Multi-Stage Growth: For high initial growth $g _ { 1 }$ for $_ n$ years transitioning to stable growth $g _ { 2 }$ , compute dividends during the high-growth phase and add the terminal value $\mathrm { T V } _ { n } = D _ { n + 1 } / ( r - g _ { 2 } )$ .

▷ No step-by-step procedure for multi-stage computation

Worked Example: A stock pays $\$ 5$ , growing $6 \%$ annually, required return $12 \%$ . $P _ { 0 } = 5 \times 1 . 0 6 / ( 0 . 1 2 -$

$0 . 0 6 ) = 8 8 . 3 3$

▷ Same parameters as an input instance

[. . . ]

# Golden Skill

name: Dividend Discount Model (DDM)

description: Valuing stocks using constant-growth and multi-stage dividend discount models, including terminal value computation.

# Dividend Discount Model (DDM)

The intrinsic value of a stock equals the present value of all future dividends: $\textstyle P _ { 0 } = \sum D _ { t } / ( 1 { + } r ) ^ { t }$ . The formula simplifies according to the dividend growth pattern.

# Three Common Cases:

• Zero growth (preferred stock): $P _ { 0 } = D / r$ .   
• Constant growth (Gordon): $P _ { 0 } = D _ { 1 } / ( r - g )$ , where $D _ { 1 } = D _ { 0 } ( 1 { + } g )$ .

Common error: using $D _ { 0 }$ instead of $D _ { 1 }$ in the numerator.

• Multi-stage: high growth $g _ { 1 }$ for $_ n$ years, then stable growth $g _ { 2 }$ forever.

# Multi-Stage Step-by-Step Procedure:

1. Compute each high-growth dividend: $D _ { t } = D _ { 0 } ( 1 { + } g _ { 1 } ) ^ { t }$ for $t = 1 , \ldots , n$   
2. Discount each to present: $\mathrm { P V } _ { t } = D _ { t } / ( 1 { + } r ) ^ { t }$   
3. Terminal value at year n $\therefore \mathrm { T V } _ { n } = D _ { n } ( 1 + g _ { 2 } ) / ( r - g _ { 2 } )$

Note: the numerator is $D _ { n + 1 } = D _ { n } ( 1 { + } g _ { 2 } )$ , not $D _ { n }$

4. Discount terminal value: $\mathrm { P V } ( \mathrm { T V } ) = \mathrm { T V } _ { n } / ( 1 + r ) ^ { n }$ .   
5. Sum: $P _ { 0 } = \sum \mathrm { P V } _ { t } + \mathrm { P V } ( \mathrm { T V } )$ .

# Critical Pitfalls:

• Wrong terminal value base: TV uses $D _ { n + 1 }$ , not $D _ { n }$   
• Missing growth step: given $D _ { 0 }$ , the first dividend is $D _ { 1 } = D _ { 0 } ( 1 { + } g )$ , not $D _ { 0 }$   
• Rate format: if $r$ and $g$ are given as percentages (e.g., $12 \%$ ), convert to decimals (0.12) before computing $r - g$

Worked Example: $D _ { 0 } = 4$ , $g _ { 1 } = 1 8 \%$ for 3 years, then $g _ { 2 } = 6 \%$ , $r = 1 4 \%$ . $D _ { 1 } = 4 . 7 2$ , $D _ { 2 } = 5 . 5 7$

$D _ { 3 } ~ = ~ 6 . 5 7$ ; PV of dividends $= 1 2 . 8 6$ . $D _ { 4 } ~ = ~ 6 . 5 7 \times 1 . 0 6 ~ = ~ 6 . 9 6$ ; $\mathrm { T V _ { 3 } } ~ = ~ 6 . 9 6 / 0 . 0 8 ~ = ~ 8 7 . 0$

$\mathrm { P V } ( \mathrm { T V } ) = 5 8 . 7$ . $P _ { 0 } = 1 2 . 8 6 + 5 8 . 7 \approx 7 1 . 6$

Revision summary. (1) The LLM draft correctly presents the DDM formulas for all three cases, but treats multi-stage growth in a single sentence without a structured computation procedure. The expert restructured the skill with an explicit five-step multi-stage template and documented the critical $D _ { n + 1 }$ vs. $D _ { n }$ pitfall by consulting the Wikipedia DDM reference. (2) The draft’s example uses parameters identical to an input instance. The expert replaced it with a newly constructed multi-stage example.

Example 2: Lah Numbers. Lah numbers count partitions of a set into non-empty ordered lists, and are frequently confused with the closely related Stirling numbers of the second kind.

# Source Material

Theorem: Lah Numbers

Field: Mathematics Subfield: Combinatorics

2 of 6 associated instances:

Q: How many ways are there to divide a set of 8 elements into 5 non-empty ordered subsets?

A: 11760 (integer)

Q: A group of 7 people is to be divided into 3 committees. Within each committee, people are ranked in a certain order. In how many ways can this be done?

A: 12600 (integer)

# LLM Drafted Skill

name: Lah Numbers

description: Compute Lah numbers for counting partitions of a set into non-empty ordered subsets.

# Lah Numbers

The Lah number $L ( n , k )$ counts the number of ways to partition a set of $n$ elements into $k$ non-empty ordered subsets (sequences).

Formula: $\begin{array} { r } { L ( n , \dot { k } ) = \frac { n ! } { k ! } \binom { n ^ { \prime } - 1 } { k - 1 } } \end{array}$

# Procedure:

1. Identify $n$ (total elements) and $k$ (number of groups).   
2. Compute $\begin{array} { r } { L ( n , k ) = \frac { n ! } { k ! } \times \binom { n - 1 } { k - 1 } } \end{array}$ .

Special Cases: $L ( n , 1 ) = n ! ; L ( n , n ) = 1$ $L ( n , n ) = 1$ .

▷ No distinction from Stirling numbers; models confuse ordered vs. unordered partitions

Worked Example: Divide 8 elements into 5 ordered subsets: $\begin{array} { r } { L ( 8 , 5 ) = \frac { 8 ! } { 5 ! } \times \binom { 7 } { 4 } = 3 3 6 \times 3 5 = 1 1 7 6 0 . } \end{array}$

▷ Same parameters as an input instance

# Golden Skill

name: Lah Numbers

description: Computing Lah numbers for counting ordered partitions of a set into non-empty ordered sequences.

# Lah Numbers

The Lah number $L ( n , k )$ counts the number of ways to partition a set of $n$ elements into $k$ non-empty ordered subsets (sequences), where the order within each subset matters.

Formula: $\begin{array} { r } { L ( n , k ) \stackrel { * } { = } \frac { n ! } { k ! } \binom { n - 1 } { k - 1 } } \end{array}$

# Step-by-Step Procedure:

1. Identify $n$ (total elements) and $k$ (number of groups).   
2. Verify the problem asks for ordered groups (sequences, ranked committees, lists)—not unordered subsets.   
3. Compute $\begin{array} { r } { L ( n , k ) = \frac { n ! } { k ! } \times \binom { n - 1 } { k - 1 } } \end{array}$   
4. Sanity check: $L ( n , 1 ) = n ! , L ( n , n ) = 1$ $L ( n , n ) = 1$ .

Distinguishing from Related Problems:   

<table><tr><td>Concept</td><td>Groups ordered?</td><td>Within-group order?</td></tr><tr><td>Lah number L(n,k)</td><td>No</td><td>Yes</td></tr><tr><td>Stirling 2nd kind S(n,k)</td><td>No</td><td>No</td></tr><tr><td>k! × S(n,k)</td><td>Yes</td><td>No</td></tr></table>

Key distinction: “subsets” r “groups” (unordered within) Stirling numbers. “Sequences,” “ranked

Worked Example: Divide 4 books into 2 non-empty ordered shelves: $\begin{array} { r } { L ( 4 , 2 ) = \frac { 4 ! } { 2 ! } \times \binom { 3 } { 1 } = 1 2 \times 3 = 3 6 } \end{array}$

Revision summary. (1) The LLM draft provides the correct formula but does not distinguish Lah numbers from the closely related Stirling numbers of the second kind, which count unordered partitions. The expert added the comparison table and recognition patterns, which are critical since problems with nearly identical phrasing (e.g., “groups” vs. “ordered groups”) require different formulas. (2) The draft’s worked example uses parameters identical to an input instance (n=8, $k { = } 5$ ). The expert replaced it with a smaller, independently constructed example ( $\mathrm { { \it n } = 4 }$ , $k { = } 2$ ) that is easier to verify by hand.

# A.2 LogicBench

Each LogicBench instance consists of a natural-language question that poses a logical reasoning problem and a ground-truth answer: either yes/no for binary question answering (BQA, $71 \%$ 5 or choice_N for four-option multiple-choice question answering (MCQA, $2 9 \%$ ). The original benchmark contains 25 logic patterns across three categories: 9 propositional, 8 first-order, and 8 non-monotonic. Among them, six first-order patterns are direct counterparts of propositional ones (e.g., first-order modus tollens tests the same underlying inference rule as propositional modus tollens). We therefore exclude these redundant variants and retain 19 distinct patterns, yielding 760 instances in total. Each retained pattern corresponds to a named inference rule, such as modus tollens, disjunctive syllogism, or default reasoning with exceptions, and is annotated with a logic category. These pattern names and category labels serve as the primary structured signals for goldskill construction. LogicBench does not provide intermediate reasoning steps, formal derivations, or worked explanations; it contains only the problem statement and the final answer. Each instance is associated with exactly one logic pattern, and each pattern is mapped to one gold skill.

# A.2.1 LLM Draft Generation

For each pattern, the LLM receives the pattern name, its logic category, all associated questions and answers, and relevant Stanford Encyclopedia of Philosophy2 pages as input. Because the dataset provides no worked solutions, the LLM must synthesize the application procedure from its own knowledge of the inference rule, using the questions to understand the scope of application and the reference pages for canonical definitions. The following prompt template is used:

# Prompt Template for LogicBench

You are a logic education expert. Given an inference rule and a set of problems that test it, write a reusable skill document that teaches how to apply this rule to solve new problems.

Inference Rule: {pattern_name} Logic Category: {logic_category} Reference material: {reference_pages} Problems testing this rule (each with its ground truth answer): {questions_and_answers} Based on the above materials, write a skill document with the following structure:

1. name: Canonical inference rule name.   
2. description: One sentence summarizing what the rule states and its scope.   
3. Content (Markdown):

• Formal statement of the inference rule.   
• When to recognize that this rule applies.   
• Step-by-step application procedure.   
• Common errors or pitfalls.   
• One worked example using a new problem (not from the problems above).

# A.2.2 Expert Revision

Beyond the general revision principles described in $\ S 3 . 2$ , LogicBench requires two dataset-specific refinements:

• Narrative premise extraction: In LogicBench, logical premises are often embedded in naturallanguage narratives that include scene-setting details, hedged expressions (e.g., “typically,” “often,” “likely”), and other distractor content. While LLM drafts usually capture the target inference rule itself, they often under-specify how to identify the premises that are logically operative in the task. The expert therefore adds structured procedures for premise extraction, including how to separate reasoning-relevant premises from background context and how to interpret qualified conditionals in a rule-like manner.   
• Related fallacy disambiguation: LogicBench also contains reasoning patterns that are easily confused with closely related fallacies (e.g., modus tollens vs. denying the antecedent, modus ponens vs. affirming the consequent). Although LLM drafts describe the target rule, they often do not explicitly contrast it with these neighboring invalid forms. The expert therefore adds distinction sections with recognition criteria to reduce confusion and prevent misapplication.

# A.2.3 Construction Examples

We present two complete construction examples from different logic categories.

Example 1: Modus Tollens. Modus tollens is a fundamental deductive rule appearing in both propositional and first-order logic; the key challenge lies in correctly extracting premises from narrative-rich problem statements.

# Source Material

Pattern: Modus Tollens

Logic Category: Propositional Logic, First-Order Logic 2 of 40 associated instances:

Q: Daniel woke up on a gloomy morning and realized that he had forgotten to take his umbrella. Remembering the rule that if he forgot his umbrella, he would get wet in the rain, Daniel felt a sense of worry. However, Daniel managed to stay dry. Does this infer that Daniel forgot to take his umbrella? A: no (BQA)   
Q: Isabella had been working hard on her project, knowing that if she finished it on time, she would receive a bonus. However, Isabella realized that she wouldn’t be receiving a bonus. Does this entail that Isabella didn’t finish her project on time? A: yes (BQA)

# LLM Drafted Skill

name: Modus Tollens description: Derive that the antecedent is false from a conditional and the denial of its consequent.

# Modus Tollens

Modus tollens (denying the consequent) derives a negative conclusion by contrapositive reasoning: Form: If P, then Q. Q is false. Therefore, P is false.

Symbolic: $P  Q$ , $\neg Q \ \vdash \ \neg P$

# Application Procedure:

1. Identify the conditional premise: extract antecedent A and consequent C.   
2. Confirm that C is negated in the premises.   
3. Conclude $\neg \mathbf { A }$ . The antecedent must be false.

▷ No guidance on narrative context or related fallacies

Worked Example: If someone forgets their umbrella, they get wet in the rain. The person stayed dry. Therefore, they did not forget their umbrella. ▷ Same scenario as an input instance [. . . ]

# Golden Skill

name: Modus Tollens

description: Denying the consequent: if P implies Q and Q is false, then P is false. Covers propositional and first-order logic.

Modus Tollens

Modus tollens is the valid argument form: $P  Q$ , $\neg Q \ \vdash \ \neg P$ .

Application Procedure:

1. Extract the conditional: Identify A (antecedent) and C (consequent) from “if A then C.”   
2. Check the second premise: Confirm that C is negated—“not C,” “C did not occur,” or equivalent.   
3. Apply modus tollens: Conclude $\neg \mathbf { A }$ .   
4. Answer the question: Compare the question’s target against ¬A.

# Common Errors:

• Narrative context confusion. A problem may describe A as having happened (scene-setting), then state $\lnot \mathbf { C }$ . The narrative mention is not a logical premise—the operative premises are the conditional rule and the denial of C.   
• Affirming the consequent (invalid): “If A then C; C; therefore A” does not follow.   
• Denying the antecedent (invalid): “If A then C; ¬A; therefore ¬C” does not follow. Modus tollens requires negation of the consequent, not the antecedent.

Worked Example: “If a server authenticates successfully, encrypted data transfer begins.” “Encrypted transfer has not begun.”

$\mathbf { A } =$ successful authentication, $\mathbf { C } =$ encrypted transfer. $\lnot \mathbf { C }$ confirmed $ \neg \mathrm { A }$ : the server did not authenticate.

BQA: “Did the server authenticate?” $\bf { \Gamma } \to \bf { n } \bf { 0 }$ .

Revision summary. (1) The LLM draft correctly formalizes the inference rule but does not address the challenge of extracting premises from narrative scenarios. LogicBench problems often embed the antecedent in scene-setting narrative (e.g., “Daniel realized that he had forgotten his umbrella”), creating an apparent contradiction with the modus tollens conclusion (¬A). The expert added narrative context guidance and contrasted modus tollens with the commonly confused invalid forms (denying the antecedent, affirming the consequent). (2) The draft’s worked example reuses the umbrella scenario from an input instance. The expert replaced it with a newly constructed server authentication example.

Example 2: Default Reasoning with Irrelevant Information. This non-monotonic reasoning pattern tests whether the model can filter out logically irrelevant premises that do not affect the application of the default rule.

# Source Material

Pattern: Default Reasoning with Irrelevant Information

Logic Category: Non-Monotonic Logic

2 of 40 associated instances:

Q: Snowflakes and crystals, which are typically white, often give the impression of being soft. However, snowflakes are not actually soft. Most white objects tend to possess a soft texture. Furthermore, snowflakes have a distinct six-sided structure. Does this entail that crystals are soft?

A: yes (BQA)

Q: Pens and pencils are commonly used in classrooms. However, pens are not used in classrooms. Pens come in different colors. What conclusion would be most appropriate?

choice_1: whiteboards are commonly used choice_2: markers are preferred over pens

choice_3: pencils are frequently used in classrooms choice_4: pencils are used on the blackboard

A: choice_3 (MCQA)

# LLM Drafted Skill

name: Default Reasoning with Irrelevant Information description: Apply default rules correctly while filtering out irrelevant premises.

# Default Reasoning with Irrelevant Information

In default reasoning, a general rule applies to all members of a class unless a specific exception is stated. Additional information that does not modify the exception should not affect the conclusion.

# Application Procedure:

1. Identify the default rule and the exception.   
2. Apply the default to non-exception members.

▷ No structured procedure for identifying and filtering distractors

Worked Example: Snowflakes and crystals are typically soft. Snowflakes are an exception. Snowflakes have six sides. Are crystals soft? Yes—the six-sided shape is irrelevant. ▷ Same entities as an input instance [. . . ]

# Golden Skill

name: Default Reasoning with Irrelevant Information

description: Applying a default rule correctly by filtering out logically irrelevant premises that do not affect the scope of the default.

# Default Reasoning with Irrelevant Information

In default reasoning, a premise set may contain information that is logically irrelevant to the conclusion. The ability to filter such distractors is as important as applying the rule itself.

# Step-by-Step Procedure:

1. Identify the default rule—look for “typically,” “normally,” “usually.”   
2. Identify the exception—which individual is flagged as deviating from the default?   
3. Actively set aside all additional information that does not modify the default or re-designate any individual as an exception.   
4. Apply the default rule to the questioned individual, who remains a non-exceptional member.   
5. Conclude accordingly.

# Why Distractors Mislead:

• Relevance inflation: assuming every provided fact must factor into the conclusion.   
• Narrative completion: letting the overall story shade the conclusion beyond what the logical structure supports.

Worked Example: “Birds typically can fly. Penguins cannot fly. Penguins live in Antarctica. Can eagles fly?”

Step 1: Default $=$ birds can fly. Step 2: Exception $=$ penguins. Step 3: “live in Antarctica” is irrelevant—set aside. Step 4: Eagles are non-exceptional birds. Step 5: yes.

Revision summary. (1) The LLM draft correctly states the default reasoning principle. Still, it reduces the procedure to two generic steps, without teaching how to identify and filter distractor premises—the core challenge of this pattern. The expert restructured the skill around an explicit five-step filtering procedure and added a “Why Distractors Mislead” section explaining the cognitive biases (relevance inflation, narrative completion) that cause models to incorporate irrelevant facts. (2) The draft’s worked example reuses entities from an input instance (snowflakes, crystals, six-sided structure). The expert replaced it with a newly constructed example.

# A.3 ToolQA

The original ToolQA benchmark covers eight data domains. We exclude GSM8K, which primarily evaluates mathematical reasoning rather than tool-augmented question answering over an external reference corpus, and retain the remaining seven domains, resulting in 1,430 instances. Each instance poses a natural-language question grounded in one domain and requires an agent to interact with domain-specific external tools to retrieve, manipulate, or compute the answer. ToolQA provides a ground-truth answer string for each instance, but does not include solution traces, worked explanations, or reusable skill annotations. In our SRA-Bench, we define 14 annotation groups by crossing the seven retained domains with two difficulty levels. Easy questions generally require extracting a single piece of information from the external corpus, whereas hard questions require more compositional tool use, such as aggregation, comparison, filtering over multiple records, or derived computation. Each annotation group is represented by one gold skill.

The retained domains are associated with three types of tool-use capabilities. The tabular domains, including flight, coffee, Airbnb, and Yelp, require structured database operations via LoadDB, FilterDB, and GetValue. The text domains, Agenda and Scirex, require semantic retrieval over textual corpora through RetrieveAgenda and RetrieveScirex. The graph domain, DBLP, requires citation-network reasoning through LoadGraph, NodeCheck, NeighbourCheck, and EdgeCheck. All retained domains can additionally use general-purpose tools, including Calculate, PythonInterpreter, and SQLInterpreter.

Crucially, the ToolQA repository publishes dataset generation code containing template-based question patterns for each domain $. \times$ difficulty. Each template specifies a natural-language question pattern, the target data fields, and the answer computation procedure. These templates, together with tool specifications and data schemas, provide the annotation-level context for skill construction and constitute construction-time knowledge beyond what the inference-time agent observes from the task prompt alone. Each instance maps to exactly one domain $\times$ difficulty annotation, and each annotation is used to construct one gold skill.

# A.3.1 LLM Draft Generation

For each domain $\times$ difficulty annotation, the LLM receives the domain name, difficulty level, tool specifications with signatures, data schema, question-generation templates from the published dataset generation code, and all associated questions and answers as input. The question templates are particularly important because they reveal the systematic mapping from natural-language question patterns to data operations and answer formats. This information is not available to the inference-time agent from the task prompt alone, but is useful for writing procedural guidance that generalizes across instances. The following prompt template is used:

# Prompt Template for ToolQA

You are a data analysis expert. Given a data domain, available tools, data schema, question templates, and QA examples, write a reusable skill document that teaches how to answer questions about this domain using the provided tools.

Domain: {domain_name} Difficulty: {difficulty} Available tools: {tool_specifications} Data schema: {schema} Question templates (from dataset generation code): {question_templates} QA pairs: {questions_and_answers}

Based on the above materials, write a skill document with the following structure:

1. name: Domain and query type.   
2. description: One sentence summarizing the skill’s scope.   
3. Content (Markdown):

• Data schema overview with key fields.   
• Step-by-step tool workflow for each question type.   
• Answer format conventions.   
• Common errors or pitfalls.

# A.3.2 Expert Revision

For ToolQA, expert revision focuses on making the drafted skills operationally usable by an inferencetime agent. Although the agent is provided with tool descriptions and general-purpose few-shot ReAct demonstrations, these resources do not specify how to chain tools for different question patterns. The initial LLM drafts can often identify the relevant columns or fields from the question templates, but they rarely provide executable workflows, such as when a query should be handled by FilterDB versus SQLInterpreter, or when document retrieval should be followed by PythonInterpreter for aggregation beyond the retriever’s top-3 results. Therefore, experts revise the skills by adding question-type-specific workflows that describe the intended sequence of tool calls, the required arguments, and the decision points for switching between tools.

Experts also add guidance on domain-specific data representations that are not explicitly documented in the original tool descriptions or schemas. For example, flight times are stored as HHMM-style numeric values (e.g., 2143.0 denotes 21:43), flight identifiers concatenate carrier codes and flight numbers (e.g., AA2319 corresponds to carrier AA and flight number 2319), and date formats may differ between natural-language questions and textual corpora. Therefore, the revised skills include lightweight decoding and normalization instructions that cover how to interpret encoded values, convert between formats, and construct valid tool inputs before execution.

# A.3.3 Construction Examples

Example 1: Flights (Single-Record Lookup).

# Source Material

Domain: flights (US domestic)

Difficulty: Easy

Tools: LoadDB[flights], FilterDB[cond, ...], GetValue[column]

Key schema fields: FlightDate, Origin, Dest, IATA_Code_Marketing_Airline,

Flight_Number_Marketing_Airline, DepTime, ArrTime, AirTime, TaxiIn, Cancelled, . . .

Question templates (3 of 10, from dataset generation code):

• “What was the departure time of {carrier}{number} from {Origin} to {Dest} on {date}?” → DepTime   
• “How many minutes did {carrier}{number} take to taxi in on {date}?” TaxiIn   
• “Was {carrier}{number} from {Origin} to {Dest} cancelled on {date}?” Cancelled

2 of 100 associated instances:

Q: What was the departure time of the AA2319 flight from MIA to LAS on 2022-06-05? A: 21:43

Q: How many minutes did the DL3538 flight take to taxi in on 2022-06-26? A: 3

# LLM Drafted Skill

name: US Domestic Flight Data — Easy Queries

description: Guide for answering single-record lookup questions about US domestic flight data.

# Available Tools:

• LoadDB[flights]: Load the flight database.   
• FilterDB[cond1, cond2, ...]: Filter rows matching conditions.   
• GetValue[column]: Retrieve values from the filtered result.

# Question Types and Target Columns:

• Departure/arrival time DepTime / ArrTime   
• Flight duration AirTime   
• Taxi time TaxiIn / TaxiOut   
• Cancellation status Cancelled

▷ Maps templates to columns but no step-by-step tool workflow or flight ID parsing

Worked Example: Q: “What was the departure time of the AA2319 flight from MIA to LAS on 2022-06- 05?” Filter by flight AA2319, origin MIA, destination LAS, date 2022-06-05. Get DepTime. ▷ Same question; HHMM format not documented

# Golden Skill

name: US Domestic Flight Data — Single-Record Lookup Guide description: Structured workflows for answering single-record lookup questions over the US domestic flight database, including schema reference, flight ID parsing, and time format interpretation.

Flight Identifier Convention: A reference like AA2319 encodes two separate columns: IATA_Code_Marketing_Airline ${ \tt \tt = A A }$ and Flight_Number_Marketing_Airline=2319. FilterDB requires them as separate conditions.

# Standard Lookup Workflow:

1. LoadDB[flights]   
2. FilterDB[FlightDate=YYYY-MM-DD, Origin=XXX, IATA_Code_Marketing_Airline=XX, Flight_Number_Marketing_Airline $\fallingdotseq$ NNNN]   
3. GetValue[target_column]

Use only the identifying fields given in the question; omitting unneeded fields is fine.

Time Column Interpretation: Time columns (DepTime, ArrTime, etc.) store values as numeric HHMM without a colon—e.g., 2143.0 represents 21:43, 835.0 represents 08:35. Strip the .0 suffix and insert a colon when reporting.

When FilterDB Is Insufficient: FilterDB compares all values as strings (lexicographic), producing incorrect results for numeric columns. For numeric comparisons, use SQLInterpreter with CAST(column AS REAL).

[. . . ]

Revision summary. (1) The LLM draft correctly maps question types to target columns (using the templates) but does not specify the multi-step tool invocation workflow. The draft says “filter by flight AA2319” without explaining that AA2319 must be split into carrier code (AA) and flight number (2319) as separate FilterDB conditions—a parsing step that the agent must perform before any filter can succeed. The expert added a structured three-step workflow (LoadDB FilterDB GetValue), the flight identifier convention, and guidance on when to switch from FilterDB to SQLInterpreter. (2) The draft does not document the HHMM time storage convention: time columns store numeric HHMM values (e.g., 2143.0 for 21:43), which the agent must decode to correctly interpret GetValue results and construct time-based queries. (3) The worked example reuses a question from an input instance.

Example 2: Agenda (Hard) — Aggregation Queries. The agenda-hard skill covers populationlevel queries (counting events, finding free meeting slots) over a personal calendar corpus; the key challenge is recognizing that the semantic retrieval tool is fundamentally insufficient and switching to programmatic corpus processing.

# Source Material

Domain: agenda (personal calendars) Difficulty: Hard Tools: RetrieveAgenda[query] (returns top 3 passages), PythonInterpreter[code], Calculate[expr]

Corpus format: JSONL; each event: “On [date], [person] has [event] at [location] from [start] to [end].” Question templates (3 of 5, from dataset generation code):

• “How many events happen on {date}?” count all events on date   
• “How many people are unavailable between {start} and {end} on {date}?” count with time overlap   
• “When should I schedule a meeting with {person} on {date}?” → comma-separated free slots

2 of 100 associated instances:

Q: How many events happen on 2022/09/25 in the agenda table?

A: 24

Q: When should I schedule a meeting with Alice from 9:00 AM to 6:00 PM on 2022/01/06 in the agenda table?

A: 9:00 AM-1:00 PM, 3:00 PM-6:00 PM

# LLM Drafted Skill

name: Agenda Event Data — Hard Queries description: Guide for answering aggregation and scheduling queries over the agenda event corpus.

# Available Tools:

• RetrieveAgenda[query]: Search for relevant agenda events.   
• Calculate[expression]: Perform arithmetic.

# Question Types:

1. Count events on a date retrieve events, count results.   
2. Count unavailable people in a time window retrieve events, count distinct people.   
3. Schedule a meeting retrieve person’s events, compute free slots.

▷ RetrieveAgenda returns only top 3 passages—cannot count all 24 events on a date

Worked Example: Q: “How many events happen on 2022/09/25?” RetrieveAgenda[events 2022/09/25] count retrieved passages. ▷ Same question; approach fundamentally wrong [. . . ]

# Golden Skill

name: Working with the Agenda Event Corpus: Aggregation and Population Queries description: Programmatic corpus processing for agenda queries that require counting, time-overlap filtering, or free-slot scheduling beyond the retriever’s top-3 limit.

Why RetrieveAgenda Cannot Answer Aggregation Queries: RetrieveAgenda returns only the top 3 passages by semantic similarity. This works for looking up a specific event but cannot support queries that require counting or enumerating across all matching records. A date may have 15–40 events; retrieving 3 will produce the wrong count.

# Using PythonInterpreter for Corpus-Wide Queries:

import json   
corpus_path $=$ "data/agenda/agenda descriptions_merged.jsonl" records $= []$ with open(corpus_path）as f: for line in f: if line.strip(): records.appendjson.loads(line))

Date Format Conversion: Questions use YYYY/MM/DD; corpus uses “Month Day, Year” (e.g., “September 25, 2022”). Convert before filtering.

Query Type Tool Summary:

<table><tr><td>Query type</td><td>Tool</td></tr><tr><td>Look up a specific event</td><td>RetrieveAgenda</td></tr><tr><td>Count events on a date</td><td>PythonInterpreter</td></tr><tr><td>Count people in time window</td><td>PythonInterpreter</td></tr><tr><td>Find free meeting slots</td><td>PythonInterpreter</td></tr></table>

Revision summary. (1) The LLM draft suggests using RetrieveAgenda for all question types, not recognizing that aggregation queries (counting events, finding free slots) require access to the entire corpus—not just the top 3 retrieved passages. The expert restructured the skill around the fundamental distinction between single-event lookup (RetrieveAgenda) and corpus-wide queries (PythonInterpreter), adding complete code recipes for loading the corpus, converting date formats, filtering by time overlap, and computing free meeting slots. (2) The draft does not address the date format mismatch: questions use YYYY/MM/DD while the corpus stores dates as “Month Day, Year” (e.g., “September 25, 2022”), requiring format conversion before records can be filtered programmatically. (3) The worked example reuses a question from an input instance.

# A.4 MedCalc-Bench

For MedCalc-Bench, we use the following fields from each instance: a patient note describing the clinical vignette, a question requesting the computation of a target clinical value, the calculator name, the ground-truth answer, and a ground-truth explanation. The ground-truth explanation provides a step-by-step walkthrough that extracts relevant values from the patient note and applies the calculator’s formula or scoring rules. The evaluation split contains 55 unique calculator names, with 20 test instances for each calculator. Each instance is associated with exactly one calculator, and we construct one gold skill for each calculator.

For skill construction, the ground-truth explanations serve as the primary source material because they explicitly identify the relevant clinical variables and demonstrate the calculation procedure. Compared with raw patient notes, these explanations provide a more information-dense basis for abstracting a reusable calculation skill. The calculator name provides the structured annotation that links each instance to its corresponding gold skill.

# A.4.1 LLM Draft Generation

For each calculator, the LLM receives the calculator name, all 20 ground-truth explanations, and the corresponding MDCalc3 reference page as input. The explanations provide worked examples from which the LLM abstracts a general procedure, while the MDCalc page provides the canonical formula specification and boundary conditions. The following prompt template is used:

# Prompt Template for MedCalc-Bench

You are a medical knowledge engineer. Given a set of worked calculation examples for a clinical calculator, write a reusable skill document that teaches how to apply this calculator to any patient.

Calculator name: {calculator_name} Reference page: {mdcalc_page} Worked examples (20 patients, each showing extracted values and step-by-step calculation): {explanations}

Based on the above materials, write a skill document with the following structure:

1. name: Canonical calculator name (include standard abbreviation if applicable).   
2. description: One sentence–-what the calculator computes and its clinical use.   
3. Content (Markdown):

• Brief clinical context (1–2 sentences).   
• Required inputs with expected units and common unit conversions.   
• Step-by-step computation procedure using symbolic variables.   
• A Python function compute_{name}(...) implementing the calculator.   
• One worked example using a new hypothetical patient (not from the examples above).

# A.4.2 Expert Revision

Beyond the general revision criteria, expert revision for MedCalc-Bench focuses on making each skill both logically complete and executable. Since instance-level explanations usually reveal only the branch taken by a particular patient case, LLM-generated drafts may miss rare or unobserved branches, such as dialysis-dependent creatinine overrides or BMI-specific handling for underweight patients. The expert thus cross-checks each drafted skill against all annotated instances and the original MDCalc reference, completing the missing conditional logic in both the textual procedure and the accompanying Python implementation. The expert also verifies the executable tool by

checking its outputs and correcting implementation details such as clamping rules, rounding behavior, and boundary conditions, ensuring that the final skill is operationally reliable rather than merely descriptive.

# A.4.3 Construction Examples

We present two construction examples illustrating how an LLM draft is revised into a finalized gold skill.

Example 1: MELD Na (UNOS/OPTN). The MELD Na score illustrates a complex calculator with multi-step conditional logic.

# Source Material

# Calculator: MELD Na (UNOS/OPTN)

Ground Truth Explanation (1 of 20 instances): The formula for computing the MELD Na is to first apply the following equation: M $\mathrm { E L D ( i ) } = 0 . 9 5 7 \times \ln ( \mathrm { C r } ) + 0 . 3 7 8 \times \ln ( \mathrm { b i l i r u b i n } ) + 1 . 1 2 0 \times \ln ( \mathrm { I N R } ) + 0 . 6 4 3 .$ If the MELD(i) is greater than 11 after rounding to the nearest tenth and multiplying the MELD(i) by 10, we apply the following equation: $\mathrm { M E L D } = \mathrm { M E L D } ( \mathrm { i } ) + 1 . 3 2 \times ( 1 3 7 - \mathrm { N a } ) - [ 0 . 0 3 3 \times \mathrm { M E L D } ( \mathrm { i } ) \times ( 1 3 7 - \mathrm { N a } ) ]$ $- \ : \mathrm { N a } ) ]$ . The MELD Na score is capped at 40. The concentration of Na is mEq/L, the concentration of bilirubin is $\mathrm { m g / d L }$ , and the concentration of creatinine is $\mathrm { m g / d L }$ . If the patient’s Na concentration is less than $1 2 5 ~ \mathrm { m E q / L }$ , we set it to $1 2 5 \mathrm { m E q / L }$ and if the patient’s the Na concentration is greater $1 3 7 \mathrm { m E q } / \mathrm { L }$ , we round it to $1 3 7 \mathrm { m E q } / \mathrm { L }$ . The concentration of creatinine is $2 . 0 \mathrm { m g / d L }$ . The patient has not went through dialysis at least twice in the past week. [. . . ] The patient’s bilirubin concentration is less than $1 . 0 \mathrm { m g / d L }$ , and so we set the bilirubin concentration to $1 . 0 \mathrm { m g / d L }$ . The patient’s INR is 1.59. The sodium concentration is greater than $1 3 7 ~ \mathrm { m E q / L }$ , and so we set the sodium concentration to $1 3 7 ~ \mathrm { m E q / L }$ . Applying the first equation gives us $0 . 9 5 7 \times \ln ( 2 . 0 ) + 0 . 3 7 8 \times \ln ( 1 . 0 ) + 1 . 1 2 0 \times \ln ( 1 . 5 9 ) + 0 . 6 4 3 = 1 . 8 2 5 7 .$ . Rounding to the nearest tenth makes the MELD(i) score 1.8. We then multiply by 10, making the MELD(i) score 18. Because the MELD(i) score is greater than 11, we then apply the second equation, giving us $1 8 + 1 . 3 2 \times$ $( 1 3 7 - 1 3 7 ) - [ 0 . 0 3 3 \times 1 . 8 2 5 7 \times ( 1 3 7 - 1 3 7 ) ] = 1 8$ . The MELD Na score is less than 40, and so we keep the score as it is. The patient’s MELDNa score, rounded to the nearest integer, is 18 points.

# LLM Drafted Skill

name: MELD Na (UNOS/OPTN)

description: Calculate the MELD Na score to assess end-stage liver disease severity for transplant prioritization.

# MELD Na (UNOS/OPTN)

The MELD Na score estimates 90-day mortality risk in patients with end-stage liver disease and is used for liver transplant prioritization.

Required Inputs: Creatinine (mg/dL), bilirubin $\mathrm { ( m g / d L ) }$ ), INR, sodium $( \mathrm { m E q } / \mathrm { L } )$ , dialysis status.

Step 1 — Value clamping: Bilirubin: if $< 1 . 0$ , set to 1.0. INR: if $< 1 . 0$ , set to 1.0. Creatinine: clamp to [1.0, 3.0]. Sodium: clamp to [125, 137]. ▷ Missing: dialysis $ C r = 3 . 0$

Step 2 — Compute MELD(i) $: \mathrm { M E L D ( i ) } = 0 . 9 5 7 \ln ( \mathrm { C r } ) + 0 . 3 7 8 \ln ( \mathrm { B i l i } ) + 1 . 1 2 0 \ln ( \mathrm { I N R } ) + 0 . 6 4 3$

Step 3 — Sodium adjustment: MELD Na = MELD(i)×10 + 1.32(137−Na) − 0.033 · MELD(i)×10 · (137−Na)

Cap at 40 and round to nearest integer. ▷ Missing: MELD(i) ≤ 11 bypass Tool:

def compute_meld_nacreatinine,bilirubin,indr,sodium,on_dialysis): if bilirubin $<  1.0$ :bilirubin $= 1.0$ if $\mathrm{inr} <   1.0$ : $\mathrm{inr} = 1.0$ creatinine $=$ max(1.0,mincreatinine,3.0)#no dialysis handling sodium $= \max (125$ min(sodium，137)) meld_i $= (0.957\ast$ math.logcreatinine） $+0.378\ast$ math.log(bilirubin) $+1.120\ast$ math.log(inr） $+0.643)$ meld_iScaled $=$ round(round(meld_i,1） $\ast 10$ ） meld_na $=$ (meld_iScaled $+1.32\ast$ (137 - sodium)-0.033 \*meld_iScaled $\ast$ (137 - sodium)) #always applied return round(min(meld_na，40))

```txt
[...] 
```

# Golden Skill

name: MELD Na (UNOS/OPTN)

description: Compute MELD Na score for liver transplant prioritization from creatinine, bilirubin, INR, and sodium.

# MELD Na (UNOS/OPTN)

The MELD Na score (Model for End-Stage Liver Disease with Sodium) is used to prioritize patients on the liver transplant waiting list. It extends the original MELD score by incorporating serum sodium to better predict 90-day mortality in end-stage liver disease.

Required Inputs: Creatinine (mg/dL), bilirubin (mg/dL), INR, sodium (mEq/L), dialysis status.

Step 1 — Value clamping before MELD(i):

• Creatinine: if on dialysis ( $2 2 \times$ /week), set to 3.0. Otherwise clamp to [1.0, 3.0].   
• Bilirubin: if $< 1 . 0$ , set to 1.0.   
• INR: if $< 1 . 0$ , set to 1.0.

Step 2 — Compute MELD(i):

$$
\mathrm {M E L D} (\mathrm {i}) = 0. 9 5 7 \ln (\mathrm {C r}) + 0. 3 7 8 \ln (\mathrm {B i l i}) + 1. 1 2 0 \ln (\mathrm {I N R}) + 0. 6 4 3
$$

Step 3 — Check sodium adjustment: Round MELD(i) to nearest tenth, multiply by 10. If ≤ 11: final score $\mathrm { \tau = M E L D ( i ) \times 1 0 }$ . If $> 1 1$ : proceed to Step 4.

Step 4 — Sodium clamping: Clamp Na to [125, 137] mEq/L.

Step 5 — Sodium adjustment:

$$
\mathrm {M E L D N a} = \mathrm {M E L D (i)} \times 1 0 + 1. 3 2 (1 3 7 - \mathrm {N a}) - 0. 0 3 3 \cdot \mathrm {M E L D (i)} \times 1 0 \cdot (1 3 7 - \mathrm {N a})
$$

Step 6 — Cap: If MELD $\mathrm { N a } > 4 0$ , set to 40. Round to nearest integer.

Tool:

def compute_meld_na(creatinine,bilirubin,indr,sodium,on_dialysis): if on_dialysis: creatinine $= 3.0$ else: creatinine $\equiv$ max(1.0,mincreatinine,3.0)) if bilirubin $<  1.0$ : bilirubin $= 1.0$ if inr $<  1.0$ :indr $= 1.0$ meld_i $= (0.957\ast$ math.logcreatinine) $+0.378\ast$ math.log(bilirubin) $+1.120\ast$ math.log(inr) $+0.643)$ meld_iScaled $\equiv$ round(round(meld_i,1） $\ast 10$ sodium $= \max (125$ min(sodium，137))   
if meld_iScaled>11: weld_na $=$ (meld_iScaled $+1.32\ast$ (137 - sodium) $-0.033\ast$ meld_iScaled $\ast$ (137 - sodium))   
else: weld_na $=$ meld_iScaled   
return round(min(meld_na,40))

Worked Example: Patient: 58-year-old man with cirrhosis. Cr 1.8, Bili 3.5, INR 1.4, Na 130. Not on dialysis.

Step 1: All values in range, no clamping needed.

Step 2–3: TOOL_CALL: compute_meld_na(1.8, 3.5, 1.4, 130, False) → 25

The patient’s MELD Na score is 25.

Revision summary. The LLM draft captures the correct formulas and clamping rules for the majority of patients, but omits two conditional branches—in both the textual procedure and the Python tool. (1) It omits the dialysis-dependent creatinine override $\mathrm { C r } \to 3 . 0$ when on dialysis $2 2 \times / \mathrm { w e e k } )$ , a branch that appears in only 2 of the 20 input instances. (2) It always applies the sodium adjustment, missing the $\mathrm { M E L D ( i ) } \leq 1 1$ $\leq 1 1$ threshold that bypasses it entirely—a condition that none of the 20 instances happen to trigger. The expert added both branches by consulting the MDCalc reference page and restructured the procedure into 6 explicit steps.

Example 2: Creatinine Clearance (Cockcroft-Gault Equation). The Cockcroft-Gault equation appears to be a single formula, but the BMI-based weight adjustment makes it procedurally complex.

# Source Material

Calculator: Creatinine Clearance (Cockcroft-Gault Equation)

Ground Truth Explanation (1 of 20 instances, abbreviated): The formula for computing Cockcroft-Gault is given by $\mathrm { C r C l } = \left( \left( 1 4 0 - \mathrm { a g e } \right) \times \mathrm { w e i g h t } \times \right.$ $\times$ gender_coefficient) / (serum creatinine $\times 7 2$ ), where the gender_coefficient is 1 if male, and 0.85 if female. The serum creatinine concentration is in mg/dL. The patient’s gender is female, which means that the gender coefficient is 0.85. The patient is 62 years old. The concentration of creatinine is 85.0 umol/L. We need to convert the concentration to mg/dL. Let’s first convert the mass of creatinine from umol to mg. [. . . 6-step molar mass conversion. . . ] This will result to $9 . 6 2 \mathrm { m g }$ creatinine $/ 1 0 . 0 \mathrm { d L } = 0 . 9 6 2 \mathrm { m g }$ creatinine/dL. [. . . ] Because the patient is overweight/obese, we use the adjusted body weight formula. For females, the ideal body weight (IBW) is calculated as follows: $4 5 . 5 \mathrm { k g } + 2 . 3 \mathrm { k g } \times \mathrm { ~ \ r ~ { ~ \AA ~ } ~ }$ (height (in inches) − 60). [. . . ] $\mathbf { A B W } = \mathbf { I B W } + 0 . 4 \times ( \mathbf { w e i g h t } - \mathbf { I B W } ) = 5 2 . 3 8 + 0 . 4$ $\mathbf { \nabla } \times ( 7 2 . 0 - 5 2 . 3 8 ) = 6 0 . 2 3 \mathrm { k g } .$ . [. . . ] Plugging the patient’s values gives us $\left( \left( 1 4 0 - 6 2 \right) \times 6 0 . 2 3 \times 0 . 8 5 \right) /$ $( 0 . 9 6 2 \times 7 2 ) = 5 7 . 6 5 ~ \mathrm { m L / m i n }$ .

# LLM Drafted Skill

name: Cockcroft-Gault Creatinine Clearance

description: Estimate creatinine clearance using the Cockcroft-Gault equation for drug dosing in renal impairment.

# Creatinine Clearance (Cockcroft-Gault)

The Cockcroft-Gault equation estimates creatinine clearance (CrCl) to guide drug dosing in renal impairment.

Required Inputs: Age (years), sex, height, weight, serum creatinine (mg/dL; from $\mu \mathrm { m o l } / \mathrm { L } \colon \div 8 8 . 4 2 )$ .

Step 1 — Weight adjustment: Compute IBW: Male $= 5 0 + 2 . 3 \times ( h _ { \mathrm { i n } } - 6 0 )$ ; Female $= 4 5 . 5 + 2 . 3 \times$ $\left( h _ { \mathrm { i n } } - 6 0 \right)$ . For overweight/obese patients, use $\mathrm { 4 B W = I B W + 0 . 4 \times ( a c t u a l - I B W ) }$ .

▷ Only overweight branch; missing underweight and normal-weight rules

Step 2 — Apply Cockcroft-Gault: $\begin{array} { r } { \mathbf { C r C l } = \frac { ( 1 4 0 - \mathrm { a g e } ) \times w _ { \mathrm { a d j } } \times g } { \mathbf { C r } \times 7 2 } } \end{array}$ where $g = 1 . 0$ (male) or 0.85 (female).

Tool:

def compute_cg_crcl(age, weight_kg, height_cm, creatinine, is_female):

h_in $=$ height_cm * 0.393701

ibw $=$ (45.5 if is_female else 50) + 2.3 * (h_in - 60)

adj_ $\mathbf { \nabla } _ { \mathbf { w } } = \mathbf { \nabla }$ ibw + 0.4 * (weight_kg - ibw) # always uses ABW

$\tt { g } = 0 . 8 5$ if is_female else 1.0

return ((140 - age) * adj_w * g) / (creatinine * 72)

▷ No BMI check; always applies overweight formula

Worked Example: Patient: 62-year-old female, $1 6 5 ~ \mathrm { c m }$ , $7 2 . 0 \mathrm { k g }$ , $\mathrm { C r 0 . 9 6 2 m g / d L }$ (from 85.0 µmol/L).

[. . . ] $\mathrm { C r C l } = 5 7 . 6 5 \ \mathrm { m L / m i n }$

▷ Values from an input instance

# Golden Skill

name: Creatinine Clearance (Cockcroft-Gault Equation)

description: Compute creatinine clearance using the Cockcroft-Gault equation with BMI-based weight adjustment.

# Creatinine Clearance (Cockcroft-Gault Equation)

The Cockcroft-Gault equation estimates creatinine clearance (CrCl) as a surrogate for GFR, widely used for drug dosing adjustments in renal impairment. It uses an adjusted body weight determined from the patient’s BMI category.

Required Inputs: Age (years), sex, height (cm; from inches: $\times 2 . 5 4$ ), weight (kg; from lbs: $\times \ : 0 . 4 5 3 6$ serum creatinine (mg/dL; from $\mu \mathrm { m o l } / \mathrm { L } \colon \div 8 8 . 4 2 )$ .

Step 1 — Calculate BMI: BMI = weight_kg / (height_cm/100)2

# Step 2 — Determine adjusted body weight:

• Underweight $\mathbf { B M I } < 1 8 . 5 $ : use actual body weight.   
• Normal (18.5–24.9): use min(IBW, actual weight).   
• Overweight/obese $( \geq 2 5 )$ ): use $\mathrm { A B W } = \mathrm { I B W } + 0 . 4 \times ( \mathrm { a c t u a l - I B W } ) .$

IBW: Mal $: = 5 0 + 2 . 3 \times ( h _ { \mathrm { i n } } - 6 0 )$ ; Female $= 4 5 . 5 + 2 . 3 \times ( h _ { \mathrm { i n } } - 6 0 )$

# Step 3 — Apply Cockcroft-Gault:

$$
\mathrm {C r C l} = \frac {(1 4 0 - \mathrm {a g e}) \times w _ {\mathrm {a d j}} \times g}{\mathrm {C r} \times 7 2} (\mathrm {m L / m i n})
$$

where $g = 1 . 0$ (male) or 0.85 (female).

Tool:

def compute_cg_crlage,weight_kg,height_cm,creatinine,is_female): $\mathrm{bmi} =$ weight_kg / (height_cm / 100) \*\*2   
h_in $=$ height_cm $\ast 0.393701$ ibw $=$ (45.5 if is_female else 50) $+2.3\ast$ (h_in - 60) $\mathbf{g} = 0.85$ if is_female else 1.0   
if bmi < 18.5: adj_w $=$ weight_kg   
elif bmi < 25: adj_w $=$ min(ibw,weight_kg)   
else: adj_w $=$ ibw $+0.4\ast$ (weight_kg - ibw)   
return ((140 - age) $\ast$ adj_w $\ast \mathbf{g}$ )/ (creatinine $\ast$ 72)

Worked Example: Patient: 65-year-old female, 160 cm, 55 kg, Cr 1.0 mg/dL.

Step 1: $\mathrm { B M I } = \bar { 5 } 5 / 1 . 6 0 ^ { 2 } = 2 1 . \bar { 5 } $ normal weight.

Step 2: $\mathrm { I B W } = 5 2 . 3 8 < 5 5$ , so $w _ { \mathrm { a d j } } = \operatorname* { m i n } ( 5 2 . 3 8 , 5 5 ) = 5 2 . 3 8 \mathrm { k g } .$

Step 3: $\mathrm { C r C l } = \left( 1 4 0 - 6 5 \right) \times 5 2 . 3 8 \times 0 . 8 5 / \left( 1 . 0 \times 7 2 \right) = 4 6 . 3 8 \mathrm { m L / m i n } .$

Revision summary. (1) The LLM draft only implements the overweight/obese ABW adjustment— the text omits underweight and normal-weight rules, and the Python tool always computes ABW regardless of BMI. The expert added the full three-way branching (underweight actual weight; normal min(IBW, actual); overweight $ \mathrm { A B W }$ ) by consulting the MDCalc reference. (2) The draft’s worked example directly reuses values from an input instance. The expert replaced it with a newly constructed patient (65-year-old female, normal BMI) that exercises the min(IBW, actual) branch.

# A.5 CHAMP

Each instance in CHAMP contains a competition mathematics question, a ground truth answer (which may be a number, algebraic expression, yes/no, or brief phrase), and official solution steps detailing the intended approach. The dataset provides 89 unique concept annotations across eight mathematical domains: number theory (29 concepts), inequality (18), polynomial (18), algebra (9), combinatorics (8), sequence (3), trigonometry (3), and function (1). Each concept is accompanied by a concept description—a formal statement of the mathematical theorem or identity (e.g., “For non-negative $x , y \colon ( x { + } y ) / 2 \geq { \sqrt { x y } }$ , with equality iff $x { = } y '$ )—which serves as annotation-level context.

CHAMP uses multi-label annotations: each instance is tagged with all concepts relevant to its solution (average 1.7 per instance), and $49 \%$ of instances require two or more concepts. Each concept corresponds to one gold skill.

# A.5.1 LLM Draft Generation

For each concept, the LLM receives the concept name, its mathematical definition, category, all associated questions with answers and official solution steps, and relevant AoPS Wiki4 pages as input. The solution steps provide concrete worked examples of how the concept is applied, giving the LLM significantly more procedural context than in TheoremQA or LogicBench where no solution steps are available. The following prompt template is used:

# Prompt Template for CHAMP

You are a mathematics education expert. Given a mathematical concept and competition problems that apply it, write a reusable skill document that teaches how to apply this concept to solve new problems.

```txt
Concept: {concept_name}
Category: {category}
Definition: {concept_defined}
Reference material:
{reference_pages}
Problems applying this concept (each with its answer and official solution steps):
{problems_with_solutions}
Based on the above materials, write a skill document with the following structure: 
```

1. name: Canonical concept name.   
2. description: One sentence summarizing what the concept computes or states.   
3. Content (Markdown):

• Statement of the concept (theorem or formula).   
• When to recognize that this concept applies.   
• Step-by-step application procedure.   
• Common errors or pitfalls.   
• One worked example using a new problem (not from the problems above).

# A.5.2 Expert Revision

In addition to the general revision criteria, the following dataset-specific activities apply:

• Recognition pattern development: Competition math problems rarely name the concept they require. The concept definition tells what a theorem states, and solution steps demonstrate specific applications, but neither systematically identifies the structural cues that signal applicability across diverse problem types. LLM drafts restate the theorem and walk through example calculations

without abstracting recognition patterns. The expert adds problem-type classification and structural signatures that indicate when the concept should be applied.

• Solution strategy systematization: Official solution steps show the concept applied in ad-hoc fashion to individual problems, but competition math concepts often encompass multiple distinct application strategies (e.g., modular arithmetic for impossibility proofs vs. residue enumeration vs. periodicity analysis). LLM drafts present the concept as a single tool with one usage mode. The expert organizes the skill around a typology of problem types, each with its own systematic procedure.

# A.5.3 Construction Examples

We present two complete construction examples from different mathematical domains.

Example 1: AM-GM Inequality (Two Variables). The AM-GM inequality is a fundamental bounding tool; the key challenge lies in recognizing when an optimization problem can be solved by identifying the right pair of non-negative terms.

# Source Material

Concept: AM-GM Inequality (Two Variables)

Category: Inequality

Definition: For non-negative $x , y$ : $( x + y ) / 2 \geq \sqrt { x y }$ , with equality iff $x = y$

2 of 6 associated instances:

Q: If $\begin{array} { r l r } { a _ { i } } & { { } > } & { 0 } \end{array}$ for $\begin{array} { r c l } { i } & { = } & { 1 , 2 , \dots , n } \end{array}$ and $\begin{array} { r l r } { a _ { 1 } a _ { 2 } \cdot \cdot \cdot a _ { n } } & { { } = } & { 1 } \end{array}$ , what is the smallest value of $( 1 + a _ { 1 } ) ( 1 + a _ { 2 } ) \cdot \cdot \cdot ( 1 + a _ { n } )$ as an expression of $n$ ?

A: $2 ^ { n }$

Q: For positive numbers $a , b , c$ , what is the smallest value of $( a + b ) ( b + c ) ( a + c ) / ( a b c ) ?$

A: 8

# LLM Drafted Skill

name: AM-GM Inequality (Two Variables)

description: Establish bounds on sums and products of non-negative numbers using the AM-GM inequality.

AM-GM Inequality (Two Variables)

Statement: For non-negative $x , y$ : $( x + y ) / 2 \geq { \sqrt { x y } }$ , equivalently $x + y \geq 2 { \sqrt { x y } }$ . Equality holds iff $x = y$ .

Procedure:

1. Identify two non-negative terms $u , v$   
2. Apply AM-GM: $u + v \geq 2 \sqrt { u v }$ .   
3. Verify that the equality condition $u = v$ is achievable.

▷ No recognition patterns—when should one try AM-GM?

Worked Example: For positive a $\therefore b , c \colon ( a + b ) ( b + c ) ( a + c ) / ( a b c )$ . Apply AM-GM to each factor: $a + b \geq$

$2 { \sqrt { a b } }$ , etc. Product $\geq 8$ .

▷ Same problem as an input instance

# Golden Skill

name: AM-GM Inequality (Two Variables)

description: Apply AM-GM for two variables to establish bounds on sums, products, and square root comparisons.

# AM-GM Inequality (Two Variables)

Statement: For non-negative $x , y$ : $( x + y ) / 2 \geq \sqrt { x y }$ . Equality holds iff $x = y$

Recognition: Use AM-GM when:

• A sum $x { + y }$ needs a lower bound in terms of $\sqrt { x y }$ , or a product needs an upper bound.   
• The expression contains $t + 1 / t$ for $t > 0$ (directly gives $t + 1 / t \geq 2 $ ).   
• A product of factors $( a + b ) ( b + c ) \cdot \cdot \cdot$ appears—apply AM-GM to each factor, then multiply.

# Procedure:

1. Identify terms: Find two non-negative terms $u , v$ whose product or sum appears in the expression.   
2. Apply AM-GM: $u + v \geq 2 \sqrt { u v }$ .   
3. Chain or multiply: To bound a product of factors, apply AM-GM to each factor and multiply the results.   
4. Verify equality: Confirm $u = v$ is achievable under the problem’s constraints.

# Common Pitfalls:

• Non-negativity: both terms must be non-negative; AM-GM does not apply otherwise.   
• Chained equality: when AM-GM is applied multiple times, all equality conditions must hold simultaneously—check that they are mutually compatible.

Worked Example: For √ $t > 0$ , find the minimum of $t + 4 / t$ . Apply AM-GM with $u = t$ , $v = 4 / t$ $t + 4 / t \geq 2 { \sqrt { 4 } } = 4$ . Equality when $t = 4 / t$ , i.e., $t = 2$ . Minimum is 4.

Revision summary. (1) The LLM draft correctly states the AM-GM formula and basic procedure but does not teach how to recognize when a problem calls for AM-GM. Problems like “minimize $( 1 + a _ { 1 } ) \cdot \cdot \cdot ( 1 { + } a _ { n } )$ given $\prod a _ { i } \bar { = } 1 ^ { \prime \prime }$ require recognizing the per-factor AM-GM pattern, which is not obvious from the formula alone. The expert added a “Recognition” section with structural cues (product-sum patterns, reciprocal expressions, products of factors). (2) The draft’s worked example uses the same problem as an input instance. The expert replaced it with a newly constructed $t + 4 / t$ example that exercises the reciprocal-expression pattern.

Example 2: Modular Arithmetic. Modular arithmetic is the most frequently applied concept in CHAMP (32 instances); the main challenge is selecting the right modulus and application strategy from among several distinct problem types.

# Source Material

Concept: Modular Arithmetic

Category: Number Theory

Definition: $( a { + } b )$ mod $m = ( ( a { \bmod { m } } ) + ( b { \bmod { m } } ) )$ mod $m$ . $( a \cdot b )$ mod $m = ( ( a { \bmod { m } } )$

(b mod m)) mod $m$ . $a ^ { k }$ mod $m = ( ( a { \bmod { m } } ) ^ { k } )$ ) mod $m$ .

2 of 32 associated instances:

Q: Are there integer solutions to the equation $( x ^ { 2 } - 1 ) ( y ^ { 2 } - 1 ) + 1 9 8 5 = z ^ { 2 } ?$

Q: Let $( x , y , z )$ be an integer-valued solution to $x ^ { 2 } + y ^ { 2 } = z ^ { 2 }$ . At least how many of the three numbers must be divisible by 5?

A: At least 1 number

# LLM Drafted Skill

name: Modular Arithmetic

description: Use modular arithmetic rules to analyze divisibility and residue patterns.

# Modular Arithmetic

Core Rules: $( a { + } b )$ mod $m = ( ( a { \bmod { m } } ) + ( b { \bmod { m } } ) )$ ) mod $m$ . Similar for multiplication and exponentiation.

# Procedure:

1. Choose a modulus $m$ .   
2. Compute residues of each side of the equation.   
3. Check if the residues are compatible.

▷ No guidance on modulus selection or problem-type classification

Worked Example: Show $x ^ { 2 } + y ^ { 2 } = z ^ { 2 }$ requires at least one variable divisible by 5: check all residues mod 5.

Squares mod 5 are $\{ 0 , 1 , 4 \}$ . If none is divisible by 5 . . . ▷ Same problem as an input instance

[. . . ]

# Golden Skill

name: Modular Arithmetic: Residue Analysis

description: Apply modular arithmetic to prove impossibility, enumerate residues, analyze periodicities, and use CRT for simultaneous divisibility.

# Modular Arithmetic: Residue Analysis

# Problem Types and Recognition:

• Type 1—Impossibility: “Find all integer solutions” choose $m$ so residue sets of each side are disjoint.   
• Type 2—Residue enumeration: “What are the possible values mod m?” → compute $f ( x )$ mod $m$ for each residue class.   
• Type 3—Divisibility: “Show $d \mid f ( x )$ for all $x ' $ verify $f ( x )$ mod $d = 0$ across all residue classes.   
• Type 4—Periodicity: “Find $f ( n )$ mod $m$ for large $n { \stackrel {  } { } } \to$ compute until a cycle appears.

# Modulus Selection:

• $m = 3 , 9$ : distinguish squares (0, 1 mod 3) and cubes.   
• $m = 4 , 8$ : distinguish squares $( 0 , 1 , 4 { \bmod { 8 } }$ ), powers of 2.   
• Start with the modulus suggested by the coefficients; try others if the first fails.

# Procedure:

1. Classify the problem type (impossibility, enumeration, divisibility, or periodicity).   
2. Select modulus based on problem structure and coefficient cues.   
3. Enumerate relevant residues (for all integers, or restricted to primes, etc.).   
4. Compare residue sets or compute the period as needed.

Worked Example: Prove that no integers satisfy $x ^ { 2 } + y ^ { 2 } + z ^ { 2 } = 8 k + 7 .$ . Type 1 (impossibility). Choose $m = 8$ : $x ^ { 2 }$ mod $8 \in \{ 0 , 1 , 4 \}$ . Possible sums of three elements from $\{ 0 , 1 , 4 \}$ : $\{ 0 , 1 , 2 , 3 , 4 , 5 , 6 \}$ —7 is not achievable. Since $8 k + 7 \equiv 7$ (mod 8) is unreachable, no solutions exist.

Revision summary. (1) The LLM draft states the modular arithmetic identities and a generic threestep procedure but does not distinguish among the different problem types or provide guidance on modulus selection—the core strategic decisions. The expert organized the skill around a four-type classification (impossibility, enumeration, divisibility, periodicity) with type-specific procedures and added concrete modulus selection heuristics by consulting the AoPS Wiki. (2) The draft’s worked example uses the same Pythagorean-triple divisibility problem from an input instance. The expert replaced it with a newly constructed impossibility example $( x ^ { 2 } + y ^ { 2 } + z ^ { 2 } \neq \bar { 8 } k + 7 )$ that demonstrates the full Type-1 workflow.

# A.6 BigCodeBench

Each instance in BigCodeBench specifies a Python function to implement. The task description defines the expected behavior, inputs and outputs, and, when applicable, error-handling requirements, together with a function signature and a set of relevant library/import requirements. The dataset also provides a canonical solution and a unit-test suite for execution-based evaluation, where model performance is commonly measured by metrics such as pass@1. BigCodeBench covers 139 Python libraries.

BigCodeBench provides multi-label library annotations: each instance is associated with the libraries used or permitted in the reference implementation, with an average of 2.8 libraries per task. In our conversion, each distinct library is converted into one gold skill, and an instance may therefore be associated with multiple gold skills. The dataset does not provide annotation-level context such as library descriptions, API summaries, or reusable usage guidance.

# A.6.1 LLM Draft Generation

For each library, the LLM receives the library name, the associated task descriptions and canonical solutions, and relevant official documentation5 pages as input. The canonical solutions are used to identify concrete API usage patterns: which functions are invoked, how parameters are configured, what objects are returned, and how different calls are composed to satisfy task requirements. This information is often critical for generating code that passes unit tests, but is not always explicitly stated in the natural-language task descriptions. To avoid instance-level leakage, the final gold skills are revised to describe reusable API contracts and task-oriented recipes, rather than to copy benchmark-specific solutions. The following prompt template is used:

# Prompt Template for BigCodeBench

```txt
You are a Python programming expert. Given a library and a set of coding tasks that use it, write a reusable skill document that teaches how to use this library's API correctly in similar tasks.  
Library: {library_name}  
Official documentation:  
{doc_pages}  
Tasks using this library (each with its task description and canonical solution):  
{tasks_and_solutions}  
Based on the above materials, write a skill document with the following structure: 
```

1. name: Library name with a descriptive subtitle.   
2. description: One sentence summarizing the practical API patterns covered.   
3. Content (Markdown):

• Core API functions with calling conventions and return types.   
• Common usage patterns with code examples.   
• Common errors or pitfalls.

# A.6.2 Expert Revision

For BigCodeBench, expert revision focuses on making each skill directly useful for code-generation tasks that are evaluated by automated unit tests. Since passing these tests often depends on exact API semantics, merely describing a library at a conceptual level is insufficient. For example, an LLMgenerated draft may state that re supports pattern matching and substitution, but omit the behavioral details that determine whether the generated code is correct, such as return types, parameter-specific effects, and edge-case behaviors. Experts thus revise such drafts into precise API contracts, including concrete notes on return values, parameter interactions, and representative examples; for instance,

re.findall returns captured group contents when the pattern contains a single capturing group, and plt.hist() returns a tuple rather than an Axes object.

In addition, experts reorganize the skills from function-centric descriptions into task-oriented recipes. A single library may support many distinct programming needs, such as URL removal, word tokenization, file matching, data validation, or visualization. Listing APIs one by one does not clearly indicate how to combine these functions for a particular task. The revised skills are therefore structured around reusable code patterns indexed by their intended use, enabling a model to identify the relevant recipe and apply it to the current programming problem.

# A.6.3 Construction Examples

We present two complete construction examples from different library domains.

Example 1: re (Regular Expressions). The re library is used across 139 instances for text processing tasks; the key challenge lies in the subtle behavioral differences between API functions— for example, re.findall with a capturing group returns group contents rather than full matches, and chained re.sub operations interact when later substitutions depend on earlier ones.

# Source Material

Library: re — Regular Expressions 2 of 139 associated instances:

Task: Find the latest log file in a directory matching a given regex pattern; return its path or None.

Canonical solution:

```python
log_files = [f for f in os.listdir(log_dir) if re.match(pattern, f)]  
log_files = sorted(log_files, key=lambda f: os.path.gettime(os.path.join(log_dir, f)), reverse=True)  
return os.path.join(log_dir, log_files[0]) \ if log_files else None 
```

Task: Create a word cloud from text after removing URLs; raise ValueError if no words remain.

Canonical solution:

```python
text = re.sub(r"http[s]?://\S+", "", text)  
if not text.strip():  
    raise ValueError("No words available...")  
wordcloud = WordCloud().generate(text)  
plt.imshow(wordcloud)  
return wordcloud 
```

# LLM Drafted Skill

name: re — Regular Expressions description: Python regular expression library for pattern matching, searching, and text substitution.

# Core Functions:

• re.match(pattern, string): Match pattern at the start of string.   
• re.search(pattern, string): Search for pattern anywhere in string.   
• re.findall(pattern, string): Return all non-overlapping matches as a list of strings.   
• re.sub(pattern, repl, string): Replace all occurrences of pattern.

▷ No behavioral contracts (e.g., findall group semantics, match anchoring)

Common Patterns:

Remove URLs: re.sub(r"http[s]?://\S+", "", text) Find all words: re.findall(r"\w+", text)

Worked Example: Remove URLs from text and generate a word cloud: text $=$ re.sub(r"http[s]?://\S+", "", text) ▷ Same task as an input instance [. . . ]

# Golden Skill

name: re — Regular Expressions Practical Reference description: Practical guide to re API semantics, findall/sub/match anchoring, case handling, operation ordering, and common text-processing recipes.

# Match Anchoring:

```python
re.match(r'\w+\', text) # start of string only  
re.search(r'\d+\', text) # scans full string  
re.fullmatch(r'\w+\', text) # entire string must match  
re.findall and Capturing Groups:  
# Without group: returns full matches  
re.findall(r'score: \d+\', text)  
# -> ['score: 80', 'score: 95']  
# With one group: returns group contents only  
re.findall(r'score: (\d+)\', text)  
# -> ['80', '95']  
re.sub - Operation Ordering:  
When steps interact, order matters: remove punctuation before introducing underscores, so that string.punctuation (which includes _) does not remove the new underscores.  
Word Tokenization Recipes:  
# Alphabetic words only (drops digits):  
re.findall(r'[A-Za-z]+', text)  
# Word-character tokens (keeps digits):  
re.findall(r'\b\w+\\b', text)  
[...] 
```

Example 2: matplotlib (Data Visualization). The matplotlib library appears in 309 instances—the most frequently annotated library—for data visualization tasks; the key challenge lies in returning the correct object type, as different plotting functions produce different return types (Axes, tuples, or ndarrays) that unit tests verify.

# Source Material

```txt
Library: matplotlib - Data Visualization  
2 of 309 associated instances:  
Task: Create a bar chart from a list of category-value pairs; return a (DataFrame, Axes) tuple.  
Canonical solution:  
df = pd.DataFrame(list_of_pairs, columns=['Category', "Value'])  
plt.figure(figsize=(10, 5))  
sns.barplot(x="Category", y="Value", data=df)  
plt.title("Category vs Value")  
ax = plt.gca()  
return df, ax  
Task: Standardize a data matrix, compute row means, and visualize their distribution as a histogram; return a (DataFrame, Axes) tuple.  
Canonical solution:  
scatter = StandardScaler()  
standardized_data = scorer.fit_transform(data_matrix)  
df = pd.DataFrame(standardized_data, columns=FEATURE_NAMES)  
df["Mean"] = df.mean(axis=1)  
plt.figure(figsize=(10, 5))  
ax = df["Mean"].plot(kind="hist", title="Distribution of Means")  
return df, ax 
```

# LLM Drafted Skill

name: matplotlib — Data Visualization description: Python library for creating plots, charts, and visualizations.

# Basic Plotting Functions:

• plt.plot(x, y): Create a line plot.   
• plt.bar(x, height): Create a bar chart.   
• plt.hist(data, bins): Create a histogram.   
• plt.scatter(x, y): Create a scatter plot.

Customization: plt.title("Title"), plt.xlabel("X"), plt.ylabel("Y"), plt.figure(figsize=(w, h)). ▷ No return type specifications—the most common source of test failures

Worked Example: Create a bar chart from a DataFrame:

```txt
plt.figure(figsize=(10,5))  
sns.barplot(x="Category",y="Value", data=df)  
ax = plt.gca()  
return df, ax  
[...] 
```

▷ Same task as an input instance

# Golden Skill

name: matplotlib — Plotting API, Return Types, and Composition Recipes description: Guide for correct matplotlib API usage: return type contracts, OOP vs state-machine style, histogram rwidth, bar chart positioning, and common pitfalls.

# Return Type Quick Reference:

```python
Expression Actual return type  
fig, ax = plt.subplot() → ax  
plt.bar_labels, vals) BarContainer  
plt.hist(data) (n, bins, patches) tuple  
df.plot() / series.plot() Axes  
df.boxplot() Axes  
ax pie (...) (wedges, texts) tuple  
OOP Style (use when returning Axes):  
fig, ax = plt.subplot()  
ax.plot(x, y, label="series")  
ax.set_title("Title")  
return ax  
Bar Charts with Numeric Positions:  
indexes = np.arange(len.labels))  
ax.bar(indexes, values)  
ax.set_xsticks(indexes) # positions first  
ax.set_xticklabels Labels) # then labels  
Omitting set_xsticks before set_xticklabels raises a ValueEr [...] 
```

Revision summary. (1) The LLM draft describes matplotlib’s plotting functions at a conceptual level but omits return type specifications—the single most common source of test failures in Big-CodeBench. Functions like plt.hist() return a (n, bins, patches) tuple rather than an Axes object, df.boxplot() does not return Axes, and ax.pie() returns a tuple of wedges and texts. Models that follow the draft’s function-level descriptions produce code that returns incorrect types and fails unit tests. The expert added a return type reference table and restructured the skill around

the OOP vs. state-machine distinction, with correct return patterns for each plotting idiom. (2) The draft’s worked example reuses a bar chart task from an input instance. The expert replaced it with newly constructed code examples that exercise different return type patterns.

# B Implementation Details

This appendix provides the prompt templates used to implement the skill-use strategies introduced in $\ S 4 . 2$ . These templates specify how retrieved skills are exposed to the model during inference. Unless otherwise stated, {original user prompt} denotes the original task prompt from the corresponding benchmark.

# B.1 Full-Skill Injection

When one or more skills are provided, either by retrieval or by Oracle annotation, their full content is prepended to the user prompt:

# Full-Skill Injection

```txt
Relevant Skill:  
{skill content}  
{original user prompt} 
```

When multiple skills are available, e.g., in Oracle mode for multi-label datasets such as CHAMP [17] and BigCodeBench [61], they are concatenated with a horizontal rule delimiter (---).

# B.2 LLM Selection

For LLM Selection, the model is first given the task query alongside a numbered list of candidate skills, where each candidate is represented by its name and description. The model is asked to output only the number of the most relevant skill:

# LLM Selection prompt

```txt
Given the following problem, select the ONE most relevant skill. Output ONLY the skill number.   
Problem:   
{query}   
Skills:   
[1] Bayes' Theorem: Posterior probability computation using prior and likelihood   
[2] Law of Total Probability: Marginal probability via exhaustive partitioning ...   
Most relevant skill number: 
```

The selected skill’s full content is then prepended to the task prompt using the same format as Full-Skill Injection in $\ S \mathrm { B } . 1$ .

# B.3 Progressive Disclosure

For Progressive Disclosure, the system prompt is augmented with a compact catalog of candidate skills and an instruction that allows the model to load full skill content on demand:

# Progressive Disclosure prompt (appended to system prompt)

You have access to a skill library. Each skill provides precise methodology and step-by-step procedures for a specific problem type –- these often contain critical details that general knowledge may miss.

To use a skill, write on its own line: LOAD_SKILL: <index> For example: LOAD_SKILL: 0 You will receive the skill’s full content and can then apply the methodology solve the problem.

Available skills: 0 –- Bayes’ Theorem –- Posterior probability computation using prior and likelihood

1 –- Law of Total Probability –- Marginal probability via exhaustive partitioning

When the model issues a LOAD_SKILL action, the framework injects the full content of the corresponding skill and prompts the model to continue. The model may load multiple skills across successive turns, up to 10 rounds.