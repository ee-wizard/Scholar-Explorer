# Learning Hierarchical Procedural Memory for LLM Agents through Bayesian Selection and Contrastive Refinement

Saman Forouzandeh, Wei Peng, Parham Moradi, Xinghuo Yu, Mahdi Jalili School of Engineering, Royal Melbourne Institute of Technology University Melbourne, VIC, Australia

# ABSTRACT

We present MACLA, a framework that decouples reasoning from learning by maintaining a frozen large language model (LLM) while performing all adaptation in an external hierarchical procedural memory. MACLA extracts reusable procedures from trajectories, tracks reliability via Bayesian posteriors, selects actions through expected-utility scoring, and refines procedures by contrasting successes vs. failures. Across four benchmarks (ALFWorld, WebShop, TravelPlanner, InterCodeSQL), MACLA achieves $7 8 . 1 \%$ average performance, outperforming all baselines. On ALFWorld unseen tasks, MACLA reaches $9 0 . 3 \%$ with $+ 3 . 1 \%$ positive generalization. The system constructs memory in 56 seconds $^ { 2 , 8 0 0 \times }$ faster than the state-of-the-art LLM parameter-training baseline), compresses 2,851 trajectories into 187 procedures (15:1). Experimental results demonstrate that structured external memory with Bayesian selection and constrastive refinement enable sample-efficient, interpretable and continually improving agents without LLM parameter updates. Code is publicly available at MACLA.

# KEYWORDS

Memory-augmented agents, Procedural memory, Bayesian decision making, Contrastive learning, LLM agents

# ACM Reference Format:

Saman Forouzandeh, Wei Peng, Parham Moradi, Xinghuo Yu, Mahdi Jalili . 2026. Learning Hierarchical Procedural Memory for LLM Agents through Bayesian Selection and Contrastive Refinement. In Proc. of the 25th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2026), Paphos, Cyprus, May 25 – 29, 2026, IFAAMAS, 21 pages.

# 1 INTRODUCTION

Large language model (LLM) agents can solve complex, interactive tasks such as web shopping [25] and embodied AI housekeeping [9], by transforming natural-language instructions into sequences of environment actions [26]. In these settings, agents navigate step-bystep through partially observable environments to pursue subgoals and ultimately complete the task [9, 22]. The resulting trajectory is the ordered record of an episode’s interaction, typically written as $( T , A , O , R )$ , where $T$ represents a task to complete, ?? are actions, $O$ stand for observations for the outcome of corresponding actions, and ?? records step-level outcomes or rewards. Trajectories thus capture the full decision process, not merely terminal success or failure, and provide dense supervision for how an agent progresses through a task [16, 22]. When a new task arrives, the agent synthesizes an appropriate trajectory (that is, a step-by-step plan and

its execution) to achieve the goal in the current context, deciding which information to gather, which tools to invoke, and which subroutines to chain in order to achieve completion [25, 26].

Early LLM agents used prompt-based planning [26] and selfcritique [15], but lack persistent “how-to” procedures — when tasks are similar but not identical, agents must re-plan from scratch, increasing cost and latency. Fine-tuning approaches [2, 27, 29] adapt agents via supervised learning or RLHF, but typically treat entire trajectories as single units weighted by terminal success/failure, neglecting rich intermediate steps. In practice, failed trajectories often contain correct substeps (e.g., “successfully navigating and retrieving an egg, but failing to boil it” [16]), while successful ones may include suboptimal actions that accidentally cancel out. Recent work [22] addresses this via step-level rewards, but requires repeated policy training on densely-labeled data, incurring substantial computational cost.

We introduce MACLA (Memory-Augmented Contrastive Learning Agent), a framework that disentangles reasoning from learning through the coupling a frozen LLM and a structured external procedural memory (Figure 1). Unlike fine-tuning approaches where reasoning and adaptation are entangled within billions of parameters, MACLA fixes the LLM as a stable semantic reasoner responsible for trajectory segmentation, abstraction, and action generation. All learning occurs externally through explicit, interpretable memory operations - maintaining human-readable procedures, updating Bayesian posteriors, and refining preconditions through contrastive analysis. MACLA operates through three core mechanisms:

(1) Bayesian procedure selection: Maintains Beta posteriors $\mathrm { B e t a } ( \alpha _ { i } , \beta _ { i } )$ over procedure success rates and ranks candidates via expected-utility scoring that balances contextual relevance, success probability, failure risk, and information gain, providing principled exploration-exploitation.   
(2) Contrastive refinement: Compares successful and failed execution contexts to tighten preconditions, repair action sequences, and refine postconditions once procedures accumulate sufficient evidence (i.e., $\geq$ a threshold), progressively improving procedure quality through memory edits rather than gradient updates.   
(3) Meta-procedural learning: Composes frequently co-occurring procedures into hierarchical “playbooks” with conditional control policies (continue, skip, repeat, abort) for long-horizon tasks, enabling strategic reuse beyond atomic skills.

This architecture yields sample-efficient, interpretable agents with human-readable procedural knowledge, closed-form utility computation, and minimal LLM usage. Specifically, this work contributes:

• Online procedural memory adaptation: Continual updates to procedural and meta-procedural memory during and

![](images/37414e3e80d0ce03db47d8a4b5816438e995bc0e7d5aaf0c6ebce3184a564315.jpg)

![](images/181421eb1fa5a25b95acac658753d529fce0e011459fe1879b1ab2fa7b833ae1.jpg)  
MACLA: Memory-Augmented Contrastive Learning Agent (Inference Time Learning)   
Figure 1: Comparison between existing LLM-based trajectory learning (top) and the proposed memory-augmented contrastive learning agent (MACLA, bottom). Existing methods train trajectories $( T , A , O , R )$ (Task, Action, Observation, Reward) into LLM parameters through post-training (finetuning and/or RLHF), whereas MACLA constructs procedural and meta-procedural memory externally through frozen LLM abstraction, segmentation, Bayesian selection, and contrastive refinement. Memories are learned during memory construction. Besides learning during memory construction, MACLA enables inference-time learning in which outputs are verified in the task environment, with feedback used for contrastive refinement on the retrieved memories. Meta-procedural learning enables the composition policy to be learned among procedures.

after episodes, enabling adaptation without weight updates, compared with offline LLM post-training approaches [17, 22, 29] that remain static at inference.

• Reasoning/learning decoupling: A frozen LLM for parsing and abstraction with all improvements occurring in an external, structured procedural memory, avoiding the computational cost and catastrophic forgetting risks of parameter fine-tuning.   
• Bayesian uncertainty-aware selection: A principled procedure selection module that maintains Beta posteriors over success rates with closed-form expected utility objectives balancing relevance, success probability, failure risk and information gain.   
• Contrastive procedural refinement: An algorithm leveraging paired successes and failures to tighten preconditions, repair action schemas, and refine postconditions of stored procedures without requiring expert demonstrations.   
• Hierarchical meta-procedural composition: Automatic discovery and maintenance of conditional playbooks with control policies (skip, repeat, abort) for long-horizon tasks, enabling compositional generalization.

We evaluate MACLA across four benchmarks (ALFWorld [9], WebShop [25], TravelPlanner [21], InterCodeSQL [24]), achieving $7 8 . 1 \%$ average performance — the highest among all methods,

including those using models $1 0 \times$ larger (later in Table 1). On ALF-World [16], MACLA reaches $8 7 . 2 \%$ on seen and $9 0 . 3 \%$ on unseen tasks, with a positive generalization gap $( + 3 . 1 \% )$ indicating compositional transfer rather than overfitting. The system achieves this with only 0.016 GPU-hours for one-time memory construction $- 2 { , } 8 0 0 \times$ faster than the state-of-the-art LLM parameter-training baseline [22], which requires 44.8 GPU-hours of iterative training — while simultaneously producing human-interpretable procedural knowledge.

# 2 RELATED WORKS

LLM agents have advanced rapidly in reasoning and decisionmaking, enabling multi-step interaction in embodied and webbased environments. Early frameworks such as ReAct[26] and Reflexion[14] integrate reasoning and acting within the same loop, while trajectory-tuning methods [2, 22] fine-tune models using expert demonstrations. However, fine-tuning is computationally expensive, requires offline data collection and training cycles, and does not support true online adaptation at inference time. To overcome this issue, a line of research augments LLM agents with memory for continuous reasoning. Memory is a foundational component of language agents, supporting competence across multiple timescales from transient working context to persistent long-term knowledge [6, 8, 31]. Research on memory for LLM agents can be usefully

organized along two directions: where memory resides and what is stored. Along the first direction, some methods such as MemGPT [11] and MemoryBank [32], use buffer-based systems to store conversational or episodic traces and retrieve them with embedding search and simple heuristics. Some others, such as HiAgent[5], A-Mem[23], MemAgent [28] use hierarchical designs to separate working buffers from episodic and long-term stores to relieve context pressure and improve persistence. Recently, SAGE [7] used reflective multi-agent controllers to curate these stores while controlling growth. The second direction concerns what is stored. Many systems retain free-form text snippets such as notes, summaries, or dialogue chunks; these are easy to write but suffer from retrieval drift and weak compositionality as repositories scale [11, 32]. More structured artifacts appear as tuples and key–value frames (e.g., tool logs or entity/event graphs), which aid filtering but still lack executable semantics for reuse. A growing line of work targets skills and procedures: agents capture reusable action patterns, tool workflows, and instruction-like steps across related tasks [3, 19, 20]. Memp [4] advances this view by treating procedural memory as a first-class object and studying its construction, retrieval, and update across domains. However, several key limitations remain; (1) it represents know-how largely as monolithic text (scripts or full trajectories) with heuristic retrieval and simple updates; (2) it lacks uncertainty-aware selection or principled exploration-exploitation balance, preventing reason about reliability or risk of retrieved memory; and (3) it lacks a mechanism to refine procedures from paired successes and failures or abstract recurring patterns into meta-procedural compositions. Comparatively, we represent experience as structured, hierarchical procedures with explicit preconditions, action schemas, and postconditions, enabling interpretable reuse and safe composition and direct schema edits when evidence warrants change. The proposed approach enables the system to continuously adapt and improve.

# 3 THE PREAMBLE

You will be assigned a submission number when you register the abstract of your paper on OpenReview. Include this number in your document using the ‘\acmSubmissionID’ command.

Then use the familiar commands to specify the title and authors of your paper in the preamble of the document. The title should be appropriately capitalised (meaning that every ‘important’ word in the title should start with a capital letter). For the final version of your paper, make sure to specify the affiliation and email address of each author using the appropriate commands. Specify an affiliation and email address separately for each author, even if two authors share the same affiliation. You can specify more than one affiliation for an author by using a separate ‘\affiliation’ command for each affiliation.

Provide a short abstract using the ‘abstract’ environment.

Finally, specify a small number of keywords characterising your work, using the ‘\keywords’ command.

# 4 PROPOSED METHOD

The key components of MACLA are described in detail below.

# 4.1 LLM-based Procedural Abstraction

The first stage transforms raw episodic trajectories into structured, reusable procedural knowledge. Given a trajectory

${ \tau } = \{ ( o _ { t } , \bar { a } _ { t } , r _ { t } ) \} _ { t = 0 } ^ { T }$ consisting of textual observations $o _ { t }$ , primitive actions $a _ { t }$ , and rewards $r _ { t }$ , the frozen LLM $\mathcal { L } _ { \theta }$ receives the full trajectory and identifies semantically coherent segments that correspond to meaningful sub-tasks:

$$
\operatorname {S e g} = \mathcal {L} _ {\theta} \left(\operatorname {P r o m p t} _ {\text {s e g m e n t}} (\tau)\right) = \left\{\left(t _ {k} ^ {\text {s t a r t}}, t _ {k} ^ {\text {e n d}}, d _ {k}\right) \right\} _ {k = 1} ^ {K}, \tag {1}
$$

where each segment $k$ spans time steps $[ t _ { k } ^ { \mathrm { s t a r t } } , t _ { k } ^ { \mathrm { e n d } } ]$ and is summarized by a description $d _ { k }$ . For each segment, MACLA constructs a structured procedure $\mathrm { P r o c } _ { k } \ : = \ : \left. \mathcal G _ { k } , \Psi _ { k } , \pi _ { k } , \Phi _ { k } \right.$ , where $\mathcal { G } _ { k }$ is a natural-language goal, $\Psi _ { k }$ are precondition patterns inferred from the observations before the segment, $\pi _ { k }$ is an abstracted action sequence, and $\Phi _ { k }$ are postcondition patterns extracted from the final observations. This decomposition produces interpretable “how-to” skills that can be invoked whenever their preconditions are met. To support retrieval and merging, each procedure is embedded into a semantic vector space using an encoder $\phi$ , $\mathbf { e } _ { k } = \phi ( [ G _ { k } ; \Psi _ { k } ; \Phi _ { k } ] ) \in$ $\mathbb { R } ^ { d }$ . When a new procedure is created, it is compared to existing ones via cosine similarity, $i ^ { * } = \arg \operatorname* { m a x } _ { i } \sin ( \mathbf { e } _ { k } , \mathbf { e } _ { i } )$ . If $\mathrm { s i m } ( \mathbf { e } _ { k } , \mathbf { e } _ { i ^ { * } } ) \ >$ $\theta _ { \mathrm { d u p } }$ , the new procedure is merged into the existing one by expanding its condition sets; otherwise, a new entry is added. This process yields a continually growing procedural library $\mathbb { M } _ { \mathrm { p r o c } } =$ $\left\{ ( \mathrm { P r o c } _ { i } , \mathbf { e } _ { i } , \alpha _ { i } , \beta _ { i } ) \right\} _ { i = 1 } ^ { N _ { p } }$ that forms the foundation for later Bayesian selection and refinement.

# 4.2 Bayesian Reliability and Utility Selection

Given the procedural library, the agent must decide which procedure to execute for the current observation. Each procedure $\mathrm { P r o c } _ { i }$ maintains a Beta posterior over its success probability $\rho _ { i } \in [ 0 , 1 ]$ :

$$
p \left(\rho_ {i} \mid \mathcal {D} _ {i}\right) = \operatorname {B e t a} \left(\rho_ {i}; \alpha_ {i}, \beta_ {i}\right) \tag {2}
$$

where $\alpha _ { i }$ and $\beta _ { i }$ accumulate successful and failed executions from history $\mathcal { D } _ { i }$ . The posterior mean $\mathbb { E } [ \rho _ { i } ] = \alpha _ { i } / ( \alpha _ { i } + \beta _ { i } )$ estimates current reliability, while the variance Var[???? ] = ???? ????(???? +???? )2 (???? +???? +1) $\begin{array} { r } { \mathrm { V a r } [ \rho _ { i } ] = \frac { \alpha _ { i } \beta _ { i } } { ( \alpha _ { i } + \beta _ { i } ) ^ { 2 } ( \alpha _ { i } + \beta _ { i } + 1 ) } } \end{array}$ quantifies epistemic uncertainty. For each candidate, we compute expected utility by integrating over the Beta posterior. Given utility ?? (?? | ???? , ??) = Rel?? (???? ) · ?? · ??max − Risk?? (???? ) · (1−??) ·??fail + ??info · ?? (??), the expected utility is:

$$
\mathrm {E U} \left(\operatorname {P r o c} _ {i} \mid o _ {t}\right) = \int_ {0} ^ {1} U (\rho \mid o _ {t}, i) \operatorname {B e t a} (\rho ; \alpha_ {i}, \beta_ {i}) d \rho \tag {3}
$$

Exploiting $\begin{array} { r } { \mathbb { E } _ { \mathrm { B e t a } ( \alpha , \beta ) } \left[ \rho \right] = \frac { \alpha } { \alpha + \beta } } \end{array}$ and $\begin{array} { r } { \mathbb { E } [ 1 - \rho ] = \frac { \beta } { \alpha + \beta } } \end{array}$ = ????+?? , this simplifies to:

$$
\begin{array}{l} \operatorname {E U} (\operatorname {P r o c} _ {i} \mid o _ {t}) = \underbrace {\operatorname {R e l} _ {i} (o _ {t}) \frac {\alpha_ {i}}{\alpha_ {i} + \beta_ {i}} R _ {\max }} _ {\text {e x p e c t e d r e w a r d}} - \underbrace {\operatorname {R i s k} _ {i} (o _ {t}) \frac {\beta_ {i}}{\alpha_ {i} + \beta_ {i}} C _ {\text {f a i l}}} _ {\text {f a i l u r e c o s t}} \\ + \underbrace {\lambda_ {\text {i n f o}} H \left[ \operatorname {B e t a} \left(\alpha_ {i} , \beta_ {i}\right) \right]} _ {\text {i n f o r m a t i o n g a i n}} \tag {4} \\ \end{array}
$$

where ${ \mathrm { R e l } } _ { i } ( o _ { t } ) = \cos ( \phi ( o _ { t } ) , \mathbf { e } _ { i } )$ is contextual similarity, $\mathrm { R i s k } _ { i } ( o _ { t } )$ is the fraction of past failures with similar contexts, and $H [ \cdot ]$ is differential entropy encouraging exploration. The selected procedure

is:

$$
\operatorname {P r o c} _ {t} ^ {*} = \arg \max  _ {\operatorname {P r o c} _ {i} \in C _ {t}} \mathrm {E U} \left(\operatorname {P r o c} _ {i} \mid o _ {t}\right) \tag {5}
$$

subject to confidence threshold $\theta _ { \mathrm { c o n f } } .$ . If $\begin{array} { r } { \operatorname* { m a x } _ { i } \operatorname { E U } ( \operatorname* { P r o c } _ { i } | o _ { t } ) < \theta _ { \mathrm { c o n f } } , } \end{array}$ the agent falls back to zero-shot LLM reasoning. This Bayesian selection mechanism balances exploitation (high ????+?? procedures), $\frac { \alpha } { \alpha + \beta }$ risk aversion (avoiding contexts similar to past failures), and exploration (high entropy procedures). The expected utility formulation naturally handles the explore-exploit tradeoff: early in learning, high entropy dominates selection, while after sufficient evidence accumulates, expected reward becomes the primary driver.

# 4.3 Contrastive Refinement of Procedures

As experience accumulates, procedures with both successful and failed instances are subjected to contrastive refinement to improve their accuracy and robustness. For a procedure $\mathrm { P r o c } _ { i }$ with sets of successful and failed contexts $S _ { i }$ and $\mathscr { F } _ { i }$ , the LLM performs discriminative comparison, D?? = ContrastiveExtract $( S _ { i } , { \mathcal { F } } _ { i } )$ , identifying differences in three dimensions: (i) precondition patterns $\langle \Delta \Psi _ { i } ^ { + }$ and $\Delta \Psi _ { i } ^ { - }$ ) that distinguish successful from failed initial contexts, (ii) action discrepancies $( \Delta \pi _ { i } )$ revealing missing or misordered actions, and (iii) postcondition mismatches $( \Delta \Phi _ { i } )$ that capture incomplete goal states. These discriminators drive explicit refinement operations

$$
\begin{array}{l} \Psi_ {i} \gets \Psi_ {i} \cup \Delta \Psi_ {i} ^ {+} \cup \{\neg \psi | \psi \in \Delta \Psi_ {i} ^ {-} \}, \\ \pi_ {i} \leftarrow \operatorname {M e r g e} \left(\pi_ {i}, \Delta \pi_ {i}\right), \\ \Phi_ {i} \leftarrow \Phi_ {i} \cup \Delta \Phi_ {i}. \tag {6} \\ \end{array}
$$

When distinct execution modes are detected, the procedure is specialized into separate variants with inherited reliability priors. This process progressively tightens applicability conditions and action precision, yielding interpretable improvements purely through memory edits rather than gradient updates.

# 4.4 Meta-procedural Composition

To extend reasoning beyond atomic skills, MACLA automatically discovers and learns meta-procedures that are structured compositions of procedures that capture recurrent long-horizon strategies. When a sequence of procedures $\langle \mathrm { P r o c } _ { i _ { 1 } } , \dots , \mathrm { P r o c } _ { i _ { m } } \rangle$ repeatedly leads to success under a common high-level goal, the agent abstracts it as $\mathrm { M P } _ { j } = \langle G _ { j } ^ { \mathrm { m e t a } } , \Psi _ { j } ^ { \mathrm { m e t a } }$ , $\{ \mathrm { P r o c } _ { i _ { 1 } } , \ldots , \mathrm { P r o c } _ { i _ { m } } \} , \Theta _ { j } \rangle$ . Here, $\Theta _ { j }$ denotes a lightweight control policy governing conditional transitions among sub-procedures based on the current observation and execution context, $\Theta _ { j } \big ( o _ { t } , \mathrm { i n d e x } \big ) \ \in \ \cdot$ {continue, skip, repeat, abort}. This policy is distilled by analyzing successful traces, where the LLM identifies observation patterns that triggered each branch—for example, repeating when postconditions are unmet, skipping when preconditions already hold, or aborting when failures recur. Each meta-procedure maintains its own Beta success posterior $\mathbf { \Gamma } _ { p } ( \sigma _ { j } \vert \mathcal { D } _ { j } ) =$ $\mathrm { B e t a } ( \alpha _ { j } , \beta _ { j } )$ and is refined periodically to add new branches, reorder sub-procedures, or prune redundant ones. Through these hierarchical compositions, MACLA acquires flexible “playbooks” that encapsulate extended strategies with conditional logic.

# 4.5 Ontological Semantic Grounding

To enable cross-context generalization (e.g., procedures learned on "mug" applying to "cup"), MACLA constructs a lightweight ontological semantic index during offline memory construction. We extract the $k _ { v o c a b }$ most frequent words from task descriptions and actions, then cluster semantically similar words using Sentence-Transformer embeddings [12] to form an implicit domain ontology:

$$
C _ {w} = \left\{w ^ {\prime} \mid \operatorname {s i m} \left(\phi (w), \phi \left(w ^ {\prime}\right)\right) > \theta_ {s i m} \right\} \tag {7}
$$

where each cluster $C _ { w }$ represents a semantic category (e.g., $C _ { \mathrm { c o n t a i n e r } } =$ {mug, cup, glass}). During retrieval, observations are mapped to these ontological categories, allowing procedures to match across lexically different but semantically equivalent contexts. This ontological grounding enables domain-adaptive generalization without requiring explicit knowledge engineering.

# 4.6 System Efficiency and Memory Management

To ensure practical scalability, MACLA employs efficient retrieval, bounded growth, and strict control over LLM usage. All procedures and meta-procedures are embedded in an approximate nearestneighbor index supporting sublinear retrieval $( O ( \log N _ { p } ) )$ for semantic search. The episode buffer stores at most $N _ { b } = 1 0 0 0$ steps, providing local context for LLM prompts and post-episode updates. Each procedure maintains a failure index limited to $K _ { \mathrm { f a i l } } = 1 5$ entries, managed through success-based removal, redundancy-aware eviction, and temporal decay, ensuring that memory remains concise and informative. To prevent memory saturation, procedures and meta-procedures are periodically pruned using a multi-factor utility score that balances reliability, usage frequency, and temporal relevance:

$$
U \left(\operatorname {P r o c} _ {i}\right) = \lambda_ {r} \cdot \frac {\alpha_ {i}}{\alpha_ {i} + \beta_ {i}} + \lambda_ {f} \cdot \frac {n _ {i}}{N _ {\text {t o t a l}}} + \lambda_ {t} \cdot e ^ {- \left(t _ {\text {c u r r e n t}} - t _ {i} ^ {\text {l a s t}}\right) / \tau} \tag {8}
$$

where $\frac { \alpha _ { i } } { \alpha _ { i } + \beta _ { i } }$ is the Bayesian success rate (reliability), $n _ { i }$ is the execution count of procedure ??, $N _ { \mathrm { t o t a l } }$ is the total invocations across all procedures in the current episode window, $t _ { \mathrm { c u r r e n t } }$ is the current episode index, $t _ { i } ^ { \mathrm { l a s t } }$ is the episode when ?? was last used, and $\tau$ is the temporal decay constant.

The weighting coefficients $\lambda _ { r } { = } 0 . 5$ , $\lambda _ { f } { = } 0 . 3$ , and $\lambda _ { t } \mathrm { = } 0 . 2$ reflect the relative importance of each factor: reliability receives the highest weight (0.5) as it directly predicts future success; frequency receives moderate weight (0.3) to favor well-tested procedures while avoiding over-retention of obsolete frequently-used skills; recency receives the lowest weight (0.2) to provide soft temporal decay without aggressive forgetting. These values were determined through grid search over $\{ 0 . 3 , 0 . 4 , 0 . 5 , 0 . 6 \} \times \{ 0 . 2 , 0 . 3 , 0 . 4 \} \times \{ 0 . 1 , 0 . 2 , 0 . 3 \}$ on ALFWorld validation, with the constraint $\lambda _ { r } + \lambda _ { f } + \lambda _ { t } = 1 . 0$ for interpretability. The selected configuration (0.5, 0.3, 0.2) yielded the best balance between retaining high-quality procedures ${ \mathrm { > } } 0 . 7$ success rate) and pruning low-utility entries $\scriptstyle ( < 0 . 4$ success rate), as validated later in Figure 4. Entries with the lowest utility are removed while ensuring diversity across goal clusters through stratified sampling. These operations keep the total memory footprint below 4 MB for hundreds of procedures.

Finally, MACLA limits LLM usage to a fixed budget of API calls per episode to cover segmentation, abstraction, and occasional refinement, while all retrieval, Bayesian scoring, and updates are

symbolic or vectorized. As a result, per-step runtime remains effectively constant and inference cost does not scale with experience. This memory-first design ensures that MACLA remains efficient, interpretable, and deployable for continual learning across long interaction horizons.The theoretical foundations are provided in Appendix D.

# 4.7 Algorithm

At runtime, MACLA executes a new task by coupling frozen semantic reasoning with memory-driven decision making. The agent receives an initial observation $o _ { 0 }$ (and, optionally, an instruction string) and embeds it as $\mathbf { h } _ { 0 } = \phi ( o _ { 0 } )$ . This embedding queries the semantic index of the external memory to retrieve a compact candidate set consisting of procedures $\{ \mathrm { P r o c } _ { i } \}$ and meta-procedures $\{ \mathrm { M P } _ { j } \}$ whose embeddings are most similar to ${ \bf h } _ { 0 }$ . Retrieval is approximate nearest neighbor over the concatenated descriptors of goals, preconditions, and postconditions, which keeps lookup sublinear in memory size.

Given the candidate set, MACLA ranks each item with a Bayesian expected-utility score that trades off contextual relevance, estimated success, risk, and information gain under the procedure’s Beta posterior. The highest-scoring item above a confidence threshold is selected; otherwise the agent falls back to zero-shot LLM reasoning for that step, logs the outcome, and continues. If a meta-procedure is chosen, execution proceeds hierarchically under its composition policy $\Theta _ { j } \big ( o _ { t } , \mathrm { i n d e x } \big ) \in$ {continue, skip, repeat, abort} until completion or abort; if an atomic procedure is chosen, the agent checks preconditions $\Psi _ { i }$ against $o _ { t }$ , invokes the action sketch $\pi _ { i }$ via the frozen LLM’s action formatter, and verifies postconditions $\Phi _ { i }$ to certify completion. In both cases the outcome updates $( \alpha , \beta )$ and appends the initial context to the corresponding success or failure set for later analysis.

After each execution, the agent re-embeds the new observation and repeats retrieval and selection until the task is solved or a horizon is reached. When a procedure accumulates both successes and failures, a contrastive pass is triggered: the LLM proposes discriminators that tighten $\Psi _ { i }$ , repair $\pi _ { i }$ , and refine $\Phi _ { i }$ , or if distinct modes are detected, specializes the procedure into variants that inherit prior counts. When successful episodes repeatedly traverse a small set of procedures in a stable order, the agent abstracts a meta-procedure with its own success posterior and a lightweight $\Theta _ { j }$ distilled from divergence points across traces. Throughout, memory remains bounded by pruning with a utility that blends reliability, frequency, and recency, and the LLM-call budget is capped, as retrieval, scoring, and updates are vectorized operations. The complete runtime procedure is outlined in Algorithm 1.

# 5 EXPERIMENTS

We evaluate MACLA on four challenging interactive agent benchmarks spanning diverse domains. All experiments use consistent hyperparameters across tasks to demonstrate generalization without task-specific tuning.

Algorithm 1 MACLA Runtime Procedure with Function Descriptions   
Require: observation $o_0$ , memory $\mathbb{M}$ (procedures, meta-procedures, in- dices), horizon $H$ 1: $\mathrm{h}\gets \phi (o_0)$ $\triangleright$ Embed observation   
2: $C\gets$ RETRIEVECANDIDATES(h, M)  Top-k ANN search   
3: while not TERMINAL and $t <   H$ do   
4: for all $c\in C$ do   
5: EU[c] $\leftarrow$ EXPECTEDUTILITY(c, $o_t,\mathbb{M})$ $\triangleright$ Compute Eq. 4   
6: end for   
7: $c^{\star}\gets \arg \max_{c\in C}$ EU[c]   
8: if EU[c] $<  \theta_{\mathrm{conf}}$ then   
9: $(o_{t + 1},y)\gets ZEROSHOTSTEP(o_t)$ LLM generates action directly   
10: else if $c^{\star}$ is MPj then   
11: $(o_{t + 1},y)\gets$ EXECUTEMETA(MPj, $\Theta_j,o_t)$ Run with control policy   
12: $(\alpha_{j},\beta_{j})\gets$ UPDATEBETA(( $\alpha_{j},\beta_{j}),y$ $\triangleright$ $\alpha \leftarrow \alpha +y$ $\beta \leftarrow \beta +(1 - y)$ 13: else $c^{\star}$ is atomic Proc i   
14: if CHECKPRE( $\Psi_i,o_t)$ then Verify preconditions match $o_t$ 15: $(o_{t + 1},y)\gets$ EXECUTEPROc( $\pi_i,o_t)$ Instantiate & execute   
16: $y\gets y\land$ CHECKPOST( $\Phi_i,o_{t + 1})$ Verify postconditions in $o_{t + 1}$ 17: else   
18: $(o_{t + 1},y)\gets$ ZEROSHOTSTEP(o_t) Preconditions failed,   
fallback   
19: end if   
20: $(\alpha_i,\beta_i)\gets$ UPDATEBETA(( $\alpha_{i},\beta_{i}),y$ 21: RECORD CONTEXT( $S_{i},F_{i},o_{t},y)$ Add to success/fail sets   
22: end if   
23: if REFINETRIGGER(c*) then If $|S|$ $|\mathcal{F}|\geq 3$ 24: CONTRASTIVEREFINE(c*) LLM compares S vs. F (\$4.3)   
25: end if   
26: h $\leftarrow$ $\phi (o_{t + 1})$ . $C\gets$ RETRIEVECANDIDATES(h,M); t $\leftarrow t + 1$ 27: end while   
28: if ELIGIBLEFORMETA(trace) then If $\geq 3$ procs in stable order   
29: EXTRACTORREFINEMETA(trace,M) Create/update meta-proc   
30: end if   
31: PRUNEANDMAINTAIN(M) Remove low-utility via Eq. 8

# 5.1 Experimental Setting

Memory Architecture: Episode buffer $N _ { b u f f e r } { = } 1 0 0 0$ (stores recent observations and actions for temporal context provision during action generation); procedural memory $N _ { p r o c } { = } 2 0 0$ (capacity for extracted reusable skills); meta-procedural memory $N _ { m e t a } { = } 5 0$ (capacity for hierarchical procedure compositions). Critically, MACLA does not store raw trajectories. Instead, the LLM segments each episode into coherent sub-tasks and extracts structured procedures (Section 4.1). Duplicate detection with similarity threshold $\theta _ { d u p } { = } 0 . 8 5$ prevents redundant storage. Through this process, the 2,851 ALFWorld training trajectories compress into approximately 187 unique procedures—demonstrating efficient knowledge distillation from experience.

Bayesian Selection. Information gain weight $\lambda _ { i n f o } { = } 0 . 1$ , failure cost $C _ { f a i l } { = } 0 . 5$ . These parameters balance exploration (trying uncertain procedures to reduce epistemic uncertainty) with exploitation (selecting high-posterior reliable procedures).

Contrastive Refinement. Minimum contexts ??????????=??????????=3. Re- $n _ { m i n } ^ { s } { = } n _ { m i n } ^ { f } { = } 3$ finement activates only when a procedure has accumulated at least 3 successes and 3 failures, ensuring sufficient statistical evidence for discriminative pattern extraction.

LLM Configuration. Llama-2-7B [18] via Ollama with 4-bit quantization and temperature $T { = } 0 . 7$ . The LLM parameters remain frozen throughout all experiments—learning occurs exclusively through external memory updates.

Benchmarks and Dataset Statistic: ALFWorld [16] (2,851 train, 274 test) is a text-based embodied environment with six household tasks (e.g., retrieval, placement). We follow the standard train/validation-seen/validation-unseen split, where test trajectories feature novel object-location configurations. WebShop [25] (1,624 train, 200 test) simulates e-commerce search over 12,087 products, requiring agents to follow natural-language instructions via multi-step navigation and filtering. TravelPlanner [21] (1,000 train, 180 validation, 45 test) involves multi-day itinerary planning under hard constraints (budget, dates) and soft preferences (cuisine, attractions). Evaluation uses Common Sense (CS) and Hard Constraint (HC) scores. InterCodeSQL [24] benchmarks interactive text-to-SQL generation over diverse schemas, requiring correct handling of schema relationships and varying query difficulty.

# 5.2 Experimental Results and Analysis

Table 1 compares MACLA against state-of-the-art baselines across all benchmarks. We organize baselines into three paradigms: promptbased methods using in-context learning, outcome refinement approaches optimizing trajectory-level rewards, and process refinement methods refining step-level generation. MACLA achieves the highest average performance $( 7 8 . 1 \% )$ while using a 7B parameter model, demonstrating that domain-agnostic procedural memory with Bayesian selection and contrastive refinement enables competitive performance without task-specific engineering.

In Table 1, MACLA achieves state-of-the-art results on TravelPlanner (83.3 CS) and ALFWorld-Unseen $( 9 0 . 3 \% )$ , outperforming methods that rely on models $1 0 \times$ larger. Its strong performance across all benchmarks demonstrates cross-domain generalization, while the positive generalization gap on ALFWorld $( + 3 . 1$ points for unseen vs. seen) indicates robust compositional transfer rather than memorization.

# Ablation Study

To understand the contribution of each component in MACLA, we conduct an ablation study by systematically removing key modules. Table 2 reports results on ALFWorld (seen and unseen splits), evaluating: (1) Bayesian procedural selection (Section 4.2), (2) contrastive learning from failed trajectories (Section 4.3), (3) metaprocedural composition (Section 4.4), and (4) ontological semantic grounding (Section 4.5). Removing Bayesian selection leads to the largest degradation (–7.7 seen, –9.1 unseen), highlighting its role in effective exploration. Meta-procedural composition is essential for compositional generalization, with a sharp drop on unseen tasks (–11.9). Contrastive learning and ontological clustering provide smaller but consistent improvements $( - 3 . 5 / - 4 . 6 $ and $- 4 . 3 / - 6 . 2$ respectively). Overall, all four components contribute synergistically to MACLA’s robustness.

Computational and Memory Efficiency Analysis

MACLA’s efficiency comes from three design choices: (1) the frozen LLM eliminates gradient updates, (2) external memory construction is trivially parallelizable, and (3) learned procedures amortize LLM costs across episodes. Table 3 summarizes training costs.

# Training and Adaptation

MACLA builds memory in 56s ( $2 , 8 0 0 \times$ faster than IPR [22]) by extracting reusable procedures with a frozen LLM, instead of iterative parameter updates. For new tasks, IPR post-trains for 44.8 GPUhrs, whereas MACLA ingests new trajectories in seconds. Memory construction scales nearly linearly with resources.

# Memory Capacity and Performance Saturation

Figure 2 reveals logarithmic performance growth across three capacity regimes: (1) Undercapacity (25–50): Sharp degradation $6 4 . 1 \%$ unseen at 25) due to insufficient task coverage, forcing frequent zero-shot fallback. Low posterior (0.61) indicates pruning removes procedures before adequate validation. (2) Optimal (100– 200): Rapid improvement $8 5 . 6 \%  9 0 . 3 \%$ unseen), capturing core reusable procedures. The system extracts 187 unique procedures from 2,851 training trajectories (15:1 compression), leaving 13 of 200 slots unused—indicating automatic discovery of task-space boundaries. (3) Overcapacity (300): Performance declines (- $- 0 . 2 \%$ unseen) despite more slots, as redundant variants introduce retrieval noise. The posterior plateau at 0.79 confirms saturation. This bounded growth (3.6 MB footprint) contrasts with neural approaches requiring unbounded parameter expansion, demonstrating ALFWorld’s task space has finite complexity discoverable through procedural abstraction.

# Bayesian Posterior Evolution and Convergence

Figure 3 demonstrates uncertainty-aware learning through Bayesian posterior evolution. Panel (a) shows diverging $\alpha$ trajectories reflecting the explore-exploit tradeoff: general-purpose procedures (Navigate) accumulate evidence fastest through frequent invocation, while specialized procedures (Heat/Cool) converge slower but maintain high posteriors when applicable. Panel (b) reveals all top procedures stabilize above the 0.75 reliability threshold within 50 test episodes, with posterior variance decreasing as evidence accumulates:

$$
\operatorname {V a r} [ \rho ] = \frac {\alpha \beta}{(\alpha + \beta) ^ {2} (\alpha + \beta + 1)} \xrightarrow {\alpha + \beta \rightarrow \infty} 0 \tag {9}
$$

By episode 50, $\alpha + \beta > 3 0$ for all procedures, yielding standard deviations $< 0 . 0 5$ —demonstrating principled uncertainty quantification. This self-reinforcing cycle ensures memory quality: poor procedures accumulate failures (high ??), receive low utility scores, and are pruned before reaching high evidence totals.

# Memory Pruning Characteristics

Figure 4 validates MACLA’s self-regulating pruning mechanism. Panel (a) shows clear distributional separation: $7 3 \%$ of pruned procedures have success rates below 0.5 (primarily spurious extractions from failed exploration trajectories), while $8 1 \%$ of retained procedures exceed 0.7. The utility-based criterion effectively discriminates signal from noise:

$$
U (p) = 0. 5 \cdot \frac {\alpha}{\alpha + \beta} + 0. 3 \cdot \min  \left(1, \frac {\text {c o u n t}}{1 0}\right) + 0. 2 \cdot \left(1 - \frac {\text {a g e}}{\text {m a x} _ {-} \text {a g e}}\right) \tag {10}
$$

Panel (b) reveals $6 8 \%$ of pruned procedures are both young $< 4 0$ trajectories old) and rarely used $^ { < 5 }$ invocations)—the system identifies unpromising candidates early rather than wasting execution budget.

Table 1: Performance comparison across four agent benchmarks. Baseline results are from [22] and [4]. All metrics report average reward or quality score (0–100 scale, higher is better). Best results per column in bold.   

<table><tr><td rowspan="2">Method</td><td rowspan="2">WebShop</td><td rowspan="2">InterCodeSQL</td><td rowspan="2">TravelPlanner</td><td colspan="2">ALFWorld</td><td rowspan="2">Avg.</td></tr><tr><td>Seen</td><td>Unseen</td></tr><tr><td colspan="7">Prompt-based Methods</td></tr><tr><td>GPT-4 [1]</td><td>63.2</td><td>38.5</td><td>71.9</td><td>42.9</td><td>38.1</td><td>50.9</td></tr><tr><td>GPT-3.5-Turbo [10]</td><td>62.4</td><td>37.8</td><td>-</td><td>7.9</td><td>10.5</td><td>29.7</td></tr><tr><td>Llama-2-7B [18]</td><td>17.9</td><td>4.0</td><td>-</td><td>0.0</td><td>0.0</td><td>5.5</td></tr><tr><td colspan="7">Outcome Refinement Methods</td></tr><tr><td>Llama-2-7B + SFT [2]</td><td>60.2</td><td>54.9</td><td>-</td><td>60.0</td><td>67.2</td><td>60.6</td></tr><tr><td>Llama-2-7B + RFT-PPO [13]</td><td>64.2</td><td>52.4</td><td>-</td><td>22.1</td><td>29.1</td><td>42.0</td></tr><tr><td>Llama-2-7B + RFT-CR [30]</td><td>63.6</td><td>56.3</td><td>-</td><td>62.9</td><td>66.4</td><td>62.3</td></tr><tr><td>Llama-2-7B + ETO [17]</td><td>67.4</td><td>57.2</td><td>-</td><td>68.6</td><td>72.4</td><td>66.4</td></tr><tr><td colspan="7">Process Refinement Methods</td></tr><tr><td>Llama-2-7B + Step-PPO [22]</td><td>64.0</td><td>60.2</td><td>-</td><td>65.7</td><td>69.4</td><td>64.8</td></tr><tr><td>Llama-2-7B + IPR [22]</td><td>71.3</td><td>61.3</td><td>-</td><td>70.3</td><td>74.7</td><td>69.4</td></tr><tr><td>Claude-3.5-Sonnet†[4]</td><td>-</td><td>-</td><td>65.5</td><td>82.5</td><td>74.7</td><td>74.2</td></tr><tr><td>Qwen2.5-72B‡[4]</td><td>-</td><td>-</td><td>63.8</td><td>85.7</td><td>77.2</td><td>75.6</td></tr><tr><td>Llama-2-7B + MACLA</td><td>70.2</td><td>59.3</td><td>83.3</td><td>87.2</td><td>90.3</td><td>78.1</td></tr></table>

†Substantially larger models (Claude-3.5: proprietary, Qwen2.5: 72B vs. 7B parameters). TravelPlanner reports Commonsense (CS) score; other benchmarks report task completion reward.

![](images/c4d996c75a5932a26a17c65a6dcb23b7e52361c035e1289b05b8d57880d12f56.jpg)

![](images/c632ebd5afcf3b71b1f8bfbb52dc4f8c5bf55d10d302cb9a1701b1bec3213fa9.jpg)  
Figure 2: Ablation study varying maximum procedural memory capacity. (a) Success rate on ALFWorld seen/unseen splits saturates beyond 150 procedures, with diminishing returns from $\mathbf { 1 5 0 } {  } 2 \mathbf { 0 0 }$ $( + 1 . 6 \%$ unseen) and slight decline at 300 $( - \mathbf { 0 . 2 \% } )$ . (b) Average Bayesian posterior $\frac { \alpha } { \alpha + \beta }$ plateaus at 0.79, showing extra capacity adds redundancy rather than quality.

Table 2: Ablation study on ALFWorld with Llama-2-7B backbone. Each component is removed in turn to assess its contribution. Results are success rates (0–100).   

<table><tr><td>Config.</td><td>Bayes.</td><td>Contr.</td><td>Meta</td><td>Ontol.</td><td>Seen</td><td>Unseen</td></tr><tr><td>Full MACLA</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>87.1</td><td>90.3</td></tr><tr><td>w/o Bayesian</td><td>X</td><td>✓</td><td>✓</td><td>✓</td><td>79.4</td><td>81.2</td></tr><tr><td>w/o Contrast.</td><td>✓</td><td>X</td><td>✓</td><td>✓</td><td>83.6</td><td>85.7</td></tr><tr><td>w/o Meta</td><td>✓</td><td>✓</td><td>X</td><td>✓</td><td>81.2</td><td>78.4</td></tr><tr><td>w/o Ontology</td><td>✓</td><td>✓</td><td>✓</td><td>X</td><td>82.8</td><td>84.1</td></tr></table>

Bayes.: probabilistic selection (Sec. 4.2); Contr.: success/failure refinement (Sec. 4.3); Meta: hierarchical composition (Sec. 4.4); Ontol.: semantic clustering (Sec. 4.5).

Critically, the top-right quadrant is empty: no high-quality procedures $_ { \mathrm { > } 0 . 7 }$ success, ${ > } 1 0$ uses) are pruned, confirming conservative

Table 3: Efficiency comparison. MACLA avoids iterative training, yielding $9 9 . 9 6 \%$ less training compute while maintaining competitive performance.   

<table><tr><td>Method</td><td>Training (GPU-hrs)</td><td>WebShop</td><td>ALFWorld Unseen</td></tr><tr><td>IPR [22]</td><td>44.8</td><td>71.3</td><td>74.7</td></tr><tr><td>SFT [2]</td><td>8.0</td><td>60.2</td><td>67.2</td></tr><tr><td>ETO [17]</td><td>20.0</td><td>67.4</td><td>72.4</td></tr><tr><td>MACLA</td><td>0.016</td><td>70.2</td><td>90.3</td></tr><tr><td>Speedup vs IPR</td><td>2,800×</td><td>-</td><td>+15.6 pts</td></tr></table>

Training cost: IPR = 5.6h on $8 \times \mathrm { A 1 0 0 }$ (44.8 GPU-hrs); MACLA = 56s on 1×RTX 3090 (0.016 GPU-hrs), representing a $2 { , } 8 0 0 \times$ reduction. MACLA’s frozen-LLM architecture eliminates iterative parameter training while achieving superior generalization on unseen tasks $_ { + 1 5 . 6 }$ points on ALFWorld-Unseen vs IPR).

![](images/70ba1ed473b0b0d98cae1644f3aa3fe82e845b7cdeab878a4e7ae2ce6293ec0a.jpg)  
(a) Evolution for Top-5 Procedures

(b) Posterior Refinement During Evaluation   
Figure 3: Bayesian learning dynamics for top-5 procedures during 200 test episodes. (a) Cumulative success count ?? grows at different rates: Navigate (blue) reaches $\mathbf { 1 5 0 + }$ invocations, while task-specific procedures (Heat/Cool, green/red) accumulate evidence more slowly due to limited applicability. (b) Posterior success rates $\frac { \alpha } { \alpha + \beta }$ converge above 0.75 within 50 episodes, with variance decreasing as $O ( 1 / ( \alpha { + } \beta ) )$ .   
![](images/5238e8580941dc453867b9dcd574c5887ff7366b99f77c9a0f28b45747f6e3bf.jpg)  
Mean: 0.38 Mean: 0.73

![](images/9e9c25125359183367b46c7fa603f382a0e11876882c7ec2b2d77497433c6f3e.jpg)  
(a) Success Rate Distribution

![](images/9d102cabf8515fec54072e3866aa426e03650ec72e2381ef66a0696b1736e5ac.jpg)  
(b) Pruned Procedure Characteristics   
Figure 4: Analysis of $\bf { 2 0 0 + }$ pruned procedures during ALFWorld training. (a) Bimodal success rate distribution: pruned procedures (red, mean 0.42) separate cleanly from retained procedures (green, mean 0.79), validating utility-based retention. (b) Scatter plot shows pruned procedures cluster in bottom-left (young $^ +$ rarely used), with no high-quality procedures (>0.7 success, $\mathbf { \Gamma } _ { > 1 0 }$ uses) pruned.

retention. This automatic quality control explains why performance plateaus at 187 procedures (mean posterior 0.79) without manual curation.

# Task-Specific Memory Effectiveness

Figure 5 explains SQL underperformance through three metrics. Low reuse $( 5 1 \% )$ : SQL queries are schema-specific, e.g., customers. does not apply to employees.experience. ALFWorld generalizes via placeholders (<object>), but SQL column names vary unpredictably. Low reliability $( 6 4 \% )$ : Schema mismatches, join com plexity, and edge cases accumulate failures ( $\beta$ counts), suppressing posteriors. Minimal composition $( 1 8 \% )$ : SQL queries are atomic (2-3 actions), too short for meta-procedures. ALFWorld tasks nat urally decompose into multi-step sub-procedures. MACLA excels when tasks have: (1) reusable actions, (2) hierarchical structure, and (3) consistent semantics — SQL violates all three.

# 6 CONCLUSION

We presented MACLA, a framework that decouples reasoning from learning by maintaining a frozen LLM and performing all adaptation in an external hierarchical procedural memory through Bayesian

selection, contrastive refinement, and meta-procedural composition. MACLA achieves $7 8 . 1 \%$ average performance across four benchmarks using only a 7B model, with state-of-the-art results on ALF-World ( $8 7 . 2 \%$ seen; $9 0 . 3 \%$ unseen) and TravelPlanner $( 8 3 . 3 \% )$ . The system compresses 2,851 ALFWorld training trajectories into 187 reusable procedures through semantic abstraction and duplicate detection, demonstrating efficient knowledge distillation.