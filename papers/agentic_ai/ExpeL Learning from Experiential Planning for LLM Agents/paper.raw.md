# ExpeL: LLM Agents Are Experiential Learners

Andrew Zhao,♠ Daniel Huang, ♣ Quentin Xu, ♣ Matthieu Lin, ♣ Yong-Jin Liu, ♣ Gao Huang ♠*

♠ Department of Automation, BNRist, Tsinghua University

♣ Department of Computer Science, BNRist, Tsinghua University

{zqc21,huang-jy22,xgd22,lyh21}@mails.tsinghua.edu.cn,

{liuyongjin,gaohuang}@tsinghua.edu.cn

![](images/d782d10291bca84918c988857f1fbc1c95a0910ded81557c4d15467935211018.jpg)

![](images/ab3594e50fd044068c5e828dc7f7b798f9ecca2fa103d029d2d3f7317ee6667a.jpg)

![](images/c14ead52fe5152ff79d93562b4de67ab3350e4b3d4ce6bc9e26a04197d2a6b96.jpg)

# Abstract

The recent surge in research interest in applying large language models (LLMs) to decision-making tasks has flourished by leveraging the extensive world knowledge embedded in LLMs. While there is a growing demand to tailor LLMs for custom decision-making tasks, finetuning them for specific tasks is resource-intensive and may diminish the model’s generalization capabilities. Moreover, state-of-the-art language models like GPT-4 and Claude are primarily accessible through API calls, with their parametric weights remaining proprietary and unavailable to the public. This scenario emphasizes the growing need for new methodologies that allow learning from agent experiences without requiring parametric updates. To address these problems, we introduce the Experiential Learning (ExpeL) agent. Our agent autonomously gathers experiences and extracts knowledge using natural language from a collection of training tasks. At inference, the agent recalls its extracted insights and past experiences to make informed decisions. Our empirical results highlight the robust learning efficacy of the ExpeL agent, indicating a consistent enhancement in its performance as it accumulates experiences. We further explore the emerging capabilities and transfer learning potential of the ExpeL agent through qualitative observations and additional experiments.1

A computer program is said to learn from experience

$E$ with respect to some class of tasks $T$ and

performance measure $P$ , if its performance at tasks in

$T$ , as measured by $P$ , improves with experience $E$ .

Tom Mitchell

# 1 Introduction

Machine learning research has long been captivated by the potential of autonomous agents and their capabilities. In recent times, incorporating large language models into these agents (Wang et al. 2023b; Xi et al. 2023) has unveiled a broad spectrum of applications, even extending beyond academia (Yang et al. 2023a; Nakajima 2023; Significant-Gravitas 2023). One of the significant advantages of LLMs lies in their world knowledge, allowing them to be inherently versatile across various scenarios (Zhao et al. 2023b).

On the one hand, previous works investigated finetuning LLMs with a large number of environment interactions (Yao et al. 2023c) or with a large amount of human-labeled datasets (Nakano et al. 2021; Shaw et al. 2023). This class of methods incurs high computational costs and needs access to the LLM’s parametric weights. Furthermore, finetuning an LLM restricts its functionalities and can hurt its generalization abilities (Du et al. 2022). On the other hand, prompting methods can augment an LLM with better sequential decision-making planning abilities with only a few in-context examples (Hao et al. 2023; Lin et al. 2023b; Sun et al. 2023). However, since current LLMs are bounded by context window size (Tworkowski et al. 2023), these agents have no recollections of what they have seen, and therefore no learning can be done outside of a few demonstrations. So, how can we strike a balance between these paradigms?

We present the Experiential Learning (ExpeL) agent as a solution. Our agent autonomously gathers experiences from a collection of training tasks through trial and error. From these experiences, it derives natural language insights and employs its own successful experiences as in-context examples during test time. Our agent’s learning process is analogous to a student studying for an exam and then taking it on a single attempt, reflecting many real-world situations. Unlike self-improvement methods like Reflexion (Shinn et al. 2023), our approach emphasizes the importance of retaining experiences across multiple tasks to enhance agent performance. Moreover, ExpeL learns without parameter updates, making it compatible with powerful closed-source models like GPT-4 or Claude. Lastly, the experience-gathering step does not require a large amount of data or human labels.

We evaluated ExpeL on three vastly different domains and consistently outperformed strong baselines. Additionally, we showcased a transfer learning scenario where our agent that accumulated knowledge from source tasks showed positive forward transfer to target tasks. Finally, we highlighted some unexpected emerged abilities the ExpeL agent gained.

In summary, our key contributions are as follows: (1) we introduced ExpeL, a novel LLM agent that autonomously learns from experience without gradient updates; (2) We evaluated ExpeL on a diverse set of tasks to showcase its learning abilities and improvement on top of existing planning methods; (3) we showed a novel setting of transfer learning for our LLM agent and demonstrated forward transferability from source tasks to target tasks. Lastly, we believe that as planning algorithms and foundational models continue to improve, ExpeL’s paradigm stands to gain significant benefits from their enhanced performances.

# 2 Related Work

We discuss the most relevant related works in this section. See Appendix A for detailed discussions on related works.

Prompt-based Learning: Prompt-based learning refines label prediction tasks by modifying the input context, facilitating swift adaptation to new tasks with minimal data (Liu et al. 2023a). This approach capitalizes on LLMs for answers without parameter tuning as they can be augmented using in-context learning (Brown et al. 2020). LAMA (Petroni et al. 2019) and GPT-3 (Brown et al. 2020) are early works that promoted this formulation. Efforts to reduce the intricacies of prompt design include automatic reasoning chains for NLP (Kojima et al. 2022; Zhang et al. 2023). Similarly, the ExpeL agent also autonomously learns from experiences using extracted insights and self-generated incontext trajectories by altering the execution prompt.

Retrieval Augmented Generation (RAG): Retrieval allows LLMs to access databases, mitigating hallucinations (Li et al. 2022; Wang, Yang, and Wei 2023; Rubin, Herzig, and Berant 2022; Liu et al. 2022). Retrieval has also been used to enhance the capabilities of decision-making agents (Humphreys et al. 2022; Zhao et al. 2023a). In contrast to these works, we focus on retrieving the ExpeL agent’s self-generated experiences, thus reducing the dependency on gold examples and leveraging domain-specific corpus.

Planning for LLM Agents: Application of LLM agents in fields like robotics, natural sciences, game-playing, and workflows has surged, with emphasis on their world knowledge in fewshot settings (Ha, Florence, and Song 2023; Mu et al. 2023; Bran et al. 2023; Boiko, MacKnight, and Gomes 2023; Yang et al. 2023b; Lin et al. 2023a; Nakano et al. 2021; Wang et al. 2023c; Liu et al. 2023b). Moreover, LLMs have demonstrated promising zero/few-shot planning and reasoning capabilities in various configurations (Sumers et al. 2023), including embodied environments and reasoning tasks (Huang et al. 2022; Yao et al. 2023a; Wei et al. 2022b; Yao et al. 2023b; Gong et al. 2023).

Self-improvement and Memory for LLM Agents: Agents like Reflexion showcase feedback-based improvement, yet often lack cross-task memory (Shinn et al. 2023). Other agents exhibit potential in persistent memory within multiagent contexts (Park et al. 2023; Maas et al. 2023). Our ExpeL agent combines these approaches, focusing on tasksolving while benefiting from self-generated in-context examples and abstracted insights from memory.

# 3 Preliminaries

Complex Interactive Tasks We work with complex interactive tasks where at each time step $i \in \{ 0 , \ldots , H \}$ , the agent receives an observation $o \in \mathcal { O }$ , and from its observation history $H _ { t }$ decides to perform action $a \in { \mathcal { A } }$ . The objective of the agent is to achieve some goal $g \in { \mathcal { G } }$ . We only deal with deterministic environments in this work.

Large Language Models A large language model is a statistical model of the natural language, typically a neural network. In our setting, we use an autoregressive language model (OpenAI 2023; Brown et al. 2020; Touvron

![](images/28af3adbcb08dc655049797895061d547e6170b61b70bd643c43db909ca3c337.jpg)

![](images/3effefa4775872682ad6b55c8c6240ac327d6ea0b39e5f59c655c6c4ea4846aa.jpg)  
Figure 1: ExpeL Agent Overview. Left: ExpeL operates in three stages: (1) Collection of success and failure experiences into a pool. (2) Extraction/abstraction of cross-task knowledge from these experiences. (3) Application of the gained insights and recall of past successes in evaluation tasks. Right: (A) Illustrates the experience gathering process via Reflexion (Shinn et al. 2023), enabling task reattempt after self-reflection on failures. (B) Illustrates the insight extraction step. When presented with success/failure pairs or a list of $L$ successes, the agent dynamically modifies an existing list of insights $\hat { \iota }$ using operations ADD, UPVOTE, DOWNVOTE, and EDIT. This process has an emphasis on extracting prevalent failure patterns or best practices.

et al. 2023b,a; Chowdhery et al. 2023), which given an ordered list of existing tokens $\mathbf x = \{ x _ { 1 } , x _ { 2 } , . . . , x _ { l - 1 } \}$ , outputs the probability of the next token $p ( x _ { l } \mid x _ { < l } )$ . An instructionfollowing LLM (Thoppilan et al. 2022; Chung et al. 2022; Wei et al. 2022a) is typically finetuned on various NLP tasks that are formatted into instruction, input, response tuples (Taori et al. 2023). Instruction-tuned models are better at following natural language instructions which alleviates the need for heavy prompt engineering (Wei et al. 2022a).

ReAct and Reflexion ReAct (Yao et al. 2023b) and Reflexion (Shinn et al. 2023) are promising frameworks enabling the aforementioned proficiency of LLMs in reasoning and self-improvement. ReAct explicitly intertwines observations, actions, and thoughts, providing a foundation for robust planning and reasoning capabilities. Building upon it, Reflexion introduces an additional reflective step before reattempting the subsequent trial of the same task, enhancing the model’s adaptive learning process.

# 4 ExpeL: An Experiential Learning Agent

Recent advancements in generative LLMs suggest an intriguing approach. Rather than altering the LLM parameters, adjusting the prompts may be more beneficial: this strategy ensures that the LLM’s inherent common sense knowledge remains intact, allowing for superior generalization (Liu et al. 2023a). Furthermore, some of the most potent language models are proprietary (OpenAI 2023; An-

thropic 2023). Thus, focusing on prompt-based methods seems promising as a way to harness the strengths of these advanced LLMs. Additionally, previous works on learning in LLM agents have primarily been trained on extensive human-labeled datasets (Lin et al. 2023a; Shaw et al. 2023) or improved via iterative retries (Shinn et al. 2023) on a single task. A relatively less explored area is facilitating agents to learn autonomously from their own experiences, similar to a student gaining insights from practicing for an exam. The student tackles practice problems multiple times to derive insights. At the exam, the student rely solely on these insights and draw memories of similar problems to answer the questions with one attempt. With this in mind, we wish to design an LLM agent that autonomously gathers experiences and extracts insights, then uses these cross-task insights and memories of similar tasks to aid its decision-making.

We aim to enhance a planning LLM agent, such as Re-Act, with learning abilities that allow it to improve through inter-task experiences without any parameter updates. Inspired by the cognitive abilities inherent in human learning, as well as the benefits observed in self-learning autonomous agents and the progress made in prompt-based methods, we developed the Experiential Learning (ExpeL) agent. During the training stage, the agent interacts with the environment, gathering experiences via trial and error. These experiences are stored in an experience pool (Lin 1992). From this pool, the agent later extracts insights, similar to off-policy learn-

ing (Watkins and Dayan 1992), in which the agent can learn from experiences of a behavior policy. During the evaluation stage, the agent attempts unseen tasks with a single try, augmented with extracted insights and successful trajectories in its experience pool gathered from the training stage. Refer to Fig. 1 for detailed information on our agent framework.

# 4.1 Gathering Experiences

To gather diverse experiences that can be useful to extract information from, we leverage Reflexion (Shinn et al. 2023) to continuously retry the training task at most $Z$ times. In particular, the agent will be given a training task $t _ { n }$ at the $z$ -th trial, fewshot examples $F _ { \mathrm { m a n u a l } }$ and past reflections $\nu _ { n , z }$ (initially, $\nu _ { n , 0 }$ is the empty string). At first, the agent will attempt the task with fewshot examples concatenated with its current trajectory $\tau _ { n , 0 }$ as the context, and use ReAct (Yao et al. 2023b) as the base planning algorithm, $\mathrm { L L M } _ { \mathrm { R e A c t } } ( \cdot { \textrm { \textsf { | } } } \tau _ { n , 0 } , F _ { \mathrm { m a n u a l } } , \nu _ { n , 0 } )$ . On the $z$ -th trial, when the agent finishes the task or the maximum number of steps $H$ is reached, the ExpeL agent’s experience pool $\boldsymbol { B }$ ingests the trajectory $\tau _ { n , z }$ . Then, if the agent succeeds, it moves on to the next task. However, if the agent fails, it will look at its failed trajectory and self-reflect to produce $\nu _ { n , z + 1 } = \mathrm { c o n c a t } ( \nu _ { n , z } , \mathrm { L L M } _ { \mathrm { r e f l e c t } } ( \tau _ { n , z } ) )$ to see where it can do better on the next retry, concatenated with the previous reflections. In the next retry, the agent will augment its context with reflection $\scriptstyle \nu _ { n , z + 1 }$ , the input to the LLM policy, $\mathbf { L L M } _ { \mathrm { R e A c t } } ( \cdot \mid \tau _ { n , z + 1 } , F _ { \mathrm { m a n u a l } } , \nu _ { n , z + 1 } )$ .

To highlight, this trial and error way of gathering experiences not only improves the chances of getting more positive examples for experience recall during evaluation but also allows for collecting valuable success/failure pairs used for comparisons during insight extraction (Sec. 4.2). The pseudo-code can be found in Alg. 1.

# 4.2 Learning from Experiences

Human learning occurs mainly either by storing successful trajectories in memory, which can be later recalled as specific examples, or by extracting high-level insights from experiences, enabling generalization to novel situations. ExpeL considers both of these learning modes to boost task performance. Concretely, an instruction $I$ given to an LLM agent can be broken down into task specifications and fewshot examples. We can augment task specifications with an agent’s extracted insights from past experiences, where an instruction-following LLM can be leveraged (OpenAI 2023) to follow them closely. For fewshot examples, we can allow the agent to retrieve from its experience pool with top- $k$ relevant examples to aid its decisions. Next, we detail our experience recall and insight extraction mechanisms.

Similar Experiences as Demonstrations Works have shown that using in-context examples that are semantically similar to the task at hand results in better performance (Liu et al. 2022). Moreover, when involved in a novel situation, humans also recall from their memory similar tasks they’ve solved as references when attempting the task (Kahneman 2011). Motivated by these observations, we propose experi-

ence recall to retrieve successful trajectories from the experience pool gathered during training based on task similarity.

Concretely, we used the Faiss vectorstore (Johnson, Douze, and Jegou ´ 2019) as the experience pool, kNN retriever and all-mpnet-base-v2 (Song et al. 2020) embedder to obtain top- $k$ successful trajectories that have the maximum inner-product task similarity with the evaluation task. The advantage of using task similarity as the retrieval rank is that if the agent repeats a task or does a task similar to an existing successful trajectory from the experience pool, the agent only needs to closely imitate the successful trajectory and have less burden on ability extrapolation.

Learning from Successes and Failures To leverage the diverse outcomes gathered during the experience collection phase, we believe the agent should analyze experiences in two distinct ways. First, we let the agent compare a failed trajectory with a successful trajectory for the same task. This comparison offers a concrete understanding of the agent’s shortcomings, highlighting the correct and incorrect actions. Second, we let the agent identify patterns within a set of successful trajectories from different tasks. This approach sheds light on common “good practices” that the agent can adopt to ensure success in evaluation tasks.

For the implementation, we give the agent’s instructionfollowing $\mathrm { L L M } _ { \mathrm { i n s i g h t s } }$ several operators to apply on an existing set of insights ˆι. We initialize the set of insights to an empty set $\hat { \iota } = \varnothing$ and iteratively provide the LLM with fail/success pairs or lists of $L$ successes (created by sampling without replacement) from the experience pool. The operations the LLM can perform are: ADD a new insight, EDIT the content of an existing insight, DOWNVOTE to disagree with an existing insight, or UPVOTE to agree with an existing insight. A newly added insight will have an initial importance count of two associated with it, and the count will increment if subsequent operators UPVOTE or EDIT are applied to it and will decrement when DOWNVOTE is applied to it. If an insight’s importance count reaches zero, it will be removed. This particular design choice robustifies the process since even successful trajectories can be suboptimal and mislead the generated insights. The prompt template we used can be found in Fig. 2. We kept the maximum size for a list of successes to $L$ and used $\mathtt { g p t - 4 - 0 6 1 3 }$ as the default $\mathbf { L L M _ { \mathrm { i n s i g h t s } } }$ . We empirically found that $\mathtt { g p t - 4 - 0 6 1 3 }$ is better than $\mathtt { g p t } - 3 . 5 \mathtt { - t u r b o - } 0 6 1 3$ at following instructions on how to use the insight extraction operators and hallucinated less. Pseudo-code for this process can be found in Alg. 2. Finally, ExpeL utilizes these generated insights $\hat { \iota }$ in the task inference phase, described next.

# 4.3 Task Inference

After the agent gathers experiences, extracts insights from them, and sets up a vectorstore of successful trajectories, it can proceed to the evaluation. For each task, the task specifications will be augmented with the concatenation of the full list of extracted insights ${ \hat { \iota } } = \operatorname { c o n c a t } ( \iota _ { 1 } , \iota _ { 2 } , \iota _ { 3 } , \ldots )$ , and the top- $k$ trajectories with the highest task similarity will be retrieved and used as fewshot in-context examples, Fsimilar tasks. Fig. 3 shows an example prompt template structure, and a

You are an advanced reasoning agent that can add,edit or remove rulesfrom yourexisting rule set,basedon forming new critiques of past task trajectories.You will be given..

# A

two previous task trials in which

[Task description]...

one successful and one unsuccessful trial. You failed the trial because..

[Task failure reasons].

Here are the two previous trials to compare and critique:

[Failed/Succeeded Trajectories]

successful tasks trials in whichyou

[Task description].

Here are the trials:

[Succeeded Trajetories]

Here are the EXISTING RULES:

[Currently existing insights]

By examining..

and contrasting to  the the successful trials... successful trial,

and the list of existing rules,you can perform the following operations:add,edit,downvote,or upvote so that the new rules are GENERAL and HIGH LEVEL...

critigues of the failed trial... insights of the successful trials...

or proposed way of Thought so they can be used...

to avoid similar failures when ashelpful tips to different encountered with different tasks in the future. questions in the future.

Have an emphasis on...

critiquing how to... tips that help the agent..

perform better Thought and Action.

Follow the below format:

<OPERATION> <RULE NUMBER>:<RULE>

The available operations are:UPVOTE lif the existing rule is strongly relevant for the task), DoWNvOTE (if one existing rule is contradictory or similar/duplicated to other existing rules),EDIT lif any existing rule is not general enough or can be enhanced, rewriteand improveit),ADD [add new rulesthat are very different from existing rules and relevant for other tasks).Each needs to CLOSELY follow their corresponding formatting below:

UPVOTE <EXISTING RULE NUMBER>: <EXISTING RULE>

DOWNVOTE <EXISTING RULE NUMBER>:<EXISTING RULE>

EDIT <EXISTING RULE NUMBER>:<NEW MODIFIED RULE>

ADD <NEW RULE NUMBER>: <NEW RULE>

Do not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE.Each rule should be concise and easy to follow. Any operation can be used MULTiPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation.Below are the operations you do to the above list of EXiSTING RULES:

pseudo-code for this step can be found in Alg. 3. We believe as the list of extracted insights grows, retrieval could be a feasible solution to manage the context window size.

Task You are QA system. Solve a question description: answering task with interleaving..

![](images/45030451c675ef2bcf93b2be85623b2644dd8b476d7fa2082c660fec55c12e95.jpg)  
Figure 2: Insight Extraction Prompt Template. The prompt template ExpeL agents used for insight extraction. The same template is used both for success/fail pairs (A, in yellow) and $L$ -sized successes (B, in green).

Extracted insights:

1. Break down complex queries into simpler,   
2.Consider that the answer might be in the observations already made...

3....

![](images/a64e9210dc6e5f4dbdc8d8b7c0012a96b847d6580ed6f9ee190b7e8db742434c.jpg)

Retrieved inexamples:

Question: Which documentary ... ?

Thought 1:  I need to search..

Action N:Finish[The Saimaa ..]

Obervation N:Answer is CORRECT

Task:Which episode of SpongeBob... ?

Trajectory:

Thought 1: I need to search..

Action 1: Search["The Clasl

Observation 1: “The Clash of Triton,.

Thought 2: The paragraph does

Action H:

Obervation H:

Finish["To SquarePants...]

Answer is CORRECT

![](images/d0607ed101bcef66c4ce5724956e384daa61b19cabb034da7af90bd7e1640753.jpg)  
Figure 3: Task Inference Prompt Template. We illustrate ExpeL’s prompt template during evaluation. The areas with a white background are identical to the base ReAct agent’s inputs. We differ by (purple areas) having additional extracted insights from past experience, and dynamically retrieved successful in-context examples from past experiences based on task similarity.

# 4.4 Transfer Learning

After demonstrating how learning by using experiences from a training set can benefit an LLM agent in solving an unseen task in the same task distribution, we investigate another interesting setting where knowledge accumulated from a source task distribution could be useful for a target task distribution with minimal target task examples for the ExpeL agent. Like most transfer learning settings, we assume that the source and target tasks exhibit common knowledge. Therefore, experiences accumulated from source tasks can benefit the agent in solving a new set of target tasks.

Similar to pretraining on source task and finetuning on target task in transfer learning literature (Zhuang et al. 2020), we propose to use the extracted insights $\hat { \iota }$ from the source task and fewshot examples from the target task to “finetune” the insights so that they are more applicable in the target task. We hypothesize that using target task fewshot examples can better ground the insights into the target task and mitigate hallucinations. An example prompt template to “finetune” extracted insights from a source domain to tailor them

Algorithm 1: ExpeL - Experience Gathering   
Initialize:   
Policy $\mathrm{LLM}_{\mathrm{ReAct}}$ Self-reflection model $\mathrm{LLM}_{\mathrm{reflect}}$ Collection of tasks $\mathcal{T}_{\mathrm{train}}$ Fewshot examples $F_{\mathrm{manual}}$ Experience pool $\mathcal{B}\gets F_{\mathrm{manual}}$ Number of training tasks $N$ Maximum retry number $Z$ Maximum step number $H$ Current task index $n\gets 1$ while task $n\leq N$ do $t_n\gets \mathcal{T}_{\mathrm{train}}[n]$ Reflection $\nu_{n,0}\gets$ "for trial $z = 0$ to $Z$ do $o_0\gets \mathrm{env.dim}(t_n)$ Initialize trajectory $\tau_{n,z}\leftarrow o_0$ for timestep $i = 0$ to $H$ do $a_i\gets \mathrm{LLM}_{\mathrm{ReAct}}(a_i\mid \tau_{n,z},F_{\mathrm{manual}},\nu_{n,z})$ $o_{i + 1},r_{i + 1}$ ,done $\leftarrow$ env_STEP(a_i) $\tau_{n,z}\gets \tau_{n,z}\cup \{(o_i,a_i,o_{i + 1},r_{i + 1})\}$ if done then break end if end for $\mathcal{B}\gets \mathcal{B}\cup \tau_{n,z}$ if done or $z = Z$ then $n\gets n + 1$ break else $\nu_{n,z + 1}\gets \mathrm{concat}(\nu_{n,z} + \mathrm{LLM}_{\mathrm{reflect}}(\tau_{n,z}))$ end if end for end while return $\mathcal{B}$

Algorithm 2: ExpeL - Insight Extraction   
Initialize:  
Experience pool $\mathcal{B}$ (from Alg. 1)  
Insight extraction model LLMinsights  
Set of insights $\hat{\imath}\gets \emptyset$ Divide the successes in $\mathcal{B}$ into $L$ -sized chunks: $C_{success} = \{\{\tau_1^{success},\tau_2^{success},\dots \tau_L^{success}\} ,$ $\{\tau_{L + 1}^{success},\tau_{L + 2}^{success},\dots \tau_{2L}^{success}\} ,\ldots \}$ Construct fail/success tuples of the same tasks in $\mathcal{B}$ : $C_{compare} = \{(\tau_1^{success},\tau_{1,0}^{fail}),(\tau_1^{success},\tau_{1,1}^{fail}),\dots,$ $(\tau_{2}^{success},\tau_{2,0}^{fail}),\dots \}$ for each $c_{compare}$ in $C_{compare}$ do $\hat{\imath}\gets \mathrm{LLM}_{\mathrm{insights}}(c_{compare},\hat{\imath})$ end for  
for each $c_{success}$ in $C_{success}$ do $\hat{\imath}\gets \mathrm{LLM}_{\mathrm{insights}}(c_{success},\hat{\imath})$ end for  
return $\hat{\imath}$

Algorithm 3: ExpeL - Evaluation   
Initialize: ExpeL agent $\mathrm{LLM_{ExpeL}}$ Text Embedder $\mathcal{E}$ Experience pool $\mathcal{B}$ (from Alg. 1) Set of insights $\hat{\iota}$ (from Alg. 2) Collection of evaluation tasks $\mathcal{T}_{\mathrm{evaluation}}$ Number of evaluation tasks $M$ Number of fewshots $k$ Number of successes $S\gets 0$ for task $m = 1$ to $M$ do $t_m\gets \mathcal{T}_{\mathrm{evaluation}}[m]$ $o_0\gets \mathrm{env.dim}(t_m)$ Initialize trajectory $\tau_{m}\leftarrow o_{0}$ $F_{\mathrm{similar~tasks}}\gets \mathrm{Faisss}(t_m,\mathcal{B},\mathcal{E},k)$ for timestep $i = 1$ to $H$ do $a_i\gets \mathrm{LLM_{ExpeL}}(a_i\mid \tau_m,F_{\mathrm{similar~tasks}},\hat{\iota})$ $o_{i + 1},r_{i + 1}$ done $\leftarrow$ env step $(a_i)$ $\tau_{m}\gets \tau_{m}\cup \{(o_{i},a_{i},o_{i + 1},r_{i + 1})\}$ if done then break end if end for if $r_{i + 1} = 1$ then $S\gets S + 1$ end if end for return $\frac{S}{M}$

to a target domain is illustrated in Fig. 4.

# 4.5 ExpeL’s Strengths

In this section, we outline the key strengths of our framework. First and foremost, ExpeL offers inherent interpretability, as both the extracted experiences and successful trajectories are presented in natural language. This design allows users to easily inspect, modify, or remove potentially harmful trajectories/insights — a challenge in finetuned models. Moreover, users can seamlessly add expert insights or trajectories to an ExpeL agent. Additionally, our learning approach is highly accessible; it demands less data, reduces computational resources, and is straightforward to implement. Furthermore, self-improvement methods like Reflexion (Shinn et al. 2023) facilitate intra-task improvements, but ExpeL enables inter-task learning. ExpeL does not rely on retries during deployment, which certain domains require. On the flexibility front, the ExpeL agent boasts a significant level of versatility. It is not restricted to specific language models and complements existing strategies aimed at enhancing LLM agent planning capabilities. Moreover, when applied in conjunction with them, ExpeL might even improve the capabilities of finetuned agents. Another strength lies in continuous improvement. Our method stands to benefit from the ongoing enhancements in foundational models. As an illustration, our experiments show that using gpt-4 to extract insights outperforms gpt-3.5-turbo (refer to Sec. 5.6). Lastly, we introduced a method for trans-

# Knowledge Finetuning

Youare a teacher agent that passes on experience to student agents.You came up with the following rules tohelp you achieve the task of {SourceTask} effectively.The number at the end are the importance you gave to each of the rules.

# RULES:

# [Extracted insights from Source Task]

Now a student agent is trying to solve a similar {Target Task}.

# Some examples of this new task are: [Fixed fewshot examples of Target

Give a concise and easy to follow instructional paragraph based on the RULES for the student agent to solve {Target Task}. Do not state where each sentence is using whichever rule,and make sure the paragraph is VERY CONCISE and EASY TO FOLLOW!

#

# Fewshot Evaluation

The following paragraph is insights a teacher agent provided to you. It is MANDATORY for you to follow these insights as CLOSELY as possible as they will help you perform the {Target Task} tasks efficiently:

# [Finetuned insights]

[Target Task desciption + fewshot]

{Target Task)

![](images/39c9fae556f54423d513affc69f1358c9b505c74171f9ed599ac22ae408b0bbc.jpg)  
Figure 4: Transfer Learning Finetuning Prompt Template. The prompt template used to finetune knowledge from source to target domain. Highlighted in grey should be formatted with concise descriptions of the tasks.

ferring extracted insights across domains using only a small amount of finetuning examples, demonstrating the advantage of our approach in diverse settings with limited data.

# 5 Experiments

# 5.1 Experimental Setup

In line with ReAct (Yao et al. 2023b), the experiments are designed based on four text-based benchmarks: HotpotQA (Yang et al. 2018), a knowledge-intensive dataset that challenges an agent to perform reasoning and question answering using the search tool Wikipedia Docstore API, ALFWorld and WebShop (Shridhar et al. 2021; Yao et al. 2022) that require the agent to perform interactive multi-step decision-making tasks in respectively a household and an online shopping website environments, and FEVER (Thorne et al. 2018), that focuses on fact verification tasks using the same API as HotpotQA which makes it suitable for knowledge transfer (Sec. 5.4). All experiments use four-fold validation, and we report the mean and standard error over the

folds. Following ReAct, for all environments, we use success rate as the evaluation metric: exact matching for HotpotQA and FEVER, completing the task in time for ALF-World, and purchasing the item that matches all attributes for WebShop. Some additional metrics are introduced when the environment offers them: mean reward (calculated using Eq. 1 in Appendix) score $r \in [ 0 , 1 ]$ for WebShop and a score breakdown per task type for ALFWorld.

We use ReAct and Act as main baselines planning LLM agents (Yao et al. 2023b), where Act does not have the reasoning steps like ReAct. All agents, including ExpeL, used gpt-3.5-turbo-0613 when performing actions during evaluation. All text generations were done with temperature 0 and greedy decoding. Imitation learning (IL) results were taken from the ReAct paper (Yao et al. 2023b). More details about the experimental setup can be found in Appendix D.

# 5.2 Main Results

The primary findings of this study are presented in Fig. 5. IL-based method struggles to efficiently perform in Web-Shop and ALFWorld, possibly due to their demand for more substantial prior and reasoning abilities, which conventional trainings from scratch fail to provide. This limitation shows the promise of leveraging knowledge-based language models to address these challenges. The following claims were made based on (1) a deep understanding of each environment; (2) extracted insights and retrievable in-context examples; and (3) statistics (e.g. number of invalid actions per trial) of the runs.

Experiential learning Augmenting agents with abstracted insights and the ability to recall successful trajectories improve performance across all environments compared to baseline agents. When restricting the ExpeL agent to only one mode of learning (insights-only or retrieval-only), HotpotQA and ALFWorld environments demonstrate contrasting quantitative distinctions $( 3 6 \% / 3 1 \%$ and $5 0 \% / 5 5 \%$ for HotpotQA and ALFWorld, respectively). The prominent influence of insights on HotpotQA can be due to its reliance on analysing (Wikipedia results) abilities. This highlights the need for general guidelines across various question types. Conversely, ALFWorld’s task completion, dependent on specific action sets, is better derived from past experiential trajectories. Furthermore, WebShop presents a unique challenge, requiring both website-based reasoning (price comparisons, query reformulation, etc.) and precise execution of actions (searching, clicking, option selection, etc.). Consequently, the performance across these tasks shows a near equilibrium, as reflected in both the success rate and score $( 3 7 \% / 3 8 \%$ and 0.675/0.67 for insights/retrieve-only respectively, see Tab. 5 in Appendix for scores). These observations highlight the synergistic interplay between abstraction and recollection in experiential learning, with ExpeL showing a quantitative advantage over baseline/restricted learning mode agents.

Cross-task learning Another important finding we observe is the comparison with the Reflexion agent (Shinn et al. 2023). ExpeL matches Reflexion’s performance $40 \%$ at R3 vs. $39 \%$ ) for HotpotQA and even outperforms it for

![](images/ceec8db202d0b184369ca740f408c200695e1a1f6df8e1f4a8534844cfb89872.jpg)

![](images/42817501f4df65558309a33a65a9d3b606a180eb10e67e9e0e532c7fe5d21656.jpg)

![](images/47f31ab64a46c59b77c57ff7e64cfa1810606e7d8eb14d643acb9645f16812b2.jpg)  
Figure 5: Main Results. Average task success rates (std. error in gray arrows) across three different domains: HotpotQA, ALFWorld, and WebShop. ReAct and Act are used as baselines. ExpeL consistently outperforms the baselines on all domains, highlighting the importance of learning from experience. Additionally, we compare ExpeL with ExpeL (retrieve-only) and ExpeL (insights-only) to highlight that both insight extraction and task similarity retrieval are essential and synergistic.

ALFWorld $54 \%$ at R3 vs. $59 \%$ ) without repeated attempts. While Reflexion improves results by iteratively refining insights through repeated task execution (R1, R2, R3...), our ExpeL agent leverages cross-task learning by accumulating task experience. However, it is noteworthy that there remains room for improvement in the context of WebShop tasks, approaching the lower side of Reflexion’s success rates.

# 5.3 Agent Behavioral Analysis

In this section, we highlight some observations made by manually inspecting the trajectories of ReAct agents and ExpeL agents, and by pinpointing possible causes of how some unexpected behaviors might have emerged. Please visit the paper’s webpage, https://andrewzh112.github.io/expel, for full trajectory demos illustrating the following findings.

Hypothesis Formulation & Constraints Adaptation After extracting the insights from experiences gathered in the training set, we noticed the agent subsequently gained the ability to reassess its whole trajectory in the last steps and conclusively end the task rather than expressing its ineptitude in providing a solution. This ability was particularly observed in HotpotQA (Fig. 16, 17 in Appendix) where a likely influential insight was stating that the agent should “consider the answer might be in the observations already made”. Therefore the agent would finish by proposing the most probable answer given its past observations rather than concluding with “Unknown” or “Information not available”.

World Model Belief Update We noticed our ExpeL agent updated its beliefs through the insights and over its gained experience. This belief thereby update enables the agent to avoid unnecessary actions and increase efficiency in solving a given task. For example, in ALFWorld, the agent completely changed the priors it had in ReAct on the likely locations of a pan (from drawers/countertops/cabinets to stoveburners). This behavior emerged from the extracted insight

claiming that “when searching for an item” it needs to “consider its nature and its typical usage” (Fig. 18 in Appendix), leading the agent to promptly and accurately find the correct item at the first step while the ReAct agent could not find it in time.

Self-correction Although ReAct was sometimes not able to reassess its situation when attempting to solve a task, ExpeL demonstrated its proficiency in identifying and rectifying missteps. Notably, when incorrectly taking an object in ALFWorld, the agent has shown its ability to put it back and resume the task by searching for the proper object (Fig. 19 in Appendix). This highlights ExpeL’s capacity to recover from errors and stay on course without hallucinating when completing tasks. This behavior is possibly encouraged by the generated insight “reassess the situation and consider alternative actions” if “an attempt does not progress the task”.

# 5.4 Transfer Learning

In this experiment, we use the HotpotQA dataset (Yang et al. 2018) as source tasks and the FEVER dataset (Thorne et al. 2018) as target tasks. Like the HotpotQA dataset, we equip the agent with the ability to navigate on Wikipedia using a Docstore API; therefore, we hypothesize that some of the knowledge obtained from HotpotQA tasks should also be beneficial when transferred to the FEVER tasks. We use gpt-4-0613 for adapting the HotpotQA insights into FEVER insights. We use the same fewshot examples to finetune the insights as the ones that will be used during task execution. We compare our ExpeL Transfer agent’s transfer learning ability with (1) ReAct; (2) Act; and (3) an agent that “finetunes” insights without task demonstrations. Notice that since source and target tasks are inherently different, we do not have an experience pool to retrieve from; thus, the ExpeL Transfer agents use the existing fixed fewshot examples as in-context examples.

Tab. 1 showcases the transfer learning results. Both agents

Table 1: Transfer Results. We transfer insights extracted from HotpotQA to FEVER. Act and ReAct are baseline agents, ExpeL w/o Task Demos does not utilize fewshot examples when altering the insights for the target task.   

<table><tr><td></td><td>FEVER (SR %)</td></tr><tr><td>Act</td><td>58 ± 0.0</td></tr><tr><td>ReAct</td><td>63 ± 0.4</td></tr><tr><td>ExpeL Transfer w/o Task Demos</td><td>65 ± 1.7</td></tr><tr><td>ExpeL Transfer</td><td>70 ± 0.7</td></tr></table>

Table 2: Success Rate on ALFWorld with Reflexion Rounds. ExpeL and Reflexion appear to be synergistic in the ALFWorld environment (Highlight $=$ ExpeL with one attempt). R1-R3 were obtained from failed R0 checkpoints.   

<table><tr><td></td><td>R0</td><td>R1</td><td>R2</td><td>R3</td></tr><tr><td>ReAct+Reflexion</td><td>40.3%</td><td>47.8%</td><td>52.2%</td><td>54.4%</td></tr><tr><td>ExpeL retrieve only</td><td>54.5%</td><td>57.5%</td><td>59.7%</td><td>60.4%</td></tr><tr><td>ExpeL+Reflexion</td><td>59.0%</td><td>60.4%</td><td>63.4%</td><td>64.2%</td></tr></table>

that transferred knowledge from the source domain saw performance gains. Notably, the agent with a few in-context examples had a more significant improvement than the one without, indicating the effectiveness of the proposed “finetuning” method in transfer learning scenarios.

# 5.5 ExpeL with Task Reattempts

While not being the central focus of our study, we present preliminary findings on the effectiveness of incorporating task reattempts into the evaluation phase using ExpeL by resuming the failed checkpoints from R0. The performance of ExpeL combined with Reflexion, alongside two baselines: ReAct/Reflexion and ExpeL without insights (ExpeL retrieve only), is detailed in Table 2. The results demonstrate a notable improvement in the success rate when ExpeL is paired with Reflexion, with the success rate increasing as the number of task reattempts grows.

# 5.6 Ablation Studies

One main component of ExpeL is the agent’s ability to autonomously gather valuable experiences benefiting its own learning. Therefore, we wish to investigate if the number of useful experiences impacts the downstream performance of ExpeL. We designed two different agents to compare our agent with. The first one only has access to initial fewshot examples and extracts insights from them. The second gathers experience using ReAct where the agent has no retries. Thus, the agent will not only get less successful trajectories but will also not have any success/failure comparison pairs during insights extraction. We conducted experiments in the HotpotQA environment and presented the results in Fig. 6. As we can see, the agent that extracts insights from the existing fewshots has no advantage compared to the Re-Act agent, illustrating that experience is essential for ExpeL to learn from. This was reflected in a significantly better performance for the two other agents having access to more experience. Furthermore, the ExpeL agent with access to a di-

verse set of experiences (failure and success pairs obtained using Reflexion) performs better than the agent using only ReAct during experience gathering.

![](images/e4c9d245eca854ad4488d8a4f286c16c49ea1fabfcd24c586de737dc9e1b20bb.jpg)  
Effects of experience pool size on performance

Figure 6: Effects of Experience on Performance. We highlight the correlation between the number of diverse experience samples and the final performance. Concretely, we compare ExpeL with (1) ReAct, (2) ExpeL that only has access to fewshot examples, and (3) ExpeL that only uses ReAct during the experience gathering step. It is evident that extra autonomously collected experiences are essential to ExpeL’s success and that diversity of success/failure data gathered using Reflexion was superior to using ReAct only.

Next, we will scrutinize the efficacy of the insight extraction step of ExpeL. Since insights had the most significant impact on the HotpotQA environment (Fig. 5), we performed the ablations on insights in this environment. We use three dimensions to ablate the design choices for insight extraction by creating the following variants of ExpeL agents: (1) human-crafted insights (Fig. 12 in Appendix), which were manually engineered by carefully studying the agent’s mistakes during the experience gathering step; (2) adding reflections $\nu$ into the insights construction step in addition to using fail/success pairs and lists of successes; (3) using gpt-3.5-turbo-0613 as the $\mathrm { L L M } _ { \mathrm { i n s i g h t s } }$ . Results in Tab. 3 show several significant findings: (1) learned insights by the agent are more advantageous than handcrafted ones; (2) using reflections in addition to success/- failure pairs and lists of successes is disadvantageous, possibly due to reflections sometimes outputting hallucinations, therefore misleading the insight extraction stage; and (3) a better LLM is more advantageous at improving ExpeL’s performance, suggesting our agent will enjoy free performance boosts with the ever-improving nature of base foundation models.

Lastly, we investigated the design choice of using task similarity as the ranking score for retrieving successful incontext examples in ALFWorld. In particular, we use (1) reason similarity by retrieving top- $k$ trajectories with the most similar reasoning step as the latest reasoning step in the current trajectory, and (2) randomly sampling successful trajectories from the experience pool. We clearly observe in Tab. 3 that retrieving with task similarity (ExpeL) performs the best. Reason similarity is still advantageous but slightly drops in performance, possibly due to dynamically changing fewshots during a single trajectory, causing instabilities. Lastly, random sampling has a significant drop in performance, suggesting that our design choice of selecting the most pertinent in-context example is advantageous.

Table 3: Ablations Results. Upper: Ablations on insight extraction. Hand-crafted insights enjoyed a performance boost over ReAct but were less effective than LLM-generated ones. Furthermore, adding reflections to the insight-generating process hurt performance. Lastly, better LLM base models give better insights. Lower: Ablations on in-context examples selection strategy. Randomly selected baseline has a significant drop in performance while ranking using reason similarity also has a noticeable dip.   

<table><tr><td></td><td>HotpotQA (SR %)</td></tr><tr><td>ReAct</td><td>28.0 ± 1.4</td></tr><tr><td>Hand-crafted insights</td><td>32.0 ± 1.1</td></tr><tr><td>Insights with reflections</td><td>29.0 ± 0.4</td></tr><tr><td>gpt-3.5-turbo insights</td><td>32.0 ± 0.4</td></tr><tr><td>ExpeL (ours)</td><td>39.0 ± 1.7</td></tr><tr><td></td><td>ALFWorld (SR %)</td></tr><tr><td>ReAct</td><td>40.0 ± 0.3</td></tr><tr><td>Reasoning similarity</td><td>48.5 ± 2.1</td></tr><tr><td>Random sampled</td><td>42.5 ± 0.8</td></tr><tr><td>ExpeL (ours)</td><td>59.0 ± 0.3</td></tr></table>

# 6 Conclusion and Limitations

Limitations In this work, we investigated tasks with textual observation, which is limiting in real-world scenarios. Thus, incorporating image observations will make our method more generally applicable. Using Vision-Language Models or captioning models to supplement the LLM to enable image observations could be an interesting new avenue of research. Additionally, we investigated the efficacy of our method by using closed-source API LLMs, which can be off-limits in some applications. Exploring LLM agents using open-source LLMs should be another promising future work (Zeng et al. 2023). Furthermore, since our extracted insights do not exceed the current LLM’s token limit, we can fit them into the agent’s context window. However, extra retrieval steps for insights might be needed for truly lifelong learning agents to ensure a manageable context window size. Lastly, unlike reinforcement learning methods, prompting techniques lack theoretical underpinnings that could potentially impact the efficiency of the resulting policies. Future research should explore the integration of these approaches to yield more effective and optimal solutions.

In summary, we introduced ExpeL, a novel learning LLM agent that autonomously gathers experience from a set of training tasks to improve its abilities in solving evaluation tasks without access to model parameters. We demonstrated its learning abilities by showing its performance gain compared to vanilla ReAct and Act agents. Furthermore, we investigated a transfer learning scenario where extracting insights from a set of source tasks can benefit the ExpeL agent in solving a target task. Lastly, we presented several unexpected emerged abilities our agent developed at the end of its training. We believe that autonomously learning from experience is essential for developing human-like intelligent agents, and our ExpeL agent is a step toward that goal.

# Acknowledgement

This work is supported in part by the National Key R&D Program of China (2022ZD0114900), the National Natural Science Foundation of China under Grants 62022048, U2336214, and 62332019, and the Guoqiang Institute of Tsinghua University.

# References

Anthropic. 2023. Introducing Claude.

Boiko, D. A.; MacKnight, R.; and Gomes, G. 2023. Emergent Autonomous Scientific Research Capabilities of Large Language Models. arXiv preprint.

Bran, A. M.; Cox, S.; White, A. D.; and Schwaller, P. 2023. ChemCrow: Augmenting Large-Language Models with Chemistry Tools. arXiv preprint.

Brown, T.; Mann, B.; Ryder, N.; Subbiah, M.; Kaplan, J. D.; Dhariwal, P.; Neelakantan, A.; Shyam, P.; Sastry, G.; Askell, A.; et al. 2020. Language Models are Few-Shot Learners. NeurIPS.

Chase, H. 2023. Langchain.

Chowdhery, A.; Narang, S.; Devlin, J.; Bosma, M.; Mishra, G.; Roberts, A.; Barham, P.; Chung, H. W.; Sutton, C.; Gehrmann, S.; Schuh, P.; Shi, K.; Tsvyashchenko, S.; Maynez, J.; Rao, A.; Barnes, P.; Tay, Y.; Shazeer, N.; Prabhakaran, V.; Reif, E.; Du, N.; Hutchinson, B.; Pope, R.; Bradbury, J.; Austin, J.; Isard, M.; Gur-Ari, G.; Yin, P.; Duke, T.; Levskaya, A.; Ghemawat, S.; Dev, S.; Michalewski, H.; Garcia, X.; Misra, V.; Robinson, K.; Fedus, L.; Zhou, D.; Ippolito, D.; Luan, D.; Lim, H.; Zoph, B.; Spiridonov, A.; Sepassi, R.; Dohan, D.; Agrawal, S.; Omernick, M.; Dai, A. M.; Pillai, T. S.; Pellat, M.; Lewkowycz, A.; Moreira, E.; Child, R.; Polozov, O.; Lee, K.; Zhou, Z.; Wang, X.; Saeta, B.; Diaz, M.; Firat, O.; Catasta, M.; Wei, J.; Meier-Hellstern, K.; Eck, D.; Dean, J.; Petrov, S.; and Fiedel, N. 2023. PaLM: Scaling Language Modeling with Pathways. JMLR.

Chung, H. W.; Hou, L.; Longpre, S.; Zoph, B.; Tay, Y.; Fedus, W.; Li, E.; Wang, X.; Dehghani, M.; Brahma, S.; et al. 2022. Scaling Instruction-Finetuned Language Models. arXiv preprint.

Du, M.; He, F.; Zou, N.; Tao, D.; and Hu, X. 2022. Shortcut Learning of Large Language Models in Natural Language Understanding: A Survey. arXiv preprint.

Gong, R.; Huang, Q.; Ma, X.; Vo, H.; Durante, Z.; Noda, Y.; Zheng, Z.; Zhu, S.-C.; Terzopoulos, D.; Fei-Fei, L.; et al. 2023. MindAgent: Emergent Gaming Interaction. arXiv preprint.

Gur, I.; Furuta, H.; Huang, A.; Safdari, M.; Matsuo, Y.; Eck, D.; and Faust, A. 2023. A Real-World WebAgent with Planning, Long Context Understanding, and Program Synthesis. arXiv preprint.

Ha, H.; Florence, P.; and Song, S. 2023. Scaling Up and Distilling Down: Language-Guided Robot Skill Acquisition. In CoRL. PMLR.

Hao, S.; Gu, Y.; Ma, H.; Hong, J. J.; Wang, Z.; Wang, D. Z.; and Hu, Z. 2023. Reasoning with Language Model is Planning with World Model. arXiv preprint.

Huang, W.; Abbeel, P.; Pathak, D.; and Mordatch, I. 2022. Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents. In ICML. PMLR.   
Humphreys, P.; Guez, A.; Tieleman, O.; Sifre, L.; Weber, T.; and Lillicrap, T. 2022. Large-scale Retrieval for Reinforcement Learning. NeurIPS.   
Johnson, J.; Douze, M.; and Jegou, H. 2019. Billion-scale´ Similarity Search with GPUs. IEEE Transactions on Big Data.   
Kahneman, D. 2011. Thinking, Fast and Slow. Farrar, Straus and Giroux.   
Kojima, T.; Gu, S. S.; Reid, M.; Matsuo, Y.; and Iwasawa, Y. 2022. Large Language Models are Zero-Shot Reasoners. NeurIPS.   
Li, H.; Su, Y.; Cai, D.; Wang, Y.; and Liu, L. 2022. A Survey on Retrieval-Augmented Text Generation. arXiv preprint.   
Lin, B. Y.; Fu, Y.; Yang, K.; Ammanabrolu, P.; Brahman, F.; Huang, S.; Bhagavatula, C.; Choi, Y.; and Ren, X. 2023a. SwiftSage: A Generative Agent with Fast and Slow Thinking for Complex Interactive Tasks. NeurIPS.   
Lin, K.; Agia, C.; Migimatsu, T.; Pavone, M.; and Bohg, J. 2023b. Text2Motion: From Natural Language Instructions to Feasible Plans. Autonomous Robots.   
Lin, L.-J. 1992. Self-Improving Reactive Agents Based on Reinforcement Learning, Planning and Teaching. Machine learning.   
Liu, J.; Shen, D.; Zhang, Y.; Dolan, B.; Carin, L.; and Chen, W. 2022. What Makes Good In-Context Examples for GPT-3? In DeeLIO. Association for Computational Linguistics.   
Liu, P.; Yuan, W.; Fu, J.; Jiang, Z.; Hayashi, H.; and Neubig, G. 2023a. Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing. ACM Computing Surveys.   
Liu, X.; Yu, H.; Zhang, H.; Xu, Y.; Lei, X.; Lai, H.; Gu, Y.; Ding, H.; Men, K.; Yang, K.; et al. 2023b. AgentBench: Evaluating LLMs as Agents. arXiv preprint.   
Liu, Z.; Bahety, A.; and Song, S. 2023. REFLECT: Summarizing Robot Experiences for Failure Explanation and Correction. In CoRL. PMLR.   
Maas; Carey; Wheeler; Saatchi; Billington; and Shamash. 2023. To Infinity and Beyond: SHOW-1 and Showrunner Agents in Multi-Agent Simulations. arXiv preprint.   
Mirchandani, S.; Xia, F.; Florence, P.; Ichter, B.; Driess, D.; Arenas, M. G.; Rao, K.; Sadigh, D.; and Zeng, A. 2023. Large Language Models as General Pattern Machines. In CoRL. PMLR.   
Mu, Y.; Zhang, Q.; Hu, M.; Wang, W.; Ding, M.; Jin, J.; Wang, B.; Dai, J.; Qiao, Y.; and Luo, P. 2023. EmbodiedGPT: Vision-Language Pre-Training via Embodied Chain of Thought. NeurIPS.   
Nakajima, Y. 2023. BabyAGI. https://github.com/ yoheinakajima/babyagi.   
Nakano, R.; Hilton, J.; Balaji, S. A.; Wu, J.; Ouyang, L.; Kim, C.; Hesse, C.; Jain, S.; Kosaraju, V.; Saunders, W.; Jiang, X.; Cobbe, K.; Eloundou, T.; Krueger, G.; Button, K.;

Knight, M.; Chess, B.; and Schulman, J. 2021. WebGPT: Browser-Assisted Question-Answering with Human Feedback. arXiv preprint.   
OpenAI. 2023. GPT-4 Technical Report.   
Ouyang, L.; Wu, J.; Jiang, X.; Almeida, D.; Wainwright, C.; Mishkin, P.; Zhang, C.; Agarwal, S.; Slama, K.; Ray, A.; Schulman, J.; Hilton, J.; Kelton, F.; Miller, L.; Simens, M.; Askell, A.; Welinder, P.; Christiano, P. F.; Leike, J.; and Lowe, R. 2022. Training Language Models to Follow Instructions with Human Feedback. In NeurIPS.   
Park, J. S.; O’Brien, J.; Cai, C. J.; Morris, M. R.; Liang, P.; and Bernstein, M. S. 2023. Generative Agents: Interactive Simulacra of Human Behavior. In ACM Symposium on User Interface Software and Technology.   
Petroni, F.; Rocktaschel, T.; Riedel, S.; Lewis, P.; Bakhtin, ¨ A.; Wu, Y.; and Miller, A. 2019. Language Models as Knowledge Bases? In EMNLP-IJCNLP. Association for Computational Linguistics.   
Qian, C.; Cong, X.; Yang, C.; Chen, W.; Su, Y.; Xu, J.; Liu, Z.; and Sun, M. 2023. Communicative Agents for Software Development. arXiv:2307.07924.   
Rubin, O.; Herzig, J.; and Berant, J. 2022. Learning To Retrieve Prompts for In-Context Learning. In NAACL. Association for Computational Linguistics.   
Schaul, T.; Quan, J.; Antonoglou, I.; and Silver, D. 2015. Prioritized Experience Replay. In ICLR.   
Shaw, P.; Joshi, M.; Cohan, J.; Berant, J.; Pasupat, P.; Hu, H.; Khandelwal, U.; Lee, K.; and Toutanova, K. 2023. From Pixels to UI Actions: Learning to Follow Instructions via Graphical User Interfaces. NeurIPS.   
Shinn, N.; Cassano, F.; Gopinath, A.; Narasimhan, K. R.; and Yao, S. 2023. Reflexion: Language Agents with Verbal Reinforcement Learning. In NeurIPS.   
Shridhar, M.; Yuan, X.; Cotˆ e, M.-A.; Bisk, Y.; Trischler, A.; ´ and Hausknecht, M. 2021. ALFWorld: Aligning Text and Embodied Environments for Interactive Learning. In ICLR.   
Significant-Gravitas. 2023. AutoGPT. https://github.com/ Significant-Gravitas/Auto-GPT.   
Song, K.; Tan, X.; Qin, T.; Lu, J.; and Liu, T.-Y. 2020. MP-Net: Masked and Permuted Pre-training for Language Understanding. NeurIPS.   
Sumers, T. R.; Yao, S.; Narasimhan, K.; and Griffiths, T. L. 2023. Cognitive Architectures for Language Agents. arXiv preprint.   
Sun, H.; Zhuang, Y.; Kong, L.; Dai, B.; and Zhang, C. 2023. AdaPlanner: Adaptive Planning from Feedback with Language Models. NeurIPS.   
Sutton, R. S.; and Barto, A. G. 2018. Reinforcement Learning: An Introduction. MIT press.   
Taori, R.; Gulrajani, I.; Zhang, T.; Dubois, Y.; Li, X.; Guestrin, C.; Liang, P.; and Hashimoto, T. B. 2023. Stanford Alpaca: An Instruction-Following LLaMA Model. https: //github.com/tatsu-lab/stanford alpaca.   
Thoppilan, R.; De Freitas, D.; Hall, J.; Shazeer, N.; Kulshreshtha, A.; Cheng, H.-T.; Jin, A.; Bos, T.; Baker, L.; Du,

Y.; et al. 2022. LaMDA: Language Models for Dialog Applications. arXiv preprint.   
Thorne, J.; Vlachos, A.; Christodoulopoulos, C.; and Mittal, A. 2018. FEVER: a Large-scale Dataset for Fact Extraction and VERification. In NAACL.   
Touvron, H.; Lavril, T.; Izacard, G.; Martinet, X.; Lachaux, M.-A.; Lacroix, T.; Roziere, B.; Goyal, N.; Hambro, E.; ` Azhar, F.; et al. 2023a. LLaMA: Open and Efficient Foundation Language Models. arXiv preprint.   
Touvron, H.; Martin, L.; Stone, K.; Albert, P.; Almahairi, A.; Babaei, Y.; Bashlykov, N.; Batra, S.; Bhargava, P.; Bhosale, S.; et al. 2023b. Llama 2: Open Foundation and Fine-Tuned Chat Models. arXiv preprint.   
Tworkowski, S.; Staniszewski, K.; Pacek, M.; Wu, Y.; Michalewski, H.; and Miłos, P. 2023. Focused Transformer: ´ Contrastive Training for Context Scaling. In NeurIPS.   
Wang, G.; Xie, Y.; Jiang, Y.; Mandlekar, A.; Xiao, C.; Zhu, Y.; Fan, L.; and Anandkumar, A. 2023a. Voyager: An Openended Embodied Agent with Large Language Models. arXiv preprint.   
Wang, L.; Ma, C.; Feng, X.; Zhang, Z.; Yang, H.; Zhang, J.; Chen, Z.; Tang, J.; Chen, X.; Lin, Y.; et al. 2023b. A Survey on Large Language Model Based Autonomous Agents. arXiv preprint.   
Wang, L.; Yang, N.; and Wei, F. 2023. Learning to Retrieve In-Context Examples for Large Language Models. arXiv preprint.   
Wang, S.; Liu, C.; Zheng, Z.; Qi, S.; Chen, S.; Yang, Q.; Zhao, A.; Wang, C.; Song, S.; and Huang, G. 2023c. Avalon’s Game of Thoughts: Battle Against Deception through Recursive Contemplation. arXiv preprint.   
Watkins, C. J.; and Dayan, P. 1992. Q-learning. Machine learning.   
Wei, J.; Bosma, M.; Zhao, V.; Guu, K.; Yu, A. W.; Lester, B.; Du, N.; Dai, A. M.; and Le, Q. V. 2022a. Finetuned Language Models are Zero-Shot Learners. In ICLR.   
Wei, J.; Wang, X.; Schuurmans, D.; Bosma, M.; Xia, F.; Chi, E.; Le, Q. V.; Zhou, D.; et al. 2022b. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. NeurIPS.   
Wu, J.; Antonova, R.; Kan, A.; Lepert, M.; Zeng, A.; Song, S.; Bohg, J.; Rusinkiewicz, S.; and Funkhouser, T. 2023. TidyBot: Personalized Robot Assistance with Large Language Models. Autonomous Robots.   
Xi, Z.; Chen, W.; Guo, X.; He, W.; Ding, Y.; Hong, B.; Zhang, M.; Wang, J.; Jin, S.; Zhou, E.; et al. 2023. The Rise and Potential of Large Language Model Based Agents: A Survey. arXiv preprint.   
Yang, S.; Nachum, O.; Du, Y.; Wei, J.; Abbeel, P.; and Schuurmans, D. 2023a. Foundation Models for Decision Making: Problems, Methods, and Opportunities. arXiv preprint.   
Yang, Z.; Li, L.; Wang, J.; Lin, K.; Azarnasab, E.; Ahmed, F.; Liu, Z.; Liu, C.; Zeng, M.; and Wang, L. 2023b. MM-REACT: Prompting ChatGPT for Multimodal Reasoning and Action. arXiv preprint.

Yang, Z.; Qi, P.; Zhang, S.; Bengio, Y.; Cohen, W.; Salakhutdinov, R.; and Manning, C. D. 2018. HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. In EMNLP. Association for Computational Linguistics.   
Yao, S.; Chen, H.; Yang, J.; and Narasimhan, K. 2022. Web-Shop: Towards Scalable Real-World Web Interaction with Grounded Language Agents. In NeurIPS.   
Yao, S.; Yu, D.; Zhao, J.; Shafran, I.; Griffiths, T. L.; Cao, Y.; and Narasimhan, K. 2023a. Tree of Thoughts: Deliberate Problem Solving with Large Language Models. NeurIPS.   
Yao, S.; Zhao, J.; Yu, D.; Du, N.; Shafran, I.; Narasimhan, K.; and Cao, Y. 2023b. ReAct: Synergizing Reasoning and Acting in Language Models. In ICLR.   
Yao, W.; Heinecke, S.; Niebles, J. C.; Liu, Z.; Feng, Y.; Xue, L.; Murthy, R.; Chen, Z.; Zhang, J.; Arpit, D.; Xu, R.; Mui, P.; Wang, H.; Xiong, C.; and Savarese, S. 2023c. Retroformer: Retrospective Large Language Agents with Policy Gradient Optimization.   
Yue, Y.; Kang, B.; Ma, X.; Huang, G.; Song, S.; and Yan, S. 2023. Offline Prioritized Experience Replay. arXiv preprint.   
Zeng, A.; Liu, M.; Lu, R.; Wang, B.; Liu, X.; Dong, Y.; and Tang, J. 2023. AgentTuning: Enabling Generalized Agent Abilities for LLMs. arXiv preprint.   
Zhang, Z.; Zhang, A.; Li, M.; and Smola, A. 2023. Automatic Chain of Thought Prompting in Large Language Models. In ICLR.   
Zhao, A.; Zhu, E.; Lu, R.; Lin, M.; Liu, Y.-J.; and Huang, G. 2023a. Augmenting Unsupervised Reinforcement Learning with Self-Reference. arXiv preprint.   
Zhao, W. X.; Zhou, K.; Li, J.; Tang, T.; Wang, X.; Hou, Y.; Min, Y.; Zhang, B.; Zhang, J.; Dong, Z.; et al. 2023b. A Survey of Large Language Models. arXiv preprint.   
Zhuang, F.; Qi, Z.; Duan, K.; Xi, D.; Zhu, Y.; Zhu, H.; Xiong, H.; and He, Q. 2020. A Comprehensive Survey on Transfer Learning. Proceedings of the IEEE.   
Zitkovich, B.; Yu, T.; Xu, S.; Xu, P.; Xiao, T.; Xia, F.; Wu, J.; Wohlhart, P.; Welker, S.; Wahid, A.; et al. 2023. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. In CoRL. PMLR.

# Appendix

# A Detailed Related Works

# A.1 Prompt-based Learning

Prompt-based learning is a paradigm where the language model that originally outputs the label y from context c improves on the label prediction task with a modified context cˆ (Liu et al. 2023a). This framework is compelling as it enables the usage of pre-trained LLMs trained on vast text volumes. Furthermore, a new prompting function supports fewshot or zero-shot learning, thereby adapting swiftly to tasks with minimal or no labeled data. Specifically, tuning-free prompting directly produces answers using a pre-trained language model’s prompt without altering its parameters. This method can be enhanced with answered prompts, a strategy termed in-context learning (Brown et al. 2020). Examples include LAMA (Petroni et al. 2019), GPT-3 (Brown et al. 2020) and CoT (Wei et al. 2022b). Its benefits include efficiency, no parameter updates, avoidance of catastrophic forgetting, and zero/fewshot setting applicability. However, it demands intricate prompt engineering and domain knowledge expertise to increase accuracy. Works like AutoPrompt and Zero-shot-CoT (Kojima et al. 2022; Zhang et al. 2023) alleviate the burden on the engineer by automatically generating reasoning chains for NLP reasoning tasks. Likewise, ExpeL agent automatically gathers experiences in sequential decision-making tasks, generates its own insights, and uses these insights alongside successful in-context examples to inform its decisions, taking the burdens away from heavy manual prompt engineering and the requirement of expert domain knowledge.

# A.2 Retrieval Augmented Generation

Retrieval augmented generation has gained popularity, which is helpful to reduce hallucination and give LLMs access to internal databases (Li et al. 2022). Several works in the field of NLP demonstrated the efficacy of retrieving in-context examples (Wang, Yang, and Wei 2023; Rubin, Herzig, and Berant 2022) from a database of gold demonstrations. On the contrary, our work explores LLM agents retrieving from their own generated experiences, which lessens the burden of the user’s engineering efforts and domain expertise.

# A.3 LLM Agents

Research involving using LLMs as the “brain” of an agent has surged in recent years. LLM agents have been instantiated in many areas such as robotics (Ha, Florence, and Song 2023; Zitkovich et al. 2023; Mu et al. 2023; Mirchandani et al. 2023; Wu et al. 2023), natural sciences (Bran et al. 2023; Boiko, MacKnight, and Gomes 2023) and automated workflows (Yang et al. 2023b; Gur et al. 2023). Most of these works leverage LLMs’ strong common sense knowledge to achieve downstream tasks in a zero or fewshot manner to keep the LLM’s strong world knowledge priors. Our ExpeL agent also leverages the powerful world knowledge of LLMs. Concretely, we use LLMs during gathering experience, extracting insights, and downstream execution steps.

Planning LLMs have demonstrated the ability to plan in embodied environments in a zero-shot manner (Huang et al. 2022). However, many works show that LLMs’ planning ability can be further enhanced by improving their reasoning capabilities (Yao et al. 2023a; Wei et al. 2022b). The ReAct agent (Yao et al. 2023b) demonstrates a combination of reasoning and acting. This approach has not only been proven to be superior to agents that only output actions in various scenarios, but also provides insight into what the agent is thinking while acting. Because of its simplicity and effectiveness, we used ReAct as our base planning algorithm.

Self-improvement A class of methods that leverages LLMs’ ability to self-reflect based on feedback from the environment has shown their superiority compared to algorithms that do not have an awareness of doing the task a second time (Shinn et al. 2023; Liu, Bahety, and Song 2023). In particular, the Reflexion agent (Shinn et al. 2023) provides a verbal hypothesis on why a task failed based on the failed trajectory/environment feedback and improved if given a second chance. However, self-reflecting methods assume the tasks are repeatable, and environment feedback is available at test time. Furthermore, selfreflection methods are stateless and cannot learn cross-task insights. Instead, our approach leverages the strengths of Reflexion and uses it to gather more failed/successful trajectories to extract insights from them and perform better at test time. Works like Voyager (Wang et al. 2023a) explored skill learning in specific environments like Minecraft.

Memory Mechanisms Agents with persistent long-term memory have demonstrated exciting results in multi-agent settings (Park et al. 2023; Maas et al. 2023; Qian et al. 2023). These works usually have multiple instantiations of generative agents that interact with each other and simulate human societies or fictional settings. In generative agents (Park et al. 2023), agents have a memory mechanism where they can retrieve information based on recency, relevance, and importance, much like how humans sometimes refer to and associate with different memories during their day. These lines of work usually are open-ended, while ExpeL agents are task-solving. Like generative agents, our work also uses memory: successful in-context examples and extracted insights as condensed memory which were both gathered from the agent’s own experience.

# A.4 Reinforcement Learning

Our agent gathers experience autonomously, reminiscent of online reinforcement learning methods (Sutton and Barto 2018). Especially, our method uses off-policy learning (Watkins and Dayan 1992), where the policy uses Reflexion during experience gathering and performs policy improvement via insight extraction and retrieval of similar tasks as in-context examples. Specifically, the retrieval step is similar to experience replay (Lin 1992), where research has been done to select which examples to give the agent for training (Schaul et al. 2015; Yue et al. 2023). However, unlike these existing methods, ExpeL doesn’t require access to model parameters, the design of complicated reward or loss functions, or a large number of environment interactions.

# B Broader Impacts

Our research focuses on LLM agents. If these autonomous programs are given internet access, there’s a risk they might cause unexpected harm. However, techniques such as RLHF could potentially mitigate these adverse effects (Nakano et al. 2021; Ouyang et al. 2022).

# C Computational Resources

All experiments were performed on a desktop: Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz with 16 cores, 64GB RAM, and a single NVIDIA GeForce RTX 2080 Ti.

# D Environment Details

# D.1 Evaluation Task Set

We employ four-fold validation for all experiments. We train on one half of the dataset and evaluate on the other half, and vice versa. All results include the mean and standard error of the results across the folds. For HotpotQA, we assess performance using 100 validation tasks from the distractor dev split of the HotPotQA dataset (Yang et al. 2018), which were also used by ReAct and Reflexion. In the case of ALFWorld (Shridhar et al. 2021), we utilized the 134 solvable tasks that ReAct and Reflexion used. Similarly, for WebShop tasks, we evaluated using the same 100 tasks used by ReAct and Reflexion.

# D.2 Prompts/Fewshot Examples

We used the same fewshot examples/prompts from ReAct and Reflexion (Yao et al. 2023b; Shinn et al. 2023) during appropriate stages. For WebShop, we added one additional fewshot to make the environment have two fewshot examples. We show our prompt templates in Appendix F and will make the code publicly available.

# D.3 WebShop Environment Specific Detail

We slightly modified WebShop environment found at https://github.com/princeton-nlp/WebShop. Our goal was to ensure each experiment instantiation was deterministic. In the original version, item prices and price constraints in instructions were generated by sampling from a uniform range. Instead, we used the average value. While this should produce a result similar to the original implementation on average, it ensures consistency across different instantiations for easier reproducibility. Lastly, we extend the items per page from 3 to 10 since recent LLMs saw an increase in context window size that can accommodate more observations.

# D.4 WebShop Reward Function

Another metric introduced in the WebShop (Yao et al. 2022) is their reward function, which converts the similarity between expected product attributes and the attributes of the purchased product into a value ranging from 0 to 1:

$$
r = \frac {\left| U _ {\text {a t t}} \cap Y _ {\text {a t t}} \right| + \left| U _ {\text {o p t}} \cap Y _ {\text {o p t}} \right| + \mathbb {I} \left[ y _ {\text {p r i c e}} \leq u _ {\text {p r i c e}} \right]}{\left| U _ {\text {a t t}} \right| + \left| U _ {\text {o p t}} \right| + 1} \cdot r _ {\text {t y p e}}, \tag {1}
$$

where,

$$
r _ {t y p e} = \left\{ \begin{array}{l l} 0, & \text {i f T e x t M a t c h = 0} \\ 0. 1, & \text {i f T e x t M a t c h <   0 . 1} \\ 0. 5, & \text {i f T e x t M a t c h \leq 0 . 2 a n d q u e r y n o t m a t c h a n d c a t e g o r y n o t m a t c h} \\ 1, & \text {o t h e r w i s e .} \end{array} \right. \tag {2}
$$

Since a single query could yield multiple appropriate items, WebShop utilizes a matching reward for assessment. The term “TextMatch” denotes the textual overlap of pronouns, nouns, and proper nouns between the selected product’s title and the target product’s title (Liu et al. 2023b).

# D.5 Base Language Model

All experiments were conducted using Langchain (Chase 2023), making API calls to the OpenAI API. For Reflexion during experience gathering, we used gpt-3.5-turbo-0613 and gpt-3.5-turbo-16k-0613 when it is over the context window limit. For insight extraction, we used gpt-4-0613. We used gpt-3.5-turbo-0613 for all evaluation stage agents. All experiments were conducted from July 10, 2023, to August 10, 2023.

E Environment, Agent, Retrieval Parameters   
Table 4: Environment, Retrieval and Agent Parameters.   

<table><tr><td colspan="2">Retrieval Parameters</td></tr><tr><td>Vectorstore</td><td>Faisss</td></tr><tr><td>Retriever type</td><td>kNN</td></tr><tr><td>Embedder</td><td>all-mpnet-base-v2</td></tr><tr><td colspan="2">Agent Hyperparameters</td></tr><tr><td>Max Reflection Retries</td><td>3</td></tr><tr><td>Reflection LLM</td><td>gpt-3.5-turbo-0613</td></tr><tr><td>Policy LLM</td><td>gpt-3.5-turbo-0613</td></tr><tr><td>Insight Extraction LLM</td><td>gpt-4-0613</td></tr><tr><td>Decoding Temperature</td><td>0</td></tr><tr><td>Decoding Strategy</td><td>greedy</td></tr><tr><td colspan="2">HotpotQA-specific Parameters</td></tr><tr><td>Number of Success Examples in Insight Extraction L</td><td>8</td></tr><tr><td>Max Number of Environment Steps H</td><td>7</td></tr><tr><td>Max Number of Fewshot Examples k</td><td>6</td></tr><tr><td>Max Number of Reflection Fewshot Examples kreflections</td><td>2</td></tr><tr><td colspan="2">WebShop-specific Parameters</td></tr><tr><td>Number of Success Examples in Insight Extraction L</td><td>4</td></tr><tr><td>Max Number of Environment Steps H</td><td>15</td></tr><tr><td>Max Number of Fewshot Examples k</td><td>2</td></tr><tr><td>Max Number of Reflection Fewshot Examples kreflections</td><td>2</td></tr><tr><td>Searched items per page</td><td>10</td></tr><tr><td colspan="2">ALFWorld-specific Parameters</td></tr><tr><td>Number of Success Examples in Insight Extraction L</td><td>8</td></tr><tr><td>Max Number of Environment Steps H</td><td>20</td></tr><tr><td>Max Number of Fewshot Examples k</td><td>2</td></tr><tr><td>Max Number of Reflection Fewshot Examples kreflections</td><td>2</td></tr><tr><td colspan="2">FEVER-specific Parameters</td></tr><tr><td>Max Number of Environment Steps H</td><td>7</td></tr><tr><td>Max Number of Fewshot Examples k</td><td>3</td></tr></table>

# F Prompt Templates

# F.1 Policy/Actor Prompt Templates

Policy/actor prompt templates were taken from ReAct (Yao et al. 2023b) (https://github.com/ysymyth/ReAct) with minimal alterations to fit extracted insights for our ExpeL agents.

You are QA system. Solve a question answering task with interleaving Thought, Action, Observation steps.Thought can reason about the current situation,and Action can be three types:

[1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists.If not, it will return some similar entities to search.   
[2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.   
(3) Finish[answer], which returns the answer and finishes the task.

The following are some experience you gather on a similar task of question answering using Wikipedia APl. Use these as references to help you perform this task:

[Extracted insights]

You may take maximum of 7 steps.

Here are some examples: [Retrieved successful trajectories]

[END OF EXAMPLES]

Now it's your turn!

```txt
Question: ... ... 
```

Action H:Finish[...]

Observation H: Answer is ...

Figure 7: ExpeL HotpotQA Acting Template.

ALFWorld {

You are Alfred. Follow the syntax of the examples closely when taking actions.

The following are some experience you gather on a similar task of completing a household task by interacting in a household environment. Use these as references to help you perform this task:

[Extracted insights]

You may take maximum of 20 steps.

Here are some examples: [Retrieved successful trajectories]

[END OF EXAMPLES]

Now it's your turn!

You are in the middle of a room. Looking quickly around you, you see a..

```txt
Your task is to: ...  
> ...  
> ...  
Task is ... 
```

You are Webshop. Follow the syntax of the examples closely when taking actions.

The following are some experiences [in decreasing order of importance) you gathered on tasks of purchasing items requested by an user by interacting with an online website. Use these experiences as useful references to help you perform better on this task:

# [Extracted insights]

You may take maximum of 15 steps.

Here are two examples:

# [Retrieved successful trajectories]

[END OF EXAMPLES]

Now it's your turn!

```txt
Instruction: ...  
... : ...  
Action: Click[Buy Now]  
Observation: Your score (min 0.0, max 1.0): ... 
```

Figure 9: ExpeL WebShop Acting Template.

![](images/ba615909d563f4022f7614a6a30e3ca65626ad62724a2a837473803cbc851228.jpg)  
Figure 10: ExpeL FEVER Acting Template.

You are fact verifier. Solve a question answering task with interleaving Thought, Action, Observation steps.Thought can reason about the current situation,and Action can be three types:

[1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists.If not, it will return some similar entities to search.   
[2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.   
[3) Finish[answer], which returns the answer and finishes the task,answer should be one of REFUTES, SUPPORTS Or NOT ENOUGH INFO.

The following paragraph are insights a teacher agent provided to you. It is MANDATORY for you to follow these insights as CLOSELY as possible as they will help you perform the fact verification tasks efficiently:

# [Transfered insights]

You may take maximum of 7 steps.

Here are some examples:

# [Fixed fewshot examples]

[END OF EXAMPLES]

Now it's your turn!

```python
Claim: ...  
...: ...  
Action H: Finish[...]  
Observation H: Episode finished, reward = ... 
```

# F.2 Transfer Learning Prompt Template

# G Example Insights

Below are some example insights extracted by $\mathtt { g p t - 4 - 0 6 1 3 }$ or humans by examining the failed and successful trajectories. Some interesting insights are highlighted in purple (including the emergent ones demonstrated in Sec. 5.3).

# G.1 HotpotQA insights

# HotpotQA

1. Always ensure to exhaust all possible search queries related to the task at hand before concluding.   
2.When searching for information, consider using different combinations of keywords,and if necessary， break down complex queries into simpler, more specific ones to increase the chances of finding the required information.   
3. If a search query does not yield the desired results,consider rephrasing, using synonyms,or altering the specificity of the query for a more effective search.   
4. When faced with a complex task, consider breaking it down into smaler, more manageable parts and tackle each part individually. This can help in better understanding the task and in formulating more effective search queries.   
5.Always verify the information obtained from the search results to ensure it directly answers the task question before concluding.   
6.When dealing with tasks involving historical events or figures, consider searching for timelines， historical records, or biographical information to provide a comprehensive understanding of the context.   
7.When a search query does not yield the expected results, consider revising the search strategy by looking for related information that can lead to the answer.   
8.If the information is not found after several attempts,consider that the answer might be in the observations already made. Review them carefully before making further searches.   
9.If the task involves finding a specific event in history， consider searching for timelines or historical records related to the topic.   
10.When comparing two or more entities based on a certain atribute like birth dates],ensure to keep track of the attribute for each entity separately to avoid confusion.   
11. Always double-check your final answer against the information obtained to ensure accuracy before concluding the task.   
12.When searching for specific episodes or events,include the name of the series or event in the search query to increase the chances of finding the required information.   
13.When comparing the atributes of multiple entities,ensure to keep a clear record of each entity's attributes to avoid confusion and ensure accurate comparison.

Figure 11: An example of Extracted Insights for HotpotQA. One component resulting to the improved performance of ExpeL on HotpotQA can be traced to several pivotal insights extracted from its past experiences. A special emphasis is placed on insights 2 and 4, which suggest breaking down complex questions into simpler queries, reminiscent of the mechanism of Auto-GPT (Significant-Gravitas 2023). Besides, as mentioned, the emergent abilities arising from insight 8 were discussed in Sec. 5.3 and illustrated in Fig. 16, 17.

# Hand-crafted

![](images/c3c7c4e060f753f654a737c60751b214f7a84447c81bb6ae1d81fc71bcf2388c.jpg)  
Figure 12: Hand-crafted insights for the HotpotQA environment. This figure summarizes insights derived through a manual examination of both successful and unsuccessful Reflexion (Shinn et al. 2023) trajectories. These insights have been carefully crafted to address the most prevalent mistakes. On a related note, we observe that GPT-4 is able to extract a variety of insights (Fig. 11) in common with these hand-crafted ones, as depicted in this figure. For instance, insights 3 and 6 underscore the importance of exhausting all steps before conceding and diversifying search keywords to achieve better results if initial attempts are inconclusive. This illustrates that our proposed method accompanied by powerful gpt-4-0613 LLM, shows traces of human-like abstraction capabilities.

1. Do single searches on the same step and not multiple actions on the same line.   
2.There are only Search[<topic>]. Lookup[<topic>], Finish[<answer>] actions,do not invent your own actions.   
3. Do the given task until the very end.   
4.When encountering "Could not find" or "Similar.J"， look at the results closely where some might be helpful searches.   
5. Use Search[topic [category)] to narrow down search when the initial search is too broad.   
6. Do not use the same query as previous steps when doing Search.   
7. Do not callfor/interact with user or experts.   
8. Do not think you can access other resources, you only have access to Wikipedia APl.   
9. Do not finish the task without providing an answer.   
10.When you successfully find a page with search, you can follow up with Lookup[<topic>] to find specific information.

# G.2 ALFWorld insights

# ALFWorld

![](images/aa1fcd18bb9a374ec8193567d33dc347018eef1939c7d062379084349c8dd893.jpg)  
Figure 13: An example of Extracted Insights for ALFWorld. We showcase insights extracted by our agent in the ALFWorld Environment. Some particularly interesting insights are highlighted in purple.

1. Always confirm the presence of an item before attempting to interact with it.   
2. Before using an appliance or storage, ensure it is ready for use. If not, take necessary actions to prepare it. Once an item is found and prepared if necessary).it should be placed in the designated location immediately to avoid confusion.   
3.If an item is not found in one location,systematically check the next likely location based on the nature of the item and common household organization, without assuming its presence.   
4. Prioritize checking locations that are most likely to contain the item you are looking for, based on the nature of the item,common household organization,and the task at hand.   
5. If an attempt to interact with an item fails or does not progress the task, reassess the situation and consider alternative actions or locations before repeating the same action.   
6.Limit the number of consecutive thoughts without actions to ensure progress in task completion and avoid reaching the maximum allowed number of steps.   
7. When an item is found,ensure to specify the item's number and location when interacting with it.   
8.Always ensure to check all possible locations before concluding that an item is not present.   
9. Keep track of the locations already checked to avoid unnecessary repetition and to save time.

10.When a task involves multiple items of the same type, do not assume that allitems will be found in different locations. Some may be found in the same location.   
11.When multiple items are involved in a task,ensure to interact with each item in the corect order as per the task requirement.   
12.When a task requires examining an item,ensure to physically interact with the item rather than just thinking about examining it.   
13.If a task requires cooling an item,consider using appliances that are specifically designed for cooling,such as a fridge or a freezer.   
14.When a task requires the use of a specific item,ensure to correctly identify and use the exact item as per the task requirement. Do not confuse it with similar items   
15.When searching for an item, consider the nature of the item and its typical usage. For example, a pan is more likely to be found on a stoveburner or countertop rather than in a cabinet or drawer.   
16.When a task requires placing an item in a specific location,ensure the location is suitable and has enough space for the item.If the location is not suitable,find an alternative location that fulfils the task requirement.   
17. When a task requires cleaning an item,consider using a sinkbasin or other suitable cleaning tools or appliances.   
18.When a task requires moving an item,ensure to check the destination location's capacity before placing the item to avoid cluttering.   
19. When a task involves multiple items of the same type, ensure to keep track of the number of items already found and interacted with to avoid confusion and ensure task completion.   
20.When a task requires heating or cooling an item,ensure to use the appropriate appliance such as a microwave for heating or a fridge for cooling.

# Webshop

1. Before purchasing,verify that the product meets allthe specific requirements mentioned in the task instruction.   
2.Always ensure to understand the product attributes like color, size,etc.] mentioned in the product description,and select them if they are selectable options to ensure the product matches the task instruction.   
3. Use precise search terms to get relevant results, but avoid being overly specific which might limit the search results.   
4. Always think and validate the product details before making a purchase.   
5.Always include the price range in the search query to narrow down the search results to the most relevant products within the specified budget.   
6.Always consider the product's price before purchasing to ensure it fals within the budget specified in the task instruction.   
7.Always double-check the product type before purchasing.Do not rely solely on the product attributes,but also consider the product's main category to ensure it matches the task instruction.   
8.Always include all the task requirements in the search query,including the price range, to get the most relevant results.   
9.Always ensure to check the product's style if it is mentioned in the task instruction.If the styleis a selectable option,make sure to select the correct style before purchasing.   
10.Always ensure the product selected is exactly what is requested in the task instruction.If the product has multiple variations,make sure to select the correct one.   
11.Always read the product description and reviews if available,to gain more information about the product and ensure it meets the task requirements.   
12.If the product has multiple pack options [like pack of 2,pack of 3,etc.J,ensure to select the correct pack that matches the task instruction before purchasing.   
13.If the product has multiple variations,ensure to select the correct one that matches the task instruction.If the correct variation is not available, consider other products from the search results.   
14. If the first product selection atempt is unsuccessful, do not repeat the same search query. Instead, consider other products from the initial search results or modify the search query to get more relevant results.   
15.If the product has selectable options,ensure to select the correct options that match the task instruction before purchasing.

Figure 14: An example of Extracted Insights for WebShop. We showcase insights extracted by our agent in the WebShop Environment. Some particularly interesting insights are highlighted in purple.

# G.4 FEVER insights

# Fever

In order to successfully complete factual verification tasks, begin by clearly understanding the claim.Then formulate a search query that is precise and directly related to the claim.Include the main subjects or context from the claim in your query. If the initial search doesn't yield desired results, consider refining the query, using synonyms,or breaking down the claim into smaller parts.Always verify the information you obtain against the claim before drawing a conclusion. If multiple searches fail, consider changing the search strategy or looking for related information that might indirectly provide the necessary information.If all else fails, consider that the answer might be found in the observations already made. When you're ready to draw a conclusion, double-check it against the information obtained and ensure its accuracy. Lastly， always be prepared to exhaust all possible search queries related to the task at hand before concluding. Remember, the claim can either be supported, refuted,or there might not be enough information to draw a conclusion.

Figure 15: An example of Transferred Insights for FEVER. We showcase insights extracted by our agent in the FEVER Environment. Some particularly interesting insights are highlighted in purple.

# H Emergent Abilities Showcase

In this section, we showcase examples of ExpeL’s emergent abilities in different environments. Irrelevant or non-representative steps are omitted for clarity and conciseness.

# H.1 HotpotQA, Forming Analytical Deduction & Awareness of Environment Constraints

#

Question:  Which movie was filmed first "The Guest" or "You're Next"?

Thought 1: I need to search "The Guest" and "You're Next" to find out which movie was filmed first.

Action 1: Search["The Guest"]

Observation 1: A hypervisor [also known as a virtual machine monitor, VMM, or virtualizer] is a type of computer software, firmware or hardware that creates and runs virtual machines...

Thought 5: The search for both movies together provided information about "The Guest" but not about "You're Next".I should try searching for "You're Next" separately.

Action 5: Search["You're Next"]

Observation 5: Invasion of the Body Snatchers is a 1956 American science-fiction horror film produced by Walter Wanger, directed by Don Siegel,and starring Kevin McCarthy and Dana Wynter...

Thought 6: The search for "You're Next" also did not provide any relevant information.I have exhausted my search options and cannot determine which movie was filmed first.

Action 6: Finish[Unknown]

Observation 6: Answer is INCORRECT

![](images/aa487934b353a2470c59a89fbf29b61779a1dfff48ce37b75db9a2fd232fef01.jpg)

#

Possible Influential Insight:

“If the information is not found after several attempts, consider that the answer might be in the observations already made. Review them carefully before making further searches."

Thought 1: I need to search for information about the movies "The Guest" and "You're Next" to find out which one was filmed first.

Action 1: Search["The Guest" movie]

Observation 1: This is a list of programmes broadcast on South Korean cable television channel OCN.

Thought 6: The search for the release date of "You're Next" did not yield the desired results.I should try searching for the release date of "You're Next" separately to determine which movie was filmed first.

Action 6: Search["You're Next" release date]

Observation 6: Invasion of the Body Snatchers is a 1956 American science-fiction horror film produced by Walter Wanger, directed by Don Siegel,and starring Kevin McCarthy and Dana Wynte....

Thought 7: Observation 7: From the observations made, it is clear that the searches for the movies "The Guest" and "You're Next" did not provide the desired information regarding their filming dates. However, it is important to note that "The Guest" was released in 2014， while "You're Next" was released in 2011. Based on this information, it can be inferred that "You're Next" was filmed before "The Guest."

Action 7: Finish[You're Next]

Observation 7: Answer is CORRECT

![](images/5b290bf3d4e8bed158e404fdfd5e67fda6bde96cd41dce17dfcfd7efc43d9481.jpg)  
Figure 16: ExpeL Emergent Abilities in HotpotQA, Example 1. ExpeL demonstrates its adaptation to its environment, in particular, by reevaluating its trajectory to formulate an educated guess and successfully answer the question, rather than conceding, as observed with the vanilla ReAct agent. We provide a possible influencing insight for this trajectory, shown in the purple box.

# Question:

What team was led to victory in 1939 by a two-time All-American basketbal player nicknamed "Scat"?

# Thought 1:

I need to find the team that was led to victory in 1939 by a two-time All-American basketball player nicknamed "Scat". I should search for information about the 1939 NCAA basketball championship winner and look for any mention of a player with the nickname "Scat".

Action 1: Search[1939 NCAA basketball championship winner]

Observation 1: Could not find [1939 NCAA basketball championship winner]. Similar: ['1939 NCAA basketball championship game.,..]

Thought 6: I was unable to find specific information about the players on the Oregon Webfoots team in 1939.I should try searching for articles or sources that provide a roster or player information for that year.

Action 6: Search[Oregon Webfoots basketball roster 1939]

Observation 6: The 1938-39 Oregon Webfoots men's basketball team was a college basketball team that represented the University of Oregon. The Webfoots,coached ..

Thought 7: The available information does not mention a player nicknamed "Scat" on the Oregon Webfoots team in 1939. It seems that the nickname "Scat" is not associated with any player on the team. I should conclude that there is no information to determine the team led to victory in 1939 by a two-time All-American basketball player nicknamed "Scat."

Action 7: Finish[No information available]

Observation 7: AnsWer is INCORRECT

![](images/b840837ba4ffe8d6a97a328e54787ff900151fbe750a6275b019113a8ac05456.jpg)

Possible

Influential

Insight:

“If the information is not found after several attempts, consider that the answer might be in the observations already made. Review them carefully before making further searches."

Thought 1: I need to find the team that was led to victory in 1939 by a two-time All-American basketball player nicknamed "Scat".

Action 1: Search[1939 basketball team led by "Scat"]

Observation 1: James Carlos McNatt [December 19，1918 - December 23，2000] was an All-American basketball player for the Oklahoma Sooners and ..

Action 6: Search[1939 NCAA basketball teams]

Observation 6: Could not find [1939 NCAA basketball teams]. Similar: ['1939 NCAA basketball tournament,...!

Thought 7: The search for the 1939 NCAA basketball teams did not yield any relevant information. I should review the observations I have made so far and see if there are any other clues that can help me find the answer.

Thought 7: Observation 1 mentioned that James Carlos McNatt led the Oklahoma Sooners to the first-ever NCAA Final Four in 1939. Since McNatt is the only player mentioned so far, it is likely that the team led to victory in 1939 by a two-time All-American basketball player nicknamed "Scat" is the Oklahoma Sooners. I should proceed with this assumption.

Action 7: Finish[Oklahoma Sooners]

Observation 7: AnswerisCORRECT

![](images/99915c53cc3bf4a5e210c9fa76b57b3730d25b511f76cb65840caa00c490ff5d.jpg)  
Figure 17: ExpeL Emergent Abilities in HotpotQA, Example 2. ExpeL is going over the observation made during its trajectory to make an educated guess. We provide a possible influencing insight for this trajectory, shown in the purple box.

You are in the middle of a room. Looking quickly around you, you see a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 3, a countertop 2, a countertop 1, a drawer 3,a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a shelf 3, a shelf 2, a shelf 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.

Your task is to: cool some pan and put it in countertop.

#

> think: To solve the task,I need to find and take apan, then cool it with the fridge,and finally Iput it on the countertop.   
> think: First, I need to find a pan.A pan is more likely to appear in the countertop [1-3], cabinet [1-6],or drawer [1-3]. I can check one by one, starting with the countertop 1.   
> go to countertop 1 On the countertop 1, you see a mug 1.   
> go to countertop 2 On the countertop 2.   
> think: On the countertop 2, you see a plate 1, a knife 1,and a spoon 1.   
>..   
> go to cabinet 1 On the cabinet 1, you see a bowl 1, and a plate 1.   
> go to cabinet 2 The cabinet 2 is closed.   
>...   
> go to drawer 1 The drawer 1is closed.   
  
> open drawer 3 You open the drawer 3. The drawer 3 is open. In it,you see a dishsponge 2.   
> think: Now I find a pan. Next, I need to take it.   
> take pan from countertop 1 Nothing happens.   
> take pan 1 from countertop 1 Nothing happens. Task FAILED.

# 国

> think: To solve the task, I need to find and take a pan, then cool it with the fridge,and finally put it on the countertop.   
> think: First, I need to find a pan. A pan is more likely to appear in thestoveburner (1-4), sinkbasin [1], countertop [1-3), cabinet [1-6),or drawer [1-3). I will start by checking the stoveburner 1.   
> go to stoveburner 1 On the stoveburner 1, you see a pot 1.   
> go to stoveburner 2 On the stoveburner 2, you see a pan 1.   
> take pan 1 from stoveburner 2 You pick up the pan 1 from the stoveburner 2.   
> think: Now that I have the pan,I need to cool it with the fridge.   
> go to fridge 1 The fridge 1is closed.   
> open fridge 1 You open the fridge 1. The fridge 1 is open.   
> cool pan 1with fridge 1 You cool the pan 1 using the fridge 1.   
> think: Now that I have cooled the pan,I need to put it on the countertop.   
> go to countertop 1 On the countertop 1, you see a mug 1.   
> put pan 1on countertop 1 Task is SOLVED.

![](images/1e0b9d4b6a46dae3be72a04852e4148f4d57f1dcdc9d864a95c7216e0f7c1086.jpg)  
Figure 18: ExpeL Emergent Abilities in ALFWorld, World Model Belief Update. Through experience, ExpeL updated its prior knowledge of a pan’s likely location from a countertop to a stoveburner, enabling successful task completion. We provide a possible influencing insight for this trajectory, shown in the purple box.

# Possible Influential Insight:

“When searching for an item， consider the nature of the item and its typical usage. For example, a pan is more likely to be found on a stoveburner or countertop rather than in a cabinet or drawer."