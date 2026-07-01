# Flow-DPPO: Divergence Proximal Policy Optimization for Flow Matching Models

Bowen Ping<sup>1,2,∗</sup> Xiangxin Zhou<sup>2,∗</sup> <sup>¶</sup> Penghui Qi<sup>3</sup> 

Minnan Luo<sup>1,‡</sup> Liefeng Bo<sup>2</sup> Tianyu Pang<sup>2,‡</sup> 

<sup>1</sup>Xi’an Jiaotong University <sup>2</sup>Tencent Hunyuan <sup>3</sup>National University of Singapore 

<sup>∗</sup>Equal contribution <sup>¶</sup>Project Lead <sup>‡</sup>Corresponding author 

Abstract. Recent work has demonstrated that online reinforcement learning (RL) can substantially improve the quality and alignment of flow matching models for image and video generation. Methods such as Flow-GRPO and CPS cast the denoising process as a Markov Decision Process and apply PPO-style ratio clipping to enforce a trust region. However, we argue that ratio clipping is structurally ill-suited for flow models: the probability ratio between new and old policies is a noisy, single-sample estimate of the true policy divergence, leading to over-constraining in some regions of the trajectory and under-constraining in others. We propose Flow-DPPO (Flow Divergence Proximal Policy Optimization), which replaces ratio clipping with a divergence proximal constraint. A key observation is that the per-step policy in flow models is Gaussian, enabling exact and cheap computation of the KL divergence between old and new policies. Flow-DPPO employs an asymmetric divergence mask that blocks gradient updates only when they simultaneously move away from the trusted region and violate the divergence threshold. Experiments show that Flow-DPPO achieves higher rewards with better KL-proximal eficiency, alleviates catastrophic forgetting, promotes balanced multi-objective optimization, and enables stable multi-epoch training where ratio clipping degrades. 

Date: June 8, 2026 

Code: https://github.com/Tencent-Hunyuan/UniRL/tree/main/FlowDPPO 

## 1 Introduction

Reinforcement learning (RL) has emerged as a core paradigm for aligning models with downstream objectives. In language models, RL methods such as DPO (Rafailov et al., 2023) and GRPO (Shao et al., 2024) have substantially improved alignment (Ouyang et al., 2022) and reasoning capabilities (Guo et al., 2025). Recently, these advances have been extended to image and video generation Liu et al. (2025); Wallace et al. (2024); Wang and Yu (2025); Xue et al. (2025a); Zheng et al. (2026), where flow matching models (Lipman et al., 2023; Liu et al., 2023) represent the dominant generative framework. Among them, Flow-GRPO (Liu et al., 2025) and DanceGRPO (Xue et al., 2025b) 


GRPO-Guard



FLUX.1-dev



Flow-GRPO


seven green croissants 

a blue dog on top of three white sheep behind seven white candles 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-01/d52561ff-9523-48be-af73-f2383cdc7168/5884c75bfbb8ffe65c5b247b17255023f5713ecf4c47193e287cafb4de759c36.jpg)



a blue girafe behind seven pink clocks to the right of an elephant



Figure 1: Qualitative comparison on FLUX.1-dev (Black Forest Labs, 2024) with GenEval2 (Kamath et al., 2025) prompts. Flow-DPPO achieves competitive compositional accuracy with notably less image quality degradation compared to Flow-GRPO (Liu et al., 2025), Flow-CPS (Wang and Yu, 2025), and GRPO-Guard (Wang et al., 2025), reflecting their superior KL-proximal eficiency.


demonstrated strong performance by transforming deterministic ODE sampling into stochastic SDE trajectories and introducing PPO-style ratio clipping to enforce trust-region optimization. 

The theoretical foundation of trust-region methods originates from Trust Region Policy Optimization (TRPO) (Schulman et al., 2015), which establishes a policy improvement bound: monotonic improvement is guaranteed when policy updates remain within a trust region defined by the divergence between the old and new policies. PPO (Schulman et al., 2017) later introduced ratio clipping as a computationally eficient first-order approximation to TRPO. However, as noted by Qi et al. (2026), each clipping decision is based on a single-sample Monte Carlo estimate of the true Total Variation (TV) divergence, rather than the divergence itself. In the continuous and high-dimensional latent space of flow models, this estimation noise becomes substantially amplified, leading to a systematic left shift in the ratio distribution, with its mean falling below one (Wang et al., 2025). We show that this bias is intrinsic to Gaussian policies: the standard PPO clipping range [1−ϵ, 1+ϵ] therefore becomes efectively asymmetric, failing to adequately constrain over-optimization for positive-advantage samples while excessively clipping negative-advantage ones. 

To mitigate this bias, GRPO-Guard (Wang et al., 2025) proposed normalizing the ratio distribution. While this re-centering alleviates the symptom, it does not address the root cause: the ratio remains a noisy, per-sample proxy for the true policy divergence. We observe that flow models ofer a structural advantage that sidesteps this problem entirely. Because each per-step policy is Gaussian with a mean µ<sub>θ</sub> determined by the velocity network and a fixed, schedule-dependent variance σ, the KL divergence between old and new policies reduces to $\| \pmb { \mu } _ { \theta _ { \mathrm { o l d } } } - \pmb { \mu } _ { \theta } \| ^ { 2 } / ( 2 \sigma ^ { 2 } )$ , which is an exact, deterministic quantity that can be computed from two forward passes already performed during training. Unlike the LLM setting, where DPPO (Qi et al., 2026) must resort to approximate divergence reductions over large vocabularies, flow models admit exact divergence computation at no additional cost. This motivates replacing ratio clipping with a direct KL-proximal trust region constraint. 

Building on this insight, we propose Flow-DPPO (Flow Divergence Proximal Policy Optimization), which replaces ratio clipping with a divergence-based mask. The mask blocks gradient updates only when two conditions are jointly met: (1) the advantage and ratio indicate that the update is moving the policy away from the old policy, and (2) the exact KL divergence already exceeds a threshold. This design directly enforces the trust region while preserving the beneficial asymmetric structure of PPO: updates that move the policy towards the old policy are never blocked, accelerating recovery from overshooting. Extensive experiments on various base models demonstrate that Flow-DPPO achieves superior reward optimization, improved KL-proximal eficiency, stronger robustness to catastrophic forgetting, balanced multi-objective optimization that mitigates reward hacking, and stable multi-epoch training that enables higher sample eficiency. Figure 1 presents qualitative generation results demonstrating that Flow-DPPO achieves competitive compositional accuracy while preserving notably higher visual quality than existing methods. 

## 2 Preliminaries

Flow matching (Lipman et al., 2023; Liu et al., 2023) learns a continuous-time velocity field that transports samples from a simple source distribution to the data distribution. Specifically, let $\pmb { x } _ { 0 } \sim \pi _ { 0 } = p _ { \mathrm { d a t a } }$ , and define an interpolating path ${ \pmb x } _ { t } = \alpha _ { t } { \pmb x } _ { 0 } + \sigma _ { t } { \pmb \epsilon }$ with $\mathbf { \epsilon } \gets \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ , where $\alpha _ { t }$ and $\sigma _ { t }$ determine the probability path between data and noise. This construction induces a conditional distribution $\pi _ { t | 0 } ( \pmb { x } _ { t } \mid \pmb { x } _ { 0 } ) = \mathcal { N } ( \alpha _ { t } \pmb { x } _ { 0 } , \sigma _ { t } ^ { 2 } \mathbf { I } )$ . The goal of flow matching is to train a time-dependent vector field ${ \pmb v } _ { \theta } ( { \pmb x } _ { t } , t )$ to match the target velocity, which is given by $\begin{array} { r } { \pmb { v } = \frac { \mathrm { d } \pmb { x } _ { t } } { \mathrm { d } t } = \dot { \alpha } _ { t } \pmb { x } _ { 0 } + \dot { \sigma } _ { t } \pmb { \epsilon } } \end{array}$ and the functinal $\dot { f } _ { t } : = \mathrm { d } f _ { t } / \mathrm { d } t$ . The model ${ \pmb v } _ { \theta } ( { \pmb x } _ { t } , t )$ is then trained by minimizing the regression objective 

$$
\mathbb {E} _ {t, \pmb {x} _ {0} \sim \pi_ {0}, \pmb {\epsilon} \sim \mathcal {N} (\pmb {0}, \mathbf {I})} \left[ w (t) \| \pmb {v} _ {\theta} (\pmb {x} _ {t}, t) - \pmb {v} \| _ {2} ^ {2} \right],\tag{1}
$$

where $w ( t )$ is a weighting function. After training, samples are generated by solving the ODE $\begin{array} { r } { \frac { \mathrm { d } \pmb { x } _ { t } } { \mathrm { d } t } = \pmb { v } _ { \theta } ( \pmb { x } _ { t } , t ) } \end{array}$ . In practice, simple numerical solvers such as Euler discretization are often suficient for high-quality sampling (Karras et al., 2022; Lu et al., 2022; Song et al., 2021a). A notable special case is rectified flow (Liu et al., 2023), which uses the linear conditional path $\alpha _ { t } = 1 - t$ and $\sigma _ { t } = t$ Under this choice, the target velocity reduces to ${ \pmb v } = { \pmb \epsilon } - { \pmb x } _ { 0 }$ . We adopt this linear schedule throughout the paper. 

## 2.1 RL Fine-Tuning for Flow Matching Models

For text-conditional flow matching models, given a conditioning prompt c, generation starts from a Gaussian latent $\mathbf { \sigma } _ { \mathbf { X } _ { T } } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ and progressively transforms it into a clean sample $\scriptstyle { \mathbf { { \vec { x } } } } _ { 0 }$ . At each timestep t, the flow model predicts a velocity field ${ \pmb v } _ { \theta } ( { \pmb x } _ { t } , t , { \pmb c } )$ , which specifies a deterministic generation direction. Applying RL algorithms such as GRPO (Shao et al., 2024) to flow matching models requires a sampler-induced stochastic policy at each denoising step. Flow-GRPO (Liu et al., 2025) constructs such a policy via an ODE-to-SDE conversion, which transforms the probability-flow ODE into an equivalent SDE with the same marginals (Albergo and Vanden-Eijnden, 2023; Albergo et al., 2024; Song et al., 2021b): d $\begin{array} { r } { \pmb { x } _ { t } = \left[ \pmb { v } _ { \theta } ( \pmb { x } _ { t } , t , \pmb { c } ) + \frac { \sigma _ { t } ^ { 2 } } { 2 t } \big ( \pmb { x } _ { t } + ( 1 - t ) \pmb { v } _ { \theta } ( \pmb { x } _ { t } , t , \pmb { c } ) \big ) \right] \mathrm { d } t + \sigma _ { t } } \end{array}$ dw, where dw denotes Wiener process increments, $\begin{array} { r } { \sigma _ { t } = a \sqrt { \frac { t } { 1 - t } } } \end{array}$ , and a is a scalar hyperparameter controlling the noise level. Applying Euler–Maruyama discretization yields the Flow-SDE sampler: 

$$
\pmb {x} _ {t - \Delta t} = \pmb {x} _ {t} + \left[ \pmb {v} _ {\theta} (\pmb {x} _ {t}, t, \pmb {c}) + \frac {\sigma_ {t} ^ {2}}{2 t} \big (\pmb {x} _ {t} + (1 - t) \pmb {v} _ {\theta} (\pmb {x} _ {t}, t, \pmb {c}) \big) \right] \Delta t + \sigma_ {t} \sqrt {\Delta t} \pmb {\epsilon},\tag{2}
$$

with $\mathbf { \epsilon } \epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ . An alternative is Coeficients-Preserving Sampling (CPS) (Wang and Yu, 2025), which reduces the excessive noise injection in Flow-SDE and better preserves the interpolation structure of the scheduler. Let $\hat { \mathbf { x } } _ { 0 } = \mathbf { x } _ { t } - t \hat { v } _ { \theta } ( \mathbf { x } _ { t } , t , c ) , \hat { \mathbf { x } } _ { 1 } = \mathbf { x } _ { t } + \left( 1 - t \right) \hat { v } _ { \theta } ( \mathbf { x } _ { t } , t , c )$ denote the predicted clean sample and noise component, respectively. CPS updates the latent as 

$$
\pmb {x} _ {t - \Delta t} = (1 - (t - \Delta t)) \hat {\pmb {x}} _ {0} + (t - \Delta t) \cos \left(\frac {\eta \pi}{2}\right) \hat {\pmb {x}} _ {1} + (t - \Delta t) \sin \left(\frac {\eta \pi}{2}\right) \pmb {\epsilon},\tag{3}
$$

where $\mathbf { \epsilon } \gets \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ and $\eta \in [ 0 , 1 ]$ controls the stochasticity. Both Flow-SDE and CPS therefore induce Gaussian per-step policies written as 

$$
p _ {\theta} (\pmb {x} _ {t - \Delta t} \mid \pmb {x} _ {t}, t, \pmb {c}) = \mathcal {N} \big (\pmb {x} _ {t - \Delta t}; \pmb {\mu} _ {\theta} (\pmb {x} _ {t}, t, \pmb {c}), \sigma^ {2} (t) \mathbf {I} \big),\tag{4}
$$

where the specific forms of $\mu _ { \theta }$ and $\sigma ( t )$ depend on the sampler. The above generative process can be formulated as a finite-horizon Markov Decision Process (MDP) (Black et al., 2024; Fan et al., 2023; Liu et al., 2025). To distinguish the discrete decision process from the underlying continuous-time flow, we use $k \in \{ 1 , \ldots , K \}$ for the MDP state index and $t \in [ 0 , 1 ]$ for the reverse-time variable of the flow. Let $0 = t _ { K } < t _ { K - 1 } < \cdot \cdot \cdot < t _ { 1 } = 1$ be a discretization of reverse time, so that state k corresponds to flow time $t _ { k }$ . The state at step $k$ is $\pmb { s } _ { k } = ( \pmb { c } , t _ { k } , \pmb { x } _ { t _ { k } } )$ . Note that $t _ { k } - t _ { k + 1 } = \Delta t$ . For $k = 1 , \ldots , K - 1$ , the action is the next latent sample, $\mathbf { \Delta } \mathbf { a } _ { k } = \mathbf { x } _ { t _ { k + 1 } } .$ , drawn from the sampler-induced policy $\pi _ { \boldsymbol { \theta } } ( \mathbf { a } _ { k } \mid s _ { k } ) = \pi _ { \boldsymbol { \theta } } ( \mathbf { x } _ { t _ { k + 1 } } \mid \mathbf { x } _ { t _ { k } } , t _ { k } , { c } )$ . Given the sampled action, the transition is deterministic, with next state $\pmb { s } _ { k + 1 } = ( \pmb { c } , t _ { k + 1 } , \pmb { x } _ { t _ { k + 1 } } )$ . The rollout starts from $c \sim p ( c )$ and $\pmb { x } _ { t _ { 1 } } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ , and terminates at $k = K$ where $t _ { K } = 0$ 

After the full generative process, a scalar reward $R ( \pmb { x } _ { 0 } , \pmb { c } )$ is provided. RL fine-tuning maximizes the expected terminal reward with a KL regularization term that penalizes deviation from the pretrained reference policy $\pi _ { \mathrm { r e f } } \colon$ max<sub>θ</sub> $\begin{array} { r } { \mathbb { E } _ { c \sim p ( c ) , \tau \sim \pi _ { \theta } } \left[ R ( \pmb { x } _ { 0 } , \pmb { c } ) - \beta \sum _ { k = 1 } ^ { K - 1 } D _ { \mathrm { K L } } \big ( \pi _ { \theta } ( \cdot \mid \pmb { s } _ { k } ) \mid \mid \pi _ { \mathrm { r e f } } ( \cdot \mid \pmb { s } _ { k } ) \big ) \right] } \end{array}$ , where $\tau = ( \boldsymbol { s } _ { 1 } , \boldsymbol { a } _ { 1 } , \boldsymbol { s } _ { 2 } , \boldsymbol { a } _ { 2 } , \ldots , \boldsymbol { s } _ { K - 1 } , \boldsymbol { a } _ { K - 1 } , \boldsymbol { s } _ { K } )$ denotes a trajectory induced by $\pi _ { \theta }$ and $\beta \geq 0$ controls the regularization strength. This KL penalty discourages reward hacking and mitigates catastrophic forgetting of the pretrained model’s capabilities. 

Flow-GRPO (Liu et al., 2025) applies GRPO to the above MDP. Given a prompt c, the current policy generates a group of G samples $\{ \boldsymbol { x } _ { 0 } ^ { i } \} _ { i = 1 } ^ { G }$ . Their rewards are normalized within the group to obtain relative advantages: $\hat { A } ^ { i } = \big ( R ( x _ { 0 } ^ { i } , c ) - \mathrm { m e a n } ( \{ R ( x _ { 0 } ^ { j } , c ) \} _ { j = 1 } ^ { G } ) \big ) / \mathrm { s t d } ( \{ R ( x _ { 0 } ^ { j } , c ) \} _ { j = 1 } ^ { G } )$ . In practice, each policy optimization iteration begins by rolling out a batch of data, which is then split into several minibatches for multiple gradient steps. This procedure introduces policy staleness: after the first update, the optimizing policy has already diverged from the behavior policy that generated the data. To control this of-policy drift, a trust region mechanism is applied. Following PPO (Schulman et al., 2017), the policy is optimized using the clipped surrogate objective 

$$
\mathcal {L} ^ {\mathrm{Flow-GRPO}} (\theta) = \mathbb {E} \left[ \frac {1}{G} \sum_ {i = 1} ^ {G} \frac {1}{K} \sum_ {k = 1} ^ {K} \left(\min \left(r _ {k} ^ {i} (\theta) \hat {A} ^ {i}, \operatorname{clip} (r _ {k} ^ {i} (\theta), 1 - \epsilon , 1 + \epsilon) \hat {A} ^ {i}\right)\right) \right],\tag{5}
$$

where we omit the KL penalty term $D _ { \mathrm { K L } } ( \pi _ { \theta } \| \pi _ { \mathrm { r e f } } )$ for brevity, and the per-step importance ratio is defined as $\begin{array} { r } { r _ { k } ^ { i } ( \theta ) = \frac { p _ { \theta } ( \pmb { x } _ { t _ { k } - \Delta t } ^ { i } | \pmb { x } _ { t _ { k } } ^ { i } , \pmb { c } ) } { p _ { \theta _ { \mathrm { o l d } } } ( \pmb { x } _ { t _ { k } - \Delta t } ^ { i } | \pmb { x } _ { t _ { k } } ^ { i } , \pmb { c } ) } } \end{array}$ . Since both Flow-SDE and CPS define Gaussian per-step policies as in Eq. (4), the log-ratio admits the same closed-form expression: 

$$
\log r _ {k} ^ {i} (\theta) = \frac {\| \boldsymbol {x} _ {t _ {k} - \Delta t} ^ {i} - \boldsymbol {\mu} _ {\theta_ {\mathrm{old}}} \| ^ {2} - \| \boldsymbol {x} _ {t _ {k} - \Delta t} ^ {i} - \boldsymbol {\mu} _ {\theta} \| ^ {2}}{2 \sigma^ {2} (t _ {k})}.\tag{6}
$$

Therefore, both samplers can be optimized within the same GRPO framework, difering only in the parameterization of the induced stochastic policy. 

## 3 Methodology

In this section, we first derive a policy improvement bound that justifies trust-region methods for flow models. Then, we show that ratio clipping is a noisy proxy for the true divergence constraint. Finally, we present Flow-DPPO, which leverages exact KL computation to enforce a deterministic divergence mask, yielding a tighter and variance-free trust-region constraint. 

## 3.1 Trust-Region Policy Optimization for Flow Matching Models

Inspired by Schulman et al. (2017); Qi et al. (2026), we adapt the trust region framework to the flow model fine-tuning setting defined in Section 2.1. This setting difers from the classical discounted RL paradigm in two important ways. First, the problem is an undiscounted episodic task with a finite horizon of K − 1 decision steps. Second, due to the terminal reward structure, advantages are estimated at the trajectory level rather than per step. These properties necessitate a tailored policy improvement guarantee. We follow the MDP defined in Section 2.1. 

Theorem 1 (Performance Diference Identity for Flow Models). In the finite-horizon flow model MDP with K − 1 decision steps, let $J ( \pi ) = \mathbb { E } _ { c \sim p ( c ) , \tau \sim \pi } [ R ( \pmb { x } _ { 0 } , \pmb { c } ) ]$ denote the expected reward. For any two policies π<sub>θ</sub> and $\pi _ { \theta _ { \mathrm { o l d } } }$ , the performance diference decomposes as: $J ( \pi _ { \theta } ) - J ( \pi _ { \theta _ { \mathrm { o l d } } } ) = L _ { \theta _ { \mathrm { o l d } } } ^ { \prime } ( \pi _ { \theta } ) -$ $\Delta ( \pi _ { \theta _ { \mathrm { o l d } } } , \pi _ { \theta } )$ , where the surrogate objective is 

$$
L _ {\theta_ {\mathrm{old}}} ^ {\prime} (\pi_ {\theta}) = \mathbb {E} _ {\tau \sim \pi_ {\theta_ {\mathrm{old}}}} \left[ R (\pmb {x} _ {0}, \pmb {c}) \sum_ {k = 1} ^ {K - 1} \left(\frac {\pi_ {\theta} (\pmb {a} _ {k} \mid \pmb {s} _ {k})}{\pi_ {\theta_ {\mathrm{old}}} (\pmb {a} _ {k} \mid \pmb {s} _ {k})} - 1\right) \right],\tag{7}
$$

and the error term is 

$$
\Delta (\pi_ {\theta_ {\mathrm{old}}}, \pi_ {\theta}) = \mathbb {E} _ {\tau \sim \pi_ {\theta_ {\mathrm{old}}}} \left[ R (\boldsymbol {x} _ {0}, \boldsymbol {c}) \sum_ {k = 1} ^ {K - 1} \left(\frac {\pi_ {\theta} (\boldsymbol {a} _ {k} \mid \boldsymbol {s} _ {k})}{\pi_ {\theta_ {\mathrm{old}}} (\boldsymbol {a} _ {k} \mid \boldsymbol {s} _ {k})} - 1\right) \left(1 - \prod_ {j = k + 1} ^ {K - 1} \frac {\pi_ {\theta} (\boldsymbol {a} _ {j} \mid \boldsymbol {s} _ {j})}{\pi_ {\theta_ {\mathrm{old}}} (\boldsymbol {a} _ {j} \mid \boldsymbol {s} _ {j})}\right) \right].
$$

The surrogate $L _ { \theta _ { \mathrm { o l d } } } ^ { \prime } ( \pi _ { \theta } )$ represents a first-order approximation to the true improvement, while the error term $\Delta$ captures higher-order interactions between per-step policy changes. To yield a practical optimization objective, we bound this error term. 

Theorem 2 (Policy Improvement Bound for Flow Models). In the finite-horizon flow model MDP with K − 1 decision steps, the policy improvement is lower-bounded by: 

$$
J (\pi_ {\theta}) - J (\pi_ {\theta_ {\mathrm{old}}}) \geq L _ {\theta_ {\mathrm{old}}} ^ {\prime} (\pi_ {\theta}) - 2 \xi (K - 1) (K - 2) \cdot D _ {\mathrm{TV}} ^ {\max} (\pi_ {\theta_ {\mathrm{old}}} \| \pi_ {\theta}) ^ {2},\tag{8}
$$

where $\begin{array} { r } { D _ { \mathrm { T V } } ^ { \operatorname* { m a x } } ( \pi _ { \theta _ { \mathrm { o l d } } } \| \pi _ { \theta } ) = \operatorname* { m a x } _ { s _ { k } } D _ { \mathrm { T V } } \left( \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot  { | } s _ { k } )  { | } | \pi _ { \theta } ( \cdot  { | } s _ { k } ) \right) } \end{array}$ is the maximum per-step Total Variation divergence, and $\xi = \operatorname* { m a x } _ { { \pmb x } _ { 0 } , { \pmb c } } | R ( { \pmb x } _ { 0 } , { \pmb c } ) |$ is the maximum absolute reward. 

Please refer to Appendix B for the detailed derivation; a tighter bound linear in K is given in Appendix B.3. This bound is structurally analogous to the policy improvement bound for LLMs derived in Qi et al. (2026). It provides a rigorous justification for trust-region methods in flow model fine-tuning: constraining the per-step divergence controls the penalty term and guarantees monotonic improvement. Similar to TRPO (Schulman et al., 2015), we can solve the following constrained optimization problem to ensure stable learning: 

$$
\max _ {\pi_ {\theta}} L _ {\theta_ {\mathrm{old}}} ^ {\prime} (\pi_ {\theta}), \qquad \text {s.t.} \quad D _ {\mathrm{TV}} ^ {\max} (\pi_ {\theta_ {\mathrm{old}}} \| \pi_ {\theta}) \leq \delta .\tag{9}
$$

Remark 3 (Exact Divergence in the Gaussian Setting). For the Gaussian per-step policies in $E q .$ (4), the TV divergence is a monotone function of the mean displacement: 

$$
D _ {\mathrm{TV}} \big (\pi_ {\theta_ {\mathrm{old}}} (\cdot | \pmb {s} _ {k}) \| \pi_ {\theta} (\cdot | \pmb {s} _ {k}) \big) = 2 \Phi \left(\frac {\| \pmb {\mu} _ {\theta_ {\mathrm{old}}} - \pmb {\mu} _ {\theta} \|}{2 \sigma (t _ {k})}\right) - 1,\tag{10}
$$

where $\Phi$ is the standard normal CDF. Constraining the TV divergence below a threshold is therefore equivalent to constraining $\| \pmb { \mu } _ { \theta _ { \mathrm { o l d } } } - \pmb { \mu } _ { \theta } \| ^ { 2 } \leq \delta ^ { \prime }$ for an appropriate $\delta ^ { \prime } ,$ which is precisely the divergence measure that Flow-DPPO employs. Moreover, the Pinsker inequality $\begin{array} { r } { D _ { \mathrm { T V } } ( p \| q ) ^ { 2 } \leq \frac { 1 } { 2 } D _ { \mathrm { K L } } ( p \| q ) } \end{array}$ ensures that our KL-based constraint also upper-bounds the TV divergence: when the per-step $D _ { \mathrm { K L } } \leq \delta _ { \mathrm {  } }$ , we have $D _ { \mathrm { T V } } ^ { \mathrm { m a x } } \leq \sqrt { \delta / 2 }$ . In the Gaussian equal-covariance case, the converse also holds since KL and TV are both monotone functions of $\| \pmb { \mu } _ { \theta _ { \mathrm { o l d } } } - \pmb { \mu } _ { \theta } \| / \sigma$ . Thus, our method is theoretically justified from both the KL and TV perspectives. Unlike the LLM setting, where the discrete vocabulary requires approximate divergence computations (Qi et al., 2026), the Gaussian structure of flow models provides exact per-step divergence at zero additional cost. 

## 3.2 Pitfalls of Ratio Clipping in Flow-GRPO

Flow-GRPO adopts PPO-style ratio clipping to enforce a trust region. For consistency with the Flow-GRPO notation (Liu et al., 2025), in this and the following subsections we index denoising steps by the flow time t (equivalently, $t = t _ { k }$ in the MDP indexing of Section 2.1). The clipping condition $| r _ { t } ^ { i } - 1 | \leq \epsilon$ is intended to prevent the new policy from deviating too far from the old one. However, the probability ratio is a fundamentally noisy proxy for the true policy divergence. By definition of the Total Variation divergence, 

$$
D _ {\mathrm{TV}} \big (\pi_ {\theta_ {\mathrm{old}}} (\cdot | \pmb {x} _ {t}) \| \pi_ {\theta} (\cdot | \pmb {x} _ {t}) \big) = \frac {1}{2} \mathbb {E} _ {\pmb {x} _ {t - \Delta t} \sim \pi_ {\theta_ {\mathrm{old}}}} \big [ | r _ {t} ^ {i} - 1 | \big ],\tag{11}
$$

so each individual $| r _ { t } ^ { i } - 1 |$ is merely a single-sample Monte Carlo estimate of $2 D _ { \mathrm { T V } }$ . While the policy improvement bound (Theorem 2) calls for constraining $D _ { \mathrm { T V } } ^ { \mathrm { m a x } }$ , ratio clipping constrains this noisy per-sample surrogate instead. This issue was identified by Qi et al. (2026) in the LLM setting; we now show that the resulting pathology is particularly severe in flow models due to the high-dimensional continuous action space. 

Recall from Eq. (6) that the log-ratio is: 

$$
\log r _ {t} ^ {i} (\theta) = \frac {\| \boldsymbol {x} _ {t - \Delta t} ^ {i} - \boldsymbol {\mu} _ {\theta_ {\mathrm{old}}} \| ^ {2} - \| \boldsymbol {x} _ {t - \Delta t} ^ {i} - \boldsymbol {\mu} _ {\theta} \| ^ {2}}{2 \sigma^ {2}}.\tag{12}
$$

Since $\pmb { x } _ { t - \Delta t } ^ { i }$ is sampled from $\mathcal { N } ( \mu _ { \theta _ { \mathrm { o l d } } } , \sigma ^ { 2 } \mathbf { I } )$ , we can write $\pmb { x } _ { t - \Delta t } ^ { i } = \pmb { \mu } _ { \theta _ { \mathrm { o l d } } } + \sigma \epsilon$ where $\epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ Substituting and letting $d = \mu _ { \theta } - \mu _ { \theta _ { \mathrm { o l d } } }$ 

$$
\log r _ {t} ^ {i} (\theta) = \frac {\| \sigma \epsilon \| ^ {2} - \| \sigma \epsilon - d \| ^ {2}}{2 \sigma^ {2}} = \frac {2 \sigma \epsilon^ {\top} d - \| d \| ^ {2}}{2 \sigma^ {2}} = \frac {\epsilon^ {\top} d}{\sigma} - \frac {\| d \| ^ {2}}{2 \sigma^ {2}}.\tag{13}
$$

The first term, $\epsilon ^ { \top } d / \sigma ,$ is a zero-mean random variable with variance $\| d \| ^ { 2 } / \sigma ^ { 2 }$ . This reveals that the log-ratio is dominated by noise: the signal (the deterministic second term $- \| d \| ^ { 2 } / ( 2 \sigma ^ { 2 } ) )$ is exactly the negative of the KL divergence, but it is corrupted by a noise term whose standard deviation $\| d \| / \sigma$ is of the same order as the signal itself. This analysis yields two key insights: 

1. High variance. The ratio $r _ { t } ^ { i }$ is inherently noisy due to the stochastic sample ϵ. Even when the true KL divergence $\| d \| ^ { 2 } / ( 2 \sigma ^ { 2 } )$ is moderate, individual ratio samples can be extreme (either very large or very small), triggering spurious clipping. 

2. Noise-dependent clipping. Whether an update is clipped depends heavily on the random noise ϵ drawn during sampling, rather than the true policy divergence. Two trajectories with identical policy parameters but diferent noise realizations may receive entirely diferent clipping decisions. 

In contrast, the true KL divergence $D _ { \mathrm { K L } } ( \pi _ { \theta _ { \mathrm { o l d } } } \| \pi _ { \theta } ) = \| \pmb { d } \| ^ { 2 } / ( 2 \sigma ^ { 2 } )$ is a deterministic function of the policy parameters alone, unafected by the sampling noise. This motivates our approach: replace the noisy ratio-based trust region with a direct divergence constraint. A detailed variance analysis is provided in Appendix D. 

## 3.3 Divergence Proximal Policy Optimization for Flow Models

We now derive the divergence between old and new policies in the flow model setting and present our Flow-DPPO algorithm. 

Exact KL divergence. Since both $\pi _ { \theta _ { \mathrm { o l d } } } ( \cdot \mid x _ { t } )$ and $\pi _ { \boldsymbol { \theta } } ( \cdot \mid x _ { t } )$ are Gaussians with the same variance $\sigma ^ { 2 } \mathbf { I }$ but diferent means, the KL divergence admits the closed form (see Appendix C for derivation): 

$$
D _ {\mathrm{KL}} \big (\pi_ {\theta_ {\mathrm{old}}} (\cdot | \boldsymbol {x} _ {t}) \| \pi_ {\theta} (\cdot | \boldsymbol {x} _ {t}) \big) = \frac {\| \boldsymbol {\mu} _ {\theta_ {\mathrm{old}}} (\boldsymbol {x} _ {t} , t) - \boldsymbol {\mu} _ {\theta} (\boldsymbol {x} _ {t} , t) \| ^ {2}}{2 \sigma^ {2}}.\tag{14}
$$

For Flow-SDE (corresponding to Eq. (2)), $\sigma ^ { 2 } = \sigma _ { t } ^ { 2 } \Delta t$ , giving: 

$$
D _ {\mathrm{KL}} ^ {\mathrm{SDE}} (\pi_ {\theta_ {\mathrm{old}}} \| \pi_ {\theta}) = \frac {\Delta t}{2} \left(\frac {\sigma_ {t} (1 - t)}{2 t} + \frac {1}{\sigma_ {t}}\right) ^ {2} \| \boldsymbol {v} _ {\theta} (\boldsymbol {x} _ {t}, t) - \boldsymbol {v} _ {\theta_ {\mathrm{old}}} (\boldsymbol {x} _ {t}, t) \| ^ {2}.\tag{15}
$$

For CPS (corresponding to Eq. (3)), with $\sigma _ { \mathrm { C P S } } = ( t - \Delta t ) \sin ( \eta \pi / 2 )$ 

$$
D _ {\mathrm{KL}} ^ {\mathrm{CPS}} (\pi_ {\theta_ {\mathrm{old}}} \| \pi_ {\theta}) = \frac {\| \pmb {\mu} _ {\theta} ^ {\mathrm{CPS}} (\pmb {x} _ {t} , t) - \pmb {\mu} _ {\theta_ {\mathrm{old}}} ^ {\mathrm{CPS}} (\pmb {x} _ {t} , t) \| ^ {2}}{2 (t - \Delta t) ^ {2} \sin^ {2} (\eta \pi / 2)}.\tag{16}
$$

Remark 4. In the LLM setting, DPPO (Qi et al., 2026) must approximate the true divergence via Binary or Top-K reductions of the vocabulary distribution, as computing exact TV or KL over $| \mathcal { V } | > 1 0 0 K$ tokens is memory-prohibitive. In flow models, the Gaussian policy structure yields exact divergence at negligible cost, namely the squared diference between two forward passes of the velocity network. This makes divergence-based trust regions strictly more natural for flow models than for LLMs. 

The Flow-DPPO mask. We define the Flow-DPPO objective as: 

$$
\mathcal {L} ^ {\text { Flow - DPPO }} (\theta) = \mathbb {E} \left[ \frac {1}{G} \sum_ {i = 1} ^ {G} \frac {1}{T} \sum_ {t = 0} ^ {T - 1} \left(M _ {t} ^ {i} \cdot r _ {t} ^ {i} (\theta) \cdot \hat {A} ^ {i} - \beta D _ {\mathrm{KL}} (\pi_ {\theta} \| \pi_ {\mathrm{ref}})\right) \right],\tag{17}
$$

where the divergence-based mask is: 

$$
M _ {t} ^ {i} = \left\{ \begin{array}{l l} 0, & \text { if } \big (\hat {A} ^ {i} > 0 \text { and } r _ {t} ^ {i} > 1 \text { and } D _ {t} > \delta \big) \\ & \text { or } \big (\hat {A} ^ {i} <   0 \text { and } r _ {t} ^ {i} <   1 \text { and } D _ {t} > \delta \big), \\ 1, & \text { otherwise }, \end{array} \right.\tag{18}
$$

with $D _ { t } \equiv { \cal D } _ { \mathrm { K L } } \big ( \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot  { | } \ x _ { t } ^ { i } ) \ | | \pi _ { \theta } ( \cdot  { | } \ x _ { t } ^ { i } ) \big )$ and δ a divergence threshold. 

Asymmetric design. The mask in Eq. (18) preserves the asymmetric structure that makes PPO efective. It only blocks updates that are already moving away from the old policy: 

• When $\hat { A } ^ { i } > 0$ and $r _ { t } ^ { i } > 1$ : the gradient is pushing the policy further from $\theta _ { \mathrm { o l d } }$ (increasing an already-increased action probability). The mask blocks this if the divergence exceeds δ. 

• When $\hat { A } ^ { i } < 0$ and $r _ { t } ^ { i } < 1$ : the gradient is decreasing an already-decreased action probability, again moving away from the old policy. The mask blocks this if divergence exceeds δ. 

• In all other cases $( \hat { A } ^ { i } > 0 , r _ { t } ^ { i } < 1 \mathrm { o r } \hat { A } ^ { i } < 0 , r _ { t } ^ { i } > 1 )$ : the gradient is moving the policy towards the old policy. These beneficial updates are never blocked, regardless of the divergence level. 

This asymmetry ensures that the trust region constraint does not impede recovery: when the policy has drifted too far, corrective updates remain uninhibited. We provide a justification of this directional condition and discuss refined mask variants in Appendix E. 

## 4 Experiments

Models and Baselines. We employ Stable Difusion 3.5 Medium (Esser et al., 2024; Stability AI, 2024) (SD3.5), FLUX2-klein-base-9B (Black Forest Labs, 2026) (FLUX2-9B) and FLUX.1-dev (Black Forest Labs, 2024) as base models to cover diverse architectures and scales. We compare our method against four competitive baselines: Flow-GRPO (Liu et al., 2025), Flow-CPS (Wang and Yu, 2025), GRPO-Guard (Wang et al., 2025) and Difusion-NFT (Zheng et al., 2026). Specifically, we evaluate two variants of our approach: Flow-DPPO (using SDE sampling from Flow-GRPO) and Flow-DPPO+CPS (using CPS-scheduled SDE sampling). Detailed configurations are deferred to Appendix F. 

Metrics and Datasets. GenEval2 (Kamath et al., 2025) and PickScore (Kirstain et al., 2023) are selected as in-domain and out-of-domain (OOD) datasets, respectively. For GenEval2, we follow the oficial template to generate 20k synthetic training prompts and evaluate on the 800 oficially released prompts. To monitor catastrophic forgetting under distribution shifts, we track PickScore (Kirstain et al., 2023), CLIP (Radford et al., 2021) score, and HPSv2 (Wu et al., 2023) during training. We report results for both single-reward optimization (GenEval2 only) and multi-reward training, where GDPO (Liu et al., 2026) aggregates advantages with equal reward weights. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-01/d52561ff-9523-48be-af73-f2383cdc7168/b83628005d6e9ab0b202e0c8b20fbfb60e4755fe61e8b92901a558509d0d6fc7.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-01/d52561ff-9523-48be-af73-f2383cdc7168/a0c7159b3c752a6e1d73f79e6d7790cc2de1dde880262c65a0cc5ef22df97e2f.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-01/d52561ff-9523-48be-af73-f2383cdc7168/80d9633f932df565357a9a44a49e3ea03e495b750cfaf19e2aae5f243d36acb0.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-01/d52561ff-9523-48be-af73-f2383cdc7168/f45af04a2d67ab5c1c7e259e9fb6964664d4a1ce17eeb42d94b5d065cc205998.jpg)



Figure 2: Training curves on FLUX2-9B for single-reward setting. Flow-DPPO variants achieve state-of-the-art performance and less catastrophic forgetting on out-of-domain rewards.



Table 1: Performance comparison on SD3.5 and FLUX2-9B. The training is applied on the In-Domain (GenEval2). The Out-of-Domain (PickScore) prompts are only used for evaluation. The corresponding training curves are in Figures 4 and 8. The full version including single-reward training is in Table 4.


<table><tr><td rowspan="2">Model</td><td colspan="4">In-Domain (GenEval2)</td><td colspan="3">Out-of-Domain (PickScore)</td></tr><tr><td>GenEval2</td><td>CLIP</td><td>PickScore</td><td>HPSv2</td><td>CLIP</td><td>PickScore</td><td>HPSv2</td></tr><tr><td colspan="8">Pretrained baselines (before RL)</td></tr><tr><td>SD3.5-medium</td><td>12.4</td><td>0.250</td><td>21.00</td><td>0.213</td><td>0.244</td><td>19.99</td><td>0.210</td></tr><tr><td>FLUX2-klein-base-9B</td><td>25.4</td><td>0.281</td><td>20.92</td><td>0.228</td><td>0.254</td><td>20.05</td><td>0.230</td></tr><tr><td>FLUX.1-dev</td><td>23.3</td><td>0.297</td><td>23.26</td><td>0.315</td><td>0.276</td><td>21.91</td><td>0.304</td></tr><tr><td colspan="8">SD3.5-medium, multi-reward RL fine-tuning</td></tr><tr><td>Flow-GRPO</td><td>39.9</td><td>0.358</td><td>25.09</td><td>0.399</td><td>0.273</td><td>22.07</td><td>0.349</td></tr><tr><td>Flow-CPS</td><td>44.6</td><td>0.359</td><td>25.51</td><td>0.407</td><td>0.265</td><td>22.08</td><td>0.343</td></tr><tr><td>GRPO-Guard</td><td>47.8</td><td>0.353</td><td>25.64</td><td>0.409</td><td>0.272</td><td>22.32</td><td>0.354</td></tr><tr><td>Diffusion-NFT</td><td>42.5</td><td>0.334</td><td>25.30</td><td>0.394</td><td>0.269</td><td>22.52</td><td>0.355</td></tr><tr><td>Flow-DPPO</td><td>48.1</td><td>0.345</td><td>25.63</td><td>0.409</td><td>0.273</td><td>22.58</td><td>0.360</td></tr><tr><td>Flow-DPPO + CPS</td><td>51.6</td><td>0.369</td><td>25.72</td><td>0.415</td><td>0.279</td><td>22.51</td><td>0.361</td></tr><tr><td colspan="8">FLUX2-klein-base-9B, multi-reward RL fine-tuning</td></tr><tr><td>Flow-GRPO</td><td>46.8</td><td>0.371</td><td>25.61</td><td>0.412</td><td>0.277</td><td>22.62</td><td>0.357</td></tr><tr><td>Flow-CPS</td><td>47.1</td><td>0.361</td><td>25.70</td><td>0.416</td><td>0.276</td><td>22.85</td><td>0.364</td></tr><tr><td>GRPO-Guard</td><td>49.0</td><td>0.375</td><td>25.27</td><td>0.411</td><td>0.269</td><td>21.99</td><td>0.349</td></tr><tr><td>Diffusion-NFT</td><td>47.3</td><td>0.336</td><td>24.87</td><td>0.389</td><td>0.274</td><td>22.47</td><td>0.351</td></tr><tr><td>Flow-DPPO</td><td>57.7</td><td>0.364</td><td>25.76</td><td>0.418</td><td>0.282</td><td>22.90</td><td>0.368</td></tr><tr><td>Flow-DPPO + CPS</td><td>55.2</td><td>0.386</td><td>26.15</td><td>0.427</td><td>0.287</td><td>22.97</td><td>0.370</td></tr></table>

## 4.1 Main results

Performance and Generalization. As summarized in Table 1, Flow-DPPO variants consistently outperform all baselines across both base models and all evaluation metrics, with particularly substantial gains in the GenEval2 reward. In the single-reward setting (optimizing GenEval2 only), Figure 2 demonstrates that our proposed variants not only achieve superior performance on FLUX2-9B compared to baselines but also exhibit a more stable training trajectory. These empirical advantages persist across SD3.5 (Figure 7) and FLUX.1-dev (Figure 9). 

We attribute this superiority to the precise divergence-based mask in Flow-DPPO. By mitigating the influence of samples falling outside the trust region, which are susceptible to reward hacking, Flow-DPPO maintains a more robust optimization gradient. This constraint prevents the model from excessively exploiting individual rewards at the expense of others, thereby achieving a superior balance across multiple optimization objectives and fostering stable convergence. This is further corroborated by the multi-reward training curves in Figure 4, where Flow-DPPO variants consistently outperform all baselines across most metrics on SD3.5, without sacrificing any individual objective. 

Out-of-domain Behavior and Catastrophic Forgetting. To investigate catastrophic forgetting, we analyze OOD metrics (PickScore, CLIP, and HPSv2) and the KL divergence from the pretrained model. As illustrated in Figure 2, OOD metrics initially increase across all methods as RL optimization drives the model toward higher visual quality. However, as training progresses, these metrics decline, indicating that the model overfits the in-domain reward (GenEval2) at the expense of OOD knowledge. Notably, Flow-DPPO variants exhibit significantly less OOD degradation, suggesting that catastrophic forgetting is efectively mitigated. Qualitative results in Figure 6 further support this, demonstrating that our methods better preserve visual fidelity on OOD prompts. Consistently, Table 2 shows that Flow-DPPO variants maintain a lower KL divergence in most settings. This reduced distribution drift aligns with OOD metric trends, collectively indicating stronger resistance to reward hacking and forgetting. Ultimately, these results highlight that the divergence-based mask acts as a safety boundary, allowing the model to learn from rewards without losing its original generative quality or falling into distribution collapse. 


Figure 3: Asymmetric masking ablation on SD3.5 with single-reward on GenEval2.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-01/d52561ff-9523-48be-af73-f2383cdc7168/d13ed849f8793957e0aabe41a1e93e53399d94ba549d2a30011488c36a6ab4da.jpg)



Table 2: KL divergence $( \times 1 0 ^ { - 3 } )$ between the RL fine-tuned model and the pre-trained reference. Lower is better. Full curves in Figure 12.


<table><tr><td rowspan="2">Method</td><td colspan="3">FLUX2-9B</td><td colspan="2">SD3.5</td></tr><tr><td>Single</td><td>Multi</td><td>+CFG</td><td>Single</td><td>Multi</td></tr><tr><td colspan="6">Flow-SDE schedule</td></tr><tr><td>Flow-GRPO</td><td>0.77</td><td>0.79</td><td>1.36</td><td>2.34</td><td>3.81</td></tr><tr><td>GRPO-Guard</td><td>1.07</td><td>1.01</td><td>1.63</td><td>2.05</td><td>3.33</td></tr><tr><td>Flow-DPPO</td><td>0.17</td><td>0.49</td><td>0.51</td><td>1.16</td><td>2.49</td></tr><tr><td colspan="6">CPS schedule</td></tr><tr><td>Flow-CPS</td><td>0.24</td><td>1.66</td><td>1.51</td><td>2.41</td><td>3.18</td></tr><tr><td>Flow-DPPO + CPS</td><td>0.68</td><td>0.70</td><td>0.83</td><td>1.60</td><td>2.52</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-01/d52561ff-9523-48be-af73-f2383cdc7168/161d117de8ba140745192f0ea86009553cb995374c0c20981370017ab16dab9f.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-01/d52561ff-9523-48be-af73-f2383cdc7168/dee4d57283e8ccdb794af6b54d5161139610476316ed37089e8cbc8b7fa13848.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-01/d52561ff-9523-48be-af73-f2383cdc7168/edfbb735b4c2c06d1bbfbeb6a7ea5f96c3dafeaf8e418d367b4d53ffb2e2ed4e.jpg)



Figure 4: Training curves on SD3.5 for multi-reward setting. Flow-DPPO variants consistently outperform all baselines across all metrics.


## 4.2 Analysis

Asymmetric Masking and Divergence Threshold. We investigate the impact of the divergence threshold and asymmetric masking in Flow-DPPO using SD3.5 with CPS sampling (Figure 3). Without asymmetric masking, the training process collapses as the trust-region regularization becomes inefective; specifically, samples falling outside the trust region are largely ignored, preventing optimization progress. Conversely, asymmetric masking constrains these samples back within the trust region, thereby stabilizing the trajectory. Regarding the divergence threshold, a looser threshold $( 1 0 ^ { - 5 } )$ results in diminished stability and suboptimal convergence. A tighter threshold $( 1 0 ^ { - 7 } )$ initially slows down learning but fosters superior stability and slightly better final performance due to more rigorous trust-region enforcement. 

Multi-epoch Training and Sample Eficiency. Given the high computational cost of rollouts, we investigate how sample reuse frequency afects optimization eficiency on SD3.5. Specifically, we vary two factors: (i) the number of groups per rollout, and (ii) the number of training epochs per rollout (inner loops). The latter determines the reuse frequency of each sample. For instance, two inner loops imply that each rollout batch is utilized for two consecutive gradient steps. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-01/d52561ff-9523-48be-af73-f2383cdc7168/5311b46d3b744c8a4ebeaefdd3f3f15d2361904c162ab5d4d056fce7c138731e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-01/d52561ff-9523-48be-af73-f2383cdc7168/5f538716e8551fdd8582ef0b5cd0f13d25685588bd98c2f316149c22491da7e6.jpg)



Figure 5: Multi-epoch training on SD3.5 (Left: Flow-SDE, Right: CPS). Flow-DPPO variants show consistent long-term gains under multi-epoch training (G64-I2 and G32-I2), while baselines plateau or even degrade.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-01/d52561ff-9523-48be-af73-f2383cdc7168/3e7f928dc7c626aefb20dbba1f9ca7a1fb7bef26fcbd8aa264ee2b400ec79462.jpg)



Figure 6: Qualitative comparison on FLUX2-9B with single-reward setting and controlled seeds for each prompt at the same training iteration. Flow-DPPO and Flow-DPPO + CPS retain competitive in-domain performance with less reward hacking while exhibiting notably less catastrophic forgetting on out-of-domain prompts.


While our main experiments use 64 groups with 1 inner loop (G64-I1), we further explore two eficiency-oriented settings: G32-I2 (half the rollout computation with samples reused twice) and G64-I2 (standard rollout computation with doubled training intensity). As shown in Figure 5, baseline methods (Flow-GRPO, Flow-CPS) struggle to achieve sustained gains under multi-epoch training, often leading to performance plateaus or degradation. In contrast, Flow-DPPO variants successfully reuse rollout samples across multiple updates, yielding consistent long-term performance improvements. This advantage stems from the divergence-based mask, which constrains updates within the trust region, ensuring eficient sample utilization. This ofers a promising direction for scenarios where rollouts are computationally expensive, such as long-video generation. 

## 5 Conclusion

We show ratio clipping in flow models is a noisy, biased proxy for divergence. To address this, we propose a divergence-based mask using the exact KL at zero extra cost. Across multiple base models, sampling schedules, and reward objectives, Flow-DPPO consistently achieves superior performance than baselines in terms of reward optimization and catastrophic forgetting. Furthermore, Flow-DPPO enables stable multi-epoch training where ratio clipping degrades, ofering a promising direction for scenarios with expensive rollouts, such as long-video generation. 

## References



Michael Samuel Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. In The Eleventh International Conference on Learning Representations, 2023. 





Michael Samuel Albergo, Mark Goldstein, Nicholas Matthew Bofi, Rajesh Ranganath, and Eric Vanden-Eijnden. Stochastic interpolants with data-dependent couplings. In International Conference on Machine Learning, pages 921–937. PMLR, 2024. 





Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training difusion models with reinforcement learning. In The Twelfth International Conference on Learning Representations, 2024. 





Black Forest Labs. Flux.1: Announcing black forest labs. https://blackforestlabs.ai/announc ing-black-forest-labs/, 2024. 





Black Forest Labs. FLUX.2 [klein]: Towards interactive visual intelligence. https://bfl.ai /blog/flux2-klein-towards-interactive-visual-intelligence, 2026. Model weights: https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B. 





Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024. 





Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Reinforcement learning for fine-tuning text-to-image difusion models. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. 





Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 





Sham Kakade and John Langford. Approximately optimal approximate reinforcement learning. In Proceedings of the nineteenth international conference on machine learning, pages 267–274, 2002. 





Amita Kamath, Kai-Wei Chang, Ranjay Krishna, Luke Zettlemoyer, Yushi Hu, and Marjan Ghazvininejad. Geneval 2: Addressing benchmark drift in text-to-image evaluation. arXiv preprint arXiv:2512.16853, 2025. 





Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of difusionbased generative models. In Advances in Neural Information Processing Systems, 2022. 





Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. Advances in neural information processing systems, 36:36652–36663, 2023. 





Junzhe Li, Yutao Cui, Tao Huang, Yinping Ma, Chun Fan, Yiming Cheng, Miles Yang, Zhao Zhong, and Liefeng Bo. Mixgrpo: Unlocking flow-based grpo eficiency with mixed ode-sde. arXiv preprint arXiv:2507.21802, 2025. 





Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023. 





Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025. 





Shih-Yang Liu, Xin Dong, Ximing Lu, Shizhe Diao, Peter Belcak, Mingjie Liu, Min-Hung Chen, Hongxu Yin, Yu-Chiang Frank Wang, Kwang-Ting Cheng, et al. Gdpo: Group reward-decoupled normalization policy optimization for multi-reward rl optimization. arXiv preprint arXiv:2601.05242, 2026. 





Xingchao Liu, Chengyue Gong, and qiang liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In The Eleventh International Conference on Learning Representations, 2023. 





Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. DPM-solver: A fast ODE solver for difusion probabilistic model sampling in around 10 steps. In Advances in Neural Information Processing Systems, 2022. 





Long Ouyang, Jefrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730– 27744, 2022. 





Penghui Qi, Xiangxin Zhou, Zichen Liu, Tianyu Pang, Chao Du, Min Lin, and Wee Sun Lee. Rethinking the trust region in llm reinforcement learning. arXiv preprint arXiv:2602.04879, 2026. 





Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 





Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023. 





John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In International conference on machine learning, pages 1889–1897. PMLR, 2015. 





John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. 





Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 





Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising difusion implicit models. In International Conference on Learning Representations, 2021a. 





Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic diferential equations. In International Conference on Learning Representations, 2021b. 





Stability AI. Stable difusion 3.5. https://stability.ai/news/introducing-stable-diffusion -3-5, 2024. Model weights: https://huggingface.co/stabilityai/stable-diffusion-3.5-m edium. 





Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Difusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8228–8238, 2024. 





Feng Wang and Zihao Yu. Coeficients-preserving sampling for reinforcement learning with flow matching. arXiv preprint arXiv:2509.05952, 2025. 





Jing Wang, Jiajun Liang, Jie Liu, Henglin Liu, Gongye Liu, Jun Zheng, Wanyuan Pang, Ao Ma, Zhenyu Xie, Xintao Wang, et al. Grpo-guard: Mitigating implicit over-optimization in flow matching via regulated clipping. arXiv preprint arXiv:2510.22319, 2025. 





Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341, 2023. 





Shuchen Xue, Chongjian Ge, Shilong Zhang, Yichen Li, and Zhi-Ming Ma. Advantage weighted matching: Aligning rl with pretraining in difusion models. arXiv preprint arXiv:2509.25050, 2025a. 





Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, et al. Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025b. 





Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, and Ming-Yu Liu. DifusionNFT: Online difusion reinforcement with forward process. In The Fourteenth International Conference on Learning Representations, 2026. 



## Appendix of Flow-DPPO: Divergence Proximal Policy Optimization for Flow Matching Models

A The Flow-DPPO Algorithm 16   
B Policy Improvement Bound for Flow Models 16   
B.1 Proof of Performance Difference Identity . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . B.2 Proof of Policy Improvement Bound . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . B.3 A Tighter Policy Improvement Bound 19   
B.4 Connection to Gaussian Per-Step Divergence 20   
C KL Divergence Between Gaussian Policies 20   
C.1 General Gaussian KL Divergence 20   
C.2 Connection Between KL and TV in the Gaussian Setting 20   
C.3 Application to Flow-SDE 21   
C.4 Application to CPS 21   
D Ratio Variance Analysis 21   
E Towards a Predictive Divergence Mask 22   
E.1 Predicting Post-Update Divergence 22   
E.2 The Predictive Mask 23   
E.3 Discussion on Mask Variants 24   
F Experimental Details 24   
F.1 Computational Resources. 24   
F.2 Hyperparameters. 24   
G Additional Experimental Results 25   
G.1 Additional Training Curves 25   
G.2 KL Divergence Curves 26   
G.3 Ablation Studies 27 

## Appendix A. The Flow-DPPO Algorithm

We summarize the complete Flow-DPPO training procedure. The algorithm adopts the CPS sampling framework (Wang and Yu, 2025) for trajectory generation, uses group-relative advantage estimation, and applies the divergence-based mask during policy optimization. 

Algorithm 1 Flow-DPPO Training
1: Input: Flow model $\boldsymbol{v}_{\theta}$ , reference model $\boldsymbol{v}_{\text{ref}}$ , reward function $R$ , prompts $\mathcal{C}$ 2: Hyperparameters: group size $G$ , divergence threshold $\delta$ , KL coefficient $\beta$ , stochasticity $\eta$ 3: for each training iteration do
4:    Sample prompts $\{c_j\} \sim \mathcal{C}$ 5:    // Rollout phase (with $\theta_{\text{old}}$ )
6:    for each prompt $c_j$ do
7:    Generate $G$ trajectories $\{(\boldsymbol{x}_T^i, \ldots, \boldsymbol{x}_0^i)\}_{i=1}^G$ via CPS (Eq. 3) using $\boldsymbol{v}_{\theta_{\text{old}}}$ 8:    Record log-probabilities $\log p_{\theta_{\text{old}}}(\boldsymbol{x}_{t-\Delta t}^i | \boldsymbol{x}_t^i)$ and means $\boldsymbol{\mu}_{\theta_{\text{old}}}(\boldsymbol{x}_t^i, t)$ 9:    Compute rewards $R(\boldsymbol{x}_0^i, \boldsymbol{c}_j)$ and advantages $\hat{A}^i$ 10:    end for
11:    // Policy optimization phase
12:    for each gradient step do
13:    Compute current means $\boldsymbol{\mu}_{\theta}(\boldsymbol{x}_t^i, t)$ via forward pass of $\boldsymbol{v}_{\theta}$ 14:    Compute divergence $D_t = \| \boldsymbol{\mu}_{\theta_{\text{old}}}(\boldsymbol{x}_t^i, t) - \boldsymbol{\mu}_{\theta}(\boldsymbol{x}_t^i, t) \|^2$ 15:    Compute ratio $r_t^i(\theta)$ from log-probabilities
16:    Compute mask $M_t^i$ (Eq. 18)
17:    Update $\theta$ by maximizing $\mathcal{L}^{\text{Flow-DPPO}}$ (Eq. 17)
18:    end for
19: $\theta_{\text{old}} \leftarrow \theta$ 20: end for 

Computational overhead. The divergence computation requires one additional forward pass of the velocity network to obtain $\mu _ { \theta } ( x _ { t } ^ { i } , t )$ at training time. However, this forward pass is already required for computing the log ratio, so the divergence $D _ { t } = \| \pmb { \mu } _ { \theta _ { \mathrm { o l d } } } - \pmb { \mu } _ { \theta } \| ^ { 2 }$ comes at zero additional cost: it is simply the squared norm of a diference that is already computed. 

## Appendix B. Policy Improvement Bound for Flow Models

We adapt the classical policy improvement theory (Kakade and Langford, 2002; Schulman et al., 2015) to the finite-horizon, undiscounted setting of flow model denoising, following the approach of Qi et al. (2026) for the LLM regime. We use the MDP notation introduced in Section 2.1: K − 1 decision steps indexed by $k \in \{ 1 , \ldots , K - 1 \}$ , states $\pmb { s } _ { k } = ( \pmb { c } , t _ { k } , \pmb { x } _ { t _ { k } } )$ , actions $\mathbf { \Delta } \mathbf { a } _ { k } = \mathbf { \Delta } \mathbf { x } _ { t _ { k + 1 } }$ , and terminal reward $R ( \pmb { x } _ { 0 } , \pmb { c } )$ 

## B.1 Proof of Performance Diference Identity

Proof [Proof of Theorem 1] We begin by expressing the performance diference via its definition. Since the reward is only a function of the terminal state $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ and the prompt $^ { c , }$ we have: 

$$
\begin{array}{r l r} & & {J (\pi_ {\theta}) - J (\pi_ {\theta_ {\mathrm{old}}}) = \mathbb {E} _ {\tau \sim \pi_ {\theta}} [ R (\pmb {x} _ {0}, \pmb {c}) ] - \mathbb {E} _ {\tau \sim \pi_ {\theta_ {\mathrm{old}}}} [ R (\pmb {x} _ {0}, \pmb {c}) ]} \\ & & {= \int \big (\pi_ {\theta} (\tau | \pmb {c}) - \pi_ {\theta_ {\mathrm{old}}} (\tau | \pmb {c}) \big) R (\pmb {x} _ {0}, \pmb {c}) \mathrm{d} \tau ,} \end{array}
$$

where the integral is over all trajectories $\tau = ( \mathbf { a } _ { 1 } , \dots , \mathbf { a } _ { K - 1 } )$ (we omit the deterministic transition structure for notational clarity). 

The core of the proof is the telescoping identity for the diference in trajectory probabilities. Since $\begin{array} { r } { \pi _ { \theta } ( \tau ~ \mathbf { \theta } | ~ \mathbf { c } ) ~ = ~ \prod _ { k = 1 } ^ { K - 1 } \pi _ { \theta } ( \mathbf { a } _ { k } ~ \mathbf { \theta } | ~ \mathbf { s } _ { k } ) } \end{array}$ , we apply the algebraic identity $\begin{array} { r } { \prod _ { k = 1 } ^ { N } a _ { k } - \prod _ { k = 1 } ^ { N } b _ { k } \ = } \end{array}$ $\begin{array} { r } { \sum _ { k = 1 } ^ { N } \big ( \prod _ { j = 1 } ^ { k - 1 } b _ { j } \big ) \big ( a _ { k } - \dot { b _ { k } } \big ) \big ( \prod _ { j = k + 1 } ^ { N } a _ { j } \big ) } \end{array}$ : 

$$
\pi_ {\theta} (\tau \mid \boldsymbol {c}) - \pi_ {\theta_ {\text {old}}} (\tau \mid \boldsymbol {c}) = \sum_ {k = 1} ^ {K - 1} \left(\prod_ {j = 1} ^ {k - 1} \pi_ {\theta_ {\text {old}}} (\boldsymbol {a} _ {j} \mid \boldsymbol {s} _ {j})\right) \cdot \left(\pi_ {\theta} (\boldsymbol {a} _ {k} \mid \boldsymbol {s} _ {k}) - \pi_ {\theta_ {\text {old}}} (\boldsymbol {a} _ {k} \mid \boldsymbol {s} _ {k})\right) \left(\prod_ {j = k + 1} ^ {K - 1} \pi_ {\theta} (\boldsymbol {a} _ {j} \mid \boldsymbol {s} _ {j})\right).
$$

Substituting into the performance diference and converting to an expectation under $\pi _ { \boldsymbol { \theta } _ { \mathrm { o l d } } } \colon$ 

$$
J (\pi_ {\theta}) - J (\pi_ {\theta_ {\mathrm{old}}}) = \mathbb {E} _ {\tau \sim \pi_ {\theta_ {\mathrm{old}}}} \left[ R (\boldsymbol {x} _ {0}, \boldsymbol {c}) \sum_ {k = 1} ^ {K - 1} \left(\frac {\pi_ {\theta} (\boldsymbol {a} _ {k} \mid \boldsymbol {s} _ {k})}{\pi_ {\theta_ {\mathrm{old}}} (\boldsymbol {a} _ {k} \mid \boldsymbol {s} _ {k})} - 1\right) \left(\prod_ {j = k + 1} ^ {K - 1} \frac {\pi_ {\theta} (\boldsymbol {a} _ {j} \mid \boldsymbol {s} _ {j})}{\pi_ {\theta_ {\mathrm{old}}} (\boldsymbol {a} _ {j} \mid \boldsymbol {s} _ {j})}\right) \right].
$$

We decompose this expression by adding and subtracting the term where the future ratio product is set to 1: 

$$
\begin{array}{l} J (\pi_ {\theta}) - J (\pi_ {\theta_ {\mathrm{old}}}) = \underbrace {\mathbb {E} _ {\tau \sim \pi_ {\theta_ {\mathrm{old}}}} \left[ R (\boldsymbol {x} _ {0} , \boldsymbol {c}) \sum_ {k = 1} ^ {K - 1} \left(\frac {\pi_ {\theta} (\boldsymbol {a} _ {k} \mid \boldsymbol {s} _ {k})}{\pi_ {\theta_ {\mathrm{old}}} (\boldsymbol {a} _ {k} \mid \boldsymbol {s} _ {k})} - 1\right) \right]} _ {L _ {\theta_ {\mathrm{old}}} ^ {\prime} (\pi_ {\theta})} \\ - \underbrace {\mathbb {E} _ {\tau \sim \pi_ {\theta_ {\mathrm{old}}}} \left[ R (\boldsymbol {x} _ {0} , \boldsymbol {c}) \sum_ {k = 1} ^ {K - 1} \left(\frac {\pi_ {\theta} (\boldsymbol {a} _ {k} \mid \boldsymbol {s} _ {k})}{\pi_ {{\theta_ {\mathrm{old}}}} (\boldsymbol {a} _ {k} \mid \boldsymbol {s} _ {k})} - 1\right) \left(1 - \prod_ {j = k + 1} ^ {K - 1} \frac {\pi_ {\theta} (\boldsymbol {a} _ {j} \mid \boldsymbol {s} _ {j})}{\pi_ {{\theta_ {\mathrm{old}}}} (\boldsymbol {a} _ {j} \mid \boldsymbol {s} _ {j})}\right) \right]} _ {\Delta (\pi_ {\theta_ {\mathrm{old}}}, \pi_ {\theta})}. \end{array}
$$

This completes the proof. 

## B.2 Proof of Policy Improvement Bound

Lemma 5 (Bound on Trajectory-Level TV Divergence). Let $\pi _ { \boldsymbol { \theta } _ { \mathrm { o l d } } }$ and $\pi _ { \theta }$ be two policies for the flow model MDP. Let $\pi _ { \theta _ { \mathrm { o l d } } , > k } ( { \cdot \ } | \ s _ { k + 1 } )$ and $\pi _ { \boldsymbol { \theta } , > k } ( . ~ \vert ~ \vert ~ \boldsymbol { s } _ { k + 1 } )$ denote the distributions over future sub-trajectories $( { \pmb a } _ { k + 1 } , \dots , { \pmb a } _ { K - 1 } )$ starting from state $s _ { k + 1 }$ . Then: 

$$
D _ {\mathrm{TV}} \big (\pi_ {\theta_ {\mathrm{old}}, > k} (\cdot | \boldsymbol {s} _ {k + 1}) \| \pi_ {\theta , > k} (\cdot | \boldsymbol {s} _ {k + 1}) \big) \leq \sum_ {j = k + 1} ^ {K - 1} \mathbb {E} _ {\boldsymbol {s} _ {j} \sim \pi_ {\theta_ {\mathrm{old}}}} \left[ D _ {\mathrm{TV}} \big (\pi_ {\theta_ {\mathrm{old}}} (\cdot | \boldsymbol {s} _ {j}) \| \pi_ {\theta} (\cdot | \boldsymbol {s} _ {j}) \big) \right],
$$

where the expectation is over states visited under $\pi _ { \boldsymbol { \theta } _ { \mathrm { o l d } } }$ starting from $s _ { k + 1 }$ 

Proof Let $P ( \tau _ { > k } ) = \pi _ { \theta _ { \mathrm { o l d } } , > k } ( \tau _ { > k } \mid s _ { k + 1 } )$ and $Q ( \tau _ { > k } ) = \pi _ { \boldsymbol { \theta } , > k } ( \tau _ { > k } \mid s _ { k + 1 } )$ , where $\tau _ { > k } = ( { \pmb a } _ { k + 1 } , \dots , { \pmb a } _ { K - 1 } )$ We have: 

$$
2 D _ {\mathrm{TV}} (P \| Q) = \int | P (\tau_ {> k}) - Q (\tau_ {> k}) | \mathrm{d} \tau_ {> k} = \int \left| \prod_ {j = k + 1} ^ {K - 1} \pi_ {\theta_ {\mathrm{old}}} (\boldsymbol {a} _ {j} \mid \boldsymbol {s} _ {j}) - \prod_ {j = k + 1} ^ {K - 1} \pi_ {\theta} (\boldsymbol {a} _ {j} \mid \boldsymbol {s} _ {j}) \right| \mathrm{d} \tau_ {> k}.
$$

Applying the telescoping identity $\begin{array} { r } { | a _ { 1 } \cdot \cdot \cdot a _ { N } - b _ { 1 } \cdot \cdot \cdot b _ { N } | \leq \sum _ { j = 1 } ^ { N } \big ( \prod _ { i = 1 } ^ { j - 1 } a _ { i } \big ) | a _ { j } - b _ { j } | \big ( \prod _ { i = j + 1 } ^ { N } b _ { i } \big ) } \end{array}$ (which follows from the triangle inequality) and integrating: 

$$
2 D _ {\mathrm{TV}} (P \| Q) \leq \sum_ {j = k + 1} ^ {K - 1} \int \left(\prod_ {i = k + 1} ^ {j - 1} \pi_ {\theta_ {\mathrm{old}}} (\boldsymbol {a} _ {i} \mid \boldsymbol {s} _ {i})\right) | \pi_ {\theta_ {\mathrm{old}}} (\boldsymbol {a} _ {j} \mid \boldsymbol {s} _ {j}) - \pi_ {\theta} (\boldsymbol {a} _ {j} \mid \boldsymbol {s} _ {j}) | \left(\prod_ {i = j + 1} ^ {K - 1} \pi_ {\theta} (\boldsymbol {a} _ {i} \mid \boldsymbol {s} _ {i})\right) \mathrm{d} \tau_ {> k}.
$$

For each term indexed by j, integrating out the future actions $\pmb { a } _ { j + 1 } , \dotsc , \pmb { a } _ { K - 1 }$ yields 1 (since $\pi _ { \theta }$ is normalized), leaving: 

$$
2 D _ {\mathrm{TV}} (P \| Q) \leq \sum_ {j = k + 1} ^ {K - 1} \int \left(\prod_ {i = k + 1} ^ {j - 1} \pi_ {\theta_ {\text { old }}} (\boldsymbol {a} _ {i} \mid \boldsymbol {s} _ {i})\right) \left(\int | \pi_ {\theta_ {\text { old }}} (\boldsymbol {a} _ {j} \mid \boldsymbol {s} _ {j}) - \pi_ {\theta} (\boldsymbol {a} _ {j} \mid \boldsymbol {s} _ {j}) |   \mathrm{d} \boldsymbol {a} _ {j}\right) \mathrm{d} \boldsymbol {a} _ {k + 1} \dots \mathrm{d} \boldsymbol {a} _ {j - 1}.
$$

The inner integral is $2 D _ { \mathrm { T V } } ( \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot \mid s _ { j } ) \| \pi _ { \theta } ( \cdot \mid s _ { j } ) )$ , and the outer integral defines an expectation over states $s _ { j }$ under policy $\pi _ { \theta _ { \mathrm { o l d } } }$ . Thus: 

$$
D _ {\mathrm{TV}} (P \| Q) \leq \sum_ {j = k + 1} ^ {K - 1} \mathbb {E} _ {\pmb {s} _ {j} \sim \pi_ {\theta_ {\mathrm{old}}}} \left[ D _ {\mathrm{TV}} \big (\pi_ {\theta_ {\mathrm{old}}} (\cdot | \pmb {s} _ {j}) \| \pi_ {\theta} (\cdot | \pmb {s} _ {j}) \big) \right].
$$

Proof [Proof of Theorem 2] From Theorem 1, we start with the exact performance diference identity: 

$$
J (\pi_ {\theta}) - J (\pi_ {\theta_ {\mathrm{old}}}) = L _ {\theta_ {\mathrm{old}}} ^ {\prime} (\pi_ {\theta}) - \Delta (\pi_ {\theta_ {\mathrm{old}}}, \pi_ {\theta}).
$$

Our goal is to upper-bound $| \Delta ( \pi _ { \theta _ { \mathrm { o l d } } } , \pi _ { \theta } ) |$ . We begin by bounding the reward by its maximum absolute value $\xi = \operatorname* { m a x } _ { { \pmb x } _ { 0 } , { \pmb c } } | R ( { \pmb x } _ { 0 } , { \pmb c } ) |$ : 

$$
\begin{array}{l} | \Delta (\pi_ {\theta_ {\mathrm{old}}}, \pi_ {\theta}) | \\ \leq \xi \cdot \mathbb {E} _ {\tau \sim \pi_ {\theta_ {\mathrm{old}}}} \left[ \sum_ {k = 1} ^ {K - 1} \left| \frac {\pi_ {\theta} (\boldsymbol {a} _ {k} \mid \boldsymbol {s} _ {k})}{\pi_ {\theta_ {\mathrm{old}}} (\boldsymbol {a} _ {k} \mid \boldsymbol {s} _ {k})} - 1 \right| \cdot \left| 1 - \prod_ {j = k + 1} ^ {K - 1} \frac {\pi_ {\theta} (\boldsymbol {a} _ {j} \mid \boldsymbol {s} _ {j})}{\pi_ {\theta_ {\mathrm{old}}} (\boldsymbol {a} _ {j} \mid \boldsymbol {s} _ {j})} \right| \right] \\ = \xi \cdot \sum_ {k = 1} ^ {K - 1} \mathbb {E} _ {\boldsymbol {s} _ {\leq k} \sim \pi_ {\theta_ {\mathrm{old}}}} \left[ \left| \frac {\pi_ {\theta} (\boldsymbol {a} _ {k} \mid \boldsymbol {s} _ {k})}{\pi_ {\theta_ {\mathrm{old}}} (\boldsymbol {a} _ {k} \mid \boldsymbol {s} _ {k})} - 1 \right| \cdot \mathbb {E} _ {\tau_ {> k} \sim \pi_ {\theta_ {\mathrm{old}}}} \left[ \left| 1 - \frac {\pi_ {\theta , > k} (\tau_ {> k} \mid \boldsymbol {s} _ {k + 1})}{\pi_ {\theta_ {\mathrm{old}} , > k} (\tau_ {> k} \mid \boldsymbol {s} _ {k + 1})} \right| \right] \right]. \end{array}\tag{19}
$$

The inner expectation over future sub-trajectories is exactly twice the TV divergence between future trajectory distributions: 

$$
\mathbb {E} _ {\tau_ {> k} \sim \pi_ {\theta_ {\mathrm{old}}}} \left[ \left| 1 - \frac {\pi_ {\theta , > k} (\tau_ {> k} \mid \pmb {s} _ {k + 1})}{\pi_ {\theta_ {\mathrm{old}} , > k} (\tau_ {> k} \mid \pmb {s} _ {k + 1})} \right| \right] = 2 D _ {\mathrm{TV}} \bigl (\pi_ {\theta_ {\mathrm{old}}, > k} (\cdot \mid \pmb {s} _ {k + 1}) \| \pi_ {\theta , > k} (\cdot \mid \pmb {s} _ {k + 1}) \bigr).
$$

Applying Theorem 5 and bounding each term by $D _ { \mathrm { T V } } ^ { \mathrm { m a x } }$ 

$$
D _ {\mathrm{TV}} \big (\pi_ {\theta_ {\mathrm{old}}, > k} (\cdot | \boldsymbol {s} _ {k + 1}) \| \pi_ {\theta , > k} (\cdot | \boldsymbol {s} _ {k + 1}) \big) \leq (K - 1 - k) D _ {\mathrm{TV}} ^ {\max} (\pi_ {\theta_ {\mathrm{old}}} \| \pi_ {\theta}).
$$

Substituting back into Eq. 19: 

$$
\begin{array}{l} | \Delta (\pi_ {\theta_ {\mathrm{old}}}, \pi_ {\theta}) | \leq \xi \cdot \sum_ {k = 1} ^ {K - 1} \mathbb {E} _ {\boldsymbol {s} _ {k} \sim \pi_ {\theta_ {\mathrm{old}}}} \left[ \mathbb {E} _ {\boldsymbol {a} _ {k} \sim \pi_ {\theta_ {\mathrm{old}}} (\cdot | \boldsymbol {s} _ {k})} \left[ \left| \frac {\pi_ {\theta} (\boldsymbol {a} _ {k} \mid \boldsymbol {s} _ {k})}{\pi_ {\theta_ {\mathrm{old}}} (\boldsymbol {a} _ {k} \mid \boldsymbol {s} _ {k})} - 1 \right| \right] \right] \cdot 2 (K - 1 - k) D _ {\mathrm{TV}} ^ {\max} \\ = 2 \xi \cdot D _ {\mathrm{TV}} ^ {\max} \sum_ {k = 1} ^ {K - 1} (K - 1 - k) \cdot \mathbb {E} _ {\boldsymbol {s} _ {k} \sim \pi_ {\theta_ {\mathrm{old}}}} \left[ 2 D _ {\mathrm{TV}} \big (\pi_ {\theta_ {\mathrm{old}}} (\cdot \mid \boldsymbol {s} _ {k}) \| \pi_ {\theta} (\cdot \mid \boldsymbol {s} _ {k}) \big) \right] \\ \leq 2 \xi \cdot D _ {\mathrm{TV}} ^ {\max} \sum_ {k = 1} ^ {K - 1} (K - 1 - k) \cdot 2 D _ {\mathrm{TV}} ^ {\max} \\ = 4 \xi \cdot D _ {\mathrm{TV}} ^ {\max 2} \sum_ {k = 1} ^ {K - 1} (K - 1 - k). \end{array}
$$

Evaluating the sum: $\begin{array} { r } { \sum _ { k = 1 } ^ { K - 1 } ( K - 1 - k ) = \sum _ { m = 0 } ^ { K - 2 } m = \frac { ( K - 1 ) ( K - 2 ) } { 2 } } \end{array}$ . Therefore: 

$$
| \Delta (\pi_ {\theta_ {\mathrm{old}}}, \pi_ {\theta}) | \leq 4 \xi \cdot \frac {(K - 1) (K - 2)}{2} \cdot D _ {\mathrm{TV}} ^ {\max 2} = 2 \xi (K - 1) (K - 2) \cdot D _ {\mathrm{TV}} ^ {\max 2}.
$$

Substituting into the performance diference identity yields the desired bound: 

$$
J (\pi_ {\theta}) - J (\pi_ {\theta_ {\mathrm{old}}}) \geq L _ {\theta_ {\mathrm{old}}} ^ {\prime} (\pi_ {\theta}) - 2 \xi (K - 1) (K - 2) \cdot D _ {\mathrm{TV}} ^ {\max} (\pi_ {\theta_ {\mathrm{old}}} \| \pi_ {\theta}) ^ {2}.
$$

This completes the proof. 

## B.3 A Tighter Policy Improvement Bound

The quadratic dependence on the horizon $K ^ { 2 }$ in Theorem 2 can be overly pessimistic. By exploiting the fact that $D _ { \mathrm { T V } } \leq 1$ , we derive a tighter bound that is linear in K. 

Starting from the intermediate step in Eq. 19, the inner expectation is $2 D _ { \mathrm { T V } } ( \pi _ { \theta _ { \mathrm { o l d } } , > k } ( \cdot \ | \ s _ { k + 1 } ) | | \pi _ { \theta , > k } ( \cdot \ | $ $s _ { k + 1 } ) )$ . Instead of applying Theorem 5, we directly use the universal bound $D _ { \mathrm { T V } } \leq 1$ 

$$
\begin{array}{r l} & {| \Delta (\pi_ {\theta_ {\mathrm{old}}}, \pi_ {\theta}) | \leq \xi \cdot \sum_ {k = 1} ^ {K - 1} \mathbb {E} _ {\boldsymbol {s} _ {k} \sim \pi_ {\theta_ {\mathrm{old}}}} \left[ \mathbb {E} _ {\boldsymbol {a} _ {k} \sim \pi_ {\theta_ {\mathrm{old}}} (\cdot | \boldsymbol {s} _ {k})} \left[ \left| \frac {\pi_ {\theta} (\boldsymbol {a} _ {k} \mid \boldsymbol {s} _ {k})}{\pi_ {\theta_ {\mathrm{old}}} (\boldsymbol {a} _ {k} \mid \boldsymbol {s} _ {k})} - 1 \right| \right] \right] \cdot 2} \\ & {\qquad = 4 \xi \cdot \mathbb {E} _ {\tau \sim \pi_ {\theta_ {\mathrm{old}}}} \left[ \sum_ {k = 1} ^ {K - 1} D _ {\mathrm{TV}} (\pi_ {\theta_ {\mathrm{old}}} (\cdot \mid \boldsymbol {s} _ {k}) \| \pi_ {\theta} (\cdot \mid \boldsymbol {s} _ {k})) \right].} \end{array}
$$

Combining both bounds, the policy improvement satisfies the composite guarantee: 

$$
J (\pi_ {\theta}) - J (\pi_ {\theta_ {\mathrm{old}}}) \geq L _ {\theta_ {\mathrm{old}}} ^ {\prime} (\pi_ {\theta}) - \min \left(2 \xi (K - 1) (K - 2) \cdot D _ {\mathrm{TV}} ^ {\max 2}, 4 \xi \cdot \mathbb {E} _ {\tau \sim \pi_ {\theta_ {\mathrm{old}}}} \left[ \sum_ {k = 1} ^ {K - 1} D _ {\mathrm{TV}, k} \right]\right),
$$

where $D _ { \mathrm { T V } , k } = D _ { \mathrm { T V } } ( \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot \ | \ s _ { k } ) | | \pi _ { \theta } ( \cdot \ | \ s _ { k } ) )$ . The quadratic bound is tighter for small policy changes, while the linear bound is tighter for larger updates or longer horizons. 

## B.4 Connection to Gaussian Per-Step Divergence

For the Gaussian policies in Eq. 4, $\pi _ { \boldsymbol { \theta } _ { \mathrm { o l d } } } ( \cdot  { | } \ s _ { k } ) = \mathcal { N } (  { \pmb { \mu } } _ { \theta _ { \mathrm { o l d } } } , \sigma ^ { 2 } ( t _ { k } ) { \bf I } )$ and $\pi _ { \boldsymbol { \theta } } ( \cdot \mid s _ { k } ) = \mathcal { N } ( \mu _ { \boldsymbol { \theta } } , \sigma ^ { 2 } ( t _ { k } ) \mathbf { I } )$ 2 the TV divergence admits the closed form: 

$$
D _ {\mathrm{TV}} \big (\pi_ {\theta_ {\mathrm{old}}} (\cdot | \boldsymbol {s} _ {k}) \| \pi_ {\theta} (\cdot | \boldsymbol {s} _ {k}) \big) = 2 \Phi \left(\frac {\| \boldsymbol {\mu} _ {\theta_ {\mathrm{old}}} - \boldsymbol {\mu} _ {\theta} \|}{2 \sigma (t _ {k})}\right) - 1,
$$

where $\Phi$ is the standard normal CDF. Since Φ is strictly monotonically increasing, the TV constraint $D _ { \mathrm { T V } } ^ { \mathrm { m a x } } \leq \delta$ is equivalent to: 

$$
\max _ {\boldsymbol {s} _ {k}} \| \boldsymbol {\mu} _ {\theta_ {\mathrm{old}}} (\boldsymbol {x} _ {t _ {k}}, t _ {k}, \boldsymbol {c}) - \boldsymbol {\mu} _ {\theta} (\boldsymbol {x} _ {t _ {k}}, t _ {k}, \boldsymbol {c}) \| ^ {2} \leq 4 \sigma^ {2} (t _ {k}) \left[ \Phi^ {- 1} \bigg (\frac {1 + \delta}{2} \bigg) \right] ^ {2} =: \delta^ {\prime}.
$$

This formally establishes that the Flow-DPPO mask, which blocks updates when $\| \pmb { \mu } _ { \theta _ { \mathrm { o l d } } } - \pmb { \mu } _ { \theta } \| ^ { 2 } > \delta$ implements a trust-region constraint equivalent (up to a monotone rescaling) to constraining the per-step TV divergence. The policy improvement bound (Theorem 2) thus provides a rigorous theoretical guarantee for Flow-DPPO: by enforcing a per-step divergence threshold, the penalty term remains controlled, ensuring monotonic policy improvement. 

## Appendix C. KL Divergence Between Gaussian Policies

In this section, we derive the KL divergence between old and new policies in flow models and establish its connection to the TV divergence used in the policy improvement bound. 

## C.1 General Gaussian KL Divergence

Let $p = \mathcal { N } ( \mu _ { 1 } , \sigma ^ { 2 } \mathbf { I } )$ and $\ v q = \mathcal { N } ( \mu _ { 2 } , \sigma ^ { 2 } \mathbf { I } )$ be two isotropic Gaussians in $\mathbb { R } ^ { d }$ with the same covariance. The KL divergence is: 

$$
D _ {\mathrm{KL}} (p \| q) = \frac {1}{2 \sigma^ {2}} \left[ 2 (\boldsymbol {\mu} _ {1} - \boldsymbol {\mu} _ {2}) ^ {\top} \underbrace {\mathbb {E} _ {p} [ \boldsymbol {x} - \boldsymbol {\mu} _ {1} ]} _ {= \boldsymbol {0}} + \| \boldsymbol {\mu} _ {1} - \boldsymbol {\mu} _ {2} \| ^ {2} \right] = \frac {\| \boldsymbol {\mu} _ {1} - \boldsymbol {\mu} _ {2} \| ^ {2}}{2 \sigma^ {2}}.\tag{20}
$$

Note that this is symmetric in the means: $D _ { \mathrm { K L } } ( p \Vert q ) = D _ { \mathrm { K L } } ( q \Vert p )$ when the covariances are identical. 

## C.2 Connection Between KL and TV in the Gaussian Setting

For the same pair of Gaussians, the TV divergence is: 

$$
D _ {\mathrm{TV}} (p, q) = 2 \Phi \left(\frac {\| \boldsymbol {\mu} _ {1} - \boldsymbol {\mu} _ {2} \|}{2 \sigma}\right) - 1.
$$

Since both KL and TV are monotone functions of the single quantity $\| \pmb { \mu } _ { 1 } - \pmb { \mu } _ { 2 } \| / \sigma$ , thresholding one is equivalent to thresholding the other. Specifically, the constraint $D _ { \mathrm { T V } } \leq \delta _ { \mathrm { T V } }$ is equivalent to $\begin{array} { r } { \| \mu _ { 1 } - \mu _ { 2 } \| ^ { 2 } \leq 4 \sigma ^ { 2 } [ \Phi ^ { - 1 } ( ( 1 + \delta _ { \mathrm { T V } } ) / 2 ) ] ^ { 2 } } \end{array}$ , which in turn is equivalent to $D _ { \mathrm { K L } } \leq 2 [ \Phi ^ { - 1 } ( ( 1 + \delta _ { \mathrm { T V } } ) / 2 ) ] ^ { 2 }$ This shows that the squared $\ell _ { 2 }$ distance $\| \pmb { \mu } _ { \theta _ { \mathrm { o l d } } } - \pmb { \mu } _ { \theta } \| ^ { 2 }$ used in our mask is a unified divergence measure equivalent (up to monotone transformations) to both KL and TV divergences. 

## C.3 Application to Flow-SDE

For Flow-SDE (Eq. 2), the per-step policy is $\pi _ { \boldsymbol { \theta } } ( \mathbf { x } _ { t - \Delta t } \mid \mathbf { x } _ { t } ) = \mathcal { N } ( \boldsymbol { \mu } _ { \boldsymbol { \theta } } , \sigma _ { t } ^ { 2 } \Delta t \cdot \mathbf { I } )$ where: 

$$
\pmb {\mu} _ {\theta} (\pmb {x} _ {t}, t) = \pmb {x} _ {t} + \left[ \pmb {v} _ {\theta} (\pmb {x} _ {t}, t) + \frac {\sigma_ {t} ^ {2}}{2 t} \big (\pmb {x} _ {t} + (1 - t) \pmb {v} _ {\theta} (\pmb {x} _ {t}, t) \big) \right] \Delta t.
$$

The diference in means is: 

$$
\boldsymbol {\mu} _ {\theta} - \boldsymbol {\mu} _ {\theta_ {\mathrm{old}}} = \left(1 + \frac {\sigma_ {t} ^ {2} (1 - t)}{2 t}\right) \Delta t \cdot (\boldsymbol {v} _ {\theta} - \boldsymbol {v} _ {\theta_ {\mathrm{old}}}).
$$

Substituting into Eq. 20 with $\sigma ^ { 2 } = \sigma _ { t } ^ { 2 } \Delta t \colon$ 

$$
D _ {\mathrm{KL}} ^ {\mathrm{SDE}} (\pi_ {\theta_ {\mathrm{old}}} \| \pi_ {\theta}) = \frac {\Delta t}{2} \left(\frac {1}{\sigma_ {t}} + \frac {\sigma_ {t} (1 - t)}{2 t}\right) ^ {2} \| \boldsymbol {v} _ {\theta} (\boldsymbol {x} _ {t}, t) - \boldsymbol {v} _ {\theta_ {\mathrm{old}}} (\boldsymbol {x} _ {t}, t) \| ^ {2}.\tag{21}
$$

## C.4 Application to CPS

For CPS (Eq. 3), the policy mean is $\pmb { \mu } _ { \theta } ^ { \mathrm { C P S } } = ( 1 - ( t - \Delta t ) ) \hat { \pmb { x } } _ { 0 } + ( t - \Delta t ) \cos ( \eta \pi / 2 ) \hat { \pmb { x } } _ { 1 }$ and the variance is $\sigma _ { \mathrm { C P S } } ^ { 2 } = ( t - \Delta t ) ^ { 2 } \sin ^ { 2 } ( \eta \pi / 2 )$ . Using $\hat { \pmb { x } } _ { 0 } = \pmb { x } _ { t } - t \pmb { v } _ { \theta }$ and $\hat { { \pmb x } } _ { 1 } = { \pmb x } _ { t } + ( 1 - t ) { \pmb v } _ { \theta }$ , the diference in means is: 

$$
\boldsymbol {\mu} _ {\theta} ^ {\mathrm{CPS}} - \boldsymbol {\mu} _ {\theta_ {\text {old}}} ^ {\mathrm{CPS}} = [ - (1 - (t - \Delta t)) t + (t - \Delta t) (1 - t) \cos (\eta \pi / 2) ] (\boldsymbol {v} _ {\theta} - \boldsymbol {v} _ {\theta_ {\text {old}}}).
$$

Let $c ( t ) = - ( 1 - ( t - \Delta t ) ) t + ( t - \Delta t ) ( 1 - t ) \cos ( \eta \pi / 2 )$ . Then: 

$$
D _ {\mathrm{KL}} ^ {\mathrm{CPS}} (\pi_ {\theta_ {\mathrm{old}}} \| \pi_ {\theta}) = \frac {c (t) ^ {2} \| \pmb {v} _ {\theta} - \pmb {v} _ {\theta_ {\mathrm{old}}} \| ^ {2}}{2 (t - \Delta t) ^ {2} \sin^ {2} (\eta \pi / 2)}.\tag{22}
$$

In previous work (Wang and Yu, 2025), the $2 \sigma _ { \mathrm { C P S } } ^ { 2 }$ normalization is dropped for numerical stability, reducing the divergence to $D ( \pi _ { \theta _ { \mathrm { o l d } } } \| \pi _ { \theta } ) = \| \pmb { \mu } _ { \theta _ { \mathrm { o l d } } } ^ { \mathrm { C P S } } - \pmb { \mu } _ { \theta } ^ { \mathrm { C P S } } \| ^ { 2 }$ . We instead retain the full normalization in Eq. 22: because $\sigma _ { \mathrm { C P S } } ^ { 2 } \propto ( t - \Delta t ) ^ { 2 }$ shrinks at later denoising steps, the $\sigma _ { \mathrm { C P S } } ^ { - 2 }$ factor amplifies the divergence where small velocity changes most afect the output, yielding a tighter constraint that prevents distribution collapse. 

## Appendix D. Ratio Variance Analysis

We provide a detailed analysis of the variance of the log-ratio in flow models. 

From Eq. 13, log $r _ { t } ^ { i } = \epsilon ^ { \top } d / \sigma - \| d \| ^ { 2 } / ( 2 \sigma ^ { 2 } )$ , where d $= \mu _ { \theta } - \mu _ { \theta _ { \mathrm { o l d } } }$ and $\mathbf { \epsilon } \gets \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ . It follows that: 

$$
\mathbb {E} [ \log r _ {t} ^ {i} ] = - \frac {\| \boldsymbol {d} \| ^ {2}}{2 \sigma^ {2}} = - D _ {\mathrm{KL}} (\pi_ {\theta_ {\mathrm{old}}} \| \pi_ {\theta}), \qquad \operatorname{Var} [ \log r _ {t} ^ {i} ] = \frac {\| \boldsymbol {d} \| ^ {2}}{\sigma^ {2}} = 2   D _ {\mathrm{KL}} (\pi_ {\theta_ {\mathrm{old}}} \| \pi_ {\theta}).
$$

Thus std[log $r _ { t } ^ { i } ] = \sqrt { 2 D _ { \mathrm { K L } } }$ . When the KL is moderate $( \mathrm { e . g . , } D _ { \mathrm { K L } } = 0 . 5 )$ , the standard deviation of the log-ratio is 1.0, meaning that individual log-ratio samples fluctuate by ±1 around the mean of −0.5. In terms of the ratio itself, this corresponds to roughly a 3× multiplicative spread. 

Implication for clipping. With a typical clip parameter $\epsilon = 0 . 2$ (i.e., clip range [0.8, 1.2]), the log-clip range is [log 0.8, log $1 . 2 ] \approx [ - 0 . 2 2 , 0 . 1 8 ]$ . Comparing this narrow range with the log-ratio standard deviation of $\sqrt { 2 D _ { \mathrm { K L } } }$ , we see that even for modest KL values, a significant fraction of samples will be clipped purely due to noise, not because the true divergence is excessive. This provides rigorous justification for replacing ratio-based clipping with direct divergence measurement. 

## Appendix E. Towards a Predictive Divergence Mask

We recall the asymmetric mask in Flow-DPPO (Eq. 18). The mask blocks the gradient $( \mathrm { i . e . , } M _ { t } ^ { i } = 0 )$ when two conditions hold simultaneously: (i) the divergence $D _ { t } > \delta$ already exceeds the trust-region threshold, and (ii) a directional condition signals that the optimization would push the policy further away from $\pi _ { \theta _ { \mathrm { o l d } } }$ . Concretely, the directional condition triggers when $\hat { A } ^ { i } > 0 \land r _ { t } ^ { i } > 1$ (the gradient would further increase an already-elevated ratio) or $\hat { A } ^ { i } < 0 \land r _ { t } ^ { i } < 1$ (the gradient would further decrease an already-reduced ratio). These two cases can be compactly unified as: 

$$
M _ {t} ^ {i} = 0 \quad \Longleftrightarrow \quad \operatorname{sgn} \bigl (\hat {A} ^ {i} \cdot (r _ {t} ^ {i} - 1) \bigr) > 0 \land D _ {t} > \delta .\tag{23}
$$

While this design is efective in practice, the directional indicator $\mathrm { s g n } \big ( \hat { A } ^ { i } ( r _ { t } ^ { i } - 1 ) \big )$ is a heuristic proxy for whether the upcoming gradient step will increase the divergence. In a ratio-based trust region (e.g., PPO clipping), this sign test is well-motivated: $r _ { t } ^ { i } - 1$ directly reflects the deviation of the single-sample Monte Carlo estimate of the importance ratio, so the sign of $\hat { A } ^ { i } ( r _ { t } ^ { i } - 1 )$ faithfully indicates whether the surrogate objective would drive the ratio further from unity. However, in a divergence-based trust region where the constraint is on $D _ { t } = D _ { \mathrm { K L } } ( \pi _ { \theta _ { \mathrm { o l d } } } \Vert \pi _ { \theta } )$ , the connection is less direct. The ratio $r _ { t } ^ { i }$ is a stochastic quantity evaluated at a single sampled action, whereas $D _ { t }$ measures a distributional distance that integrates over all actions. A positive $\hat { A } ^ { i } ( r _ { t } ^ { i } - 1 )$ does not guarantee that the gradient step will increase $D _ { t }$ , nor does a negative value guarantee a decrease. 

In this section we exploit the Gaussian structure of flow model policies to derive a more principled masking criterion. We first predict how a single gradient step changes $D _ { t } \ ( \ S \mathrm { E . 1 } )$ , obtaining a closed form expression that decomposes into a first-order directional term and a second-order magnitude term. The sign of the first-order term yields an exact directional criterion sgn $\left( \hat { A } \cdot ( \log r _ { t } - D _ { t } ) \right)$ , which recovers the current sign test in the small-divergence regime but reveals a correction when the policy has already drifted. The full expression further accounts for the step size and gradient magnitude, leading to a predictive mask (§E.2) that directly forecasts whether the post-update divergence will exceed δ. 

## E.1 Predicting Post-Update Divergence

Fix a denoising step with state $\mathbf { \mathcal { x } } _ { t }$ and suppress the time index for brevity. Write $\pmb { \mu } \equiv \pmb { \mu } _ { \theta } ( \pmb { x } _ { t } , t )$ 2 $\pmb { \mu } _ { \mathrm { o l d } } \equiv \pmb { \mu } _ { \theta _ { \mathrm { o l d } } } ( \pmb { x } _ { t } , t ) , \pmb { d } = \pmb { \mu } - \pmb { \mu } _ { \mathrm { o l d } }$ , and $D _ { t } = \| d \| ^ { 2 } / ( 2 \sigma ^ { 2 } )$ . The sampled action is ${ \pmb x } _ { t - \Delta t } = { \pmb \mu } _ { \mathrm { o l d } } + \sigma { \pmb \epsilon }$ with $\mathbf { \epsilon } \epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ 

We derive how a single gradient step on the surrogate objective $L = r _ { t } \cdot { \hat { A } }$ changes the divergence $D _ { t }$ The policy gradient with respect to $\pmb { \mu }$ is: 

$$
\nabla_ {\boldsymbol {\mu}} L = \hat {A} \cdot r _ {t} \cdot \nabla_ {\boldsymbol {\mu}} \log r _ {t} = \frac {\hat {A} \cdot r _ {t}}{\sigma^ {2}} (\sigma \boldsymbol {\epsilon} - \boldsymbol {d}).
$$

With efective learning rate $\eta ,$ the updated mean is $\pmb { \mu } _ { \mathrm { n e w } } = \pmb { \mu } + \pmb { \eta } \cdot \nabla _ { \pmb { \mu } } L$ . Let $g = \sigma \epsilon - d .$ The predicted post-update divergence is: 

$$
\begin{array}{r} D _ {t} ^ {\mathrm{new}} = \frac {\| \pmb {\mu} _ {\mathrm{new}} - \pmb {\mu} _ {\mathrm{old}} \| ^ {2}}{2 \sigma^ {2}} = \frac {1}{2 \sigma^ {2}} \left\| \pmb {d} + \frac {\eta \hat {A} r _ {t}}{\sigma^ {2}} \pmb {g} \right\| ^ {2} \\ = D _ {t} + \frac {\eta \hat {A} r _ {t}}{\sigma^ {4}} \pmb {g} ^ {\top} \pmb {d} + \frac {\eta^ {2} \hat {A} ^ {2} r _ {t} ^ {2}}{2 \sigma^ {6}} \| \pmb {g} \| ^ {2}. \end{array}\tag{24}
$$

From the ratio decomposition (Eq. 13), log $r _ { t } = \epsilon ^ { \top } d / \sigma - \| d \| ^ { 2 } / ( 2 \sigma ^ { 2 } )$ , which gives $g ^ { \top } d = \sigma ^ { 2 } ( \log r _ { t } -$ $D _ { t } )$ . The first-order term thus simplifies to $( \eta \hat { A } r _ { t } / \sigma ^ { 2 } ) ( \log r _ { t } - D _ { t } )$ . The three terms in Eq. 24 have clear interpretations: (1) the current divergence $D _ { t } ; \mathbf { \Omega } ( 2 )$ a first-order term whose sign determines whether the gradient step increases or decreases the divergence; (3) a non-negative second-order term that grows with the step size $\eta$ and gradient magnitude $\| g \|$ , always contributing positively to $D _ { t } ^ { \mathrm { n e w } }$ 

The first-order directional criterion. The direction of divergence change is mainly determined by the sign of the first-order term. Since $r _ { t } > 0$ and $\eta > 0$ , this sign equals: 

$$
\operatorname{sgn} \Bigl (\hat {A} \cdot \bigl (\log r _ {t} - D _ {t} \bigr) \Bigr).\tag{25}
$$

When this is positive, the gradient step increases $D _ { t } ;$ when negative, it decreases $D _ { t }$ . Equivalently, this is the sign of the inner product $\langle \nabla _ { \mu } L , \nabla _ { \mu } D _ { t } \rangle$ , confirming that the surrogate gradient projects onto the divergence-increasing direction. 

Recovery of the current mask. In the small-divergence regime $D _ { t } \ll 1$ (which is the typical operating range when the trust region is efective), the correction $D _ { t } \approx 0$ and the criterion simplifies to $\operatorname { s g n } ( \hat { A } \cdot \log r _ { t } )$ . Since $\operatorname { s g n } ( \log r _ { t } ) = \operatorname { s g n } ( r _ { t } - 1 )$ , this is equivalent to $\mathrm { s g n } \big ( \hat { A } \cdot ( r _ { t } - 1 ) \big )$ , which is exactly the directional condition in Eq. 23. Thus, the current Flow-DPPO mask implements the correct first-order divergence-increasing criterion in this regime. 

The correction term. When $D _ { t }$ is non-negligible (i.e., the policy has already drifted appreciably), the true divergence-change direction is sgn $\left( \hat { A } \cdot ( \log r _ { t } - D _ { t } ) \right)$ rather than sgn $\left( { \hat { A } } \cdot ( r _ { t } - 1 ) \right)$ . The subtracted term $D _ { t }$ shifts the decision boundary: a sample must have log $r _ { t } > D _ { t } > 0$ (rather than merely log $r _ { t } > 0 )$ before the positive-advantage gradient is classified as divergence-increasing. Intuitively, when the policy has already moved away from $\pi _ { \theta _ { \mathrm { o l d } } } .$ , a moderately elevated ratio does not necessarily push it further; only suficiently large ratios do. This yields a first natural refinement of the mask: replacing sgn $\left( \hat { A } ( r _ { t } - 1 ) \right)$  with sgn $\left( \hat { A } \cdot ( \log r _ { t } - D _ { t } ) \right)$ as the directional indicator, which we call the first-order predictive mask: 

$$
M _ {t} ^ {(1)} = \left\{ \begin{array}{l l} 0, & \text { if } \operatorname{sgn} \Big (\hat {A} \cdot \big (\log r _ {t} - D _ {t} \big) \Big) > 0 \land D _ {t} > \delta , \\ 1, & \text { otherwise. } \end{array} \right.\tag{26}
$$

This mask uses only quantities already computed during training $( \hat { A } , r _ { t } , D _ { t } )$ and requires no additional hyperparameters beyond the existing threshold δ. 

## E.2 The Predictive Mask

Based on Eq. 24, we define the $( f u l l )$ predictive mask that blocks updates whenever the predicted post-update divergence would exceed δ: 

$$
M _ {t} ^ {\text {pred}} = \left\{ \begin{array}{l l} 0, & \text {if} D _ {t} ^ {\text {new}} > \delta , \\ 1, & \text {otherwise.} \end{array} \right.\tag{27}
$$

Comparison with the first-order mask. The first-order mask $\left( \mathrm { E q . 2 6 } \right)$ only considers the direction of divergence change and still relies on the separate threshold condition $D _ { t } > \delta$ . The full predictive mask unifies both into a single inequality: whether the gradient increases or decreases divergence is automatically encoded in the predicted value $D _ { t } ^ { \mathrm { n e w } }$ , and the threshold comparison is applied to the predicted (rather than current) divergence. This has two consequences. First, when $D _ { t } \ll \delta$ , even a divergence-increasing step may be permitted if the predicted $D _ { t } ^ { \mathrm { n e w } }$ remains below δ. Second, when $D _ { t }$ is close to $\delta ,$ the second-order term $\eta ^ { 2 } \| \pmb { g } \| ^ { 2 }$ may push $D _ { t } ^ { \mathrm { n e w } }$ above $\delta$ even when the first-order direction is $^ { 6 } \mathrm { s a f e } ^ { 3 } \ ( \mathrm { i . e . }$ , the first-order mask would not fire), correctly blocking large gradient steps near the trust-region boundary. 

Recovery of the existing mask. In the limit $\eta  0$ , the second-order term vanishes and $D _ { t } ^ { \mathrm { n e w } } > \delta$ reduces to requiring that the first-order direction is positive and $D _ { t } > \delta .$ Combined with the small-divergence approximation $( D _ { t } \approx 0 )$ , this exactly recovers the current Flow-DPPO mask (Eq. 23). 

## E.3 Discussion on Mask Variants

Hierarchy of masks. The three masks form a natural hierarchy of increasing fidelity: 

$$
\underbrace {\operatorname{sgn} \bigl (\hat {A} (r _ {t} - 1) \bigr)} _ {\text {current (Eq. 23)}} \subset \underbrace {\operatorname{sgn} \bigl (\hat {A} (\log r _ {t} - D _ {t}) \bigr)} _ {\text {first - order (Eq. 26)}} \subset \underbrace {D _ {t} ^ {\text {new}} > \delta} _ {\text {full predictive (Eq. 27)}}.
$$

The current mask is the cheapest (no additional computation) and sufices when the trust region keeps $D _ { t }$ small throughout training. The first-order mask refines the directional decision with zero additional hyperparameters. The full predictive mask additionally requires an efective learning rate estimate but provides quantitative divergence prediction. 

Local approximation. The analysis treats µ as a free vector, whereas in practice it is the output of a neural network. The actual change in $\pmb { \mu } ( \pmb { x } _ { t } , t )$ is coupled to changes at all other inputs through shared parameters. The predictive mask is thus a local approximation that is most accurate when the efective learning rate is small and the network Jacobian is approximately preserved across one step. 

We leave empirical validation of the predictive masks to future work. The key contribution of this analysis is twofold: it provides a theoretical justification for the existing asymmetric condition (showing it is the correct first-order criterion in the small-divergence regime), and it charts a principled path toward more refined trust-region enforcement that exploits the Gaussian structure of flow model policies. 

## Appendix F. Experimental Details

## F.1 Computational Resources.

All experiments are conducted on NVIDIA H20 96GB GPUs. The main results in Table 1 require approximately 90K GPU hours in total (across SD3.5, FLUX2-klein-base-9B, and FLUX1-dev with all methods and reward configurations). Including all ablation studies, multi-epoch experiments, and auxiliary runs, the overall computational cost for all experiments reported in this paper is approximately 140K GPU hours. 

## F.2 Hyperparameters.

LoRA is used for all models. We use LoRA $r = 3 2$ and $\alpha = 6 4$ for SD3.5, $r = 6 4$ and $\alpha = 1 2 8$ for FLUX2-9B and FLUX.1-dev. The learning rate is set to $3 \times 1 0 ^ { - 4 }$ for all models aligning to previous works. We set the training resolution to $5 1 2 \times 5 1 2$ , number of denoising steps to 10 for SD3.5 and 14 for FLUX2-9B. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-01/d52561ff-9523-48be-af73-f2383cdc7168/c11cfcc990ff4a81fab235b1465fbc938d75a3b6bd4c69ad0fa760a33b833802.jpg)



<sup>F</sup>low-<sup>GRPO F</sup>low-<sup>CPS GRPO</sup>-<sup>G</sup>uard <sup>F</sup>low-<sup>DPPO F</sup>low-<sup>DPPO</sup> + <sup>CPS D</sup>iffusion-<sup>NF</sup>T


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-01/d52561ff-9523-48be-af73-f2383cdc7168/a9b1f5d94fdc10caa627d7c0e1d3922a000cae3fd87aafcbdcada32077504cb0.jpg)



<sup>F</sup>low-<sup>GRPO F</sup>low-<sup>CPS GRPO</sup>-<sup>G</sup>uard <sup>F</sup>low-<sup>DPPO F</sup>low-<sup>DPPO</sup> + <sup>CPS D</sup>iffusion-<sup>NF</sup>T



Figure 7: Training curves on SD3.5 for single-reward setting, including Difusion-NFT (Zheng et al., 2026) as an additional baseline. Flow-DPPO variants achieve state-of-the-art performance and less catastrophic forgetting on out-of-domain rewards, consistent with the main results.



Figure 8: Training curves on FLUX2-9B for multi-reward setting (GPU hours). Flow-DPPO variants consistently outperform the baselines across all metrics, with a notable improvement on the GenEval2 reward.


For GRPO setting, we use group size 16 and number of groups 64 per epoch for all methods. The PPO clip threshold is set to $1 \times 1 0 ^ { - 4 }$ for Flow-GRPO and Flow-CPS, and $4 \times 1 0 ^ { - 6 }$ for GRPO-Guard, following the oficial recommendation. The thresholds for KL-clipping are set to $1 \times 1 0 ^ { - 7 }$ for Flow-DPPO and $1 \times 1 0 ^ { - 6 }$ for Flow-DPPO+CPS due to their diferent KL-scaling factors. We applied the stragegy proposed in MixGRPO (Li et al., 2025) on all baselines and proposed methods for faster convergence and better performance. Specifically, we mix ODE and SDE sampling and randomly select 3 steps out of first half of the denoising steps for SDE sampling. The noise level for SDE sampling (η in CPS sampling) is set to 0.8. 

For Difusion-NFT, we follow the oficial implementation for SD3.5 for the rest of the hyperparameters, such as EMA schedule. 

## Appendix G. Additional Experimental Results

## G.1 Additional Training Curves

We provide the training curves on SD3.5 for the single-reward setting in Figure 7 (the multi-reward setting is in Figure 4 in the main body). We also provide the FLUX2-9B multi-reward training curves in Figure 8 and FLUX.1-dev in Figure 9. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-01/d52561ff-9523-48be-af73-f2383cdc7168/513df980fd296499798978dc8995a729ab4b69032cca18435df576be0b7e0afa.jpg)



Figure 9: Training curves on FLUX.1-dev for single-reward setting.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-01/d52561ff-9523-48be-af73-f2383cdc7168/6e6cc4a11502f71f64a6cc743e11454b821d8f57e6ebe83d21a2517a996ece4b.jpg)



Figure 10: Training curves on FLUX2-9B with CFG scale 4.0. Flow-DPPO variants remain robust under CFG, achieving strong performance with less catastrophic forgetting.


We additionally provide training curves on FLUX.1-dev in Figure 9. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-01/d52561ff-9523-48be-af73-f2383cdc7168/7f78d5bacc36f2ab72b4ec8a586e64472d4f291100643b6b43a1495c8069465b.jpg)



Figure 11: Training reward curves under three $D _ { \mathrm { K L } } ( \pi _ { \theta } \Vert \pi _ { \mathrm { r e f } } )$ regularization strengths (β) on FLUX2- klein-base-9B (multi-reward GDPO, CPS schedule). A moderate $\beta { = } 1 0 ^ { - 3 }$ suppresses early reward hacking on PickScore and HPSv2, balancing cross-reward gradients and boosting final GenEval2 performance without hurting end-of-training performance on any individual reward.


## G.2 KL Divergence Curves

Figure 12 visualises the per-step KL divergence between the current and reference (pre-trained) model across all six training settings and two SDE schedules. The corresponding end-of-training values are reported in Table 2 of the main body. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-01/d52561ff-9523-48be-af73-f2383cdc7168/fe778179411bd813dfa4573bc149f392ea8faf1232c582f8410fd85295cfe134.jpg)



Figure 12: KL-divergence between the current and reference (pre-trained) model during training, across six training settings (columns: four single-reward — SD3.5, FLUX2-9B $\mathrm { w / o }$ CFG, FLUX2-9B $\mathrm { w } / $ CFG, FLUX.1-dev; two multi-reward — SD3.5 multi, FLUX2-9B multi) and two SDE schedules (rows: Flow-SDE, CPS). For each schedule, Flow-DPPO variants maintain a lower KL divergence with the pre-trained model, indicating less catastrophic forgetting and reward hacking. The only exception is in the FLUX2-9B $\mathrm { w / o }$ CFG setting under the CPS schedule, where Flow-DPPO + CPS shows a higher KL divergence than the Flow-CPS baseline after about epoch 500. The Flow-CPS run on FLUX2-9B multi collapsed at epoch 480; we plot its full logged trajectory and report its end-of-training KL at the run’s last logged step in Table 2.


## G.3 Ablation Studies

## G.3.1 Classifier-Free Guidance

Previous works found that CFG heavily afects the training convergence and performance (Zheng et al., 2026). Here, we study the efect of CFG on the training of Flow-DPPO on FLUX2-9B, as shown in Figure 10, where the CFG scale is set to 4.0 following the oficial recommendation. With CFG, Flow-DPPO variants still achieve state-of-the-art performance on the training reward (GenEval2) and mitigate catastrophic forgetting on the out-of-domain prompts, consistent with the observations in previous discussions. This shows that the divergence-based mask is robust under CFG and continues to deliver strong performance. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-01/d52561ff-9523-48be-af73-f2383cdc7168/8ea5b5fb7b99483b78a5a90509fbea1db25307ef208a4427b064ca44e9818ddf.jpg)



Figure 13: $D _ { \mathrm { K L } } ( \pi _ { \theta } \Vert \pi _ { \mathrm { r e f } } )$ during training for diferent $\beta$ settings.



Table 3: End-of-training Soft $\mathrm { T I F A _ { G M } }$ on GenEval2 (%) across six training configurations (columns) and five RL algorithms (rows). The six columns correspond, left-to-right, to Figs. 2, 8, 10, 7, 4, 9. Per-column bold and underline mark the top-1 and top-2 methods; blue rows highlight our two contributions.


<table><tr><td rowspan="2">Method</td><td colspan="3">FLUX2-9B</td><td colspan="2">SD3.5</td><td>FLUX.1-dev</td></tr><tr><td>Single</td><td>Multi</td><td>+CFG</td><td>Single</td><td>Multi</td><td>Single</td></tr><tr><td>Flow-GRPO</td><td>84.5</td><td>46.8</td><td>54.6</td><td>56.6</td><td>39.9</td><td>87.8</td></tr><tr><td>Flow-CPS</td><td>82.7</td><td>47.1</td><td>89.0</td><td>74.8</td><td>44.6</td><td>91.2</td></tr><tr><td>GRPO-Guard</td><td>82.8</td><td>49.0</td><td>78.8</td><td>85.8</td><td>47.8</td><td>87.6</td></tr><tr><td>Diffusion-NFT</td><td>-</td><td>47.3</td><td>-</td><td>64.5</td><td>42.5</td><td>-</td></tr><tr><td>Flow-DPPO</td><td>85.1</td><td>57.7</td><td>87.4</td><td>78.9</td><td>48.1</td><td>90.7</td></tr><tr><td>Flow-DPPO + CPS</td><td>92.6</td><td>55.2</td><td>91.0</td><td>84.1</td><td>51.6</td><td>91.6</td></tr></table>

## G.3.2 Reference KL Regularization Strength

We ablate the strength of the $D _ { \mathrm { K L } } ( \pi _ { \theta } \Vert \pi _ { \mathrm { r e f } } )$ regularization term (controlled by $\beta )$ on FLUX2-kleinbase-9B under the multi-reward GDPO setting with CPS scheduling. Figure 11 shows the training reward curves and Figure 13 shows the KL divergence from the pretrained model. A moderate regularization strength $\left( \beta = 1 0 ^ { - 3 } \right)$ further mitigates early-stage reward hacking on auxiliary objectives (PickScore, HPSv2, etc.), thereby balancing the gradients across rewards and yielding an additional improvement in final GenEval2 performance over the unregularized baseline, without degrading end-of-training performance on any individual reward. 

## G.4 Quantitative Summary on GenEval2

To complement the per-setting training-curve figures above, Tables 4 and 3 report the end-of-training Soft $\mathrm { T I F A _ { G M } }$ score on GenEval2 for each method. Table 4 additionally reports end-of-training ancillary CLIP, PickScore, and HPSv2 rewards on both the in-domain GenEval2 prompt set and the held-out out-of-domain PickScore validation prompts, contextualising both SD3.5-medium and FLUX2-klein-base-9B by stacking six blocks: published reference numbers for state-of-the-art textto-image systems, the corresponding pretrained-baseline scores (no RL), and the five RL fine-tuning algorithms applied to each base model under both the single-reward (GenEval2-only) and multi-reward $\mathrm { ( G e n E v a l 2 + C L I P + P i c k S c o r e + H P S v 2 ) }$ configurations. Table 3 then expands the per-method Soft $\mathrm { T I F A _ { G M } }$ comparison to all five training settings reported in this paper. 


Table 4: GenEval2 [Soft TIFA , defined in (Kamath et al., 2025)] together with ancillary CLIP, PickScore, and HPSv2 rewards at the end of training. The four in-domain columns are evaluated on the GenEval2 prompt set (the oficial released evaluation set of 800 prompts); the three out-ofdomain columns are evaluated on the PickScore prompt set. Within each RL block, bold marks the per-column top-1 method and underline the per-column top-2 method. Blue rows highlight our two contributions.


<table><tr><td rowspan="2">Model</td><td colspan="4">In-Domain (GenEval2)</td><td colspan="3">Out-of-Domain (PickScore)</td></tr><tr><td>GenEval2</td><td>CLIP</td><td>PickScore</td><td>HPSv2</td><td>CLIP</td><td>PickScore</td><td>HPSv2</td></tr><tr><td colspan="8">State-of-the-Art T2I Models</td></tr><tr><td>SD3.5-large</td><td>22.8</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Bagel + CoT</td><td>23.1</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Qwen-Image</td><td>33.8</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Gemini 2.5 Flash Image</td><td>44.6</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td colspan="8">Pretrained baselines (before RL)</td></tr><tr><td>SD3.5-medium</td><td>12.4</td><td>0.250</td><td>21.00</td><td>0.213</td><td>0.244</td><td>19.99</td><td>0.210</td></tr><tr><td>FLUX2-klein-base-9B</td><td>25.4</td><td>0.281</td><td>20.92</td><td>0.228</td><td>0.254</td><td>20.05</td><td>0.230</td></tr><tr><td>FLUX.1-dev</td><td>23.3</td><td>0.297</td><td>23.26</td><td>0.315</td><td>0.276</td><td>21.91</td><td>0.304</td></tr><tr><td colspan="8">SD3.5-medium, single-reward RL fine-tuning</td></tr><tr><td>Flow-GRPO</td><td>56.6</td><td>0.297</td><td>21.21</td><td>0.219</td><td>0.252</td><td>19.33</td><td>0.206</td></tr><tr><td>Flow-CPS</td><td>74.8</td><td>0.313</td><td>21.68</td><td>0.235</td><td>0.260</td><td>19.94</td><td>0.220</td></tr><tr><td>GRPO-Guard</td><td>85.8</td><td>0.328</td><td>22.03</td><td>0.252</td><td>0.265</td><td>19.94</td><td>0.214</td></tr><tr><td>Diffusion-NFT</td><td>64.5</td><td>0.307</td><td>21.69</td><td>0.251</td><td>0.262</td><td>20.24</td><td>0.239</td></tr><tr><td>Flow-DPPO</td><td>78.9</td><td>0.319</td><td>22.06</td><td>0.263</td><td>0.265</td><td>20.45</td><td>0.253</td></tr><tr><td>Flow-DPPO + CPS</td><td>84.1</td><td>0.316</td><td>21.99</td><td>0.262</td><td>0.272</td><td>20.50</td><td>0.246</td></tr><tr><td colspan="8">SD3.5-medium, multi-reward RL fine-tuning</td></tr><tr><td>Flow-GRPO</td><td>39.9</td><td>0.358</td><td>25.09</td><td>0.399</td><td>0.273</td><td>22.07</td><td>0.349</td></tr><tr><td>Flow-CPS</td><td>44.6</td><td>0.359</td><td>25.51</td><td>0.407</td><td>0.265</td><td>22.08</td><td>0.343</td></tr><tr><td>GRPO-Guard</td><td>47.8</td><td>0.353</td><td>25.64</td><td>0.409</td><td>0.272</td><td>22.32</td><td>0.354</td></tr><tr><td>Diffusion-NFT</td><td>42.5</td><td>0.334</td><td>25.30</td><td>0.394</td><td>0.269</td><td>22.52</td><td>0.355</td></tr><tr><td>Flow-DPPO</td><td>48.1</td><td>0.345</td><td>25.63</td><td>0.409</td><td>0.273</td><td>22.58</td><td>0.360</td></tr><tr><td>Flow-DPPO + CPS</td><td>51.6</td><td>0.369</td><td>25.72</td><td>0.415</td><td>0.279</td><td>22.51</td><td>0.361</td></tr><tr><td colspan="8">FLUX2-klein-base-9B, single-reward RL fine-tuning</td></tr><tr><td>Flow-GRPO</td><td>84.5</td><td>0.314</td><td>21.82</td><td>0.276</td><td>0.264</td><td>20.84</td><td>0.280</td></tr><tr><td>Flow-CPS</td><td>82.7</td><td>0.311</td><td>21.82</td><td>0.261</td><td>0.275</td><td>21.15</td><td>0.267</td></tr><tr><td>GRPO-Guard</td><td>82.8</td><td>0.312</td><td>20.52</td><td>0.210</td><td>0.230</td><td>18.45</td><td>0.167</td></tr><tr><td>Flow-DPPO</td><td>85.1</td><td>0.331</td><td>22.22</td><td>0.294</td><td>0.278</td><td>21.27</td><td>0.285</td></tr><tr><td>Flow-DPPO + CPS</td><td>92.6</td><td>0.315</td><td>21.97</td><td>0.279</td><td>0.265</td><td>20.79</td><td>0.272</td></tr><tr><td colspan="8">FLUX2-klein-base-9B, multi-reward RL fine-tuning</td></tr><tr><td>Flow-GRPO</td><td>46.8</td><td>0.371</td><td>25.61</td><td>0.412</td><td>0.277</td><td>22.62</td><td>0.357</td></tr><tr><td>Flow-CPS</td><td>47.1</td><td>0.361</td><td>25.70</td><td>0.416</td><td>0.276</td><td>22.85</td><td>0.364</td></tr><tr><td>GRPO-Guard</td><td>49.0</td><td>0.375</td><td>25.27</td><td>0.411</td><td>0.269</td><td>21.99</td><td>0.349</td></tr><tr><td>Diffusion-NFT</td><td>47.3</td><td>0.336</td><td>24.87</td><td>0.389</td><td>0.274</td><td>22.47</td><td>0.351</td></tr><tr><td>Flow-DPPO</td><td>57.7</td><td>0.364</td><td>25.76</td><td>0.418</td><td>0.282</td><td>22.90</td><td>0.368</td></tr><tr><td>Flow-DPPO + CPS</td><td>55.2</td><td>0.386</td><td>26.15</td><td>0.427</td><td>0.287</td><td>22.97</td><td>0.370</td></tr><tr><td colspan="8">FLUX.1-dev, single-reward RL fine-tuning</td></tr><tr><td>Flow-GRPO</td><td>87.8</td><td>0.331</td><td>23.03</td><td>0.311</td><td>0.291</td><td>21.85</td><td>0.311</td></tr><tr><td>Flow-CPS</td><td>91.2</td><td>0.328</td><td>23.20</td><td>0.317</td><td>0.288</td><td>21.98</td><td>0.307</td></tr><tr><td>GRPO-Guard</td><td>87.6</td><td>0.333</td><td>22.69</td><td>0.293</td><td>0.286</td><td>21.03</td><td>0.276</td></tr><tr><td>Flow-DPPO</td><td>90.7</td><td>0.331</td><td>23.15</td><td>0.323</td><td>0.290</td><td>21.60</td><td>0.300</td></tr><tr><td>Flow-DPPO + CPS</td><td>91.6</td><td>0.331</td><td>23.29</td><td>0.322</td><td>0.289</td><td>21.91</td><td>0.305</td></tr></table>