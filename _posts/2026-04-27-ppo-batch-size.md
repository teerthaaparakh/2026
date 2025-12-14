---
layout: distill
title: "The Trade-off Between Parallel Environments and Steps in PPO"
description: This blog post explores batch size in PPO-what happens when we increase the number of parallel environments versus the number of rollout steps, while keeping the total samples per update fixed. We discuss how this affects bias and variance in gradient estimation.
date: 2026-11-13
future: true
htmlwidgets: true
hidden: true

# Mermaid diagrams
mermaid:
  enabled: true
  zoomable: true

# Anonymize when submitting
authors:
  - name: Anonymous

# must be the exact same name as your blogpost
bibliography: 2026-04-27-ppo-batch-size.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: "Clarifying Terminology: Batch vs Mini-Batch"
  - name: "Background: PPO Gradient Computation"
  - name: Bias and Variance Due to Batch Size
  - name: Bias and Variance Due to Mini-batches
  - name: "What role does GAE-λ play here?"
 

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

### Introduction
In this post, we are going to explore a common dilemma we face when tuning PPO <d-cite key="schulman2017ppo"></d-cite> hyperparameters: constructing the batch size.

It's easy to think of batch size as just a single number, but in PPO, it is actually the product of two distinct levers we can pull:

- The number of parallel environments ($N$).
- The number of steps collected per environment ($T$).

Does it matter if we reach a batch size of 2,048 by running 2 environments for 1,024 steps, or by running 1,024 environments for 2 steps?

Recent work has shown that data collection strategy is not a trivial detail. Multiple studies—including <d-cite key="mayor2025the"></d-cite> and <d-cite key="sapg2024"></d-cite>—highlight that how we distribute experience across environments and rollout lengths can meaningfully influence the stability and effectiveness of on-policy RL methods like PPO. As simulation becomes faster and large-scale parallelism more accessible, understanding these choices is becoming increasingly important.

In this post, we'll dig into why the structure of the batch matters at all. In particular, we'll look at how increasing $N$ and increasing $T$ affect the bias and variance of PPO’s gradient estimates, theoretically and empirically.
<!-- We’ll keep the heavy equations to a minimum and rely on illustrations to build intuition for what’s happening under the hood. -->

### Clarifying Terminology: Batch vs. Mini-Batch
When reading PPO implementations across different libraries, you may notice that the term batch size is used in slightly different ways. To keep things consistent in this post, we’ll use the following terminology as is used in the original PPO paper <d-cite key="schulman2017ppo"></d-cite>:

- Rollout Buffer (Total Batch Size): The full dataset collected before a policy update. It is the product of the number of parallel environments ($N$) and the number of steps collected per environment ($T$).

$$\text{Total Batch Size} = N \times T$$

- Mini-Batch: A subset of the Rollout Buffer used for a single gradient descent step. The Rollout Buffer is shuffled and divided into these smaller chunks during training.

**Source of Confusion.**
Different RL libraries sometimes use the word batch to refer to what we’re calling a mini-batch. For example, Stable Baselines3 <d-cite key="stable-baselines3"></d-cite>, Dopamine <d-cite key="castro18dopamine"></d-cite>, OpenAI Baselines (PPO1) <d-cite key="baselines"></d-cite>, and Ray RLlib <d-cite key="liang2018rllib,liang2021rllib"></d-cite> commonly adopt this convention. This can lead to ambiguity, especially when tuning PPO’s data collection parameters.

As noted by <d-cite key="shengyi2022the37implementation"></d-cite>:
<blockquote style="font-size:0.9em; margin:6px 0; padding-left:12px; border-left:3px solid #ccc;">Some common mis-implementations include (1) always using the whole batch for the update, and (2) implementing mini-batches by randomly fetching from the training data, which does not guarantee all training data points are fetched.</blockquote>

Being precise about terminology makes it clear that when we adjust hyperparameters like $N$ and $T$, we’re modifying the amount of experience collected instead of just changing how much data goes into each optimization step (which is determined by the size of each mini-batch).

<!-- ### Background: MDP Setup and PPO Overview

Before discussing how PPO constructs its batches, we briefly summarize the reinforcement-learning setup. We consider an agent interacting with an environment modeled as a Markov Decision Process (MDP), defined by the state space ($S$), action space ($A$), transition dynamics ($P$), reward function ($r$), and discount factor ($\gamma$). At each step, the agent samples an action from its policy $a_t \sim \pi_\theta(a \mid s_t)$, receives a reward, and transitions to a new state. The goal is to find $\theta$ to maximize expected discounted return: $$J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{\infty} \gamma^t r_t\right]$$. PPO is an on-policy algorithm that iteratively improves the policy using new data collected from the current version of $\pi_\theta$.

PPO (High-level Pseudocode):
```
Initialize policy πθ and value function Vϕ

repeat:
    # --- Data Collection ---
    For each of N environments:
        Rollout T steps using πθ
        Store (s, a, r, logπθ(a|s), Vϕ(s)) in the rollout buffer

    Compute advantages Â using GAE or Monte Carlo returns

    # --- Policy Update ---
    For K epochs:
        Shuffle the rollout buffer
        For each mini-batch M:
            Compute ratio:      r = πθ(a|s) / old_logπθ(a|s)
            Compute clipped objective:
                Lclip = min(r Â, clip(r, 1-ε, 1+ε) Â)
            Update policy parameters θ via gradient ascent on Lclip
            Update value function parameters ϕ via regression on returns

until convergence
``` -->
<!-- 
### Data Distribution

In reinforcement learning, a trajectory $\tau = (s_0,a_0,s_1,a_1,\dots,s_T)$ is generated jointly by the policy and environment dynamics. Its probability under policy $\pi_\theta$ and transition matrix $P$ is:

$$
P(\tau\mid \pi_\theta) = p(s_0)\, \prod_{t=0}^{T-1} \pi_\theta(a_t\mid s_t)\, P(s_{t+1}\mid s_t,a_t).
$$

As the policy updates online, this trajectory distribution shifts between updates. PPO optimizes expected return using samples from the current distribution; near convergence, updates become small and the distribution stabilizes.

Our focus here is not convergence, but how structuring a fixed total batch $NT$ (many short vs. few long rollouts) influences the variance of gradient estimates. 

-->
### Background - PPO Gradient Computation 

Before diving into bias and variance, let's write down the gradient PPO computes during policy updates. Ignoring value and entropy terms, the per-sample PPO objective is:

$$L_t^{\text{PPO}}(\theta) = -\min\left(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_t\right),$$

with probability ratio:

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}.$$

During a training epoch, PPO updates the policy using mini-batches sampled from the rollout buffer of size $N \times T$. For a mini-batch $M$, the gradient estimate is:

$$G_{mb} = \frac{1}{|M|} \sum_{t \in M} \nabla_\theta L_t^{\text{PPO}}(\theta).$$

Because PPO performs multiple mini-batch updates per iteration, the policy parameters $\theta$ change between these updates. As a result, later mini-batches observe a policy that differs from the one that generated the data, meaning $r_t(\theta) \neq 1$ for most mini-batches. 
<details>
<summary><strong>Additional Info on Clipping</strong></summary>

The clipping term prevents the update from moving too far from the behavior policy that produced the data, improving stability during multiple gradient steps by ensuring the new policy stays sufficiently close to the old one for the importance-sampling ratio $r_\theta(t)$ to remain reliable.

</details>



### Bias and Variance Due to Batch Size

<div style="border-left: 4px solid #335c67; padding: 0.5em 1em; background: #fff3b0b5;">
<strong>Note:</strong><br>
In this section, we assume that the gradient is computed using the <strong>full batch</strong> of collected data. This allows us to isolate and analyze the inherent sources of bias and variance that arise purely from the sampled batch itself.
<br>
As mentioned in previous section, PPO uses <strong>mini-batch</strong> stochastic gradient descent, which introduces additional sources of bias and variance due to sub-sampling. These effects will be discussed in the later section.
</div>
<br> 

#### <span style="color:#2ca6a4;">▶</span> Simplified Gradient Equation

If we assume that the gradient is computed using the <strong>full batch</strong> of collected data, we can isolate and analyze the inherent sources of bias and variance that arise purely from the sampled batch itself. Under this assumption, the gradient estimator takes the same form as the standard (vanilla) policy gradient:

<div style="border: 1px solid #4a79bc01; padding: 0.2em 0.3em; border-radius: 4px; background: #f6f2f8ff;">
$$
G_B = - \frac{1}{N T} \sum_{(s_t, a_t) \in \mathcal{D}}
A^{\theta_{\text{old}}}(s_t, a_t)\,
\nabla_\theta \log \pi_\theta(a_t \mid s_t)
$$
</div>

Here, the policy gradient estimate $G_B$ is simply the average of the per-sample policy-gradient contributions $g_i$.

Although the PPO gradient is ultimately derived from the standard (vanilla) policy gradient in a way that supports mini-batch sampling and multiple gradient updates per iteration, it is helpful to “backtrack’’ from the PPO gradient to this full-batch (vanilla) version for clarity.

The policy $\pi_{\theta_{old}}$ is the one used to collect the data, while $\pi_\theta$ is the policy being updated. For the very first full-batch gradient computation of an iteration, we have $\pi_\theta = \pi_{\theta_{old}}$, which implies that the probability ratio satisfies $r = 1$. Because of this, the clipping term does not affect the gradient at this stage. The PPO loss therefore reduces to:

$$L = -\frac{1}{N T} \sum_{(s_t, a_t) \in \mathcal{D}}
\underbrace{\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}}_{r} \,
A^{\theta_{\text{old}}}(s_t, a_t)$$

Taking the gradient of this loss yields:

$$G_B = \nabla_\theta L = -\frac{1}{N T} \sum_{(s_t, a_t) \in \mathcal{D}}
\nabla_\theta \left[ r \cdot A^{\theta_{\text{old}}}(s_t, a_t) \right]$$

Since the advantage $A$ is constant with respect to the new policy $\theta$, and knowing that $\nabla_\theta r = r \cdot \nabla_\theta \log \pi_\theta(a_t \mid s_t)$, the gradient simplifies to:

$$\nabla_\theta L = -\frac{1}{N T} \sum_{(s_t, a_t) \in \mathcal{D}}
r \, A^{\theta_{\text{old}}}(s_t, a_t) \, \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

Since $r=1$ in the full-batch case, the equation becomes:

$$G_B = - \frac{1}{N T} \sum_{(s_t, a_t) \in \mathcal{D}} \underbrace{ A^{\theta_{\text{old}}}(s_t, a_t) \, \nabla_\theta \log \pi_\theta(a_t \mid s_t) }_{g_i}$$


#### <span style="color:#2ca6a4;">▶</span> Sources of Noise in Gradient Estimation

Let us define $G_T$ as the ideal gradient we would obtain with infinite data, or full distribution of trajectories. When PPO estimates a policy gradient, it computes ${G_B}$ from a finite set of sampled trajectories rather than the true gradient $G_T$. This estimator ${G_B}$ is noisy, and it is helpful to describe this noise in terms of two main contributions:

$$G_B \approx G_T + \color{brown}{\text{sampling noise}} + \color{blue}{\text{advantage estimation noise}},$$

where $G_T$ denotes the ideal gradient under the policy's trajectory distribution.

 - The first term, $\color{brown}{\text{sampling noise}}$, is the noise that arises from working with a finite number of samples.
 - The second term, $\color{blue}{\text{advantage estimation noise}}$, appears
  because each gradient $g_i$ is scaled by an advantage estimate
  $A^{\theta_{\text{old}}}(s_t, a_t)$, which itself is noisy. Since advantage
  estimation depends on multi-step returns and bootstrapping, two noise sources
  arise:
    - $\color{blue}{\text{Trajectory Variability Noise}}$ comes from randomness in environment transitions and policy stochasticity. Even small random differences early in a trajectory can lead to large differences in later outcomes and returns.
    - $\color{blue}{\text{Credit Assignment Noise}}$ reflects how difficult it is to determine which early actions contributed to rewards that appear much later. This makes advantage estimates inherently noisy, especially in sparse or delayed reward environments.

<!-- Together, these components explain why RL gradient estimates tend to exhibit more variability than those in supervised learning, even when the nominal batch size is large. -->
Overall, these noise sources appear in the gradient estimator in two ways:
- Variance: how much ${G_B}$ would fluctuate if we collected a different batch from the *same* policy.
- Bias: the systematic deviation between the expected estimate $\mathbb{E}[{G_B}]$ and the true gradient.
<div align="center" style="display:flex; justify-content:center; gap:16px;"> 
  <div style="flex:1; max-width:550px;"> <img src="{{ site.baseurl }}/assets/img/2026-04-27-ppo-batch-size/data_distribution.png" alt="data distribution" style="width:100%; border-radius:6px;"/> 
    <div class="explain-box" style="margin-top:8px;"> <strong>Figure</strong>  Policy Update:  Updating a policy alters the trajectory distribution. A policy update from iteration $<i>i</i>$ to $<i>i+1</i>$ shifts the probability mass over
      state–action trajectories, which is what the gradient aims to achieve. 
    </div> 
  </div> 
</div>

<div align="center" style="display:flex; justify-content:center; gap:16px;"> 
  <div style="flex:1; max-width:400px;"> <img src="{{ site.baseurl }}/assets/img/2026-04-27-ppo-batch-size/batch_gradient.png" alt="PPO bias-variance and policy update illustration" style="width:85%; border-radius:6px;"/> 
    <div class="explain-box" style="margin-top:8px;"> <strong>Figure</strong> Bias and variance: The estimated gradient <i>Ĝ</i> differs from the true gradient $<i>G_T</i>$ due to sampling variability (variance) and systematic estimation error
      (bias). Both influence how the policy moves from iteration $<i>i</i>$ to $<i>i+1</i>$. 
    </div> 
  </div>
</div>



#### <span style="color:#2ca6a4;">▶</span> Effect of Batch Size on Variance

The variance of a mean is critical because it tells us how noisy $G_B$ is. High variance means we can't trust the update.

The total variance of our gradient estimate $G_B$ can be mathematically decomposed into two parts:

$$\text{Var}(G_B) = \text{Var}\Bigg(\frac{1}{N T} \sum_i g_i \Bigg) = \color{blue}{\underbrace{\frac{1}{(N T)^2} \sum_i \text{Var}(g_i)}_{\text{Term 1: Individual Variance}}} + \color{brown}{\underbrace{\frac{1}{(N T)^2} \sum_{i \neq j} \text{Cov}(g_i, g_j)}_{\text{Term 2: Covariance (Correlation)}}}$$

<!-- This decomposition holds the key to the $N$ vs. $T$ trade-off:

- $\color{blue}{\text{Term 1: Individual Variance}}$

This is the noise coming from each step $g_i$. Due to noise in advantage $ A^{\theta_{\text{old}}}(s_t, a_t)$ estimation each gradient $g_i$ is also noisy. Since the sum is divided by $(NT)^2$, this term shrinks rapidly as the Total Batch Size ($NT$) increases. If every sample were independent, this is all we would have to worry about. 

- $\color{brown}{\text{Term 2: Covariance (The Correlation Problem)}}$

This is the term that makes RL different from Supervised Learning. $\text{Cov}(g_i, g_j)$ is not zero because $g_i$ and $g_{i+1}$ come from consecutive steps in the same environment. Consecutive states are highly related, so the policy gradients derived from them are also highly correlated.

The Impact: When $T$ is large, we have many highly correlated steps in our batch, leading to a large positive covariance term. This effectively lowers the "effective sample size" of our batch. Even if $NT$ is large, if $T$ is too long, the covariance term can keep the overall $\text{Var}(G_B)$ high this increses the sampling noise because effective sample size has reduced. Also if $T$ is large, then the individual variance term is also high because for long horizon the advanateg estimation has high variance, the role of gae-lambda in advantage estimation is dicussed later. But greater the rollout length, higher the variance of this term.

When $T$ is small, the advantage estomation does not have variance and also the effective sample size is not muche reduced as compared to the case when we had long horizon.

The take-away: To aggressively reduce variance, we must minimize the covariance term. This requires breaking the temporal correlation, which means we need more independent starting points—that is, increasing the number of parallel environments, $\mathbf{N}$. -->


This decomposition connects directly to the noise sources discussed earlier and holds the key to the $N$ vs.\ $T$ trade-off:

- $\color{blue}{\text{Term 1: Individual Variance}}$

  This term reflects the variability of individual gradient contributions.
  In practice, this variability appears as noise in the advantage estimates,
  since each $g_i$ is scaled by a noisy advantage value. Because this term scales
  with $1/(NT)^2$, it decreases rapidly as the batch size grows.  
  If every sample were independent, this would be the only variance term we would need to consider.

- $\color{brown}{\text{Term 2: Covariance (The Correlation Problem)}}$

  This is the term that makes RL different from supervised learning. The covariance $\mathrm{Cov}(g_i, g_j)$ is not zero because $g_i$ and $g_{i+1}$ come from consecutive steps in the same environment. Consecutive states are highly related, so the policy gradients computed from them are also highly correlated.

**Impact:**  
When $T$ is **large**, the batch contains many highly correlated steps, leading to a large positive covariance term. This effectively reduces the *effective sample size* of the batch. This $\color{brown}{\text{increases the sampling noise}}$ because the effective sample size has been reduced. Therefore, a long horizon $T$ can keep $\mathrm{Var}(G_B)$ high. In addition, when $T$ is large, the $\color{blue}{\text{individual variance term also increases}}$, since advantage estimation has high variance over long horizons. (The role of $\lambda$ in GAE will be discussed later.)

When $T$ is small, advantage estimates have lower variance, and the effective sample size is not reduced as severely compared to the long-horizon case, therefore the total variance in the gradient estimate is much lower.

**Take-away:**  
To aggressively reduce variance, we must minimize the covariance term. This requires breaking temporal correlations, which means collecting more independent trajectories—that is, increasing the number of parallel environments, $\mathbf{N}$.

**Some FrozenLake-v1 Environment Experiments:**  
To investigate whether policy gradients are correlated along a trajectory, we use the 8×8 Slippery FrozenLake-v1 environment. PPO is trained with
<code style="background:#f6f2f8ff; padding:2px 4px; border-radius:4px;">num_envs = 4</code> and
<code style="background:#f6f2f8ff; padding:2px 4px; border-radius:4px;">num_steps = 32</code>,
which produces a total batch size of
<code style="background:#f6f2f8ff; padding:2px 4px; border-radius:4px;">batch_size = 4 × 32 = 128</code> samples per update.

Unlike a fixed-horizon setting, each trajectory here spans from the episode start until termination—either reaching the goal or falling into a hole. Thus, trajectory lengths vary depending on when the episode ends.

To measure correlation within a trajectory, we extract one full episode trajectory at a given update during training. For that trajectory, we compute:
- The mean gradient is defined as:

  $$
  \bar{g} = \frac{1}{L} \sum_{t=1}^{L} g_t,
  $$

  where \(L\) is the episode length (not necessarily 32).

- The cosine similarity between each per-step gradient \(g_t\) and the trajectory mean \(\bar{g}\) is:

  $$
  \cos(g_t, \bar{g})
  = \frac{g_t \cdot \bar{g}}
        {\lVert g_t \rVert \, \lVert \bar{g} \rVert }.
  $$

  If gradients along a trajectory are correlated, then
  $\cos(g_t, \bar{g})$ will remain high across steps.

<!-- <div align="center" style="display:flex; justify-content:center; gap:16px;">

  <div style="flex:1; max-width:650px;">
    <img src="{{ site.baseurl }}/assets/img/2026-04-27-ppo-batch-size/traj_cos_update150_batch1_traj1.png"
         alt="Subplot A"
         style="width:100%; border-radius:6px;"/>
    <div class="explain-box" style="margin-top:8px;">
      <strong>(a)</strong> The x-axis is the step number for a aprticualr trajectory and ya-xis is cosine similiaty of that trajcteory data points with the trajectory mean for update number 150
    </div>
  </div>

  <div style="flex:1; max-width:650px;">
    <img src="{{ site.baseurl }}/assets/img/2026-04-27-ppo-batch-size/traj_cos_update250_batch2_traj3.png"
         alt="Subplot B"
         style="width:100%; border-radius:6px;"/>
    <div class="explain-box" style="margin-top:8px;">
      <strong>(b)</strong> The x-axis is the step number for a aprticualr trajectory and ya-xis is cosine similiaty of that trajcteory data points with the trajectory mean for update number 250.
    </div>
  </div>

</div> -->

<div align="center" style="display:flex; justify-content:center; gap:16px;"> 
  <div style="flex:1; max-width:650px;"> <img src="{{ site.baseurl }}/assets/img/2026-04-27-ppo-batch-size/traj_cos_update150_batch1_traj1.png" alt="Subplot A" style="width:105%; border-radius:6px;"/> 
    <!-- <div class="explain-box" style="margin-top:8px;"> <strong>(a)</strong> Cosine similarity between each step’s gradient and the trajectory’s mean gradient at update 150. Higher values indicate stronger within-trajectory correlation. 
    </div>  -->
  </div> 
  <div style="flex:1; max-width:650px;"> <img src="{{ site.baseurl }}/assets/img/2026-04-27-ppo-batch-size/traj_cos_update250_batch2_traj3.png" alt="Subplot B" style="width:105%; border-radius:6px;"/> 
    <!-- <div class="explain-box" style="margin-top:8px;"> <strong>(b)</strong> Cosine similarity for a different trajectory at update 250, again showing pronounced correlation gradients of a trajectory. 
    </div>  -->
  </div> 
</div>
<div class="explain-box" style="max-width:900px; margin:1rem auto;">
  <strong>Figure:</strong> Cosine similarity between per-step gradients and the
  trajectory’s mean gradient for two different training instances. These figures illustrate strong within-trajectory correlation, but this pattern may not necessarily appear uniformly across all stages of training.
</div>


The strong correlation we observed within a trajectory suggests that <code style="background:#f6f2f8ff; padding:2px 4px; border-radius:4px;">num_steps</code> play a significant role in gradient variance. To study this effect more systematically, we compare two PPO settings: `num_envs = 4`, `num_steps = 32` (long trajectories) and `num_envs = 32`, `num_steps = 4` (short trajectories).
<!-- <code style="background:#f6f2f8ff; padding:2px 4px; border-radius:4px;">num_envs = 4, num_steps = 32</code> (long trajectories) and 
<code style="background:#f6f2f8ff; padding:2px 4px; border-radius:4px;">num_envs = 32, num_steps = 4</code> (short trajectories).   -->

For each setting, we randomly sample five batches from the rollout buffer (`batch0` … `batch4`).  
From each batch, we compute the **mean gradient vector**, and then measure how the per-sample gradients of `batch0` align with the mean gradients of all batches. Formally, for every batch \(j\), we compute:

$$
\cos(g_i^{(0)}, \bar{g}^{(j)}) 
= \frac{g_i^{(0)} \cdot \bar{g}^{(j)}}{\lVert g_i^{(0)} \rVert \, \lVert \bar{g}^{(j)} \rVert }.
$$

This allows us to visualize whether gradients from `batch0` tend to point in the same direction as gradients from other batches.  

Importantly, the differences we observe between batches represent the overall variance of the gradient estimator. This includes both the per-sample variance (noise in each $(g_i)$) and the additional covariance created by temporal correlations within trajectories.

In the **long-horizon case (4×32)**, we observe that gradients from `batch0` can be strongly **positively** or **negatively** correlated with gradients from other batches. Negative correlation means that the gradient estimate from one batch points in the *opposite* direction of another batch’s mean—indicating high variance in gradient estimation.  

In contrast, in the **short-horizon case (32×4)**, the cosine similarities remain consistent, suggesting lower variability in gradient direction across batches.

This pattern is shown below at a certain step. This represents instance; the behavior may not necessarily generalize to all timesteps.

<style>
  .fixed-row {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
    margin-top: 1rem;
    margin-bottom: 1rem;
  }

  .scroll-window {
    max-height: 300px;  /* adjust to control visible scroll area */
    overflow-y: auto;
    border: 1px solid #ccc;
    padding: 5px;
    border-radius: 6px;
  }

  .scroll-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
  }

  img {
    width: 100%;
    height: auto;
  }
</style>
<style>
  .row-header {
    font-weight: bold;
    margin-top: 1.2rem;
    margin-bottom: 0.3rem;
    font-size: 1.05rem;
    text-align: left;
    padding-left: 4px;
    border-left: 4px solid #444;
  }
</style>
<style>
  .explain-box {
    border: 1px solid #aaa;
    border-left: 4px solid #444;
    padding: 12px 16px;
    background: #f7f7f7;
    border-radius: 6px;
    margin-top: 1.5rem;
    margin-bottom: 1.5rem;
    font-size: 0.95rem;
    line-height: 1.45;
  }

  .explain-box code {
    background: #eee;
    padding: 2px 4px;
    border-radius: 4px;
    font-size: 0.95em;
  }
</style>

<style>
  .column-headers {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
    font-weight: bold;
    text-align: center;
    font-size: 1.2rem;
  }
  .header {
    padding-bottom: 4px;
    border-bottom: 2px solid #444;
  }
</style>


{% assign steps = "100,200,300,400,500,600,700,800,900,1100,1200,1300,1400,1500" | split: "," %}

<label for="stepSlider">Training Step:</label>
<input type="range" id="stepSlider"
       min="0" max="{{ steps.size | minus: 1 }}" step="1" value="3"
       oninput="updateImages(this.value)">
<span id="stepLabel">{{ steps[0] }}</span>

<div class="column-headers">
  <div class="header">4vs32</div>
  <div class="header">32vs4</div>
</div>

<div class="row-header">Cosine Similarity within Batch</div>

<!-- ALWAYS VISIBLE FIXED ROW -->
<div class="fixed-row">
  <img id="fixed0">
  <img id="fixed1">
</div>

<div class="row-header">Cosine Similarity across Batches</div>

<!-- SCROLLABLE ROWS (batch1..batch4) -->
<div class="scroll-window">
  <div class="scroll-grid">
    <!-- rows 1 to 4 → 8 images -->
    {% for i in (0..7) %}
      <img id="scroll{{ i }}">
    {% endfor %}
  </div>
</div>


<div class="explain-box">
  <!-- <strong>What these plots show:</strong><br><br>
  Each row compares the policy gradients obtained from different mini-batch
  pairs during PPO training at a particular training timestep (selected using the
  slider). The left column (<em>4vs32</em>) contains gradient comparisons from the
  experiment where batch size = 4 and minibatch size = 32, while the right column
  (<em>32vs4</em>) contains comparisons from the experiment with batch size = 32 and
  minibatch size = 4.<br><br>
  The top row (always visible) shows the diagonal comparison
  <code>batch0_vs_batch0</code>, which serves as a self-similarity reference.
  The scrollable section below shows the off-diagonal comparisons
  <code>batch0_vs_batch[i]</code> for <i>i = 1…4</i>, allowing inspection of how
  gradients vary across different batch selections. -->
  Cosine similarity between gradients from `batch0` and the mean gradients of other batches.  
  In the long-horizon setting (4×32) on the left, gradient directions vary substantially across batches—including negative alignment—indicating high gradient variance.  
  In the short-horizon setting (32×4) on the right, gradients remain consistently aligned, suggesting lower variance.
</div>


<script>
  const steps = {{ steps | jsonify }};
  const base = "{{ site.baseurl }}";

  function updateImages(stepIndex) {
    const step = steps[stepIndex];
    document.getElementById("stepLabel").textContent = step;

    // --- FIXED ALWAYS-VISIBLE ROW (batch0_vs_batch0) ---
    document.getElementById("fixed0").src =
      `${base}/assets/img/2026-04-27-ppo-batch-size/4vs32/update_${step}_batch0_vs_batch0.png`;

    document.getElementById("fixed1").src =
      `${base}/assets/img/2026-04-27-ppo-batch-size/32vs4/update_${step}_batch0_vs_batch0.png`;

    // --- SCROLLABLE ROWS: batch1..4 ---
    for (let i = 1; i <= 4; i++) {

      // left column = 4vs32
      document.getElementById(`scroll${(i-1)*2}`).src =
        `${base}/assets/img/2026-04-27-ppo-batch-size/4vs32/update_${step}_batch0_vs_batch${i}.png`;

      // right column = 32vs4
      document.getElementById(`scroll${(i-1)*2 + 1}`).src =
        `${base}/assets/img/2026-04-27-ppo-batch-size/32vs4/update_${step}_batch0_vs_batch${i}.png`;
    }
  }

  document.addEventListener("DOMContentLoaded", () => updateImages(3));
</script>


#### <span style="color:#2ca6a4;">▶</span> Effect of Batch Size on Bias

Bias describes consistent displacement between the estimated gradient and the true gradient:

$$\text{Bias} = \mathbb{E}[{G_B}] - G_T.$$

In PPO, bias arises mainly through how returns or advantages are estimated. Generalized Advantage Estimation rely on bootstrapping via Value function estimates, and can introduce bias if the value function is inaccurate, but provides lower variance. 

The effect of the rollout length $T$ on bias is as follows:

- When $T$ is small, advantage estimates rely more heavily on bootstrapping, which increases bias. This shows up in practice as noisier advantage estimates.

- When $T$ is large, advantage estimates use more actual returns and rely less on the value function, so bias becomes smaller.

Importantly, correlation between samples (the covariance term) affects variance but does not affect bias as it is determined by how advantages are computed.
<!-- but shift the expectation of the estimator away from the exact policy gradient. Additionally, PPO's clipped objective deliberately modifies the update direction to constrain policy deviation, introducing a controlled form of bias intended to improve stability. -->

### Bias and Variance Due to Mini-batches
<!-- In practice, PPO rarely updates from the full $N \times T$ batch at once.  -->
Mini-batches are used because the rollout buffer is typically large, and multiple updates per iteration improve computational efficiency. However, mini-batching also introduces sub-sampling variance, since each mini-batch represents only a portion of the available data. Clipped objectives modify the effective update direction, and introduce another form of bias in the gradient estimate.  
<!-- Moreover, because PPO updates the policy after each mini-batch, later mini-batches are evaluated under a slightly different policy than the one that generated their data. 
This policy drift contributes an additional source of bias, which the clipping mechanism partially mitigates. -->

  <!-- const base = "{{ site.baseurl }}"; -->

<div align="center">

  <img src="{{ site.baseurl }}/assets/img/2026-04-27-ppo-batch-size/mini_batch.png"
       alt="PPO bias-variance and policy update illustration"
       style="width:75%; max-width:500px;"/>

  <div class="explain-box" style="max-width:500px; margin-top:12px;">
    <strong>Figure:</strong> Mini-batch gradients: Mini-batch updates compute gradient estimates from subsets of the full
      batch. These estimates vary across mini-batches and accumulate over an
      update epoch, introducing sub-sampling variance and policy-drift-related bias.
  </div>

</div>

<!-- <div align="center">

  <img src="{{ site.baseurl }}/assets/img/2026-04-27-ppo-batch-size/Articulation.jpeg"
       alt="PPO bias-variance and policy update illustration"
       style="width:100%; max-width:900px;"/>

  <div class="explain-box" style="max-width:900px; margin-top:12px;">
    <strong>Figure:</strong> Conceptual illustration of policy updates and gradient
    estimation in PPO. (a) Policy Update:  Updating a policy alters the trajectory distribution. A policy update from iteration <i>i</i> to <i>i+1</i> shifts the probability mass over
      state–action trajectories, which is what the gradient aims to achieve. (b) Bias and variance: The estimated gradient <i>Ĝ</i> differs from the true gradient <i>G_T</i> due to sampling variability (variance) and systematic estimation error
      (bias). Both influence how the policy moves from iteration <i>i</i> to <i>i+1</i>. (c) Mini-batch gradients: Mini-batch updates compute gradient estimates from subsets of the full
      batch. These estimates vary across mini-batches and accumulate over an
      update epoch, introducing sub-sampling variance and policy-drift-related bias.
  </div>

</div> -->



<!-- ### Deconstructing the Gradient and Its Variance
We’ve set the stage: our goal is to manage the bias (from short rollouts $T$) and the variance (from correlation, or high $T$). Now, let’s look at the PPO math to see exactly where the variance comes from.

**PPO Gradient Equation**: The core of PPO's objective is to maximize the expected advantage, weighted by the probability ratio. The PPO objective (without the clipping mechanism initially) is:

$$L = \frac{1}{N T} \sum_{(s_t, a_t) \in \mathcal{D}}
\underbrace{\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}}_{r} \,
A^{\theta_{\text{old}}}(s_t, a_t)$$

(To keep things simple, we assume using entire collected batch $\mathcal{D}$ for a single policy update, ignoring mini-batches and hence clipping).

Where:

- $N \times T$ is our Total Batch Size.
- $r$ is the Probability Ratio, comparing the new policy $\pi_\theta$ to the old $\pi_{\theta_{\text{old}}}$.
- $A^{\theta_{\text{old}}}(s_t, a_t)$ is the Advantage Estimate calculated on the collected data.

Taking the gradient of this loss function gives us our gradient estimate, $G_B$:

$$G_B = \nabla_\theta L = \frac{1}{N T} \sum_{(s_t, a_t) \in \mathcal{D}}
\nabla_\theta \left[ r \cdot A^{\theta_{\text{old}}}(s_t, a_t) \right]$$

Since the advantage $A$ is constant with respect to the new policy $\theta$, and knowing that $\nabla_\theta r = r \cdot \nabla_\theta \log \pi_\theta(a_t \mid s_t)$, the gradient simplifies to:

$$\nabla_\theta L = \frac{1}{N T} \sum_{(s_t, a_t) \in \mathcal{D}}
r \, A^{\theta_{\text{old}}}(s_t, a_t) \, \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

For the initial full-batch gradient calculation, the old and new policies are very close, so $r \approx 1$. This gives us the simplest form of the gradient estimate:

$$G_B = \frac{1}{N T} \sum_{(s_t, a_t) \in \mathcal{D}} \underbrace{ A^{\theta_{\text{old}}}(s_t, a_t) \, \nabla_\theta \log \pi_\theta(a_t \mid s_t) }_{g_i}$$

This tells us something fundamental: Our policy gradient estimate $G_B$ is simply the mean of the individual policy gradients ($g_i$) calculated for every single step in our batch. -->



### What role does GAE-$\lambda$ play here?

$$A^{\text{GAE}}(\gamma, \lambda)_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ and corresponds to the one-step temporal-difference error. 

For $\lambda = 0$, this reduces to 
$$A^{\text{GAE}}(\gamma, 0)_t = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$
which only considers immediate rewards and bootstrapped values. This has low variance (the environment stochasiticity only affects a single step) but can be biased due to the bootstrapping from the value function, which is still being learned.

For $\lambda = 1$, it becomes the Monte Carlo return:
$$A^{\text{GAE}}(\gamma, 1)_t = \sum_{l=0}^{\infty} \gamma^l \delta_{t+l} = \sum_{l=0}^{\infty} \gamma^l r_{t+l} - V(s_t)$$
which incorporates the full sequence of rewards until the end of the episode. This is a accurate (unbiased) estimate of the advantage, as it is based on  actual observed returns, opposed to boostrapped values from value estimates. However, it has high variance because the returns can be highly variable, especially in stochastic and sensitive environments.

$\lambda$ thus controls how much we rely on immediate rewards vs long-term returns. Now, the advantages are computed for the collected batch of size $NT$, and "how far into the future" is limited by $T$, the rollout steps. When $T$ is much smaller than the episode length (especially for recurrent environments <d-cite key="schulman2017ppo"></d-cite>), the advantage estimates will be truncated, effectively creating the same effect as a lower $\lambda$. For $T > $ episode length, the advantage estimates is unaffected. 

<!-- #### Where does the bias and variance from GAE appear in the gradient variance decomposition?
$$G_B = \frac{1}{N T} \sum_{(s_t, a_t) \in \mathcal{D}} A^{\theta_{\text{old}}}(s_t, a_t) \, \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$ -->


<!-- By tuning $\lambda$, we can control the bias-variance trade-off in advantage estimation. A lower $\lambda$ (closer to 0) uses more immediate rewards, reducing variance but increasing bias. A higher $\lambda$ (closer to 1) incorporates longer-term rewards, which can increase variance due to the correlation of returns over time. -->



<!-- ## Final Thoughts -->
<!-- In this post, we explored how the structure of the batch size in PPO—specifically the trade-off between the number of parallel environments ($N$) and the number of steps per environment ($T$)—affects the variance of policy gradient estimates. -->
<!-- 
#### Note on Compute vs. Storage Constraints
It is to note that, while one may think that increasing num_envs is always better increasing num_rollouts, it is note that (a) if given constant storage, there is 1:1 relation in replacing additional rollout data with additional env data, however (b) if compute is assumed constant, then the 1:1 relation may not hold. In particular, the parallel data collection across n_envs is enabled by simulation and the compute constraint may limit how much n_env a simulation can efficiently process in parallel.  -->