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

In this post, we'll dig into why the structure of the batch matters at all. In particular, we'll look at how increasing $N$ and increasing $T$ affect the bias and variance of PPO’s gradient estimates, theoretically and empirically. Code for the empirical analysis is provided [here](https://drive.google.com/drive/folders/1z8w_T0Ree9XwfBjK6yvL3zk4SlJZ_flG?usp=sharing)

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

<div style="border-left: 4px solid #335c67; padding: 0.5em 1em; background: #d7cade94;">
<strong>Note:</strong><br>
In this section, we assume that the gradient is computed using the <strong>full batch</strong> of collected data. This allows us to isolate and analyze the inherent sources of bias and variance that arise purely from the sampled batch itself.
<br>
As mentioned in previous section, PPO uses <strong>mini-batch</strong> stochastic gradient descent, which introduces additional sources of bias and variance due to sub-sampling. These effects will be discussed in the later section.
</div>
<br> 

#### <span style="color:#2ca6a4;">▶</span> Simplified Gradient Equation

If we assume that the gradient is computed using the <strong>full batch</strong> of collected data, we can isolate and analyze the inherent sources of bias and variance that arise purely from the sampled batch itself. Under this assumption, the gradient estimator takes the same form as the standard (vanilla) policy gradient:

<div style="border: 1px solid #4a79bc01; padding: 0.2em 0.3em; border-radius: 4px; background: #b1d5bbac;">
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
    <div class="explain-box" style="margin-top:8px;"> <strong>Figure</strong>  Policy Update:  Updating a policy alters the trajectory distribution. A policy update from iteration <i>i</i> to <i>i+1</i> shifts the probability mass over
      state–action trajectories, which is what the gradient aims to achieve. 
    </div> 
  </div> 
</div>

<div align="center" style="display:flex; justify-content:center; gap:16px;"> 
  <div style="flex:1; max-width:400px;"> <img src="{{ site.baseurl }}/assets/img/2026-04-27-ppo-batch-size/batch_gradient.png" alt="PPO bias-variance and policy update illustration" style="width:85%; border-radius:6px;"/> 
    <div class="explain-box" style="max-width:700px; margin-top:8px;">
      <strong>Figure</strong>
      Bias and variance: The estimated gradient <i>Ĝ</i> differs from the true gradient
      <i>G<sub>T</sub></i> due to sampling variability (variance) and systematic
      estimation error (bias). Both influence how the policy moves from iteration
      <i>i</i> to <i>i+1</i>.
    </div> 
  </div>
</div>


#### <span style="color:#2ca6a4;">▶</span> Effect of Batch Size on Variance

The variance of a mean is critical because it tells us how noisy $G_B$ is. High variance means we can't trust the update.

The total variance of our gradient estimate $G_B$ can be mathematically decomposed into two parts:

$$\text{Var}(G_B) = \text{Var}\Bigg(\frac{1}{N T} \sum_i g_i \Bigg) = \color{blue}{\underbrace{\frac{1}{(N T)^2} \sum_i \text{Var}(g_i)}_{\text{Term 1: Individual Variance}}} + \color{brown}{\underbrace{\frac{1}{(N T)^2} \sum_{i \neq j} \text{Cov}(g_i, g_j)}_{\text{Term 2: Covariance (Correlation)}}}$$


This decomposition connects directly to the noise sources discussed earlier and holds the key to the $N$ vs. $T$ trade-off:

- $\color{blue}{\text{Term 1: Individual Variance}}$

  This term reflects the variability of individual gradient contributions.
  In practice, this variability arises from both the stochastic policy $\pi_\theta(a_t \mid s_t)$ as well as noise in the advantage estimates,
  since each $g_i$ is scaled by a noisy advantage value. Importantly, higher individual variance does not necessarily translate into higher variance of the batch-averaged gradient, as gradient cancellation effects and covariance between samples also play a crucial role.

- $\color{brown}{\text{Term 2: Covariance (The Correlation Problem)}}$

  This is the term that makes RL different from supervised learning. The covariance $\mathrm{Cov}(g_i, g_j)$ need not be zero because $g_i$ and $g_{j}$ may originate from the same episode. Since states within a single episode are temporally correlated, the policy gradients computed from those states are also correlated.

**Impact:**  
When $T$ is **large**, the batch contains many highly correlated steps, leading to either large positive or large negative covariance term. This effectively reduces the *effective sample size* of the batch. This $\color{brown}{\text{increases the sampling noise}}$ because the effective sample size has been reduced. Therefore, a long horizon $T$ can keep $\mathrm{Var}(G_B)$ high. In addition, when $T$ is large, the $\color{blue}{\text{individual variance term also increases}}$, since advantage estimation has high variance over long horizons. (The role of $\lambda$ in GAE will be discussed later.)

When $T$ is small, advantage estimates have lower variance, and the effective sample size is less severely reduced than in the long-horizon case, since fewer temporally correlated states are included. Consequently, the overall variance of the gradient estimator is much lower.

**Take-away:**  
To aggressively reduce variance, we must minimize the covariance term. This requires breaking temporal correlations, which means collecting more independent trajectories—that is, increasing the number of parallel environments, $\mathbf{N}$.

**Some Pendulum-v1 Environment Experiments:**  
To investigate whether policy gradients are correlated along a trajectory, we use the Pendulum-v1 gym environment. PPO is trained with
<code style="background:#f6f2f8ff; padding:2px 4px; border-radius:4px;">num_envs = 4</code> and
<code style="background:#f6f2f8ff; padding:2px 4px; border-radius:4px;">num_steps = 256</code>,
which produces a total batch size of
<code style="background:#f6f2f8ff; padding:2px 4px; border-radius:4px;">batch_size = 4 × 256 = 1024</code> samples per update.

In the pendulum environment, episodes do not terminate due to task completion or failure; instead, they are truncated after a fixed horizon of 200 time steps <d-cite key="towers2024gymnasium"></d-cite>. Consequently, all trajectories have a fixed length of 200 steps.


To quantify the correlation between gradients computed from two data points, let us use the cosine similarity between gradients $g_i$ and $g_j$, defined as:
$$
  \cos(g_i, g_j)
  = \frac{g_i \cdot g_j}
        {\lVert g_i \rVert \, \lVert g_j \rVert }.
$$

At a certain training update, we analyze gradient correlation at two levels:
For a particular step in training, we compute following:
- **Within-trajectory correlation**: cosine similarity between gradients $g_t$ and $g_{t+k}$ sampled from the same trajectory at temporal lag $k$.
- **Across-trajectory correlation:** cosine similarity between gradients sampled from two different trajectories, each selected at random.
For within trajectory cosine similarity we test it for different values of lag k.

Gradients within the same trajectory are expected to be more strongly correlated, while gradients sampled across trajectories provide a baseline that serves as approximately independent samples.

The below figure compares the cosine similarity of gradients within a trajectory to that of gradients across different trajectories at two different points during training.
<div align="center">

  <img src="{{ site.baseurl }}/assets/img/2026-04-27-ppo-batch-size/gradient_similarity_399360.png"
       alt="gradient similarity distribution for step# 399360"
       style="width:100%; max-width:700px;"/>

  <div class="explain-box" style="max-width:700px; margin-top:12px;">
    <strong>Figure:</strong> Comparison of within-trajectory and across-trajectory gradient cosine similarity at global step 399360 for lag values \(k = 1, 3, 5, 7\).
  </div>

</div>

<div style="height:3rem;"></div>


<div align="center">

  <img src="{{ site.baseurl }}/assets/img/2026-04-27-ppo-batch-size/gradient_similarity_450560.png"
       alt="gradient similarity distribution for step# 450560"
       style="width:100%; max-width:700px;"/>

  <div class="explain-box" style="max-width:700px; margin-top:12px;">
    <strong>Figure:</strong> Comparison of within-trajectory and across-trajectory gradient cosine similarity at global step 450560 for lag values \(k = 1, 5, 10, 40\).
  </div>

</div>


The strong correlation we observed within a trajectory suggests that <code style="background:#f6f2f8ff; padding:2px 4px; border-radius:4px;">num_steps</code> play a significant role in gradient variance. To study this effect more systematically, let us analyze two settings: `num_envs = 4`, `num_steps = 256` (long trajectories) and `num_envs = 16`, `num_steps = 64` (short trajectories). Let us focus on training step 450560 now, as the correlation effects remain visible even at large lags (i.e. $k$=40).

For each setting, we randomly sample five batches from the rollout buffer (`batch0` … `batch4`). From each batch, we compute the **mean gradient vector**, and then compute pairwise cosine similarities:

$$
\cos(\bar{g}^{(i)}, \bar{g}^{(j)}) 
= \frac{\bar{g}^{(i)} \cdot \bar{g}^{(j)}}{\lVert \bar{g}^{(i)} \rVert \, \lVert \bar{g}^{(j)} \rVert }.
$$

This allows us to visualize whether the mean gradient from one batch tends to point in the same direction as the mean gradient from another batch.  

Importantly, the differences we observe between batches represent the overall variance of the gradient estimator. This includes both the per-sample variance (noise in each $(g_i)$) and the additional covariance created by temporal correlations within trajectories.

In the **long-horizon case (4×256)**, the off-diagonal cells of the cosine similarity heatmap shows that mean gradients from different batches can be strongly **positively** or **negatively** correlated. Negative correlation indicates that the gradient estimate from one batch points in the *opposite* direction of another batch's mean gradient, revealing substantial variance in the gradient estimator.
In contrast, in the **short-horizon case (16x64)**, while negatively correlated batch mean gradients do occur, they are noticeably less severe than in the long-horizon setting, indicating reduced gradient variance. 



<div align="center">

  <img src="{{ site.baseurl }}/assets/img/2026-04-27-ppo-batch-size/batch_grad_var.png"
       alt="analysis of variance in gradient for step# 450560"
       style="width:100%; max-width:700px;"/>

  <div class="explain-box" style="max-width:700px; margin-top:12px;">
    <strong>Figure:</strong> Cosine similarity between batch mean gradients for short-horizon and long-horizon settings.
The long-horizon case (4×256) on the right exhibits stronger positive and negative off-diagonal correlations, suggesting increased variability in gradient estimates.
  </div>

</div>



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
    border: 1px solid #ffffffff;
    border-left: 4px solid #335c67;
    padding: 12px 16px;
    background: #b1d5bbac;
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
      update epoch, introducing sub-sampling variance and policy drift related bias.
  </div>

</div>


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

Based on this reasoning, one might expect the impact of $\lambda$ on gradient variance to depend strongly on the rollout horizon $T$. However, empirical results do not show a clean separation between short-horizon and long-horizon settings. Instead, varying $\lambda$ produces qualitatively similar effects across both regimes.

In particular, changing $\lambda$ does not eliminate temporal correlation between gradients sampled along a trajectory. Even for $\lambda = 0$, gradients remain correlated across time due to shared state visitation and common policy. What $\lambda$ primarily influences is directional coherence: as $\lambda$ increases, per-sample gradient directions become more variable, which weakens alignment within a batch.

Across both short-horizon and long-horizon settings, the following patterns emerge:

- Temporal correlation between gradients persists for all values of $\lambda$.

- Increasing $\lambda$ increases directional variability in per-sample gradients.

- As a consequence, batch mean gradients become less directionally aligned, even though individual gradient samples remain temporally correlated.

<div align="center">

<img src="{{ site.baseurl }}/assets/img/2026-04-27-ppo-batch-size/lambda_similarity.png" alt="lambda similarity" style="width:100%; max-width:700px;"/>

<div class="explain-box" style="max-width:700px; margin-top:12px;"> <strong>Figure:</strong> Within-trajectory and across-trajectory cosine similarity for the 4×256 setup. Even for $\lambda = 0$ and $\lambda = 1$, gradients remain correlated at large temporal lags (here, $k = 40$). </div> </div> <div style="height:2rem;"></div> <div align="center">

<img src="{{ site.baseurl }}/assets/img/2026-04-27-ppo-batch-size/lambda_mean_similarity.png" alt="lambda mean similarity" style="width:100%; max-width:700px;"/>

<div class="explain-box" style="max-width:700px; margin-top:12px;"> <strong>Figure:</strong> Cosine similarity between each per-sample gradient and the batch mean gradient, computed as $ \cos(g_i, \bar{g}^{(i)}) = \frac{g_i \cdot \bar{g}^{(i)}}{\lVert g_i \rVert \, \lVert \bar{g}^{(i)} \rVert}. $ As $\lambda$ increases, the distribution becomes more concentrated around zero, indicating increased variability in per-sample gradient directions. </div> </div>

Varying the GAE parameter $\lambda$ therefore primarily affects the directional variability of per-sample gradients within a batch rather than removing temporal correlation altogether. For larger values of $\lambda$, gradients tend to point in more diverse directions, leading to partial cancellation when averaged. As a result, batch mean gradients may appear less extreme and, in some cases, more consistently aligned across batches.

For smaller values of $\lambda$, per-sample gradients are often more directionally consistent within a batch. This can produce strongly aligned batch mean gradients, which may result in either low or high batch-to-batch variance depending on the rollout and trajectory correlations.

<div align="center">

<img src="{{ site.baseurl }}/assets/img/2026-04-27-ppo-batch-size/batch_mean_lambda.png" alt="batch mean gradient" style="width:100%; max-width:700px;"/>

<div class="explain-box" style="max-width:700px; margin-top:12px;"> <strong>Figure:</strong> Pairwise cosine similarity between batch mean gradients. For $\lambda = 0$, batch-to-batch variance can be either low or high depending on the rollout. For larger $\lambda$, cancellation within batches can reduce extreme batch-to-batch variance. </div> </div>


### Wrapping-up
This post examined how the rollout length $T$ and the number of parallel environments $N$ affects the variance of gradient estimates. Longer rollouts introduce strong temporal correlations between gradients computed from neighboring time steps, which can substantially increase variance despite a fixed total batch size. Increasing the number of parallel environments instead reduces these correlations by collecting more independent trajectories.

In practice, PPO implementations further mitigate correlation effects by shuffling the rollout buffer and performing updates using mini-batches. This randomization breaks up temporally adjacent samples and partially reduces gradient correlation. Nevertheless, the choice of $N$ and $T$ directly influences the stability of the learning dynamics.
