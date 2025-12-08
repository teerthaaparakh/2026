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
  - name: Clearing up the Terminology
  - name: Data Distribution
  - name: Bias and Variance in Policy Gradients
  - name: Sources of variance in RL
  - name: Using Mini-batches
  - name: Deconstructing the Gradient and Its Variance
  - name: Unpacking the Variance

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

## Introduction
In this post, we are going to explore a common dilemma we face when tuning PPO hyperparameters: constructing the batch size.

It's easy to think of batch size as just a single number, but in PPO, it is actually the product of two distinct levers we can pull:

- The number of parallel environments ($N$).
- The number of steps collected per environment ($T$).

Does it matter if we reach a batch size of 2,048 by running 2 environments for 1,024 steps, or by running 1,024 environments for 2 steps?

We will look at how this choice affects the bias and variance of our gradient estimation.

We will keep the heavy equations to a minimum and rely on illustrations to build an intuition for what is happening under the hood.

## Clearing up the Terminology: Defining Batch vs. Mini-Batch
If you have ever looked at PPO implementations across different libraries, you might have noticed that "batch size" doesn't always mean the same thing.

To keep things clear in this post, here is the hierarchy we will use:

- Rollout Buffer (Total Batch Size): This is the full dataset collected before a policy update. It is the product of the number of parallel environments ($N$) and the number of steps collected per environment ($T$).

$$\text{Total Batch Size} = N \times T$$

- Mini-Batch: This is the subset of the Rollout Buffer used for a single Gradient Descent step. We shuffle the Rollout Buffer and chop it into these smaller pieces to perform the updates.

**Why is this confusing?**

The confusion stems from the fact that popular libraries label these variables differently. For instance, Stable Baselines3 <d-cite key="stable-baselines3"></d-cite>, Dopamine <d-cite key="castro18dopamine"></d-cite>, OpenAI Baselines (PPO1) <d-cite key="baselines"></d-cite>, and Ray RLlib <d-cite key="liang2018rllib,liang2021rllib"></d-cite> often refer to the mini-batch variable simply as batch.

This naming convention can lead to subtle but critical implementation errors. As pointed out by <d-cite key="shengyi2022the37implementation"></d-cite>:

<blockquote style="font-size:0.9em; margin:6px 0; padding-left:12px; border-left:3px solid #ccc;">Some common mis-implementations include (1) always using the whole batch for the update, and (2) implementing mini-batches by randomly fetching from the training data, which does not guarantee all training data points are fetched.</blockquote>

If we aren't careful, we might think we are tuning the total experience collected ($N \times T$), when we are actually just changing how much data fits into a single GPU update step.


### Data Distribution

In reinforcement learning, a trajectory $\tau = (s_0,a_0,s_1,a_1,\dots,s_T)$ is generated jointly by the policy and environment dynamics. Its probability under policy $\pi_\theta$ and transition matrix $P$ is:

$$
P(\tau\mid \pi_\theta) = p(s_0)\, \prod_{t=0}^{T-1} \pi_\theta(a_t\mid s_t)\, P(s_{t+1}\mid s_t,a_t).
$$

As the policy updates online, this trajectory distribution shifts between updates. PPO optimizes expected return using samples from the current distribution; near convergence, updates become small and the distribution stabilizes.

Our focus here is not convergence, but how structuring a fixed total batch $NT$ (many short vs. few long rollouts) influences the variance of gradient estimates.
{% include figure.liquid path="assets/img/2026-04-27-ppo-batch-size/data_distribution.png" class="img-fluid" %}

### Bias and Variance in Policy Gradients
Assumption for this section: gradients refer to those computed from the full collected batch per update ($N$ environments $\times$ $T$ steps), not mini-batches.

We don’t see all possible trajectories—only a sample from the current distribution—so our policy gradient estimate is noisy, similar to stochastic gradients in supervised learning. Variance describes how much the gradient would change if we re-sampled another batch from the same distribution; larger effective sample sizes lower this variance.

When variance is high, successive updates can point in very different directions, which makes learning unstable. Our goal here is to reason about one important source of variance: how rollout length $T$ (and thus temporal correlation) interacts with the number of environments $N$ when $NT$ is fixed.

Bias can arise through how we estimate advantages and values from limited or skewed data (e.g., sparse rewards or highly stochastic transitions), nudging updates off the true gradient direction. Bias is intuitively a systematic drift of the mean gradients from the true gradients arising due to insufficient knowledge of the environment, introduced via incorrect estimates of the value function.

<!-- **[placeholder for updated figure]** -->
{% include figure.liquid path="assets/img/2026-04-27-ppo-batch-size/batch_true_gradient.png" class="img-fluid" %}

We established that noise in gradient updates makes training unstable. To control it, we must understand its sources.

Unlike supervised learning, where noise usually just comes from drawing random samples from a fixed dataset, Reinforcement Learning adds other sources of inherent variability: environment dynamics and the tricky problem of credit assignment.

### Sources of variance in RL
We can think of our sampled gradient estimate ($G_B$) as the ideal gradient ($G_*$) plus these various sources of error:

$$G_B \approx G_* + \text{sampling noise} + \color{blue}{\text{trajectory variability}} + \color{brown}{\text{credit assignment noise}},$$

Sampling Noise: This is the familiar variance from using a finite batch size, common in all stochastic gradient methods.

$\color{blue}{\text{Trajectory Variability}}$: This is noise inherent to the RL loop. It accounts for how much trajectories can differ due to environment stochasticity, sensitive dynamics (a small change in $s_0$ leads to a vastly different $s_T$), or policy stochasticity.

- Mitigation: Increasing the effective sample size—meaning increasing the number of independent trajectories ($N$).

$\color{brown}{\text{Credit Assignment Noise}}$: This reflects the difficulty of figuring out which early actions caused a distant reward, especially with sparse or delayed feedback.

- Mitigation: Improving the advantage estimation, primarily by adjusting the Generalized Advantage Estimation (GAE) parameters or using longer rollouts ($T$).

<!-- 
While sampling noise is present in both supervised learning and RL, RL adds more moving parts. A simple mental model is:

$$
G_B \approx G_* + \text{sampling noise} + \color{blue}{\text{trajectory variability}} + \color{brown}{\text{credit assignment noise}},
$$

where \(G_*\) is the ideal gradient. The blue term groups environment stochasticity/sensitivity and policy stochasticity: trajectories can vary a lot even under the same policy. The brown term reflects the difficulty of attributing rewards to earlier actions, especially with sparse or delayed rewards. Increasing the effective sample size (more independent trajectories) mainly helps with the blue term; improving advantage estimation (e.g., suitable GAE parameters) helps with the brown term. -->

### Practical Consideration: using Mini-batches

The full batch ($N\times T$) is often too large for a single gradient update due to memory limits or compute cost. To address this, we split the batch into mini-batches. Intuitively, let $G_B$ denote the gradient computed from the full batch. Let $G_{MB}$ denote the gradient from a mini-batch. In expectation, $G_{MB}$ updates parameters in approximately the same direction as $G_B$, as illustrated below.

<div style="width:55%; margin: 0 auto;">
{% include figure.liquid path="assets/img/2026-04-27-ppo-batch-size/mini_batch.png" class="img-fluid" %}
</div>

Mini-batches introduce additional variance because each mini-batch is a smaller sample of the full batch, yielding noisier gradient estimates. To isolate the effects we care about, in our analysis we compute gradients from the entire collected batch per update and avoid policy changes during collection.

Under this setup, the probability ratio $r$ (defined below) is approximately 1 across the batch, since $\theta$ is not updated between samples. In practice, PPO uses clipping to enable multiple mini-batch updates while preventing the policy from changing too much between mini-batch steps.

### Deconstructing the Gradient and Its Variance
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

This tells us something fundamental: Our policy gradient estimate $G_B$ is simply the mean of the individual policy gradients ($g_i$) calculated for every single step in our batch.

### Unpacking the Variance

The variance of a mean is critical because it tells us how noisy $G_B$ is. High variance means we can't trust the update.

The total variance of our gradient estimate $G_B$ can be mathematically decomposed into two parts:

$$\text{Var}(G_B) = \text{Var}\Bigg(\frac{1}{N T} \sum_i g_i \Bigg) = \underbrace{\frac{1}{(N T)^2} \sum_i \text{Var}(g_i)}_{\text{Term 1: Individual Variance}} + \underbrace{\frac{1}{(N T)^2} \sum_{i \neq j} \text{Cov}(g_i, g_j)}_{\text{Term 2: Covariance (Correlation)}}$$

This decomposition holds the key to the $N$ vs. $T$ trade-off:

- Term 1: Individual Variance

This is the noise coming from each step $g_i$. Since the sum is divided by $(NT)^2$, this term shrinks rapidly as the Total Batch Size ($NT$) increases. If every sample were independent, this is all we would have to worry about.

- Term 2: Covariance (The Correlation Problem)

This is the term that makes RL different from Supervised Learning. $\text{Cov}(g_i, g_j)$ is not zero because $g_i$ and $g_{i+1}$ come from consecutive steps in the same environment. Consecutive states are highly related, so the policy gradients derived from them are also highly correlated.

The Impact: When $T$ is large, we have many highly correlated steps in our batch, leading to a large positive covariance term. This effectively lowers the "effective sample size" of our batch. Even if $NT$ is large, if $T$ is too long, the covariance term can keep the overall $\text{Var}(G_B)$ high.

The take-away is clear: To aggressively reduce variance, we must minimize the covariance term. This requires breaking the temporal correlation, which means we need more independent starting points—that is, increasing the number of parallel environments, $\mathbf{N}$.
<style>
.crop {
  width: 120px;      /* final visible width  */
  height: 120px;     /* final visible height */
  overflow: hidden; 
}

.crop img {
  width: 200px;      /* scale image */
  height: auto;
  margin-left: -40px; /* shift crop window */
  margin-top: -20px;
}
</style>

<div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px;">
  <div class="crop"><img src="assets/img/2026-04-27-ppo-batch-size/4vs32/update_500_batch0_vs_batch0.png"></div>
  <div class="crop"><img src="assets/img/2026-04-27-ppo-batch-size/4vs32/update_500_batch0_vs_batch1.png"></div>
  <div class="crop"><img src="assets/img/2026-04-27-ppo-batch-size/4vs32/update_500_batch0_vs_batch2.png"></div>
  <div class="crop"><img src="assets/img/2026-04-27-ppo-batch-size/4vs32/update_500_batch0_vs_batch3.png"></div>
  <div class="crop"><img src="assets/img/2026-04-27-ppo-batch-size/4vs32/update_500_batch0_vs_batch4.png"></div>

  <div class="crop"><img src="assets/img/2026-04-27-ppo-batch-size/32vs4/update_500_batch0_vs_batch0.png"></div>
  <div class="crop"><img src="assets/img/2026-04-27-ppo-batch-size/32vs4/update_500_batch0_vs_batch1.png"></div>
  <div class="crop"><img src="assets/img/2026-04-27-ppo-batch-size/32vs4/update_500_batch0_vs_batch2.png"></div>
  <div class="crop"><img src="assets/img/2026-04-27-ppo-batch-size/32vs4/update_500_batch0_vs_batch3.png"></div>
  <div class="crop"><img src="assets/img/2026-04-27-ppo-batch-size/32vs4/update_500_batch0_vs_batch4.png"></div>
</div>