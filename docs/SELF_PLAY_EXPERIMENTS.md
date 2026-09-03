# Self-Play Experiments and Final Design

## Scope

Before the released dual-loop method, we tested several ways to optimize a
user-side policy jointly with a Qwen3-8B assistant. Every run used multi-turn
emotion feedback from a frozen DeepSeek-V3 scorer. The main references were:

- RLVER with a frozen DeepSeek-V3 user: training emotion reached about 0.44 at
  step 50, 0.55 at step 100, and above 0.60 later.
- Frozen Qwen3-8B user baseline: about 0.38 at step 50, with a transient peak
  near 0.65 around step 100 before declining.
- Original joint V2DR with continuous emotion reward and user loss weight 0.3:
  about 0.30 at step 50 and 0.55 at step 270.

These values are training diagnostics and are not interchangeable with the
held-out SAGE Overall values reported for the released method.

## Tested Variants

| Variant | User-side objective | User loss | Main observation |
|---|---|---:|---|
| Gaussian | `pass * Gaussian(scene pass rate; center=0.5)` | 1.0 | Pass rate was usually zero; 22% of steps had zero user advantage. Stopped at step 35. |
| Gaussian Additive | `0.8 * dense threshold proximity + 0.2 * Gaussian` | 1.0 | Training emotion rose slowly from about 0.16 to 0.26; 27/114 steps still had zero user advantage. Stopped at step 118. |
| Dense User | Continuous proximity to emotion threshold 60 | 0.3 | Stable gradient but no capability gain; training emotion remained near 0.21-0.22 through step 153. |
| USP v1 | Hand-written engagement/information/fidelity/naturalness proxy plus challenge | 0.1 | Proxy stayed in a narrow 0.63-0.68 range. Held-out SAGE was 48 at both step 0 and step 50. |
| USP v2 | Expanded response/emotion/information/fidelity/effect proxy | 0.1 | Training emotion had transient peaks, but held-out SAGE remained 48 at step 50. |
| HiddenUSP Dense | `0.7 * (1-emotion) + consistency - penalties`; hidden and visible user tokens | 0.3 | The user became increasingly adversarial and training emotion fell to 0.07. Hidden-state match stayed zero. Stopped at step 16. |
| HiddenUSP OriginalSR | Pass-rate balance, threshold miss, consistency, penalties | 0.3 | Pass-rate balance alternated between 0 and 1; hidden-state match stayed zero; SAGE remained 48 at step 15. |
| HiddenUSP Group | One shared half-pass/half-fail reward for a scene; hidden state only | 0.0 for visible user | Training emotion declined from about 0.34 to 0.27 by step 34; the run did not establish an advantage. |
| SEAD PID-Profile | Controller changes profile difficulty; user tokens frozen | 0.0 | This isolated adaptive environment allocation from user policy learning and motivated the released controller design. |

Runs marked as stopped were terminated once their optimization signal was
clearly sparse, misaligned, or non-transferable. Pending intermediate runs are
not presented as positive evidence.

## Why Joint User Training Failed

### One terminal score cannot identify a good user action

The SAGE terminal emotion score measures the assistant's effect on the complete
interaction. It does not distinguish whether a low score was caused by a
plausible difficult user turn, an incoherent user turn, stochastic assistant
generation, or scorer noise. Broadcasting a transformed terminal score to all
user tokens therefore gives weak and confounded credit.

### A 50% pass rate is not itself evidence of learnability

Half-pass/half-fail is useful only when the condition is held fixed and the
statistic is accumulated over time. As a direct user reward, a batch pass rate
near 0.5 can be produced by generation variance or an unstable simulator. It
does not tell an individual user trajectory which behavior should become more
likely.

### Joint optimization makes the environment non-stationary

The assistant must learn a long-horizon emotional-support policy while the
user's language, disclosure pattern, and acceptance behavior change at the
same time. The verifier still evaluates trajectories under its original
semantics. This moving target degraded sample efficiency and made training
reward less predictive of held-out SAGE.

### Direct opposition creates the wrong game

Objectives based on `1 - emotion` reward the user for making the assistant
fail. The cheapest strategy is not necessarily a realistic boundary case; it
can be refusal to improve, inconsistent affect, hidden-state leakage, or an
unresolvable conversation. The HiddenUSP Dense run exhibited this collapse.

### Hand-written proxies were easy to satisfy

Length, lexical diversity, emotion words, profile overlap, and leak blacklists
were nearly constant across generated trajectories and weakly coupled to
empathetic learnability. Their optimization changed training diagnostics
without changing held-out SAGE.

### The latent control variable was not behaviorally grounded

The proposed HiddenUSP JSON state received reward, but visible user behavior
did not reliably realize that state. A latent action that the downstream actor
can ignore has no stable causal path to the final outcome.

## Released Resolution

The final method freezes the user simulator and verifier. Instead of training
user tokens, it defines 24 explicit simulator-only interaction states over:

- disclosure openness: 3 levels;
- emotional activation: 2 levels;
- relational trust: 4 levels.

Each state has deterministic natural-language realization rules. The original
persona, event, and support intention remain fixed. A held-out audit recovered
disclosure, activation, and trust from generated dialogues with 86.7%, 92.5%,
and 82.5% accuracy; 92.5% of trajectories retained the original scenario
meaning.

For one scenario group, all four GRPO rollouts share the same interaction
state. The assistant is optimized with continuous terminal SAGE emotion. The
outer controller alone thresholds outcomes at 50, records one pass-rate
observation per complete group, and favors intent-state units whose stabilized
historical pass rate is near 0.5. Hierarchical evidence sharing, uncertainty
exploration, and 10% uniform rehearsal keep sparse estimates from collapsing
the sampling distribution.

This preserves the useful part of the competence-boundary idea while removing
the unreliable user-token policy gradient. In protocol-matched evaluation, the
released method obtains a three-run SAGE mean of 79.24, compared with 72.01 for
uniform emotion-reward RL under the same rollout budget.

## Practical Takeaway

For this task, the successful object of adaptation is the distribution of
controlled user conditions, not the user's natural-language policy. A user
condition becomes valuable when the current assistant shows mixed verified
outcomes under repeated, semantically fixed rollouts. The user simulator should
continue to realize that condition consistently; it should not be rewarded for
lowering the assistant's score.
