<div align="center">

# think-fast-jev

**Models built to decide vs a model built to talk.**

[Headline](#headline) • [Results](#results) • [Dataset](#dataset) • [Method](#method) • [Limitations](#limitations) • [Reproduce](#reproduce)

</div>

A small, reproducible benchmark. It compares single-pass decision models ([Jev](https://typesafe.ai) and its open-weight
clone [Kev](https://github.com/jaredpalmer/kev)) with an LLM (Claude Haiku 5.5) on quality, calibration, latency and
cost.

**Powered by [CoinGecko](https://www.coingecko.com/en/api).** Not affiliated with or endorsed by CoinGecko or TypeSafe.

## Headline

**Jev** matched **Haiku** on router accuracy once advice questions are set aside (97.8%), at about 1/6 of the cost and
1/7 of the latency, with better calibration. **Kev-4B** was the fastest and cheapest, on a local network and with an estimated
cost, but it scored lower on routing (91.2%) and had the worst calibration. Every arm scored 100% on the guardrail, so
this set does not separate them.

## Results

> [!TIP]
> See [Metrics](#metrics) for what each value means.

### Quality

| Decision node | Arm | `accuracy` | `brier` | `agreement` | `order_stable` | `order_spread` |
|---|---|---|---|---|---|---|
| Router | Jev | 0.94 | 0.084 | n/a | 0.96 | 0.036 |
| Router | Kev-4B | 0.92 | 0.220 | n/a | 0.91 | 0.119 |
| Router | Claude Haiku 5.5 | 0.90 | 0.163 | 1.00 | n/a | n/a |
| Guardrail | Jev | 1.00 | 0.002 | n/a | 1.00 | 0.005 |
| Guardrail | Kev-4B | 1.00 | 0.032 | n/a | 1.00 | 0.032 |
| Guardrail | Claude Haiku 5.5 | 1.00 | 0.007 | 1.00 | n/a | n/a |

### Latency and cost

Latency is in ms and cost is in USD per 1,000 decisions. The error rate is 0 for every arm.

| Decision node | Arm | `model_p50` | `model_p95` | `e2e_p50` | `e2e_p95` | `cost_per_1k` |
|---|---|---|---|---|---|---|
| Router | Jev | n/a | n/a | 262 | 349 | $0.029 |
| Router | Kev-4B | 55 | 65 | 63 | 88 | $0.008 (estimate) |
| Router | Claude Haiku 5.5 | 1,647 | 1,961 | 1,898 | 2,227 | $0.172 |
| Guardrail | Jev | n/a | n/a | 249 | 309 | $0.016 |
| Guardrail | Kev-4B | 36 | 51 | 44 | 58 | $0.004 (estimate) |
| Guardrail | Claude Haiku 5.5 | 1,307 | 3,024 | 1,561 | 3,309 | $0.056 |

![Quality against latency per decision](results/charts/quality_latency.png)

![Cost per 1,000 decisions](results/charts/cost.png)

### Where the router misses are

Most of Haiku's misses are advice questions, such as "Should I sell my Dogecoin to pay off my car loan?". The label
answer for these is `unsupported`, and Haiku often selects `price`. An application that runs the guardrail first does not send them
to the router.

| Arm | Misses on advice questions (13) | Misses on all other questions (137) | Accuracy on the other 137 |
|---|---|---|---|
| Jev | 6 | 3 | 97.8% |
| Kev-4B | 0 | 12 | 91.2% |
| Claude Haiku 5.5 | 12 | 3 | 97.8% |

The raw decisions are in [`results/router.jsonl`](results/router.jsonl) and
[`results/guardrail.jsonl`](results/guardrail.jsonl), with the option-order runs in `results/*.permute.jsonl`. Match
`item_id` to `id` in [`data/queries.jsonl`](data/queries.jsonl). The full metrics are in
[`results/scores.md`](results/scores.md).

### What if you only trust confident answers?

Each model gives every answer a confidence score. Imagine a simple rule: trust the answer only if the model is sure
enough, and pass the question to a slower, costlier LLM if it is not. These are the 150 router questions under that rule:

| Rule | Model | Questions answered | Correct answers | Passed on |
|---|---|---|---|---|
| At least 90% sure | **Jev** | 126 | 126 | 24 |
| At least 90% sure | **Kev-4B** | 6 | 6 | 144 |
| At least 90% sure | **Claude Haiku 5.5** | 25 | 25 | 125 |
| At least 70% sure | **Jev** | 134 | 131 | 16 |
| At least 70% sure | **Kev-4B** | 71 | 71 | 79 |
| At least 70% sure | **Claude Haiku 5.5** | 127 | 122 | 23 |

At the 90% rule, all three were right every time they were that sure. The difference is how often they were that sure.
**Jev** would answer 126 questions itself and pass on only 24, while **Kev-4B** would pass on almost everything.

Haiku is itself an LLM, so it would not pass questions on. Its rows only compare how confident it is.

> [!NOTE]
> The benchmark did not run this setup. These counts come from `results/router.jsonl`, on the same small query set.

**Why is Kev-4B so unsure?** It ships with a fitted temperature of 2.41 (see `results/router.meta.json`), which flattens
its probabilities. A recalculation from the saved router probabilities suggests that a lower temperature, fitted on half
of the questions and scored on the other half, would improve its router Brier score from 0.22 to about 0.12 without
changing its accuracy. This is an estimate, and it assumes standard softmax scaling. The benchmark did not rerun Kev at
another temperature, so all results here use its shipped setting.

### What this means

- **Jev** is the strongest candidate in this test. On the router, it matched **Haiku** outside advice questions at about
  1/6 of the cost and 1/7 of the latency, with better calibration.
- **Kev-4B** gives up router accuracy and calibration for speed and cost. It was the fastest and cheapest, and it missed
  no advice questions, but it missed 12 of the other router questions. It also needs your own GPU, and its cost is an
  estimate.
- **Haiku** lost on speed and cost, but not clearly on accuracy. Most of its misses were advice questions, which a
  guardrail that runs first would stop.
- The guardrail comparison is still open. Every arm scored 100%, so a harder set is needed to separate them.

These results support a production trial of **Jev** for decisions like these, not a direct switch. The queries are not real
traffic, and this is one run on one day.

## Dataset

**[amadzai/think-fast-jev-queries](https://huggingface.co/datasets/amadzai/think-fast-jev-queries)** on Hugging Face
(CC-BY-4.0): 150 user questions, each with a router label and a guardrail label.

```python
from datasets import load_dataset

queries = load_dataset("amadzai/think-fast-jev-queries", split="test")
```

| Field | Description |
|---|---|
| `id` | The query ID. |
| `text` | The user question. |
| `router` | The router label: one of 14 options. |
| `guardrail` | The guardrail label: `allow`, `advice` or `off_topic`. |
| `source` | `seed` or `generated`. |

## Method

This benchmark tests two decisions that come up when answering crypto market questions: which data tool to call, and
whether to answer or to decline because the question asks for personal financial advice or is off topic.

Jev and Kev do not generate text. They return a probability for each option from one prefill pass.

### Decision nodes

Each node runs on its own: an item goes in, a decision comes out. There is no pipeline and no answer generation.

| Decision node | Input | Options |
|---|---|---|
| Router | A user question | 14: 13 data-tool areas and `unsupported` |
| Guardrail | A user question | 3: `allow`, `advice`, `off_topic` |

> [!NOTE]
> The code also has a third decision node, coin sector. Its results are not published in this repo.

Option names and descriptions are in [`src/bench/nodes.py`](src/bench/nodes.py). Every arm gets the same item text and
the same options in the same order.

The guardrail options:

| Option | Meaning | Example |
|---|---|---|
| `allow` | A question about crypto markets or crypto data that asks for facts. | What is Bitcoin dominance today? |
| `advice` | A request for personal financial advice: what to buy, sell or hold, or when. | ETH dropped 10% today. Should I buy the dip? |
| `off_topic` | A question that is not about crypto markets. | Can you recommend a good laptop for programming under $1,500? |

The benchmark only scores the selected option, and no step acts on it. In an application, `allow` could go on to an
answer, `advice` could get a reply that offers facts instead of advice, and `off_topic` could get a redirect.

### Arms

| Arm | Role | Model | How the benchmark calls it |
|---|---|---|---|
| Claude Haiku 5.5 | Incumbent | `claude-haiku-5-5` | Anthropic API. Thinking disabled, effort `low`. Structured output with one probability per option. 3 runs per item. |
| Jev | Candidate | `jev-latest` | TypeSafe System One API. One prefill pass. |
| Kev-4B | Candidate | `kev-latest` (checkpoint `jaredpalmer/kev-4b`) | The Kev server on a local GPU. Same API as Jev. |

### Procedure

- The runner calls one arm at a time and finishes one item before the next. The arm order rotates per item.
- The first 3 calls of each arm in each `bench run` are warm-up and do not count toward the latency metrics. For Kev, the
  keep-warm calls count as calls.
- A failed call gets up to 3 attempts. After that, the runner records an error.
- Haiku runs each item 3 times. The main scores use only its first run, so Haiku gets no voting advantage. A separate
  row shows the majority of the 3 runs.
- Quality scores use only the items that every arm answered validly.
- `bench permute` asks Jev and Kev each router and guardrail item under 6 option orders: the original order and 5
  shuffled orders.
- `bench throughput` measures Kev's decisions per hour with 32 calls at a time. This number gives Kev's cost.

### Metrics

| Metric | Meaning | What is good |
|---|---|---|
| `accuracy` | The share of items where the selected option is the label. | Higher. 1 is perfect. |
| `brier` | Calibration, as a multi-class Brier score. | Lower. 0 is perfect and 2 is the worst. |
| `agreement` | **Haiku:** the share of items where all 3 runs selected the same answer. | Higher. 1 means the runs always agree. |
| `order_stable` | **Jev and Kev:** the share of items where all option orders selected the same option. | Higher. 1 means order never changed the answer. |
| `order_spread` | **Jev and Kev:** the mean of the largest change of one option's probability between the option orders. | Lower. 0 means order had no effect. |
| `model_p50`, `model_p95` | Model time in ms, median and 95th percentile. | Lower. |
| `e2e_p50`, `e2e_p95` | End-to-end time in ms as the client measures it, median and 95th percentile. | Lower. |
| `cost_per_1k` | USD per 1,000 decisions. | Lower. |
| `error_rate` | The share of decisions with an error, from the last record per decision. Kev's rate includes failed keep-warm calls. | Lower. 0 means no errors. |

### Setup

- **Run date:** 2026-11-03.
- **Model versions:** `jev-1.14.0`, `claude-haiku-5-5`, and `kev-latest` with the checkpoint `jaredpalmer/kev-4b`.
- **Client:** a Mac in Malaysia. Jev and Haiku go over the internet. Kev goes over the local network.
- **Kev server:** RTX 5070 Ti (16 GB), WSL2 Ubuntu 24.04, PyTorch 2.8.0, `KEV_PREFIX_CACHE=1`.
- **Prices:** Jev $0.042 per 1M input tokens, output free. Haiku $0.10 per 1M input tokens and $0.50 per 1M output
  tokens. Kev: AWS g6.xlarge (1 x L4) on-demand in us-east-1, $0.8048 per hour ($1.0138 in ap-southeast-5, Malaysia).

## Limitations

### How the arms are set up

| Limitation | What it means |
|---|---|
| **Haiku runs at its lowest setting** | Thinking is disabled and effort is `low`, so that Haiku answers directly, like the System One models. A higher setting can change its quality, latency and cost. |
| **Haiku's probabilities are self-reported** | The Anthropic API gives no logprobs. Haiku writes one number per option. For the router and guardrail, the client scales them to sum to 1, and the Brier score uses the scaled numbers. |
| **Kev gets a keep-warm call** | On the test PC, Kev's model time rises from about 60 ms to about 180–290 ms after 8 s or more of idle. The runner sends one unrecorded call before each recorded call. A production server with gaps between requests pays this penalty. With keep-warm, about 7% of calls in a test still took 300–500 ms, which can reach Kev's p95. |
| **Option order can change the answer** | The [TypeSafe docs](https://docs.typesafe.ai/primitives/choice) say that Jev 1.13 can lean toward the first option. The main run uses one fixed order. `order_stable` and `order_spread` show the effect for Jev and Kev, not for Haiku. |

### How latency and cost are measured

| Limitation | What it means |
|---|---|
| **Model time differs per arm** | Jev returns none, so it has only end-to-end time. Haiku's is the `server-timing` origin time at the Cloudflare edge, which includes Anthropic's internal network. Kev's is its server's `latency_ms` field. |
| **End-to-end time depends on location** | Jev and Haiku go from Malaysia over the internet. Kev goes over the local network. |
| **Kev's cost is an estimate** | The throughput comes from an RTX 5070 Ti, not the L4 in the g6.xlarge. The Kev README lists an L40S or H100 for Kev-4B, not an L4. The cost assumes a GPU that is always busy. |

### What the data can and cannot show

| Limitation | What it means |
|---|---|
| **The router gaps are small** | With 150 queries, the accuracy gap between Jev and Haiku is not statistically significant (exact McNemar test, p = 0.11). |
| **The router scores depend on a label rule** | Advice questions are labelled `unsupported`. A different rule changes the ranking, as [the miss table](#where-the-router-misses-are) shows. |
| **The guardrail set is too easy** | Every arm scored 100%, so the set does not separate them. Only 22 of the 150 questions are `advice` or `off_topic`. |
| **The queries are not real traffic** | The seeds cover each option about equally, so the accuracy is not a production number. The set leaves out multi-step questions and price predictions, because they do not have one clear label. |
| **This is one run on one day** | Hosted models and their latency change over time. |

## Reproduce

You need Python 3.12 and [uv](https://docs.astral.sh/uv/). Add your API keys to `.env`:

| Setting | Used for |
|---|---|
| `TYPESAFE_API_KEY` | Jev |
| `ANTHROPIC_API_KEY` | Claude Haiku 5.5 |
| `KEV_BASE_URL`, `KEV_API_KEY` | Kev (optional, see below) |

Then run the benchmark. Replace `<your GPU>` with your own hardware:

```sh
uv sync
uv run bench check
for node in router guardrail; do uv run bench run --node "$node"; done
for node in router guardrail; do uv run bench permute --node "$node"; done
for node in router guardrail; do uv run bench throughput --node "$node"; done
uv run bench score results/router.jsonl results/guardrail.jsonl
uv run bench chart --note "Run $(date +%F). Kev-4B on <your GPU>."
uv run bench export
```

Scores go to `results/scores.csv` and `results/scores.md`, charts to `results/charts/`, and the dataset to
`dist/dataset/`.

> [!TIP]
> A rerun of `bench run` or `bench permute` skips the decisions that already have a result, so you can continue after a
> failure.

### Kev needs your own server

Kev is self-hosted, so you need your own GPU machine running the [Kev server](https://github.com/jaredpalmer/kev) with
`jaredpalmer/kev-4b`. Then set `KEV_BASE_URL` and `KEV_API_KEY` on the machine that runs `bench`.

> [!IMPORTANT]
> After a fresh start of the Kev server, do one throwaway run first, because the first calls are slow. Without the two
> Kev settings, the commands run without Kev, and `bench throughput` does not run.
