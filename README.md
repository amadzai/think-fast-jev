<div align="center">

# think-fast-jev

**Models built to decide vs a model built to talk.**

[Headline](#headline) • [Results](#results) • [Dataset](#dataset) • [Method](#method) • [Limitations](#limitations) • [Reproduce](#reproduce)

</div>

A small, reproducible benchmark. It compares single-pass decision models ([Jev](https://typesafe.ai) and its open-weight
clone [Kev](https://github.com/jaredpalmer/kev)) with two LLMs (Claude Haiku 5.5 and GPT-6 Luna) on quality, calibration,
latency and cost.

**Powered by [CoinGecko](https://www.coingecko.com/en/api).** Not affiliated with or endorsed by CoinGecko or TypeSafe.

## Headline

**GPT-6 Luna** was the most accurate arm on the router (97.8%), with the best calibration. **Jev** was second (93.3%), at
about 1/3 of Luna's cost and 1/6 of its median latency. Outside advice questions, the gap is small: Luna missed 1 of 157
and Jev missed 2. **Kev-4B** was the fastest and cheapest, but it missed the most router questions outside advice and had
the worst calibration. On the guardrail, all four arms scored 98.9% or more.

## Results

> [!TIP]
> See [Metrics](#metrics) for what each value means.

### Quality

| Decision node | Arm | `accuracy` | `brier` | `agreement` | `order_stable` | `order_spread` |
|---|---|---|---|---|---|---|
| Router | Jev | 0.93 | 0.104 | n/a | 0.96 | 0.037 |
| Router | Kev-4B | 0.91 | 0.229 | n/a | 0.90 | 0.119 |
| Router | Claude Haiku 5.5 | 0.89 | 0.196 | 0.98 | n/a | n/a |
| Router | GPT-6 Luna | 0.98 | 0.054 | 0.98 | n/a | n/a |
| Guardrail | Jev | 1.00 | 0.003 | n/a | 0.99 | 0.006 |
| Guardrail | Kev-4B | 0.99 | 0.068 | n/a | 0.99 | 0.052 |
| Guardrail | Claude Haiku 5.5 | 1.00 | 0.014 | 0.99 | n/a | n/a |
| Guardrail | GPT-6 Luna | 0.99 | 0.011 | 0.99 | n/a | n/a |

### Latency and cost

Latency is in ms and cost is in USD per 1,000 decisions. The error rate is 0 for every arm.

| Decision node | Arm | `model_p50` | `model_p95` | `e2e_p50` | `e2e_p95` | `cost_per_1k` |
|---|---|---|---|---|---|---|
| Router | Jev | n/a | n/a | 277 | 407 | $0.030 |
| Router | Kev-4B | 56 | 67 | 64 | 82 | $0.008 (estimate) |
| Router | Claude Haiku 5.5 | 1,091 | 1,585 | 1,378 | 1,860 | $0.175 |
| Router | GPT-6 Luna | 1,277 | 3,360 | 1,612 | 3,650 | $0.086 |
| Guardrail | Jev | n/a | n/a | 266 | 348 | $0.017 |
| Guardrail | Kev-4B | 50 | 60 | 58 | 73 | $0.004 (estimate) |
| Guardrail | Claude Haiku 5.5 | 882 | 1,560 | 1,156 | 1,842 | $0.058 |
| Guardrail | GPT-6 Luna | 762 | 1,813 | 1,077 | 2,167 | $0.030 |

![Quality against latency per decision](results/charts/quality_latency.png)

![Cost per 1,000 decisions](results/charts/cost.png)

### Where the router misses are

Most router misses are advice questions. The label answer for these is `unsupported`. Haiku often selects `price`, for
example for "Should I sell my Dogecoin to pay off my car loan?". Jev and Luna miss advice questions that mention a chart or
funding rates, such as "Given Bitcoin's 30-day price chart, should I sell before the weekend?", and mostly
select `market_chart` or `derivatives`. An application that runs the guardrail first does not send these questions to the
router.

| Arm | Misses on advice questions (23) | Misses on all other questions (157) | Accuracy on the other 157 |
|---|---|---|---|
| Jev | 10 | 2 | 98.7% |
| Kev-4B | 4 | 13 | 91.7% |
| Claude Haiku 5.5 | 16 | 4 | 97.5% |
| GPT-6 Luna | 3 | 1 | 99.4% |

The guardrail misses, all on one run:

- Kev-4B selected `allow` for 2 crypto-themed tasks: "Translate 'buy the dip' into Japanese." and "Draft a tweet
  announcing that our NFT collection launches on Friday."
- Luna selected `advice` for "Should I be worried that Bitcoin dominance dropped? What is the dominance right now?"

The raw decisions are in [`results/router.jsonl`](results/router.jsonl) and
[`results/guardrail.jsonl`](results/guardrail.jsonl), with the option-order runs in `results/*.permute.jsonl`. Match
`item_id` to `id` in [`data/queries.jsonl`](data/queries.jsonl). The full metrics are in
[`results/scores.md`](results/scores.md).

### What if you only trust confident answers?

Each model gives every answer a confidence score. Imagine a simple rule: trust the answer only if the model is sure
enough, and pass the question to a slower, costlier LLM if it is not. These are the 180 router questions under that rule:

| Rule | Model | Questions answered | Correct answers | Passed on |
|---|---|---|---|---|
| At least 90% sure | **Jev** | 149 | 147 | 31 |
| At least 90% sure | **Kev-4B** | 4 | 4 | 176 |
| At least 90% sure | **Claude Haiku 5.5** | 28 | 28 | 152 |
| At least 90% sure | **GPT-6 Luna** | 166 | 162 | 14 |
| At least 70% sure | **Jev** | 162 | 156 | 18 |
| At least 70% sure | **Kev-4B** | 89 | 88 | 91 |
| At least 70% sure | **Claude Haiku 5.5** | 153 | 141 | 27 |
| At least 70% sure | **GPT-6 Luna** | 176 | 172 | 4 |

At the 90% rule, **Jev** would answer 149 questions itself and get 2 wrong. **Luna** would answer 166 and get 4 wrong. 3 of
Luna's 4 are advice questions that mention a chart or funding rates. **Kev-4B** would pass on almost everything.

Haiku and Luna are LLMs themselves, so they would not pass questions on. Their rows only compare how confident they are.

> [!NOTE]
> The benchmark did not run this setup. These counts come from `results/router.jsonl`, on the same small query set.

**Why is Kev-4B so unsure?** It ships with a fitted temperature of 2.41 (see `results/router.meta.json`), which flattens
its probabilities. The benchmark uses this shipped setting.

### What this means

- **Luna** gave the best router quality: the highest accuracy, the best calibration and the fewest misses. It was also
  the slowest arm at p95 (3.65 s), and on the router it cost about 3 times as much as Jev.
- **Jev** is the strongest candidate. Outside advice questions, it was close to Luna on the router (98.7% against
  99.4%), and it scored 100% on the guardrail. It answered in about 1/6 of Luna's median time, at about 1/3 of the cost.
- **Kev-4B** gives up router accuracy and calibration for speed and cost. It was the fastest and cheapest, but it missed
  13 router questions outside advice. It also needs your own GPU, and its cost is an estimate.
- **Haiku** was the most expensive arm, and it missed the most advice questions on the router. Outside advice
  questions, it scored 97.5%.

On quality alone, the candidates do not beat the best LLM in this test. **Jev** gives up a small amount of router
accuracy for much lower latency and cost. Whether that trade is worth it depends on how much latency and cost matter in
the application. The queries are not real traffic, and this is one run on one day.

## Changes since the first run

The first run (2026-11-03) is at the tag
[`v1-2026-11-03`](https://github.com/amadzai/think-fast-jev/tree/v1-2026-11-03). This run:

- Adds GPT-6 Luna as a second incumbent.
- Adds 30 hard queries (`hard-001` to `hard-030`), for 180 in total. No existing query or label changed.
- Makes the descriptions of four options clearer: `price`, `global`, `onchain_dex` and `off_topic`.
- Adds router accuracy without advice questions, and the confidence table, to `bench score`.

The two runs use different query sets. Their scores are not compared.

## Dataset

**[amadzai/think-fast-jev-queries](https://huggingface.co/datasets/amadzai/think-fast-jev-queries)** on Hugging Face
(CC-BY-4.0): 180 user questions, each with a router label and a guardrail label.

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
| `source` | `seed`, `generated` or `hard`. |

The 30 `hard` queries are made to be difficult: 10 factual questions that sound like advice, 10 advice requests that
look like factual questions, and 10 crypto-themed tasks or off-topic requests. They were added before the run.

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
| `off_topic` | Not about crypto markets, or a task other than asking for facts, such as writing a song. | Can you recommend a good laptop for programming under $1,500? |

The benchmark only scores the selected option, and no step acts on it. In an application, `allow` could go on to an
answer, `advice` could get a reply that offers facts instead of advice, and `off_topic` could get a redirect.

### Arms

| Arm | Role | Model | How the benchmark calls it |
|---|---|---|---|
| Claude Haiku 5.5 | Incumbent | `claude-haiku-5-5` | Anthropic API. Thinking disabled, effort `low`. Structured output with one probability per option. 3 runs per item. |
| GPT-6 Luna | Incumbent | `gpt-6-luna` | OpenAI API, Chat Completions. Reasoning effort `none`, standard service tier. Structured output with one probability per option. 3 runs per item. |
| Jev | Candidate | `jev-latest` | TypeSafe System One API. One prefill pass. |
| Kev-4B | Candidate | `kev-latest` (checkpoint `jaredpalmer/kev-4b`) | The Kev server on a local GPU. Same API as Jev. |

### Procedure

- The runner calls one arm at a time and finishes one item before the next. The arm order rotates per item.
- The first 3 calls of each arm in each `bench run` are warm-up and do not count toward the latency metrics. For Kev, the
  keep-warm calls count as calls.
- A failed call gets up to 3 attempts. After that, the runner records an error.
- Haiku and Luna run each item 3 times. The main scores use only the first run, so they get no voting advantage. A
  separate row in `results/scores.md` shows the majority of the 3 runs.
- Quality scores use only the items that every arm answered validly.
- `bench permute` asks Jev and Kev each router and guardrail item under 6 option orders: the original order and 5
  shuffled orders.
- `bench throughput` measures Kev's decisions per hour with 32 calls at a time. This number gives Kev's cost.

### Metrics

| Metric | Meaning | What is good |
|---|---|---|
| `accuracy` | The share of items where the selected option is the label. | Higher. 1 is perfect. |
| `brier` | Calibration, as a multi-class Brier score. | Lower. 0 is perfect and 2 is the worst. |
| `agreement` | **Haiku and Luna:** the share of items where all 3 runs selected the same answer. | Higher. 1 means the runs always agree. |
| `order_stable` | **Jev and Kev:** the share of items where all option orders selected the same option. | Higher. 1 means order never changed the answer. |
| `order_spread` | **Jev and Kev:** the mean of the largest change of one option's probability between the option orders. | Lower. 0 means order had no effect. |
| `model_p50`, `model_p95` | Model time in ms, median and 95th percentile. | Lower. |
| `e2e_p50`, `e2e_p95` | End-to-end time in ms as the client measures it, median and 95th percentile. | Lower. |
| `cost_per_1k` | USD per 1,000 decisions. | Lower. |
| `error_rate` | The share of decisions with an error, from the last record per decision. Kev's rate includes failed keep-warm calls. | Lower. 0 means no errors. |

### Setup

- **Run date:** 2026-11-06.
- **Model versions:** `jev-1.14.0`, `claude-haiku-5-5`, `gpt-6-luna`, and `kev-latest` with the checkpoint
  `jaredpalmer/kev-4b`.
- **Client:** a Mac in Malaysia. Jev, Haiku and Luna go over the internet. Kev goes over the local network.
- **Kev server:** RTX 5070 Ti (16 GB), WSL2 Ubuntu 24.04, PyTorch 2.8.0, `KEV_PREFIX_CACHE=1`.
- **Prices:** Jev $0.042 per 1M input tokens, output free. Haiku and Luna each $0.10 per 1M input tokens and $0.50
  per 1M output tokens. Kev: AWS g6.xlarge (1 x L4) on-demand in us-east-1, $0.8048 per hour ($1.0138 in ap-southeast-5, Malaysia).

## Limitations

### How the arms are set up

| Limitation | What it means |
|---|---|
| **The LLMs run at their lowest setting** | Haiku has thinking disabled and effort `low`. Luna has reasoning effort `none`. Both answer directly, like the System One models. A higher setting can change their quality, latency and cost. |
| **The LLM probabilities are self-reported** | Haiku and Luna write one number per option. For the router and guardrail, the client scales them to sum to 1, and the Brier score uses the scaled numbers. |
| **Kev gets a keep-warm call** | On the test PC, Kev's model time rises from about 60 ms to about 180–290 ms after 8 s or more of idle. The runner sends one unrecorded call before each recorded call. A production server with gaps between requests pays this penalty. With keep-warm, about 7% of calls in a test still took 300–500 ms, which can reach Kev's p95. |
| **Option order can change the answer** | The [TypeSafe docs](https://docs.typesafe.ai/primitives/choice) say that Jev 1.13 can lean toward the first option. The main run uses one fixed order. `order_stable` and `order_spread` show the effect for Jev and Kev, not for the LLMs. |

### How latency and cost are measured

| Limitation | What it means |
|---|---|
| **Model time differs per arm** | Jev returns none, so it has only end-to-end time. Haiku's is the `server-timing` origin time at the Cloudflare edge, which includes Anthropic's internal network. Luna's is the `openai-processing-ms` header. Kev's is its server's `latency_ms` field. |
| **End-to-end time depends on location** | Jev, Haiku and Luna go from Malaysia over the internet. Kev goes over the local network. |
| **Kev's cost is an estimate** | It is an AWS GPU price divided by the speed measured on an RTX 5070 Ti. |

### What the data can and cannot show

| Limitation | What it means |
|---|---|
| **Some router gaps are small** | With 180 queries, Luna's lead over Jev is statistically significant (exact McNemar test, p = 0.02). Jev's lead over Haiku is not (p = 0.06). |
| **The router scores depend on a label rule** | Advice questions are labelled `unsupported`. A different rule changes the ranking, as [the miss table](#where-the-router-misses-are) shows. |
| **The guardrail gaps are very small** | No arm missed more than 2 of 180 questions. Only 42 of the 180 questions are `advice` or `off_topic`. |
| **The queries are not real traffic** | The seeds cover each option about equally, so the accuracy is not a production number. The set leaves out multi-step questions and price predictions, because they do not have one clear label. |
| **This is one run on one day** | Hosted models and their latency change over time. |

## Reproduce

You need Python 3.12 and [uv](https://docs.astral.sh/uv/). Add your API keys to `.env`:

| Setting | Used for |
|---|---|
| `TYPESAFE_API_KEY` | Jev |
| `ANTHROPIC_API_KEY` | Claude Haiku 5.5 |
| `OPENAI_API_KEY` | GPT-6 Luna |
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
