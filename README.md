# think-fast-jev

A small, reproducible benchmark. It compares single-pass decision models ([Jev](https://typesafe.ai) and its
open-weight clone [Kev](https://github.com/jaredpalmer/kev)) with a frontier LLM at the decision steps of a crypto
market-data assistant. It measures quality, calibration, latency and cost.

**Powered by [CoinGecko](https://www.coingecko.com/en/api).** Not affiliated with or endorsed by CoinGecko or TypeSafe.

## Headline

**Question:** can a single-pass decision model take the decision layer of a crypto market-data assistant from a
frontier LLM, and what does that do to quality, latency and cost?

<!-- After the full run: the answer in 2-3 sentences, with the three numbers for each node. Say where the candidates lose. -->
_The full run is not done yet. This section gets the result after the run._

## Method

Before an assistant like this writes an answer, it makes small decisions. This benchmark tests three of them. Each
decision node runs on its own: an item goes in, a decision comes out. There is no assistant pipeline and no answer
generation.

| Decision node | Input | Options |
|---|---|---|
| Router | A user question | 14: 13 data-tool areas of the public CoinGecko API, and `unsupported` |
| Guardrail | A user question | 3: `allow`, `advice`, `off_topic` |
| Coin sector | A coin: name, symbol and description | 14 sectors, one yes/no question each |

The option names and descriptions are in [`src/bench/nodes.py`](src/bench/nodes.py). Every arm gets the same item text
and the same options in the same order.

| Arm | Role | Model | How the benchmark calls it |
|---|---|---|---|
| Claude Haiku 5.5 | Incumbent | `claude-haiku-5-5` | Anthropic API. Thinking disabled, effort `low`. Structured output with one probability per option. 3 runs per item. |
| Jev | Candidate | `jev-latest` | TypeSafe System One API. One prefill pass returns a probability per option. |
| Kev-4B | Candidate | `kev-latest` (checkpoint `jaredpalmer/kev-4b`) | The Kev server on a local GPU. Same API as Jev. |

**Data**

- Router and guardrail: one set of user questions, each with a router label and a guardrail label. The set is on
  Hugging Face: <!-- dataset link after `bench export` and upload -->_link after publication_.
- Coin sector: the top 200 coins by market cap from the CoinGecko API, without the coins that have no description.
  The labels are each coin's current CoinGecko categories, mapped to the 14 sectors in `nodes.py`.

**Procedure**

- The runner calls one arm at a time and finishes one item before the next. The arm order rotates per item.
- The first 3 calls of each arm in each `bench run` are warm-up. They are not in the latency metrics. For Kev, the
  keep-warm calls count as calls.
- A failed call gets up to 3 attempts. After that, the runner records an error.
- Haiku runs each item 3 times. The main scores use only its first run, so Haiku gets no voting advantage. A separate
  row shows the majority of the 3 runs.
- The quality scores use only the items that every arm answered validly.
- `bench permute` asks Jev and Kev each router and guardrail item under 6 option orders: the original order and 5
  shuffled orders.
- `bench throughput` measures Kev's decisions per hour with 32 calls at a time. This number gives Kev's cost.

**Metrics**

| Metric | Meaning |
|---|---|
| `accuracy` | Router and guardrail: the share of items where the selected option is the label. |
| `macro_f1` | Coin sector: the mean F1 over the sectors, with yes at a probability of 0.5 or more. |
| `brier` | Calibration. 0 is perfect. Router and guardrail: multi-class (2 is the worst). Coin sector: per yes/no question (1 is the worst). |
| `agreement` | Haiku: the share of items where all 3 runs selected the same answer. |
| `order_stable` | Jev and Kev: the share of items where all option orders selected the same option. |
| `order_spread` | Jev and Kev: the mean of the largest change of one option's probability between the option orders. |
| `model_p50`, `model_p95` | Model time in ms, median and 95th percentile. |
| `e2e_p50`, `e2e_p95` | End-to-end time in ms as the client measures it, median and 95th percentile. |
| `cost_per_1k` | USD per 1,000 decisions. |
| `error_rate` | The share of decisions with an error. |

**Setup**

<!-- After the full run: date, model versions from the run metadata, and the client location. -->
- Run date: _after the run_.
- Model versions: _from the run metadata_. `jev-latest` is an alias, so the run records the version that each
  response reports.
- Client: a Mac in Malaysia. Jev and Haiku go over the internet. Kev goes over the local network.
- Kev server: RTX 5070 Ti (16 GB), WSL2 Ubuntu 24.04, PyTorch 2.8.0, `KEV_PREFIX_CACHE=1`.
- Prices: Jev $0.042 per 1M input tokens, output free. Haiku $0.10 per 1M input tokens and $0.50 per 1M output
  tokens. Kev: AWS g6.xlarge (1 x L4) on-demand in us-east-1, $0.8048 per hour.

## Results

<!-- After the full run: paste results/scores.md here, then run `bench chart`. -->
_The full run is not done yet._

![Quality against latency per decision](results/charts/quality_latency.png)

![Cost per 1,000 decisions](results/charts/cost.png)

## Limitations

- **Haiku's probabilities are self-reported.** The Anthropic API gives no logprobs and Haiku 5.5 does not accept a
  temperature. Haiku writes one number per option in structured output. For the router and the guardrail, the client scales
  these numbers to sum 1. Haiku's Brier score measures the scaled numbers.
- **Haiku runs at its lowest reasoning setting:** thinking disabled and effort `low`. A higher setting can change its
  quality, latency and cost.
- **Model time is not the same measure for each arm.** Hosted Jev returns no model time, so Jev has only end-to-end
  time. Haiku's model time is the origin time in the `server-timing` header at the Cloudflare edge. It includes
  Anthropic's internal network. Kev's model time is the `latency_ms` field of its server.
- **End-to-end time depends on the location.** Jev and Haiku go from Malaysia over the internet. Kev goes over the
  local network.
- **Kev gets a keep-warm call.** On the test PC, Kev's model time rises from about 60 ms to about 180–290 ms after
  8 s or more of idle. Tests did not find the cause. The runner sends one unrecorded call to Kev right before each recorded
  call. Without it, a production server with gaps between requests pays this idle penalty. With keep-warm, a few calls
  (about 7% in a test) still took 300–500 ms. These calls can reach Kev's p95.
- **Kev's cost is an estimate.** The throughput comes from an RTX 5070 Ti, not from the L4 in the g6.xlarge. The client
  sends the calls from a Mac over the local network, so the time includes the network. The cost assumes that the GPU is
  busy all the time. In ap-southeast-5 (Malaysia), the g6.xlarge costs $1.0138 per hour.
- **Option order can change the answer.** The [TypeSafe docs](https://docs.typesafe.ai/primitives/choice) say that
  Jev 1.13 can lean toward the first option. The main
  run uses one fixed order for every arm. `order_stable` and `order_spread` show the effect for Jev and Kev. This benchmark
  does not measure Haiku's option-order sensitivity. An item where two options have the same probability can show as
  unstable.
- **The coin sector labels are CoinGecko's current categories.** Some coins do not have a category that fits them. An
  arm that says yes to such a sector counts as wrong. The 14 sectors are broad, but sectors with few coins give an
  unstable F1. Some sectors overlap, for example governance and DeFi. The labels are from the fetch date.
- **The query set is small and has one labeller.** Claude drafted the seed queries, and the author reviewed and edited
  them. Haiku is also a Claude model. A model that is not Claude wrote the variations. The author reviewed every query
  and label.
- **The error rate uses the last record per decision.** An error that a rerun fixed does not count. Kev's error rate
  includes failed keep-warm calls (errors that start with `keep-warm call: `).
- **This is one run on one day.** Hosted models and their latency change over time.

## Reproduce

You need API keys for TypeSafe, Anthropic and the CoinGecko Demo API in `.env` (`TYPESAFE_API_KEY`,
`ANTHROPIC_API_KEY`, `COINGECKO_API_KEY`). For Kev, start the [Kev server](https://github.com/jaredpalmer/kev) with
`jaredpalmer/kev-4b` and set `KEV_BASE_URL` and `KEV_API_KEY`. Without these two settings, the commands run without
Kev, and `bench throughput` does not run. After a fresh start of the Kev server, do one throwaway run first, because
the first calls are slow.

```sh
uv sync
uv run bench check
uv run bench fetch
for node in router guardrail coin_sector; do uv run bench run --node "$node"; done
for node in router guardrail; do uv run bench permute --node "$node"; done
for node in router guardrail coin_sector; do uv run bench throughput --node "$node"; done
uv run bench score
uv run bench chart --note "Run 2026-11-03. Kev-4B on an RTX 5070 Ti."
uv run bench export
```

A rerun of `bench run` or `bench permute` skips the decisions that already have a result.
