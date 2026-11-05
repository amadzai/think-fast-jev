# think-fast-jev

A benchmark of single-pass decision models against LLMs at the decision steps of answering crypto market questions.

## Language

**Arm**:
One system under test in the benchmark.
_Avoid_: Variant, variant A/B, model under test

**Incumbent**:
An LLM arm that the candidates try to replace. A benchmark can have more than one.
_Avoid_: Baseline, control

**Candidate**:
An arm that tries to replace the incumbents at a decision node.
_Avoid_: Challenger, alternative

**Decision node**:
One step where an arm selects from a fixed set of options.
_Avoid_: Classifier, decision layer step

**Option**:
One of the fixed answers that a decision node can select, with a short name and a one-line description.
_Avoid_: Label, class, choice

**Label**:
The correct option for one item, written down before the run. Scoring compares each arm's answer with it. The arms never see it.
_Avoid_: Ground truth, gold answer, target

**Router**:
The decision node that sends a user question to the one data tool that can answer it.
_Avoid_: Intent classifier, dispatcher

**Guardrail**:
The decision node that decides if a user question may be answered.
_Avoid_: Filter, moderation, safety check

**Coin sector**:
The decision node that decides which sectors a coin belongs to.
_Avoid_: Classification

**Unsupported**:
The router option for a question that no data tool can answer.
_Avoid_: Other, fallback, none
