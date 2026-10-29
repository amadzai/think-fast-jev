# think-fast-jev

A benchmark of single-pass decision models against a frontier LLM at the decision steps of a crypto data assistant.

## Language

**Arm**:
One system under test in the benchmark.
_Avoid_: Variant, variant A/B, model under test

**Incumbent**:
The frontier LLM arm that the candidates try to replace.
_Avoid_: Baseline, control

**Candidate**:
An arm that tries to replace the incumbent at a decision node.
_Avoid_: Challenger, alternative

**Decision node**:
One step of the assistant where an arm selects from a fixed set of options.
_Avoid_: Classifier, decision layer step

**Option**:
One of the fixed answers that a decision node can select, with a short name and a one-line description.
_Avoid_: Label, class, choice

**Router**:
The decision node that sends a user question to the one data tool that can answer it.
_Avoid_: Intent classifier, dispatcher

**Unsupported**:
The router option for a question that no data tool can answer.
_Avoid_: Other, fallback, none
