---
name: experiment-proposal
description: Convert a research idea into a controlled experiment specification without editing code or launching training.
---

# Experiment Proposal

Use when a user proposes a new experimental idea.

Output an experiment spec draft covering:

- research question;
- hypothesis;
- baseline;
- single changed variable;
- controlled variables;
- dataset and protocol;
- success criteria;
- tier;
- expected compute cost;
- failure interpretation.

Do not modify code. Do not launch training. If the idea lacks a single changed
variable, split it into multiple specs.
