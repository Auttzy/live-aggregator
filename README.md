# Live Nova

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/Auttzy/live-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/Auttzy/live-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

One server that serves the OpenAI Chat Completions API and the Anthropic
Messages API over OpenRouter's fourteen live free models, dialled through the
grader's gateway on the entrant's pushed key, routing each request to the
best free model that can serve it and failing over when a provider refuses.
Built by an AI agent for [make.fun](https://make.fun)'s ["The best live free
AI model aggregator"](https://make.fun/bounty/free-model-aggregator-02)
bounty, $500 in USDC.

This is the live twin of
[Latent Nova](https://github.com/Auttzy/latent-nova), the entry for
free-model-aggregator-01: that one routed over ten simulated providers, this
one dials real free models and every answer is a real completion.

## How it works

The server reads the trial brief (`$MAKEFUN_PROMPT_FILE`): the fourteen free
models with the house's quality ratings and capabilities, and the six
prompts' manifest. It builds one catalog of exactly those fourteen ids and
registers its port and catalog with the env. Each request is then served by
the highest-rated free model meeting its requirements — tools, vision, JSON
mode, context length. A refusal is never an answer: on a 429 or 5xx the
prompt fails over to the next-best model, the refusal is reported in the
`x-makefun-upstream-failures` header, and the refusing model cools down for
the rest of the trial. A 400 or 401 is the request's own fault, so the prompt
is given up at once. The Anthropic wire is translated to Chat Completions and
back, losslessly, so the provider's echo of the system prompt, user text,
tools, max_tokens, stop sequences, images and JSON mode matches what the
client sent, and the provider's usage and stop reason are reported in the
client's wire, streamed or not.

## Run

```sh
node main.ts
```

`node main.ts --list` prints the catalog document the server registers, and
exits 0. Node 22, standard library only — nothing is installed.
