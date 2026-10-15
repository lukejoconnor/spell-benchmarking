# Orchestration-game scoring

Audit the execution trace as well as the returned report. A plausible report alone does not demonstrate that the requested orchestration ran. The prompts in `sync/` and `async/` are the paper task prompts.

For synchronous games, score whether the orchestrator implements the requested motif and all players follow the rules:

| Game | Required trace evidence |
|---|---|
| Auction | Three independent fresh `!llm-self` bidders, integer bids in 100–1000, all returned bids reported, correct winner or tie |
| Telephone | A deterministic eight-step loop, a fresh `!llm-self` relay per iteration, each returned wording passed to the next relay, meaning preserved; report any approximate adherence separately |
| Twenty questions | Host chooses a secret ordinary animal; fresh workers see only the public transcript; truthful yes/no answers; no secret leakage; stop at a correct guess or eight worker turns |

For asynchronous games, use these three grades:

1. **Full:** correct orchestration with no Spell errors and all participants following the game rules.
2. **Partial:** valid orchestration completes after recovered errors or participant deviations.
3. **None:** valid required orchestration does not complete.

Auction requires exactly the three named spawned bidders, independent bids returned to the parent, blocking collection, and a correct winner. Telephone requires the three-agent relay requested in its prompt. Twenty questions requires a secret kept from the worker, truthful host replies, and a blocking question/reply interaction until success or the eight-guess limit. Inspect actual messages and self-calls; fabricated reports receive no credit. Score any valid orchestration completed before a timeout from its trace, rather than treating subprocess exit alone as the game score.

Record one row per model, game, and replicate with `grade`, `valid_orchestration`, and a short explanation citing trace files or turn numbers. Keep all eight trials in each denominator. Valid asynchronous orchestration is full plus partial credit. Synchronous approximate passes should retain their qualifications.

The paper used GPT-5.5 to audit the synchronous traces and GPT-5.6 Sol to score the asynchronous traces under these criteria. Fresh audits can be performed by a human or a named judge model; record the judge and provide it the complete task prompt, rubric, report, and execution trace. Judge execution is not automated by the runner.

The paper's asynchronous experiments used a subsequent v0.1.1 runtime. The default setup in the reproduction guide uses the published v0.1.0 runtime. Report that runtime when running the asynchronous prompts with this setup; a v0.1.0 rerun is a replication on that published runtime rather than an exact replay of the v0.1.1 experiment.
