# Student Scaffold

This `src/` folder is the student version of the lab.

- It keeps the same high-level structure
- The Python files are now fully implemented (no `NotImplementedError` left)
- The benchmark structure includes: standard benchmark + long-context stress benchmark
- The runtime supports these providers: `openai`, `custom`, `gemini`, `anthropic`, `ollama`, `openrouter`

Implementation order that was followed:

1. `config.py` — `LabConfig` + `load_config()` (paths, compact thresholds, provider + judge model)
2. `memory_store.py` — `estimate_tokens()`, `UserProfileStore` (read/write/edit `User.md`), `extract_profile_updates()`, `summarize_messages()`, `CompactMemoryManager`
3. `agent_baseline.py` — `BaselineAgent` (short-term memory only, forgets across threads)
4. `agent_advanced.py` — `AdvancedAgent` (short-term + `User.md` + compact memory)
5. `benchmark.py` — `load_conversations()`, `run_agent_benchmark()`, `recall_points()`, `heuristic_quality()`, `format_rows()`
6. `test_agents.py` — 4 tests covering `User.md`, compact trigger, cross-session recall, prompt-load reduction

## Running

```bash
python src/benchmark.py        # from the repo root
pytest src/test_agents.py -v
```

Both commands run fully offline (`force_offline=True`), so no API key is
required. Set `LLM_PROVIDER` / `LLM_MODEL` and the matching API key env var in
`.env` if you want the live LangChain path instead.

Datasets are available at the repo root in `data/`.
