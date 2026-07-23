# voxkit

A thin, event-driven Python library for building real-time voice agents on top of LangChain/LangGraph. STT, the agent, and TTS communicate over typed events (`STTEvent`, `LLMEvent`, `TTSEvent`) pushed through `asyncio.Queue`s; `VoxkitPipeline` (`voxkit/core/pipeline.py`) is the orchestrator. See `README.md` for the full architecture and usage.

## Setup & commands

- Package/dependency manager: `uv` (Python 3.13, see `.python-version`).
- Install deps: `uv sync`
- Run the example voice agent (mic in/out via `sounddevice`): `uv run main.py`
- Build docs locally: `uv pip install -e ".[docs]"` then `mkdocs serve`
- No test suite exists yet.

## Code conventions

- Google-style docstrings and full type annotations on all public modules/classes/methods — keep these accurate, since `mkdocstrings` renders them directly into the published API reference (`docs/reference/*.md`).
- New STT/TTS backends implement the `STTProvider`/`TTSProvider` ABCs (`voxkit/stt/base.py`, `voxkit/tts/base.py`) — queue-in/queue-out, typed events. `VoxkitPipeline` only depends on these interfaces, never on a specific vendor.
- Keep the library thin: no frame-based transport bus, no call/room management, no bundled VAD abstraction — STT/TTS providers report voice activity themselves.
- `voxkit/exceptions.py` and `voxkit/vad/` are empty stub files, not implemented or used anywhere. Don't assume they contain anything; don't build on them without discussing scope first.

## Docs site

`docs/` + `mkdocs.yml` + `.github/workflows/docs.yml` deploy an mkdocs-material/mkdocstrings site to GitHub Pages on every push to `master` that touches `voxkit/`, `docs/`, `mkdocs.yml`, or `pyproject.toml`.

## Documentation policy

**Every code change or new feature must update docs in the same change, not as a follow-up.** Docs are the contract for anyone consuming this library — treat an undocumented change as incomplete, not done. Concretely, when you touch code:

- Update or add Google-style docstrings + type annotations on anything public you added or changed (new/changed classes, methods, functions, options). `mkdocstrings` renders these directly — a stale docstring is a stale published doc.
- If you add a new module, provider, or public class, add a `::: voxkit.module` block to the relevant page under `docs/reference/`, and a nav entry in `mkdocs.yml` if it's a new top-level area.
- Update `README.md` (quick start, public API list, event-type table, "adding a new provider" section) whenever the change affects what's shown there — new install steps, new options, changed pipeline behavior, new supported providers.
- Update `docs/architecture.md` if the change affects turn-taking, event flow, or barge-in behavior.
- Before calling a task done, sanity-check the docs build: `mkdocs build --strict` should pass with no new warnings.
