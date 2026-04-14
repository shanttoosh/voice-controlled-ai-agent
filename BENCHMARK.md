# Model & latency benchmarks — Voice-Controlled AI Agent

This document is written for **course articles and technical blogs**. Section **5** lists **measured** values from `scripts/benchmark_latency.py` (you can re-run and update). Optional fields (CPU, §5.3 codegen) remain for you to fill.

---

## 1. Purpose

We compare **end-to-end perceived speed** of the pipeline and **per-step** cost drivers:

| Stage | Service | Default model (this repo) |
|--------|---------|---------------------------|
| Speech-to-text | Groq Audio | `whisper-large-v3-turbo` |
| Intent + tools (chat) | Groq Chat | `llama-3.1-8b-instant` (override via `GROQ_LLM_MODEL`) |

Groq optimizes for **low latency** on hosted inference; numbers below are **indicative** and depend on **audio length**, **prompt size**, **network**, and **API load**.

---

## 2. Test environment

| Field | Your value |
|--------|------------|
| OS | Windows 10 |
| CPU | _(optional — add if you want)_ |
| RAM | _(optional)_ |
| Python | _(your venv version, e.g. 3.11.x)_ |
| Network | _(e.g. home Wi‑Fi)_ |
| Date of run | 2026-04-15 |

---

## 3. What we measure

| Metric | Definition |
|--------|------------|
| **STT latency** | Time from sending audio bytes to Groq until transcript text is returned. |
| **LLM latency (intent)** | Time for one chat completion that returns JSON intent (short output). |
| **LLM latency (generation)** | Time for code generation or long answer (larger `max_tokens`). |
| **Cold vs warm** | First request after idle may be slower; we recommend **3 runs**, report **median**. |

**Throughput note:** Groq publishes aggregate throughput; your **wall-clock** is what users feel in Streamlit.

---

## 4. Optional: collect numbers locally

From the project root (with `.env` containing `GROQ_API_KEY`):

```bash
python scripts/benchmark_latency.py
```

With a sample audio file (WAV/MP3) for STT:

```bash
python scripts/benchmark_latency.py --audio path/to/sample.wav
```

Paste the script output into the tables in **Section 5**.

---

## 5. Results (measured)

Commands: `python scripts/benchmark_latency.py` and  
`python scripts/benchmark_latency.py --audio "path/to/testing.wav"`  
(Audio file used: **testing.wav**, ~800 KB, recorded locally.)

### 5.1 Groq Whisper (`whisper-large-v3-turbo`)

| Audio | Run 1 (s) | Run 2 (s) | Run 3 (s) | **Median (s)** |
|--------|-----------|-----------|-----------|----------------|
| `testing.wav` (~800 KB) | 0.960 | 0.806 | 0.975 | **0.960** |

**Observation:** Three STT runs cluster around **0.8–1.0 s** for this file; wall-clock includes upload + API processing on Groq.

---

### 5.2 Groq LLM — intent-style short call (same script as intent/JSON workload)

Model: **`llama-3.1-8b-instant`** (repo default)

| Run | Latency (s) |
|-----|-------------|
| 1 | 0.677 |
| 2 | 0.232 |
| 3 | 0.267 |
| **Median** | **0.267** |

**Same script, LLM-only run (no `--audio`):** runs **0.684, 0.218, 0.256** → median **0.256** s. First call is slower (cold start); medians match within ~0.01 s across sessions.

---

### 5.2b Summary — LLM vs STT (this machine, same session as §5.1)

| Stage | Median (s) |
|--------|------------|
| LLM short call | 0.267 |
| Whisper STT (`testing.wav`) | 0.960 |

---

---

### 5.3 Optional model comparison (same prompts)

If your API access allows, repeat **§5.2** with another Groq chat model, e.g. `llama-3.3-70b-versatile` for higher quality (set `GROQ_LLM_MODEL` in `.env`).

| Model | Median intent latency (s) | Notes |
|--------|---------------------------|--------|
| `llama-3.1-8b-instant` | 0.267 | Measured §5.2 (2026-04-15) |
| `llama-3.3-70b-versatile` | _(not run — fill if you compare)_ | Often slower; may suit complex JSON / code |

---

## 6. Interpretation for your article

1. **Bottleneck:** For short utterances, **LLM tool steps** often dominate; for long recordings, **STT** grows with duration.
2. **Why Groq:** Good fit when **local GPU** for Whisper + 70B-class models is unavailable; trade-off is **network dependency** and **API policy**.
3. **Fair comparison:** Same audio file, same prompts, **median of 3** runs, stable network.

---

## 7. Limitations

- Numbers are **not** reproducible across regions and time of day.
- This repo does **not** benchmark **local** Whisper/Ollama unless you add them yourself.
- **Cost** (USD per 1M tokens) should be taken from **current Groq pricing** on their site, not hardcoded here.

---

## 8. How to use this file

- **Dev.to / Hashnode:** Paste the Markdown body directly (diagrams: use mermaid or images).
- **Medium:** No `.md` import; paste sections manually or use a Markdown → Medium converter, and **upload architecture images** as PNG.
- **Substack:** Paste Markdown; check preview for code blocks.

---

*Voice-Controlled AI Agent project. Update §5 if you re-benchmark; add your GitHub link in the blog post.*
