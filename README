# Animal-like Core (ALC)

A biologically-inspired **spiking neural associative memory** written in D. This release
is two modules:

| Module | File | Role |
| --- | --- | --- |
| `ALC` — Animal-like Core | `core.d` | Tokenizer, sparse spiking network, learning, generation, persistence |
| `bmfLoader` | `bmfLoader.d` | Loads a `BMF` file into question / answer arrays |

This is the core only: there is no front-end and no I/O loop. You drive these two modules
from your own program. A trained instance is a *model* — the core itself is agnostic to
what you name it.

The design goal is **no hardcoded answers**. The network only ever returns text it has
learned; when it has not learned enough it returns an empty string, and you decide what
to say (see `doubtReply`).

## Architecture

Everything cognitive lives in `core.d`.

- **Tokenizer** — an ASCII word is one token (so `Java` ≠ `JavaScript`); a non-ASCII
  grapheme (CJK, punctuation, codec symbols) is one token.
- **Sparse spiking network** — an entorhinal-like layer (`N_E = 16384`) plus a
  dentate-gyrus-like expansion (`N_DG = 2^23`) for pattern separation. A token activates
  `K_ASM = 12` EC cells; a context (the last `CTXW = 10` tokens) maps to `K_DG = 24`
  DG cells.
- **Learning** — STDP-like association between a context's DG code and the EC assembly
  of the next token. `sleep()` consolidates (normalizes weights) and resets the
  neuromodulators.
- **Neuromodulation** — four scalar modulators (dopamine, serotonin, acetylcholine,
  norepinephrine). `right()` rewards a pair, `wrong()` weakens it.
- **Generation** — `generate` builds its input from the current context and looks for the
  best-matching memory. It counts something as "remembered" only once the evidence
  reaches `RECALL_FLOOR = 3.5`; below that it returns an empty string. While writing, if
  per-step evidence falls under `CONT_FLOOR = 3.0`, it abandons the thread and returns
  nothing.

The core is agnostic about what a token *means*; any string is a "question". Because it
never inspects the meaning or modality of its input, a caller can feed it text, or any
other signal first encoded to a token string, without changing the core.

## API

### `core.d` — `module ALC`

| Function | Description |
| --- | --- |
| `string generate(string prefix)` | Free-generate a continuation from a cue. Empty string if unknown / lost thread. |
| `string talk(string msg)` | Alias for `generate`. |
| `string doubtReply()` | Produces an "I don't know" phrase. The wording is learned memory — never a hardcoded literal — and the choice among variants is driven by an entropy-seeded PRNG. |
| `void teach(string[] usd, string[] asd)` | Train question/answer pairs (each pair becomes `BOS q QEND a EOS`). Throws if the lengths differ. |
| `void trainSentence(string sentence)` | Train a single sentence as `BOS sentence EOS`. |
| `void right(string q, string a)` | Reinforce a pair (dopamine up). |
| `void wrong(string q, string a)` | Weaken a pair (negative association). |
| `void sleep()` | Consolidate: normalize weights, reset neuromodulators. |
| `void consolidate()` | Alias for `sleep`. |
| `void clearHippocampus()` | Reserved (no-op). |
| `ulong lexiconCount()` | Number of tokens in the lexicon. |
| `void saveCortex(string path)` | Write the store as `STO12` (no-op if already mapped). |
| `void openCortex(string path)` | Memory-map a `STO12` store. |

Codec constants wrapped around training text: `BOS = "␂"`, `QEND = "␁"`, `EOS = "␃"`.

### `bmfLoader.d` — `module bmfLoader`

| Symbol | Description |
| --- | --- |
| `string[] usd;` | Questions, filled by `loadBMF`. |
| `string[] asd;` | Answers, filled by `loadBMF`. |
| `void loadBMF(ubyte[] bmf)` | Parse a `BMF` byte buffer into `usd` / `asd`. Throws on a bad header or truncated data. |

## Typical lifecycle

```d
import ALC;
import bmfLoader;
import std.file : exists, read;
import std.string : strip;

void main() {
    if (exists("cortex.sto"))
        openCortex("cortex.sto");         // Instant, memory-mapped
    else {
        loadBMF(cast(ubyte[])read("data.bmf"));
        teach(bmfLoader.usd, bmfLoader.asd);
        sleep();
        saveCortex("cortex.sto");         // May be multi-gigabyte; takes a few minutes
    }

    // A caller decides the fallback wording:
    string reply = talk("...");
    if (reply.strip.length == 0)
        reply = doubtReply();
}
```

First training walks the whole corpus and consolidates; expect a few minutes and a
multi-gigabyte file. Subsequent runs memory-map the store.

## File formats

Both formats are produced / consumed by these two modules.

### BMF — `bmfLoader.d`

```
"BMF"
u64 block count (little-endian)
then, per block:
    u8  type          0 = question, 1 = answer
    u64 byte length   (little-endian)
    ... bytes         UTF-8 text
```

### STO12 — `core.d`

```
"STO12"
lexicon            (length-prefixed UTF-8 strings, padded to 8 bytes)
offsets[N_DG + 1]  (u64)
posts[E]           (u16)      -- postsynaptic targets
weights[E]         (f64)
neuromod           (4 x f64)
```

A presynaptic neuron's entries are the contiguous range
`[offsets[pre], offsets[pre + 1])`. The file is memory-mapped, so RAM use is independent
of on-disk size.

## Parameters

| Constant | Value | Meaning |
| --- | --- | --- |
| `N_E` | 16384 | EC (entorhinal-like) cell count |
| `N_DG` | `1 << 23` | DG (pattern-separation) cell count |
| `K_ASM` | 12 | EC cells activated per token |
| `K_DG` | 24 | DG cells per context code |
| `CTXW` | 10 | Context window, in tokens |
| `MAX_SKIP` | 4 | Tolerated trailing noise tokens in a cue |
| `RECALL_FLOOR` | 3.5 | Evidence needed to call a memory "remembered" |
| `CONT_FLOOR` | 3.0 | Per-step floor before the thread is abandoned |
| `N_I` | 2048 | Inhibitory pool size |
| `TAU_M` / `V_TH` | 12.0 / 1.0 | Membrane time constant / firing threshold |

## Notes and limitations

- Research prototype, not a product. Whatever it "knows" it learned from the training
  pairs you feed it.
- Long free-form continuations can drift or hallucinate; there is no hallucination
  suppression, by design.
- The core is modality-agnostic: any signal you can encode as a token string works as a
  cue without touching `core.d`.
- A `STO12` store is not portable across builds with different `N_DG` / `N_E`.

## Dependencies

- A D compiler with Phobos — tested with **DMD64 v2.113.0**. LDC / GDC should also work.
- **No third-party packages.** Standard-library modules only:
  - `core.d` — `std.uni`, `std.conv`, `std.random`, `std.mmfile`, `std.stdio`, `std.string`
  - `bmfLoader.d` — `std.conv` (and `ALC`)

## Build

Add both files to your D build, for example:

```
dmd -O -release your_main.d core.d bmfLoader.d
```

Tested with DMD64 v2.113.0 (Windows). The sources are plain D and should build on other
platforms.