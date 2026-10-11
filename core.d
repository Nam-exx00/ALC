module ALC;

/*
    Animal-like Core

    Copyright 2026 Nam_exx00

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

import std.uni : byGrapheme;
import std.conv : to;
import std.random : unpredictableSeed;
import std.mmfile : MmFile;
import std.stdio : File;
import std.string : strip;

/*
    Network
*/
private enum size_t N_E = 16384; // EC
private enum size_t N_DG = 1 << 23; // Mode Separation Space
private enum size_t K_ASM = 12; // EC Character Set Size
private enum size_t K_DG = 24; // DG Context Code Size
private enum size_t CTXW = 10; // Max Context Window (number of tokens)
private enum size_t MAX_OUT = 128; // Max outgoing connections per presynaptic neuron
private enum size_t N_I = 2048; // Number of inhibitory intermediate neurons (population)

/*
    Synapse / STDP
*/
private enum double ETA = 1.0;
private enum double A_MINUS = 0.20;
private enum double W_MAX = 100000.0;
private enum double W_FLOOR = 0.0;
private enum double OUT_TARGET = 1.0; // Total outgoing weight per presynaptic neuron

/*
    Generative dynamics (LIF   E/I   theta/gamma)
*/
private enum size_t MAX_LEN = 240;
private enum size_t LOOP_WIN = 6;
private enum size_t T_THETA = 12; // theta period (steps/tokens)
private enum size_t T_GAMMA = 3; // gamma period (steps)
private enum double TAU_M = 12.0; // Membrane time constant (steps)
private enum double V_TH = 1.0; // Firing threshold
private enum double W_IE = 3.0; // I->E inhibition gain
private enum double W_EI = 0.6; // E->I drive gain
private enum double TAU_I = 5.0; // Inhibitory pool time constant
private enum double I_NOISE = 0.12; // Spontaneous firing noise
private enum double EOS_MARGIN = 1.4; // Stop symbol must be relative majority
private enum size_t MAX_SKIP = 4; // Cue completion: tolerated trailing noise tokens
private enum double RECALL_FLOOR = 3.5; // Memory confirmation threshold: reach this evidence to consider "remembered," otherwise treated as completely unknown
private enum double CONT_FLOOR = 3.0; // Per-step continuation confidence floor: if evidence drops below this while writing, the thread is lost -> abort and admit "unknown"

private enum double NM_DOPA_LR = 0.10;

private enum string EOS = "␃"; // Sequence terminator (codec constant)
private enum string BOS = "␂"; // Sequence starter (codec constant)
private enum string QEND = "␁"; // Question terminator (codec constant)


/*
    Math / Random / Hash
*/
private double clampF(double x, double lo, double hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

private uint xorshift32(ref uint state) {
    uint x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state = x;
    return x;
}

private uint gRandState = 0x9E3779B9u;

private uint nextRand() {
    if (gRandState == 0) gRandState = 1;
    return xorshift32(gRandState);
}

private double randUnit() {
    return cast(double)(nextRand() >> 8) / 16777216.0;
}

private uint seedRand(string s) {
    uint h = 2166136261u;
    for (ulong ptr = 0; ptr < s.length; ptr++) {
        h ^= cast(uint)(cast(ubyte)s[ptr]);
        h *= 16777619u;
    }
    if (h == 0) h = 1;
    return h;
}

private double mySqrt(double x) {
    if (x <= 0.0) return 0.0;
    double y = x;
    for (ulong ptr = 0; ptr < 20; ptr++) {
        double ny = 0.5 * (y + x / y);
        if (ny == y) break;
        y = ny;
    }
    return y;
}

private double mySin(double x) {
    const double PI = 3.14159265;
    while (x >  PI) x -= 2.0 * PI;
    while (x < -PI) x += 2.0 * PI;
    double B = 4.0 / PI;
    double C = -4.0 / (PI * PI);
    double ax = x < 0.0 ? -x : x;
    double y = B * x + C * x * ax;
    double ay = y < 0.0 ? -y : y;
    const double P = 0.225;
    return P * (y * ay - y) + y;
}

/*
    Tokenizer: the minimum unit is a whole ASCII word (so "Java" != "JavaScript")
    or a single non-ASCII grapheme (CJK, punctuation, codec symbols).
*/

private bool isAsciiAlnumChar(char c) {
    return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

private bool isAsciiAlnum(string g) {
    if (g.length != 1) return false;
    return isAsciiAlnumChar(g[0]);
}

private bool isWordJoin(string g) {
    if (g.length != 1) return false;
    char c = g[0];
    return c == '#' || c == '+' || c == '_' || c == '-';
}

private bool isSpacer(string g) {
    if (g.length != 1) return false;
    char c = g[0];
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\u3000';
}

private string[] tokenize(string s) {
    string[] res;
    string cur;
    foreach (chunk; s.byGrapheme) {
        auto g = chunk[].to!string;
        if (isSpacer(g)) {
            if (cur.length) {
                res ~= cur;
                cur = "";
            }
            continue;
        }
        if (isAsciiAlnum(g)) {
            cur ~= g;
            continue;
        }
        if (cur.length && isWordJoin(g)) {
            cur ~= g;
            continue;
        }
        if (cur.length) {
            res ~= cur;
            cur = "";
        }
        res ~= g;
    }
    if (cur.length) res ~= cur;
    return res;
}

/*
    Sensory encoder: grapheme -> EC cell assembly
*/

private string[] lexicon;
private ulong[string] lexId;
private ulong[][] assemblies;

/*
    CA3 associative memory: presynaptic = DG code neurons; postsynaptic = EC grapheme assembly neurons.

    Two backing modes:
      - Learning (RAM): jagged arrays, allocated only while teaching.
      - Running (mapped): the whole network is a memory-mapped file; the dense DG index is
        never allocated, so RAM stays small and the store may grow to many gigabytes on disk.
*/
private ulong[][] assocPost;
private double[][] assocW;

private MmFile gMap;
private const(ulong)[] gOff;
private const(ushort)* gPosts;
private const(double)* gWts;
private bool gMapped = false;

// Glyph ids of the codec symbols (BOS/EOS/QEND); size_t.max means "not yet resolved"
private __gshared size_t gBos = size_t.max, gEos = size_t.max, gQend = size_t.max;

private void refreshCodec() {
    gBos  = glyphId(BOS);
    gEos  = glyphId(EOS);
    gQend = glyphId(QEND);
}

private void allocNet() {
    lexicon = null;
    lexId = null;
    assemblies = null;
    gMap = null;
    gOff = null;
    gPosts = null;
    gWts = null;
    gMapped = false;
    assocPost = null;
    assocW = null;
}

// Allocate the RAM store lazily; if the network is currently mapped, pull it back into RAM first (learning after a load).
private void ensureRam() {
    if (gMapped) {
        auto post = new ulong[][](N_DG);
        auto w = new double[][](N_DG);
        for (ulong ptr = 0; ptr < N_DG; ptr++) {
            ulong cnt = gOff[ptr + 1] - gOff[ptr];
            if (cnt == 0) continue;
            post[ptr] = new ulong[cnt];
            w[ptr] = new double[cnt];
            ulong o = gOff[ptr];
            for (ulong i = 0; i < cnt; i++) {
                post[ptr][i] = gPosts[o + i];
                w[ptr][i] = gWts[o + i];
            }
        }
        assocPost = post;
        assocW = w;
        gMap = null;
        gOff = null;
        gPosts = null;
        gWts = null;
        gMapped = false;
        return;
    }
    if (assocPost is null) {
        assocPost = new ulong[][](N_DG);
        assocW = new double[][](N_DG);
    }
}

private ulong[] makeAssembly(string g) {
    uint st = seedRand("A:" ~ g);
    if (st == 0) st = 1;
    auto res = new ulong[K_ASM];
    auto seen = new bool[N_E];
    size_t got = 0;
    uint guard = 0;
    while (got < K_ASM && guard < K_ASM * 64) {
        uint r = xorshift32(st);
        guard++;
        size_t index = r % N_E;
        if (!seen[index]) {
            seen[index] = true;
            res[got++] = index;
        }
    }
    if (got == 0) {
        res.length = 1;
        res[0] = 0;
    } else if (got < K_ASM)
        res.length = got;
    return res;
}

private ulong glyphId(string g) {
    if (auto p = g in lexId) return *p;
    ulong id = lexicon.length;
    lexicon ~= g;
    lexId[g] = id;
    assemblies ~= makeAssembly(g);
    return id;
}

/*
    Dentate gyrus (DG): context window -> sparse near-orthogonal code (pattern separation)
*/
private ulong[] dgCode(ulong[] ctx) {
    uint h = 2166136261u;
    for (ulong ptr = 0; ptr < ctx.length; ptr++) {
        uint v = cast(uint)ctx[ptr] * 2654435761u + 0x9E3779B9u;
        h ^= v;
        h = (h << 13) | (h >> 19);
        h *= 16777619u;
        h ^= h >> 15;
    }
    if (h == 0) h = 1;
    uint st = h;
    auto res = new ulong[K_DG];
    size_t got = 0;
    uint guard = 0;
    while (got < K_DG && guard < K_DG * 64) {
        uint r = xorshift32(st);
        guard++;
        ulong index = r % N_DG;
        bool dup = false;
        for (ulong i = 0; i < got; i++)
            if (res[i] == index) {
                dup = true;
                break;
            }
        if (!dup) res[got++] = index;
    }
    if (got == 0) {
        res.length = 1;
        res[0] = h % N_DG;
    } else if (got < K_DG)
        res.length = got;
    return res;
}

/*
    CA3 synapse storage
*/
private void addAssoc(ulong pre, ulong post, double delta) {
    if (gMapped) ensureRam();
    auto posts = assocPost[pre];
    for (ulong ptr = 0; ptr < posts.length; ptr++) {
        if (posts[ptr] == post) {
            double nw = assocW[pre][ptr] + delta;
            if (nw > W_MAX) nw = W_MAX;
            if (nw < W_FLOOR) nw = W_FLOOR;
            assocW[pre][ptr] = nw;
            return;
        }
    }
    if (delta <= 0.0) return;
    if (posts.length >= MAX_OUT) return;
    assocPost[pre] ~= post;
    assocW[pre] ~= delta;
}

/*
    Neuromodulators
*/
private struct Neuromod {
    double dopamine       = 0.5;
    double serotonin      = 0.5;
    double acetylcholine  = 0.5;
    double norepinephrine = 0.5;
}
private Neuromod nm;

private double plasticityGain() {
    return 0.4 + 0.6 * nm.dopamine * (0.5 + 0.5 * nm.acetylcholine);
}

private void neuromodUpdate(double reward) {
    nm.dopamine += NM_DOPA_LR * (reward - nm.dopamine);
    nm.dopamine = clampF(nm.dopamine, 0.0, 1.0);
    nm.acetylcholine  = 0.3 + 0.6 * nm.dopamine;
    nm.norepinephrine = 0.3 + 0.6 * nm.dopamine;
}

/*
    Learning: multi-scale context -> next grapheme
*/
private void learnSeq(ulong[] ids) {
    if (ids.length < 2) return;
    double gain = plasticityGain();
    for (ulong ptr = 1; ptr <= CTXW; ptr++) {
        if (ids.length <= ptr) continue;
        for (ulong i = ptr; i < ids.length; i++) {
            if (ptr == 1 && ids[i - 1] == gQend) continue; // Avoid the over-general association [QEND] -> arbitrary first glyph of the answer
            auto code  = dgCode(ids[i - ptr .. i]);
            auto posts = assemblies[ids[i]];
            for (ulong j = 0; j < code.length; j++)
                for (ulong k = 0; k < posts.length; k++)
                    addAssoc(code[j], posts[k], ETA * gain);
        }
    }
}

private ulong[] tokensToIds(string text) {
    auto gs = tokenize(text);
    auto ids = new ulong[gs.length];
    for (ulong ptr = 0; ptr < gs.length; ptr++) ids[ptr] = glyphId(gs[ptr]);
    return ids;
}

// Generation: CA3 association-driven -> LIF E/I competition within a theta cycle (WTA)
private bool tailLoop(string[] gs) {
    if (gs.length < LOOP_WIN * 2) return false;
    auto tail = gs[$ - LOOP_WIN .. $];
    for (ulong ptr = 0; ptr < gs.length - LOOP_WIN * 2 + 1; ptr++) {
        bool ok = true;
        for (ulong i = 0; i < LOOP_WIN; i++)
            if (gs[ptr + i] != tail[i]) {
                ok = false;
                break;
            }
        if (ok) return true;
    }
    return false;
}

private double buildInput(ulong[] ids, size_t w, double[] In) {
    In[] = 0.0;
    size_t win = ids.length < w ? ids.length : w;
    if (win == 0) return 0.0;
    auto code = dgCode(ids[ids.length - win .. $]);
    if (gMapped) {
        for (ulong ptr = 0; ptr < code.length; ptr++) {
            ulong o = gOff[code[ptr]];
            ulong e = gOff[code[ptr] + 1];
            for (ulong i = o; i < e; i++) In[gPosts[i]] += gWts[i];
        }
    } else {
        for (ulong ptr = 0; ptr < code.length; ptr++) {
            auto posts = assocPost[code[ptr]];
            auto ws    = assocW[code[ptr]];
            for (ulong i = 0; i < posts.length; i++) In[posts[i]] += ws[i];
        }
    }
    double dmax = 0.0;
    for (ulong ptr = 0; ptr < assemblies.length; ptr++) {
        double s = 0.0;
        for (ulong i = 0; i < assemblies[ptr].length; i++) s += In[assemblies[ptr][i]];
        if (s > dmax) dmax = s;
    }
    return dmax;
}

string generate(string prefix) {
    if (assemblies.length == 0) return "";
    auto gs = tokenize(prefix);
    if (gs.length == 0) return "";
    if (gQend == size_t.max) refreshCodec();

    auto In  = new double[N_E];
    auto Ve  = new double[N_E]; Ve[] = 0.0;
    auto rG  = new byte[N_E];
    auto spk = new double[N_E];
    string result;
    string[] emitted;
    ulong[] ids;
    bool lostThread = false; // Set when continuation confidence collapses -> discard and admit "unknown"

    for (ulong ptr = 0; ptr < MAX_LEN; ptr++) {
        double dmax = 0.0;
        if (ptr == 0) {
            // Feed the cue in as-is; do not delete or alter it. Trailing noise (punctuation/newline, etc.) is absorbed by pattern completion.
            // First, use the full-match score of the whole candidate to drop noisy candidates
            auto tmp = new double[N_E];
            double bestFull = -1.0;
            ulong[] bestIds;
            for (ulong i = 0; i < MAX_SKIP + 1; i++) {
                if (i > gs.length) break;
                size_t keep = gs.length - i;
                auto cand = new ulong[keep + 2];
                cand[0] = gBos;
                for (ulong j = 0; j < keep; j++) cand[j + 1] = glyphId(gs[j]);
                cand[keep + 1] = gQend;
                size_t fullW = cand.length < CTXW ? cand.length : CTXW;
                double d = buildInput(cand, fullW, tmp);
                if (d > bestFull) {
                    bestFull = d;
                    bestIds = cand;
                }
            }
            ids  = bestIds;
            // Pattern separation: the whole question must be familiar. A partial (short-window) match
            // is a coincidence of one shared token, not recognition -> treat it as unknown instead.
            size_t fullW = ids.length < CTXW ? ids.length : CTXW;
            dmax = buildInput(ids, fullW, In);
            if (dmax < RECALL_FLOOR) dmax = 0.0; // Nothing familiar enough -> silence
        } else {
            size_t fullW = ids.length < CTXW ? ids.length : CTXW;
            dmax = buildInput(ids, fullW, In); // Continue writing: use the most recent and specific context
            if (dmax < CONT_FLOOR) { lostThread = true; break; } // Lost the thread: abandon this answer
        }
        if (dmax <= 0.0) break; // No match -> silence

        double gain = cast(double)K_ASM / dmax; // Sensory gain normalization
        for (ulong i = 0; i < N_E; i++) In[i] *= gain;
        for (ulong i = 0; i < N_E; i++)
            if (rG[i] > 0) rG[i]--; // Afterhyperpolarization decay

        // A theta-cycle LIF pulse simulation (gamma-gated E->I->E inhibition)
        Ve[] = 0.0; spk[] = 0.0;
        double Vic = 0.0;
        for (ulong i = 0; i < T_THETA; i++) {
            double gamma = 0.6 + 0.4 * mySin(6.2831853 * cast(double)i / cast(double)T_GAMMA);
            double theta = 0.6 + 0.4 * mySin(6.2831853 * cast(double)i / cast(double)T_THETA);
            double inh = W_IE * Vic;
            ulong stepSpikes = 0;
            for (ulong j = 0; j < N_E; j++) {
                if (rG[j] > 0) continue;
                double noise = I_NOISE * (randUnit() * 2.0 - 1.0);
                double drive = In[j] * gamma * theta + noise - inh;
                Ve[j] += (0.0 - Ve[j]) / TAU_M + drive;
                if (Ve[j] >= V_TH) {
                    Ve[j] = 0.0;
                    rG[j] = 2;
                    stepSpikes++;
                    spk[j] += 1.0;
                }
            }
            Vic = Vic * (1.0 - 1.0 / TAU_I) + W_EI * (cast(double)stepSpikes / cast(double)K_ASM);
        }

        // Decode: the glyph with the highest total pulse sum wins
        size_t G = assemblies.length;
        size_t win = size_t.max;
        double best = 0.0, eosSpk = 0.0;
        for (ulong i = 0; i < G; i++) {
            if (i == gBos || i == gQend) continue;
            double s = 0.0;
            for (ulong j = 0; j < assemblies[i].length; j++) s += spk[assemblies[i][j]];
            if (lexicon[i] == EOS) {
                eosSpk = s;
                continue;
            }
            if (s > best) {
                best = s;
                win = i;
            }
        }
        if (ptr == 0 && eosSpk > best) break;
        if (eosSpk > 0.0 && eosSpk > EOS_MARGIN * best) break;
        if (win == size_t.max || best <= 0.0) break;

        auto ng = lexicon[win];
        if (result.length && isAsciiAlnumChar(cast(char)result[$ - 1]) && isAsciiAlnumChar(cast(char)ng[0]))
            result ~= " ";
        result ~= ng;
        emitted ~= ng;
        ids ~= win;

        if (tailLoop(emitted)) break;
    }
    return lostThread ? "" : result;
}

string talk(string msg) {
    return generate(msg);
}

// Internal "I don't know" behavior: the wording is learned memory (trained under a private cue),
// not a literal in the caller. The choice among learned variants is driven by the network's own
// stochastic state (entropy-seeded), so the utterance varies across calls like a real speaker's.
private uint gDoubtState = 0;
string doubtReply() {
    immutable string[] cues = ["\u2404", "\u2405", "\u2406", "\u2407"];
    if (gDoubtState == 0) {
        gDoubtState = unpredictableSeed;
        if (gDoubtState == 0) gDoubtState = 1;
    }
    for (size_t ptr = 0; ptr < cues.length; ptr++) {
        gDoubtState = xorshift32(gDoubtState);
        string res = generate(cues[gDoubtState % cues.length]);
        if (res.strip.length) return res;
    }
    return "";
}


// Public training interface
void teach(string[] usd, string[] asd) {
    if (usd.length != asd.length)
        throw new Exception("Size Unmatched.");
    ensureRam();
    refreshCodec();
    for (ulong ptr = 0; ptr < usd.length; ptr++) {
        if (usd[ptr].length == 0) continue;
        learnSeq(tokensToIds(BOS ~ usd[ptr] ~ QEND ~ asd[ptr] ~ EOS));
    }
}

void trainSentence(string sentence) {
    if (sentence.length == 0) return;
    ensureRam();
    refreshCodec();
    learnSeq(tokensToIds(BOS ~ sentence ~ EOS));
}

void right(string q, string a) {
    ensureRam();
    refreshCodec();
    neuromodUpdate(1.0);
    learnSeq(tokensToIds(BOS ~ q ~ QEND ~ a ~ EOS));
}

void wrong(string q, string a) {
    ensureRam();
    refreshCodec();
    neuromodUpdate(0.0);
    auto gq = tokenize(q);
    auto ga = tokenize(a);
    if (gq.length == 0 || ga.length == 0) return;
    auto code = dgCode([glyphId(gq[$ - 1])]);
    auto ga0  = assemblies[glyphId(ga[0])];
    for (ulong ptr = 0; ptr < code.length; ptr++)
        for (ulong i = 0; i < ga0.length; i++)
            addAssoc(code[ptr], ga0[i], -A_MINUS);
}

// Sleep / Consolidation
void sleep() {
    if (assocW is null) return;
    for (ulong ptr = 0; ptr < N_DG; ptr++) {
        auto ws = assocW[ptr];
        if (ws.length == 0) continue;
        double tot = 0.0;
        for (ulong i = 0; i < ws.length; i++) tot += ws[i];
        if (tot > 0.0) {
            double k = OUT_TARGET / tot;
            for (ulong i = 0; i < ws.length; i++) ws[i] *= k;
        }
    }
    nm.dopamine       = 0.5;
    nm.serotonin      = 0.5;
    nm.acetylcholine  = 0.5;
    nm.norepinephrine = 0.5;
}

void consolidate() { sleep(); }
void clearHippocampus() { }
ulong lexiconCount() { return lexicon.length; }

/*
    STO12 serialization (compact, memory-mapped):
        "STO12" | lexicon | pad8 | offsets[N_DG + 1] | posts[E] (ushort) | pad8 | weights[E] (double) | neuromod (4 doubles)
    posts and weights share the offsets; a presynaptic neuron's entries are the contiguous
    range [offsets[pre], offsets[pre + 1]). Running maps the file directly, so RAM use is
    independent of the model size and the store may grow to many gigabytes on disk.
*/
private size_t align8(size_t x) {
    return (x + 7) & ~cast(size_t)7;
}

private void writeU(ref ubyte[] buf, ulong v) {
    for (ulong ptr = 0; ptr < 8; ptr++) {
        buf ~= cast(ubyte)(v & 0xFF);
        v >>= 8;
    }
}
private void writeS(ref ubyte[] buf, string s) {
    writeU(buf, s.length);
    for (ulong ptr = 0; ptr < s.length; ptr++) buf ~= cast(ubyte)s[ptr];
}
private ulong readU(const(ubyte)[] buf, ref size_t pos) {
    if (pos + 8 > buf.length) throw new Exception("Truncated .sto (u)");
    ulong v = 0;
    for (ulong ptr = 0; ptr < 8; ptr++) v |= cast(ulong)buf[pos + ptr] << (8 * ptr);
    pos += 8;
    return v;
}
private string readS(const(ubyte)[] buf, ref size_t pos) {
    ulong len = readU(buf, pos);
    if (pos + len > buf.length) throw new Exception("Truncated .sto (s)");
    string s = cast(string)buf[pos .. pos + len].idup;
    pos += len;
    return s;
}

// Persist the RAM store as a compact, mmap-friendly file. No-op when already mapped.
void saveCortex(string path) {
    if (gMapped) return;
    auto off = new ulong[N_DG + 1];
    ulong acc = 0;
    for (ulong ptr = 0; ptr < N_DG; ptr++) {
        off[ptr] = acc;
        acc += assocPost[ptr].length;
    }
    off[N_DG] = acc;

    auto f = File(path, "wb");

    ubyte[] hdr;
    hdr ~= cast(ubyte)'S'; hdr ~= cast(ubyte)'T';
    hdr ~= cast(ubyte)'O'; hdr ~= cast(ubyte)'1'; hdr ~= cast(ubyte)'2';
    writeU(hdr, lexicon.length);
    for (ulong ptr = 0; ptr < lexicon.length; ptr++) writeS(hdr, lexicon[ptr]);
    while (hdr.length % 8) hdr ~= cast(ubyte)0;
    f.rawWrite(hdr);
    f.rawWrite(off);

    // Posts (ushort), streamed in 1 MiB chunks.
    auto buf = new ubyte[1 << 20];
    size_t bi = 0;
    for (ulong ptr = 0; ptr < N_DG; ptr++) {
        auto posts = assocPost[ptr];
        for (ulong i = 0; i < posts.length; i++) {
            if (bi + 2 > buf.length) {
                f.rawWrite(buf[0 .. bi]);
                bi = 0;
            }
            buf[bi++] = cast(ubyte)(posts[i] & 0xFF);
            buf[bi++] = cast(ubyte)((posts[i] >> 8) & 0xFF);
        }
    }
    if (bi > 0) f.rawWrite(buf[0 .. bi]);
    if ((acc * 2) % 8 != 0) {
        ubyte[8] pad;
        f.rawWrite(pad[0 .. 8 - (acc * 2) % 8]);
    }

    // Weights (double), streamed in chunks.
    auto wbuf = new double[1 << 17];
    size_t wi = 0;
    for (ulong ptr = 0; ptr < N_DG; ptr++) {
        auto ws = assocW[ptr];
        for (ulong i = 0; i < ws.length; i++) {
            if (wi == wbuf.length) {
                f.rawWrite(wbuf[0 .. wi]);
                wi = 0;
            }
            wbuf[wi++] = ws[i];
        }
    }
    if (wi > 0) f.rawWrite(wbuf[0 .. wi]);

    double[4] nmv = [nm.dopamine, nm.serotonin, nm.acetylcholine, nm.norepinephrine];
    f.rawWrite(nmv);
}

// Load the store by memory-mapping it; the network itself is never copied into the heap.
void openCortex(string path) {
    lexicon = null;
    lexId = null;
    assemblies = null;

    // Validate the header before mapping, so a rejected file is never left locked.
    {
        auto f = File(path, "rb");
        ubyte[5] magic;
        auto got = f.rawRead(magic);
        if (got.length != 5 || magic[0] != 'S' || magic[1] != 'T' || magic[2] != 'O' || magic[3] != '1' || magic[4] != '2')
            throw new Exception("Unsupported .sto version");
    }

    gMap = new MmFile(path);
    auto buf = cast(const(ubyte)[])gMap[];
    if (buf.length < 5)
        throw new Exception("Truncated .sto");
    size_t pos = 5;
    ulong lexN = readU(buf, pos);
    for (ulong ptr = 0; ptr < lexN; ptr++) glyphId(readS(buf, pos));

    size_t offBase = align8(pos);
    size_t postsBase = offBase + (N_DG + 1) * 8;
    gOff = cast(const(ulong)[])buf[offBase .. postsBase];
    ulong E = gOff[N_DG];
    size_t wBase = align8(postsBase + cast(size_t)(E * 2));
    gPosts = cast(const(ushort)*)&buf[postsBase];
    gWts = cast(const(double)*)&buf[wBase];
    gMapped = true;
    refreshCodec();
}

static this() {
    allocNet();
}