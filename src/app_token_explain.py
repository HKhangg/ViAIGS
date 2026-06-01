"""
Token Attribution Heatmap Viewer — Streamlit
=============================================
Chạy:
    pip install streamlit
    streamlit run token_heatmap_app.py

Upload file JSON (array of samples) có format:
[
  {
    "sample_id": 0,
    "text": "...",
    "source": "fb",
    "category": "TP",
    "predicted_label": 1,
    "true_label": 1,
    "tokens": [
      {"position": 0, "raw_token": "<s>", "clean_token": "<s>", "score": 0.0},
      ...
    ]
  },
  ...
]
"""

import json
import math
import streamlit as st
import streamlit.components.v1 as components

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Token Heatmap Viewer",
    page_icon="🔬",
    layout="wide",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&display=swap');

.token-wrap   { line-height: 2.8; word-spacing: 2px; padding: 12px 0; }
.token-chip   { display: inline-block; padding: 2px 4px; margin: 0 1px;
                border-radius: 0; font-family: 'IBM Plex Mono', monospace;
                font-size: 13px; cursor: default; }
.legend-wrap  { margin-top: 10px; }
.legend-bar   { display: flex; border-radius: 5px; overflow: hidden;
                width: 260px; height: 22px; }
.legend-cell  { flex: 1; display: flex; align-items: center; justify-content: center;
                font-size: 10px; font-family: 'IBM Plex Mono', monospace; font-weight: 600; }
.meta-row     { display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
                margin-bottom: 8px; }
.badge        { display: inline-block; padding: 2px 10px; border-radius: 5px;
                font-size: 11px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─── Color helpers ─────────────────────────────────────────────────────────────
def lerp_rgb(a, b, t):
    return [round(a[i] + (b[i] - a[i]) * t) for i in range(3)]

def score_to_rgb(score, max_abs):
    NEUTRAL  = [248, 247, 244]
    RED_MID  = [230, 120, 80]
    RED_MAX  = [175, 45,  45]
    BLUE_MID = [100, 165, 220]
    BLUE_MAX = [24,  75,  160]
    t = min(abs(score) / (max_abs or 0.3), 1.0)
    if score > 0:
        bg = lerp_rgb(NEUTRAL, RED_MID, t*2) if t < 0.5 else lerp_rgb(RED_MID, RED_MAX, (t-0.5)*2)
    elif score < 0:
        bg = lerp_rgb(NEUTRAL, BLUE_MID, t*2) if t < 0.5 else lerp_rgb(BLUE_MID, BLUE_MAX, (t-0.5)*2)
    else:
        bg = NEUTRAL
    lum = (0.299*bg[0] + 0.587*bg[1] + 0.114*bg[2]) / 255
    fg = '#1a1a1a' if lum > 0.45 else '#f5f5f0'
    return f"rgb({bg[0]},{bg[1]},{bg[2]})", fg

def render_heatmap_component(tokens, max_abs, sample_id):
    """Group SentencePiece sub-tokens → words, compute mean score, render heatmap."""

    # ── 1. Group sub-tokens into words by ▁ boundary ──────────────────────────
    SKIP_TOKENS = {"<s>", "</s>"}
    words = []
    for tok in tokens:
        ct = tok["clean_token"]
        if ct in SKIP_TOKENS:
            continue
        is_new = ct.startswith("\u2581") or not words
        part   = ct.replace("\u2581", "")
        if not part.strip():
            continue
        if is_new:
            words.append({"label": part, "scores": [tok["score"]], "positions": [tok["position"]]})
        else:
            words[-1]["label"]    += part
            words[-1]["scores"]   .append(tok["score"])
            words[-1]["positions"].append(tok["position"])

    # ── 2. Build Python list of word dicts ─────────────────────────────────────
    word_data = []
    for w in words:
        mean_score = sum(w["scores"]) / len(w["scores"])
        bg, fg     = score_to_rgb(mean_score, max_abs)
        sign       = "+" if mean_score >= 0 else ""
        score_str  = f"{sign}{mean_score:.5f}"
        sub_detail = " | ".join(
            f"pos{p}:{('+' if sc>=0 else '')}{sc:.4f}"
            for p, sc in zip(w["positions"], w["scores"])
        )
        word_data.append({
            "label":    w["label"],
            "bg":       bg,
            "fg":       fg,
            "score":    score_str,
            "nTokens":  len(w["scores"]),
            "detail":   sub_detail,
            "pos":      w["positions"][0],
        })

    # ── 3. Legend cells ────────────────────────────────────────────────────────
    legend_cells = ""
    for v in [-1, -0.5, 0, 0.5, 1]:
        s      = v * max_abs
        bg, fg = score_to_rgb(s, max_abs)
        lbl    = ("+" if v >= 0 else "") + f"{s:.2f}"
        legend_cells += f'<div class="lcell" style="background:{bg};color:{fg}">{lbl}</div>'

    # ── 4. Serialize to JSON safely via Python json module ─────────────────────
    import json as _json
    tokens_json  = _json.dumps(word_data)
    legend_html  = legend_cells

    # ── 5. Build HTML — NO f-string for the JS block to avoid brace escaping ──
    html_parts = []
    html_parts.append("""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
html,body{overflow:visible;background:transparent;font-family:system-ui,sans-serif;padding:4px 2px}
#wrap{line-height:3;word-spacing:2px;padding:4px 0 8px}
.chip{
  display:inline-block;padding:2px 3px;margin:0 1px;border-radius:0;
  font-family:'IBM Plex Mono',monospace;font-size:14px;cursor:pointer;
  transition:outline 0.1s;
}
.chip:hover,.chip.pinned{outline:2px solid #333}
#floatip{
  display:none;position:fixed;
  background:#111827;color:#f9fafb;
  font-size:11px;line-height:1.8;padding:7px 11px;border-radius:7px;
  white-space:nowrap;z-index:99999;pointer-events:none;
  box-shadow:0 2px 10px rgba(0,0,0,.4);
  font-family:'IBM Plex Mono',monospace;
}
#floatip.show{display:block}
.arr{content:'';position:absolute;left:50%;border:5px solid transparent;}
.arr.up{top:100%;border-top-color:#111827;}
.arr.dn{bottom:100%;border-bottom-color:#111827;}
.legend{margin-top:12px}
.ll{font-size:10px;font-weight:600;color:#9ca3af;letter-spacing:.06em;text-transform:uppercase;margin-bottom:4px}
.lbar{display:flex;border-radius:5px;overflow:hidden;width:260px;height:22px}
.lcell{flex:1;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:600;font-family:'IBM Plex Mono',monospace}
.lann{display:flex;justify-content:space-between;width:260px;font-size:10px;margin-top:3px}
</style></head><body>
<div id="wrap"></div>
<div id="floatip"><span class="arr up" id="arr"></span><span id="tip-content"></span></div>
<div class="legend">
  <div class="ll">Attribution scale</div>
  <div class="lbar">""")

    html_parts.append(legend_html)

    html_parts.append("""</div>
  <div class="lann">
    <span style="color:#1264a3">&#8592; negative</span>
    <span style="color:#b43232">positive &#8594;</span>
  </div>
</div>
<script>
(function() {
  var DATA = """)

    html_parts.append(tokens_json)

    html_parts.append(""";
  var wrap    = document.getElementById('wrap');
  var floatip = document.getElementById('floatip');
  var tipContent = document.getElementById('tip-content');
  var arr     = document.getElementById('arr');
  var pinned  = null;

  function showTip(chip, d) {
    var sub = d.nTokens > 1
      ? '<br><span style="opacity:.65;font-size:10px">' + d.nTokens + ' sub-tokens &middot; mean</span>'
        + '<br><span style="opacity:.55;font-size:9px">' + d.detail + '</span>'
      : '<br><span style="opacity:.65;font-size:10px">pos: ' + d.pos + '</span>';
    tipContent.innerHTML = '<strong>' + d.label + '</strong><br>score: ' + d.score + sub;
    floatip.classList.add('show');
    positionTip(chip);
  }

  function positionTip(chip) {
    var r  = chip.getBoundingClientRect();
    var tw = floatip.offsetWidth;
    var th = floatip.offsetHeight;
    var vw = window.innerWidth;
    var vh = window.innerHeight;
    var cx = r.left + r.width / 2;
    var left = Math.max(6, Math.min(cx - tw / 2, vw - tw - 6));
    if (r.top >= th + 14) {
      floatip.style.top  = (r.top - th - 10) + 'px';
      floatip.style.left = left + 'px';
      arr.className = 'arr up';
    } else {
      floatip.style.top  = (r.bottom + 8) + 'px';
      floatip.style.left = left + 'px';
      arr.className = 'arr dn';
    }
    arr.style.left      = Math.max(10, Math.min(cx - left - 5, tw - 15)) + 'px';
    arr.style.transform = 'none';
  }

  DATA.forEach(function(d) {
    var chip = document.createElement('span');
    chip.className        = 'chip';
    chip.style.background = d.bg;
    chip.style.color      = d.fg;
    chip.textContent      = d.label;

    chip.addEventListener('mouseenter', function() { showTip(chip, d); });
    chip.addEventListener('mouseleave', function() { if (pinned !== chip) floatip.classList.remove('show'); });
    chip.addEventListener('click', function(e) {
      e.stopPropagation();
      if (pinned === chip) {
        pinned = null;
        chip.classList.remove('pinned');
        floatip.classList.remove('show');
      } else {
        if (pinned) pinned.classList.remove('pinned');
        pinned = chip;
        chip.classList.add('pinned');
        showTip(chip, d);
      }
    });

    wrap.appendChild(chip);
    wrap.appendChild(document.createTextNode(''));
  });

  document.addEventListener('click', function() {
    if (pinned) { pinned.classList.remove('pinned'); pinned = null; }
    floatip.classList.remove('show');
  });
})();
</script></body></html>""")

    html = "".join(html_parts)
    estimated_height = max(160, min(len(word_data) * 7 + 150, 520))
    components.html(html, height=estimated_height, scrolling=False)


def render_legend_html(max_abs):
    stops = [-1, -0.5, 0, 0.5, 1]
    cells = ''
    for v in stops:
        s = v * max_abs
        bg, fg = score_to_rgb(s, max_abs)
        label = f"{'+' if v >= 0 else ''}{s:.2f}"
        cells += f'<div class="legend-cell" style="background:{bg};color:{fg}">{label}</div>'
    return f'''
    <div class="legend-wrap">
      <div style="font-size:10px;font-weight:600;color:#9ca3af;letter-spacing:.06em;
                  text-transform:uppercase;margin-bottom:4px">Attribution scale</div>
      <div class="legend-bar">{cells}</div>
      <div style="display:flex;justify-content:space-between;width:260px;
                  font-size:10px;margin-top:3px">
        <span style="color:#1264a3">← negative</span>
        <span style="color:#b43232">positive →</span>
      </div>
    </div>'''

# ─── Data helpers ──────────────────────────────────────────────────────────────
def load_json(uploaded_file):
    raw = json.loads(uploaded_file.read())
    if isinstance(raw, list):
        return raw
    for key in ('samples', 'data', 'results'):
        if key in raw and isinstance(raw[key], list):
            return raw[key]
    # fallback: first list value
    for v in raw.values():
        if isinstance(v, list):
            return v
    raise ValueError("Cannot find samples array in JSON")

def max_abs_score(samples):
    return max(
        (abs(t['score']) for s in samples for t in s['tokens']),
        default=0.3
    )

def sample_max_score(sample):
    return max((abs(t['score']) for t in sample['tokens']), default=0.0)

def top_tokens(sample, n=3):
    skip = {'<s>', '</s>', '▁'}
    filtered = [t for t in sample['tokens'] if t['clean_token'] not in skip]
    return sorted(filtered, key=lambda t: abs(t['score']), reverse=True)[:n]

# ─── Session state ─────────────────────────────────────────────────────────────
if 'samples' not in st.session_state:
    st.session_state.samples = None

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 Token Heatmap")
    st.markdown("---")

    uploaded = st.file_uploader("Upload JSON file", type=["json"])
    if uploaded:
        try:
            st.session_state.samples = load_json(uploaded)
            st.success(f"Loaded {len(st.session_state.samples):,} samples")
        except Exception as e:
            st.error(f"Parse error: {e}")

    if st.session_state.samples:
        st.markdown("---")
        st.markdown("**Filters**")
        samples = st.session_state.samples

        categories = ['All'] + sorted(set(s['category'] for s in samples))
        sources     = ['All'] + sorted(set(s['source']   for s in samples))

        sel_cat     = st.selectbox("Category", categories)
        sel_src     = st.selectbox("Source",   sources)
        sel_correct = st.selectbox("Prediction", ['All', 'Correct only', 'Wrong only'])
        search_txt  = st.text_input("Search text / ID", placeholder="keyword or sample id…")
        sort_by     = st.selectbox("Sort by", ['Sample ID', 'Max score ↓', 'Max score ↑'])

        st.markdown("---")
        if st.button("🗑 Unload file"):
            st.session_state.samples = None
            st.rerun()

# ─── Main area ─────────────────────────────────────────────────────────────────
if st.session_state.samples is None:
    st.markdown("## 👈 Upload a JSON file to get started")
    st.markdown("""
Expected format — a JSON array of samples:
```json
[
  {
    "sample_id": 0,
    "text": "nội dung văn bản...",
    "source": "fb",
    "category": "TP",
    "predicted_label": 1,
    "true_label": 1,
    "tokens": [
      {"position": 0, "raw_token": "<s>", "clean_token": "<s>", "score": 0.0},
      {"position": 1, "raw_token": "▁cơ", "clean_token": "▁cơ", "score": 0.010}
    ]
  }
]
```
    """)
    st.stop()

# ─── Apply filters ─────────────────────────────────────────────────────────────
samples = st.session_state.samples
filtered = samples

if sel_cat != 'All':
    filtered = [s for s in filtered if s['category'] == sel_cat]
if sel_src != 'All':
    filtered = [s for s in filtered if s['source'] == sel_src]
if sel_correct == 'Correct only':
    filtered = [s for s in filtered if s['predicted_label'] == s['true_label']]
elif sel_correct == 'Wrong only':
    filtered = [s for s in filtered if s['predicted_label'] != s['true_label']]
if search_txt:
    q = search_txt.lower()
    filtered = [s for s in filtered if q in s['text'].lower() or q in str(s['sample_id'])]

if sort_by == 'Max score ↓':
    filtered = sorted(filtered, key=sample_max_score, reverse=True)
elif sort_by == 'Max score ↑':
    filtered = sorted(filtered, key=sample_max_score)

# ─── Stats bar ────────────────────────────────────────────────────────────────
global_max_abs = max_abs_score(samples)
n_correct = sum(1 for s in filtered if s['predicted_label'] == s['true_label'])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Showing",  f"{len(filtered):,} / {len(samples):,}")
col2.metric("Correct",  f"{n_correct:,}",  f"{n_correct/max(len(filtered),1)*100:.1f}%")
col3.metric("Wrong",    f"{len(filtered)-n_correct:,}")
col4.metric("Max |score|", f"{global_max_abs:.4f}")

st.markdown("---")

# ─── Sample table + expandable rows ───────────────────────────────────────────
if not filtered:
    st.warning("No samples match your filters.")
    st.stop()

# Table header
h1, h2, h3, h4, h5, h6 = st.columns([1, 4, 1.2, 1, 1.2, 1.5])
h1.markdown("**ID**")
h2.markdown("**Top tokens + text**")
h3.markdown("**Category**")
h4.markdown("**Source**")
h5.markdown("**Max |score|**")
h6.markdown("**Prediction**")
st.markdown('<hr style="margin:4px 0 8px">', unsafe_allow_html=True)

# Pagination — show 50 at a time for perf
PAGE = 50
total_pages = math.ceil(len(filtered) / PAGE)
if total_pages > 1:
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1) - 1
else:
    page = 0

page_samples = filtered[page*PAGE : (page+1)*PAGE]

for s in page_samples:
    is_correct  = s['predicted_label'] == s['true_label']
    max_s       = sample_max_score(s)
    top3        = top_tokens(s, 3)

    c1, c2, c3, c4, c5, c6 = st.columns([1, 4, 1.2, 1, 1.2, 1.5])

    c1.markdown(f"<small style='color:#9ca3af;font-family:monospace'>#{s['sample_id']}</small>", unsafe_allow_html=True)

    # Top tokens as colored chips + truncated text
    chips = ''
    for t in top3:
        bg, fg = score_to_rgb(t['score'], global_max_abs)
        label = t['clean_token'].replace('▁', '')
        chips += f'<span style="background:{bg};color:{fg};padding:1px 3px;border-radius:0;font-size:11px;font-family:monospace;margin-right:1px">{label}</span>'
    preview = s['text'][:70] + ('…' if len(s['text']) > 70 else '')
    c2.markdown(f"{chips} <small style='color:#d1d5db'>{preview}</small>", unsafe_allow_html=True)

    c3.markdown(f'<span style="background:#dbeafe;color:#1e40af;padding:2px 8px;border-radius:5px;font-size:11px;font-weight:600">{s["category"]}</span>', unsafe_allow_html=True)
    c4.markdown(f"<small style='color:#6b7280'>{s['source']}</small>", unsafe_allow_html=True)
    c5.markdown(f"<code style='font-size:12px'>{max_s:.4f}</code>", unsafe_allow_html=True)

    pred_bg  = '#dcfce7' if is_correct else '#fee2e2'
    pred_fg  = '#166534' if is_correct else '#991b1b'
    pred_lbl = f"✓ {s['predicted_label']}" if is_correct else f"✗ pred={s['predicted_label']} true={s['true_label']}"
    c6.markdown(f'<span style="background:{pred_bg};color:{pred_fg};padding:2px 8px;border-radius:5px;font-size:11px;font-weight:600">{pred_lbl}</span>', unsafe_allow_html=True)

    # Expandable heatmap
    with st.expander(f"▶ Show heatmap for sample #{s['sample_id']}"):
        st.markdown(f"**{s['text']}**")
        st.caption("Hover over a token to see its score · Click to pin")
        render_heatmap_component(s['tokens'], global_max_abs, s['sample_id'])

    st.markdown('<hr style="margin:4px 0">', unsafe_allow_html=True)

if total_pages > 1:
    st.caption(f"Page {page+1} of {total_pages} · {len(filtered):,} samples total")