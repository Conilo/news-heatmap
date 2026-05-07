"""Curate golden rows for live SLM pytest eval (expected_* fields).

Run as: streamlit run src/dashboard.py — open **Label eval cases** in the sidebar.
"""

from __future__ import annotations

import csv
import io
import os
import sys

import pandas as pd
import streamlit as st

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
for _p in (_REPO_ROOT, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
from geo_normalize import normalize_state
from store import load

VALID_STATES = config.VALID_STATES
VALID_EVENT_TYPES = config.VALID_EVENT_TYPES

DRAFT_BASENAME = "slm_eval_label_draft.csv"

STATE_SELECT_OPTIONS: tuple[str, ...] = (*VALID_STATES, "(custom)")
EVENT_SELECT_OPTIONS: tuple[str, ...] = (*VALID_EVENT_TYPES, "(custom)")


def _draft_csv_path() -> str:
    return os.path.join(_REPO_ROOT, config.DATA_DIR, DRAFT_BASENAME)


EXPORT_COLS: list[str] = [
    "case_id",
    *config.CSV_COLUMNS,
    "expected_state",
    "notes",
    "expected_municipality",
    "expected_group",
    "expected_event_type",
]

st.set_page_config(page_title="Label eval cases", layout="wide")


def _match_state_option(row_state: str) -> tuple[int, str]:
    """Return selectbox index into STATE_SELECT_OPTIONS and default custom text."""
    raw = str(row_state or "").strip()
    custom_ix = len(VALID_STATES)
    if not raw:
        return custom_ix, ""

    norm_row = normalize_state(raw)
    for i, cand in enumerate(VALID_STATES):
        if cand.strip() == raw or normalize_state(cand) == norm_row:
            return i, ""

    return custom_ix, raw


def _match_event_option(row_et: str) -> tuple[int, str]:
    raw = str(row_et or "").strip()
    custom_ix = len(VALID_EVENT_TYPES)
    if raw in VALID_EVENT_TYPES:
        return VALID_EVENT_TYPES.index(raw), ""
    return custom_ix, raw


def _next_case_seq_from_rows(rows: list) -> int:
    best = -1
    for r in rows:
        cid = str(r.get("case_id") or "").strip()
        if len(cid) >= 2 and cid[0].lower() == "c" and cid[1:].isdigit():
            best = max(best, int(cid[1:]))
    return max(0, best + 1)


def _load_draft_rows() -> list[dict]:
    path = _draft_csv_path()
    if not os.path.isfile(path):
        return []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return []
            rows: list[dict] = []
            seen_urls: set[str] = set()
            for raw in reader:
                row = {k: str(raw.get(k, "") or "") for k in EXPORT_COLS}
                url = row.get("url", "").strip()
                if url and url in seen_urls:
                    continue
                if url:
                    seen_urls.add(url)
                rows.append(row)
            return rows
    except Exception:
        return []


def _persist_draft(rows: list[dict]) -> None:
    os.makedirs(os.path.join(_REPO_ROOT, config.DATA_DIR), exist_ok=True)
    path = _draft_csv_path()
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=EXPORT_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: str(r.get(k, "") or "") for k in EXPORT_COLS})


def _remove_draft_file() -> None:
    path = _draft_csv_path()
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


def _init_session() -> None:
    if "golden_rows" not in st.session_state:
        loaded = _load_draft_rows()
        st.session_state.golden_rows = loaded
        st.session_state.case_seq = _next_case_seq_from_rows(loaded)
    elif "case_seq" not in st.session_state:
        st.session_state.case_seq = _next_case_seq_from_rows(st.session_state.golden_rows)


def _filter_df(df: pd.DataFrame, q: str) -> pd.DataFrame:
    if not q.strip():
        return df
    qq = q.casefold()
    cols = ("url", "title", "description", "state", "municipality", "group", "event_type")

    def ok(row: pd.Series) -> bool:
        return any(qq in str(row.get(c, "") or "").casefold() for c in cols)

    return df[df.apply(ok, axis=1)].reset_index(drop=True)


def _golden_url_set(rows: list) -> set[str]:
    return {
        str(r.get("url", "") or "").strip()
        for r in rows
        if str(r.get("url", "") or "").strip()
    }


def _exclude_buffered(df: pd.DataFrame, buffered_urls: set[str]) -> pd.DataFrame:
    if not buffered_urls:
        return df
    urls = df["url"].fillna("").astype(str).str.strip()
    return df[~urls.isin(buffered_urls)].reset_index(drop=True)


def _golden_to_csv(rows: list[dict[str, str]]) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=EXPORT_COLS, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow({k: str(r.get(k, "") or "") for k in EXPORT_COLS})
    return buf.getvalue().encode("utf-8")


def _buffer_toolbar_horizontal() -> None:
    cols = st.columns(2)
    with cols[0]:
        if st.session_state.golden_rows and st.button("Clear golden buffer"):
            st.session_state.golden_rows = []
            st.session_state.case_seq = 0
            _remove_draft_file()
            st.rerun()
    with cols[1]:
        if st.session_state.golden_rows:
            st.download_button(
                "Download cases CSV",
                data=_golden_to_csv(st.session_state.golden_rows),
                file_name="slm_eval_cases.csv",
                mime="text/csv",
                help=f"Columns match tests/fixtures/slm_eval/cases.csv ({EXPORT_COLS[0]}, …)",
            )


def _render_buffer_table() -> None:
    if not st.session_state.golden_rows:
        return
    with st.expander(f"Golden buffer preview ({len(st.session_state.golden_rows)} rows)", expanded=False):
        st.dataframe(
            pd.DataFrame(st.session_state.golden_rows)[EXPORT_COLS],
            use_container_width=True,
            height=min(420, 120 + 36 * len(st.session_state.golden_rows)),
        )


def main() -> None:
    _init_session()

    st.title("Label eval cases")
    st.caption(
        f"Golden buffer → `{config.DATA_DIR}/{DRAFT_BASENAME}` · export merges into "
        "`tests/fixtures/slm_eval/cases.csv`."
    )

    df = load()
    if df.empty:
        st.warning(
            "No articles in cache. Run **Refresh** on the main Heatmap page to fill `articles.csv` first."
        )
        if st.session_state.golden_rows:
            st.info(
                f"**{len(st.session_state.golden_rows)}** buffered row(s) restored from `{_draft_csv_path()}`."
            )
            _buffer_toolbar_horizontal()
            _render_buffer_table()
        return

    buffered = _golden_url_set(st.session_state.golden_rows)

    col_f, col_p = st.columns([1.2, 2.0])
    with col_f:
        q = st.text_input(
            "Filter",
            "",
            placeholder="URL, title, estado, grupo, tipo…",
            label_visibility="collapsed",
        )

    filt = _filter_df(df, q)
    filt_avail = _exclude_buffered(filt, buffered)

    st.sidebar.metric("Buffered", len(st.session_state.golden_rows))

    if filt.empty:
        st.info("No rows match the filter.")
        _buffer_toolbar_horizontal()
        _render_buffer_table()
        return

    if filt_avail.empty:
        st.sidebar.metric("Pickable", 0)
        n_buf = len(buffered & set(filt["url"].fillna("").astype(str).str.strip()))
        st.success(
            f"Every match is already buffered (**{n_buf}** in this slice). Clear buffer or widen filter."
        )
        _buffer_toolbar_horizontal()
        _render_buffer_table()
        return

    n_hidden = len(filt) - len(filt_avail)
    st.sidebar.metric("Pickable", len(filt_avail))
    st.sidebar.caption(f"Filtered **{len(filt)}** / **{len(df)}** cached total")

    labels: dict[int, str] = {}
    for j in range(len(filt_avail)):
        r = filt_avail.iloc[j]
        slug = (
            ((r.get("title") or "")[:56] + ("…" if len(str(r.get("title"))) > 56 else ""))
            .strip()
            or "(no title)"
        )
        labels[j] = f"{slug} · {(str(r.get('url', '') or '')[:42])}"

    with col_p:
        pos = st.selectbox(
            "Pick article",
            options=list(range(len(filt_avail))),
            format_func=lambda j_: labels[j_],
            label_visibility="collapsed",
        )

    row = filt_avail.iloc[pos]
    rd = row.to_dict()
    url = str(rd.get("url", "") or "")

    if n_hidden > 0:
        st.caption(f"Hiding **{n_hidden}** URL(s) already in buffer.")

    t1, t2, t3, t4 = st.columns([2.1, 1, 1, 1])
    title_txt = str(rd.get("title") or "").strip() or "(no title)"
    if len(title_txt) > 100:
        title_txt = title_txt[:99] + "…"
    with t1:
        if url and len(url) > 60:
            st.caption(f"**{title_txt}**\n[{url[:60]}…]({url})")
        elif url:
            st.caption(f"**{title_txt}**\n[{url}]({url})")
        else:
            st.caption(f"**{title_txt}**")
    with t2:
        et_s = rd.get("event_type") or "—"
        st.caption(f"SLM tipo `{et_s}`")
    with t3:
        st_s = rd.get("state") or "—"
        st.caption(f"SLM estado `{st_s}`")
    with t4:
        cnf = rd.get("confidence") or "—"
        st.caption(f"conf. **{cnf}**")

    ctx_l, form_r = st.columns([1.15, 0.85])
    with ctx_l:
        desc = str(rd.get("description") or "").strip()
        body = str(rd.get("body") or "").strip()
        with st.expander("Article excerpt & body"):
            if desc:
                st.markdown(desc[:1200] + ("…" if len(desc) > 1200 else ""))
            st.text_area(
                "body_preview",
                body[:config.ARTICLE_BODY_MAX_CHARS_SLM] + ("…" if len(body) > config.ARTICLE_BODY_MAX_CHARS_SLM else ""),
                height=200,
                disabled=True,
                label_visibility="collapsed",
            )

    st_ix, st_custom_def = _match_state_option(str(rd.get("state") or ""))
    et_ix, et_custom_def = _match_event_option(str(rd.get("event_type") or ""))

    with form_r:
        with st.form("add_golden"):
            r1a, r1b = st.columns(2)
            with r1a:
                sel_st = st.selectbox(
                    "expected_state",
                    STATE_SELECT_OPTIONS,
                    index=st_ix,
                    help="Same enum list as SLM constrained output.",
                )
                st_custom = ""
                if sel_st == "(custom)":
                    st_custom = st.text_input(
                        "custom estado",
                        value=st_custom_def,
                        label_visibility="collapsed",
                        placeholder="Custom state label",
                    ).strip()

            with r1b:
                sel_et = st.selectbox(
                    "expected_event_type",
                    EVENT_SELECT_OPTIONS,
                    index=et_ix,
                )
                et_custom = ""
                if sel_et == "(custom)":
                    et_custom = st.text_input(
                        "custom tipo",
                        value=et_custom_def,
                        label_visibility="collapsed",
                        placeholder="Custom event_type",
                    ).strip()

            r2a, r2b = st.columns(2)
            with r2a:
                exp_muni = st.text_input(
                    "municipality",
                    value=str(rd.get("municipality") or "").strip(),
                    placeholder="expected_municipality",
                    label_visibility="collapsed",
                ).strip()
            with r2b:
                exp_group = st.text_input(
                    "group",
                    value=str(rd.get("group") or "").strip(),
                    placeholder="expected_group",
                    label_visibility="collapsed",
                ).strip()

            notes = st.text_area("notes", height=72, placeholder="Optional notes")

            submitted = st.form_submit_button("Add → golden buffer", use_container_width=True)

    if submitted:
        if sel_st == "(custom)":
            exp_state_final = st_custom if st_custom else "Desconocido"
        else:
            exp_state_final = sel_st

        if sel_et == "(custom)":
            exp_et_final = et_custom if et_custom else "otro"
        else:
            exp_et_final = sel_et

        if url and any(gr.get("url") == url for gr in st.session_state.golden_rows):
            st.warning("URL already buffered; skipped.")
        else:
            cid = f"c{st.session_state.case_seq:02d}"
            st.session_state.case_seq += 1
            rec = {
                "case_id": cid,
                **{col: str(rd.get(col, "") or "") for col in config.CSV_COLUMNS},
                "expected_state": exp_state_final,
                "notes": notes.strip(),
                "expected_municipality": exp_muni,
                "expected_group": exp_group,
                "expected_event_type": exp_et_final,
            }
            st.session_state.golden_rows.append(rec)
            _persist_draft(st.session_state.golden_rows)
            st.success(f"Added **{cid}**.")
            st.rerun()

    st.divider()
    _buffer_toolbar_horizontal()
    _render_buffer_table()


main()
