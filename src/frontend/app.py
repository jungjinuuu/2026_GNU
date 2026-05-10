import argparse
import html
import json
import math
import os
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from models.predict import DEFAULT_ARTIFACT_DIR, predict


GU_CODES = {
    "강남구": "11680",
    "강동구": "11740",
    "강북구": "11305",
    "강서구": "11500",
    "관악구": "11620",
    "광진구": "11215",
    "구로구": "11530",
    "금천구": "11545",
    "노원구": "11350",
    "도봉구": "11320",
    "동대문구": "11230",
    "동작구": "11590",
    "마포구": "11440",
    "서대문구": "11410",
    "서초구": "11650",
    "성동구": "11200",
    "성북구": "11290",
    "송파구": "11710",
    "양천구": "11470",
    "영등포구": "11560",
    "용산구": "11170",
    "은평구": "11380",
    "종로구": "11110",
    "중구": "11140",
    "중랑구": "11260",
}

PROPERTY_TYPES = {
    "아파트": "apt",
    "오피스텔": "officetel",
    "연립/다세대": "rh",
    "단독/다가구": "detached",
    "도시형생활주택": "urban",
    "복합/기타": "other",
}

PROPERTY_TYPE_LABELS = {
    "apt": "아파트",
    "officetel": "오피스텔",
    "rh": "연립/다세대",
    "detached": "단독/다가구",
    "urban": "도시형생활주택",
    "other": "복합/기타",
}

RISK_LABELS = {
    "safe": "안전 매물",
    "fair": "적정 매물",
    "caution": "주의 매물",
    "high_risk": "위험 매물",
}

RISK_CLASS = {
    "safe": "safe",
    "fair": "fair",
    "caution": "caution",
    "high_risk": "danger",
}


STYLE = """
:root {
  color-scheme: light;
  --ink: #17202a;
  --muted: #667085;
  --line: #d9dee7;
  --surface: #ffffff;
  --bg: #f4f6f8;
  --brand: #155eef;
  --safe: #0f8a5f;
  --fair: #2d6cdf;
  --caution: #b7791f;
  --danger: #c2410c;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--ink);
  background: var(--bg);
}
header {
  padding: 28px 32px 18px;
  background: #102033;
  color: white;
}
header h1 {
  margin: 0;
  font-size: 28px;
  line-height: 1.2;
  font-weight: 760;
}
header p {
  margin: 8px 0 0;
  color: #cbd5e1;
  font-size: 14px;
}
main {
  max-width: 1180px;
  margin: 0 auto;
  padding: 28px 20px 44px;
}
.layout {
  display: grid;
  grid-template-columns: minmax(340px, 0.9fr) minmax(420px, 1.1fr);
  gap: 18px;
  align-items: start;
}
.panel {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 8px 24px rgba(16, 32, 51, 0.06);
}
.panel h2 {
  margin: 0 0 16px;
  font-size: 17px;
}
.form-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
}
.field.full { grid-column: 1 / -1; }
label {
  display: block;
  margin-bottom: 6px;
  color: #344054;
  font-size: 13px;
  font-weight: 650;
}
input, select {
  width: 100%;
  height: 42px;
  padding: 0 11px;
  border: 1px solid #c8d0dc;
  border-radius: 7px;
  background: #fff;
  color: var(--ink);
  font-size: 14px;
}
input:focus, select:focus {
  outline: 2px solid rgba(21, 94, 239, 0.18);
  border-color: var(--brand);
}
button {
  width: 100%;
  height: 46px;
  margin-top: 18px;
  border: 0;
  border-radius: 7px;
  background: var(--brand);
  color: #fff;
  font-weight: 750;
  font-size: 15px;
  cursor: pointer;
}
button:hover { background: #0f4bd6; }
.result-top {
  display: grid;
  grid-template-columns: 1fr 150px;
  gap: 14px;
  align-items: stretch;
}
.hero-number {
  border-bottom: 1px solid var(--line);
  padding-bottom: 16px;
}
.hero-number span {
  display: block;
  color: var(--muted);
  font-size: 13px;
  margin-bottom: 5px;
}
.hero-number strong {
  display: block;
  font-size: 34px;
  line-height: 1.1;
  letter-spacing: 0;
}
.badge {
  display: grid;
  place-items: center;
  border-radius: 8px;
  color: white;
  text-align: center;
  padding: 12px;
  min-height: 108px;
}
.badge.safe { background: var(--safe); }
.badge.fair { background: var(--fair); }
.badge.caution { background: var(--caution); }
.badge.danger { background: var(--danger); }
.badge strong { display: block; font-size: 25px; }
.badge span { display: block; margin-top: 6px; font-size: 13px; }
.metrics {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 10px;
  margin-top: 16px;
}
.metric {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 13px;
  min-height: 84px;
}
.metric span {
  display: block;
  color: var(--muted);
  font-size: 12px;
  margin-bottom: 7px;
}
.metric strong {
  display: block;
  font-size: 18px;
  line-height: 1.25;
}
.range {
  margin-top: 18px;
  padding: 14px;
  border-radius: 8px;
  background: #f8fafc;
  border: 1px solid var(--line);
}
.range-line {
  position: relative;
  height: 10px;
  border-radius: 999px;
  background: linear-gradient(90deg, #4f8df7, #27ae7e, #f59e0b);
  margin: 16px 0 10px;
}
.range-values {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  color: #475467;
  font-size: 13px;
}
.explain {
  margin-top: 16px;
  color: var(--muted);
  font-size: 13px;
  line-height: 1.55;
}
.error {
  border: 1px solid #fecaca;
  background: #fff1f2;
  color: #9f1239;
  padding: 14px;
  border-radius: 8px;
  font-size: 14px;
}
.empty {
  color: var(--muted);
  line-height: 1.6;
  font-size: 14px;
}

.score-value { font-size: 20px; }
.score-value small { color: var(--muted); font-size: 12px; font-weight: 650; margin-left: 4px; }
.cluster-panel { margin-top: 18px; }
.cluster-plot {
  margin-top: 18px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
  overflow: hidden;
}
.cluster-plot-head {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: baseline;
  padding: 13px 14px 0;
}
.cluster-plot-head strong { font-size: 15px; }
.cluster-plot-head span { color: var(--muted); font-size: 12px; }
.cluster-plot svg {
  display: block;
  width: 100%;
  height: auto;
}
.axis-label {
  fill: #667085;
  font-size: 11px;
}
.current-dot-label {
  fill: #17202a;
  font-size: 12px;
  font-weight: 750;
}
.cluster-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 10px;
  margin-top: 12px;
}
.cluster-card {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 11px;
  background: #fff;
  min-height: 118px;
}
.cluster-card.active { border-color: var(--brand); box-shadow: 0 0 0 2px rgba(21, 94, 239, 0.12); }
.cluster-card strong { display: block; font-size: 15px; margin-bottom: 6px; }
.cluster-card span { display: block; color: var(--muted); font-size: 12px; line-height: 1.45; }

.similar-summary {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 10px;
  margin-top: 12px;
}
.similar-item {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 13px;
  background: #f8fafc;
}
.similar-item span { display: block; color: var(--muted); font-size: 12px; margin-bottom: 7px; }
.similar-item strong { display: block; font-size: 17px; line-height: 1.25; }
.similar-note {
  margin-top: 10px;
  padding: 11px 12px;
  border-radius: 8px;
  background: #f8fafc;
  border: 1px solid var(--line);
  color: var(--muted);
  font-size: 12px;
  line-height: 1.5;
}

.scope-card {
  margin-top: 12px;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 13px;
  background: #f8fafc;
}
.scope-card.warning {
  border-color: #f59e0b;
  background: #fffbeb;
}
.scope-card span { display: block; color: var(--muted); font-size: 12px; margin-bottom: 6px; }
.scope-card strong { display: block; font-size: 17px; margin-bottom: 6px; }
.scope-card p { margin: 0; color: var(--muted); font-size: 12px; line-height: 1.5; }

.cluster-detail { margin-top: 12px; }
.source-note {
  color: var(--muted);
  font-size: 12px;
  margin-top: 8px;
}

.radio-group {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 8px;
}
.radio-card input { position: absolute; opacity: 0; pointer-events: none; }
.radio-card span {
  display: grid;
  place-items: center;
  min-height: 42px;
  border: 1px solid #c8d0dc;
  border-radius: 7px;
  background: #fff;
  color: #344054;
  font-size: 13px;
  font-weight: 700;
  cursor: pointer;
}
.radio-card input:checked + span {
  border-color: var(--brand);
  background: rgba(21, 94, 239, 0.08);
  color: var(--brand);
}

@media (max-width: 860px) {
  .layout { grid-template-columns: 1fr; }
  .result-top { grid-template-columns: 1fr; }
  .metrics { grid-template-columns: 1fr; }
  .cluster-grid { grid-template-columns: 1fr; }
  .similar-summary { grid-template-columns: 1fr; }
  header { padding: 22px 20px 14px; }
  header h1 { font-size: 23px; }
}
"""


def won(value) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    if not pd.notna(value):
        return "-"
    eok = value / 100_000_000
    if abs(eok) >= 1:
        return f"{eok:,.2f}억"
    man = value / 10_000
    return f"{man:,.0f}만"


def pct(value) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    if not pd.notna(value):
        return "-"
    return f"{value:.1f}%"


def score_text(value) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    if not pd.notna(value):
        return "-"
    return f"{value:.0f}점"


def safe_text(value) -> str:
    if value is None or (isinstance(value, float) and not pd.notna(value)):
        return "-"
    return html.escape(str(value))


def load_cluster_artifacts(artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> dict:
    artifacts_path = artifact_dir / "artifacts.joblib"
    if not artifacts_path.exists():
        return {}
    try:
        artifacts = joblib.load(artifacts_path)
    except Exception:
        return {}
    return {
        "profile": artifacts.get("cluster_profile"),
        "points": artifacts.get("unsupervised", {}).get("cluster_plot_points", []),
    }


def summarize_dict(values: dict) -> str:
    if not values:
        return "-"
    parts = []
    for key, value in list(values.items())[:3]:
        label = PROPERTY_TYPE_LABELS.get(str(key), str(key))
        parts.append(f"{label} {value}")
    return ", ".join(parts)


def selected(current, value) -> str:
    return " selected" if str(current) == str(value) else ""


def form_value(values: dict, key: str, default: str) -> str:
    return html.escape(str(values.get(key, default)))


def parse_float(values: dict, key: str, default: float | None = None) -> float | None:
    raw = str(values.get(key, "")).replace(",", "").strip()
    if raw == "":
        return default
    return float(raw)


def build_payload(values: dict) -> dict:
    gu_name = str(values.get("gu_name", "강남구"))
    property_label = str(values.get("property_label", "아파트"))
    housing_type = property_label
    return {
        "gu_name": gu_name,
        "gu_code": GU_CODES.get(gu_name, ""),
        "ym": str(values.get("ym", "2025-01")),
        "property_type": PROPERTY_TYPES.get(property_label, property_label),
        "주택유형": housing_type,
        "유형": housing_type,
        "area_m2_clean": parse_float(values, "area_m2_clean", 59.0),
        "room_count_clean": parse_float(values, "room_count_clean", 3),
        "household_size_clean": 2,
        "deposit_won": parse_float(values, "deposit_won", None),
    }


def render_form(values: dict) -> str:
    gu = values.get("gu_name", "강남구")
    property_label = values.get("property_label", "아파트")
    gu_options = "".join(f'<option value="{name}"{selected(gu, name)}>{name}</option>' for name in GU_CODES)
    property_options = "".join(
        f'<label class="radio-card"><input type="radio" name="property_label" value="{name}"'
        f'{" checked" if str(property_label) == str(name) else ""}><span>{name}</span></label>'
        for name in PROPERTY_TYPES
    )
    return f"""
    <section class="panel">
      <h2>매물 정보 입력</h2>
      <form method="post">
        <div class="form-grid">
          <div>
            <label>지역</label>
            <select name="gu_name">{gu_options}</select>
          </div>
          <div>
            <label>계약 월</label>
            <input name="ym" type="month" value="{form_value(values, 'ym', '2025-01')}">
          </div>
          <div class="full">
            <label>주택유형</label>
            <div class="radio-group">{property_options}</div>
          </div>
          <div>
            <label>전용 면적 m²</label>
            <input name="area_m2_clean" inputmode="decimal" value="{form_value(values, 'area_m2_clean', '59')}">
          </div>
          <div>
            <label>방 개수</label>
            <input name="room_count_clean" inputmode="decimal" value="{form_value(values, 'room_count_clean', '3')}">
          </div>
          <div class="full">
            <label>제시 보증금 원</label>
            <input name="deposit_won" inputmode="numeric" value="{form_value(values, 'deposit_won', '330000000')}">
          </div>
        </div>
        <div class="source-note">주변 전세 중위가는 지역/유형/계약월 기반 내부 데이터에서 자동 추정합니다.</div>
        <button type="submit">예측하기</button>
      </form>
    </section>
    """


def _scale(value: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    if hi == lo:
        return (out_lo + out_hi) / 2
    return out_lo + (value - lo) / (hi - lo) * (out_hi - out_lo)


def as_float(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not pd.notna(value):
        return None
    return value


def render_cluster_plot(row: pd.Series) -> str:
    data = load_cluster_artifacts()
    points = data.get("points") or []
    current_x = as_float(row.get("cluster_x"))
    current_y = as_float(row.get("cluster_y"))
    active_cluster = str(row.get("cluster_id", ""))
    if current_x is None or current_y is None:
        return ""

    def log_coord(value: float) -> float:
        return math.log1p(max(value, 0.0))

    plot_points = []
    xs = [log_coord(current_x)]
    ys = [log_coord(current_y)]
    for point in points:
        x = as_float(point.get("x"))
        y = as_float(point.get("y"))
        if x is None or y is None:
            continue
        cluster_id = str(point.get("cluster_id", ""))
        plot_x = log_coord(x)
        plot_y = log_coord(y)
        plot_points.append((plot_x, plot_y, cluster_id))
        xs.append(plot_x)
        ys.append(plot_y)

    width = 620
    height = 340
    left = 48
    right = 22
    top = 24
    bottom = 44
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    pad_x = (max_x - min_x) * 0.06 or 1
    pad_y = (max_y - min_y) * 0.08 or 1
    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y

    colors = {
        "1": "#155eef",
        "2": "#0f8a5f",
        "3": "#b7791f",
        "4": "#c2410c",
        "5": "#7c3aed",
        "6": "#0891b2",
    }

    circles = []
    for x, y, cluster_id in plot_points:
        cx = _scale(x, min_x, max_x, left, width - right)
        cy = _scale(y, min_y, max_y, height - bottom, top)
        color = colors.get(cluster_id, "#667085")
        opacity = "0.42" if cluster_id == active_cluster else "0.18"
        radius = "3.1" if cluster_id == active_cluster else "2.4"
        circles.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius}" fill="{color}" opacity="{opacity}"></circle>'
        )

    current_plot_x = log_coord(current_x)
    current_plot_y = log_coord(current_y)
    current_cx = _scale(current_plot_x, min_x, max_x, left, width - right)
    current_cy = _scale(current_plot_y, min_y, max_y, height - bottom, top)
    active_color = colors.get(active_cluster, "#155eef")
    x_mid = (left + width - right) / 2
    y_mid = (top + height - bottom) / 2
    y_label = f"{current_y:.2f}"

    return f"""
      <div class="cluster-plot">
        <div class="cluster-plot-head">
          <strong>Cluster map</strong>
          <span>active cluster {html.escape(active_cluster or "-")}</span>
        </div>
        <svg viewBox="0 0 {width} {height}" role="img" aria-label="cluster scatter plot">
          <rect x="{left}" y="{top}" width="{width - left - right}" height="{height - top - bottom}" fill="#f8fafc"></rect>
          <line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#c8d0dc"></line>
          <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#c8d0dc"></line>
          <line x1="{left}" y1="{y_mid:.1f}" x2="{width - right}" y2="{y_mid:.1f}" stroke="#e4e7ec"></line>
          <line x1="{x_mid:.1f}" y1="{top}" x2="{x_mid:.1f}" y2="{height - bottom}" stroke="#e4e7ec"></line>
          {''.join(circles)}
          <circle cx="{current_cx:.1f}" cy="{current_cy:.1f}" r="8" fill="#ffffff" stroke="{active_color}" stroke-width="4"></circle>
          <circle cx="{current_cx:.1f}" cy="{current_cy:.1f}" r="3.2" fill="{active_color}"></circle>
          <text class="current-dot-label" x="{min(current_cx + 12, width - 118):.1f}" y="{max(current_cy - 10, top + 16):.1f}">current</text>
          <text class="axis-label" x="{left}" y="{height - 16}">log area: {current_x:.1f} m2</text>
          <text class="axis-label" x="{width - 198}" y="{height - 16}">log rent median: {html.escape(y_label)} eok</text>
        </svg>
      </div>
    """


def render_cluster_profile(active_cluster: str | None, row: pd.Series) -> str:
    data = load_cluster_artifacts()
    profile = data.get("profile")
    if not profile:
        return ""
    profiles = profile.get("profiles", [])
    active = str(active_cluster) if active_cluster is not None else None
    item = next((p for p in profiles if str(p.get("cluster_id")) == active), None)
    if item is None and profiles:
        item = profiles[0]
    if not item:
        return ""

    rows = int(item.get("rows", 0))
    avg_area = float(item.get("avg_area_m2") or 0)
    median_rent = item.get("median_rent_deposit_won")
    current_area = float(row.get("area_m2_clean") or 0)
    current_rent = row.get("rent_deposit_median")
    current_property = PROPERTY_TYPE_LABELS.get(str(row.get("property_type")), str(row.get("property_type", "-")))
    top_gu = html.escape(summarize_dict(item.get("top_gu_name", {})))
    top_property = html.escape(summarize_dict(item.get("top_property_type", {})))

    return f"""
      <div class="cluster-panel">
        <h2>유사 매물 기준</h2>
        <div class="similar-summary">
          <div class="similar-item"><span>현재 매물</span><strong>{safe_text(row.get('gu_name'))} · {html.escape(current_property)}</strong></div>
          <div class="similar-item"><span>현재 면적 / 유사 평균</span><strong>{current_area:.1f}m² / {avg_area:.1f}m²</strong></div>
          <div class="similar-item"><span>현재 주변 전세 / 유사 중위</span><strong>{won(current_rent)} / {won(median_rent)}</strong></div>
          <div class="similar-item"><span>유사 사례 수</span><strong>{rows:,}건</strong></div>
          <div class="similar-item"><span>주요 지역</span><strong>{top_gu}</strong></div>
          <div class="similar-item"><span>주요 주택유형</span><strong>{top_property}</strong></div>
        </div>
        <div class="similar-note">유사 기준은 주택유형, 면적, 방 개수, 주변 전세 수준이 가까운 학습 사례를 묶은 값입니다. 그림 대신 실제 판단에 쓰이는 기준값만 표시합니다.</div>
      </div>
    """


def render_result(result: pd.DataFrame | None, error: str | None) -> str:
    if error:
        return f'<section class="panel"><h2>예측 결과</h2><div class="error">{html.escape(error)}</div></section>'
    if result is None:
        return """
        <section class="panel">
          <h2>예측 결과</h2>
          <div class="empty">왼쪽에 매물 정보를 입력하면 적정 전세금, 95% 예측구간, 위험 판단 지수가 표시됩니다.</div>
        </section>
        """
    row = result.iloc[0]
    label = str(row.get("risk_label", "safe"))
    label_ko = RISK_LABELS.get(label, label)
    klass = RISK_CLASS.get(label, "safe")
    gap = row.get("deposit_gap_won")
    gap_text = won(gap)
    scope_status = str(row.get("model_scope_status", "in_domain"))
    scope_class = "warning" if scope_status != "in_domain" else ""
    return f"""
    <section class="panel">
      <h2>예측 결과</h2>
      <div class="result-top">
        <div class="hero-number">
          <span>예측 적정 전세금</span>
          <strong>{won(row.get('predicted_fair_deposit_won'))}</strong>
          <div class="source-note">주변 전세 중위가: {won(row.get('rent_deposit_median'))} · 추정 기준: {safe_text(row.get('rent_deposit_median_source'))}</div>
        </div>
        <div class="badge {klass}">
          <div>
            <strong>{pct(row.get('risk_probability_percent'))}</strong>
            <span>{html.escape(label_ko)}</span>
          </div>
        </div>
      </div>
      <div class="metrics">
        <div class="metric"><span>95% 하한</span><strong>{won(row.get('fair_deposit_lower_95_won'))}</strong></div>
        <div class="metric"><span>95% 상한</span><strong>{won(row.get('fair_deposit_upper_95_won'))}</strong></div>
        <div class="metric"><span>가격 이상 점수</span><strong class="score-value">{score_text(row.get('overprice_score'))}<small>{safe_text(row.get('price_risk_grade'))}</small></strong><div class="source-note">고가 {score_text(row.get('high_price_score'))} · 저가 {score_text(row.get('low_price_score'))}</div></div>
        <div class="metric"><span>제시가 차이</span><strong>{gap_text}</strong></div>
      </div>
      <div class="scope-card {scope_class}">
        <span>모델 적용범위</span>
        <strong>{safe_text(row.get('model_scope_label'))}</strong>
        <p>{safe_text(row.get('model_scope_message'))}</p>
      </div>
      <div class="range">
        <strong>유의수준 5% 기준 예측구간</strong>
        <div class="range-line"></div>
        <div class="range-values">
          <span>{won(row.get('fair_deposit_lower_95_won'))}</span>
          <span>{won(row.get('predicted_fair_deposit_won'))}</span>
          <span>{won(row.get('fair_deposit_upper_95_won'))}</span>
        </div>
      </div>
      <div class="explain">
        위험 판단은 제시 보증금이 LH 기준 적정가와 95% 예측 상한 대비 높은 정도를 중심으로 판단합니다. 예측구간 기준: {safe_text(row.get('interval_source'))}.
      </div>
      {render_cluster_plot(row)}
      {render_cluster_profile(row.get('cluster_id'), row)}
    </section>
    """


def page(values: dict | None = None, result: pd.DataFrame | None = None, error: str | None = None) -> bytes:
    values = values or {}
    body = f"""
    <!doctype html>
    <html lang="ko">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>전세 적정가 및 위험 판단</title>
        <style>{STYLE}</style>
      </head>
      <body>
        <header>
          <h1>전세 적정가 및 위험 매물 판단</h1>
          <p>최종 가격 예측 모델 기반 위험 판단</p>
        </header>
        <main>
          <div class="layout">
            {render_form(values)}
            {render_result(result, error)}
          </div>
        </main>
      </body>
    </html>
    """
    return body.encode("utf-8")


class AppHandler(BaseHTTPRequestHandler):
    artifact_dir = DEFAULT_ARTIFACT_DIR

    def send_html(self, content: bytes, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:
        self.send_html(page())

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        values = {key: vals[0] for key, vals in parse_qs(raw).items()}
        try:
            payload = build_payload(values)
            result = predict(pd.DataFrame([payload]), self.artifact_dir)
            self.send_html(page(values, result=result))
        except Exception as exc:
            self.send_html(page(values, error=str(exc)), status=200)

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Local frontend for fair jeonse deposit and risk prediction.")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8501")))
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    AppHandler.artifact_dir = Path(args.artifact_dir)
    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    url = f"http://{args.host}:{args.port}"
    browser_host = "localhost" if args.host in {"0.0.0.0", "::"} else args.host
    browser_url = f"http://{browser_host}:{args.port}"
    print(f"serving {url}")
    if args.host == "0.0.0.0":
        print("public mode: listening on all network interfaces")
    print(f"artifact_dir={AppHandler.artifact_dir}")
    if not args.no_browser and not os.environ.get("RENDER"):
        try:
            webbrowser.open(browser_url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
