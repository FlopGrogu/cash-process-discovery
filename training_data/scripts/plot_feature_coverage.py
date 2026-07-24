#!/usr/bin/env python3
"""Plot real, augmented, and synthetic log coverage in the six-axis feature space.

The visual follows GEDI Figure 4: project the feature space to two principal
components, plot every log, and draw a convex hull around each collection.
Count-like axes are log10 transformed, then standardized against the real logs,
matching the feature-space distance convention used by the GEDI pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.spatial import ConvexHull, QhullError
from sklearn.decomposition import PCA

from process_discovery_cash.data.features import (
    attrs_from_children,
    has_non_self_loop_repetition,
    local_name,
    open_xes,
)
from process_discovery_cash.generation.feature_space import (
    LOG10_FEATURES,
    TARGET_FEATURES,
    RealAnchor,
    to_target_space,
)

plt.switch_backend("Agg")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANCHOR = ROOT / "data/synthetic/gedi/anchor_features.csv"
DEFAULT_GEDI_MANIFEST = ROOT / "data/synthetic/gedi/manifest.csv"
DEFAULT_AUGMENTED_MANIFEST = ROOT / "data/augmented/manifest.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/feature-coverage"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--gedi-manifest", type=Path, default=DEFAULT_GEDI_MANIFEST)
    parser.add_argument("--augmented-manifest", type=Path, default=DEFAULT_AUGMENTED_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--inline-html",
        type=Path,
        help="Also write a theme-aware Codex inline-visualization fragment.",
    )
    parser.add_argument(
        "--refresh-augmented-cache",
        action="store_true",
        help="Recompute augmented repetition prevalence from processed Parquet files.",
    )
    return parser


def _accepted_gedi_features(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    accepted = manifest.loc[manifest["status"].eq("accepted")].copy()
    rows: list[dict[str, float | str]] = []
    for record in accepted.to_dict("records"):
        features = json.loads(record["achieved_features_json"])
        rows.append(
            {
                "log_id": record["log_id"],
                "num_traces": features["rs4pd_total_traces"],
                "avg_trace_length": features["rs4pd_trace_length_avg"],
                "num_activities": features["rs4pd_distinct_events"],
                "variant_ratio": features["ratio_unique_traces_per_trace"],
                "dfg_density": features["rs4pd_flow_density"],
                "repetition_prevalence": features["relative_number_of_traces_with_repetition"],
            }
        )
    return pd.DataFrame(rows)


def _timestamp_format(value: str) -> str:
    separator = "T" if "T" in value else " "
    fraction = ".%f" if re.search(r"\d{2}:\d{2}:\d{2}\.\d+", value) else ""
    return f"%Y-%m-%d{separator}%H:%M:%S{fraction}%z"


def _timestamp_key(
    value: object,
    event_index: int,
    timestamp_format: str | None,
) -> tuple[int, float, int]:
    if isinstance(value, str):
        try:
            parsed = datetime.strptime(value, timestamp_format or _timestamp_format(value))
            return 0, parsed.timestamp(), event_index
        except ValueError:
            pass
    return 1, 0.0, event_index


def _six_axes_from_xes(path: Path) -> dict[str, float]:
    """Stream the canonical six axes without materializing the whole event log."""
    n_traces = 0
    n_events = 0
    repetitions = 0
    timestamp_format: str | None = None
    activities: set[str] = set()
    variants: set[tuple[str, ...]] = set()
    directly_follows: set[tuple[str, str]] = set()
    with open_xes(path) as handle:
        for _, trace in ET.iterparse(handle, events=("end",)):
            if local_name(trace.tag) != "trace":
                continue
            n_traces += 1
            trace_events: list[tuple[tuple[int, float, int], str | None]] = []
            event_index = 0
            for child in list(trace):
                if local_name(child.tag) != "event":
                    continue
                event_index += 1
                n_events += 1
                attrs = attrs_from_children(child)
                activity = attrs.get("concept:name")
                activity = str(activity) if activity is not None else None
                timestamp = attrs.get("time:timestamp")
                if timestamp_format is None and isinstance(timestamp, str):
                    timestamp_format = _timestamp_format(timestamp)
                trace_events.append(
                    (_timestamp_key(timestamp, event_index, timestamp_format), activity)
                )
            trace_events.sort(key=lambda item: item[0])
            ordered = tuple(activity for _, activity in trace_events if activity is not None)
            activities.update(ordered)
            variants.add(ordered)
            directly_follows.update(zip(ordered, ordered[1:], strict=False))
            repetitions += has_non_self_loop_repetition(ordered)
            trace.clear()
    n_activities = len(activities)
    return {
        "num_traces": float(n_traces),
        "avg_trace_length": n_events / n_traces,
        "num_activities": float(n_activities),
        "variant_ratio": len(variants) / n_traces,
        "dfg_density": len(directly_follows) / (n_activities * n_activities),
        "repetition_prevalence": repetitions / n_traces,
    }


def _augmented_features(
    manifest_path: Path,
    cache_path: Path,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    accepted = manifest.loc[manifest["status"].eq("accepted")].copy()
    rows: list[dict[str, float | str]] = []
    total = len(accepted)
    for index, record in enumerate(accepted.to_dict("records"), start=1):
        log_id = str(record["child_log_id"])
        xes_path = ROOT / str(record["output_path"])
        print(f"[{index:02d}/{total}] {log_id}", flush=True)
        rows.append({"log_id": log_id, **_six_axes_from_xes(xes_path)})
    result = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(cache_path, index=False)
    return result


def _load_augmented_features(
    manifest_path: Path,
    cache_path: Path,
    *,
    refresh: bool,
) -> pd.DataFrame:
    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path)
    return _augmented_features(manifest_path, cache_path)


def _valid_axes(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    cleaned = frame[["log_id", *TARGET_FEATURES]].copy()
    values = cleaned[TARGET_FEATURES].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(values.to_numpy()).all(axis=1)
    if not valid.all():
        print(f"Dropped {(~valid).sum()} invalid {label} feature rows", flush=True)
    cleaned[TARGET_FEATURES] = values
    return cleaned.loc[valid].reset_index(drop=True)


def _project(
    real: pd.DataFrame,
    augmented: pd.DataFrame,
    gedi: pd.DataFrame,
) -> tuple[pd.DataFrame, PCA]:
    anchor = RealAnchor.from_features(real)
    frames = []
    for group, frame in (
        ("Real", real),
        ("Augmented", augmented),
        ("Synthetic", gedi),
    ):
        transformed = to_target_space(frame)
        standardized = anchor.standardize(transformed)
        block = pd.DataFrame(standardized, columns=TARGET_FEATURES)
        block.insert(0, "log_id", frame["log_id"].to_numpy())
        block.insert(1, "group", group)
        frames.append(block)
    combined = pd.concat(frames, ignore_index=True)
    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(combined[TARGET_FEATURES])
    combined["pc1"] = coordinates[:, 0]
    combined["pc2"] = coordinates[:, 1]
    return combined, pca


def _hull_vertices(points: np.ndarray) -> np.ndarray | None:
    if len(points) < 3:
        return None
    try:
        hull = ConvexHull(points)
    except QhullError:
        return None
    vertices = points[hull.vertices]
    return np.vstack([vertices, vertices[0]])


def _plot(projected: pd.DataFrame, pca: PCA, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
        }
    )
    styles = {
        "Real": {"color": "#4477AA", "marker": "o", "size": 34, "alpha": 0.92},
        "Augmented": {"color": "#228B7E", "marker": "^", "size": 30, "alpha": 0.72},
        "Synthetic": {"color": "#EE7733", "marker": "x", "size": 28, "alpha": 0.70},
    }
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    fig.subplots_adjust(left=0.11, right=0.98, top=0.98, bottom=0.13)
    for group in ("Synthetic", "Augmented", "Real"):
        subset = projected.loc[projected["group"].eq(group)]
        points = subset[["pc1", "pc2"]].to_numpy()
        style = styles[group]
        hull = _hull_vertices(points)
        if hull is not None:
            ax.fill(
                hull[:, 0],
                hull[:, 1],
                color=style["color"],
                alpha=0.09,
                zorder=1,
            )
            ax.plot(
                hull[:, 0],
                hull[:, 1],
                color=style["color"],
                linewidth=1.35,
                zorder=2,
            )
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=style["size"],
            c=style["color"],
            marker=style["marker"],
            alpha=style["alpha"],
            linewidths=1.1,
            edgecolors="white" if group != "Synthetic" else None,
            zorder=3,
        )

    explained = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}% explained variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}% explained variance)")
    ax.grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    legend_handles = []
    for group in ("Real", "Augmented", "Synthetic"):
        style = styles[group]
        count = int(projected["group"].eq(group).sum())
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=style["marker"],
                color="none" if group != "Synthetic" else style["color"],
                linestyle="none",
                markerfacecolor=style["color"] if group != "Synthetic" else "none",
                markeredgecolor=style["color"],
                markersize=7,
                label=f"{group} (n={count})",
            )
        )
    legend_handles.append(Patch(facecolor="#777777", alpha=0.10, label="Convex hull"))
    ax.legend(handles=legend_handles, loc="best", frameon=True, framealpha=0.92)

    for suffix in ("png", "svg", "pdf"):
        path = output_dir / f"real-augmented-synthetic-feature-coverage.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


def _write_inline_html(projected: pd.DataFrame, pca: PCA, path: Path) -> None:
    width, height = 736, 500
    left, right, top, bottom = 72, 22, 22, 66
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_min, x_max = projected["pc1"].min(), projected["pc1"].max()
    y_min, y_max = projected["pc2"].min(), projected["pc2"].max()
    x_pad = max((x_max - x_min) * 0.04, 0.1)
    y_pad = max((y_max - y_min) * 0.06, 0.1)
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad

    def px(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def py(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    lines = [
        '<div id="feature-coverage-pca-inline">',
        '  <svg class="coverage-chart" viewBox="0 0 736 500" role="img" '
        'aria-labelledby="coverage-chart-title coverage-chart-desc">',
        '    <title id="coverage-chart-title">Feature-space coverage of real, augmented, '
        "and synthetic logs</title>",
        '    <desc id="coverage-chart-desc">PCA scatter plot with convex hulls. Real logs '
        "are circles, augmented logs are triangles, and synthetic logs are crosses.</desc>",
    ]
    for value in np.linspace(x_min, x_max, 6):
        x = px(float(value))
        lines.extend(
            [
                f'    <line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" '
                f'y2="{top + plot_height}"/>',
                f'    <text class="tick" x="{x:.2f}" y="{top + plot_height + 24}" '
                f'text-anchor="middle">{value:.1f}</text>',
            ]
        )
    for value in np.linspace(y_min, y_max, 6):
        y = py(float(value))
        lines.extend(
            [
                f'    <line class="grid" x1="{left}" y1="{y:.2f}" '
                f'x2="{left + plot_width}" y2="{y:.2f}"/>',
                f'    <text class="tick" x="{left - 12}" y="{y + 4:.2f}" '
                f'text-anchor="end">{value:.1f}</text>',
            ]
        )
    lines.extend(
        [
            f'    <line class="axis" x1="{left}" y1="{top + plot_height}" '
            f'x2="{left + plot_width}" y2="{top + plot_height}"/>',
            f'    <line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>',
        ]
    )
    for group in ("Synthetic", "Augmented", "Real"):
        slug = group.lower()
        points = projected.loc[projected["group"].eq(group), ["pc1", "pc2"]].to_numpy()
        hull = _hull_vertices(points)
        if hull is not None:
            coords = " ".join(f"{px(x):.2f},{py(y):.2f}" for x, y in hull)
            lines.append(f'    <polygon class="hull {slug}" points="{coords}"/>')
    for record in projected.to_dict("records"):
        group = str(record["group"])
        slug = group.lower()
        x, y = px(float(record["pc1"])), py(float(record["pc2"]))
        label = f"{record['log_id']}: PC1 {record['pc1']:.2f}, PC2 {record['pc2']:.2f}"
        if group == "Real":
            mark = f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5"/>'
        elif group == "Augmented":
            mark = (
                f'<path d="M {x:.2f} {y - 5:.2f} L {x + 5:.2f} {y + 4:.2f} '
                f'L {x - 5:.2f} {y + 4:.2f} Z"/>'
            )
        else:
            mark = (
                f'<path d="M {x - 4:.2f} {y - 4:.2f} L {x + 4:.2f} {y + 4:.2f} '
                f'M {x + 4:.2f} {y - 4:.2f} L {x - 4:.2f} {y + 4:.2f}"/>'
            )
        lines.append(
            f'    <g class="point {slug}" aria-label="{label}">{mark}<title>{label}</title></g>'
        )
    explained = pca.explained_variance_ratio_ * 100
    lines.extend(
        [
            f'    <text class="axis-label" x="{left + plot_width / 2:.2f}" y="490" '
            f'text-anchor="middle">PC1 ({explained[0]:.1f}% explained variance)</text>',
            f'    <text class="axis-label" x="18" y="{top + plot_height / 2:.2f}" '
            f'text-anchor="middle" transform="rotate(-90 18 {top + plot_height / 2:.2f})">'
            f"PC2 ({explained[1]:.1f}% explained variance)</text>",
            '    <g class="legend" transform="translate(92 40)">',
            '      <circle class="legend-mark real" cx="0" cy="0" r="5"/>',
            '      <text x="14" y="4">Real (n=21)</text>',
            '      <path class="legend-mark augmented" d="M 0 18 L 6 29 L -6 29 Z"/>',
            '      <text x="14" y="29">Augmented (n=77)</text>',
            '      <path class="legend-mark synthetic" d="M -5 43 L 5 53 M 5 43 L -5 53"/>',
            f'      <text x="14" y="52">Synthetic '
            f'(n={int(projected["group"].eq("Synthetic").sum())})</text>',
            "    </g>",
            "  </svg>",
            "</div>",
            "<style>",
            "#feature-coverage-pca-inline { width: 100%; color: var(--foreground); }",
            "#feature-coverage-pca-inline .coverage-chart { display: block; width: 100%; "
            "height: auto; overflow: visible; }",
            "#feature-coverage-pca-inline text { fill: var(--foreground); "
            "font: 400 var(--font-size-base) system-ui, sans-serif; }",
            "#feature-coverage-pca-inline .grid { stroke: var(--border); stroke-width: 1; }",
            "#feature-coverage-pca-inline .axis { stroke: var(--foreground); stroke-width: 1.5; }",
            "#feature-coverage-pca-inline .tick { fill: var(--muted-foreground); }",
            "#feature-coverage-pca-inline .axis-label { font-weight: 500; }",
            "#feature-coverage-pca-inline .real { --series: var(--viz-series-1); }",
            "#feature-coverage-pca-inline .augmented { --series: var(--viz-series-2); }",
            "#feature-coverage-pca-inline .synthetic { --series: var(--viz-series-3); }",
            "#feature-coverage-pca-inline .hull { fill: color-mix(in srgb, var(--series) 10%, "
            "transparent); stroke: var(--series); stroke-width: 2; }",
            "#feature-coverage-pca-inline .point { fill: var(--series); stroke: var(--background); "
            "stroke-width: 1.5; opacity: .82; }",
            "#feature-coverage-pca-inline .point.synthetic { fill: none; stroke: var(--series); "
            "stroke-width: 2; }",
            "#feature-coverage-pca-inline .legend-mark { fill: var(--series); "
            "stroke: var(--series); stroke-width: 2; }",
            "#feature-coverage-pca-inline .legend-mark.synthetic { fill: none; }",
            "</style>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "augmented-feature-axes.csv"

    real = pd.read_csv(args.anchor).rename(columns={"log_id": "log_id"})
    gedi = _accepted_gedi_features(args.gedi_manifest)
    augmented = _load_augmented_features(
        args.augmented_manifest,
        cache_path,
        refresh=args.refresh_augmented_cache,
    )
    real = _valid_axes(real, "real")
    augmented = _valid_axes(augmented, "augmented")
    gedi = _valid_axes(gedi, "synthetic")

    projected, pca = _project(real, augmented, gedi)
    projected.to_csv(args.output_dir / "feature-coverage-pca.csv", index=False)
    summary = {
        "axes": TARGET_FEATURES,
        "log10_axes": sorted(LOG10_FEATURES),
        "standardization": "mean and population standard deviation fitted on real logs",
        "pca_fit": (
            "all real, augmented, and accepted synthetic logs after real-log standardization"
        ),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "counts": projected.groupby("group").size().to_dict(),
        "components": {
            f"PC{index + 1}": dict(zip(TARGET_FEATURES, component.tolist(), strict=True))
            for index, component in enumerate(pca.components_)
        },
    }
    (args.output_dir / "feature-coverage-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot(projected, pca, args.output_dir)
    if args.inline_html:
        _write_inline_html(projected, pca, args.inline_html)
    print(f"Wrote feature-coverage outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
