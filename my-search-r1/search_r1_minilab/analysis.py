"""Checkpoint metric loading and plotting helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


@dataclass(frozen=True)
class CheckpointSpec:
    """One checkpoint label and JSONL filename pair."""

    label: str
    filename: str


DEFAULT_CHECKPOINTS = (
    CheckpointSpec("Base", "eval_results.jsonl"),
    CheckpointSpec("Step 20", "eval_results_rl_step_20.jsonl"),
    CheckpointSpec("Step 50", "eval_results_rl_step_50.jsonl"),
    CheckpointSpec("Step 100", "eval_results_rl_step_100.jsonl"),
    CheckpointSpec("Step 150", "eval_results_rl_step_150.jsonl"),
    CheckpointSpec("Step 200", "eval_results_rl_step_200.jsonl"),
)


def parse_checkpoint_spec(value: str) -> CheckpointSpec:
    """Parse a CLI checkpoint spec in Label=filename form."""
    label, separator, filename = value.partition("=")
    if not separator or not label.strip() or not filename.strip():
        raise ValueError("checkpoint spec must use non-empty Label=filename")
    return CheckpointSpec(label.strip(), filename.strip())


def default_or_existing_specs(result_dir: Path) -> tuple[CheckpointSpec, ...]:
    """Use original checkpoint defaults, or a current trajectories file when present."""
    if any((result_dir / spec.filename).is_file() for spec in DEFAULT_CHECKPOINTS):
        return DEFAULT_CHECKPOINTS
    if (result_dir / "trajectories.jsonl").is_file():
        return (CheckpointSpec("Current", "trajectories.jsonl"),)
    return DEFAULT_CHECKPOINTS


def load_metrics_from_jsonl(path: Path) -> dict[str, float]:
    """Load em/macro and format/rate from summary or trajectory records."""
    summary_metrics: dict[str, Any] | None = None
    trajectories: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON in {path}:{line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"record in {path}:{line_number} must be an object")
            if record.get("type") == "summary":
                metrics = record.get("metrics")
                if not isinstance(metrics, dict):
                    raise ValueError(f"summary metrics are missing in {path}")
                summary_metrics = metrics
            else:
                trajectories.append(record)

    if summary_metrics is not None:
        return _required_metrics(summary_metrics, path)
    if trajectories:
        return _metrics_from_trajectories(trajectories)
    raise ValueError(f"no summary or trajectory records found in {path}")


def load_metric_series(
    result_dir: Path,
    checkpoints: tuple[CheckpointSpec, ...],
) -> tuple[list[str], list[float], list[float]]:
    """Load checkpoint labels, macro EM, and format rate."""
    labels: list[str] = []
    macro_em: list[float] = []
    format_rate: list[float] = []
    for spec in checkpoints:
        path = result_dir / spec.filename
        if not path.is_file():
            raise FileNotFoundError(f"missing evaluation result: {path}")
        metrics = load_metrics_from_jsonl(path)
        labels.append(spec.label)
        macro_em.append(metrics["em/macro"])
        format_rate.append(metrics["format/rate"])
    return labels, macro_em, format_rate


def configure_style() -> None:
    """Apply a restrained paper-style Matplotlib theme."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.edgecolor": "#202020",
            "axes.linewidth": 1.0,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def make_figure(
    labels: list[str],
    macro_em: list[float],
    format_rate: list[float],
) -> plt.Figure:
    """Create a two-panel checkpoint comparison figure."""
    configure_style()
    x_positions = list(range(len(labels)))
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 5.4),
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )
    em_axis, format_axis = axes
    best_em_index = max(range(len(macro_em)), key=macro_em.__getitem__) if macro_em else 0

    em_colors = ["#D94A4A"] * len(labels)
    if em_colors:
        em_colors[best_em_index] = "#F28E2B"
    bars = em_axis.bar(
        x_positions,
        macro_em,
        width=0.66,
        color=em_colors,
        edgecolor="#202020",
        linewidth=1.0,
        alpha=0.96,
        zorder=3,
    )
    if bars:
        bars[best_em_index].set_hatch("///")
    _add_value_labels(em_axis, x_positions, macro_em, offset=0.025)
    em_axis.set_title("(a) Macro Exact Match", pad=14)
    em_axis.set_ylabel("Score")

    format_axis.plot(
        x_positions,
        format_rate,
        color="#3388B8",
        marker="s",
        markersize=7,
        markerfacecolor="#57B8D2",
        markeredgecolor="#202020",
        markeredgewidth=0.9,
        linewidth=2.1,
        zorder=4,
    )
    if format_rate:
        format_axis.scatter(
            [best_em_index],
            [format_rate[best_em_index]],
            s=90,
            facecolor="#F28E2B",
            edgecolor="#202020",
            linewidth=1.0,
            zorder=5,
        )
    _add_value_labels(format_axis, x_positions, format_rate, offset=0.025)
    format_axis.set_title("(b) Valid Answer Format Rate", pad=14)

    for axis in axes:
        axis.set_xticks(x_positions, labels)
        axis.set_xlabel("Model / Checkpoint", labelpad=10)
        axis.set_ylim(0.0, 1.08)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        axis.grid(
            axis="y",
            color="#C7C7C7",
            linestyle="-.",
            linewidth=0.8,
            alpha=0.75,
            zorder=0,
        )
        axis.tick_params(
            axis="both",
            which="major",
            direction="in",
            top=True,
            right=True,
            length=5,
            width=0.9,
        )
        axis.margins(x=0.06)

    figure.suptitle("Search-R1 Checkpoint Evaluation", fontsize=17, y=0.995)
    figure.text(
        0.5,
        0.935,
        "Metrics read from persisted JSONL summaries or trajectory records",
        ha="center",
        va="top",
        fontsize=10,
        color="#555555",
    )
    figure.text(
        0.5,
        0.015,
        f"Orange indicates the best observed Macro EM checkpoint: {labels[best_em_index]}.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.18, top=0.84)
    return figure


def save_figure(
    labels: list[str],
    macro_em: list[float],
    format_rate: list[float],
    output: Path,
    *,
    dpi: int = 300,
) -> None:
    """Create and save a checkpoint comparison figure."""
    figure = make_figure(labels, macro_em, format_rate)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _required_metrics(metrics: dict[str, Any], path: Path) -> dict[str, float]:
    try:
        return {
            "em/macro": float(metrics["em/macro"]),
            "format/rate": float(metrics["format/rate"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"expected numeric em/macro and format/rate metrics in {path}"
        ) from error


def _metrics_from_trajectories(records: list[dict[str, Any]]) -> dict[str, float]:
    by_source: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        source = str(record.get("data_source") or "unknown")
        by_source.setdefault(source, []).append(record)
    source_scores = [
        _mean([float(record.get("exact_match") is True) for record in source_records])
        for source_records in by_source.values()
    ]
    return {
        "em/macro": _mean(source_scores),
        "format/rate": _mean(
            [float(record.get("valid_format") is True) for record in records]
        ),
    }


def _add_value_labels(
    axis: plt.Axes,
    x_positions: list[int],
    values: list[float],
    *,
    offset: float,
) -> None:
    for x_position, value in zip(x_positions, values, strict=True):
        axis.text(
            x_position,
            value + offset,
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#202020",
        )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
