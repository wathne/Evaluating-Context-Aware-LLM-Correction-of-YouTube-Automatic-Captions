# Plot module.


from copy import deepcopy
import matplotlib.pyplot as plt
from mean_evaluation_results_wrapper import Record as MeanRecord
from mean_evaluation_results_wrapper import RecordList as MeanRecordList
from typing import Literal


def bar_chart_mean_records(
    mean_records: MeanRecordList,
    bar_legends: list[str],
    bar_colors: list[str],
    bar_chart_path: str,
    bar_chart_title: str | None = None,
    show_stdev: bool = False,
    show_variance: bool = False,
    reference_new_asr_mean_record: MeanRecord | None = None,
    reference_new_bar_legend: str | None = None,
    reference_new_bar_color: str | None = None,
    reference_old_asr_mean_record: MeanRecord | None = None,
    reference_old_bar_legend: str | None = None,
    reference_old_bar_color: str | None = None,
) -> None:
    bar_width: float = 0.8
    bar_bottom: float | None = 0.0
    bar_alignment: Literal["center", "edge"] = "center"

    set_ylim_hundred: bool = False

    try:
        show_stdev = bool(show_stdev)
    except (TypeError, ValueError):
        show_stdev = False
    try:
        show_variance = bool(show_variance)
    except (TypeError, ValueError):
        show_variance = False

    # Allows stdev or variance, but not both.
    if show_stdev and show_variance:
        show_variance = False

    bar_colors_copy: list[str] = deepcopy(bar_colors)
    bar_legends_copy: list[str] = deepcopy(bar_legends)

    bar_positions: list[float] = [] # x
    bar_heights: list[float] = [] # y
    bar_widths: list[float] = []
    bar_labels: list[str] = []
    y_errors: list[float] = []

    fig = plt.figure(
        num=None,
        figsize=[8, 4],
        dpi=600.0,
    )

    ax = plt.subplot(1, 1 ,1)

    mean_record: MeanRecord
    if show_stdev:
        if reference_new_asr_mean_record is not None:
            if (
                reference_new_asr_mean_record.mean is not None and
                reference_new_asr_mean_record.stdev is not None
            ):
                bar_positions.insert(0, 0 + bar_width/4)
                bar_heights.insert(0, float(reference_new_asr_mean_record.mean))
                bar_widths.insert(0, bar_width/2)
                bar_labels.insert(0, "*")
                y_errors.insert(0, float(reference_new_asr_mean_record.stdev))
                if reference_new_bar_color is None:
                    bar_colors_copy.insert(0, "#696969") # darkgrey
                else:
                    bar_colors_copy.insert(0, str(reference_new_bar_color))
                if reference_new_bar_legend is None:
                    bar_legends_copy.insert(
                        0,
                        "* - new YT captions (no LLM)",
                    )
                else:
                    bar_legends_copy.insert(0, str(reference_new_bar_legend))

        if reference_old_asr_mean_record is not None:
            if (
                reference_old_asr_mean_record.mean is not None and
                reference_old_asr_mean_record.stdev is not None
            ):
                bar_positions.insert(0, 0 - bar_width/4)
                bar_heights.insert(0, float(reference_old_asr_mean_record.mean))
                bar_widths.insert(0, bar_width/2)
                bar_labels.insert(0, "*")
                y_errors.insert(0, float(reference_old_asr_mean_record.stdev))
                if reference_old_bar_color is None:
                    bar_colors_copy.insert(0, "#a9a9a9") # dimgrey
                else:
                    bar_colors_copy.insert(0, str(reference_old_bar_color))
                if reference_old_bar_legend is None:
                    bar_legends_copy.insert(
                        0,
                        "* - old YT captions (no LLM)",
                    )
                else:
                    bar_legends_copy.insert(0, str(reference_old_bar_legend))

        for mean_record in mean_records:
            if mean_record.identifier is None:
                # TODO(wathne): Temporary, too fragile.
                raise Exception("mean_record.identifier is None")

            bar_positions.append(float(mean_record.identifier))
            bar_widths.append(bar_width)
            bar_labels.append(str(mean_record.identifier))

            if (
                mean_record.mean is not None and
                mean_record.stdev is not None
            ):
                bar_heights.append(float(mean_record.mean))
                y_errors.append(float(mean_record.stdev))
                if not set_ylim_hundred:
                    if float(mean_record.mean) > 1.0:
                        set_ylim_hundred = True
            else:
                bar_heights.append(0.0)
                y_errors.append(0.0)

        for i in range(0, len(bar_positions), 1):
            ax.bar(
                x=bar_positions[i],
                height=bar_heights[i],
                width=bar_widths[i],
                bottom=bar_bottom,
                align=bar_alignment,
                facecolor=bar_colors_copy[i],
                label=bar_legends_copy[i],
                yerr=y_errors[i],
            )

        ax.set_ylabel(ylabel="mean & std dev")

    elif show_variance:
        if reference_new_asr_mean_record is not None:
            if (
                reference_new_asr_mean_record.mean is not None and
                reference_new_asr_mean_record.variance is not None
            ):
                bar_positions.insert(0, 0 + bar_width/4)
                bar_heights.insert(0, float(reference_new_asr_mean_record.mean))
                bar_widths.insert(0, bar_width/2)
                bar_labels.insert(0, "*")
                y_errors.insert(0, float(
                    reference_new_asr_mean_record.variance
                ))
                if reference_new_bar_color is None:
                    bar_colors_copy.insert(0, "#696969") # darkgrey
                else:
                    bar_colors_copy.insert(0, str(reference_new_bar_color))
                if reference_new_bar_legend is None:
                    bar_legends_copy.insert(
                        0,
                        "* - new YT captions (no LLM)",
                    )
                else:
                    bar_legends_copy.insert(0, str(reference_new_bar_legend))

        if reference_old_asr_mean_record is not None:
            if (
                reference_old_asr_mean_record.mean is not None and
                reference_old_asr_mean_record.variance is not None
            ):
                bar_positions.insert(0, 0 - bar_width/4)
                bar_heights.insert(0, float(reference_old_asr_mean_record.mean))
                bar_widths.insert(0, bar_width/2)
                bar_labels.insert(0, "*")
                y_errors.insert(0, float(
                    reference_old_asr_mean_record.variance
                ))
                if reference_old_bar_color is None:
                    bar_colors_copy.insert(0, "#a9a9a9") # dimgrey
                else:
                    bar_colors_copy.insert(0, str(reference_old_bar_color))
                if reference_old_bar_legend is None:
                    bar_legends_copy.insert(
                        0,
                        "* - old YT captions (no LLM)",
                    )
                else:
                    bar_legends_copy.insert(0, str(reference_old_bar_legend))

        for mean_record in mean_records:
            if mean_record.identifier is None:
                # TODO(wathne): Temporary, too fragile.
                raise Exception("mean_record.identifier is None")

            bar_positions.append(float(mean_record.identifier))
            bar_widths.append(bar_width)
            bar_labels.append(str(mean_record.identifier))

            if (
                mean_record.mean is not None and
                mean_record.variance is not None
            ):
                bar_heights.append(float(mean_record.mean))
                y_errors.append(float(mean_record.variance))
                if not set_ylim_hundred:
                    if float(mean_record.mean) > 1.0:
                        set_ylim_hundred = True
            else:
                bar_heights.append(0.0)
                y_errors.append(0.0)

        for i in range(0, len(bar_positions), 1):
            ax.bar(
                x=bar_positions[i],
                height=bar_heights[i],
                width=bar_widths[i],
                bottom=bar_bottom,
                align=bar_alignment,
                facecolor=bar_colors_copy[i],
                label=bar_legends_copy[i],
                yerr=y_errors[i],
            )

        ax.set_ylabel(ylabel="mean & variance")

    else:
        if reference_new_asr_mean_record is not None:
            if reference_new_asr_mean_record.mean is not None:
                bar_positions.insert(0, 0 + bar_width/4)
                bar_heights.insert(0, float(reference_new_asr_mean_record.mean))
                bar_widths.insert(0, bar_width/2)
                bar_labels.insert(0, "*")
                if reference_new_bar_color is None:
                    bar_colors_copy.insert(0, "#696969") # darkgrey
                else:
                    bar_colors_copy.insert(0, str(reference_new_bar_color))
                if reference_new_bar_legend is None:
                    bar_legends_copy.insert(
                        0,
                        "* - new YT captions (no LLM)",
                    )
                else:
                    bar_legends_copy.insert(0, str(reference_new_bar_legend))

        if reference_old_asr_mean_record is not None:
            if reference_old_asr_mean_record.mean is not None:
                bar_positions.insert(0, 0 - bar_width/4)
                bar_heights.insert(0, float(reference_old_asr_mean_record.mean))
                bar_widths.insert(0, bar_width/2)
                bar_labels.insert(0, "*")
                if reference_old_bar_color is None:
                    bar_colors_copy.insert(0, "#a9a9a9") # dimgrey
                else:
                    bar_colors_copy.insert(0, str(reference_old_bar_color))
                if reference_old_bar_legend is None:
                    bar_legends_copy.insert(
                        0,
                        "* - old YT captions (no LLM)",
                    )
                else:
                    bar_legends_copy.insert(0, str(reference_old_bar_legend))

        for mean_record in mean_records:
            if mean_record.identifier is None:
                # TODO(wathne): Temporary, too fragile.
                raise Exception("mean_record.identifier is None")

            bar_positions.append(float(mean_record.identifier))
            bar_widths.append(bar_width)
            bar_labels.append(str(mean_record.identifier))

            if mean_record.mean is not None:
                bar_heights.append(float(mean_record.mean))
                if not set_ylim_hundred:
                    if float(mean_record.mean) > 1.0:
                        set_ylim_hundred = True
            else:
                bar_heights.append(0.0)

        for i in range(0, len(bar_positions), 1):
            ax.bar(
                x=bar_positions[i],
                height=bar_heights[i],
                width=bar_widths[i],
                bottom=bar_bottom,
                align=bar_alignment,
                facecolor=bar_colors_copy[i],
                label=bar_legends_copy[i],
                yerr=None,
            )

        ax.set_ylabel(ylabel="mean")

    ax.set_xticks(
        ticks=bar_positions,
        labels=bar_labels,
    )

    if set_ylim_hundred:
        ax.set_ylim(bottom=0.0, top=100.0)
    else:
        ax.set_ylim(bottom=0.0, top=1.0)

    if bar_chart_title is not None:
        ax.set_title(label=str(bar_chart_title))

    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

    fig.tight_layout()

    plt.savefig(fname=str(bar_chart_path))

    return None
