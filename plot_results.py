import argparse
import json
import os
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, rcParams

from benchmark import RESULTS_FOLDER
from dataset import Datasets
from engine import Engines
from remixer import RemixedDataset


Color = Tuple[float, float, float]


def rgb_from_hex(x: str) -> Color:
    x = x.strip('# ')
    assert len(x) == 6
    return int(x[:2], 16) / 255, int(x[2:4], 16) / 255, int(x[4:], 16) / 255


BLACK = rgb_from_hex('#000000')
GREY1 = rgb_from_hex('#3F3F3F')
GREY2 = rgb_from_hex('#5F5F5F')
GREY3 = rgb_from_hex('#7F7F7F')
GREY4 = rgb_from_hex('#9F9F9F')
GREY5 = rgb_from_hex('#BFBFBF')
WHITE = rgb_from_hex('#FFFFFF')
BLUE = rgb_from_hex('#377DFF')
ORANGE = rgb_from_hex('#FFA500')

ENGINE_COLORS = {
    Engines.MOZILLA_RNNOISE: GREY1,
    Engines.PICOVOICE_KOALA: BLUE,
}

ENGINE_PRINT_NAMES = {
    Engines.MOZILLA_RNNOISE: "Mozilla RNNoise",
    Engines.ORIGINAL: "Original (baseline)",
    Engines.PICOVOICE_KOALA: "Picovoice Koala",
}

ENGINE_ORDER_KEYS = {
    Engines.ORIGINAL: 0,
    Engines.PICOVOICE_KOALA: 10,
}

METRIC_LABELS = {
    'stoi': 'Intelligibility score (STOI)',
    'rtf': 'Real-time factor',
}


def plot_results(
        dataset: Datasets,
        snrs_db: Sequence[float],
        metric: str,
        show: bool = True,
        save_path: Optional[str] = None,
        bar_plot: bool = False,
        include_baseline: bool = False,
        limits: Optional[Tuple[float, float]] = None,
        fonts_folder: Optional[str] = None,
        font_family: Optional[str] = None,
        font_size: int = 13,
        text_color: Color = BLACK) -> None:

    if fonts_folder is not None:
        if os.path.isdir(fonts_folder):
            for x in os.listdir(fonts_folder):
                for font in font_manager.findSystemFonts(os.path.join(fonts_folder, x)):
                    font_manager.fontManager.addfont(font)
        else:
            print(f"Given fonts folder `{fonts_folder}` does not exist")

    if font_family is not None:
        rcParams['font.family'] = font_family

    fig, ax = plt.subplots(figsize=(6, 4))

    sorted_engines = sorted(Engines, key=lambda e: (ENGINE_ORDER_KEYS.get(e, 1), ENGINE_PRINT_NAMES.get(e, e.value)))
    if not include_baseline:
        sorted_engines = [e for e in sorted_engines if e is not Engines.ORIGINAL]

    bar_width = 1 / (len(sorted_engines) + 1)

    for engine_index, engine_type in enumerate(sorted_engines):
        engine_results = []
        for snr_db in snrs_db:
            dataset_name = RemixedDataset.remixed_name(dataset.value, snr_db)
            results_path = os.path.join(RESULTS_FOLDER, dataset_name, engine_type.value + '.json')

            if not os.path.exists(results_path):
                print(f"No results file for engine `{engine_type.value}` on dataset `{dataset_name}`")
                engine_results.append(np.nan)
                continue

            with open(results_path, 'r') as f:
                results_json = json.load(f)

            if metric not in results_json:
                print(f"No `{metric}` results for engine `{engine_type.value}` on dataset `{dataset_name}`")

            engine_results.append(results_json.get(metric, np.nan))

        if bar_plot:
            positions = np.arange(len(snrs_db)) + (engine_index - len(Engines) / 2) * bar_width
            ax.bar(
                positions,
                engine_results,
                width=bar_width,
                color=ENGINE_COLORS.get(engine_type, WHITE),
                align='edge',
                edgecolor=text_color,
                label=ENGINE_PRINT_NAMES.get(engine_type, engine_type.value))
        else:
            ax.plot(
                snrs_db,
                engine_results,
                color=ENGINE_COLORS.get(engine_type, BLACK),
                label=ENGINE_PRINT_NAMES.get(engine_type, engine_type.value))

    for pos, spine in ax.spines.items():
        spine.set_color(text_color if pos in ('bottom', 'left') else 'none')

    if limits is not None:
        ax.set_ylim(*limits)

    xtick_pos = range(len(snrs_db)) if bar_plot else snrs_db
    ax.set_xticks(xtick_pos, [f'{snr:+g} dB' for snr in snrs_db], fontsize=font_size)
    ax.tick_params(axis='both', colors=text_color)

    ax.set_xlabel('Signal-to-noise ratio', fontsize=font_size)
    ax.xaxis.label.set_color(text_color)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=font_size)
    ax.yaxis.label.set_color(text_color)

    legend = plt.legend(framealpha=0, fontsize=font_size, ncol=2, bbox_to_anchor=(0.5, 1.0), loc='lower center')
    for x in legend.texts:
        x.set_color(text_color)

    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, facecolor=WHITE, transparent=True)

    if show:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=[ds.value for ds in Datasets])
    parser.add_argument('--snrs-db', nargs='+', default=[0, 10, 20, 30], type=float)
    parser.add_argument('--metric', choices=METRIC_LABELS.keys(), default='stoi')
    parser.add_argument('--limits', nargs=2, type=float)
    parser.add_argument('--line-plot', action='store_true')
    parser.add_argument('--include-baseline', action='store_true')
    parser.add_argument('--font-family')
    parser.add_argument('--fonts-folder')
    parser.add_argument('--hide', action='store_true')
    parser.add_argument('--save-path')
    args = parser.parse_args()

    plot_results(
        dataset=Datasets(args.dataset),
        snrs_db=args.snrs_db,
        metric=args.metric,
        bar_plot=not args.line_plot,
        limits=args.limits,
        fonts_folder=args.fonts_folder,
        font_family=args.font_family,
        include_baseline=args.include_baseline,
        show=not args.hide,
        save_path=args.save_path,
    )


if __name__ == '__main__':
    main()


__all__ = [
    'Color',
    'plot_results',
    'rgb_from_hex',
]
