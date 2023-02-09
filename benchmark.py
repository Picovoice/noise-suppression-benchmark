import argparse
import json
import os
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, Optional, Sequence

import numpy as np
import soundfile

from dataset import *
from engine import *
from metrics import *
from remixer import *


DEFAULT_CACHE_FOLDER = os.path.join(os.path.dirname(__file__), 'cache')
RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), 'results')


@dataclass
class Result:
    data_name: str
    rtf: float
    stoi: float


def process_chunk(
        engine_type: Engines,
        engine_params: Dict[str, Any],
        data: Sequence[DataPair],
        output_folder: Optional[str] = None) -> Sequence[Result]:
    engine = Engine.create(engine_type, **engine_params)
    results = []

    for data_pair in data:
        engine_result = engine.process(data_pair.noisy_path)

        clean_pcm, sample_rate = soundfile.read(data_pair.clean_path)
        length_sec = clean_pcm.size / sample_rate

        rtf = engine_result.runtime / length_sec
        stoi = eval_stoi(clean_pcm, engine_result.denoised_pcm, sample_rate)

        results.append(Result(data_name=data_pair.name, rtf=rtf, stoi=stoi))

        if output_folder is not None:
            soundfile.write(
                os.path.join(output_folder, data_pair.name + '.wav'),
                engine_result.denoised_pcm,
                sample_rate)

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=[ds.value for ds in Datasets], required=True)
    parser.add_argument('--data-folder', required=True)
    parser.add_argument('--remix-snr-db', type=float)
    parser.add_argument('--cache-folder', default=DEFAULT_CACHE_FOLDER)
    parser.add_argument('--output-folder')
    parser.add_argument('--engine', choices=[en.value for en in Engines], required=True)
    parser.add_argument('--picovoice-access-key')
    parser.add_argument('--rnnoise-demo-path')
    args = parser.parse_args()

    engine_type = Engines(args.engine)
    engine_params = dict()
    if engine_type is Engines.MOZILLA_RNNOISE:
        if args.rnnoise_demo_path is None:
            raise ValueError(f"Engine {engine_type.value} requires --rnnoise-demo-path")
        engine_params.update(demo_path=args.rnnoise_demo_path)
    elif engine_type is Engines.PICOVOICE_KOALA:
        if args.picovoice_access_key is None:
            raise ValueError(f"Engine {engine_type.value} requires --picovoice-access-key")
        engine_params.update(access_key=args.picovoice_access_key)

    print("Loading data...")
    start_time = perf_counter()
    dataset = Dataset.load(Datasets(args.dataset), args.data_folder)

    if args.remix_snr_db is not None:
        dataset = RemixedDataset(dataset, snr_db=args.remix_snr_db, cache_folder=args.cache_folder)

    data_pairs = [dataset.get(i) for i in range(dataset.size())]
    print(f"...done in {perf_counter() - start_time:.2f} seconds")

    output_folder = args.output_folder
    if output_folder is not None:
        output_folder = os.path.join(output_folder, str(dataset), engine_type.value)
        os.makedirs(output_folder, exist_ok=True)

    results = process_chunk(engine_type, engine_params, data_pairs, output_folder)

    if output_folder is not None:
        results_dict = [vars(res) for res in results]
        results_dict = {res.pop('data_name'): res for res in results_dict}
        with open(os.path.join(output_folder, 'results.json'), 'w') as f:
            json.dump(results_dict, f, indent=4)
        print(f"Individual results stored in {output_folder}")

    mean_rtf = np.mean([res.rtf for res in results])
    mean_stoi = np.mean([res.stoi for res in results])
    print(f"Mean RTF: {mean_rtf:.4f}")
    print(f"Mean STOI: {mean_stoi:.4f}")

    results_folder = os.path.join(RESULTS_FOLDER, str(dataset))
    os.makedirs(results_folder, exist_ok=True)
    results_path = os.path.join(results_folder, engine_type.value + '.json')
    with open(results_path, 'w') as f:
        json.dump(dict(rtf=mean_rtf, stoi=mean_stoi), f, indent=4)
    print(f"Results written to {results_path}")


if __name__ == '__main__':
    main()
