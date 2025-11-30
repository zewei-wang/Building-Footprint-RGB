#!/usr/bin/env python3
"""Compare RGB-only model checkpoints on the held-out building dataset."""

import argparse
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.io

import libs.solaris as sol
from networks.vgg16_unet import get_modified_vgg16_unet

warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class ModelRun:
    name: str
    config_path: Path
    weights_path: Path
    pred_dir: Path
    prop_dir: Path


def clear_run_outputs(run: ModelRun) -> None:
    """Remove stale prediction/proposal files before a new inference run."""
    for directory in (run.pred_dir, run.prop_dir):
        if not directory.exists():
            continue
        for file_path in directory.glob('*'):
            if file_path.is_file():
                file_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run inference + scoring for multiple RGB-only checkpoints and compare them.'
    )
    parser.add_argument(
        '--model-run',
        action='append',
        nargs=3,
        metavar=('NAME', 'CONFIG', 'WEIGHTS'),
        required=True,
        help=(
            'Register a model comparison entry. Provide a run NAME, the config YAML path, '
            'and the checkpoint/weights path. Repeat the flag for each model you want to compare.'
        ),
    )
    parser.add_argument(
        '--inference-csv',
        default='./data/buildings/split_blind_test.csv',
        help='CSV file listing the inference image paths (default: split_blind_test.csv).',
    )
    parser.add_argument(
        '--data-dir',
        default='./data/buildings/',
        help='Root directory that contains RGB+ELEV/, geojson_buildings/, mask_buildings/.',
    )
    parser.add_argument(
        '--prefix',
        default='SN2_buildings_train_AOI_2_Vegas_',
        help='Filename prefix used by the building dataset.',
    )
    parser.add_argument(
        '--sample-chip',
        default='img1423',
        help='Chip ID to visualize when comparing predictions.',
    )
    parser.add_argument(
        '--output-root',
        default='./results/buildings/comparisons',
        help='Base directory where prediction, polygon, and figure outputs are written.',
    )
    parser.add_argument(
        '--save-figure',
        default=None,
        help='Optional path to save the sample comparison figure. Defaults inside output_root.',
    )
    parser.add_argument(
        '--single-image',
        default=None,
        help=(
            'Path to a single RGB+ELEV GeoTIFF to evaluate. When set, the script ignores '
            '--inference-csv and only processes that image.'
        ),
    )
    return parser.parse_args()


def build_model_runs(args: argparse.Namespace) -> List[ModelRun]:
    runs = []
    output_root = Path(args.output_root)
    for name, config_path, weights_path in args.model_run:
        run_root = output_root / name
        pred_dir = run_root / 'pred_mask'
        prop_dir = run_root / 'prop_geojson'
        runs.append(
            ModelRun(
                name=name,
                config_path=Path(config_path),
                weights_path=Path(weights_path),
                pred_dir=pred_dir,
                prop_dir=prop_dir,
            )
        )
    return runs


def load_sample_arrays(
    data_dir: Path,
    prefix: str,
    chip_id: Optional[str],
    sample_image_path: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    if sample_image_path is not None:
        img_path = Path(sample_image_path)
        if not img_path.exists():
            raise FileNotFoundError(f'Sample image does not exist: {img_path}')
        mask_file = os.path.basename(img_path).replace('RGB+ELEV', 'mask_buildings')
    else:
        if chip_id is None:
            raise ValueError('chip_id is required when --single-image is not provided.')
        img_file = prefix + 'RGB+ELEV_' + chip_id + '.tif'
        mask_file = prefix + 'mask_buildings_' + chip_id + '.tif'
        img_path = data_dir / 'RGB+ELEV' / img_file
    mask_path = data_dir / 'mask_buildings' / mask_file

    if not mask_path.exists():
        raise FileNotFoundError(f'Sample mask does not exist: {mask_path}')

    img = skimage.io.imread(img_path)
    mask = skimage.io.imread(mask_path)
    return {
        'img_path': img_path,
        'rgb': img[..., :3],
        'elev': img[..., -1],
        'mask': mask,
    }


def prepare_inferer(run: ModelRun, inference_csv: Optional[str]) -> Tuple[dict, sol.nets.infer.Inferer]:
    config = sol.utils.config.parse(str(run.config_path))
    config['train'] = False
    if inference_csv:
        config['inference_data_csv'] = inference_csv
    os.makedirs(run.pred_dir, exist_ok=True)
    os.makedirs(run.prop_dir, exist_ok=True)
    config['inference']['output_dir'] = str(run.pred_dir)

    custom_model = get_modified_vgg16_unet(in_channels=config['data_specs']['channels'])
    custom_model_dict = {
        'model_name': 'modified_vgg16_unet',
        'weight_path': str(run.weights_path),
        'weight_url': None,
        'arch': custom_model,
    }
    inferer = sol.nets.infer.Inferer(config, custom_model_dict=custom_model_dict)
    return config, inferer


def infer_dataframe(inferer: sol.nets.infer.Inferer, df: pd.DataFrame) -> None:
    # Solaris inferer expects a DataFrame with at least an 'image' column.
    inferer(pd.DataFrame({'image': df['image']}))


def convert_predictions_to_polys(run: ModelRun, test_df: pd.DataFrame) -> None:
    pred_dir = run.pred_dir
    prop_dir = run.prop_dir
    img_lookup = {os.path.basename(path): path for path in test_df['image']}
    pred_files = sorted(f for f in os.listdir(pred_dir) if f.endswith('.tif'))
    if not pred_files:
        raise RuntimeError(f'No prediction TIFFs found under {pred_dir}.')

    for pred_file in pred_files:
        pred_path = pred_dir / pred_file
        if pred_file not in img_lookup:
            print(
                f'Warning: skipping prediction {pred_file} because no source image '
                'was found in the current inference set.'
            )
            continue
        img_path = img_lookup[pred_file]
        pred = skimage.io.imread(pred_path)[..., 0]
        prop_file = pred_file.replace('RGB+ELEV', 'geojson_buildings').replace('.tif', '.geojson')
        prop_path = prop_dir / prop_file
        sol.vector.mask.mask_to_poly_geojson(
            pred_arr=pred,
            reference_im=img_path,
            do_transform=True,
            min_area=1e-10,
            output_path=prop_path,
        )


def evaluate_scores(run: ModelRun, bldg_dir: Path) -> pd.DataFrame:
    prop_files = sorted(f for f in os.listdir(run.prop_dir) if f.endswith('.geojson'))
    if not prop_files:
        raise RuntimeError(f'No proposal GeoJSONs found under {run.prop_dir}.')

    score_list = []
    for prop_file in prop_files:
        prop_path = run.prop_dir / prop_file
        bldg_path = bldg_dir / prop_file
        evaluator = sol.eval.base.Evaluator(str(bldg_path))
        evaluator.load_proposal(str(prop_path), conf_field_list=[])
        score = evaluator.eval_iou(miniou=0.5, calculate_class_scores=False)
        score_list.append(score[0])
    return pd.DataFrame.from_records(score_list)


def aggregate_metrics(score_df: pd.DataFrame) -> Dict[str, float]:
    tp_agg = score_df['TruePos'].sum()
    fp_agg = score_df['FalsePos'].sum()
    fn_agg = score_df['FalseNeg'].sum()
    precision = tp_agg / (tp_agg + fp_agg)
    recall = tp_agg / (tp_agg + fn_agg)
    f1 = 2 * precision * recall / (precision + recall)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_pos': tp_agg,
        'false_pos': fp_agg,
        'false_neg': fn_agg,
    }


def plot_sample(sample_data: Dict[str, np.ndarray], preds: List[Tuple[str, np.ndarray]], figure_path: Path) -> None:
    cols = 2 + len(preds)
    fig, ax = plt.subplots(1, cols, figsize=(4 * cols, 4))
    ax[0].imshow(sample_data['rgb'])
    ax[0].set_title('RGB')
    ax[1].imshow(sample_data['mask'], cmap='Blues')
    ax[1].set_title('Ground Truth')
    for idx, (name, pred) in enumerate(preds, start=2):
        ax[idx].imshow(pred > 0, cmap='Blues')
        ax[idx].set_title(f'Prediction\n{name}')
    for axis in ax:
        axis.axis('off')
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    model_runs = build_model_runs(args)
    data_dir = Path(args.data_dir)
    bldg_dir = data_dir / 'geojson_buildings'
    single_image_path = Path(args.single_image) if args.single_image else None

    if single_image_path is not None:
        if not single_image_path.exists():
            raise FileNotFoundError(f'--single-image path does not exist: {single_image_path}')
        inference_df = pd.DataFrame({'image': [str(single_image_path)]})
    else:
        inference_df = pd.read_csv(args.inference_csv)
        if 'image' not in inference_df.columns:
            raise ValueError(
                f"Inference CSV {args.inference_csv} must contain an 'image' column."
            )

    sample_data = load_sample_arrays(
        data_dir,
        args.prefix,
        None if single_image_path else args.sample_chip,
        sample_image_path=single_image_path,
    )
    sample_file = os.path.basename(sample_data['img_path'])

    all_metrics = []
    sample_predictions: List[Tuple[str, np.ndarray]] = []

    for run in model_runs:
        print(f'\n=== Running inference for {run.name} ===')
        clear_run_outputs(run)
        config_inference_path = args.inference_csv if single_image_path is None else None
        config, inferer = prepare_inferer(run, config_inference_path)

        # Run inference on the sample chip.
        sample_df = pd.DataFrame({'image': [str(sample_data['img_path'])]})
        infer_dataframe(inferer, sample_df)
        sample_pred_path = run.pred_dir / sample_file
        sample_pred = skimage.io.imread(sample_pred_path)[..., 0]
        sample_predictions.append((run.name, sample_pred))

        # Run inference on the entire test set.
        infer_dataframe(inferer, inference_df)
        convert_predictions_to_polys(run, inference_df)
        score_df = evaluate_scores(run, bldg_dir)
        metrics = aggregate_metrics(score_df)
        metrics_row = {'model': run.name, **metrics}
        all_metrics.append(metrics_row)
        pd.DataFrame(metrics_row, index=[0]).to_csv(
            run.pred_dir.parent / 'metrics_summary.csv', index=False
        )
        print(
            'Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f}'.format(**metrics)
        )

    metrics_df = pd.DataFrame(all_metrics)
    print('\n=== Comparison summary ===')
    print(metrics_df)
    comparison_csv = Path(args.output_root) / f'comparison_metrics_{args.sample_chip}.csv'
    comparison_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(comparison_csv, index=False)

    figure_path = (
        Path(args.save_figure)
        if args.save_figure
        else Path(args.output_root) / f'sample_{args.sample_chip}_comparison.png'
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plot_sample(sample_data, sample_predictions, figure_path)
    print(f'Saved sample comparison figure to {figure_path}')


if __name__ == '__main__':
    main()
