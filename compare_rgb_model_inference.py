#!/usr/bin/env python3
"""Compare RGB-only model checkpoints on the held-out building dataset."""

import argparse
import os
import re
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
        '--output-root',
        default='./results/buildings/comparisons',
        help='Base directory where prediction, polygon, and figure outputs are written.',
    )
    parser.add_argument(
        '--single-image',
        default=None,
        help=(
            'Path to a single RGB+ELEV GeoTIFF to evaluate. When set, the script ignores '
            '--inference-csv and only processes that image.'
        ),
    )
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.5,
        help='Minimum IoU threshold for a prediction to be considered a true positive (default: 0.5).',
    )
    parser.add_argument(
        '--compute-pixel-iou',
        action='store_true',
        help='Also compute pixel-wise IoU in addition to polygon-based IoU.',
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


def extract_chip_id(filename: str) -> str:
    """Extract chip ID (e.g., img1423) from a filename."""
    match = re.search(r'(img\d+)', filename)
    if match:
        return match.group(1)
    # Fallback: use filename without extension
    return Path(filename).stem


def load_image_arrays(
    data_dir: Path,
    image_path: Path,
) -> Dict[str, np.ndarray]:
    """Load RGB, elevation, and mask arrays for a given image."""
    if not image_path.exists():
        raise FileNotFoundError(f'Image does not exist: {image_path}')
    
    mask_file = os.path.basename(image_path).replace('RGB+ELEV', 'mask_buildings')
    mask_path = data_dir / 'mask_buildings' / mask_file

    if not mask_path.exists():
        raise FileNotFoundError(f'Mask does not exist: {mask_path}')

    img = skimage.io.imread(image_path)
    mask = skimage.io.imread(mask_path)
    return {
        'img_path': image_path,
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
    """Run inference on all images in the dataframe."""
    inferer(pd.DataFrame({'image': df['image']}))


def convert_predictions_to_polys(run: ModelRun, test_df: pd.DataFrame) -> None:
    """Convert prediction masks to polygon GeoJSON files."""
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


def compute_pixel_iou(pred_mask: np.ndarray, truth_mask: np.ndarray, threshold: float = 0.5) -> float:
    """
    Compute pixel-wise IoU between prediction and ground truth masks.
    
    Arguments
    ---------
    pred_mask : np.ndarray
        Prediction mask (can be binary or probability mask).
    truth_mask : np.ndarray
        Ground truth binary mask.
    threshold : float
        Threshold for binarizing prediction mask.
    
    Returns
    -------
    iou : float
        Pixel-wise IoU score.
    """
    return sol.eval.pixel.iou(truth_mask, pred_mask, prop_threshold=threshold, verbose=False)


def evaluate_per_image_scores(
    run: ModelRun,
    bldg_dir: Path,
    iou_threshold: float = 0.5,
    compute_pixel_iou_flag: bool = False,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Evaluate IoU scores for each image individually.
    
    Uses polygon-based IoU evaluation:
    - Each predicted polygon is matched to ground truth polygons
    - IoU is calculated for each match
    - A prediction is a True Positive if IoU >= iou_threshold
    - Mean IoU is calculated from all matched pairs
    
    Arguments
    ---------
    run : ModelRun
        Model run configuration.
    bldg_dir : Path
        Directory containing ground truth GeoJSON files.
    iou_threshold : float
        Minimum IoU for a prediction to be considered a true positive.
    compute_pixel_iou_flag : bool
        Whether to also compute pixel-wise IoU.
    data_dir : Path, optional
        Data directory for loading masks (needed for pixel IoU).
    
    Returns
    -------
    pd.DataFrame
        DataFrame with per-image metrics including:
        image, precision, recall, f1, mean_iou, true_pos, false_pos, false_neg, pixel_iou (optional)
    """
    prop_files = sorted(f for f in os.listdir(run.prop_dir) if f.endswith('.geojson'))
    if not prop_files:
        raise RuntimeError(f'No proposal GeoJSONs found under {run.prop_dir}.')

    per_image_metrics = []
    for prop_file in prop_files:
        prop_path = run.prop_dir / prop_file
        bldg_path = bldg_dir / prop_file
        evaluator = sol.eval.base.Evaluator(str(bldg_path))
        evaluator.load_proposal(str(prop_path), conf_field_list=[])
        score = evaluator.eval_iou(miniou=iou_threshold, calculate_class_scores=False)
        score_dict = score[0]
        
        tp = score_dict['TruePos']
        fp = score_dict['FalsePos']
        fn = score_dict['FalseNeg']
        
        # Calculate mean IoU from matched polygons
        # The evaluator stores IoU scores in the proposal_GDF after evaluation
        # Column name format is "{iou_field_prefix}_{class_id}", e.g., "iou_score_all"
        mean_iou = 0.0
        
        if hasattr(evaluator, 'proposal_GDF') and not evaluator.proposal_GDF.empty:
            # Find the IoU score column (could be 'iou_score_all' or similar)
            iou_columns = [col for col in evaluator.proposal_GDF.columns if col.startswith('iou_score')]
            
            if iou_columns:
                # Use the first matching IoU column (typically 'iou_score_all')
                iou_series = evaluator.proposal_GDF[iou_columns[0]]
                # Filter out values below threshold (only count true positives for mean IoU)
                valid_ious = [iou for iou in iou_series if pd.notna(iou) and iou >= iou_threshold]
                if valid_ious:
                    mean_iou = np.mean(valid_ious)
        
        # Handle division by zero cases
        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)
        
        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        chip_id = extract_chip_id(prop_file)
        metrics = {
            'image': chip_id,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mean_iou': mean_iou,
            'true_pos': tp,
            'false_pos': fp,
            'false_neg': fn,
        }
        
        # Optionally compute pixel-wise IoU
        if compute_pixel_iou_flag and data_dir is not None:
            try:
                # Load prediction mask
                pred_file = prop_file.replace('geojson_buildings', 'RGB+ELEV').replace('.geojson', '.tif')
                pred_mask_path = run.pred_dir / pred_file
                if pred_mask_path.exists():
                    pred_mask = skimage.io.imread(pred_mask_path)
                    if len(pred_mask.shape) == 3:
                        pred_mask = pred_mask[..., 0]
                    
                    # Load ground truth mask
                    mask_file = prop_file.replace('geojson_buildings', 'mask_buildings').replace('.geojson', '.tif')
                    truth_mask_path = data_dir / 'mask_buildings' / mask_file
                    if truth_mask_path.exists():
                        truth_mask = skimage.io.imread(truth_mask_path)
                        if len(truth_mask.shape) == 3:
                            truth_mask = truth_mask[..., 0]
                        
                        # Ensure same shape
                        if pred_mask.shape == truth_mask.shape:
                            pixel_iou = compute_pixel_iou(pred_mask, truth_mask, threshold=0.5)
                            metrics['pixel_iou'] = pixel_iou
            except Exception as e:
                print(f'Warning: Could not compute pixel IoU for {chip_id}: {e}')
                metrics['pixel_iou'] = np.nan
        
        per_image_metrics.append(metrics)
    
    return pd.DataFrame(per_image_metrics)


def compute_average_metrics(per_image_df: pd.DataFrame) -> Dict[str, float]:
    """Compute average metrics across all images."""
    metrics = {
        'avg_precision': per_image_df['precision'].mean(),
        'avg_recall': per_image_df['recall'].mean(),
        'avg_f1': per_image_df['f1'].mean(),
        'avg_mean_iou': per_image_df['mean_iou'].mean(),
        'total_true_pos': per_image_df['true_pos'].sum(),
        'total_false_pos': per_image_df['false_pos'].sum(),
        'total_false_neg': per_image_df['false_neg'].sum(),
        'num_images': len(per_image_df),
    }
    
    # Add pixel IoU average if present
    if 'pixel_iou' in per_image_df.columns:
        metrics['avg_pixel_iou'] = per_image_df['pixel_iou'].mean()
    
    return metrics


def plot_single_image_comparison(
    image_data: Dict[str, np.ndarray],
    model_predictions: List[Tuple[str, np.ndarray]],
    output_path: Path,
    chip_id: str,
) -> None:
    """Generate and save a comparison plot for a single image across all models."""
    cols = 2 + len(model_predictions)
    fig, ax = plt.subplots(1, cols, figsize=(4 * cols, 4))
    
    ax[0].imshow(image_data['rgb'])
    ax[0].set_title('RGB')
    ax[1].imshow(image_data['mask'], cmap='Blues')
    ax[1].set_title('Ground Truth')
    
    for idx, (model_name, pred) in enumerate(model_predictions, start=2):
        ax[idx].imshow(pred > 0, cmap='Blues')
        ax[idx].set_title(f'Prediction\n{model_name}')
    
    for axis in ax:
        axis.axis('off')
    
    fig.suptitle(f'Image: {chip_id}', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    model_runs = build_model_runs(args)
    data_dir = Path(args.data_dir)
    bldg_dir = data_dir / 'geojson_buildings'
    output_root = Path(args.output_root)
    single_image_path = Path(args.single_image) if args.single_image else None

    # Create output directories
    image_comparison_dir = output_root / 'image-comparison'
    image_comparison_dir.mkdir(parents=True, exist_ok=True)

    # Build inference dataframe
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

    # Storage for all models' results
    all_model_avg_metrics = []
    all_per_image_metrics = []  # (model_name, per_image_df)
    model_predictions_cache: Dict[str, Dict[str, np.ndarray]] = {}  # model_name -> {filename: pred_array}

    # Run inference for each model
    for run in model_runs:
        print(f'\n=== Running inference for {run.name} ===')
        clear_run_outputs(run)
        config_inference_path = args.inference_csv if single_image_path is None else None
        config, inferer = prepare_inferer(run, config_inference_path)

        # Run inference on all images
        infer_dataframe(inferer, inference_df)
        convert_predictions_to_polys(run, inference_df)
        
        # Evaluate per-image scores
        per_image_df = evaluate_per_image_scores(
            run,
            bldg_dir,
            iou_threshold=args.iou_threshold,
            compute_pixel_iou_flag=args.compute_pixel_iou,
            data_dir=data_dir,
        )
        per_image_df['model'] = run.name
        all_per_image_metrics.append(per_image_df)
        
        # Save per-model detailed metrics
        per_model_csv = run.pred_dir.parent / 'per_image_metrics.csv'
        per_image_df.to_csv(per_model_csv, index=False)
        print(f'Saved per-image metrics to {per_model_csv}')
        
        # Compute and store average metrics
        avg_metrics = compute_average_metrics(per_image_df)
        avg_metrics['model'] = run.name
        all_model_avg_metrics.append(avg_metrics)
        
        # Save model summary
        model_summary_csv = run.pred_dir.parent / 'metrics_summary.csv'
        pd.DataFrame([avg_metrics]).to_csv(model_summary_csv, index=False)
        
        print(
            f"Avg Precision={avg_metrics['avg_precision']:.4f} "
            f"Avg Recall={avg_metrics['avg_recall']:.4f} "
            f"Avg F1={avg_metrics['avg_f1']:.4f} "
            f"Avg Mean IoU={avg_metrics['avg_mean_iou']:.4f}"
        )
        if 'avg_pixel_iou' in avg_metrics:
            print(f"  Avg Pixel IoU={avg_metrics['avg_pixel_iou']:.4f}")
        
        # Cache predictions for comparison plots
        model_predictions_cache[run.name] = {}
        for pred_file in os.listdir(run.pred_dir):
            if pred_file.endswith('.tif'):
                pred_path = run.pred_dir / pred_file
                pred_array = skimage.io.imread(pred_path)[..., 0]
                model_predictions_cache[run.name][pred_file] = pred_array

    # Combine all per-image metrics into one CSV
    combined_per_image_df = pd.concat(all_per_image_metrics, ignore_index=True)
    # Reorder columns for clarity
    base_cols = ['model', 'image', 'precision', 'recall', 'f1', 'mean_iou']
    if 'pixel_iou' in combined_per_image_df.columns:
        base_cols.append('pixel_iou')
    base_cols.extend(['true_pos', 'false_pos', 'false_neg'])
    cols_order = [col for col in base_cols if col in combined_per_image_df.columns]
    combined_per_image_df = combined_per_image_df[cols_order]
    combined_per_image_csv = output_root / 'all_models_per_image_metrics.csv'
    combined_per_image_df.to_csv(combined_per_image_csv, index=False)
    print(f'\nSaved combined per-image metrics to {combined_per_image_csv}')

    # Save average comparison metrics
    avg_metrics_df = pd.DataFrame(all_model_avg_metrics)
    # Build column order dynamically based on what's available
    base_cols = ['model', 'avg_precision', 'avg_recall', 'avg_f1', 'avg_mean_iou']
    if 'avg_pixel_iou' in avg_metrics_df.columns:
        base_cols.append('avg_pixel_iou')
    base_cols.extend(['total_true_pos', 'total_false_pos', 'total_false_neg', 'num_images'])
    cols_order_avg = [col for col in base_cols if col in avg_metrics_df.columns]
    avg_metrics_df = avg_metrics_df[cols_order_avg]
    comparison_csv = output_root / 'comparison_average_metrics.csv'
    avg_metrics_df.to_csv(comparison_csv, index=False)
    print(f'Saved comparison average metrics to {comparison_csv}')

    print('\n=== Comparison Summary (Average Metrics) ===')
    print(avg_metrics_df.to_string(index=False))

    # Generate comparison plots for each image
    print('\n=== Generating comparison plots for each image ===')
    for idx, row in inference_df.iterrows():
        image_path = Path(row['image'])
        image_filename = os.path.basename(image_path)
        chip_id = extract_chip_id(image_filename)
        
        try:
            image_data = load_image_arrays(data_dir, image_path)
        except FileNotFoundError as e:
            print(f'Warning: Skipping plot for {chip_id}: {e}')
            continue
        
        # Collect predictions from all models for this image
        model_predictions = []
        for run in model_runs:
            if image_filename in model_predictions_cache[run.name]:
                pred = model_predictions_cache[run.name][image_filename]
                model_predictions.append((run.name, pred))
        
        if not model_predictions:
            print(f'Warning: No predictions found for {chip_id}, skipping plot.')
            continue
        
        # Generate and save comparison plot
        plot_path = image_comparison_dir / f'{chip_id}_comparison.png'
        plot_single_image_comparison(image_data, model_predictions, plot_path, chip_id)
        print(f'  Saved: {plot_path}')

    print(f'\n=== Done! All outputs saved to {output_root} ===')


if __name__ == '__main__':
    main()
