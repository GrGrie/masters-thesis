import argparse

from src.utils.config import get_output_dir, load_config, validate_config
from src.utils.logging import get_logger, log_environment, setup_logging
from src.utils.seed import set_seed


logger = get_logger(__name__)

def run_training(config):
    logger.info("Starting training pipeline...")
    # TODO: Initialize dataset, model, and training loop
    # Example:
    # from src.datasets.waterbirds import get_waterbirds_dataloaders
    # from src.models.cl_vit import get_cl_vit
    # from src.training.train import train_model
    pass

def run_evaluation(config):
    logger.info("Starting evaluation pipeline...")
    # TODO: Evaluate average and worst-group accuracy
    # Example:
    # from src.evaluation.performance import evaluate_worst_group_accuracy
    pass

def run_analysis(config):
    logger.info("Starting representation analysis pipeline...")
    # TODO: Run the layer-wise mechanistic analysis
    # Example:
    # from src.analysis.attention import analyze_attention_distance
    # from src.analysis.spectral import plot_umap
    pass

def run_smoke(config):
    logger.info("Starting smoke check...")
    from src.utils.smoke import run_smoke_check
    result = run_smoke_check(config)
    logger.info("Smoke check passed: %s", result)

def main():
    parser = argparse.ArgumentParser(description="Master's Thesis Pipeline")
    parser.add_argument('--config', type=str, default='configs/train.yaml', help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'analyze', 'smoke', 'all'], required=True,
                        help='Mode to run the pipeline in')
    
    args = parser.parse_args()
    config = load_config(args.config)
    validate_config(config, args.mode)

    output_dir = get_output_dir(config)
    setup_logging(output_dir)
    set_seed(int(config['experiment']['seed']))

    logger.info(f"Loaded config: {args.config}")
    logger.info(f"Running in mode: {args.mode}")
    log_environment(logger)

    if args.mode == 'smoke':
        run_smoke(config)
        return

    if args.mode in ['train', 'all']:
        run_training(config)
    
    if args.mode in ['evaluate', 'all']:
        run_evaluation(config)

    if args.mode in ['analyze', 'all']:
        run_analysis(config)

if __name__ == "__main__":
    main()
