"""
Toroidal Attention Mechanism - Main Entry Point

Provides a unified interface for:
- Running unit tests
- Training models
- Evaluating models
- Running ablation studies
- Interactive demos
"""

import argparse
import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))


def run_tests(comprehensive=False):
    """Run test suite."""
    if comprehensive:
        # Run comprehensive test suite with all categories
        from tests.run_all_tests import main as run_comprehensive
        print("Running comprehensive test suite...")
        run_comprehensive()
    else:
        # Run basic unit tests
        from tests.test_toroidal_attention import run_all_tests
        print("Running unit tests...")
        run_all_tests()


def run_training(args):
    """Run training."""
    from scripts.train_toroidal import (
        TrainingConfig,
        run_ablation_study,
        train_toroidal_attention,
        load_training_config_yaml,
    )

    if args.ablation:
        run_ablation_study()
    else:
        if args.config:
            config = load_training_config_yaml(Path(args.config))
        else:
            config = TrainingConfig(
                depth=args.depth,
                fusion_mode=args.fusion_mode,
                dataset_type=args.dataset,
                batch_size=args.batch_size,
                num_epochs=args.epochs,
                learning_rate=args.lr,
                n_train=args.n_train,
                n_val=args.n_val,
                backend=args.backend,
                window_size=args.window_size,
                allow_flash2=not args.no_flash2,
                latent_dim=args.latent_dim,
                latent_update=args.latent_update,
            )

        train_toroidal_attention(config)


def run_evaluation(args):
    """Run evaluation."""
    from scripts.evaluate import comprehensive_evaluation

    comprehensive_evaluation(
        checkpoint_path=Path(args.checkpoint),
        config_path=Path(args.config),
        output_dir=Path(args.output),
    )


def run_demo():
    """Run interactive demo."""
    from scripts.load_phi2 import main as phi2_demo

    print("Running Phi-2 integration demo...")
    phi2_demo()


def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Toroidal Attention Mechanism for LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run unit tests
  python main.py test

  # Train with periodic data
  python main.py train --dataset periodic --epochs 10

  # Run ablation study
  python main.py train --ablation

  # Evaluate a trained model
  python main.py eval --checkpoint checkpoints/best_model.pt --config checkpoints/config.json

  # Run demo
  python main.py demo
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Test command
    test_parser = subparsers.add_parser('test', help='Run test suite')
    test_parser.add_argument('--comprehensive', action='store_true',
                             help='Run comprehensive test suite (all categories)')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train toroidal attention')
    train_parser.add_argument('--config', type=str, default=None, help='Path to YAML training config')
    train_parser.add_argument('--depth', type=int, default=4, help='Number of depth platters')
    train_parser.add_argument('--fusion_mode', type=str, default='low_rank',
                             choices=['low_rank', 'attention', 'mean'],
                             help='Depth fusion mode')
    train_parser.add_argument('--dataset', type=str, default='periodic',
                             choices=['periodic', 'sinusoidal', 'openwebtext'],
                             help='Dataset type')
    train_parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    train_parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    train_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    train_parser.add_argument('--n_train', type=int, default=1000, help='Training samples')
    train_parser.add_argument('--n_val', type=int, default=200, help='Validation samples')
    train_parser.add_argument('--ablation', action='store_true', help='Run ablation study')
    # Backend/options
    train_parser.add_argument('--backend', type=str, default='sdpa', choices=['sdpa', 'flash2'], help='Attention backend')
    train_parser.add_argument('--window_size', type=int, default=None, help='Sliding window size (None disables)')
    train_parser.add_argument('--no_flash2', action='store_true', help='Disable flash-attn even if available')
    # Latent streaming opts (inference oriented, for completeness here)
    train_parser.add_argument('--latent_dim', type=int, default=None, help='Enable latent streaming with given dimension')
    train_parser.add_argument('--latent_update', type=str, default='gru', choices=['gru', 'linear'], help='Latent update type')

    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate trained model')
    eval_parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    eval_parser.add_argument('--config', type=str, required=True, help='Path to config')
    eval_parser.add_argument('--output', type=str, default='evaluation', help='Output directory')

    # Demo command
    subparsers.add_parser('demo', help='Run interactive demo')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Dispatch to appropriate function
    if args.command == 'test':
        run_tests(comprehensive=args.comprehensive)
    elif args.command == 'train':
        run_training(args)
    elif args.command == 'eval':
        run_evaluation(args)
    elif args.command == 'demo':
        run_demo()


if __name__ == "__main__":
    main()
