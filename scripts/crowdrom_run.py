#!/usr/bin/env python
"""
CrowdROM Command-Line Interface.

Usage:
    crowdrom run --config /path/to/config.json --outdir /path/to/output \\
        --make-movies-for 1 --save-dtype float64 --mass-tol 1e-12 \\
        --adf-alpha 0.01 --movie-fps 20 --movie-max-frames 500 --quiet false

Exit codes:
    0: Success
    2: Mass conservation check failed
    3: Invalid configuration
    4: I/O error
"""

import argparse
import json
import sys
from pathlib import Path

from rectsim.crowdrom_runner import CrowdROMRunner


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CrowdROM End-to-End Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit Codes:
  0 - Success
  2 - Mass conservation check failed
  3 - Invalid configuration
  4 - I/O error

Examples:
  # Basic run with default settings
  crowdrom run --config config.json --outdir output/run_001

  # Generate movies for first 3 simulations
  crowdrom run --config config.json --outdir output/run_002 --make-movies-for 1,2,3

  # High precision output, strict mass tolerance
  crowdrom run --config config.json --outdir output/run_003 \\
      --save-dtype float64 --mass-tol 1e-14

  # Disable all movies
  crowdrom run --config config.json --outdir output/run_004 --make-movies-for none
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # 'run' command
    run_parser = subparsers.add_parser("run", help="Execute complete pipeline")
    
    run_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to JSON configuration file"
    )
    
    run_parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Output directory (created if missing)"
    )
    
    run_parser.add_argument(
        "--make-movies-for",
        type=str,
        default="1",
        help="Comma-separated simulation indices (1-based) for movie generation, or 'none' (default: 1)"
    )
    
    run_parser.add_argument(
        "--save-dtype",
        type=str,
        choices=["float32", "float64"],
        default="float64",
        help="CSV float precision (default: float64)"
    )
    
    run_parser.add_argument(
        "--mass-tol",
        type=float,
        default=1e-12,
        help="Mass conservation tolerance (default: 1e-12)"
    )
    
    run_parser.add_argument(
        "--adf-alpha",
        type=float,
        default=0.01,
        help="ADF test significance level (default: 0.01)"
    )
    
    run_parser.add_argument(
        "--adf-max-lags",
        type=str,
        default="auto",
        help="Max lags for ADF test (integer or 'auto' for BIC selection, default: auto)"
    )
    
    run_parser.add_argument(
        "--movie-fps",
        type=int,
        default=20,
        help="Movie frames per second (default: 20)"
    )
    
    run_parser.add_argument(
        "--movie-max-frames",
        type=int,
        default=500,
        help="Maximum frames in movies, subsamples if exceeded (default: 500)"
    )
    
    run_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-error log messages"
    )
    
    return parser.parse_args()


def main():
    """Main CLI entry point."""
    args = parse_args()
    
    if args.command != "run":
        print("Error: Please specify 'run' command", file=sys.stderr)
        sys.exit(3)
    
    # Load configuration
    config_path = args.config
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(3)
    
    try:
        with open(config_path, 'r') as f:
            cfg = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file: {e}", file=sys.stderr)
        sys.exit(3)
    except Exception as e:
        print(f"Error: Failed to load config: {e}", file=sys.stderr)
        sys.exit(4)
    
    # Parse movies_for
    if args.make_movies_for.lower() == "none":
        movies_for = ()
    else:
        try:
            movies_for = tuple(int(x.strip()) for x in args.make_movies_for.split(","))
        except ValueError:
            print(f"Error: Invalid --make-movies-for format: {args.make_movies_for}", file=sys.stderr)
            print("Use comma-separated integers (e.g., '1,2,3') or 'none'", file=sys.stderr)
            sys.exit(3)
    
    # Parse adf_max_lags
    if args.adf_max_lags.lower() == "auto":
        adf_max_lags = None
    else:
        try:
            adf_max_lags = int(args.adf_max_lags)
        except ValueError:
            print(f"Error: Invalid --adf-max-lags: {args.adf_max_lags}", file=sys.stderr)
            print("Must be an integer or 'auto'", file=sys.stderr)
            sys.exit(3)
    
    # Create runner
    runner = CrowdROMRunner(
        cfg=cfg,
        outdir=str(args.outdir),
        mass_tol=args.mass_tol,
        adf_alpha=args.adf_alpha,
        adf_max_lags=adf_max_lags,
        save_dtype=args.save_dtype,
        movie_fps=args.movie_fps,
        movie_max_frames=args.movie_max_frames,
        movies_for=movies_for,
        quiet=args.quiet
    )
    
    # Execute pipeline
    exit_code = runner.run()
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
