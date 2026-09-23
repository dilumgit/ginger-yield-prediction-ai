"""Main Command-Line Entry Point for Ginger Yield Prediction System.

Module: Research Methods and Scientific Writing (IT41012)
Title: An Explainable AI-Based Dynamic In-Season Ginger Yield Prediction System
       for Sri Lankan Farmers.

Usage:
    python main.py --help
    python main.py --inspect
    python main.py --prepare
    python main.py --train
    python main.py --evaluate
    python main.py --dynamic
    python main.py --explain
    python main.py --demo
    python main.py --phase7
    python main.py --ui
    python main.py --test
    python main.py --pipeline
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def print_banner():
    """Print project header banner."""
    print("=" * 80)
    print("  EXPLAINABLE AI-BASED DYNAMIC IN-SEASON GINGER YIELD PREDICTION SYSTEM")
    print("  Research Prototype for Sri Lankan Smallholders | IT41012")
    print("=" * 80)


def run_inspect():
    """Execute Phase 1 exploratory data inspection."""
    from scripts.inspect_data import main as cmd_inspect
    cmd_inspect()


def run_prepare():
    """Execute Phase 2 preprocessing and feature engineering."""
    from scripts.prepare_data import main as cmd_prepare
    cmd_prepare()


def run_train():
    """Execute Phase 3 model training, tuning, and benchmarking."""
    from scripts.train import main as cmd_train
    cmd_train()


def run_evaluate():
    """Execute Phase 3 evaluation of serialized best model."""
    from scripts.evaluate import main as cmd_eval
    cmd_eval()


def run_dynamic():
    """Execute Phase 4 dynamic in-season checkpoint evaluation."""
    from scripts.evaluate_dynamic_prediction import main as cmd_dyn
    cmd_dyn()


def run_simulate():
    """Execute Phase 4 dynamic simulation of in-season cultivation updates."""
    from scripts.simulate_dynamic_prediction import main as cmd_sim
    cmd_sim()


def run_explain():
    """Execute Phase 5 TreeSHAP explainability and farmer translation pipeline."""
    from scripts.explain_predictions import main as cmd_explain
    cmd_explain()


def run_demo():
    """Execute Phase 6 end-to-end system integration demonstration."""
    from scripts.demo_end_to_end import main as cmd_demo
    cmd_demo()


def run_phase7():
    """Execute Phase 7 final research evaluation and scientific evidence generation."""
    from scripts.run_phase7_evaluation import main as cmd_phase7
    cmd_phase7()


def run_ui():
    """Launch the interactive Streamlit web dashboard."""
    app_path = PROJECT_ROOT / "src" / "ui" / "app.py"
    print(f"Launching Streamlit Web Prototype from: {app_path.resolve()}")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nStreamlit application stopped by user.")


def run_tests():
    """Execute the automated test suite."""
    print("Executing automated test suite (pytest)...")
    cmd = [sys.executable, "-m", "pytest", "-v"]
    res = subprocess.run(cmd)
    sys.exit(res.returncode)


def run_full_pipeline():
    """Execute complete end-to-end research reproduction pipeline."""
    print_banner()
    print("\n>>> Running Full End-to-End Research Pipeline...\n")
    print("[Step 1/6] Data Inspection & Quality Analysis (Phase 1)...")
    run_inspect()
    print("\n[Step 2/6] Preprocessing & In-Season Feature Engineering (Phase 2)...")
    run_prepare()
    print("\n[Step 3/6] Model Training & Cross-Validation Benchmarking (Phase 3)...")
    run_train()
    print("\n[Step 4/6] Dynamic Developmental Checkpoint Evaluation (Phase 4)...")
    run_dynamic()
    print("\n[Step 5/6] Explainable AI (TreeSHAP) & Farmer Translations (Phase 5)...")
    run_explain()
    print("\n[Step 6/6] Final Research Validation & Statistical Testing (Phase 7)...")
    run_phase7()
    print("\n" + "=" * 80)
    print("  FULL RESEARCH PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Explainable AI-Based Dynamic In-Season Ginger Yield Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --inspect      Run data inspection and quality checks
  python main.py --prepare      Preprocess dataset and engineer features
  python main.py --train        Train and tune all candidate models
  python main.py --evaluate     Evaluate saved best model on hold-out test set
  python main.py --dynamic      Evaluate accuracy across in-season stage horizons
  python main.py --simulate     Simulate continuous farm update journey
  python main.py --explain      Generate SHAP explanations and farmer narratives
  python main.py --demo         Run end-to-end integration demo (Phase 6)
  python main.py --phase7       Run master scientific validation suite (Phase 7)
  python main.py --ui           Launch interactive Streamlit web dashboard
  python main.py --test         Run automated pytest test suite (110 tests)
  python main.py --pipeline     Run complete reproduction pipeline (Phases 1-7)
        """,
    )

    group = parser.add_argument_group("Pipeline Execution Commands")
    group.add_argument("--inspect", action="store_true", help="Execute Phase 1 data inspection and exploratory analysis")
    group.add_argument("--prepare", action="store_true", help="Execute Phase 2 feature engineering and dataset preprocessing")
    group.add_argument("--train", action="store_true", help="Execute Phase 3 model training, tuning, and benchmark generation")
    group.add_argument("--evaluate", action="store_true", help="Evaluate serialized best model on hold-out test set")
    group.add_argument("--dynamic", action="store_true", help="Execute Phase 4 in-season developmental checkpoint evaluation")
    group.add_argument("--simulate", action="store_true", help="Simulate longitudinal in-season crop update trajectory")
    group.add_argument("--explain", action="store_true", help="Execute Phase 5 TreeSHAP attribution and farmer explanation layer")
    group.add_argument("--demo", action="store_true", help="Execute Phase 6 end-to-end integration demo")
    group.add_argument("--phase7", "--validate", action="store_true", help="Execute Phase 7 scientific evaluation and statistical tests")
    group.add_argument("--ui", action="store_true", help="Launch interactive Streamlit web interface")
    group.add_argument("--test", action="store_true", help="Run automated pytest test suite")
    group.add_argument("--pipeline", action="store_true", help="Execute complete end-to-end research reproduction pipeline")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # If no flags passed, print banner and help
    if not any(vars(args).values()):
        print_banner()
        print("\nNo command flag specified. Below is the usage overview:\n")
        parser.print_help()
        print("\nTip: Run 'python main.py --ui' to launch the web dashboard, or 'python main.py --demo' for a quick demo.")
        return

    if args.inspect:
        print_banner()
        run_inspect()
    elif args.prepare:
        print_banner()
        run_prepare()
    elif args.train:
        print_banner()
        run_train()
    elif args.evaluate:
        print_banner()
        run_evaluate()
    elif args.dynamic:
        print_banner()
        run_dynamic()
    elif args.simulate:
        print_banner()
        run_simulate()
    elif args.explain:
        print_banner()
        run_explain()
    elif args.demo:
        print_banner()
        run_demo()
    elif args.phase7:
        print_banner()
        run_phase7()
    elif args.ui:
        run_ui()
    elif args.test:
        run_tests()
    elif args.pipeline:
        run_full_pipeline()


if __name__ == "__main__":
    main()
