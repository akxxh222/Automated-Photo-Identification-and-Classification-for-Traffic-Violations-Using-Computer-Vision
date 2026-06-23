#!/usr/bin/env python3
"""
End-to-end orchestration script for the complete traffic enforcement platform.
Guides users through setup, training, and deployment.
"""

import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrafficEnforcementOrchestrator:
    """Orchestrates the complete platform pipeline."""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.steps = []
    
    def log_section(self, title, level="INFO"):
        """Log a formatted section header."""
        width = 80
        logger.log(logging.getLevelName(level), "=" * width)
        logger.log(logging.getLevelName(level), title.center(width))
        logger.log(logging.getLevelName(level), "=" * width)
    
    def run_command(self, cmd, description, critical=False):
        """Run a command and handle errors."""
        logger.info(f"\n► {description}")
        logger.info(f"  Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
        
        try:
            result = subprocess.run(cmd, capture_output=False, check=False)
            if result.returncode != 0 and critical:
                logger.error(f"✗ CRITICAL ERROR: {description} failed")
                return False
            elif result.returncode != 0:
                logger.warning(f"⚠ WARNING: {description} had issues (non-critical)")
            else:
                logger.info(f"✓ {description} completed")
            return result.returncode == 0
        except Exception as e:
            logger.error(f"✗ Error running: {description} - {e}")
            return not critical
    
    def step_dependencies(self):
        """Step 1: Install dependencies."""
        self.log_section("STEP 1: ENVIRONMENT SETUP")
        logger.info("Installing Python dependencies...")
        
        return self.run_command(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            "Pip dependency installation",
            critical=True
        )
    
    def step_prepare_datasets(self):
        """Step 2: Extract and prepare datasets."""
        self.log_section("STEP 2: DATASET PREPARATION")
        logger.info("Extracting and organizing training datasets...")
        
        return self.run_command(
            [sys.executable, "scripts/prepare_datasets.py"],
            "Dataset extraction and preparation",
            critical=True
        )
    
    def step_train_models(self, skip_forecasting=False):
        """Step 3: Train all models."""
        self.log_section("STEP 3: MODEL TRAINING")
        logger.info("Training YOLO detection models and forecasters...")
        logger.info("This may take 30-120 minutes depending on GPU")
        
        cmd = [sys.executable, "scripts/train_enhanced.py"]
        if skip_forecasting:
            cmd.append("--skip-forecasting")
        
        return self.run_command(cmd, "Enhanced model training", critical=False)
    
    def step_evaluation(self):
        """Step 4: Evaluate trained models."""
        self.log_section("STEP 4: MODEL EVALUATION")
        logger.info("Evaluating model performance on test sets...")
        
        return self.run_command(
            [sys.executable, "scripts/evaluate_all.py"],
            "Model evaluation",
            critical=False
        )
    
    def step_demo(self):
        """Step 5: Run demo."""
        self.log_section("STEP 5: DEMO RUN")
        logger.info("Running inference demo on sample images...")
        
        return self.run_command(
            [sys.executable, "scripts/run_demo.py"],
            "Demo inference run",
            critical=False
        )
    
    def step_api(self):
        """Step 6: Start API server."""
        self.log_section("STEP 6: API SERVER")
        logger.info("Starting FastAPI server at http://localhost:8000...")
        logger.info("Press Ctrl+C to stop the server")
        
        return self.run_command(
            [sys.executable, "-m", "uvicorn", "src.api.app:app", "--reload", "--port", "8000"],
            "FastAPI server startup",
            critical=False
        )
    
    def step_dashboard(self):
        """Step 7: Start dashboard."""
        self.log_section("STEP 7: COMMAND CENTER DASHBOARD")
        logger.info("Starting Streamlit dashboard at http://localhost:8501...")
        logger.info("Press Ctrl+C to stop the dashboard")
        
        return self.run_command(
            [sys.executable, "-m", "streamlit", "run", "app/app.py"],
            "Streamlit dashboard startup",
            critical=False
        )
    
    def run_full_pipeline(self):
        """Run the complete pipeline."""
        self.log_section("🚀 TRAFFIC ENFORCEMENT PLATFORM - FULL SETUP", "WARNING")
        logger.info("This will:")
        logger.info("  1. Install dependencies")
        logger.info("  2. Extract and prepare datasets")
        logger.info("  3. Train all models (may take 1-2 hours)")
        logger.info("  4. Evaluate model performance")
        logger.info("  5. Run a demo inference")
        logger.info("  6. Start the API server")
        logger.info("  7. Start the command center dashboard")
        
        user_input = input("\nProceed? (yes/no): ").strip().lower()
        if user_input not in ["yes", "y"]:
            logger.info("Setup cancelled.")
            return False
        
        # Step 1
        if not self.step_dependencies():
            logger.error("Dependency installation failed. Cannot proceed.")
            return False
        
        # Step 2
        if not self.step_prepare_datasets():
            logger.warning("Dataset preparation had issues. Continuing anyway...")
        
        # Step 3
        if not self.step_train_models():
            logger.warning("Model training failed or incomplete.")
        
        # Step 4
        self.step_evaluation()
        
        # Step 5
        self.step_demo()
        
        return True
    
    def run_quick_setup(self):
        """Run quick setup (deps + datasets)."""
        self.log_section("⚡ QUICK SETUP", "WARNING")
        
        if not self.step_dependencies():
            return False
        
        if not self.step_prepare_datasets():
            return False
        
        logger.info("\n✓ Quick setup complete!")
        logger.info("Next steps:")
        logger.info("  - Run full training: python scripts/train_enhanced.py")
        logger.info("  - Or download pretrained weights and run: python scripts/run_demo.py")
        
        return True
    
    def run_training_only(self):
        """Run only model training."""
        self.log_section("🤖 MODEL TRAINING ONLY", "WARNING")
        return self.step_train_models()
    
    def run_eval_and_demo(self):
        """Run evaluation and demo."""
        self.log_section("📊 EVALUATION & DEMO", "WARNING")
        self.step_evaluation()
        self.step_demo()
        return True


def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Traffic Enforcement Platform - Orchestration CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full setup and training (1-2 hours)
  python scripts/orchestrate.py --full
  
  # Quick setup (deps + datasets only)
  python scripts/orchestrate.py --quick
  
  # Train models only
  python scripts/orchestrate.py --train
  
  # Evaluation and demo
  python scripts/orchestrate.py --eval-demo
  
  # Start API server
  python scripts/orchestrate.py --api
  
  # Start dashboard
  python scripts/orchestrate.py --dashboard
        """
    )
    
    parser.add_argument("--full", action="store_true", 
                        help="Run complete setup pipeline")
    parser.add_argument("--quick", action="store_true",
                        help="Quick setup (deps + datasets only)")
    parser.add_argument("--train", action="store_true",
                        help="Train models only")
    parser.add_argument("--eval-demo", action="store_true",
                        help="Run evaluation and demo")
    parser.add_argument("--api", action="store_true",
                        help="Start API server")
    parser.add_argument("--dashboard", action="store_true",
                        help="Start command center dashboard")
    
    args = parser.parse_args()
    
    orchestrator = TrafficEnforcementOrchestrator()
    
    # If no argument, show interactive menu
    if not any([args.full, args.quick, args.train, args.eval_demo, args.api, args.dashboard]):
        logger.info("\n" + "="*60)
        logger.info("TRAFFIC ENFORCEMENT PLATFORM")
        logger.info("="*60)
        logger.info("\nAvailable options:")
        logger.info("  1. Full setup (setup → train → evaluate → demo)")
        logger.info("  2. Quick setup (setup only)")
        logger.info("  3. Train models")
        logger.info("  4. Evaluation & Demo")
        logger.info("  5. Start API Server")
        logger.info("  6. Start Dashboard")
        logger.info("  0. Exit")
        
        choice = input("\nSelect option (0-6): ").strip()
        
        if choice == "1":
            orchestrator.run_full_pipeline()
        elif choice == "2":
            orchestrator.run_quick_setup()
        elif choice == "3":
            orchestrator.run_training_only()
        elif choice == "4":
            orchestrator.run_eval_and_demo()
        elif choice == "5":
            orchestrator.step_api()
        elif choice == "6":
            orchestrator.step_dashboard()
        else:
            logger.info("Exiting.")
            return 0
    
    # Command-line arguments
    if args.full:
        orchestrator.run_full_pipeline()
    elif args.quick:
        orchestrator.run_quick_setup()
    elif args.train:
        orchestrator.run_training_only()
    elif args.eval_demo:
        orchestrator.run_eval_and_demo()
    elif args.api:
        orchestrator.step_api()
    elif args.dashboard:
        orchestrator.step_dashboard()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
