import argparse
import os
import sys

def setup_wandb_api_key():
    """Loads a locally saved W&B API key from .env, or prompts and saves it."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bori_root = os.path.dirname(os.path.dirname(script_dir)) # /Users/brandon.baek/Development/Bori
    env_path = os.path.join(bori_root, ".env")
    
    # 1. Try to load from existing .env
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                if line.startswith("WANDB_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    os.environ["WANDB_API_KEY"] = key
                    print("🔑 Loaded W&B API Key from local .env config.")
                    return True

    # 2. If not authenticated already, prompt and save
    try:
        import wandb
    except ImportError:
        print("❌ Error: 'wandb' package is not installed. Please run: pip install wandb")
        sys.exit(1)

    try:
        api = wandb.Api()
        api.viewer
        return True
    except Exception:
        pass

    if not os.environ.get("WANDB_API_KEY"):
        print("\n🔑 W&B Authentication Required.")
        user_key = input("👉 Please paste your W&B API Key (it will be saved locally in .env): ").strip()
        if user_key:
            os.environ["WANDB_API_KEY"] = user_key
            with open(env_path, "w") as f:
                f.write(f"WANDB_API_KEY={user_key}\n")
            print(f"💾 Saved API key to {env_path} (Git-ignored) for future runs.")
            return True
        else:
            print("❌ Error: W&B API Key required to upload checkpoints.")
            sys.exit(1)
    return True

def main():
    parser = argparse.ArgumentParser(description="Upload Local Model Checkpoints to W&B Artifacts")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Local directory containing the model files")
    parser.add_argument("--project", type=str, default="Bori-V2", help="W&B project name")
    parser.add_argument("--entity", type=str, default="brandon_baek", help="W&B username/entity")
    parser.add_argument("--artifact_name", type=str, default="bori-2-135m-sft", help="W&B artifact name")
    parser.add_argument("--alias", type=str, default=None, help="Specific version alias to assign (e.g. 'step-400')")
    args = parser.parse_args()

    print("======================================================================")
    print("🌾 Bori (보리): Checkpoint Uploader to W&B Artifacts")
    print("======================================================================\n")

    # W&B API Key setup
    setup_wandb_api_key()

    # 1. Interactive input fallbacks
    checkpoint_dir = args.checkpoint_dir
    if not checkpoint_dir:
        checkpoint_dir = input("💬 Path to local checkpoint folder (e.g. ./sft_checkpoints/checkpoint-400): ").strip()

    # Validate local directory
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Error: Local directory '{checkpoint_dir}' does not exist.")
        return
    if not os.path.isdir(checkpoint_dir):
        print(f"❌ Error: Path '{checkpoint_dir}' is not a directory.")
        return

    # Check if folder is actually a HF model
    required_files = ["config.json", "model.safetensors", "pytorch_model.bin"]
    has_model_file = any(os.path.exists(os.path.join(checkpoint_dir, f)) for f in required_files)
    if not has_model_file:
        print(f"⚠️ Warning: '{checkpoint_dir}' does not appear to contain standard model weights (config.json, model.safetensors).")
        proceed = input("❓ Do you still want to upload this folder? [y/N]: ").strip().lower()
        if proceed not in ["y", "yes"]:
            return

    project = args.project
    if not args.checkpoint_dir:
        user_input = input(f"💬 Enter W&B Project [Default: {project}]: ").strip()
        project = user_input if user_input else project

    artifact_name = args.artifact_name
    if not args.checkpoint_dir:
        user_input = input(f"💬 Enter W&B Artifact Name [Default: {artifact_name}]: ").strip()
        artifact_name = user_input if user_input else artifact_name

    alias = args.alias
    if not alias:
        # Try to automatically extract step number from folder name (e.g., 'checkpoint-400' -> 'step-400')
        folder_basename = os.path.basename(os.path.normpath(checkpoint_dir))
        default_alias = "latest"
        if "checkpoint-" in folder_basename:
            step_num = folder_basename.split("-")[-1]
            if step_num.isdigit():
                default_alias = f"step-{step_num}"
                
        user_input = input(f"💬 Enter version alias [Default: {default_alias}]: ").strip()
        alias = user_input if user_input else default_alias

    # 2. Perform upload
    import wandb
    print(f"\n🚀 Connecting to W&B (Project: {project}, Entity: {args.entity})...")
    
    try:
        # Launch a lightweight uploading run
        run = wandb.init(
            project=project,
            entity=args.entity,
            job_type="upload_checkpoint",
            group="Local-Uploads",
            tags=["local", "upload"],
            name=f"upload-{artifact_name}-{alias}"
        )
        
        print(f"📦 Creating W&B Artifact: {artifact_name} (Type: model)...")
        artifact = wandb.Artifact(name=artifact_name, type="model")
        
        print(f"📂 Adding local directory: {checkpoint_dir} ...")
        artifact.add_dir(checkpoint_dir)
        
        print(f"📤 Uploading checkpoint folder to W&B...")
        # Assign both the user-selected alias and the default 'latest' tag
        run.log_artifact(artifact, aliases=[alias, "latest"])
        
        print(f"🏁 Finalizing upload run...")
        run.finish()
        
        print("\n" + "="*70)
        print(f"🎉 Success! Checkpoint uploaded to W&B Artifacts!")
        print(f"👉 Full Path: {args.entity}/{project}/{artifact_name}:{alias}")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"❌ Failed to upload checkpoint: {e}")
        print("💡 Tip: Verify your internet connection, credentials, and folder permissions.")

if __name__ == "__main__":
    main()
