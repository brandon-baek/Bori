import os
import argparse

def upload_kaggle_checkpoint(checkpoint_dir, artifact_name="bori-2-135m-sft", project="Bori-V2", entity="brandon_baek", alias="latest"):
    """
    Self-contained function to upload Kaggle checkpoint directories to W&B.
    Copy-paste this directly into a Kaggle Notebook cell.
    """
    try:
        import wandb
    except ImportError:
        print("📥 Installing wandb package...")
        os.system("pip install wandb -q")
        import wandb

    if not os.path.exists(checkpoint_dir):
        print(f"❌ Error: Checkpoint directory '{checkpoint_dir}' does not exist.")
        return

    # Auto-extract step if present in directory name (e.g. 'checkpoint-400' -> 'step-400')
    folder_name = os.path.basename(os.path.normpath(checkpoint_dir))
    if "checkpoint-" in folder_name:
        step_num = folder_name.split("-")[-1]
        if step_num.isdigit() and alias == "latest":
            alias = f"step-{step_num}"

    # Check if running in Kaggle and load Secret if available
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.exists("/kaggle/input"):
        try:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            os.environ["WANDB_API_KEY"] = user_secrets.get_secret("WANDB_API_KEY")
            print("🔑 Authenticated silently using Kaggle Secret 'WANDB_API_KEY'.")
        except Exception:
            pass

    print(f"\n🚀 Connecting to W&B (Project: {project}, Entity: {entity})...")
    
    # Check if already authenticated; if not, invoke login
    try:
        api = wandb.Api()
        api.viewer
    except Exception:
        print("🔑 Authentication required. Please log in to W&B:")
        wandb.login()

    try:
        # Start a lightweight upload run
        run = wandb.init(
            project=project,
            entity=entity,
            job_type="kaggle_upload",
            name=f"kaggle-{artifact_name}-{alias}"
        )
        
        print(f"📦 Creating W&B Artifact: {artifact_name} (Type: model)...")
        artifact = wandb.Artifact(name=artifact_name, type="model")
        
        print(f"📂 Adding files from directory: {checkpoint_dir} ...")
        artifact.add_dir(checkpoint_dir)
        
        print(f"📤 Uploading checkpoint from Kaggle to W&B...")
        run.log_artifact(artifact, aliases=[alias, "latest"])
        
        print("🏁 Wrapping up run...")
        run.finish()
        
        print("\n" + "="*70)
        print(f"🎉 Success! Checkpoint uploaded from Kaggle to W&B Artifacts!")
        print(f"👉 Full Path: {entity}/{project}/{artifact_name}:{alias}")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"❌ Failed to upload checkpoint from Kaggle: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Kaggle Checkpoint Uploader to W&B")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to checkpoint folder in Kaggle")
    parser.add_argument("--project", type=str, default="Bori-V2", help="W&B project name")
    parser.add_argument("--entity", type=str, default="brandon_baek", help="W&B username")
    parser.add_argument("--artifact_name", type=str, default="bori-2-135m-sft", help="W&B artifact name")
    parser.add_argument("--alias", type=str, default="latest", help="Version alias")
    args = parser.parse_args()
    
    upload_kaggle_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        artifact_name=args.artifact_name,
        project=args.project,
        entity=args.entity,
        alias=args.alias
    )
