import wandb

def delete_checkpoint_folders(entity: str, project: str):
    api = wandb.Api()

    print(f"Fetching runs for {entity}/{project}...")
    runs = api.runs(f"{entity}/{project}")

    for run in runs:
        print(f"\nChecking run: {run.id} ({run.name})")

        # Fetch all files in the run (top-level only)
        files = run.files()

        # Find folder named "checkpoints"
        checkpoint_folders = [f for f in files if "checkpoints" in f.name]

        if not checkpoint_folders:
            print("  No checkpoints folder found.")
            continue

        for folder in checkpoint_folders:
            print(f"  Deleting folder: {folder.name}")
            try:
                folder.delete()
                print("  Deleted successfully.")
            except Exception as e:
                print(f"  Failed to delete folder: {e}")


if __name__ == "__main__":
    # 🔧 UPDATE THESE
    ENTITY = "182-research-project"
    PROJECT = "omniglot-pretraining"

    delete_checkpoint_folders(ENTITY, PROJECT)
