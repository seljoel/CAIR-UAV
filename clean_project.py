import os
import shutil

def clean_dir(dir_path):
    if os.path.exists(dir_path):
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            # Keep .gitkeep if it exists so directories stay in version control
            if item == ".gitkeep":
                continue
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            except Exception as e:
                print(f"Could not remove {item_path}: {e}")
        print(f"Cleaned directory: {dir_path}")

def clean_models():
    for sub_dir in ["cair", "standard", "baselines"]:
        path = os.path.join("models", sub_dir)
        clean_dir(path)
        
def clean_logs():
    clean_dir("logs")
    clean_dir("runs")

def clean_results():
    clean_dir("results")
    # Also clean the plots subdirectory if it got deleted and recreated
    plots_dir = os.path.join("results", "plots")
    if not os.path.exists(plots_dir) and os.path.exists("results"):
        os.makedirs(plots_dir, exist_ok=True)
    clean_dir(plots_dir)

def main():
    print("Purging model weights, training logs, and evaluation results...")
    clean_models()
    clean_logs()
    clean_results()
    print("Cleanup complete! The slate is clean for retraining.")

if __name__ == "__main__":
    main()
