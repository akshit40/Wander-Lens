import kagglehub
import os
import shutil

def setup_dataset():
    """
    Downloads the Kaggle dataset and copies the pre-split train/test
    directories directly into the project's data folder using the
    correct, verified folder structure.
    """
    print("--- 📥 Checking/Downloading dataset from Kaggle Hub... ---")
    dataset_path = kagglehub.dataset_download("danushkumarv/indian-monuments-image-dataset")
    print(f"✅ Dataset is available at: {dataset_path}")

    # --- Configuration ---
    base_dir = os.getcwd()
    destination_data_dir = os.path.join(base_dir, 'data')

    # --- THIS IS THE FINAL, CORRECTED PATH based on the inspector output ---
    source_train_dir = os.path.join(dataset_path, 'Indian-monuments', 'images', 'train')
    source_test_dir = os.path.join(dataset_path, 'Indian-monuments', 'images', 'test')

    # Define the destination train/test folders for our project
    destination_train_dir = os.path.join(destination_data_dir, 'train')
    destination_test_dir = os.path.join(destination_data_dir, 'test')

    print("\n--- 📂 Copying pre-split dataset... ---")

    # Check if source directories exist before trying to copy
    if not os.path.exists(source_train_dir) or not os.path.exists(source_test_dir):
        print(f"❌ Error: Could not find the 'train' and 'test' folders in the downloaded dataset.")
        print(f"Looked for train folder at: {source_train_dir}")
        print(f"Looked for test folder at: {source_test_dir}")
        return

    # Use shutil.copytree to copy the entire directory trees
    print(f"Copying from {source_train_dir}...")
    shutil.copytree(source_train_dir, destination_train_dir)

    print(f"Copying from {source_test_dir}...")
    shutil.copytree(source_test_dir, destination_test_dir)

    print("\n--- ✨ Dataset setup complete! ---")

if __name__ == '__main__':
    setup_dataset()