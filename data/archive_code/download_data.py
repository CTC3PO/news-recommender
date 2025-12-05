"""
Download and prepare MIND dataset from Kaggle
Microsoft News Dataset - Small version for faster processing
"""
import os
import requests
import zipfile
import kaggle
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import time
import subprocess
import sys


def setup_kaggle():
    """Setup Kaggle API credentials if not already configured"""
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'

    if kaggle_json.exists():
        print("✓ Kaggle API already configured")
        return True

    print("\n⚠️  Kaggle API not configured.")
    print("To download from Kaggle, you need to:")
    print("1. Go to https://www.kaggle.com/[your-username]/account")
    print("2. Click 'Create New API Token'")
    print("3. Download kaggle.json")
    print("4. Place it in ~/.kaggle/ directory")
    print("\nOr manually download from: https://www.kaggle.com/datasets/arashnic/mind-news-dataset")

    # Ask if user wants to manually enter credentials
    response = input("\nDo you have Kaggle credentials? (yes/no): ").lower()
    if response in ['yes', 'y']:
        print("\nPlease enter your Kaggle credentials:")
        username = input("Username: ").strip()
        key = input("API Key: ").strip()

        # Create kaggle directory
        kaggle_dir.mkdir(exist_ok=True, mode=0o700)

        # Write credentials
        with open(kaggle_json, 'w') as f:
            f.write(f'{{"username":"{username}","key":"{key}"}}')

        # Set permissions
        kaggle_json.chmod(0o600)
        print("✓ Kaggle credentials saved")
        return True

    return False


def download_from_kaggle():
    """Download MIND dataset from Kaggle"""
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Kaggle dataset identifier
    dataset = "arashnic/mind-news-dataset"

    print(f"\n{'='*60}")
    print(f"Downloading MIND dataset from Kaggle...")
    print(f"Dataset: {dataset}")
    print(f"{'='*60}")

    try:
        # Download using Kaggle API
        print("Starting download...")
        kaggle.api.dataset_download_files(
            dataset,
            path=data_dir,
            unzip=True,
            quiet=False
        )

        # Check what was downloaded
        downloaded_files = list(data_dir.rglob("*"))
        print(f"\nDownloaded {len(downloaded_files)} files/directories")

        # The Kaggle dataset might have a different structure
        # Let's check and reorganize if needed
        reorganize_kaggle_structure(data_dir)

        print(f"✓ Kaggle download complete")

    except Exception as e:
        print(f"✗ Kaggle download failed: {e}")
        print("\nTrying alternative: Download via Kaggle CLI command...")

        # Try using command line
        try:
            cmd = f"kaggle datasets download -d {dataset} -p {data_dir} --unzip"
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True)

            if result.returncode == 0:
                print("✓ Kaggle CLI download successful")
                reorganize_kaggle_structure(data_dir)
            else:
                print(f"✗ Kaggle CLI failed: {result.stderr}")
                raise Exception("Kaggle download failed")

        except Exception as cli_error:
            print(f"✗ All Kaggle methods failed: {cli_error}")
            raise


def reorganize_kaggle_structure(data_dir):
    """Reorganize Kaggle downloaded files into expected structure"""
    print("\nOrganizing downloaded files...")

    # Check what files we have
    all_files = list(data_dir.rglob("*"))

    # Look for train and dev directories or files
    train_dir = data_dir / "train"
    dev_dir = data_dir / "dev"

    # Create directories if they don't exist
    train_dir.mkdir(exist_ok=True)
    dev_dir.mkdir(exist_ok=True)

    # Map possible file patterns to our expected structure
    file_patterns = {
        'news.tsv': ['news.tsv', 'news.csv', 'news.parquet', 'news.*'],
        'behaviors.tsv': ['behaviors.tsv', 'behaviors.csv', 'behaviors.*', 'user-behaviors.*']
    }

    moved_count = 0

    # Look for train files
    for file in all_files:
        if file.is_file():
            filename = file.name.lower()

            # Check if it's a train file
            if 'train' in filename or 'small_train' in filename:
                if 'news' in filename:
                    new_path = train_dir / "news.tsv"
                    file.rename(new_path)
                    print(f"  Moved: {file.name} → train/news.tsv")
                    moved_count += 1
                elif 'behavior' in filename or 'behaviour' in filename:
                    new_path = train_dir / "behaviors.tsv"
                    file.rename(new_path)
                    print(f"  Moved: {file.name} → train/behaviors.tsv")
                    moved_count += 1

            # Check if it's a dev/validation file
            elif 'dev' in filename or 'val' in filename or 'small_dev' in filename:
                if 'news' in filename:
                    new_path = dev_dir / "news.tsv"
                    file.rename(new_path)
                    print(f"  Moved: {file.name} → dev/news.tsv")
                    moved_count += 1
                elif 'behavior' in filename or 'behaviour' in filename:
                    new_path = dev_dir / "behaviors.tsv"
                    file.rename(new_path)
                    print(f"  Moved: {file.name} → dev/behaviors.tsv")
                    moved_count += 1

    print(f"✓ Organized {moved_count} files")

    # Check if we have the required files
    required_files = [
        train_dir / "news.tsv",
        train_dir / "behaviors.tsv",
        dev_dir / "news.tsv",
        dev_dir / "behaviors.tsv"
    ]

    missing_files = [f for f in required_files if not f.exists()]

    if missing_files:
        print(f"\n⚠️  Missing files: {[f.name for f in missing_files]}")
        print("Checking for alternative formats...")

        # Try to find alternative formats
        for missing in missing_files:
            parent = missing.parent
            alt_files = list(parent.glob("*"))
            if alt_files:
                print(f"  Found in {parent}: {[f.name for f in alt_files]}")

                # If it's a different format, try to convert
                for alt_file in alt_files:
                    if alt_file.suffix in ['.csv', '.parquet', '.json']:
                        try:
                            convert_to_tsv(alt_file, missing)
                            print(
                                f"  ✓ Converted {alt_file.name} to {missing.name}")
                        except Exception as e:
                            print(
                                f"  ✗ Failed to convert {alt_file.name}: {e}")


def convert_to_tsv(source_file, target_file):
    """Convert different file formats to TSV"""
    if source_file.suffix == '.csv':
        df = pd.read_csv(source_file)
        df.to_csv(target_file, sep='\t', index=False)
    elif source_file.suffix == '.parquet':
        df = pd.read_parquet(source_file)
        df.to_csv(target_file, sep='\t', index=False)
    elif source_file.suffix == '.json':
        df = pd.read_json(source_file)
        df.to_csv(target_file, sep='\t', index=False)


def download_file(url, destination, max_retries=3):
    """Fallback: Direct download with progress bar"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(destination, 'wb') as f, tqdm(
                desc=destination.name,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        size = f.write(chunk)
                        pbar.update(size)

            return True

        except requests.exceptions.RequestException as e:
            print(f"Download error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return False

    return False


def fallback_download():
    """Fallback to direct download if Kaggle fails"""
    print("\nTrying fallback download from alternative sources...")

    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Alternative sources (direct links)
    urls = {
        "train_news": "https://raw.githubusercontent.com/msnews/MIND/master/download/MINDsmall_train/news.tsv",
        "train_behaviors": "https://raw.githubusercontent.com/msnews/MIND/master/download/MINDsmall_train/behaviors.tsv",
        "dev_news": "https://raw.githubusercontent.com/msnews/MIND/master/download/MINDsmall_dev/news.tsv",
        "dev_behaviors": "https://raw.githubusercontent.com/msnews/MIND/master/download/MINDsmall_dev/behaviors.tsv"
    }

    for name, url in urls.items():
        parts = name.split("_")
        split = parts[0]
        file_type = parts[1]

        split_dir = data_dir / split
        split_dir.mkdir(exist_ok=True)

        dest_file = split_dir / f"{file_type}.tsv"

        if dest_file.exists():
            print(f"✓ {name} already exists")
            continue

        print(f"Downloading {name}...")
        success = download_file(url, dest_file)

        if not success:
            raise Exception(f"Failed to download {name}")


def load_news(split="train"):
    """Load news articles from TSV file"""
    news_path = Path(f"data/raw/{split}/news.tsv")

    if not news_path.exists():
        # Try to find the file with different extensions
        alt_files = list(news_path.parent.glob("news.*"))
        if alt_files:
            print(f"Found alternative file: {alt_files[0]}")
            return load_news_file(alt_files[0])
        raise FileNotFoundError(f"News file not found: {news_path}")

    return load_news_file(news_path)


def load_news_file(file_path):
    """Load news articles from file (handles different formats)"""
    print(f"\nLoading news from {file_path}...")

    # Column names from MIND dataset documentation
    columns = [
        'news_id', 'category', 'subcategory', 'title',
        'abstract', 'url', 'title_entities', 'abstract_entities'
    ]

    # Read based on file extension
    if file_path.suffix == '.tsv':
        df = pd.read_csv(file_path, sep='\t', header=None,
                         names=columns, on_bad_lines='warn')
    elif file_path.suffix == '.csv':
        df = pd.read_csv(file_path, header=None,
                         names=columns, on_bad_lines='warn')
    else:
        # Try to infer format
        try:
            df = pd.read_csv(file_path, sep='\t', header=None,
                             names=columns, on_bad_lines='warn')
        except:
            df = pd.read_csv(file_path, header=None,
                             names=columns, on_bad_lines='warn')

    # Create combined text field for embedding
    df['text'] = df['title'] + " " + df['abstract'].fillna("")

    # Keep only necessary columns
    df = df[['news_id', 'category', 'subcategory', 'title', 'abstract', 'text']]

    print(f"✓ Loaded {len(df):,} news articles")
    return df


def load_behaviors(split="train"):
    """Load user behavior data"""
    behavior_path = Path(f"data/raw/{split}/behaviors.tsv")

    if not behavior_path.exists():
        # Try to find the file with different extensions
        alt_files = list(behavior_path.parent.glob("behaviors.*"))
        if alt_files:
            print(f"Found alternative file: {alt_files[0]}")
            return load_behaviors_file(alt_files[0])
        raise FileNotFoundError(f"Behavior file not found: {behavior_path}")

    return load_behaviors_file(behavior_path)


def load_behaviors_file(file_path):
    """Load behaviors from file (handles different formats)"""
    print(f"\nLoading behaviors from {file_path}...")

    columns = [
        'impression_id', 'user_id', 'time', 'history', 'impressions'
    ]

    # Read based on file extension
    if file_path.suffix == '.tsv':
        df = pd.read_csv(file_path, sep='\t', header=None,
                         names=columns, on_bad_lines='warn')
    elif file_path.suffix == '.csv':
        df = pd.read_csv(file_path, header=None,
                         names=columns, on_bad_lines='warn')
    else:
        # Try to infer format
        try:
            df = pd.read_csv(file_path, sep='\t', header=None,
                             names=columns, on_bad_lines='warn')
        except:
            df = pd.read_csv(file_path, header=None,
                             names=columns, on_bad_lines='warn')

    print(f"✓ Loaded {len(df):,} user behaviors")
    return df


def display_sample_data(news_df, behaviors_df):
    """Display sample data for verification"""
    print("\n" + "="*60)
    print("SAMPLE DATA")
    print("="*60)

    print("\n📰 Sample News Article:")
    sample = news_df.iloc[0]
    print(f"  ID: {sample['news_id']}")
    print(f"  Category: {sample['category']}")
    print(f"  Title: {sample['title'][:80]}...")
    print(
        f"  Abstract: {sample['abstract'][:100] if sample['abstract'] else 'N/A'}...")

    print("\n👤 Sample User Behavior:")
    sample_behavior = behaviors_df.iloc[0]
    print(f"  User ID: {sample_behavior['user_id']}")
    print(f"  Time: {sample_behavior['time']}")
    if pd.notna(sample_behavior['history']):
        history = sample_behavior['history'].split()
        print(f"  History: {len(history)} articles read")
        print(f"  Sample IDs: {' '.join(history[:3])}...")

    print("\n📊 Dataset Statistics:")
    print(f"  Total Articles: {len(news_df):,}")
    print(f"  Total Behaviors: {len(behaviors_df):,}")
    print(f"  Unique Users: {behaviors_df['user_id'].nunique():,}")
    print(f"  Categories: {news_df['category'].nunique()}")
    print(
        f"  Top Categories: {', '.join(news_df['category'].value_counts().head(5).index.tolist())}")


if __name__ == "__main__":
    print("="*60)
    print("MIND DATASET DOWNLOADER (Kaggle Version)")
    print("="*60)

    try:
        # Try Kaggle first
        print("\nAttempting to download from Kaggle...")

        # Setup Kaggle API if needed
        if not setup_kaggle():
            print("\n⚠️  Kaggle API not configured.")
            print("Please configure Kaggle or use fallback method.")
            use_kaggle = input("Use Kaggle API? (yes/no): ").lower()
            if use_kaggle not in ['yes', 'y']:
                raise Exception("Kaggle API not configured")

        # Download from Kaggle
        download_from_kaggle()

    except Exception as e:
        print(f"\n❌ Kaggle download failed: {e}")
        print("\nTrying fallback download from GitHub...")
        fallback_download()

    # Load and display
    news_df = load_news("train")
    behaviors_df = load_behaviors("train")

    display_sample_data(news_df, behaviors_df)

    print("\n✅ Data preparation complete!")
    print("\nNext step: Run feature engineering")
    print("  → python data/feature_engineering.py")
