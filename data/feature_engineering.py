"""
Feature Engineering for News Recommendation
Creates user profiles and article statistics
"""
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from tqdm import tqdm


class FeatureEngineer:
    """Extract features from raw data"""

    def __init__(self):
        self.user_profiles = []
        self.article_stats = {}

    def process_user_history(self, behaviors_df, news_df):
        """
        Extract user reading patterns and preferences

        For each user, we calculate:
        - Number of articles read
        - Favorite categories
        - Category diversity
        - Recent reading history
        """
        print("\n" + "="*60)
        print("Processing User Histories")
        print("="*60)

        user_features = []
        news_lookup = news_df.set_index('news_id')

        # Group by user for efficiency
        user_groups = behaviors_df.groupby('user_id')

        for user_id, user_data in tqdm(user_groups, desc="Processing users"):
            # Collect all history across impressions
            all_history = []

            for history in user_data['history'].dropna():
                all_history.extend(history.split())

            if len(all_history) == 0:
                continue

            # Remove duplicates while preserving order
            unique_history = list(dict.fromkeys(all_history))

            # Get article details
            try:
                history_articles = news_lookup.loc[
                    [nid for nid in unique_history if nid in news_lookup.index]
                ]
            except KeyError:
                continue

            if len(history_articles) == 0:
                continue

            # Calculate category distribution
            category_counts = Counter(history_articles['category'])
            top_categories = category_counts.most_common(3)

            # Build user profile
            profile = {
                'user_id': user_id,
                'num_articles_read': len(unique_history),
                'unique_categories': len(category_counts),
                'top_category': top_categories[0][0] if top_categories else None,
                'top_category_count': top_categories[0][1] if top_categories else 0,
                'second_category': top_categories[1][0] if len(top_categories) > 1 else None,
                'third_category': top_categories[2][0] if len(top_categories) > 2 else None,
                'category_diversity': len(category_counts) / max(len(unique_history), 1),
                # Last 20 articles
                'recent_history': ','.join(unique_history[-20:])
            }

            user_features.append(profile)

        user_df = pd.DataFrame(user_features)

        print(f"\n- Processed {len(user_df):,} user profiles")
        print(
            f"  Avg articles per user: {user_df['num_articles_read'].mean():.1f}")
        print(
            f"  Avg category diversity: {user_df['category_diversity'].mean():.2f}")

        return user_df

    def process_article_features(self, news_df, behaviors_df):
        """
        Calculate article-level statistics

        For each article, we calculate:
        - Number of times shown (impressions)
        - Number of times clicked
        - Click-through rate (CTR)
        - Popularity score
        """
        print("\n" + "="*60)
        print("Processing Article Features")
        print("="*60)

        # Count impressions and clicks
        article_impressions = Counter()
        article_clicks = Counter()

        print("Analyzing user interactions...")
        for impressions_str in tqdm(behaviors_df['impressions'].dropna(), desc="Processing impressions"):
            for impression in impressions_str.split():
                try:
                    news_id, clicked = impression.split('-')
                    article_impressions[news_id] += 1
                    if clicked == '1':
                        article_clicks[news_id] += 1
                except ValueError:
                    continue  # Skip malformed entries

        # Add statistics to news dataframe
        news_df = news_df.copy()
        news_df['impressions'] = news_df['news_id'].map(
            article_impressions).fillna(0).astype(int)
        news_df['clicks'] = news_df['news_id'].map(
            article_clicks).fillna(0).astype(int)

        # Calculate CTR
        news_df['ctr'] = news_df.apply(
            lambda x: x['clicks'] /
            x['impressions'] if x['impressions'] > 0 else 0,
            axis=1
        )

        # Popularity score (log-scaled impressions)
        news_df['popularity'] = np.log1p(news_df['impressions'])

        # Add binary flag for whether article was ever clicked
        news_df['has_clicks'] = (news_df['clicks'] > 0).astype(int)

        print(f"\n- Article statistics computed")
        print(
            f"  Articles with impressions: {(news_df['impressions'] > 0).sum():,}")
        print(f"  Articles with clicks: {(news_df['clicks'] > 0).sum():,}")
        print(
            f"  Average CTR: {news_df[news_df['impressions'] > 0]['ctr'].mean():.4f}")

        return news_df

    def add_category_features(self, news_df):
        """Add category-level aggregate features"""
        print("\nAdding category features...")

        # Category-level CTR
        category_ctr = news_df.groupby('category')['ctr'].mean().to_dict()
        news_df['category_avg_ctr'] = news_df['category'].map(category_ctr)

        # Category popularity
        category_popularity = news_df.groupby(
            'category')['impressions'].sum().to_dict()
        news_df['category_popularity'] = news_df['category'].map(
            category_popularity)

        return news_df

    def save_features(self, user_df, news_df, output_dir="data/processed"):
        """Save processed features to parquet files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*60)
        print("Saving Features")
        print("="*60)

        # Save user features
        user_path = output_path / "user_features.parquet"
        user_df.to_parquet(user_path, index=False)
        print(f"- User features saved: {user_path}")
        print(f"  Size: {user_path.stat().st_size / 1024 / 1024:.2f} MB")

        # Save news features
        news_path = output_path / "news_features.parquet"
        news_df.to_parquet(news_path, index=False)
        print(f"- News features saved: {news_path}")
        print(f"  Size: {news_path.stat().st_size / 1024 / 1024:.2f} MB")

    def generate_summary_stats(self, user_df, news_df):
        """Print summary statistics"""
        print("\n" + "="*60)
        print("FEATURE SUMMARY")
        print("="*60)

        print("\n- User Features:")
        print(f"  Total users: {len(user_df):,}")
        print(
            f"  Avg articles read: {user_df['num_articles_read'].mean():.1f}")
        print(f"  Max articles read: {user_df['num_articles_read'].max()}")
        print(
            f"  Avg categories explored: {user_df['unique_categories'].mean():.1f}")

        print("\n- Article Features:")
        print(f"  Total articles: {len(news_df):,}")
        print(f"  With interactions: {(news_df['impressions'] > 0).sum():,}")
        print(f"  Avg impressions: {news_df['impressions'].mean():.1f}")
        print(
            f"  Avg CTR: {news_df[news_df['impressions'] > 0]['ctr'].mean():.4f}")

        print("\n- Top Categories by Impressions:")
        top_cats = news_df.groupby('category')['impressions'].sum(
        ).sort_values(ascending=False).head(5)
        for cat, imps in top_cats.items():
            print(f"  {cat}: {imps:,} impressions")


def load_news(split="train"):
    """Load news articles from TSV file"""
    news_path = Path(f"data/raw/news.tsv")

    if not news_path.exists():
        # Check for alternative file extensions
        for ext in ['.csv', '.parquet', '.json']:
            alt_path = Path(f"data/raw/news{ext}")
            if alt_path.exists():
                print(f"Found alternative format: {alt_path}")
                news_path = alt_path
                break

    if not news_path.exists():
        raise FileNotFoundError(f"News file not found: {news_path}")

    print(f"\nLoading news from {news_path}...")

    # Load based on file extension
    if news_path.suffix == '.tsv':
        columns = [
            'news_id', 'category', 'subcategory', 'title',
            'abstract', 'url', 'title_entities', 'abstract_entities'
        ]
        df = pd.read_csv(news_path, sep='\t', header=None,
                         names=columns, on_bad_lines='warn')
    elif news_path.suffix == '.csv':
        # Assume it has headers if it's CSV
        df = pd.read_csv(news_path, on_bad_lines='warn')
    elif news_path.suffix == '.parquet':
        df = pd.read_parquet(news_path)
    elif news_path.suffix == '.json':
        df = pd.read_json(news_path)
    else:
        # Try to read as TSV first, then CSV
        try:
            columns = [
                'news_id', 'category', 'subcategory', 'title',
                'abstract', 'url', 'title_entities', 'abstract_entities'
            ]
            df = pd.read_csv(news_path, sep='\t', header=None,
                             names=columns, on_bad_lines='warn')
        except:
            df = pd.read_csv(news_path, on_bad_lines='warn')

    # Create combined text field for embedding
    if 'title' in df.columns and 'abstract' in df.columns:
        df['text'] = df['title'] + " " + df['abstract'].fillna("")
    elif 'title' in df.columns:
        df['text'] = df['title']
    else:
        # If no title column, use the first text column we find
        text_cols = [col for col in df.columns if 'text' in col.lower()
                     or 'content' in col.lower()]
        if text_cols:
            df['text'] = df[text_cols[0]]
        else:
            df['text'] = ""

    # Ensure we have required columns
    required_cols = ['news_id', 'category', 'text']
    for col in required_cols:
        if col not in df.columns:
            # Try to find similar column names
            matching_cols = [c for c in df.columns if col in c.lower()]
            if matching_cols:
                df = df.rename(columns={matching_cols[0]: col})
            else:
                # Create placeholder column
                df[col] = ""

    # Keep only necessary columns
    keep_cols = ['news_id', 'category',
                 'subcategory', 'title', 'abstract', 'text']
    keep_cols = [col for col in keep_cols if col in df.columns]
    df = df[keep_cols]

    print(f"- Loaded {len(df):,} news articles")
    return df


def load_behaviors(split="train"):
    """Load user behavior data"""
    behavior_path = Path(f"data/raw/behaviors.tsv")

    if not behavior_path.exists():
        # Check for alternative file extensions
        for ext in ['.csv', '.parquet', '.json']:
            alt_path = Path(f"data/raw/behaviors{ext}")
            if alt_path.exists():
                print(f"Found alternative format: {alt_path}")
                behavior_path = alt_path
                break

    if not behavior_path.exists():
        raise FileNotFoundError(f"Behavior file not found: {behavior_path}")

    print(f"\nLoading behaviors from {behavior_path}...")

    # Load based on file extension
    if behavior_path.suffix == '.tsv':
        columns = ['impression_id', 'user_id',
                   'time', 'history', 'impressions']
        df = pd.read_csv(behavior_path, sep='\t', header=None,
                         names=columns, on_bad_lines='warn')
    elif behavior_path.suffix == '.csv':
        # Assume it has headers if it's CSV
        df = pd.read_csv(behavior_path, on_bad_lines='warn')
    elif behavior_path.suffix == '.parquet':
        df = pd.read_parquet(behavior_path)
    elif behavior_path.suffix == '.json':
        df = pd.read_json(behavior_path)
    else:
        # Try to read as TSV first, then CSV
        try:
            columns = ['impression_id', 'user_id',
                       'time', 'history', 'impressions']
            df = pd.read_csv(behavior_path, sep='\t', header=None,
                             names=columns, on_bad_lines='warn')
        except:
            df = pd.read_csv(behavior_path, on_bad_lines='warn')

    # Rename columns to expected names
    column_mapping = {
        'impression_id': ['impression_id', 'impressionid', 'impression', 'id'],
        'user_id': ['user_id', 'userid', 'user', 'uid'],
        'time': ['time', 'timestamp', 'date', 'datetime'],
        'history': ['history', 'user_history', 'read_history', 'clicks'],
        'impressions': ['impressions', 'impression_list', 'candidates', 'news_list']
    }

    for expected_col, possible_names in column_mapping.items():
        if expected_col not in df.columns:
            for possible in possible_names:
                if possible in df.columns:
                    df = df.rename(columns={possible: expected_col})
                    break

    print(f"- Loaded {len(df):,} user behaviors")
    return df


def main():
    """Main feature engineering pipeline"""
    print("="*60)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*60)

    # Load raw data
    print("\n Loading raw data...")
    news_df = load_news("train")
    behaviors_df = load_behaviors("train")

    # Initialize feature engineer
    fe = FeatureEngineer()

    # Process features
    user_df = fe.process_user_history(behaviors_df, news_df)
    news_df = fe.process_article_features(news_df, behaviors_df)
    news_df = fe.add_category_features(news_df)

    # Save
    fe.save_features(user_df, news_df)

    # Summary
    fe.generate_summary_stats(user_df, news_df)

    print("\n Feature engineering complete!")
    print("\nNext step: Generate embeddings")
    print("  → python models/embeddings.py")


if __name__ == "__main__":
    main()
