"""
Train ML ranking model to predict click probability
Uses gradient boosting on hand-crafted features
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import pickle
from pathlib import Path
from tqdm import tqdm
import time
import sys
import os

# Add the project root to the Python path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_behaviors(split="train"):
    """Load user behavior data directly from TSV file"""
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

    print(f"✓ Loaded {len(df):,} user behaviors")
    return df


class SimpleRanker:
    """
    Ranking model that predicts click probability
    Combines user features, article features, and user-article interactions
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=5):
        """Initialize gradient boosting model"""
        print(f"\n{'='*60}")
        print(f"Initializing Ranking Model")
        print(f"{'='*60}")

        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42,
            verbose=0
        )

        self.feature_names = None

        print(f"  Model: Gradient Boosting")
        print(f"  Trees: {n_estimators}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Max depth: {max_depth}")

    def prepare_training_data(self, behaviors_df, news_df, user_df, sample_size=None):
        """
        Create training examples from impression logs

        Each impression (article shown to user) becomes a training example:
        - Positive example if clicked (label=1)
        - Negative example if not clicked (label=0)
        """
        print(f"\n{'='*60}")
        print(f"Preparing Training Data")
        print(f"{'='*60}")

        # Create lookup tables for efficiency
        news_lookup = news_df.set_index('news_id').to_dict('index')
        user_lookup = user_df.set_index('user_id').to_dict('index')

        examples = []

        # Sample behaviors if needed
        if sample_size and sample_size < len(behaviors_df):
            behaviors_df = behaviors_df.sample(n=sample_size, random_state=42)
            print(f"- Sampling {sample_size:,} behaviors for faster training")

        print(f"Processing {len(behaviors_df):,} behaviors...")

        for _, behavior in tqdm(behaviors_df.iterrows(), total=len(behaviors_df), desc="Creating examples"):
            if pd.isna(behavior['impressions']):
                continue

            user_id = behavior['user_id']

            # Skip if user not in profile (no history)
            if user_id not in user_lookup:
                continue

            user_profile = user_lookup[user_id]

            # Parse impressions
            impressions = behavior['impressions'].split()

            for impression in impressions:
                try:
                    news_id, clicked = impression.split('-')
                except ValueError:
                    continue

                # Skip if article not in dataset
                if news_id not in news_lookup:
                    continue

                article = news_lookup[news_id]

                # Extract features
                features = self._extract_features(user_profile, article)
                features['label'] = int(clicked)

                examples.append(features)

        df = pd.DataFrame(examples)

        print(f"\n✓ Created {len(df):,} training examples")
        print(
            f"  Positive (clicked): {df['label'].sum():,} ({df['label'].mean():.2%})")
        print(
            f"  Negative (not clicked): {(~df['label'].astype(bool)).sum():,}")

        return df

    def _extract_features(self, user_profile, article):
        """
        Extract features for ranking

        Feature categories:
        1. User features: reading behavior, preferences
        2. Article features: popularity, quality
        3. Match features: user-article compatibility
        """
        features = {}

        # User features
        features['user_read_count'] = user_profile.get('num_articles_read', 0)
        features['user_category_diversity'] = user_profile.get(
            'category_diversity', 0)
        features['user_unique_categories'] = user_profile.get(
            'unique_categories', 0)

        # Article features
        features['article_ctr'] = article.get('ctr', 0)
        features['article_impressions'] = np.log1p(
            article.get('impressions', 0))
        features['article_clicks'] = np.log1p(article.get('clicks', 0))
        features['article_popularity'] = article.get('popularity', 0)
        features['article_has_clicks'] = article.get('has_clicks', 0)

        # Category features
        features['category_avg_ctr'] = article.get('category_avg_ctr', 0)
        features['category_popularity'] = np.log1p(
            article.get('category_popularity', 0))

        # Match features
        top_category = user_profile.get('top_category')
        article_category = article.get('category')

        features['category_match_top'] = 1 if article_category == top_category else 0
        features['category_match_second'] = 1 if article_category == user_profile.get(
            'second_category') else 0
        features['category_match_third'] = 1 if article_category == user_profile.get(
            'third_category') else 0

        # Text length features
        title_len = len(article.get('title', ''))
        abstract_len = len(article.get('abstract', '') or '')
        features['title_length'] = np.log1p(title_len)
        features['abstract_length'] = np.log1p(abstract_len)
        features['has_abstract'] = 1 if abstract_len > 0 else 0

        return features

    def train(self, train_df):
        """Train the ranking model"""
        print(f"\n{'='*60}")
        print(f"Training Model")
        print(f"{'='*60}")

        # Separate features and labels
        feature_cols = [c for c in train_df.columns if c != 'label']
        self.feature_names = feature_cols

        X = train_df[feature_cols]
        y = train_df['label']

        print(f"\nFeatures ({len(feature_cols)}):")
        for feat in feature_cols:
            print(f"  • {feat}")

        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"\nDataset split:")
        print(f"  Train: {len(X_train):,} examples")
        print(f"  Validation: {len(X_val):,} examples")

        # Train model
        print(f"\nTraining...")
        start = time.time()
        self.model.fit(X_train, y_train)
        elapsed = time.time() - start

        print(f"✓ Training completed in {elapsed:.2f}s")

        # Evaluate
        self._evaluate(X_val, y_val, "Validation")

        return self.model

    def _evaluate(self, X, y, split_name="Test"):
        """Evaluate model performance"""
        print(f"\n{'='*60}")
        print(f"{split_name} Results")
        print(f"{'='*60}")

        # Predictions
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        y_pred = self.model.predict(X)

        # Metrics
        auc = roc_auc_score(y, y_pred_proba)
        ap = average_precision_score(y, y_pred_proba)

        print(f"\n📊 Metrics:")
        print(f"  AUC-ROC: {auc:.4f}")
        print(f"  Average Precision: {ap:.4f}")

        # Feature importance
        self._print_feature_importance()

        return auc

    def _print_feature_importance(self):
        """Print top features by importance"""
        if self.feature_names is None:
            return

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\n- Top 10 Features:")
        for i, row in importance_df.head(10).iterrows():
            bar_len = int(row['importance'] * 50)
            bar = '█' * bar_len
            print(f"  {row['feature']:30s} {bar} {row['importance']:.4f}")

    def predict(self, features_df):
        """Predict click probability for new examples"""
        if self.feature_names:
            features_df = features_df[self.feature_names]
        return self.model.predict_proba(features_df)[:, 1]

    def save(self, path="models/ranker.pkl"):
        """Save trained model"""
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            'model': self.model,
            'feature_names': self.feature_names
        }

        with open(model_path, 'wb') as f:
            pickle.dump(save_dict, f)

        print(f"\n- Model saved: {model_path}")
        print(f"  Size: {model_path.stat().st_size / 1024:.2f} KB")

    def load(self, path="models/ranker.pkl"):
        """Load trained model"""
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)

        self.model = save_dict['model']
        self.feature_names = save_dict['feature_names']

        print(f"- Model loaded: {path}")


def main():
    """Main training pipeline"""
    print("="*60)
    print("RANKING MODEL TRAINING PIPELINE")
    print("="*60)

    # Check if processed data exists
    news_path = Path("data/processed/news_features.parquet")
    user_path = Path("data/processed/user_features.parquet")

    if not news_path.exists() or not user_path.exists():
        print("\n❌ Processed data not found!")
        print("Please run feature engineering first:")
        print("  python data/feature_engineering.py")
        return

    # Load data
    print("\n- Loading data...")
    news_df = pd.read_parquet(news_path)
    user_df = pd.read_parquet(user_path)
    behaviors_df = load_behaviors("train")

    print(f"✓ News articles: {len(news_df):,}")
    print(f"✓ User profiles: {len(user_df):,}")
    print(f"✓ Behaviors: {len(behaviors_df):,}")

    # Initialize ranker
    ranker = SimpleRanker(n_estimators=100, learning_rate=0.1, max_depth=5)

    # Prepare training data (sample for speed - remove sample_size for full dataset)
    train_df = ranker.prepare_training_data(
        behaviors_df,
        news_df,
        user_df,
        sample_size=10000  # Remove this line to train on full dataset
    )

    # Train
    ranker.train(train_df)

    # Save
    ranker.save()

    print("\n** Training complete!")
    print("\nNext step: Build web application")
    print("  → python app/api.py")


if __name__ == "__main__":
    main()
