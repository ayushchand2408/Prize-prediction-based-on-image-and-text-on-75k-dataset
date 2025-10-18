import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def calculate_smape(y_true, y_pred):
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denominator != 0
    smape_values = np.zeros_like(numerator)
    smape_values[mask] = numerator[mask] / denominator[mask]
    return 100.0 * np.mean(smape_values)

# ==================== TEXT CLEANING / HELPERS ====================
def clean_text(t):
    t = str(t).lower()
    t = re.sub(r'[^a-z0-9\s\.\-/]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

# Robust membership check for label encoder fallback
def safe_label_encode(series, existing_classes=None):
    # returns encoded series and classes array
    if existing_classes is None:
        classes = np.unique(series.astype(str))
    else:
        classes = np.array(existing_classes)
    mapping = {c: i for i, c in enumerate(classes)}
    return series.astype(str).map(lambda x: mapping[x] if x in mapping else -1).astype(int), classes

# ==================== AGGRESSIVE FEATURE EXTRACTION ====================
def extract_aggressive_features(df, use_clean_text=True):
    print("  Extracting AGGRESSIVE features...")
    s = df['catalog_content'].fillna('').astype(str)
    if use_clean_text:
        s_clean = s.apply(clean_text)
    else:
        s_clean = s

    features = pd.DataFrame(index=df.index)

    def get_value(txt):

        t = str(txt).lower()
        m = re.search(r'value[:\s]*([0-9]+(?:\.[0-9]+)?)', t)
        if m:
            return float(m.group(1))
        m2 = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*(ml|l|g|kg|oz)', t)
        if m2:
            val = float(m2.group(1))
            unit = m2.group(2)
            if unit == 'l':
                return val * 1000.0
            if unit == 'kg':
                return val * 1000.0
            if unit == 'oz':
                return val * 28.3495
            return val
        # fallback numeric
        m3 = re.search(r'\b([0-9]{2,4})\b', t)
        if m3:
            return float(m3.group(1))
        return 0.0

    features['value'] = s_clean.map(get_value).astype(float)
    features['value_log'] = np.log1p(features['value'])

    # ---- Pack / quantity ----
    def get_pack(txt):
        t = str(txt).lower()
        patterns = [
            r'pack\s*of\s*(\d+)', r'\((?:pack|pk)\s*of\s*(\d+)\)',
            r'(\d+)\s*pack\b', r'\bpk\s*(\d+)\b', r'case\s*of\s*(\d+)', r'(\d+)\s*x\b'
        ]
        for p in patterns:
            m = re.search(p, t)
            if m:
                qty = int(m.group(1))
                if 1 <= qty <= 1000:
                    return qty
        return 1

    features['pack'] = s_clean.map(get_pack).astype(int)
    features['value_times_pack'] = features['value'] * features['pack']

    # ---- Unit normalization (derived unit from text) ----
    def get_unit(txt):
        t = str(txt).lower()
        if re.search(r'\b(ml|milliliter|millilitre)\b', t):
            return 'ml'
        if re.search(r'\b(l|litre|liter)\b', t):
            return 'l'
        if re.search(r'\b(g|gram|grams)\b', t):
            return 'g'
        if re.search(r'\b(kg|kilogram)\b', t):
            return 'kg'
        if re.search(r'\b(count|each|piece|pcs|pack)\b', t):
            return 'count'
        return 'other'
    features['unit'] = s_clean.map(get_unit)

    # ---- Basic text stats ----
    features['text_length'] = s_clean.map(len)
    features['word_count'] = s_clean.map(lambda x: len(x.split()))

    # ---- Category heuristics ----
    def get_category(txt):
        t = str(txt).lower()
        if any(w in t for w in ['drink', 'juice', 'coffee', 'tea', 'cola', 'soda']):
            return 'beverage'
        if any(w in t for w in ['snack', 'bar', 'chip', 'cookie', 'cracker']):
            return 'snack'
        if any(w in t for w in ['sauce', 'oil', 'spice', 'salt', 'pepper']):
            return 'condiment'
        if any(w in t for w in ['beauty', 'skincare', 'soap', 'shampoo']):
            return 'personal'
        return 'other'
    features['category'] = s_clean.map(get_category)

    # ---- Brand heuristic 
    def get_brand(txt):
        t = str(txt).strip()
        m = re.match(r'^\s*([A-Za-z0-9&\.]{2,30})\s*[-:]\s*', t)
        if m:
            return m.group(1).lower()
        first = t.split()[0].lower() if len(t.split())>0 else ''
        if first and not re.match(r'^\d', first) and len(first)>1:
            return first
        return 'unknown'
    features['brand'] = df['catalog_content'].fillna('').map(get_brand)

    return features

# ==================== OOF TARGET ENCODING (KFold) ====================
def target_encode_oof(train_df, test_df, col, target, n_splits=5, aggfunc='median', random_state=42):
    """Return (train_encoded_series, test_encoded_series, global_train_map)"""
    print(f"  Target-encoding {col} (OOF KFold)...")
    train = train_df[[col]].copy()
    test = test_df[[col]].copy()
    train_encoded = pd.Series(index=train.index, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for tr_idx, val_idx in kf.split(train):
        tr = train.iloc[tr_idx]
        val = train.iloc[val_idx]
        mapping = train_df.iloc[tr_idx].groupby(col)[target].agg(aggfunc)
        # map for val
        train_encoded.iloc[val_idx] = val[col].map(mapping).fillna(train_df[target].median())
    # global mapping 
    global_map = train_df.groupby(col)[target].agg(aggfunc)
    test_encoded = test[col].map(global_map).fillna(train_df[target].median())
    return train_encoded.astype(float), test_encoded.astype(float), global_map.to_dict()

# ==================== TF-IDF + SVD ====================
def get_text_features(train_texts, test_texts, n_features=700, n_svd=100):
    print("  Generating TF-IDF + SVD features...")
    # combined cleaning
    train_c = [clean_text(t) for t in train_texts.fillna('')]
    test_c = [clean_text(t) for t in test_texts.fillna('')]

    vectorizer = TfidfVectorizer(
        max_features=n_features, stop_words='english',
        ngram_range=(1, 3), analyzer='word', min_df=2
    )
    tfidf_train = vectorizer.fit_transform(train_c)
    tfidf_test = vectorizer.transform(test_c)

    # Reduce dimensionality for downstream models
    if n_svd > 0:
        svd = TruncatedSVD(n_components=min(n_svd, tfidf_train.shape[1]-1), random_state=42)
        tfidf_train_svd = svd.fit_transform(tfidf_train)
        tfidf_test_svd = svd.transform(tfidf_test)
        tfidf_cols = [f'tfidf_svd_{i}' for i in range(tfidf_train_svd.shape[1])]
        return pd.DataFrame(tfidf_train_svd, columns=tfidf_cols), pd.DataFrame(tfidf_test_svd, columns=tfidf_cols)
    else:
        tfidf_cols = [f'tfidf_{i}' for i in range(tfidf_train.shape[1])]
        return pd.DataFrame(tfidf_train.toarray(), columns=tfidf_cols), pd.DataFrame(tfidf_test.toarray(), columns=tfidf_cols)

# ==================== PREPARE FEATURES (MAIN) ====================
def prepare_features(train_df, test_df):
    print("="*70)
    print("AGGRESSIVE FEATURE ENGINEERING (IMPROVED)")
    print("="*70)

    train_feats = extract_aggressive_features(train_df)
    test_feats = extract_aggressive_features(test_df)
    print(f"✓ Base features extracted: {train_feats.shape[1]}")

    # OOF target-encoding for brand, category (use price)
    train_te_brand, test_te_brand, brand_map = target_encode_oof(train_df.assign(brand=train_feats['brand']),
                                                                 test_df.assign(brand=test_feats['brand']),
                                                                 col='brand', target='price', n_splits=5)
    train_feats['brand_price_te'] = train_te_brand
    test_feats['brand_price_te'] = test_te_brand

    train_te_cat, test_te_cat, cat_map = target_encode_oof(train_df.assign(category=train_feats['category']),
                                                          test_df.assign(category=test_feats['category']),
                                                          col='category', target='price', n_splits=5)
    train_feats['category_price_te'] = train_te_cat
    test_feats['category_price_te'] = test_te_cat

    # TF-IDF + SVD
    tfidf_train_df, tfidf_test_df = get_text_features(train_df['catalog_content'], test_df['catalog_content'],
                                                      n_features=800, n_svd=120)
    print(f"✓ TF-IDF SVD dims: {tfidf_train_df.shape[1]}")

    # Combine
    X_train = pd.concat([train_feats.reset_index(drop=True), tfidf_train_df.reset_index(drop=True)], axis=1)
    X_test = pd.concat([test_feats.reset_index(drop=True), tfidf_test_df.reset_index(drop=True)], axis=1)

    # Label encoding for small-cardinality cats (unit)
    unit_vals = X_train['unit'].astype(str).unique()
    X_train['unit_enc'], unit_classes = safe_label_encode(X_train['unit'], existing_classes=unit_vals)
    X_test['unit_enc'], _ = safe_label_encode(X_test['unit'], existing_classes=unit_vals)
    X_train.drop(columns=['unit'], inplace=True)
    X_test.drop(columns=['unit'], inplace=True)

    X_train.drop(columns=['brand', 'category'], inplace=True)
    X_test.drop(columns=['brand', 'category'], inplace=True)

    print(f"✓ Total features after combine: {X_train.shape[1]}")
    return X_train, X_test

# ==================== MODEL TRAINING: OOF LGBM + STACKING ====================
def train_and_stack(X, y, X_test, n_splits=5):
    print("="*70)
    print("TRAINING OOF LightGBM + STACK (Ridge) ")
    print("="*70)

    X_values = X.values
    X_test_values = X_test.values
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(X))
    test_preds = np.zeros((n_splits, X_test.shape[0]))
    models = []
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.03,
        'num_leaves': 64,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'seed': 42,
        'verbosity': -1
    }

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_values)):
        print(f"[fold {fold+1}/{n_splits}]")
        X_tr, X_val = X_values[tr_idx], X_values[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        # scale numeric features (robust)
        scaler = RobustScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test_values)

        # train LightGBM on log1p target
        lgb_train = lgb.Dataset(X_tr_s, label=np.log1p(y_tr))
        lgb_val = lgb.Dataset(X_val_s, label=np.log1p(y_val), reference=lgb_train)

        m = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=5000,
        valid_sets=[lgb_train, lgb_val],
        callbacks=[
        lgb.early_stopping(stopping_rounds=150),
        lgb.log_evaluation(period=200)
            ]
        )

        models.append((m, scaler))
        val_pred = np.expm1(m.predict(X_val_s, num_iteration=m.best_iteration))
        oof_preds[val_idx] = val_pred
        test_preds[fold] = np.expm1(m.predict(X_test_s, num_iteration=m.best_iteration))

        sm = calculate_smape(y_val, val_pred)
        print(f"  Fold SMAPE: {sm:.3f}%")

    meta_train = oof_preds.reshape(-1, 1)
    meta_test = test_preds.mean(axis=0).reshape(-1, 1)

    meta = Ridge(alpha=1.0)
    meta.fit(meta_train, y)
    final_test_pred = meta.predict(meta_test)

    # overall metrics
    overall_smape = calculate_smape(y, oof_preds)
    mae = mean_absolute_error(y, oof_preds)
    rmse = np.sqrt(mean_squared_error(y, oof_preds))
    r2 = r2_score(y, oof_preds)

    print("="*50)
    print(f"OOF SMAPE: {overall_smape:.3f}%")
    print(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.4f}")
    print("="*50)

    return models, meta, oof_preds, final_test_pred, (mae, rmse, r2, overall_smape)

# ==================== PREDICT ====================
def predict_test(models, meta, X_test):

    test_preds = []
    for m, scaler in models:
        X_test_s = scaler.transform(X_test.values)
        p = np.expm1(m.predict(X_test_s, num_iteration=m.best_iteration))
        test_preds.append(p)
    test_stack = np.mean(np.vstack(test_preds), axis=0).reshape(-1, 1)
    final = meta.predict(test_stack)
    return final

# ==================== MAIN ====================
def main(train_csv, test_csv, output_csv):
    print("="*70)
    print("AGGRESSIVE PRICING MODEL - IMPROVED")
    print("="*70)

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    print(f"✓ Train: {train_df.shape}")
    print(f"✓ Test: {test_df.shape}")

    X_train, X_test = prepare_features(train_df, test_df)
    y_train = train_df['price'].values

    models, meta, oof_preds, test_final_pred, metrics = train_and_stack(X_train, y_train, X_test, n_splits=5)

    final_preds = test_final_pred
    out_df = pd.DataFrame({'sample_id': test_df['sample_id'], 'price': final_preds})
    out_df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved predictions to {output_csv}")
    print("="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Validation SMAPE (OOF): {metrics[3]:.2f}%")
    print(f"MAE: {metrics[0]:.2f}, RMSE: {metrics[1]:.2f}, R2: {metrics[2]:.4f}")
    print("="*70)
    return out_df

if __name__ == "__main__":
    TRAIN_CSV = "train.csv"
    TEST_CSV = "test.csv"
    OUTPUT_CSV = "test_out.csv"
    results = main(TRAIN_CSV, TEST_CSV, OUTPUT_CSV)
    print("✓ Complete!")