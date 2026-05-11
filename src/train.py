


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    # Clip negatives (price can't be < 0)
    y_pred = np.maximum(y_pred, 0)

    mae   = mean_absolute_error(y_test, y_pred)
    rmse  = mean_squared_error(y_test, y_pred) ** 0.5
    r2    = r2_score(y_test, y_pred)
    mape  = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100

    metrics = {
        "MAE":  round(float(mae),  2),
        "RMSE": round(float(rmse), 2),
        "R2":   round(float(r2),   4),
        "MAPE": round(float(mape), 2),
    }
    return metrics, y_pred


def train():
    print("═" * 55)
    print("  MODEL TRAINING — Car Price Predictor")
    print("═" * 55)

    # ── 1. Load & clean data ──────────────────────────────────
    print("\n[1/5] Loading and cleaning data …")
    df = full_pipeline(DATA_PATH)

    # ── 2. Feature / target split ─────────────────────────────
    print("\n[2/5] Building feature matrix …")
    X, y = build_features(df)
    print(f"      X shape : {X.shape}")
    print(f"      y shape : {y.shape}")

    # ── 3. Train / test split (80/20) ─────────────────────────
    print("\n[3/5] Splitting data  (test_size = 0.20) …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"      Train : {X_train.shape[0]} samples")
    print(f"      Test  : {X_test.shape[0]} samples")

    # ── 4. Build & train pipeline ─────────────────────────────
    print("\n[4/5] Training Linear Regression pipeline …")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    print("      ✓ Training complete")

    # ── 5. Evaluate ───────────────────────────────────────────
    print("\n[5/5] Evaluating on test set …")
    metrics, _ = evaluate(pipeline, X_test, y_test)

    print("\n  ┌─────────────────────────────────────┐")
    print(f"  │  MAE   : ₹ {metrics['MAE']:>14,.0f}          │")
    print(f"  │  RMSE  : ₹ {metrics['RMSE']:>14,.0f}          │")
    print(f"  │  R²    :   {metrics['R2']:>14.4f}          │")
    print(f"  │  MAPE  :   {metrics['MAPE']:>13.2f} %         │")
    print("  └─────────────────────────────────────┘")

    # ── Save pipeline ─────────────────────────────────────────
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\n✅ Model saved → {MODEL_PATH}")

    # ── Save metadata for the web app ─────────────────────────
    # Unique values for dropdowns
    unique_companies  = sorted(df["company"].unique().tolist())
    unique_fuel_types = sorted(df["fuel_type"].unique().tolist())
    unique_names      = sorted(df["name"].unique().tolist())
    year_range        = [int(df["year"].min()), int(df["year"].max())]
    kms_range         = [int(df["kms_driven"].min()), int(df["kms_driven"].max())]

    meta = {
        "metrics":       metrics,
        "companies":     unique_companies,
        "fuel_types":    unique_fuel_types,
        "car_names":     unique_names,
        "year_range":    year_range,
        "kms_range":     kms_range,
        "train_samples": int(X_train.shape[0]),
        "test_samples":  int(X_test.shape[0]),
        "features":      CAT_FEATURES + NUM_FEATURES,
    }

    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✅ Metadata saved → {META_PATH}")

    return pipeline, metrics


if __name__ == "__main__":
    train()
