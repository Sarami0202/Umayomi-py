from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
import shap

from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    brier_score_loss
)

# Classifierモデルの検証関数
def verify_classifier_model(
    engine,
    bet_type,
    x_train,
    y_train,
    cat_cols,
    x_test,
    y_test,
    test_df
):

    # ==========================
    # Classifierモデル
    # ==========================

    model = LGBMClassifier(
        objective="binary",
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )

    model.fit(
        x_train,
        y_train,
        categorical_feature=cat_cols
    )

    # ==========================
    # 複勝確率
    # ==========================
    prob = model.predict_proba(x_test)[:, 1]

    # ==========================
    # DataFrameへ
    # ==========================
    result_df = test_df.copy()
    result_df["prob"] = prob    
    result_df["horse_no"] = result_df["horse_no"].astype(str)

    # 評価処理

    # ==========================
    # AUC
    # ==========================
    auc = roc_auc_score(y_test, prob)

    print(f"AUC: {auc:.4f}")

    # ==========================
    # LogLoss 小さいほど良い
    # ==========================
    ll = log_loss(y_test, prob)

    print(f"LogLoss: {ll:.5f}")

    # ==========================
    # Brier Score 小さいほど良い
    # ==========================
    brier = brier_score_loss(y_test, prob)

    print(f"Brier Score: {brier:.5f}")

    # ==========================
    # レースごとの1位
    # ==========================
    top1 = (
        result_df
        .sort_values(
            ["race_id", "prob"],
            ascending=[True, False]
        )
        .groupby("race_id")
        .head(1)
    )

    print(
        f"予測1位馬の複勝率: "
        f"{top1['target'].mean():.2%}"
    )

    # ==========================
    # レースごとの上位3頭
    # ==========================
    top3 = (
        result_df
        .sort_values(
            ["race_id", "prob"],
            ascending=[True, False]
        )
        .groupby("race_id")
        .head(3)
    )

    print(
        f"予測上位3頭の複勝率: "
        f"{top3['target'].mean():.2%}"
    )

    # ==========================
    # ROI評価
    # ==========================
    payouts = pd.read_sql(
        f"""
        SELECT race_id, bet_type, combination, payout
        FROM race_payouts
        WHERE bet_type = '{bet_type}'
        """,
        engine
    )

    # 型を合わせる
    result_df["horse_no"] = result_df["horse_no"].astype(str)
    payouts["combination"] = payouts["combination"].astype(str)

 
    # ==========================
    # Top1 ROI
    # ==========================
    top1 = (
        result_df
        .sort_values(["race_id", "score"], ascending=[True, False])
        .groupby("race_id")
        .head(1)
    )

    merged = top1.merge(
        payouts,
        left_on=["race_id", "horse_no"],
        right_on=["race_id", "combination"],
        how="left"
    )

    merged["payout"] = merged["payout"].fillna(0)

    race_result = merged.groupby("race_id", as_index=False).agg({
        "payout": "sum"
    })
    
    race_result["bet"] = 100
    race_result["roi"] = race_result["payout"] / race_result["bet"]

    investment = race_result["bet"].sum()
    returned = race_result["payout"].sum()
    roi = returned / investment

    print("\n===== Top1 ROI =====")
    print(f"購入レース数: {len(race_result):,}")
    print(f"投資額: {investment:,}円")
    print(f"回収額: {returned:,.0f}円")
    print(f"ROI: {roi:.2%}")

    # ==========================
    # Top3 ROI（複勝）
    # ==========================
    if bet_type == "複勝":
        top3 = (
            result_df
            .sort_values(["race_id", "score"], ascending=[True, False])
            .groupby("race_id")
            .head(3)
        )

        merged3 = top3.merge(
            payouts,
            left_on=["race_id", "horse_no"],
            right_on=["race_id", "combination"],
            how="left"
        )

        merged3["payout"] = merged3["payout"].fillna(0)

        race_result3 = merged3.groupby("race_id", as_index=False).agg({
            "payout": "sum"
        })

        investment3 = len(race_result3) * 300
        returned3 = race_result3["payout"].sum()
        roi3 = returned3 / investment3

        print("\n===== Top3 ROI =====")
        print(f"購入レース数: {len(race_result3):,}")
        print(f"投資額: {investment3:,}円")
        print(f"回収額: {returned3:,.0f}円")
        print(f"ROI: {roi3:.2%}")


    # ==========================
    # 予測順位
    # ==========================
    result_df["pred_rank"] = (
        result_df
        .groupby("race_id")["prob"]
        .rank(
            ascending=False,
            method="first"
        )
    )

    print("\n===== 人気 vs モデル比較 =====")

    compare = pd.DataFrame({
        "popularity_hit_rate":
            result_df.groupby("popularity")["target"].mean(),

        "model_rank_hit_rate":
            result_df.groupby("pred_rank")["target"].mean()
    })

    print(compare.head(10))

    # 特徴量分析処理
 
    # ==========================
    # 特徴量重要度
    # ==========================
    importance = pd.DataFrame({
        "feature": x_train.columns,
        "importance": model.feature_importances_
    }).sort_values(
        "importance",
        ascending=False
    )

    print("\n===== Feature Importance =====")
    print(f"総特徴量数: {importance["importance"].sum()}")
    print(importance.head(30))

    # ==========================
    # SHAP値の可視化
    # =========================
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(shap_values, x_test)

    return model, result_df