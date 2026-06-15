from lightgbm import LGBMRanker
import pandas as pd
import numpy as np
import shap
from sklearn.metrics import ndcg_score

def calc_ndcg(result_df, k):
    ndcg_list = []

    for race_id, df in result_df.groupby("race_id"):

        # 重要：馬数が少ない場合を制御
        if len(df) < k:
            continue

        y_true = df["target"].values
        y_score = df["score"].values

        # 安定化のため順位を明示的にソート
        order = np.argsort(-y_score)

        y_true = y_true[order]
        y_score = y_score[order]

        ndcg = ndcg_score(
            y_true.reshape(1, -1),
            y_score.reshape(1, -1),
            k=k
        )

        ndcg_list.append(ndcg)

    return float(np.mean(ndcg_list))

# Rankerモデルの検証関数
def verify_ranker_model(engine, bet_type, x_train, y_train, group_train, cat_cols,
                 x_test, y_test, test_df):

    # ==========================
    # Rankerモデル
    # ==========================
    model = LGBMRanker(
        # Ranker用の目的関数
        objective="lambdarank",
        # NDCG評価を使用
        metric="ndcg",
        # NDCGの評価で上位k件を重視
        ndcg_eval_at=[1, 3, 5],
        # 決定木の数は多め
        n_estimators=1000,
        # 学習率は小さめにして、過学習を防止
        learning_rate=0.03,
        # 木の複雑さ
        num_leaves=31,
        # 過学習防止のためのパラメータ
        min_child_samples=50,
        # 
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )

    model.fit(
        x_train,
        y_train,
        group=group_train,
        categorical_feature=cat_cols
    )

    # ==========================
    # 予測（スコア）
    # ==========================
    score = model.predict(x_test)

    # ==========================
    # DataFrameへ
    # ==========================
    result_df = test_df.copy()
    result_df["score"] = score

    # 評価処理

    # ==========================
    # レースごとの1位的中率
    # ==========================
    top1 = (
        result_df
        .sort_values(["race_id", "score"], ascending=[True, False])
        .groupby("race_id")
        .head(1)
    )

    print(f"予測1位馬の{bet_type}率: {top1['target'].mean():.2%}")

    # ==========================
    # レースごとの上位3頭
    # ==========================
    top3 = (
        result_df
        .sort_values(["race_id", "score"], ascending=[True, False])
        .groupby("race_id")
        .head(3)
    )

    print(f"予測上位3頭の{bet_type}率: {top3['target'].mean():.2%}")

    # ==========================
    # NDCG評価
    # ==========================
    ndcg1 = calc_ndcg(result_df, k=1)
    ndcg3 = calc_ndcg(result_df, k=3)
    ndcg5 = calc_ndcg(result_df, k=5)

    print("\n===== NDCG評価 =====")
    print(f"NDCG@1: {ndcg1:.4f}")
    print(f"NDCG@3: {ndcg3:.4f}")
    print(f"NDCG@5: {ndcg5:.4f}")
    
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
    # レースサイズ
    # ==========================
    print("\n===== レースサイズ =====")
    print(len(top1))
    print(len(merged))

    # ==========================
    # 予測スコア順の順位付け（追加）
    # =========================
    result_df["pred_rank"] = (
    result_df
    .groupby("race_id")["score"]
    .rank(ascending=False)
    )

    print("\n===== 人気 vs モデル比較 =====")

    compare = pd.DataFrame({
        "popularity_hit_rate": result_df.groupby("popularity")["target"].mean(),
        "model_rank_hit_rate": result_df.groupby("pred_rank")["target"].mean()
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
    # 相関分析
    # ==========================
    print("\n===== Correlation Analysis =====")

    corr = x_train.corr(numeric_only=True)

    # 相関が高いペアを抽出
    high_corr = []

    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            corr_value = corr.iloc[i, j]

            if abs(corr_value) >= 0.9:
                high_corr.append([
                    corr.columns[i],
                    corr.columns[j],
                    corr_value
                ])

    high_corr_df = pd.DataFrame(
        high_corr,
        columns=["feature1", "feature2", "correlation"]
    ).sort_values(
        "correlation",
        key=abs,
        ascending=False
    )

    print(high_corr_df.head(50))

    # ==========================
    # SHAP値の可視化
    # =========================
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(shap_values, x_test)

    
    return model, result_df
