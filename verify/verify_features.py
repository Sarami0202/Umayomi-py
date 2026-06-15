import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 検証用特徴量生成関数
def create_features(engine, bet_type, train_date, test_date):
    class_map = {
        "新馬": 0,
        "未勝利": 1,
        "1勝クラス": 2,
        "2勝クラス": 3,
        "3勝クラス": 4,
        "オープン": 5,
        "L": 6,
        "重賞": 7,
        "G3": 7,
        "G2": 8,
        "G1": 9,
    }

    #現在の特徴量
    # 過去5戦の平均着順
    # 過去5戦の平均3Fタイム
    # 過去5戦の平均人気
    # 過去5走の勝率
    # 過去5戦のうち3着以内だった割合
    # 前走の着順
    # 前走の3Fタイム
    # 前走と2走前の差分
    # 前走と2走前の上がり差分
    # 前走と2走前の人気差分
    # 距離差分
    # 休み明け
    # クラス
    # 頭数
    # 人気
    # 斤量
    # 馬体重
    # 馬体重の増減
    # 年齢
    # 開催場所
    # 天気
    # 馬場状態
    # コース
    # 距離
    # 枠番
    # 騎手ID
    # 調教師ID

    df = pd.read_sql("""
    SELECT
        rr.*,
        r.date AS race_date,
        r.place,
        r.weather,
        r.ground,
        r.course,
        r.distance,
        r.class,
        h.horse_name,
        h.trainer_id
    FROM race_results rr
    JOIN races r
        ON rr.race_id = r.race_id
    JOIN horses h
        ON rr.horse_id = h.horse_id
    WHERE r.course IN ('芝', 'ダ')
    """, engine)

    df["race_date"] = pd.to_datetime(df["race_date"])

    df = df.sort_values(
        ["horse_id", "race_date"]
    )

    # ==========================
    # 特徴量
    # ==========================

    # 過去5戦の平均着順
    df["avg_rank_5"] = (
        df.groupby("horse_id")["rank"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    # 過去5戦の平均3Fタイム（追加） - 3Fタイムも重要な特徴量なので追加してみる
    df["avg_last3f_5"] = (
        df.groupby("horse_id")["last3f"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    # 過去5戦の平均人気
    df["avg_popularity_5"] = (
        df.groupby("horse_id")["popularity"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    
    # 過去5走の勝率
    df["win_rate_5"] = (
        df.groupby("horse_id")["rank"]
        .transform(lambda x: x.shift(1).eq(1).rolling(5, min_periods=1).mean())
    )

    # 過去5戦のうち3着以内だった割合
    df["place_rate_5"] = (
        df.groupby("horse_id")["rank"]
        .transform(lambda x: x.shift(1).le(3).rolling(5, min_periods=1).mean())
    )

    # 前走の着順
    df["last_rank"] = df.groupby("horse_id")["rank"].shift(1)
    # 前走の3Fタイム
    df["last_last3f"] = df.groupby("horse_id")["last3f"].shift(1)

    # 前走と2走前の差分
    df["rank_change"] = (
        df.groupby("horse_id")["rank"]
        .transform(lambda x: x.shift(1).diff())
    )

    # 前走と2走前の上がり差分
    df["last3f_change"] = (
        df.groupby("horse_id")["last3f"]
        .transform(lambda x: x.shift(1).diff())
    )

    # 前走と2走前の人気差分
    df["popularity_change"] = (
        df.groupby("horse_id")["popularity"]
        .transform(lambda x: x.shift(1).diff())
    )
    
    # 距離差分
    df["distance_diff"] = (
        df["distance"]
        - df.groupby("horse_id")["distance"].shift(1)
    )

    # 休み明け    
    df["days_since_last"] = (
        df["race_date"]
        - df.groupby("horse_id")["race_date"].shift(1)
    ).dt.days

    # 頭数
    df["field_size"] = df.groupby("race_id")["horse_id"].transform("count")

    
    # 複勝用
    # target（3着以内なら1、そうでなければ0）
    if bet_type == "複勝":
        df["target"] = (df["rank"] <= 3).astype(int)
    elif bet_type == "単勝":
        df["target"] = (df["rank"] == 1).astype(int)
    else:
        raise ValueError("対応していないbet_typeです")
    
    # ==========================
    # 整数化
    # ==========================

    # クラスを整数化
    df["race_class_num"] = df["class"].map(class_map)

    # ==========================
    # カテゴリー変数化
    # ==========================

    # カテゴリ変数に変換
    cat_cols = [
        "place",
        "weather",
        "ground",
        "course",
        "jockey_id",
        "trainer_id",
        "frame_no",
    ]
    for col in cat_cols:
        df[col] = df[col].astype("category")

    

    # ==========================
     # 特徴量
    # ==========================

    features = [
        "avg_rank_5",
        "avg_last3f_5",
        "avg_popularity_5",
        "win_rate_5",
        "place_rate_5",
        "last_rank",
        "last_last3f",
        "rank_change",
        "last3f_change",        
        "popularity_change",
        "distance_diff",
        "days_since_last",
        "race_class_num",
        "field_size",
        "popularity",
        "weight",
        "body_weight",
        "body_weight_diff",
        "age",
        "place",
        "weather",
        "ground",
        "course",
        "distance",
        "jockey_id",
        "trainer_id",
        "frame_no",
    ]
    # ==========================
    # train / test split
    # ==========================

    train_df = df[df["race_date"] < train_date]
    test_df = df[df["race_date"] >= test_date]

    x_train = train_df[features]
    y_train = train_df["target"]
    x_train = x_train.fillna(-1)

    x_test = test_df[features]
    y_test = test_df["target"]
    x_test = x_test.fillna(-1)

    # ==========================
    # ★ Ranker用 group作成（重要）
    # ==========================

    group_train = train_df.groupby("race_id").size().tolist()

    return x_train, y_train, group_train, cat_cols, x_test, y_test,test_df