import pandas as pd
from sklearn.preprocessing import LabelEncoder

def trend_3_3(x):
    recent = x.shift(1).rolling(3, min_periods=3).mean()
    past   = x.shift(4).rolling(3, min_periods=3).mean()
    return recent - past

# 検証用特徴量生成関数
def create_features(engine, bet_type, train_open, train_end):
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

    # target（3着以内なら1、そうでなければ0）
    if bet_type == "複勝":
        df["target"] = (df["rank"] <= 3).astype(int)
    elif bet_type == "単勝":
        df["target"] = (df["rank"] == 1).astype(int)
    else:
        raise ValueError("対応していないbet_typeです")
    
    # 単勝フラグ
    df["win_flag"] = (df["rank"] == 1).astype(int)
    # 複勝フラグ
    df["place_flag"] = (df["rank"] <= 3).astype(int)
    # ==========================
    # 特徴量
    # ==========================

    # 前走と2走前の差分
    df["rank_change"] = (
        df.groupby("horse_id")["rank"]
        .transform(lambda x: x.shift(1).diff())
    )

    # 前走と2走前の人気差分
    df["popularity_change"] = (
        df.groupby("horse_id")["popularity"]
        .transform(lambda x: x.shift(1).diff())
    )
    # 過去5戦の平均人気
    df["avg_popularity_5"] = (
        df.groupby("horse_id")["popularity"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
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


    # ==========================
    # 着順
    # ==========================
    # 前走の着順
    df["last_rank"] = df.groupby("horse_id")["rank"].shift(1)
    # 直近5戦の平均着順
    df["avg_rank_5"] = (
        df.groupby("horse_id")["rank"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )   
    # 直近5走の勝率
    df["win_rate_5"] = (
        df.groupby("horse_id")["win_flag"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    # 直近5戦の勝率レース内順位
    df["win_rate_5_rank"] = (
        df.groupby("race_id")["win_rate_5"]
        .rank(ascending=False)
    )
    # 直近5戦のうち複勝割合
    df["place_rate_5"] = (
        df.groupby("horse_id")["place_flag"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    # 直近5戦のうち複勝割合レース内順位
    df["place_rate_5_rank"] = (
        df.groupby("race_id")["place_rate_5"]
        .rank(ascending=False)
    )
    # 勝率トレンド
    df["win_rate_trend"] = (
        df.groupby("horse_id")["win_flag"]
        .transform(trend_3_3)
    )
    # 複勝率トレンド
    df["place_rate_trend"] = (
        df.groupby("horse_id")["place_flag"]
        .transform(trend_3_3)
    )

    # ==========================
    # 上がり3F
    # ==========================
    # レース内上がり順位（計算用、リークになるから使わない）
    df["last3f_rank"] = (
        df.groupby("race_id")["last3f"]
        .rank(method="min", ascending=True)
    )
    # 前走の上がり3Fタイム
    df["last_last3f"] = (
        df.groupby("horse_id")["last3f"]
        .shift(1)
    )
    # 前走の上がり順位
    df["last_last3f_rank"] = (
        df.groupby("horse_id")["last3f_rank"]
        .shift(1)
    )
    # 過去平均上がり3F
    df["avg_last3f"] = (
        df.groupby("horse_id")["last3f"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    # 過去平均上がり順位
    df["avg_last3f_rank"] = (
        df.groupby("race_id")["avg_last3f"]
        .rank(method="min", ascending=True)
    )
    # 前走と2走前の上がり3F差分
    df["last3f_change"] = (
        df.groupby("horse_id")["last3f"]
        .transform(lambda x: x.shift(1).diff())
    )
    # 前走と2走前の上がり順位差分
    df["last3f_rank_change"] = (
        df.groupby("horse_id")["last3f_rank"]
        .transform(lambda x: x.shift(1).diff())
    )
    # 直近5戦の平均上がり3Fタイム
    df["avg_last3f_5"] = (
        df.groupby("horse_id")["last3f"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    # 直近5走の平均上がり3F順位
    df["avg_last3f_rank_5"] = (
        df.groupby("horse_id")["last3f_rank"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )    
    # 距離ごとの平均上がり3Fタイム
    df["avg_last3f_distance"] = (
        df.groupby(["horse_id", "distance"])["last3f"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    # 距離ごとの平均上がり3F順位
    df["avg_last3f_distance_rank"] = (
        df.groupby("race_id")["avg_last3f_distance"]
        .rank(method="min", ascending=True)
    )

    # ==========================
    # 馬のコース適正
    # ==========================
    # 馬のコース適正数
    df["horse_course_count"] = (
        df.groupby(["horse_id", "course"])["race_id"]
        .transform("count")
    )
    # 馬のコース単勝適正
    df["horse_course_win_rate"] = (
        df.groupby(["horse_id", "course"])["win_flag"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    # 馬のコース複勝適正
    df["horse_course_place_rate"] = (
        df.groupby(["horse_id", "course"])["place_flag"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    # 馬の距離適性数
    df["horse_distance_count"] = (
        df.groupby(["horse_id", "distance"])["race_id"]
        .transform("count")
    )
    # 馬の距離単勝適正
    df["horse_distance_win_rate"] = (
        df.groupby(["horse_id", "distance"])["win_flag"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    # 馬の距離複勝適正
    df["horse_distance_place_rate"] = (
        df.groupby(["horse_id", "distance"])["place_flag"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    # 馬の馬場適性数
    df["horse_ground_count"] = (
        df.groupby(["horse_id", "ground"])["race_id"]
        .transform("count")
    )
    # 馬の馬場単勝適正
    df["horse_ground_win_rate"] = (
        df.groupby(["horse_id", "ground"])["win_flag"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    # 馬の馬場複勝適正
    df["horse_ground_place_rate"] = (
        df.groupby(["horse_id", "ground"])["place_flag"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    # 馬の馬場適性数
    df["horse_weather_count"] = (
        df.groupby(["horse_id", "weather"])["race_id"]
        .transform("count")
    )
    # 馬の天候単勝適正
    df["horse_weatherr_win_rate"] = (
        df.groupby(["horse_id", "weather"])["win_flag"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    # 馬の天候複勝適正
    df["horse_weatherr_place_rate"] = (
        df.groupby(["horse_id", "weather"])["place_flag"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    
    # ==========================
    # クラス
    # ==========================

    # クラス
    df["class_num"] = df["class"].map(class_map)
    # 前走クラス
    df["prev_class_num"] = (
    df.groupby("horse_id")["class_num"]
      .shift(1)
    )
    # クラスアップフラグ
    df["class_up_flag"] = (
        df["class_num"] > df["prev_class_num"]
    ).astype("int8")
    # クラスダウンフラグ
    df["class_down_flag"] = (
        df["class_num"] < df["prev_class_num"]
    ).astype("int8")
    # クラス差分
    df["class_diff"] = (
        df["class_num"]
        - df["prev_class_num"]
    )
    # ==========================
    # 騎手
    # ==========================

    # 騎手データ計算のため騎手でソート
    df = df.sort_values(["jockey_id", "race_date"])

    # 騎手過去30日単勝率
    df["jockey_win_rate_30d"] = (
        df.groupby("jockey_id")
        .apply(
            lambda g:
            g.set_index("race_date")["win_flag"]
            .shift(1)
            .rolling("30D")
            .mean()
        )
        .reset_index(level=0, drop=True)
        .values
        )
    # 騎手過去30日複勝率
    df["jockey_place_rate_30d"] = (
        df.groupby("jockey_id")
        .apply(
            lambda g:
            g.set_index("race_date")["place_flag"]
            .shift(1)
            .rolling("30D")
            .mean()
        )
        .reset_index(level=0, drop=True)
        .values
    )
    # 騎手過去1年単勝率
    df["jockey_win_rate_365d"] = (
    df.groupby("jockey_id")
      .apply(
          lambda g:
          g.set_index("race_date")["win_flag"]
           .shift(1)
           .rolling("365D")
           .mean()
      )
      .reset_index(level=0, drop=True)
      .values
    )# 騎手過去1年複勝率
    df["jockey_place_rate_365d"] = (
    df.groupby("jockey_id")
      .apply(
          lambda g:
          g.set_index("race_date")["place_flag"]
           .shift(1)
           .rolling("365D")
           .mean()
      )
      .reset_index(level=0, drop=True)
      .values
    )
    # 騎手コース単勝率
    df["jockey_course_win_rate"] = (
        df.groupby(
            ["jockey_id", "course"]
        )["win_flag"].transform(
            lambda x:
            x.shift(1)
            .expanding()
            .mean()
        )
    )
    # 騎手コース複勝率
    df["jockey_course_place_rate"] =(
        df.groupby(
            ["jockey_id", "course"]
        )["place_flag"].transform(
            lambda x:
            x.shift(1)
            .expanding()
            .mean()
        )
    )
    # 騎手距離単勝率
    df["jockey_distance_win_rate"] = (
        df.groupby(
            ["jockey_id", "distance"]
        )["win_flag"]
        .transform(
            lambda x:
            x.shift(1)
            .expanding()
            .mean()
        )
    )
    # 騎手距離複勝率
    df["jockey_distance_place_rate"] = (
        df.groupby(
            ["jockey_id", "distance"]
        )["place_flag"]
        .transform(
            lambda x:
            x.shift(1)
            .expanding()
            .mean()
        )
    )
    # 騎手距離単勝トレンド
    df["jockey_distance_win_rate_trend"] = (
        df.groupby(
            ["jockey_id", "distance"]
        )["win_flag"]
        .transform(trend_3_3)
    )
    # 騎手距離複勝トレンド
    df["jockey_distance_place_rate_trend"] = (
        df.groupby(
            ["jockey_id", "distance"]
        )["place_flag"]
        .transform(trend_3_3)
    )
    # 騎手の勝率トレンド
    df["jockey_win_rate_trend"] = (
        df.groupby("jockey_id")["win_flag"]
        .transform(trend_3_3)
    )
    # 騎手の複勝率トレンド
    df["jockey_place_rate_trend"] = (
        df.groupby("jockey_id")["place_flag"]
        .transform(trend_3_3)
    )
    # 騎手の勝率トレンドレース内順位
    df["jockey_win_rate_trend_rank"] = (
        df.groupby("race_id")["jockey_win_rate_trend"]
        .rank(ascending=False)
    )
    # 騎手の複勝率トレンドレース内順位
    df["jockey_place_rate_trend_rank"] = (
        df.groupby("race_id")["jockey_place_rate_trend"]
        .rank(ascending=False)
    )

    
    
    # ==========================
    # 調教師
    # ==========================
    # 調教師データ計算のため調教師でソート
    df = df.sort_values(["trainer_id", "race_date"])

    # 調教師過去1年単勝率
    df["trainer_win_rate_365d"] = (
    df.groupby("trainer_id")
      .apply(
          lambda g:
          g.set_index("race_date")["win_flag"]
           .shift(1)
           .rolling("365D")
           .mean()
      )
      .reset_index(level=0, drop=True)
      .values
    )
    # 調教師過去1年複勝率
    df["trainer_place_rate_365d"] = (
    df.groupby("trainer_id")
      .apply(
          lambda g:
          g.set_index("race_date")["place_flag"]
           .shift(1)
           .rolling("365D")
           .mean()
        )
        .reset_index(level=0, drop=True)
        .values
    )
    # 調教師コース単勝率
    df["trainer_course_win_rate"] = (
        df.groupby(
            ["trainer_id", "course"]
        )["win_flag"]
        .transform(
            lambda x:
            x.shift(1)
            .expanding()
            .mean()
        )
    )
    # 調教師コース複勝率
    df["trainer_course_place_rate"] = (
        df.groupby(
            ["trainer_id", "course"]
        )["place_flag"]
        .transform(
            lambda x:
            x.shift(1)
            .expanding()
            .mean()
        )
    )



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
    # ==========================
    # 勝率
    # ==========================
    "avg_rank_5",
    "win_rate_5",
    "place_rate_5",
    "win_rate_5_rank",
    "place_rate_5_rank",
    "win_rate_trend",
    "place_rate_trend",

    # ==========================
    # 前走着順
    # ==========================
    "last_rank",
    "rank_change",
    
    # ==========================
    # 前走上がり3F
    # ==========================
    "last_last3f",
    "last_last3f_rank",
    "last3f_change",
    "last3f_rank_change",
    "avg_last3f",
    "avg_last3f_rank",
    "avg_last3f_5",
    "avg_last3f_rank_5",
    "avg_last3f_distance",
    "avg_last3f_distance_rank",
    # ==========================
    # クラス関連
    # ==========================
    "class_num",
    "prev_class_num",
    "class_diff",
    # ==========================
    # 人気関連
    # ==========================
    # "avg_popularity_5",
    # "popularity_change",
    # "popularity",

    # ==========================
    # レース条件
    # ==========================
    "distance",
    "distance_diff",
    "field_size",
    # 場所
    "place",
    "weather",
    "ground",
    "course",
    "frame_no",

    # ==========================
    # 馬情報
    # ==========================
    "weight",
    "body_weight",
    "body_weight_diff",
    "age",
    "days_since_last",

    # ==========================
    # 馬のコース適正
    # ==========================
    "horse_course_count",
    "horse_course_win_rate",
    "horse_course_place_rate",
    "horse_distance_count",
    "horse_distance_win_rate",
    "horse_distance_place_rate",
    "horse_ground_count",
    "horse_ground_win_rate",
    "horse_ground_place_rate",
    "horse_weather_count",
    "horse_weatherr_win_rate",
    "horse_weatherr_place_rate",

    # ==========================
    # 騎手
    # ==========================
    "jockey_id",
    "jockey_win_rate_30d",
    "jockey_place_rate_30d",
    "jockey_win_rate_365d",
    "jockey_place_rate_365d",
    "jockey_course_win_rate",
    "jockey_course_place_rate",
    "jockey_distance_win_rate",
    "jockey_distance_place_rate",
    "jockey_win_rate_trend",
    "jockey_place_rate_trend",
    "jockey_win_rate_trend_rank",
    "jockey_place_rate_trend_rank",
    "jockey_distance_win_rate_trend",
    "jockey_distance_place_rate_trend",
    # ==========================
    # 調教師
    # ==========================
    "trainer_id",
    "trainer_win_rate_365d",
    "trainer_place_rate_365d",
    "trainer_course_win_rate",
    "trainer_course_place_rate",
    ]
    # ==========================
    # train / test split
    # ==========================

    # 学習には以前のデータ、検証には以降のデータを使用
    train_df = df[(df["race_date"] >= train_open)
                    & (df["race_date"] <= train_end)]
    test_df = df[ train_end < df["race_date"] ]
    
    # データの並び替え（重要）rankerモデルはデータの順番が重要なので、race_idでソート
    train_df = train_df.sort_values("race_id")
    test_df = test_df.sort_values("race_id")

    # 特徴量と目的変数の分割
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

    return x_train, y_train, group_train, cat_cols, x_test, y_test, test_df, features