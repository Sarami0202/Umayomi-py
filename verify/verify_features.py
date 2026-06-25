import pandas as pd
from sklearn.preprocessing import LabelEncoder

# トレンド計算関数
def trend_3_3(x):
    recent = x.shift(1).rolling(3, min_periods=3).mean()
    past   = x.shift(4).rolling(3, min_periods=3).mean()
    return recent - past
# 最後通過順
def get_last_corner(x):
    return int(str(x).split("-")[-1])
# 馬の脚質を判定する関数
def style(pos):
    if pd.isna(pos):
        return 4  # 不明
    
    if pos <= 3:
        return 1  # 逃げ先行
    elif pos <= 7:
        return 2  # 差し
    else:
        return 3  # 追込
# 平均コーナー通過順位を計算する関数
def get_avg_corner(x):
    corners = [int(v) for v in str(x).split("-")]
    return sum(corners) / len(corners)

# 検証用特徴量生成関数
def create_features(engine, bet_type, train_open, train_end, feature_open):
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

    df = pd.read_sql(f"""
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
    AND r.date >= '{feature_open}'
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

    # 枠率
    df["frame_ratio"] = df["frame_no"] / df["field_size"]

    # 馬番率
    df["horse_no_ratio"] = df["horse_no"] / df["field_size"]

    # 馬体重の増減の絶対値
    df["body_weight_diff_abs"] = df["body_weight_diff"].abs()

    # 出場回数
    df["career_count"] = df.groupby("horse_id").cumcount()
    # 初出走フラグ
    df["is_first_race"] = (df["career_count"] == 0).astype(int)

    
    # ==========================
    # 着順
    # ==========================
    # 前走の着順
    df["last_rank"] = df.groupby("horse_id")["rank"].shift(1)
    # 前走の成績
    df["last_rank_rank"] = (
        df.groupby("race_id")["last_rank"]
        .rank(ascending=True)
    )
    # 直近5戦の平均着順
    df["avg_rank_5"] = (
        df.groupby("horse_id")["rank"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    ) 
    # 直近5戦の平均着順レース内順位
    df["avg_rank_5_rank"] = (
        df.groupby("race_id")["avg_rank_5"]
        .rank(ascending=True)
    )
    # 直近3戦の平均着順
    df["avg_rank_3"] = (
        df.groupby("horse_id")["rank"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    # 直近3戦の平均着順レース内順位
    df["avg_rank_3_rank"] = (
        df.groupby("race_id")["avg_rank_3"]
        .rank(ascending=True)
    ) 
    # 前走と2走前の差分
    df["rank_change"] = (
        df.groupby("horse_id")["rank"]
        .transform(lambda x: x.shift(1).diff())
    )
    # 直近5戦のべスト着順
    df["best_rank_5"] = (
        df.groupby("horse_id")["rank"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).min())
    )
    # 着順トレンド
    df["rank_trend"] = (
        df.groupby("horse_id")["rank"]
        .transform(trend_3_3)
    )
    # 着順の安定性
    df["rank_std_5"] = (
        df.groupby("horse_id")["rank"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=2).std())
    )
    # 直近5戦の平均着順距離ごと
    df["avg_rank_distance"] = (
        df.groupby(["horse_id", "distance"])["rank"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    # 直近5戦の平均着順距離ごとレース内順位
    df["avg_rank_distance_rank"] = (
        df.groupby("race_id")["avg_rank_distance"]
        .rank(ascending=True)
    )
    # レースごとの直近5戦の平均着順(中間特徴量)
    df["avg_rank_5_race_avg"] = (
        df.groupby("race_id")["avg_rank_5"]
        .transform("mean")
    )
    # レースごとの直近5戦の平均着順との差分
    df["avg_rank_5_diff"] = (
        df["avg_rank_5"] - df["avg_rank_5_race_avg"]
    )
    # DF断片化対策
    df = df.copy()
    # ==========================
    # 勝率
    # ==========================
    # 全体の単勝率
    df["win_rate"] = (
        df.groupby("horse_id")["win_flag"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    # 全体の単勝率レース内順位
    df["win_rate_rank"] = (
        df.groupby("race_id")["win_rate"]
        .rank(ascending=False)
    )
    # 全体の複勝率
    df["place_rate"] = (
        df.groupby("horse_id")["place_flag"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    # 全体の複勝率レース内順位
    df["place_rate_rank"] = (
        df.groupby("race_id")["place_rate"]
        .rank(ascending=False)
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
    # 勝率の安定性
    df["win_rate_std_5"] = (
        df.groupby("horse_id")["win_flag"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=2).std())
    )
    # 複勝率の安定性
    df["place_rate_std_5"] = (
        df.groupby("horse_id")["place_flag"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=2).std())
    )
    # DF断片化対策
    df = df.copy()
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
    # 上がり3Fのトレンド
    df["last3f_trend"] = (
        df.groupby("horse_id")["last3f"]
        .transform(trend_3_3)
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
    # 上がり3Fの安定性
    df["last3f_std_5"] = (
        df.groupby("horse_id")["last3f"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=2).std())
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
    # コースごとの平均上がり3Fタイム
    df["avg_last3f_course"] = (
        df.groupby(["horse_id", "course"])["last3f"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    # コースごとの平均上がり3F順位
    df["avg_last3f_course_rank"] = (
        df.groupby("race_id")["avg_last3f_course"]
        .rank(method="min", ascending=True)
    )
    # レースごとの直近5戦の平均上がり3F(中間特徴量)
    df["avg_last3f_race_avg"] = (
        df.groupby("race_id")["avg_last3f"]
        .transform("mean")
    )
    # レースごとの直近5戦の平均上がり3Fとの差分
    df["avg_last3f_diff"] = (
        df["avg_last3f"] - df["avg_last3f_race_avg"]
    )
    # 直近5戦のベスト3Fタイム
    df["best_last3f_5"] = (
        df.groupby("horse_id")["last3f"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).min())
    )
    # DF断片化対策
    df = df.copy()
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
    # 馬のコース単勝適正レース内順位
    df["horse_course_win_rate_rank"] = (
        df.groupby("race_id")["horse_course_win_rate"]
        .rank(ascending=False)
    )
    # 馬のコース複勝適正
    df["horse_course_place_rate"] = (
        df.groupby(["horse_id", "course"])["place_flag"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    # 馬のコース複勝適正レース内順位
    df["horse_course_place_rate_rank"] = (
        df.groupby("race_id")["horse_course_place_rate"]
        .rank(ascending=False)
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
    # 馬の距離単勝適正レース内順位
    df["horse_distance_win_rate_rank"] = (
        df.groupby("race_id")["horse_distance_win_rate"]
        .rank(ascending=False)
    )
    # 馬の距離複勝適正
    df["horse_distance_place_rate"] = (
        df.groupby(["horse_id", "distance"])["place_flag"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    # 馬の距離複勝適正レース内順位
    df["horse_distance_place_rate_rank"] = (
        df.groupby("race_id")["horse_distance_place_rate"]
        .rank(ascending=False)
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
    
    # DF断片化対策
    df = df.copy()
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
    # 同じクラスの出場数
    df["class_count"] = (
        df.groupby(["horse_id", "class_num"])
        .cumcount()
    )
    # クラス別の単勝率
    df["class_win_rate"] = (
        df.groupby("class_num")["win_flag"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    # クラス別の複勝率
    df["class_place_rate"] = (
        df.groupby("class_num")["place_flag"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    # クラス別の単勝率レース内順位
    df["class_win_rate_rank"] = (
        df.groupby("race_id")["class_win_rate"]
        .rank(ascending=False)
    )
    # クラス別の複勝率レース内順位
    df["class_place_rate_rank"] = (
        df.groupby("race_id")["class_place_rate"]
        .rank(ascending=False)
    )
    # クラス別の単勝率トレンド
    df["class_win_rate_trend"] = (
        df.groupby("class_num")["win_flag"]
        .transform(trend_3_3)
    )
    # クラス別の複勝率トレンド
    df["class_place_rate_trend"] = (
        df.groupby("class_num")["place_flag"]
        .transform(trend_3_3)
    )
    # クラス別の単勝率安定性
    df["class_win_rate_std"] = (
        df.groupby("class_num")["win_flag"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=2).std())
    )
    # クラスの複勝率安定性
    df["class_place_rate_std"] = (
        df.groupby("class_num")["place_flag"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=2).std())
    )
    # クラス別の単勝率安定性レース内順位
    df["class_win_rate_std_rank"] = (
        df.groupby("race_id")["class_win_rate_std"]
        .rank(ascending=True)
    )
    # クラス別の複勝率安定性レース内順位
    df["class_place_rate_std_rank"] = (
        df.groupby("race_id")["class_place_rate_std"]
        .rank(ascending=True)
    )    
    # DF断片化対策
    df = df.copy()
    
    
    # ==========================
    # 通過順
    # ==========================
    # 最後通過順(中間特徴量)
    df["last_corner"] = df["passing"].apply(get_last_corner)
    # 前走の最後通過順
    df["last_last_corner"] = (
        df.groupby("horse_id")["last_corner"]
        .shift(1)
    )
    # 前走の最後通過順の直近5戦の平均
    df["avg_last_corner_5"] = (
        df.groupby("horse_id")["last_corner"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    # 前走の最後通過順の直近5戦の平均レース内順位
    df["avg_last_corner_5_rank"] = (
        df.groupby("race_id")["avg_last_corner_5"]
        .rank(ascending=True)
    )
    # 通過順の平均(中間特徴量)
    df["avg_corner"] = df["passing"].apply(get_avg_corner)
    # 前走の通過順の平均
    df["last_avg_corner"] = (
        df.groupby("horse_id")["avg_corner"]
        .shift(1)
    )
    # 前走の通過順の平均のレース内順位
    df["last_avg_corner_rank"] = (
        df.groupby("race_id")["last_avg_corner"]
        .rank(ascending=True)
    )
    # 前走の通過順の平均の直近5戦の平均
    df["avg_last_avg_corner_5"] = (
        df.groupby("horse_id")["avg_corner"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    # 前走の通過順の平均の直近5戦の平均レース内順位
    df["avg_last_avg_corner_5_rank"] = (
        df.groupby("race_id")["avg_last_avg_corner_5"]
        .rank(ascending=True)
    )
    # 脚質
    df["running_style"] = df["avg_last_avg_corner_5"].apply(style)
    # 脚質フラグ
    df["front_runner_flag"] = (
        df["running_style"] == 1
    ).astype(int)
    # 同じレースの逃げ馬の数
    df["race_front_runner_count"] = (
        df.groupby("race_id")["front_runner_flag"]
        .transform("sum")
    )
    # 同じ馬の過去の逃げ馬の数
    df["horse_front_runner_count"] = (
    df.groupby("horse_id")["front_runner_flag"]
      .transform(lambda x: x.shift(1).cumsum())
    )
    # 同じ馬の過去の逃げ馬の割合
    df["front_runner_rate"] = (
    df.groupby("horse_id")["front_runner_flag"]
      .transform(
          lambda x: x.shift(1).expanding().mean()
      )
    )
    

    # DF断片化対策
    df = df.copy()
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
    # 騎手過去30日単勝率レース内ランク
    df["jockey_win_rate_30d_rank"] = (
        df.groupby("race_id")["jockey_win_rate_30d"]
        .rank(ascending=False)
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
    # 騎手過去30日複勝率レース内ランク
    df["jockey_place_rate_30d_rank"] = (
        df.groupby("race_id")["jockey_place_rate_30d"]
        .rank(ascending=False)
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
    )
    # 騎手過去1年単勝率レース内ランク
    df["jockey_win_rate_365d_rank"] = (
        df.groupby("race_id")["jockey_win_rate_365d"]
        .rank(ascending=False)
    )
    # 騎手過去1年複勝率
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
    # 騎手過去1年複勝率レース内ランク
    df["jockey_place_rate_365d_rank"] = (
        df.groupby("race_id")["jockey_place_rate_365d"]
        .rank(ascending=False)
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
    # 騎手距離単勝安定性
    df["jockey_distance_win_rate_std"] = (
        df.groupby(
            ["jockey_id", "distance"]
        )["win_flag"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(5, min_periods=2)
            .std()
        )
    )
    # 騎手距離複勝安定性    
    df["jockey_distance_place_rate_std"] = (
        df.groupby(
            ["jockey_id", "distance"]
        )["place_flag"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(5, min_periods=2)
            .std()
        )
    )
    # 騎手距離単勝安定性レース内順位
    df["jockey_distance_win_rate_std_rank"] = (
        df.groupby("race_id")["jockey_distance_win_rate_std"]
        .rank(ascending=True)
    )
    # 騎手距離複勝安定性レース内順位
    df["jockey_distance_place_rate_std_rank"] = (
        df.groupby("race_id")["jockey_distance_place_rate_std"]        
        .rank(ascending=True)
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
    # 騎手の過去1年単勝率レース内平均(中間特徴量)
    df["jockey_win_rate_365d_race_avg"] = (
        df.groupby("race_id")["jockey_win_rate_365d"]
        .transform("mean")
    )
    # 騎手の過去1年単勝率とレース内平均の差分
    df["jockey_win_rate_365d_diff"] = (
        df["jockey_win_rate_365d"] - df["jockey_win_rate_365d_race_avg"].mean()
    )
    # 騎手の過去1年複勝率レース内平均(中間特徴量)
    df["jockey_place_rate_365d_race_avg"] = (
        df.groupby("race_id")["jockey_place_rate_365d"]
        .transform("mean")
    )
    # 騎手の過去1年複勝率とレース内平均の差分
    df["jockey_place_rate_365d_diff"] = (
        df["jockey_place_rate_365d"] - df["jockey_place_rate_365d_race_avg"].mean()
    )
    # 騎手の同じ馬に騎乗した回数
    df["jockey_horse_count"] = (
        df.groupby(["jockey_id", "horse_id"])["race_id"]
        .cumcount()
    )
    # 騎手の同じ馬に騎乗した単勝率
    df["jockey_horse_win_rate"] = (
        df.groupby(["jockey_id", "horse_id"])["win_flag"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    # 騎手の同じ馬に騎乗した単勝率レース内順位
    df["jockey_horse_win_rate_rank"] = (
        df.groupby("race_id")["jockey_horse_win_rate"]
        .rank(ascending=False)
    )
    # 騎手の同じ馬に騎乗した複勝率
    df["jockey_horse_place_rate"] = (
        df.groupby(["jockey_id", "horse_id"])["place_flag"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    # 騎手の同じ馬に騎乗した複勝率レース内順位
    df["jockey_horse_place_rate_rank"] = (
        df.groupby("race_id")["jockey_horse_place_rate"]
        .rank(ascending=False)
    )
    # DF断片化対策
    df = df.copy()
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
    # 調教師過去1年単勝率レース内ランク
    df["trainer_win_rate_365d_rank"] = (
        df.groupby("race_id")["trainer_win_rate_365d"]
        .rank(ascending=False)
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
    # 調教師過去1年複勝率レース内ランク
    df["trainer_place_rate_365d_rank"] = (
        df.groupby("race_id")["trainer_place_rate_365d"]
        .rank(ascending=False)
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
    # 調教師単勝安定率
    df["trainer_win_rate_365d_std"] = (
        df.groupby("trainer_id")["win_flag"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(5, min_periods=2)
            .std()
        )
    )
    # 調教師複勝安定率
    df["trainer_place_rate_365d_std"] = (
        df.groupby("trainer_id")["place_flag"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(5, min_periods=2)
            .std()
        )
    )

    # DF断片化対策
    df = df.copy()
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
        "running_style",
    ]
    for col in cat_cols:
        df[col] = df[col].astype("category")

    

    # ==========================
     # 特徴量
    # ==========================

    features = [    
    # ==========================
    # 前走着順
    # ==========================
    "last_rank",
    "last_rank_rank",
    "avg_rank_5",
    "avg_rank_5_rank",
    "avg_rank_3",
    "avg_rank_3_rank",
    "rank_change",
    "best_rank_5",
    "rank_trend",
    "rank_std_5",
    "avg_rank_distance",
    "avg_rank_distance_rank",
    "avg_rank_5_diff",

    # ==========================
    # 勝率
    # ==========================
    "win_rate",
    "place_rate",
    "win_rate_rank",
    "place_rate_rank",
    "win_rate_5",
    "place_rate_5",
    "win_rate_5_rank",
    "place_rate_5_rank",
    "win_rate_trend",
    "place_rate_trend",
    "win_rate_std_5",
    "place_rate_std_5",
    
    # ==========================
    # 前走上がり3F
    # ==========================
    "last_last3f",
    "last_last3f_rank",
    "last3f_change",
    "last3f_rank_change",
    "best_last3f_5",
    "last3f_trend",
    "avg_last3f",
    "avg_last3f_rank",
    "avg_last3f_5",
    "avg_last3f_rank_5",
    "avg_last3f_distance",
    "avg_last3f_distance_rank",
    "avg_last3f_course",
    "avg_last3f_course_rank",
    "last3f_std_5",
    "avg_last3f_diff",
    # ==========================
    # クラス関連
    # ==========================
    "class_num",
    "class_count",
    "prev_class_num",
    "class_diff",
    "class_win_rate",
    "class_place_rate",
    "class_win_rate_rank",
    "class_place_rate_rank",
    "class_win_rate_trend",
    "class_place_rate_trend",
    "class_win_rate_std",
    "class_place_rate_std",
    "class_win_rate_std_rank",
    "class_place_rate_std_rank",
    
    # ==========================
    # 通過順
    # ==========================
    "last_last_corner",
    "avg_last_corner_5",
    "avg_last_corner_5_rank",
    "running_style",
    "race_front_runner_count",
    "horse_front_runner_count",
    "front_runner_rate",
    "avg_last_avg_corner_5",
    "avg_last_avg_corner_5_rank",
    "last_avg_corner",
    "last_avg_corner_rank",
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
    # 率
    "frame_ratio",
    "horse_no_ratio",

    # ==========================
    # 馬情報
    # ==========================
    "weight",
    "body_weight",
    "body_weight_diff",
    "body_weight_diff_abs",
    "age",
    "days_since_last",
    "career_count",
    "is_first_race",
    # ==========================
    # 馬のコース適正
    # ==========================
    "horse_course_count",
    "horse_course_win_rate",
    "horse_course_place_rate",
    "horse_course_win_rate_rank",
    "horse_course_place_rate_rank",
    "horse_distance_count",
    "horse_distance_win_rate",
    "horse_distance_place_rate",
    "horse_distance_win_rate_rank",
    "horse_distance_place_rate_rank",
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
    "jockey_win_rate_30d_rank",
    "jockey_place_rate_30d_rank",
    "jockey_win_rate_365d",
    "jockey_place_rate_365d",
    "jockey_win_rate_365d_rank",
    "jockey_place_rate_365d_rank",
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
    "jockey_distance_win_rate_std",
    "jockey_distance_place_rate_std",
    "jockey_distance_win_rate_std_rank",
    "jockey_distance_place_rate_std_rank",
    "jockey_win_rate_365d_diff",
    "jockey_place_rate_365d_diff",
    "jockey_horse_count",
    "jockey_horse_win_rate",
    "jockey_horse_place_rate",
    "jockey_horse_win_rate_rank",
    "jockey_horse_place_rate_rank",
    # ==========================
    # 調教師
    # ==========================
    "trainer_id",
    "trainer_win_rate_365d",
    "trainer_place_rate_365d",
    "trainer_win_rate_365d_rank",
    "trainer_place_rate_365d_rank",
    "trainer_course_win_rate",
    "trainer_course_place_rate",
    "trainer_win_rate_365d_std",
    "trainer_place_rate_365d_std",
    ]
    # ==========================
    # train / test split
    # ==========================# 
    # 全特徴量生成後
    df = df.copy()

    # 学習には以前のデータ、検証には以降のデータを使用
    train_df = df[(df["race_date"] >= train_open)
                    & (df["race_date"] <= train_end)]
    # テストは新馬戦のデータは使用しない
    test_df = df[
        (df["race_date"] > train_end)
        & (df["class"] != "新馬")
    ]
    # データの並び替え（重要）rankerモデルはデータの順番が重要なので、race_idでソート
    train_df = train_df.sort_values("race_id")
    test_df = test_df.sort_values("race_id")

    # 特徴量と目的変数の分割
    x_train = train_df[features]
    y_train = train_df["target"]

    x_test = test_df[features]
    y_test = test_df["target"]

    # 欠損値を-1で埋める（LightGBMは欠損値はnanのままがいいかも）
    # x_train = x_train.fillna(-1)
    # x_test = x_test.fillna(-1)
    
    # ==========================
    # ★ Ranker用 group作成（重要）
    # ==========================

    group_train = train_df.groupby("race_id").size().tolist()

    return x_train, y_train, group_train, cat_cols, x_test, y_test, test_df, features