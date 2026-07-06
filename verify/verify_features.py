import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

# ほかの特徴量を使用して追加する場合はif文の特徴量からif分を外すこと

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
# コーナー推進力を計算する関数
def get_corner_progress(x):
    corners = [int(v) for v in str(x).split("-")]
    return  corners[0] - corners[-1] 
# 時間を秒に変換する関数
def time_to_sec(t):
    if pd.isna(t):
        return np.nan

    m, s = t.split(":")
    return int(m) * 60 + float(s)

# 検証用特徴量生成関数
def create_features(engine, bet_type, features, cat_cols, train_open, train_end, feature_open):
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
    if "popularity_change" in features:
        df["popularity_change"] = (
            df.groupby("horse_id")["popularity"]
            .transform(lambda x: x.shift(1).diff())
        )
    # 過去5戦の平均人気
    if "avg_popularity_5" in features:
        df["avg_popularity_5"] = (
            df.groupby("horse_id")["popularity"]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        )
    
    # 距離差分
    if "distance_diff" in features:
        df["distance_diff"] = (
            df["distance"]
            - df.groupby("horse_id")["distance"].shift(1)
        )
        
    # コースチェンジ
    if "course_changed" in features:
        df["course_changed"] = (
            df["course"]
            != df.groupby("horse_id")["course"].shift(1)
        ).astype(int)

    # 馬場チェンジ
    if "ground_changed" in features:
        df["ground_changed"] = (
            df["ground"]
            != df.groupby("horse_id")["ground"].shift(1)
        ).astype(int)

    # 天候チェンジ
    if "weather_changed" in features:
        df["weather_changed"] = (
            df["weather"]
            != df.groupby("horse_id")["weather"].shift(1)
        ).astype(int)

    # 休み明け    
    if "days_since_last" in features:
        df["days_since_last"] = (
            df["race_date"]
            - df.groupby("horse_id")["race_date"].shift(1)
        ).dt.days

    # 頭数
    df["field_size"] = df.groupby("race_id")["horse_id"].transform("count")

    # 枠率
    if "frame_ratio" in features:
        df["frame_ratio"] = df["frame_no"] / df["field_size"]

    # 馬番率
    if "horse_no_ratio" in features:
        df["horse_no_ratio"] = df["horse_no"] / df["field_size"]

    # 馬体重の増減の絶対値
    if "body_weight_diff_abs" in features:
        df["body_weight_diff_abs"] = df["body_weight_diff"].abs()

    # 出場回数
    df["career_count"] = df.groupby("horse_id").cumcount()
    # 初出走フラグ
    if "is_first_race" in features:
        df["is_first_race"] = (df["career_count"] == 0).astype(int)

    # 単勝数
    df["win_count"] = (
        df.groupby("horse_id")["win_flag"]
        .transform(lambda x: x.shift(1).cumsum())
    )
    # 複勝数
    df["place_count"] = (
        df.groupby("horse_id")["place_flag"]
        .transform(lambda x: x.shift(1).cumsum())
    )

    
    # ==========================
    # 着順
    # ==========================
    # 前走の着順
    df["last_rank"] = df.groupby("horse_id")["rank"].shift(1)
    # 前走の成績
    if "last_rank_rank" in features:
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
    if "avg_rank_5_rank" in features:
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
    if "avg_rank_3_rank" in features:
        df["avg_rank_3_rank"] = (
            df.groupby("race_id")["avg_rank_3"]
            .rank(ascending=True)
        ) 
    # 前走と2走前の差分
    if "rank_change" in features:
        df["rank_change"] = (
            df.groupby("horse_id")["rank"]
            .transform(lambda x: x.shift(1).diff())
        )
    # 直近5戦のべスト着順
    if "best_rank_5" in features:
        df["best_rank_5"] = (
            df.groupby("horse_id")["rank"]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).min())
        )
    # 着順トレンド
    if "rank_trend" in features:
        df["rank_trend"] = (
            df.groupby("horse_id")["rank"]
            .transform(trend_3_3)
        )
    # 着順の安定性
    if "rank_std_5" in features:
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
    if "avg_rank_distance_rank" in features:
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
    if "avg_rank_5_diff" in features:
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
    if "win_rate_rank" in features:
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
    if "place_rate_rank" in features:
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
    if "win_rate_5_rank" in features:
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
    if "place_rate_5_rank" in features:
        df["place_rate_5_rank"] = (
            df.groupby("race_id")["place_rate_5"]
            .rank(ascending=False)
        )
    # 勝率トレンド
    if "win_rate_trend" in features:
        df["win_rate_trend"] = (
            df.groupby("horse_id")["win_flag"]
            .transform(trend_3_3)
        )
    # 複勝率トレンド
    if "place_rate_trend" in features:
        df["place_rate_trend"] = (
            df.groupby("horse_id")["place_flag"]
            .transform(trend_3_3)
        )
    # 勝率の安定性
    if "win_rate_std_5" in features:
        df["win_rate_std_5"] = (
            df.groupby("horse_id")["win_flag"]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=2).std())
        )
    # 複勝率の安定性
    if "place_rate_std_5" in features:
        df["place_rate_std_5"] = (
            df.groupby("horse_id")["place_flag"]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=2).std())
        )
    # DF断片化対策
    df = df.copy()

    # ==========================
    # タイム
    # ==========================
    # タイムを秒に変換
    df["race_time_sec"] = df["time"].apply(time_to_sec)
    # 前走タイム
    df["last_time"] = (
        df.groupby("horse_id")["race_time_sec"]
        .shift(1)
    )
    # 直近5戦の平均タイム
    if "avg_time_5" in features:
        df["avg_time_5"] = (
            df.groupby("horse_id")["race_time_sec"]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        )
    # 距離別前走タイム
    if "last_time_distance" in features:
        df["last_time_distance"] = (
            df.groupby(["horse_id", "distance"])["race_time_sec"]
            .shift(1)
        )
    # 距離別平均タイム
    if "avg_time_distance" in features:
        df["avg_time_distance"] = (
            df.groupby(["horse_id", "distance"])["race_time_sec"]
            .transform(lambda x: x.shift(1).expanding().mean())
        )
    # 距離別直近5戦の平均タイム
    df["avg_time_distance_5"] = (
        df.groupby(["horse_id", "distance"])["race_time_sec"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    # 距離別直近5戦の平均タイムレース内順位
    if "avg_time_distance_5_rank" in features:
        df["avg_time_distance_5_rank"] = (
            df.groupby("race_id")["avg_time_distance_5"]
            .rank(ascending=True)
        )
    # コース別の前走タイム
    if "last_time_course" in features:
        df["last_time_course"] = (
            df.groupby(["horse_id", "course"])["race_time_sec"]
            .shift(1)
        )
    # コース別の平均タイム
    if "avg_time_course" in features:
        df["avg_time_course"] = (
            df.groupby(["horse_id", "course"])["race_time_sec"]
            .transform(lambda x: x.shift(1).expanding().mean())
        )
    # コース別の直近5戦の平均タイム
    df["avg_time_course_5"] = (
        df.groupby(["horse_id", "course"])["race_time_sec"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    # コース別の直近5戦の平均タイムレース内順位
    if "avg_time_course_5_rank" in features:
        df["avg_time_course_5_rank"] = (
            df.groupby("race_id")["avg_time_course_5"]
            .rank(ascending=True)
        )
    # 前走勝ち馬のタイム差
    df["last_time_diff"] = (
        df["last_time"] - df.groupby("race_id")["last_time"].transform("min")
    )
    # 前走勝ち馬のタイム差平均
    if "avg_last_time_diff" in features:
        df["avg_last_time_diff"] = (
            df.groupby("horse_id")["last_time_diff"]
            .transform(lambda x: x.shift(1).expanding().mean())
        )
    # 前走勝ち馬のタイム差直近5戦の平均
    df["avg_last_time_diff_5"] = (
        df.groupby("horse_id")["last_time_diff"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    # 前走勝ち馬のタイム差直近5戦の平均レース内順位
    if "avg_last_time_diff_5_rank" in features:
        df["avg_last_time_diff_5_rank"] = (
            df.groupby("race_id")["avg_last_time_diff_5"]
            .rank(ascending=True)
        )
    # 距離ごとの前走勝ち馬のタイム差
    if "last_time_diff_distance" in features:
        df["last_time_diff_distance"] = (
            df.groupby(["horse_id", "distance"])["last_time_diff"]
            .shift(1)
        )
    # 距離ごとの前走勝ち馬のタイム差平均
    if "avg_last_time_diff_distance" in features:
        df["avg_last_time_diff_distance"] = (
            df.groupby(["horse_id", "distance"])["last_time_diff"]
            .transform(lambda x: x.shift(1).expanding().mean())
        )
    # 距離ごとの前走勝ち馬のタイム差直近5戦の平均
    df["avg_last_time_diff_distance_5"] = (
        df.groupby(["horse_id", "distance"])["last_time_diff"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    # 距離ごとの前走勝ち馬のタイム差直近5戦の平均レース内順位
    if "avg_last_time_diff_distance_5_rank" in features:
        df["avg_last_time_diff_distance_5_rank"] = (
            df.groupby("race_id")["avg_last_time_diff_distance_5"]
            .rank(ascending=True)
        )
    # 直近5走の中で最も良かった勝ち馬とのタイム差
    df["best_last_time_diff_5"] = (
        df.groupby("horse_id")["last_time_diff"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).min())
    )
    # 直近5走の中で最も良かった勝ち馬とのタイム差レース内順位
    if "best_last_time_diff_5_rank" in features:
        df["best_last_time_diff_5_rank"] = (
            df.groupby("race_id")["best_last_time_diff_5"]
            .rank(ascending=True)
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
    if "last_last3f" in features:
        df["last_last3f"] = (
            df.groupby("horse_id")["last3f"]
            .shift(1)
        )
    # 上がり3Fのトレンド
    if "last3f_trend" in features:
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
    if "avg_last3f_rank" in features:
        df["avg_last3f_rank"] = (
            df.groupby("race_id")["avg_last3f"]
            .rank(method="min", ascending=True)
        )
    # 上がり3Fの安定性
    if "last3f_std_5" in features:
        df["last3f_std_5"] = (
            df.groupby("horse_id")["last3f"]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=2).std())
        )
    # 前走と2走前の上がり3F差分
    df["last3f_change"] = (
        df.groupby("horse_id")["last3f"]
        .transform(lambda x: x.shift(1).diff())
    )
    # 前走と2走前の上がり3F差分のレース内順位
    if "last3f_change_rank" in features:
        df["last3f_change_rank"] = (
            df.groupby("race_id")["last3f_change"]
            .rank(method="min", ascending=True)
        )
    # 前走と2走前の上がり順位差分
    if "last3f_rank_change" in features:
        df["last3f_rank_change"] = (
            df.groupby("horse_id")["last3f_rank"]
            .transform(lambda x: x.shift(1).diff())
        )
    # 直近5戦の平均上がり3Fタイム
    df["avg_last3f_5"] = (
        df.groupby("horse_id")["last3f"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    # 直近5戦の平均上がり3Fタイムとの差分
    if "avg_last3f_5_diff" in features:
        df["avg_last3f_5_diff"] = (
            df["avg_last3f"] - df["avg_last3f_5"]
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
    if "avg_last3f_distance_rank" in features:
        df["avg_last3f_distance_rank"] = (
            df.groupby("race_id")["avg_last3f_distance"]
            .rank(method="min", ascending=True)
        )
    # 距離ごとの平均上がり3Fタイムの安定性
    if "avg_last3f_distance_std" in features:
        df["avg_last3f_distance_std"] = (
            df.groupby(["horse_id", "distance"])["last3f"]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=2).std())
        )
    # 距離との平均上がり3Fタイムとの差分
    if "avg_last3f_distance_diff" in features:
        df["avg_last3f_distance_diff"] = (
            df["avg_last3f"] - df["avg_last3f_distance"]
        )
    # コースごとの平均上がり3Fタイム
    df["avg_last3f_course"] = (
        df.groupby(["horse_id", "course"])["last3f"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    # コースごとの平均上がり3F順位
    if "avg_last3f_course_rank" in features:
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
    # 直近5戦のベスト3Fタイムレース内順位
    if "best_last3f_5_rank" in features:
        df["best_last3f_5_rank"] = (
            df.groupby("race_id")["best_last3f_5"]
            .rank(method="min", ascending=True)
        )
    # 前走の上がり3F順位の割合
    df["last3f_rank_rate"] = (
        df["last_last3f_rank"] / df["field_size"]
    )
    # 直近5戦の平均上がり3F順位の割合
    df["avg_last3f_rank_5_rate"] = (
        df["avg_last3f_rank_5"] / df["field_size"]
    )
    # 直近5戦の平均上がり3F順位の割合レース内順位
    if "avg_last3f_rank_5_rate_rank" in features:
        df["avg_last3f_rank_5_rate_rank"] = (
            df.groupby("race_id")["avg_last3f_rank_5_rate"]
            .rank(method="min", ascending=True)
        )
    # 前走の上がり3F順位のパーセンタイル
    if "last3f_percentile" in features:
        df["last3f_percentile"] = (
            (df["field_size"] - df["last_last3f_rank"])
            / (df["field_size"] - 1)
        )
    # 直近5戦の平均上がり3F順位のパーセンタイル
    df["avg_last3f_rank_5_percentile"] = (
        (df["field_size"] - df["avg_last3f_rank_5"])
        / (df["field_size"] - 1)
    )
    # 直近5戦の平均上がり3F順位のパーセンタイルのレース内順位
    if "avg_last3f_rank_5_percentile_rank" in features:
        df["avg_last3f_rank_5_percentile_rank"] = (
            df.groupby("race_id")["avg_last3f_rank_5_percentile"]
            .rank(method="min", ascending=False)
        )
    # DF断片化対策
    df = df.copy()
    # ==========================
    # 馬のコース適正
    # ==========================
    # 馬のコース適正数
    if "horse_course_count" in features:
        df["horse_course_count"] = (
            df.groupby(["horse_id", "course"])["race_id"]
            .cumcount()
        )
    # 馬のコース単勝適正
    df["horse_course_win_rate"] = (
        df.groupby(["horse_id", "course"])["win_flag"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    # 馬のコース単勝適正レース内順位
    if "horse_course_win_rate_rank" in features:
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
    if "horse_course_place_rate_rank" in features:
        df["horse_course_place_rate_rank"] = (
            df.groupby("race_id")["horse_course_place_rate"]
            .rank(ascending=False)
        )
    # 馬の距離適性数
    if "horse_distance_count" in features:
        df["horse_distance_count"] = (
            df.groupby(["horse_id", "distance"])["race_id"]
            .cumcount()
        )
    # 馬の距離単勝適正
    df["horse_distance_win_rate"] = (
        df.groupby(["horse_id", "distance"])["win_flag"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    # 馬の距離単勝適正レース内順位
    if "horse_distance_win_rate_rank" in features:
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
    if "horse_distance_place_rate_rank" in features:
        df["horse_distance_place_rate_rank"] = (
            df.groupby("race_id")["horse_distance_place_rate"]
            .rank(ascending=False)
        )
    # 馬の馬場適性数
    if "horse_ground_count" in features:
        df["horse_ground_count"] = (
            df.groupby(["horse_id", "ground"])["race_id"]
            .cumcount()
        )
    # 馬の馬場単勝適正
    if "horse_ground_win_rate" in features:
        df["horse_ground_win_rate"] = (
            df.groupby(["horse_id", "ground"])["win_flag"]
            .transform(lambda x: x.shift(1).expanding().mean())
        )
    # 馬の馬場複勝適正
    if "horse_ground_place_rate" in features:
        df["horse_ground_place_rate"] = (
            df.groupby(["horse_id", "ground"])["place_flag"]
            .transform(lambda x: x.shift(1).expanding().mean())
        )
    # 馬の馬場適性数
    if "horse_weather_count" in features:
        df["horse_weather_count"] = (
            df.groupby(["horse_id", "weather"])["race_id"]
            .cumcount()
        )
    # 馬の天候単勝適正
    if "horse_weather_win_rate" in features:
        df["horse_weather_win_rate"] = (
            df.groupby(["horse_id", "weather"])["win_flag"]
            .transform(lambda x: x.shift(1).expanding().mean())
        )
    # 馬の天候複勝適正
    if "horse_weather_place_rate" in features:
        df["horse_weather_place_rate"] = (
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
    if "class_up_flag" in features:
        df["class_up_flag"] = (
            df["class_num"] > df["prev_class_num"]
        ).astype("int8")
    # クラスダウンフラグ
    if "class_down_flag" in features:
        df["class_down_flag"] = (
            df["class_num"] < df["prev_class_num"]
        ).astype("int8")
    # クラス差分
    df["class_diff"] = (
        df["class_num"]
        - df["prev_class_num"]
    )
    # 同じクラスの出場数
    if "class_count" in features:
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
    if "class_win_rate_rank" in features:
        df["class_win_rate_rank"] = (
            df.groupby("race_id")["class_win_rate"]
            .rank(ascending=False)
        )
    # クラス別の複勝率レース内順位
    if "class_place_rate_rank" in features:
        df["class_place_rate_rank"] = (
            df.groupby("race_id")["class_place_rate"]
            .rank(ascending=False)
        )
    # クラス別の単勝率トレンド
    if "class_win_rate_trend" in features:
        df["class_win_rate_trend"] = (
            df.groupby("class_num")["win_flag"]
            .transform(trend_3_3)
        )
    # クラス別の複勝率トレンド
    if "class_place_rate_trend" in features:
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
    if "class_win_rate_std_rank" in features:
        df["class_win_rate_std_rank"] = (
            df.groupby("race_id")["class_win_rate_std"]
            .rank(ascending=True)
        )
    # クラス別の複勝率安定性レース内順位
    if "class_place_rate_std_rank" in features:
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
    if "avg_last_corner_5_rank" in features:
        df["avg_last_corner_5_rank"] = (
            df.groupby("race_id")["avg_last_corner_5"]
            .rank(ascending=True)
        )
    # 前走の最後通過順の直近5戦の安定性
    if "last_corner_std_5" in features:
        df["last_corner_std_5"] = (
            df.groupby("horse_id")["last_corner"]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=2).std())
        )
    # 通過順の平均(中間特徴量)
    df["avg_corner"] = df["passing"].apply(get_avg_corner)
    # 前走の通過順の平均
    df["last_avg_corner"] = (
        df.groupby("horse_id")["avg_corner"]
        .shift(1)
    )
    # 前走の通過順の平均のレース内順位
    if "last_avg_corner_rank" in features:
        df["last_avg_corner_rank"] = (
            df.groupby("race_id")["last_avg_corner"]
            .rank(ascending=True)
        )
    # 通過順の平均の直近5戦の平均
    df["avg_avg_corner_5"] = (
        df.groupby("horse_id")["avg_corner"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    # 通過順の平均の直近5戦の平均レース内順位
    if "avg_avg_corner_5_rank" in features:
        df["avg_avg_corner_5_rank"] = (
            df.groupby("race_id")["avg_avg_corner_5"]
            .rank(ascending=True)
        )
    # 通過順の平均の直近5戦の安定性
    if "avg_avg_corner_5_std" in features:
        df["avg_avg_corner_5_std"] = (
            df.groupby("horse_id")["avg_corner"]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=2).std())
        )
    # 脚質
    df["running_style"] = df["avg_avg_corner_5"].apply(style)
    # 脚質フラグ
    df["front_runner_flag"] = (
        df["running_style"] == 1
    ).astype(int)
    # 脚質の安定性
    if "running_style_std_5" in features:
        df["running_style_std_5"] = (
            df.groupby("horse_id")["running_style"]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=2).std())
        )
    # 同じレースの逃げ馬の数
    df["race_front_runner_count"] = (
        df.groupby("race_id")["front_runner_flag"]
        .transform("sum")
    )
    # 同じレースの逃げ馬の割合
    df["race_front_runner_rate"] = (
        df["race_front_runner_count"]
        / df["field_size"]
    )
    # 同じ馬の過去の逃げ馬の数
    if "horse_front_runner_count" in features:
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
    #ペースマッチ
    df["pace_match"] = df["front_runner_rate"] - df["race_front_runner_rate"]
    # ペースマッチの直近5戦の平均
    df["avg_pace_match_5"] = (
        df.groupby("horse_id")["pace_match"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    # ペースマッチの直近5戦の平均レース内順位
    if "avg_pace_match_5_rank" in features:
        df["avg_pace_match_5_rank"] = (
            df.groupby("race_id")["avg_pace_match_5"]
            .rank(ascending=True)
        )
    # 通過順を距離ごとに平均
    df["avg_corner_distance"] = (
        df.groupby(["horse_id", "distance"])["avg_corner"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    # 通過順を距離ごとに平均レース内順位
    if "avg_corner_distance_rank" in features:
        df["avg_corner_distance_rank"] = (
            df.groupby("race_id")["avg_corner_distance"]
            .rank(ascending=True)
        )
    # 通過順をもとにした推進力（中間特徴量）
    df["corner_progress"] = df["passing"].apply(get_corner_progress)
    # 前走の通過順をもとにした推進力
    if "last_corner_progress" in features:
        df["last_corner_progress"] = (
            df.groupby("horse_id")["corner_progress"]
            .shift(1)
        )
    # 通過順をもとにした推進力の直近5戦の平均
    df["avg_corner_progress_5"] = (
        df.groupby("horse_id")["corner_progress"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    # 通過順をもとにした推進力の直近5戦の平均レース内順位
    if "avg_corner_progress_5_rank" in features:
        df["avg_corner_progress_5_rank"] = (
            df.groupby("race_id")["avg_corner_progress_5"]
            .rank(ascending=True)
        )
    # 通過順と頭数をもとにした推進力のレート（中間特徴量）
    df["corner_progress_rate"] = (
        df["corner_progress"] / df["field_size"]
    )
    # 前走の通過順と頭数をもとにした推進力のレート
    if "last_corner_progress_rate" in features:
        df["last_corner_progress_rate"] = (
            df.groupby("horse_id")["corner_progress_rate"]
            .shift(1)
        )
    # 通過順と頭数をもとにした推進力のレートの直近5戦の平均
    df["avg_corner_progress_rate_5"] = (
        df.groupby("horse_id")["corner_progress_rate"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    # 通過順と頭数をもとにした推進力のレートの直近5戦の平均レース内順位
    if "avg_corner_progress_rate_5_rank" in features:
        df["avg_corner_progress_rate_5_rank"] = (
            df.groupby("race_id")["avg_corner_progress_rate_5"]
            .rank(ascending=True)
        )

    # 前走の最終コーナーから着順までの伸び
    df["last_finish_kick"] =  df["last_last_corner"] - df["last_rank"]
    # 直近5戦の平均最終コーナーから着順までの伸び
    df["avg_finish_kick_5"] =  (
        df.groupby("horse_id")["last_finish_kick"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    # 直近5戦の平均最終コーナーから着順までの伸びレース内順位
    if "avg_finish_kick_5_rank" in features:
        df["avg_finish_kick_5_rank"] = (
            df.groupby("race_id")["avg_finish_kick_5"]
            .rank(ascending=True)
        )
    # 直近5戦の平均最終コーナーから着順までの伸びの安定性
    if "avg_finish_kick_5_std" in features:
        df["avg_finish_kick_5_std"] = (
            df.groupby("horse_id")["last_finish_kick"]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=2).std())
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
    if "jockey_win_rate_30d_rank" in features:
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
    if "jockey_place_rate_30d_rank" in features:
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
    if "jockey_win_rate_365d_rank" in features:
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
    if "jockey_place_rate_365d_rank" in features:
        df["jockey_place_rate_365d_rank"] = (
            df.groupby("race_id")["jockey_place_rate_365d"]
            .rank(ascending=False)
        )
    # 騎手コース単勝率
    if "jockey_course_win_rate" in features:
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
    if "jockey_course_place_rate" in features:
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
    if "jockey_distance_win_rate" in features:
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
    if "jockey_distance_place_rate" in features:
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
    if "jockey_distance_win_rate_std_rank" in features:
        df["jockey_distance_win_rate_std_rank"] = (
            df.groupby("race_id")["jockey_distance_win_rate_std"]
            .rank(ascending=True)
        )
    # 騎手距離複勝安定性レース内順位
    if "jockey_distance_place_rate_std_rank" in features:
        df["jockey_distance_place_rate_std_rank"] = (
            df.groupby("race_id")["jockey_distance_place_rate_std"]        
            .rank(ascending=True)
        )
    # 騎手距離単勝トレンド
    if "jockey_distance_win_rate_trend" in features:
        df["jockey_distance_win_rate_trend"] = (
            df.groupby(
                ["jockey_id", "distance"]
            )["win_flag"]
            .transform(trend_3_3)
        )
    # 騎手距離複勝トレンド
    if "jockey_distance_place_rate_trend" in features:
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
    if "jockey_win_rate_trend_rank" in features:
        df["jockey_win_rate_trend_rank"] = (
            df.groupby("race_id")["jockey_win_rate_trend"]
            .rank(ascending=False)
        )
    # 騎手の複勝率トレンドレース内順位
    if "jockey_place_rate_trend_rank" in features:
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
    if "jockey_win_rate_365d_diff" in features:
        df["jockey_win_rate_365d_diff"] = (
            df["jockey_win_rate_365d"] - df["jockey_win_rate_365d_race_avg"]
        )
    # 騎手の過去1年複勝率レース内平均(中間特徴量)
    df["jockey_place_rate_365d_race_avg"] = (
        df.groupby("race_id")["jockey_place_rate_365d"]
        .transform("mean")
    )
    # 騎手の過去1年複勝率とレース内平均の差分
    if "jockey_place_rate_365d_diff" in features:
        df["jockey_place_rate_365d_diff"] = (
            df["jockey_place_rate_365d"] - df["jockey_place_rate_365d_race_avg"]
        )
    # 騎手の同じ馬に騎乗した回数
    if "jockey_horse_count" in features:
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
    if "jockey_horse_win_rate_rank" in features:
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
    if "jockey_horse_place_rate_rank" in features:
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
    if "trainer_win_rate_365d_rank" in features:
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
    if "trainer_place_rate_365d_rank" in features:
        df["trainer_place_rate_365d_rank"] = (
            df.groupby("race_id")["trainer_place_rate_365d"]
            .rank(ascending=False)
        )
    # 調教師コース単勝率
    if "trainer_course_win_rate" in features:
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
    if "trainer_course_place_rate" in features:
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
    if "trainer_win_rate_365d_std" in features:
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
    if "trainer_place_rate_365d_std" in features:
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
    for col in cat_cols:
        df[col] = df[col].astype("category")
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

    return x_train, y_train, group_train, x_test, y_test, test_df