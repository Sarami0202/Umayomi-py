from verify.verify_features import create_features
from verify.verify_ranker_model import verify_ranker_model
from verify.verify_classifier_model import verify_classifier_model
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import pandas as pd
 
# 検証実行

# # データ収集
# 特徴量生成
# train/testで精度検証
# 特徴量改善
# モデル検証　<-いまここ
# モデル最適化
# 最終的に全データで再学習
# モデル保存
# 次開催のレースを予測
# 定期的にモデルの再生成（例：週に1回など）

# カテゴリ変数
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
# タイム
# ==========================
"last_time",
"avg_time_5",
"last_time_distance",
"avg_time_distance",
"avg_time_distance_5",
"avg_time_distance_5_rank",
"last_time_course",
"avg_time_course",
"avg_time_course_5",
"avg_time_course_5_rank",
"last_time_diff",
"avg_last_time_diff",
"avg_last_time_diff_5",
"avg_last_time_diff_5_rank",
"last_time_diff_distance",
"avg_last_time_diff_distance",
"avg_last_time_diff_distance_5",
"avg_last_time_diff_distance_5_rank",
"best_last_time_diff_5",
"best_last_time_diff_5_rank",

# ==========================
# 前走上がり3F
# ==========================
"last_last3f",
"last_last3f_rank",
"last3f_change",
"last3f_change_rank",
"last3f_rank_change",
"last3f_trend",
"avg_last3f",
"avg_last3f_rank",
"avg_last3f_5",
"avg_last3f_5_diff",
"avg_last3f_rank_5",
"avg_last3f_distance",
"avg_last3f_distance_rank",
"avg_last3f_distance_std",
"avg_last3f_distance_diff",
"avg_last3f_course",
"avg_last3f_course_rank",
"last3f_std_5",
"avg_last3f_diff",
"best_last3f_5",
"best_last3f_5_rank",
"last3f_rank_rate",
"avg_last3f_rank_5_rate",
"avg_last3f_rank_5_rate_rank",
"last3f_percentile",
"avg_last3f_rank_5_percentile",
"avg_last3f_rank_5_percentile_rank",    
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
"last_corner_std_5",
"running_style",
"running_style_std_5",
"race_front_runner_count",
"race_front_runner_rate",
"horse_front_runner_count",
"front_runner_rate",
"avg_avg_corner_5",
"avg_avg_corner_5_rank",
"avg_avg_corner_5_std",
"last_avg_corner",
"last_avg_corner_rank",
"avg_corner_distance",
"avg_corner_distance_rank",
"last_corner_progress",
"avg_corner_progress_5",
"avg_corner_progress_5_rank",
"last_corner_progress_rate",
"avg_corner_progress_rate_5",
"avg_corner_progress_rate_5_rank",
"last_finish_kick",
"avg_finish_kick_5",
"avg_finish_kick_5_rank",
"avg_finish_kick_5_std",
"pace_match",
"avg_pace_match_5",
"avg_pace_match_5_rank",
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
# チェンジ
"course_changed",
"ground_changed",
"weather_changed",

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
"win_count",
"place_count",
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
"horse_weather_win_rate",
"horse_weather_place_rate",

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


if __name__ == "__main__":
    load_dotenv()

    engine = create_engine(
        f"mysql+pymysql://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@"
        f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_DATABASE')}"
    )

    bet_type= "複勝"  # 予測対象の賭式を指定


    # 特徴量生成
    x_train, y_train, group_train, x_test, y_test, test_df = create_features(
        engine,
        bet_type,
        features,
        cat_cols, 
        # 学習データの開始期間を指定
        "2024-01-01",
        # 学習データの終了期間を指定＆テストデータの開始期間を指定
        "2025-12-31",
        # 特徴量生成のための過去データの開始期間を指定 
        "2023-01-01")
    # Rankerモデルの検証
    verify_ranker_model(engine, bet_type,x_train, y_train, group_train, cat_cols,
                 x_test, y_test, test_df, features,'only')
    # Classifierモデルの検証
    verify_classifier_model(engine,  bet_type, x_train, y_train, cat_cols, x_test, y_test, test_df, features)