from verify.verify_features import create_features
from verify.verify_ranker_model import verify_ranker_model
from verify.verify_classifier_model import verify_classifier_model
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
 
# 検証実行

# # データ収集
# 特徴量生成
# train/testで精度検証
# 特徴量改善
# 最終的に全データで再学習[
# モデル保存
# 次開催のレースを予測
# 定期的にモデルの再生成（例：週に1回など）


if __name__ == "__main__":
    load_dotenv()

    engine = create_engine(
        f"mysql+pymysql://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@"
        f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_DATABASE')}"
    )

    bet_type= "複勝"  # 予測対象の賭式を指定

    # 特徴量生成
    x_train, y_train, group_train, group_test, cat_cols, x_test, y_test, test_df, features = create_features(
        engine, bet_type,
        # 学習データの開始期間を指定
        "2024-01-01",
        # 学習データの終了期間を指定＆テストデータの開始期間を指定
        "2025-12-31",
        # 特徴量生成のための過去データの開始期間を指定 
        "2023-01-01")
    # Rankerモデルの検証
    verify_ranker_model(engine, bet_type,x_train, y_train, group_train, cat_cols,
                 x_test, y_test, test_df, features)
    # Classifierモデルの検証
    # verify_classifier_model(engine,  bet_type, x_train, y_train, cat_cols, x_test, y_test, test_df
    #                         , features)