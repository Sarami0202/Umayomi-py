from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup
from get_data.get_result_data import get_result_data
from get_data.get_horse_data import get_horse_data_all
import pandas as pd
import time

# 過去のレース結果とレース情報をDBに保存する関数

places = [
    "01",  # 札幌
    "02",  # 函館
    "03",  # 福島
    "04",  # 新潟
    "05",  # 東京
    "06",  # 中山
    "07",  # 中京
    "08",  # 京都
    "09",  # 阪神
    "10",  # 小倉
]  
# 環境変数の読み込み
load_dotenv()
# データベース接続のためのエンジンを作成
engine = create_engine( f"mysql+pymysql://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_DATABASE')}" )
# 既存のレースIDを取得して重複を避ける
existing_race_ids = pd.read_sql("SELECT DISTINCT race_id FROM races", engine)["race_id"].tolist()

# get_result_data(202408070611, engine)
# 過去のレース結果とレース情報をDBに保存する関数
year = 2025
stop = False
for place in places:

    for kai in range(1, 7):      # 開催回数

        for day in range(1, 13): # 開催日

            for race_no in range(1, 13): # 1R～12R

                race_id = int(
                    f"{year}"
                    f"{place}"
                    f"{kai:02}"
                    f"{day:02}"
                    f"{race_no:02}"
                )

                # if race_id < 202501030611:
                #     continue

                if race_id in existing_race_ids:
                    print(f"{race_id} は既にDBに存在するためスキップします")
                    continue
                
                print(f"{race_id} にアクセス中...")

                # レース情報をDBに保存
                access_flg = get_result_data(race_id, engine)
                if not access_flg:
                    print(f"{race_id} のレースデータの取得に失敗したため、スキップします")
                    continue
                elif access_flg == 403:
                    print(f"{race_id} にアクセスできませんでした。しばらく待ってから再試行してください。")
                    stop = True
                    break

            if stop:
                break
        if stop:
            break
    if stop:
        break

if stop:
    print("データ取得を中断しました。再度実行してください。")
else:
    print("全てのレースデータの取得が完了しました。")
    print("次に、馬の情報をDBに保存します...")
    # 馬の情報をDBに保存
    get_horse_data_all(engine)