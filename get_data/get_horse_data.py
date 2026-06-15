from sqlalchemy import text
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import time

# 馬の情報を取得してDBに保存する関数
def get_horse_data(horse_id, engine):

    url = f"https://db.netkeiba.com/horse/{horse_id}/"

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, headers=headers)
    response.encoding = response.apparent_encoding
    
    print('3秒待機中...')
    time.sleep(3)
    soup = BeautifulSoup(response.text, "lxml")
    # 馬名
    horse_name = soup.select_one("title").get_text().split(" | ")[0]
    # 性別
    sex = None
    sex_text = soup.select_one(".horse_title .txt_01").get_text(" ", strip=True)
    # 調教師
    trainer = None
    profile_table = soup.select_one("table.db_prof_table")
    # 調教師ID取得
    trainer_link = profile_table.select_one(
        'a[href*="/trainer/"]'
    )
    trainer_id = trainer_link["href"].rstrip("/").split("/")[-1]
    print(f"調教師ID: {trainer_id}")
    # 性別は馬名の後ろ
    if sex_text:
        sex = re.search(r'[牡牝セ]', sex_text).group()

    # 調教師はプロフィールテーブルから取得
    if profile_table:
        trainer_link = profile_table.select_one(
            'a[href*="/trainer/"]'
        )

        if trainer_link:
            trainer = trainer_link.get_text(strip=True)

    horse = {
        "horse_id": horse_id,
        "horse_name": horse_name,
        "sex": sex,
        "trainer": trainer,
        "trainer_id": trainer_id,
        "created_at": pd.Timestamp.now(),
    }

    # DBに保存
    df = pd.DataFrame([horse])
    df.to_sql(
        "horses",
        con=engine,
        if_exists="append",
        index=False
    )
    print(f"{horse_id} の情報を保存しました")

# DBに保存されていない全ての馬の情報を取得して保存する関数
def get_horse_data_all(engine):
    # DBから全ての馬IDを取得
    horse_ids = pd.read_sql(
        """
        SELECT DISTINCT horse_id
        FROM race_results
        """,
        engine
    )
    # DBに保存されていない馬IDを取得
    horse_ids = pd.read_sql(
    """
    SELECT DISTINCT rr.horse_id
    FROM race_results rr
    LEFT JOIN horses h
        ON rr.horse_id = h.horse_id
    WHERE h.horse_id IS NULL
    """,
    engine
    )

    for horse_id in horse_ids["horse_id"]:
        print(f"{horse_id} の情報を保存中...")
        get_horse_data(horse_id, engine)
        
    print("全ての馬の情報を保存しました")
