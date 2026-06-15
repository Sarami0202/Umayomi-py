import requests
import re
from bs4 import BeautifulSoup
from io import StringIO
import pandas as pd
import time
from get_data.get_race_data import get_race_data

def get_result_data(race_id, engine):        
    # レース結果のURLを構築
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}&rf=race_list"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    # HTTPリクエストを送信してHTMLを取得
    response = requests.get(
    url,
    headers=headers,
    timeout=30
    )
    response.encoding = response.apparent_encoding
    soup = BeautifulSoup(response.text, "lxml")
    table = soup.select_one("table.RaceTable01")

    print('2秒待機中...')
    time.sleep(2)
    
    if response.status_code != 200:
        print(f"{race_id} にアクセスできませんでした")
        return False
    if table is None:
        return False
    # テーブルのヘッダーを取得
    headers = [
        th.get_text(strip=True)
        for th in table.select("tr th")
    ]
    idx = {
        name: headers.index(name)
        for name in [
            "着順",
            "馬番",
            '枠',
            "斤量",
            "騎手",
            "タイム",
            "コーナー通過順",
            "後3F",
            "単勝オッズ",
            "人気",
            "馬体重(増減)",
            "性齢",
        ]
    }
    # テーブルのデータを行ごとに処理
    rows = []
    for tr in table.select("tr")[1:]:
        cols = [td.get_text(strip=True) for td in tr.find_all("td")]

        if len(cols) < len(headers):
            continue

        # 馬ID取得
        horse_link = tr.select_one('a[href*="/horse/"]')

        if not horse_link:
            continue

        horse_id = int(
            horse_link["href"].rstrip("/").split("/")[-1]
        )

        # 騎手ID取得
        jockey_link = tr.select_one('a[href*="/jockey/result/recent/"]')
        if not jockey_link:
            continue
        jockey_id = int(
            jockey_link["href"].rstrip("/").split("/")[-1]
        )
        # 馬体重
        
        body_weight_text = cols[idx["馬体重(増減)"]]

        body_weight = None
        body_weight_diff = None

        # 馬体重だけ取得
        weight_match = re.search(r"(\d+)", body_weight_text)

        if weight_match:
            body_weight = int(weight_match.group(1))

        # 増減取得
        diff_match = re.search(r"\(([+-]?\d+)\)", body_weight_text)

        if diff_match:
            body_weight_diff = int(diff_match.group(1))
        #馬性齢から馬年齢を取得
        age=int(re.search(r"(\d+)", cols[idx["性齢"]]).group(1))

        # 着順が数字でない場合はスキップ
        rank = cols[idx["着順"]]

        if not rank.isdigit():
            continue

        # 上り3Fをfloatに変換できない場合はNoneにする
        last3f = cols[idx["後3F"]]
        try:
            last3f = float(last3f)
        except:
            last3f = None

        # 騎手の減量マーク除去
        jockey = cols[idx["騎手"]]
        jockey = jockey.lstrip("☆▲△◇★")

        rows.append({
            "race_id": int(race_id),
            "horse_id": horse_id,
            "rank": int(cols[idx["着順"]]),
            "frame_no": int(cols[idx["枠"]]),
            "horse_no": int(cols[idx["馬番"]]),
            "weight": float(cols[idx["斤量"]]),
            "jockey": jockey,
            "jockey_id": jockey_id,
            "time": cols[idx["タイム"]],
            "passing": cols[idx["コーナー通過順"]],
            "last3f": float(cols[idx["後3F"]]),
            "odds": float(cols[idx["単勝オッズ"]]),
            "popularity": int(cols[idx["人気"]]),
            "body_weight": body_weight,
            "body_weight_diff": body_weight_diff,
            "age": age,
            "created_at": pd.Timestamp.now(),
        })

    df = pd.DataFrame(rows)

    # RaceResultデータベースに接続してデータを保存
    df.to_sql(
        "race_results",
        con=engine,
        if_exists="append",
        index=False
    )
    print(f"レースID {race_id} の結果データを保存しました")

    
    # レースの情報をDBに保存
    get_race_data(race_id, response, engine)
    return True
        
        