import requests
import re
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import time
import unicodedata

def get_race_data(race_id, response, engine):

    soup = BeautifulSoup(response.text, "lxml")

    # レース名
    race_name = soup.select_one("h1.RaceName").get_text(strip=True)

    # レース情報
    race_data_01 = soup.select_one("div.RaceData01")
    if not race_data_01:
        print(f"{race_id}: レース情報取得失敗")
        return

    race_text = race_data_01.get_text(" ", strip=True)

    # 日付
    kaisai_link = soup.select_one('a[href*="kaisai_date="]')
    date = None

    if kaisai_link:
        match = re.search(
            r"kaisai_date=(\d{8})",
            kaisai_link["href"]
        )
        if match:
            date = datetime.strptime(
                match.group(1),
                "%Y%m%d"
            )
    else:
        print(f"{race_id}: 開催日取得失敗")
        return

    # 開催場
    race_data_02 = soup.select_one("div.RaceData02")

    spans = race_data_02.select("span")

    place = spans[1].get_text(strip=True)

    # レース番号
    race_no = int(str(race_id)[-2:])

    # コース
    course_match = re.search(r"(芝|ダ|障)", race_text)

    course = course_match.group(1)

    # 距離
    distance_match = re.search(r"(\d+)m", race_text)

    distance = int(distance_match.group(1))

    # 天候
    weather_match = re.search(
    r"天候[:：]?\s*(\S+)",
    race_text
)

    weather = weather_match.group(1)

    # 馬場    
    ground_match = re.search(
    r"馬場[:：]?\s*(\S+)",
    race_text
    )

    ground = ground_match.group(1)

    # クラス
    race_class = None
    # ==========================
    # 重賞・リステッド判定
    # ==========================
    race_name_txt = soup.select_one("h1.RaceName")

    if race_name_txt:
        icon_classes = set()

        for span in race_name_txt.select("span"):
            icon_classes.update(span.get("class", []))

        if "Icon_GradeType1" in icon_classes:
            race_class = "G1"

        if "Icon_GradeType2" in icon_classes:
            race_class = "G2"

        if "Icon_GradeType3" in icon_classes:
            race_class = "G3"

        if "Icon_GradeType4" in icon_classes:
            race_class = "重賞"

        if "Icon_GradeType15" in icon_classes:
            race_class = "L"

        if "Icon_GradeType10" in icon_classes:
            race_class = "JG1"

        if "Icon_GradeType11" in icon_classes:
            race_class = "JG2"

        if "Icon_GradeType12" in icon_classes:
            race_class = "JG3"

    # ==========================
    # OP以下
    # ==========================
    if not race_class:
        race_data02 = soup.select_one(".RaceData02")

        if race_data02:

            spans = [
                unicodedata.normalize("NFKC", s.get_text(strip=True))
                for s in race_data02.select("span")
            ]

            CLASS_LIST = [
                "オープン",
                "3勝クラス",
                "2勝クラス",
                "1勝クラス",
                "未勝利",
                "新馬",
            ]

            race_class = next(
                (s for s in spans if s in CLASS_LIST),
                None
            )

    race = {
        "race_id": int(race_id),
        "date": date,
        "class": race_class,
        "place": place,
        "race_no": race_no,
        "race_name": race_name,
        "course": course,
        "distance": distance,
        "weather": weather,
        "ground": ground,
        "created_at": pd.Timestamp.now(),
    }


    df = pd.DataFrame([race])
    df.to_sql(
        "races",
        con=engine,
        if_exists="append",
        index=False
    )

    payout_rows = []

    for tr in soup.select("table.Payout_Detail_Table tr"):
        bet_type = tr.select_one("th").get_text(strip=True)

        # 馬番取得
        results = [
            span.get_text(strip=True)
            for span in tr.select("td.Result span")
            if span.get_text(strip=True)
        ]

        # 払戻取得
        payout_text = (
            tr.select_one("td.Payout span")
            .get_text("\n", strip=True)
        )

        payouts = [
            int(
                x.replace("円", "")
                .replace(",", "")
            )
            for x in payout_text.split("\n")
            if x.strip()
        ]

        # 組み合わせ生成
        if bet_type in ["単勝", "複勝"]:
            combinations = results

        elif bet_type in ["枠連", "馬連", "馬単", "ワイド"]:
            combinations = [
                f"{results[i]}-{results[i+1]}"
                for i in range(0, len(results), 2)
            ]

        elif bet_type in ["3連複", "3連単"]:
            combinations = [
                "-".join(results[i:i+3])
                for i in range(0, len(results), 3)
            ]

        else:
            continue

        # レコード作成
        for combination, payout in zip(combinations, payouts):

            payout_rows.append({
                "race_id": int(race_id),
                "bet_type": bet_type,
                "combination": combination,
                "payout": payout,
                "created_at": pd.Timestamp.now()
            })
    payout_df = pd.DataFrame(payout_rows)
    payout_df.to_sql(
        "race_payouts",
        con=engine,
        if_exists="append",
        index=False
    )
    print(f"{race_id} のレース情報を保存しました")