"""Parsers on small inline fixtures copied from real NSE file layouts."""
import datetime as dt
import io
import zipfile

import numpy as np

from impi.ingest import corp_actions as CA
from impi.ingest import indices, nse_bhav, nse_fo, participant, surveillance

SEC = (b"SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, LAST_PRICE, CLOSE_PRICE, "
       b"AVG_PRICE, TTL_TRD_QNTY, TURNOVER_LACS, NO_OF_TRADES, DELIV_QTY, DELIV_PER\n"
       b"20MICRONS, EQ, 01-Jun-2020, 26.15, 27.00, 27.50, 26.30, 27.00, 26.90, 26.97, 35038, 9.45, 497, 23295, 66.48\n"
       b"ABC, BE, 01-Jun-2020, 10.00, 10.00, 10.50, 9.90, 10.10, 10.20, 10.10, 1000, 0.10, 5, -, -\n")


def _zip(name: str, data: bytes) -> bytes:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr(name, data)
    return b.getvalue()


CM = _zip("cm01JUN2015bhav.csv",
          b"SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,TOTALTRADES,ISIN,\n"
          b"20MICRONS,EQ,33.8,33.8,32.05,32.25,32.45,33.35,17196,561001.7,01-JUN-2015,154,INE144J01027,\n"
          b"XYZ,EQ,10,11,9,10.5,10.5,10,500,5000,01-JUN-2015,3,INE000000000,\n")
MTO = (b"Security Wise Delivery Position - Compulsory Rolling Settlement\n"
       b"10,MTO,01062015,275467372,0001534\n"
       b"Trade Date <01-JUN-2015>,Settlement Type <N>,Settlement No <2015101>,Settlement Date <03-JUN-2015>\n"
       b"Record Type,Sr No,Name of Security,Quantity Traded,Deliverable Quantity(gross across client level),% of Deliverable Quantity to Traded Quantity\n"
       b"20,1,20MICRONS,EQ,17196,9000,52.34\n"
       b"20,2,XYZ,EQ,499,100,20.04\n")


def test_sec_bhav_parse_and_date():
    assert nse_bhav.sec_file_date(SEC) == dt.date(2020, 6, 1)
    df = nse_bhav.parse_sec(SEC)
    r = df.iloc[0]
    assert r["symbol"] == "20MICRONS" and r["close"] == 26.90 and r["volume"] == 35038
    assert abs(r["turnover_rs"] - 945000) < 1e-6 and r["deliv_pct"] == 66.48
    assert np.isnan(df.iloc[1]["deliv_qty"])                 # '-' -> missing


def test_legacy_cm_mto_merge_and_volume_guard():
    assert nse_bhav.cm_file_date(CM) == dt.date(2015, 6, 1)
    assert nse_bhav.mto_file_date(MTO) == dt.date(2015, 6, 1)
    df = nse_bhav.merge_legacy(nse_bhav.parse_cm(CM), nse_bhav.parse_mto(MTO))
    a = df[df.symbol == "20MICRONS"].iloc[0]
    assert a["deliv_qty"] == 9000 and a["deliv_pct"] == 52.34 and a["volume"] == 17196
    b = df[df.symbol == "XYZ"].iloc[0]                       # MTO qty 499 != bhav 500 -> delivery dropped
    assert np.isnan(b["deliv_qty"])


def test_html_and_wrong_date_files_rejected():
    assert nse_bhav.sec_file_date(b"<!DOCTYPE html><html>holiday</html>") is None
    assert nse_bhav.cm_file_date(b"not a zip") is None


def test_fo_formats_agree():
    legacy = _zip("fo.csv", b"INSTRUMENT,SYMBOL,EXPIRY_DT,STRIKE_PR,OPTION_TYP,OPEN,HIGH,LOW,CLOSE,SETTLE_PR,CONTRACTS,VAL_INLAKH,OPEN_INT,CHG_IN_OI,TIMESTAMP,\n"
                         b"FUTSTK,ABB,25-Jul-2024,0,XX,1,1,1,8700,8725.9,10,1,1380375,100,05-JUL-2024,\n"
                         b"FUTSTK,ABB,29-Aug-2024,0,XX,1,1,1,8790,8790.25,10,1,36875,5,05-JUL-2024,\n"
                         b"OPTSTK,ABB,25-Jul-2024,9000,CE,1,1,1,1,1,1,1,673250,0,05-JUL-2024,\n"
                         b"OPTSTK,ABB,25-Jul-2024,8000,PE,1,1,1,1,1,1,1,356125,0,05-JUL-2024,\n")
    hdr = (b"TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,XpryDt,FininstrmActlXpryDt,StrkPric,OptnTp,"
           b"FinInstrmNm,OpnPric,HghPric,LwPric,ClsPric,LastPric,PrvsClsgPric,UndrlygPric,SttlmPric,OpnIntrst,ChngInOpnIntrst,"
           b"TtlTradgVol,TtlTrfVal,TtlNbOfTxsExctd,SsnId,NewBrdLotQty,Rmks,Rsvd1,Rsvd2,Rsvd3,Rsvd4\n")
    rows = (b"2024-07-05,2024-07-05,FO,NSE,STF,1,,ABB,,2024-07-25,2024-07-25,,,X,1,1,1,8700,1,1,8700,8725.9,1380375,100,10,1,1,F1,125,,,,,\n"
            b"2024-07-05,2024-07-05,FO,NSE,STF,2,,ABB,,2024-08-29,2024-08-29,,,X,1,1,1,8790,1,1,8700,8790.25,36875,5,10,1,1,F1,125,,,,,\n"
            b"2024-07-05,2024-07-05,FO,NSE,STO,3,,ABB,,2024-07-25,2024-07-25,9000,CE,X,1,1,1,1,1,1,8700,1,673250,0,1,1,1,F1,125,,,,,\n"
            b"2024-07-05,2024-07-05,FO,NSE,STO,4,,ABB,,2024-07-25,2024-07-25,8000,PE,X,1,1,1,1,1,1,8700,1,356125,0,1,1,1,F1,125,,,,,\n")
    udiff = _zip("u.csv", hdr + rows)
    assert nse_fo.legacy_file_date(legacy) == nse_fo.udiff_file_date(udiff) == dt.date(2024, 7, 5)
    a = nse_fo.aggregate_stock(nse_fo.parse_legacy(legacy)).iloc[0]
    b = nse_fo.aggregate_stock(nse_fo.parse_udiff(udiff)).iloc[0]
    for c in ["fut_oi", "near_settle", "next_settle", "call_oi", "put_oi"]:
        assert a[c] == b[c], c
    assert a["fut_oi"] == 1380375 + 36875 and a["near_expiry"].day == 25


def test_participant_ban_index():
    po = (b'""Participant wise Open Interest (no. of contracts) in Equity Derivatives as on Oct 08, 2026"",,,\n'
          b"Client Type,Future Index Long,Future Index Short,Future Stock Long,Future Stock Short       ,Option Index Call Long,"
          b"Option Index Put Long,Option Index Call Short,Option Index Put Short,Option Stock Call Long,Option Stock Put Long,"
          b"Option Stock Call Short,Option Stock Put Short,Total Long Contracts,Total Short Contracts\n"
          b"FII,10,20,30,40,1,1,1,1,1,1,1,1,100,90\nTOTAL,100,100,200,200,1,1,1,1,1,1,1,1,1000,1000\n")
    assert participant.file_date(po) == dt.date(2026, 10, 8)
    p = participant.parse(po, dt.date(2026, 10, 8))
    assert p[p.participant == "FII"].iloc[0]["fut_stk_short"] == 40
    ban = b"Securities in Ban For Trade Date 09-OCT-2026:\n1,AMBUJACEM\n2,BANDHANBNK\n"
    assert surveillance.ban_file_date(ban) == dt.date(2026, 10, 9)
    assert list(surveillance.parse_ban(ban, dt.date(2026, 10, 9)).symbol) == ["AMBUJACEM", "BANDHANBNK"]
    nil = b"Securities in Ban For Trade Date 01-JUN-2015:NIL"
    assert surveillance.parse_ban(nil, dt.date(2015, 6, 1)).empty
    old = b"Securities in Ban For Trade Date 02-Jan-2014,\n1,APOLLOTYRE\n"
    assert surveillance.ban_file_date(old) == dt.date(2014, 1, 2)
    idx = (b"Index Name,Index Date,Open Index Value,High Index Value,Low Index Value,Closing Index Value,Points Change,Change(%),Volume,Turnover (Rs. Cr.),P/E,P/B,Div Yield\n"
           b"CNX Nifty,02/01/2014,6301.25,6358.3,6211.3,6221.15,1,1,1,1,1,1,1\nCNX 500,02/01/2014,1,2,0.5,4848.95,1,1,1,1,1,1,1\n")
    assert indices.file_date(idx) == dt.date(2014, 1, 2)
    i = indices.parse(idx, dt.date(2014, 1, 2)).set_index("index")
    assert i.loc["nifty50", "close"] == 6221.15 and i.loc["nifty500", "close"] == 4848.95


def test_corporate_action_subjects():
    cases = {
        "Bonus 1:1": ("bonus", 0.5), "Bonus 3:2": ("bonus", 0.4),
        "Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share": ("split", 0.2),
        "Face Value Split Rs.10/- To Re.1/- Per Share": ("split", 0.1),
        "Bonus 1:1 / Face Value Split From 10/- To Face Value 2/-": ("bonus+split", 0.1),
        "Consolidation Of Equity Shares From Re 1 Per Share To Rs 10 Per Share": ("consolidation", 10.0),
        "Annual General Meeting/Dividend - Rs 10 Per Share/Face Value Split (Sub-Division) - From Rs 10/- Per Share To Re 1/- Per Share": ("split", 0.1),
        "Scheme Of Demerger": ("demerger", None),
        " Fv Splt Frm Rs 10 To Rs 2": ("split", 0.2),
        "Bonus Preference Shares 21:1": (None, None),
        "Dividend - Rs 5 Per Share": (None, None),
    }
    for s, (kind, f) in cases.items():
        k, ff = CA.parse_subject(s)
        assert k == kind, s
        if f is not None:
            assert abs(ff - f) < 1e-9, s


def test_symbol_change_chain_and_conflict():
    import pandas as pd
    df = pd.DataFrame({"date": pd.to_datetime(["2015-01-01", "2017-01-01", "2021-01-01", "2016-01-01"]),
                       "symbol": ["A", "B", "C", "Z"]})
    ch = pd.DataFrame({"old": ["A", "B", "Y"], "new": ["B", "C", "Z"],
                       "date": pd.to_datetime(["2016-06-01", "2020-01-01", "2019-01-01"])})
    out, log = CA.apply_symbol_changes(df, ch)
    assert list(out.symbol) == ["C", "C", "C", "Z"]


def test_participant_file_without_header_row():
    po = (b'"Participant wise Open Interest (no. of contracts) in Equity Derivatives as on February 24, 2014",,,,,,,,,,,,,,\n'
          b"Client,164528,209021,642990,217739,481786,745687,686388,883452,218869,118370,134204,84901,2372230,2215705\n"
          b"TOTAL,471850,471850,1,1,1,1,1,1,1,1,1,1,5127997,5127997\n")
    assert participant.file_date(po) == dt.date(2014, 2, 24)
    p = participant.parse(po, dt.date(2014, 2, 24)).set_index("participant")
    assert p.loc["Client", "fut_idx_long"] == 164528 and p.loc["TOTAL", "total_short"] == 5127997


def test_recent_missing_files_are_not_remembered(tmp_path, monkeypatch):
    import datetime as dt
    from impi.ingest import download as D
    monkeypatch.setattr(D, "NO_DATA", tmp_path / "no_data.csv")
    dl = D.Downloader()
    dl.no_data = set()
    today = dt.date.today().isoformat()
    old = (dt.date.today() - dt.timedelta(days=30)).isoformat()
    dl.mark_no_data(today, "fo_udiff", "404/html")
    dl.mark_no_data(old, "fo_udiff", "404/html")
    assert (today, "fo_udiff") not in dl.no_data and (old, "fo_udiff") in dl.no_data
