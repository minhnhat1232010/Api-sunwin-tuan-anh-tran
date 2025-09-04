import sqlite3
import requests
import math
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime, timezone
import numpy as np
from collections import Counter
import logging

# Cấu hình cơ bản
DB = "tx_predict.db"
SOURCE_API = "https://ahihidonguoccut-2b5i.onrender.com/mohobomaycai"
TELE_ID = "Văn Nhật ComeBack Hâhhahha"
PATTERN_LENGTH = 20
HISTORY_LIMIT = 500

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TX Predictor API - Advanced Ensemble",
    description="Advanced Tai/Xiu prediction with pattern analysis and ensemble AI",
    version="2.0"
)

# ---------- Database Helpers ----------
def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session INTEGER,
        d1 INTEGER, d2 INTEGER, d3 INTEGER,
        total INTEGER,
        result TEXT,
        ts TEXT
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS meta (
        k TEXT PRIMARY KEY,
        v TEXT
    )
    """)
    conn.commit()
    conn.close()

def save_round(session: int, d: list, total: int, result: str, ts: str):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute(
        "INSERT INTO history (session,d1,d2,d3,total,result,ts) VALUES (?,?,?,?,?,?,?)",
        (session, d[0], d[1], d[2], total, result, ts)
    )
    conn.commit()
    conn.close()

def get_last_n(n: int = 200) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT session,d1,d2,d3,total,result,ts FROM history ORDER BY id DESC LIMIT ?", (n,))
    rows = c.fetchall()
    conn.close()
    out = []
    for r in rows:
        out.append({
            "session": r[0],
            "dices": [r[1], r[2], r[3]],
            "total": r[4],
            "result": r[5],
            "ts": r[6]
        })
    return out[::-1]  # Chronological order

def get_pattern_last_k(k: int = 20) -> str:
    rows = get_last_n(k)
    return "".join(["T" if r["result"].upper().startswith("T") else "X" for r in rows])

# ---------- Feature Engineering ----------
def extract_features(history: List[Dict]) -> Dict[str, Any]:
    features = {}
    
    # A. Window/lag features
    for k in [10, 30, 100]:
        last_k = history[-k:] if len(history) >= k else history
        p_tai = sum(1 for r in last_k if r["result"].upper().startswith("T")) / len(last_k) if last_k else 0.5
        totals = [r["total"] for r in last_k]
        features[f"p_tai_last_{k}"] = p_tai
        features[f"mean_total_last_{k}"] = np.mean(totals) if totals else 0
        features[f"std_total_last_{k}"] = np.std(totals) if totals else 0

    # B. Streak features
    if history:
        last_result = "T" if history[-1]["result"].upper().startswith("T") else "X"
        run_len = 1
        for i in range(len(history)-2, -1, -1):
            if history[i]["result"].upper().startswith(last_result):
                run_len += 1
            else:
                break
        features["current_run_len"] = run_len
        features["current_run_type"] = last_result
        max_run = max(
            (sum(1 for _ in g) for k, g in itertools.groupby(
                "T" if r["result"].upper().startswith("T") else "X" for r in history[-100:]
            )), default=0
        )
        features["max_run_len_last_100"] = max_run

    # C. Transition features
    trans = {"T->T": 0, "T->X": 0, "X->T": 0, "X->X": 0}
    for i in range(1, len(history)):
        a = "T" if history[i-1]["result"].upper().startswith("T") else "X"
        b = "T" if history[i]["result"].upper().startswith("T") else "X"
        trans[f"{a}->{b}"] += 1
    features["trans_rates"] = trans

    # D. Dice-level features
    counts = Counter(d for r in history[-30:] for d in r["dices"])
    for i in range(1, 7):
        features[f"freq_face_{i}_last_30"] = counts.get(i, 0) / (30 * 3) if history else 0
    high_faces = sum(counts.get(i, 0) for i in [4, 5, 6])
    features["p_high_faces_last_30"] = high_faces / (30 * 3) if history else 0

    # E. Entropy
    seq = [1 if r["result"].upper().startswith("T") else 0 for r in history[-50:]]
    p = sum(seq) / len(seq) if seq else 0.5
    ent = -(p * math.log2(p + 1e-10) + (1-p) * math.log2(1-p + 1e-10)) if 0 < p < 1 else 0
    features["entropy_last_50"] = ent

    # F. Time features
    if len(history) >= 2:
        t1 = datetime.fromisoformat(history[-1]["ts"])
        t2 = datetime.fromisoformat(history[-2]["ts"])
        features["delta_time_last"] = (t1 - t2).total_seconds()
        features["hour_bucket"] = t1.hour // 4

    return features

# ---------- Pattern Analysis ----------
def detect_pattern(history: List[Dict], k: int = 20) -> Dict[str, Any]:
    pattern = get_pattern_last_k(k)
    patterns = {
        "1-1": r"TX|XT",
        "1-2-1": r"TXXT|XTXX",
        "2-1-2": r"TTXTT|XTTXX",
        "3-1": r"TTTX|XXXT",
        "1-3": r"TXXX|XTTT",
        "2-2": r"TTXX|XXTT",
        "2-3": r"TTXXX|XXTTT",
        "3-2": r"TTTXX|XXTTT",
        "4-1": r"TTTTX|XXXXT",
        "1-4": r"TXXXX|XTTTT"
    }
    
    matched_patterns = []
    follow_counts = {"T": 0, "X": 0}
    seq = "".join(["T" if r["result"].upper().startswith("T") else "X" for r in history])
    
    for name, regex in patterns.items():
        for match in re.finditer(regex, seq):
            start, end = match.span()
            if end < len(seq):
                next_char = seq[end]
                follow_counts[next_char] += 1
                if end == len(seq) - 1:  # Match at the end
                    matched_patterns.append(name)
    
    denom = sum(follow_counts.values())
    p_tai = follow_counts["T"] / denom if denom > 0 else 0.5
    
    return {
        "current_pattern": pattern,
        "matched_patterns": matched_patterns,
        "follow_counts": follow_counts,
        "p_tai": p_tai
    }

# ---------- Advanced Analysers ----------
def analyser_freq_window(history: List[Dict], window=30):
    last = history[-window:] if len(history) >= 1 else history
    count_t = sum(1 for r in last if r["result"].upper().startswith("T"))
    p = count_t / len(last) if last else 0.5
    return {
        "name": f"freq_window_{window}",
        "p_tai": p,
        "explain": f"Frequency of TAI in last {len(last)} rounds: {count_t}/{len(last)} = {p:.3f}"
    }

def analyser_markov_order1(history: List[Dict]):
    if len(history) < 2:
        return {"name": "markov1", "p_tai": 0.5, "explain": "Insufficient data for Markov(1)"}
    trans = {"T->T": 0, "T->X": 0, "X->T": 0, "X->X": 0}
    for i in range(1, len(history)):
        a = "T" if history[i-1]["result"].upper().startswith("T") else "X"
        b = "T" if history[i]["result"].upper().startswith("T") else "X"
        trans[f"{a}->{b}"] += 1
    last = "T" if history[-1]["result"].upper().startswith("T") else "X"
    denom = trans[f"{last}->T"] + trans[f"{last}->X"]
    p = trans[f"{last}->T"] / denom if denom > 0 else 0.5
    return {
        "name": "markov1",
        "p_tai": p,
        "explain": f"Markov(1) P(T|{last})={p:.3f}. Transitions: {trans}"
    }

def analyser_markov_order2(history: List[Dict]):
    if len(history) < 3:
        return {"name": "markov2", "p_tai": 0.5, "explain": "Insufficient data for Markov(2)"}
    counts = {}
    for i in range(2, len(history)):
        key = "".join(["T" if history[i-2]["result"].upper().startswith("T") else "X",
                       "T" if history[i-1]["result"].upper().startswith("T") else "X"])
        next_r = "T" if history[i]["result"].upper().startswith("T") else "X"
        counts.setdefault(key, {"T": 0, "X": 0})
        counts[key][next_r] += 1
    last_key = "".join(["T" if history[-2]["result"].upper().startswith("T") else "X",
                        "T" if history[-1]["result"].upper().startswith("T") else "X"])
    p = counts.get(last_key, {"T": 0, "X": 0})["T"] / sum(counts.get(last_key, {"T": 0, "X": 0}).values()) if last_key in counts else 0.5
    return {
        "name": "markov2",
        "p_tai": p,
        "explain": f"Markov(2) key={last_key}, counts={counts.get(last_key, '-')}, p={p:.3f}"
    }

def analyser_beta_window(history: List[Dict], window=30, alpha=1, beta=1):
    last = history[-window:] if len(history) >= 1 else history
    a = sum(1 for r in last if r["result"].upper().startswith("T"))
    b = len(last) - a
    p = (alpha + a) / (alpha + beta + a + b)
    return {
        "name": f"beta_{window}",
        "p_tai": p,
        "explain": f"Beta posterior (α={alpha},β={beta}) on last {len(last)}: a={a},b={b}, E[θ]={p:.3f}"
    }

def analyser_streak_detector(history: List[Dict]):
    if not history:
        return {"name": "streak", "p_tai": 0.5, "explain": "No history"}
    last_result = "T" if history[-1]["result"].upper().startswith("T") else "X"
    run = 1
    for i in range(len(history)-2, -1, -1):
        if history[i]["result"].upper().startswith(last_result):
            run += 1
        else:
            break
    p = 0.25 if last_result == "T" and run >= 5 else 0.75 if last_result == "X" and run >= 5 else 0.5 + (0.02 * (run-1) * (-1 if last_result == "T" else 1))
    p = max(0, min(1, p))
    return {
        "name": "streak",
        "p_tai": p,
        "explain": f"Run of {last_result} len={run}. {'Mean-reversion' if run >= 5 else 'Weak inertia'} -> p_tai={p:.3f}"
    }

def analyser_pattern_match(history: List[Dict], k=20):
    pattern_info = detect_pattern(history, k)
    return {
        "name": "pattern_match",
        "p_tai": pattern_info["p_tai"],
        "explain": f"Pattern last {k}: {pattern_info['current_pattern']}. Matched: {pattern_info['matched_patterns']}. Follow counts: {pattern_info['follow_counts']}. p={pattern_info['p_tai']:.3f}"
    }

def analyser_face_freq(history: List[Dict], window=50):
    last = history[-window:] if len(history) > 0 else history
    counts = Counter(d for r in last for d in r["dices"])
    total_faces = sum(counts.values()) or 1
    high = sum(counts.get(i, 0) for i in [4, 5, 6])
    low = sum(counts.get(i, 0) for i in [1, 2, 3])
    p = 0.5 + (high - low) / total_faces * 0.25 if total_faces else 0.5
    p = max(0, min(1, p))
    return {
        "name": "face_freq",
        "p_tai": p,
        "explain": f"Face counts: {counts}. High={high}, Low={low}, p_tai={p:.3f}"
    }

def analyser_entropy(history: List[Dict], window=50):
    last = history[-window:] if len(history) > 0 else history
    seq = [1 if r["result"].upper().startswith("T") else 0 for r in last]
    p = sum(seq) / len(seq) if seq else 0.5
    ent = -(p * math.log2(p + 1e-10) + (1-p) * math.log2(1-p + 1e-10)) if 0 < p < 1 else 0
    bias_strength = 0.5 - ent
    p_tai = 0.5 + (p - 0.5) * (1 + bias_strength)
    p_tai = max(0, min(1, p_tai))
    return {
        "name": "entropy",
        "p_tai": p_tai,
        "explain": f"Last {len(last)} p_TAI={p:.3f}, entropy={ent:.3f}, p_tai={p_tai:.3f}"
    }

def analyser_transition_rates(history: List[Dict], window=100):
    last = history[-window:] if len(history) > 0 else history
    if len(last) < 2:
        return {"name": "trans_rates", "p_tai": 0.5, "explain": "Insufficient data"}
    trans = {"T->T": 0, "T->X": 0, "X->T": 0, "X->X": 0}
    for i in range(1, len(last)):
        a = "T" if last[i-1]["result"].upper().startswith("T") else "X"
        b = "T" if last[i]["result"].upper().startswith("T") else "X"
        trans[f"{a}->{b}"] += 1
    last_state = "T" if last[-1]["result"].upper().startswith("T") else "X"
    denom = trans[f"{last_state}->T"] + trans[f"{last_state}->X"]
    p = trans[f"{last_state}->T"] / denom if denom > 0 else 0.5
    return {
        "name": "trans_rates",
        "p_tai": p,
        "explain": f"Transitions (last {len(last)}): {trans}. Last state {last_state}, p={p:.3f}"
    }

def analyser_weighted_history(history: List[Dict], half_life=30):
    if not history:
        return {"name": "ewm", "p_tai": 0.5, "explain": "No history"}
    wsum, wtot = 0.0, 0.0
    for i, r in enumerate(history[::-1]):
        weight = math.exp(-i / half_life)
        val = 1.0 if r["result"].upper().startswith("T") else 0.0
        wsum += weight * val
        wtot += weight
    p = wsum / wtot if wtot > 0 else 0.5
    return {
        "name": "ewm",
        "p_tai": p,
        "explain": f"EWMA half_life={half_life}, p_tai={p:.3f}"
    }

# ---------- Ensemble Combiner ----------
DEFAULT_ANALYSERS = [
    analyser_freq_window,
    analyser_markov_order1,
    analyser_markov_order2,
    analyser_beta_window,
    analyser_streak_detector,
    analyser_pattern_match,
    analyser_face_freq,
    analyser_entropy,
    analyser_transition_rates,
    analyser_weighted_history
]

# Tối ưu hóa trọng số dựa trên hiệu suất lịch sử (giả định đơn giản)
DEFAULT_WEIGHTS = [1.2, 1.0, 1.0, 1.1, 1.3, 1.5, 1.0, 1.1, 1.0, 1.0]

def run_analysers(history):
    results = []
    for f in DEFAULT_ANALYSERS:
        try:
            res = f(history)
            if "p_tai" in res:
                results.append(res)
        except Exception as e:
            logger.error(f"Error in analyser {f.__name__}: {e}")
            results.append({"name": f.__name__, "p_tai": 0.5, "explain": f"Error: {e}"})
    return results

def combine_ensemble(results, weights=None):
    if not results:
        return {"p_final": 0.5, "du_doan": "XIU", "do_tin_cay": 0.0, "giai_thich": "No results"}
    weights = weights or DEFAULT_WEIGHTS[:len(results)]
    wsum = sum(weights)
    weights = [w / wsum for w in weights] if wsum > 0 else [1/len(results)] * len(results)
    
    p_total = 0.0
    explains = []
    for i, res in enumerate(results):
        p_total += res["p_tai"] * weights[i]
        explains.append(f"[{res['name']}] p_tai={res['p_tai']:.3f}; {res['explain']}")
    
    p_final = p_total
    du_doan = "TAI" if p_final >= 0.5 else "XIU"
    conf = abs(p_final - 0.5) * 2 * 100
    conf = round(min(conf, 100), 2)
    giai_thich = "\n".join(explains)
    
    return {
        "p_final": p_final,
        "du_doan": du_doan,
        "do_tin_cay": conf,
        "giai_thich": giai_thich
    }

# ---------- Parse Source Payload ----------
def parse_source_payload(payload: dict):
    session = payload.get("Phien") or payload.get("session") or payload.get("id")
    dices = None
    if "dices" in payload and isinstance(payload["dices"], list):
        dices = payload["dices"]
    else:
        found = []
        for k in payload:
            lk = k.lower().replace(" ", "")
            if any(kpart in lk for kpart in ["xuc_xac_1", "xuc_xac_2", "xuc_xac_3", "d1", "d2", "d3"]):
                try:
                    found.append(int(payload[k]))
                except:
                    pass
        if len(found) == 3:
            dices = found
    total = payload.get("Tong") or payload.get("total")
    result = payload.get("Ket_qua") or payload.get("result")
    
    if dices and total is None:
        total = sum(dices)
    if result is None and total is not None:
        result = "TAI" if total >= 11 else "XIU"
    
    if not all([session, dices, total, result]):
        raise ValueError(f"Unable to parse payload. Got: {payload}")
    
    return int(session), dices, int(total), str(result)

# ---------- API Response Model ----------
class PredictResponse(BaseModel):
    session: int
    dice: str
    total: int
    result: str
    next_session: int
    du_doan: str
    do_tin_cay: float
    giai_thich: str
    pattern: str
    matched_patterns: List[str]
    ty_le: Dict[str, float]
    id: str
    features: Dict[str, Any]

# ---------- API Endpoints ----------
@app.on_event("startup")
def startup():
    init_db()
    logger.info("Database initialized")

@app.get("/predict", response_model=PredictResponse)
def predict_from_source():
    try:
        r = requests.get(SOURCE_API, timeout=10)
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        logger.error(f"Error fetching source API: {e}")
        raise HTTPException(status_code=502, detail=f"Error fetching source API: {e}")

    try:
        session, dices, total, result = parse_source_payload(payload)
    except Exception as e:
        logger.error(f"Parse error: {e}")
        raise HTTPException(status_code=500, detail=f"Parse error: {e}")

    ts = datetime.now(timezone.utc).astimezone().isoformat()
    try:
        save_round(session, dices, total, result, ts)
    except Exception as e:
        logger.warning(f"DB save error: {e}")

    history = get_last_n(HISTORY_LIMIT)
    pattern_info = detect_pattern(history, PATTERN_LENGTH)
    features = extract_features(history)
    results = run_analysers(history)
    ensemble = combine_ensemble(results)

    return {
        "session": session,
        "dice": f"{dices[0]}-{dices[1]}-{dices[2]}",
        "total": total,
        "result": result,
        "next_session": session + 1,
        "du_doan": ensemble["du_doan"],
        "do_tin_cay": ensemble["do_tin_cay"],
        "giai_thich": (
            f"Pattern Analysis: Current pattern: {pattern_info['current_pattern']}\n"
            f"Matched patterns: {pattern_info['matched_patterns']}\n"
            f"Follow counts: {pattern_info['follow_counts']}\n"
            f"Pattern-based p_tai: {pattern_info['p_tai']:.3f}\n\n"
            f"Ensemble Analysis:\n{ensemble['giai_thich']}"
        ),
        "pattern": pattern_info["current_pattern"],
        "matched_patterns": pattern_info["matched_patterns"],
        "ty_le": {
            "Tai": round(ensemble["p_final"] * 100, 2),
            "Xiu": round((1 - ensemble["p_final"]) * 100, 2)
        },
        "id": TELE_ID,
        "features": features
    }

@app.get("/history")
def api_history(limit: int = 200):
    return get_last_n(limit)

@app.get("/pattern")
def api_pattern(k: int = 20):
    history = get_last_n(k)
    pattern_info = detect_pattern(history, k)
    return {
        "pattern": pattern_info["current_pattern"],
        "matched_patterns": pattern_info["matched_patterns"],
        "follow_counts": pattern_info["follow_counts"],
        "p_tai": pattern_info["p_tai"]
    }

@app.post("/push")
def push(payload: dict):
    try:
        session, dices, total, result = parse_source_payload(payload)
        ts = datetime.now(timezone.utc).astimezone().isoformat()
        save_round(session, dices, total, result, ts)
        return {"ok": True, "session": session}
    except Exception as e:
        logger.error(f"Push error: {e}")
        raise HTTPException(status_code=400, detail=f"Parse error: {e}")

@app.get("/")
def root():
    return {"Hahahahahahahhahahahahhahahahahhahahahahahhahahahahahhahahahhahahhah Con Chó Ngu Nào Lạc Vào đây vậy ? 😂😂😂 thích bu đồ free mà ngon á về bú sửa mẹ cho ngon em nhá ở đây đéo có gì là miễn phí mà ngon cả 😂😂😂😂😂😂😂😂😂😂"}
