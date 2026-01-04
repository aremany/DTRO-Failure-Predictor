import os
import sqlite3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import uvicorn
import tempfile
from datetime import datetime

# --- 설정 --- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 상위 폴더의 DB를 우선적으로 찾음
PARENT_DB_PATH = os.path.join(os.path.dirname(BASE_DIR), "incident_reports.db")
SQLITE_DB_PATH = PARENT_DB_PATH

# 만약 상위 폴더에 없으면 현재 폴더 확인 (독립 실행 시)
if not os.path.exists(PARENT_DB_PATH):
    LOCAL_DB_PATH = os.path.join(BASE_DIR, "incident_reports.db")
    if os.path.exists(LOCAL_DB_PATH):
        SQLITE_DB_PATH = LOCAL_DB_PATH
        print(f"현재 폴더의 데이터베이스를 사용합니다: {SQLITE_DB_PATH}")

def robust_text_factory(x):
    try:
        return x.decode('utf-8')
    except UnicodeDecodeError:
        try:
            return x.decode('cp949')
        except UnicodeDecodeError:
            return x.decode('utf-8', 'ignore')

app = FastAPI(title="Independent Predictor", description="독립 실행형 장애 예측 프로그램")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic 모델 정의 --- #
class PredictRequest(BaseModel):
    fault_type: str
    target_year: Optional[int] = None

class PredictionResult(BaseModel):
    month: str
    predicted_count: int

class PredictResponse(BaseModel):
    fault_type: str
    predictions: List[PredictionResult]
    used_model: str
    debug_info: str
    total_predicted: int | None = None
    confidence: Optional[Dict[str, Optional[float]]] = None

# --- GRU 모델 정의 --- #
class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1, num_layers=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

# --- 예측 로직 (기존 코드 재사용) --- #
def calc_confidence(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Optional[float]]:
    try:
        y_pred = np.array(y_pred, dtype=float).flatten()
        std = float(np.std(y_pred)) if y_pred.size > 0 else None
        mean = float(np.mean(y_pred)) if y_pred.size > 0 else None
        return {"std": std, "mean": mean}
    except Exception:
        return {"std": None, "mean": None}

# --- 예측 함수 정의 (dtro.py에서 가져옴) --- #
def _improve_lstm_realism(predictions, data):
    """GRU 예측값을 현실적으로 조정 - 과도한 예측 방지"""
    try:
        if predictions is None or len(predictions) == 0:
            return predictions
        
        # 과거 데이터 분석
        past_values = data
        past_nonzero = past_values[past_values > 0]
        
        if len(past_nonzero) == 0:
            return np.zeros(len(predictions), dtype=int)
        
        # 과거 통계
        past_mean = np.mean(past_values)  # 0 포함한 전체 평균
        past_annual_total = np.sum(past_values) * (12 / len(past_values)) if len(past_values) > 0 else 0
        occurrence_rate = len(past_nonzero) / len(past_values)  # 발생 확률
        
        # 초기 예측값
        adjusted = np.array(predictions, dtype=float)
        initial_sum = np.sum(adjusted)
        
        # 1. 매우 보수적인 연간 총합 제한
        max_annual = past_annual_total * 0.8
        
        if initial_sum > max_annual and initial_sum > 0:
            scale_factor = max_annual / initial_sum
            adjusted = adjusted * scale_factor
        
        # 2. 희소성 강화
        if occurrence_rate < 0.5:
            target_months = max(1, int(12 * occurrence_rate * 0.8))
            if np.count_nonzero(adjusted) > target_months:
                threshold_idx = np.argsort(adjusted)[-target_months]
                threshold = adjusted[threshold_idx] if target_months > 0 else np.max(adjusted)
                mask = adjusted >= threshold
                top_indices = np.argsort(adjusted)[-target_months:]
                new_adjusted = np.zeros_like(adjusted)
                new_adjusted[top_indices] = adjusted[top_indices]
                adjusted = new_adjusted
        
        # 3. 과거 최대값으로 개별 월 제한
        past_max = np.max(past_values)
        adjusted = np.clip(adjusted, 0, past_max)
        
        # 4. 최종 연간 총합 재검증
        final_sum = np.sum(adjusted)
        if final_sum > past_annual_total * 0.6:
            excess_ratio = (final_sum - past_annual_total * 0.6) / final_sum
            if excess_ratio > 0:
                adjusted = adjusted * (1 - excess_ratio)
        
        # 5. 정수화 및 최종 검증
        result = np.round(adjusted).astype(int)
        result = np.clip(result, 0, past_max)
        
        # 6. 마지막 안전장치
        final_total = np.sum(result)
        max_allowed_total = max(1, int(past_annual_total * 0.6))
        
        if final_total > max_allowed_total:
            nonzero_indices = np.where(result > 0)[0]
            if len(nonzero_indices) > 0:
                reduce_count = int(final_total - max_allowed_total)
                for _ in range(min(reduce_count, len(nonzero_indices))):
                    if len(nonzero_indices) > 0:
                        idx = np.random.choice(nonzero_indices)
                        if result[idx] > 0:
                            result[idx] -= 1
                            if result[idx] == 0:
                                nonzero_indices = nonzero_indices[nonzero_indices != idx]
        return result
    except Exception:
        return np.round(predictions).astype(int) if predictions is not None else predictions

def predict_gru(data, future_units):
    min_required = 6
    if len(data) < min_required:
        return None, f"[GRU 스킵] 데이터 부족 ({len(data)}<{min_required})"
    
    zero_ratio = (data == 0).sum() / len(data)
    if zero_ratio >= 0.5:
        return None, f"[GRU 스킵] 결측(0) 비율이 높음: {zero_ratio:.1%}"

    try:
        seq_len = min(9, len(data) - 1) if len(data) > 1 else 1
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len])
        X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(np.array(y), dtype=torch.float32)
        
        model = GRUModel()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        for epoch in range(50):
            model.train()
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output.squeeze(), y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            last_seq = torch.tensor(data[-seq_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            forecast = []
            for _ in range(len(future_units)):
                pred = model(last_seq).item()
                forecast.append(pred)
                last_seq = torch.cat([last_seq[:, 1:], torch.tensor([[pred]], dtype=torch.float32).unsqueeze(-1)], dim=1)
        
        raw_predictions = np.array(forecast)
        realistic_predictions = _improve_lstm_realism(raw_predictions, data)
        return realistic_predictions, "GRU"
    except Exception as e:
        return None, f"[GRU 오류] {e}"

def predict_knn(agg_df, group_key, future_units):
    try:
        y = agg_df.set_index(group_key)['count']
        min_required = 3
        if len(y) < min_required: return None, "[KNN 스킵] 데이터 부족"
        
        zero_ratio = (y.values == 0).sum() / len(y) if len(y) > 0 else 0
        if zero_ratio >= 0.8: return None, f"[KNN 스킵] 결측 비율 높음: {zero_ratio:.1%}"

        n_neighbors = 1 if len(y) < 6 else 2
        model = KNeighborsRegressor(n_neighbors=n_neighbors)
        model.fit(np.arange(len(y)).reshape(-1, 1), y.values)
        x_future = np.arange(len(y), len(y) + len(future_units)).reshape(-1, 1)
        y_pred = model.predict(x_future)
        y_pred = np.clip(np.round(y_pred), 0, y.max()).astype(int)
        
        if np.all((y_pred == 0) | (y_pred == 1)):
             temp, _ = predict_mean(agg_df)
             if temp is not None: return np.array(temp, dtype=int), "KNN"
        
        if np.any(y_pred > 0): return y_pred, "KNN"
        return None, "[KNN] 예측값 0 또는 무효"
    except Exception as e:
        return None, f"[KNN 오류] {e}"

def predict_linear(agg_df, group_key, future_units):
    try:
        y = agg_df.set_index(group_key)['count']
        min_required = 3
        if len(y) < min_required: return None, "[선형회귀 스킵] 데이터 부족"
        
        zero_ratio = (y.values == 0).sum() / len(y) if len(y) > 0 else 0
        if zero_ratio >= 0.90: return None, f"[선형회귀 스킵] 결측 비율 높음: {zero_ratio:.1%}"

        x = np.arange(len(y)).reshape(-1, 1)
        model = LinearRegression(fit_intercept=True, positive=True)
        model.fit(x, y.values)
        x_future = np.arange(len(y), len(y) + len(future_units)).reshape(-1, 1)
        y_pred = model.predict(x_future)
        max_past = y.max()
        mean_past = y.mean()
        y_pred = np.clip(np.round(y_pred), 0, max_past * 1.5).astype(int)
        
        if np.all(y_pred == 0) or np.std(y_pred) < 0.01:
            return np.array([int(round(mean_past))] * len(future_units)), "선형회귀(평균대체)"
        
        if np.any(y_pred > mean_past * 3):
            y_pred = np.where(y_pred > mean_past * 3, int(round(mean_past)), y_pred)
            return y_pred, "선형회귀(부분조정)"
            
        if np.all(y_pred <= 0):
            return np.array([int(round(mean_past))] * len(future_units)), "선형회귀(평균대체-음수)"

        return y_pred, "선형회귀"
    except Exception as e:
        return None, f"[선형회귀 오류] {e}"

def predict_mean(agg_df):
    try:
        if agg_df.empty or 'count' not in agg_df.columns: return None, "[평균 스킵] 데이터 없음"
        
        df_sorted = agg_df.sort_values('month')
        win_n = 36 # 기본값
        if len(df_sorted) > win_n: win_df = df_sorted.tail(win_n)
        else: win_df = df_sorted

        vals = np.array(win_df['count'].values, dtype=float)
        if vals.size == 0: return None, "[평균 스킵] 데이터 없음"
        if np.count_nonzero(vals) < 1: return None, "[평균 스킵] 0이 아닌 값 부족"
        if np.count_nonzero(vals) == 1: return None, "[평균 스킵] 1건으로는 예측 불가"

        max_past = int(np.nanmax(vals)) if vals.size > 0 else 0
        mean_past = float(np.nanmean(vals)) if vals.size > 0 else 0.0
        
        # Case A: 희소 데이터
        if max_past <= 1:
            p = float(np.count_nonzero(vals) / len(vals))
            rng = np.random.default_rng()
            # 간단히 Bernoulli 사용
            p_cap = 0.75
            p = min(p, p_cap)
            bernoulli = rng.binomial(n=1, p=max(0.0, min(1.0, p)), size=12).astype(int)
            return bernoulli, "평균(확률)"

        # Case B: 계절 평균
        months = pd.to_datetime(win_df['month']).dt.month
        monthly_mean = win_df.assign(_m=months).groupby('_m')['count'].mean().reindex(range(1,13), fill_value=mean_past)
        floats = monthly_mean.values.astype(float)
        
        floors = np.floor(floats)
        rema = floats - floors
        target = int(np.round(np.sum(floats)))
        pred = floors.astype(int)
        remaining = int(target - np.sum(pred))
        
        if remaining > 0:
            idx_order = np.argsort(-rema)
            for i in idx_order:
                if remaining <= 0: break
                pred[i] += 1
                remaining -= 1
        elif remaining < 0:
            idx_order = np.argsort(rema)
            for i in idx_order:
                if remaining >= 0: break
                if pred[i] > 0:
                    pred[i] -= 1
                    remaining += 1
        
        pred = np.clip(pred, 0, max_past).astype(int)
        return pred, "평균(계절)"
    except Exception as e:
        return None, f"[평균 오류] {e}"

# --- API 엔드포인트 --- #

async def _predict_internal(fault_type: str, target_year: Optional[int] = None):
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.text_factory = robust_text_factory
    try:
        query = "SELECT 장애일시 FROM incident_data WHERE 장애명 = ?"
        df = pd.read_sql_query(query, conn, params=[fault_type])
        
        if df.empty:
            # 데이터가 없어도 에러가 아닌 빈 결과 반환
            return PredictResponse(
                fault_type=fault_type, 
                predictions=[], 
                used_model="데이터없음", 
                debug_info="No data found in DB", 
                total_predicted=0,
                confidence={"std": 0.0, "mean": 0.0}
            )

        df['장애일시'] = pd.to_datetime(df['장애일시'], errors='coerce')
        df.dropna(subset=['장애일시'], inplace=True)
        df['month'] = df['장애일시'].dt.to_period('M').dt.to_timestamp('M')
        agg_df = df.groupby('month').size().reset_index(name='count')
        agg_df = agg_df.set_index('month').asfreq('ME', fill_value=0).reset_index()

        t_year = target_year if target_year else pd.to_datetime('now').year + 1
        future_units = [f"{t_year}-{m:02}" for m in range(1, 13)]
        data = agg_df['count'].values

        predictions = None
        used_model = "미분류"
        debug_info = ""

        # 1. GRU
        if predictions is None:
            preds, info = predict_gru(data, future_units)
            if preds is not None and np.any(preds > 0):
                predictions, used_model = preds, "GRU"
                debug_info += f"[1차 GRU] {info}\n"

        # 2. KNN
        if predictions is None:
            preds, info = predict_knn(agg_df, 'month', future_units)
            if preds is not None and np.any(preds > 0):
                predictions, used_model = preds, "KNN"
                debug_info += f"[2차 KNN] {info}\n"

        # 3. Linear
        if predictions is None:
            preds, info = predict_linear(agg_df, 'month', future_units)
            if preds is not None and np.any(preds > 0):
                predictions, used_model = preds, "선형회귀"
                debug_info += f"[3차 선형회귀] {info}\n"

        # 4. Mean
        if predictions is None:
            preds, info = predict_mean(agg_df)
            if preds is not None:
                predictions, used_model = preds, "평균"
                debug_info += f"[4차 평균] {info}\n"
            else:
                predictions = [0] * 12
                used_model = "예측불가"

        if len(predictions) != 12:
            predictions = list(predictions) + [0] * (12 - len(predictions))
            predictions = predictions[:12]

        res_preds = [PredictionResult(month=m, predicted_count=int(p)) for m, p in zip(future_units, predictions)]
        
        return PredictResponse(
            fault_type=fault_type,
            predictions=res_preds,
            used_model=used_model,
            debug_info=debug_info,
            total_predicted=sum(p.predicted_count for p in res_preds),
            confidence=calc_confidence(data, predictions)
        )
    except Exception as e:
        print(f"Error in _predict_internal for {fault_type}: {e}")
        import traceback
        traceback.print_exc()
        return None # Caller handles None
    finally:
        conn.close()

@app.post("/predict", response_model=PredictResponse)
async def predict_fault(request: PredictRequest):
    try:
        print(f"DEBUG: Requesting prediction for {request.fault_type}")
        result = await _predict_internal(request.fault_type, request.target_year)
        if result is None:
             return JSONResponse(content={
                 "fault_type": request.fault_type,
                 "predictions": [],
                 "total_predicted": 0,
                 "used_model": "Failure",
                 "debug_info": "Prediction returned None (insufficient data or model error)"
             })
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_all")
async def predict_all(request: PredictRequest):
    try:
        print("DEBUG: Starting predict_all")
        conn = sqlite3.connect(SQLITE_DB_PATH)
        conn.text_factory = robust_text_factory
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT 장애명 FROM incident_data WHERE 장애명 IS NOT NULL AND 장애명 != ''")
        fault_types = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        print(f"DEBUG: Found {len(fault_types)} fault types")

        results = []
        for ft in fault_types:
            try:
                res = await _predict_internal(ft, request.target_year)
                if res:
                    results.append(res)
            except Exception as e:
                print(f"WARN: Prediction failed for {ft}: {e}")
                continue
        
        results.sort(key=lambda x: x.total_predicted, reverse=True)
        print(f"DEBUG: predict_all finished with {len(results)} results")
        return results
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/download_xlsx")
async def download_xlsx(request: List[PredictResponse]):
    try:
        rows = []
        for res in request:
            row = {
                "장애명": res.fault_type,
                "총 예측 건수": res.total_predicted,
                "사용 모델": res.used_model,
                "신뢰도(std)": res.confidence['std'] if res.confidence else None
            }
            for pred in res.predictions:
                row[pred.month] = pred.predicted_count
            rows.append(row)
        
        df = pd.DataFrame(rows)
        cols = ["장애명", "총 예측 건수", "사용 모델", "신뢰도(std)"] + [p.month for p in request[0].predictions]
        df = df[cols]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            tmp_path = tmp.name
            df.to_excel(tmp_path, index=False)
        
        filename = f"prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        return FileResponse(tmp_path, filename=filename, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', background=BackgroundTasks().add_task(os.remove, tmp_path))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/fault_types")
async def get_fault_types():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.text_factory = robust_text_factory
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT 장애명 FROM incident_data WHERE 장애명 IS NOT NULL AND 장애명 != ''")
        types = [row[0] for row in cursor.fetchall()]
        return {"fault_types": sorted(types)}
    finally:
        conn.close()

# --- UI 서빙 --- #
# 정적 파일 서빙 (로고 이미지 등)
app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")

@app.get("/io.png")
async def get_logo():
    logo_path = os.path.join(BASE_DIR, "io.png")
    if os.path.exists(logo_path):
        return FileResponse(logo_path)
    raise HTTPException(status_code=404, detail="Logo not found")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open(os.path.join(BASE_DIR, "index.html"), "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    print(f"독립 예측기 시작... (DB 경로: {SQLITE_DB_PATH})")
    uvicorn.run(app, host="0.0.0.0", port=8001)
