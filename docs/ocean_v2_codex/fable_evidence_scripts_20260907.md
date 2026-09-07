# Fable이 09-07 04:30~05:00 KST에 실제로 실행한 근거 계산 코드 (원문 그대로)

Codex 검증 피드백(`FABLE_FINAL_DAY_VERIFICATION_FEEDBACK_20260907.md`) 요청에 따라 원 코드를 공개한다. 실행 환경 `.venv-p1`, 저장소 루트에서 stdin 스크립트로 실행. **Codex의 재계산이 더 정확하며(아래 P2의 PAVA는 근사 구현), 계획서 수치는 Codex 값으로 대체했다.**

## P2 투영 검사 (입력: `artifacts/p2_c3_multiseed_completion_20260906_v1/evaluation.npz`, `observations.csv`)

알려진 결함: (1) PAVA를 "인접 위반쌍 평균 2회 반복"으로 근사해 3점 위반(예: [3,2,1])이 수렴하지 않음 → 정확한 등장회귀가 아니다. (2) 불완전 프로필(2층)에도 적용. (3) 방향 `sign(deep−T1)`이 0 또는 NaN이면 원값 유지. (4) 시각 조인은 UTC `time` 문자열을 `pd.to_datetime(utc=True)`로 파싱해 observations의 층별 pivot을 `reindex`. 블록 분모는 npz의 `fold` 라벨 행 수 그대로. 이 결함들이 Codex 값(B1 +0.000121, B2 −0.008293, pooled 1.236732)과의 차이(B1 +0.0028, B2 0.0000, pooled 1.23962)를 설명할 가능성이 높다. B3 0.4633, B8 0.2065는 일치.

```python
import numpy as np, pandas as pd
z=np.load("artifacts/p2_c3_multiseed_completion_20260906_v1/evaluation.npz",allow_pickle=True)
key=z["key"]; truth=z["truth"]; fold=z["fold"]; layer=z["layer"]; pred=z["natural_L120"]; tm=pd.to_datetime(z["time"],utc=True)
D=r"C:\Users\cedis\Downloads\p2\데이터셋_P2\P2_profile_restore"
o=pd.read_csv(f"{D}/observations.csv",usecols=["layer","time","temp"]); o["time"]=pd.to_datetime(o.time,utc=True)
w=o.pivot_table(index="time",columns="layer",values="temp")
T1=w[1].reindex(tm).to_numpy(); T5=w[5].reindex(tm).to_numpy(); T6=w[6].reindex(tm).to_numpy(); T7=w[7].reindex(tm).to_numpy(); T8=w[8].reindex(tm).to_numpy()
deep=np.where(np.isfinite(T5),T5,np.where(np.isfinite(T6),T6,np.where(np.isfinite(T7),T7,T8)))
lo=np.fmin(T1,deep); hi=np.fmax(T1,deep)
ok=np.isfinite(lo)&np.isfinite(hi)
clip=pred.copy(); clip[ok]=np.clip(pred[ok],lo[ok],hi[ok])          # 1) clip 먼저
df=pd.DataFrame({"t":tm,"layer":layer,"p":clip,"truth":truth,"fold":fold,"dir":np.sign(deep-T1)})
def pava3(g):                                                        # 2) 근사 PAVA (결함: 2회 인접 평균)
    g=g.sort_values("layer"); v=g.p.to_numpy().astype(float); d=g["dir"].iloc[0]
    if len(v)<2 or not np.isfinite(d) or d==0: return g.p.to_numpy()
    x=(v if d>0 else -v).copy()
    for _ in range(2):
        for i in range(len(x)-1):
            if x[i]>x[i+1]:
                m=(x[i]+x[i+1])/2; x[i]=x[i+1]=m
    return x if d>0 else -x
out=df.groupby("t",sort=False,group_keys=False).apply(lambda g: pd.Series(pava3(g),index=g.index))
df["pava"]=out.reindex(df.index).to_numpy()
def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))
for name,arr in [("L120 raw",pred),("clip only",clip),("clip+PAVA",df.pava.to_numpy())]:
    print(name, rmse(arr,truth), rmse(arr[fold=='B3'],truth[fold=='B3']), rmse(arr[fold=='B8'],truth[fold=='B8']))
```

출력(당시): raw pooled 1.25177 / B3 0.48351 / B8 0.21771; clip only 1.23970 / 0.46441 / 0.20763; clip+PAVA 1.23962 / 0.46332 / 0.20650; 블록별 B1 1.9855→1.9883, B2 1.3256→1.3256, B3 →0.4633, B4 →0.0387, B5 →0.7014, B6 →1.8759, B7 →0.9505, B8 →0.2065.

## P3 shrink 검사 (입력: `artifacts/p3_numeric_lead_forward_gpu_20260906_v2/candidate_oof.parquet`)

```python
import pandas as pd, numpy as np
c=pd.read_parquet("artifacts/p3_numeric_lead_forward_gpu_20260906_v2/candidate_oof.parquet")
def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))
t=c.target_hs.to_numpy(); p=c.persistence.to_numpy(); f=c.final_prediction.to_numpy(); L=c.lead_h.to_numpy()
long=np.isin(L,[12,18,24])
routed=f.copy(); routed[long]=(f[long]-0.2*p[long])/0.8               # 12/18/24h만 shrink 복원
for w in [0.0,0.1,0.2,0.3,0.4]:
    g=routed.copy(); g[long]=(1-w)*routed[long]+w*p[long]; print(w, rmse(g,t), rmse(g[long],t[long]))
for l in [3,6,9,12,18,24]:                                           # 변화량 보정 OLS: t-p ~ a*(routed-p)
    m=L==l; d=routed[m]-p[m]; r=t[m]-p[m]; a=float((d*r).sum()/(d*d).sum()); print(l, rmse(p[m],t[m]), rmse(routed[m],t[m]), a, rmse(p[m]+a*d,t[m]))
low=c.current_hs.to_numpy()<1.7; print(rmse(p[low],t[low]), rmse(f[low],t[low]), rmse(routed[low],t[low]))
rng=np.random.default_rng(20260907); ep=(c.station.astype(str)+"_"+c.episode_id.astype(str)).to_numpy(); ue=np.unique(ep)
idx={e:np.where(ep==e)[0] for e in ue}; deltas=[]
for _ in range(500):
    s=rng.choice(ue,len(ue),replace=True); ii=np.concatenate([idx[e] for e in s]); deltas.append(rmse(routed[ii],t[ii])-rmse(f[ii],t[ii]))
deltas=np.array(deltas); print(deltas.mean(), np.quantile(deltas,.05), np.quantile(deltas,.95), (deltas<0).mean())
```

출력(당시): w=0 0.67983 / w=0.2 0.68354; 장기 리드 0.77775 / 0.78423; a_L 3h 0.706, 6h 0.851, 9h 0.847, 12h 0.920, 18h 0.955, 24h 0.973 (변화량 보정식 `p + a·(routed−p)` 기준); hs0<1.7: persistence 0.62, shrink0.2 0.6189, 무shrink 0.6294; 부트스트랩 500회 Δ −0.00374, CI90 [−0.00775, +0.00085], P(개선) 0.912. Codex 10,000회: −0.003707, [−0.008301, +0.000982], 0.9076.

## P1 범위 검사 (입력: P1 train.csv / test.csv, 채점본 57844ef2)

정찰(09-06)과 첫 계획서의 805/123/103/20행은 **고정 [0, 35]℃** 규칙의 결과다(코드: `oor=(temp<0)|(temp>35)`). 계획서에 "학습 라벨 0 min/max로 유도"라고 쓴 것은 오기이며, 그 규칙은 다른 규칙이다. 09-07 추가 계산:

- train label 0 min/max = 5.0017 / 30.8965℃ (Codex와 일치). 이 범위 밖 test 행 653(Codex) — 내 계산 644(<min) + 9(>max).
- **test의 [0, 5.0017)℃ 행 521개**는 G-ORS 3월 56, S-ORS 2월 228·3월 82·4~6월 268 등으로, |Δ| 중앙값 0.044℃, 79%가 |Δ|<0.2℃ → 자연스러운 저수온 연속 관측으로 보인다(2026년 겨울이 학습 겨울보다 차가움: test 최저 G-ORS 3.09, I-ORS 1.66℃). 학습 min 기준 규칙은 이 행들을 대량 양성 처리한다.
- 고정 [0,35] 규칙의 test 이탈 123행은 모두 S-ORS이며 최저 −20.66℃. 학습 라벨 0에는 5℃ 미만 행이 없어 **0~5℃ 구간에서 이 규칙의 FP 여부를 학습 자료로 검증할 수 없다.** 따라서 "FP 0 확실"은 철회한다.
