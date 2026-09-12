# Generates the mockup page; each slide's markup is written once and emitted for A and B.
HEAD = r'''<title>Ocean AI 본선 슬라이드 시안</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{
  --bg:#F2F4F6; --ink:#16202B; --muted:#5B6B7B; --rule:#D3DAE1; --accent:#0A7F79; --panel:#FFFFFF;
  --sans:"IBM Plex Sans KR","Noto Sans KR","Apple SD Gothic Neo","Malgun Gothic",system-ui,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){ --bg:#0D1218; --ink:#E6EBF0; --muted:#98A6B4; --rule:#26313D; --accent:#2BA693; --panel:#141B23; }
}
:root[data-theme="dark"]{ --bg:#0D1218; --ink:#E6EBF0; --muted:#98A6B4; --rule:#26313D; --accent:#2BA693; --panel:#141B23; }

body{ background:var(--bg); color:var(--ink); font-family:var(--sans); margin:0; padding-inline:20px; padding-block:32px 56px; line-height:1.55; }
.wrap{ max-width:1240px; margin:0 auto; }
header{ max-width:72ch; margin-bottom:36px; }
.kicker{ font-size:12px; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); font-weight:500; }
h1{ font-size:clamp(24px,3.2vw,34px); line-height:1.2; margin:6px 0 12px; font-weight:700; text-wrap:balance; }
header p{ margin:0 0 10px; color:var(--muted); font-size:15px; }
header .rules{ display:flex; flex-wrap:wrap; gap:8px 18px; font-size:13px; margin-top:14px; color:var(--ink); }
header .rules span::before{ content:""; display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--accent); margin-right:7px; vertical-align:1px; }
section{ margin-top:44px; padding-top:22px; border-top:1px solid var(--rule); }
.sec-head{ display:flex; flex-wrap:wrap; align-items:baseline; gap:6px 16px; margin-bottom:14px; }
.sec-head h2{ font-size:20px; margin:0; font-weight:600; }
.sec-head .sub{ color:var(--muted); font-size:14px; }
.pair{ display:grid; grid-template-columns:1fr 1fr; gap:22px; }
@media (max-width:900px){ .pair{ grid-template-columns:1fr; } }
figure{ margin:0; min-width:0; }
figcaption{ display:flex; justify-content:space-between; gap:12px; font-size:13px; color:var(--muted); margin-top:8px; }
figcaption b{ color:var(--ink); font-weight:600; }
.notes{ margin-top:40px; padding-top:22px; border-top:1px solid var(--rule); max-width:78ch; }
.notes h2{ font-size:18px; margin:0 0 10px; }
.notes ul{ margin:0; padding-left:20px; font-size:14px; }
.notes li{ margin:5px 0; }
.notes .src{ color:var(--muted); font-size:13px; margin-top:12px; overflow-wrap:anywhere; }

/* ---------- slide boards ---------- */
.slide{ container-type:inline-size; aspect-ratio:16/9; width:100%; box-sizing:border-box; border:1px solid var(--rule); overflow:hidden; font-family:var(--sans); display:grid; grid-template-rows:auto auto 1fr auto; row-gap:1.7cqw; padding:3cqw 4cqw 2.6cqw; background:var(--sg); color:var(--si); }
.slide.A{ --sg:#F7F9FB; --si:#0F2540; --sm:#4A5D74; --sr:#D5DEE6; --sa:#0A9A92; --sat:#087C76; --sw:#C4701F; --sn:#A7B4C2; --sbox:#FFFFFF; }
.slide.B{ --sg:#10161D; --si:#E8EDF2; --sm:#9AA8B6; --sr:#2A3542; --sa:#2BA693; --sat:#4FC0AE; --sw:#C48447; --sn:#5E6C7C; --sbox:#161E27; }
.slide .eyebrow{ font-size:1.6cqw; letter-spacing:.06em; color:var(--sm); font-weight:500; display:flex; gap:1.6cqw; flex-wrap:wrap; }
.slide .eyebrow .sha{ font-family:var(--mono); letter-spacing:0; }
.slide h3{ margin:0; font-size:3.1cqw; line-height:1.22; font-weight:700; text-wrap:balance; letter-spacing:-.01em; }
.slide h3 em{ font-style:normal; color:var(--sat); }
.slide .body{ min-height:0; display:grid; gap:1.8cqw; align-content:start; }
.slide .foot{ font-size:1.45cqw; color:var(--sm); border-top:1px solid var(--sr); padding-top:1.1cqw; display:flex; gap:2.2cqw; flex-wrap:wrap; line-height:1.35; }
.slide .foot span::before{ content:"·"; margin-right:.5cqw; }
.slide svg{ width:100%; height:auto; display:block; overflow:visible; }
.slide svg text{ font-family:var(--sans); fill:var(--si); }
.slide svg .m{ fill:var(--sm); }
.slide svg .mono{ font-family:var(--mono); }
.slide svg .box{ fill:var(--sbox); stroke:var(--sr); }
.slide svg .box.hi{ stroke:var(--sa); stroke-width:1.5; }
.slide svg .ln{ stroke:var(--sm); stroke-width:1.2; fill:none; }
.slide svg .ax{ stroke:var(--sr); }
.slide svg .zero{ stroke:var(--sm); }
.slide svg .bc{ fill:var(--sa); } .slide svg .bb{ fill:var(--sn); } .slide svg .bw{ fill:var(--sw); }
.slide svg marker path{ fill:var(--sm); }
.slide .facts{ display:grid; grid-template-columns:repeat(3,1fr); gap:2.4cqw; align-items:start; }
.slide .fact .k{ font-size:1.4cqw; color:var(--sm); letter-spacing:.06em; }
.slide .fact .v{ font-size:3.2cqw; font-weight:600; line-height:1.15; font-variant-numeric:tabular-nums; letter-spacing:-.01em; }
.slide .fact .v.s{ font-size:2.3cqw; }
.slide .fact .v small{ font-size:1.6cqw; font-weight:500; color:var(--sm); margin-left:.5cqw; }
.slide .fact .d{ font-size:1.45cqw; color:var(--sm); margin-top:.4cqw; line-height:1.35; }

/* slide 7 */
.s7 .pipe{ display:grid; grid-template-columns:1fr auto 1fr auto 1fr auto 1fr; align-items:stretch; gap:.9cqw; font-size:1.55cqw; line-height:1.28; }
.s7 .pipe .step{ padding:.9cqw 1.2cqw; border:1px solid var(--sr); background:var(--sbox); min-width:0; }
.s7 .pipe .step.new{ border-color:var(--sa); box-shadow:inset .45cqw 0 0 var(--sa); }
.s7 .pipe .step b{ display:block; font-weight:600; font-size:1.65cqw; margin-bottom:.2cqw; }
.s7 .pipe .step span{ color:var(--sm); font-size:1.35cqw; display:block; }
.s7 .pipe .arrow{ align-self:center; color:var(--sm); font-size:1.8cqw; line-height:1; }
.s7 .charts{ display:grid; grid-template-columns:1fr 1.15fr; gap:3cqw; align-items:start; }
.s7 .ct{ font-size:1.6cqw; font-weight:600; margin-bottom:.4cqw; }
.s7 .charts svg{ max-height:17.5cqw; }
.s7 .ct small{ font-weight:400; color:var(--sm); margin-left:.5cqw; }
.s7 .delta{ font-size:1.6cqw; font-weight:600; margin-top:.4cqw; }
.s7 .delta small{ font-weight:400; color:var(--sm); margin-left:.5cqw; }

/* slide 11 */
.s11 .steps{ display:grid; grid-template-columns:repeat(5,1fr); gap:1.6cqw; position:relative; }
.s11 .steps::before{ content:""; position:absolute; left:0; right:0; top:1.05cqw; height:1px; background:var(--sr); }
.s11 .st{ position:relative; padding-top:2.5cqw; font-size:1.42cqw; line-height:1.36; min-width:0; }
.s11 .st::before{ content:attr(data-n); position:absolute; top:0; left:0; width:2.2cqw; height:2.2cqw; border-radius:50%; background:var(--sg); border:1px solid var(--sa); color:var(--sat); font-size:1.25cqw; font-weight:600; display:grid; place-items:center; font-variant-numeric:tabular-nums; }
.s11 .st.hit::before{ background:var(--sa); color:var(--sg); }
.s11 .st .who{ font-size:1.25cqw; letter-spacing:.08em; color:var(--sm); text-transform:uppercase; display:block; margin-bottom:.3cqw; }
.s11 .st b{ display:block; font-weight:600; margin-bottom:.35cqw; font-size:1.65cqw; }
.s11 .st code{ font-family:var(--mono); font-size:1.3cqw; background:var(--sbox); border:1px solid var(--sr); padding:0 .4cqw; }
.s11 .st .chk{ display:grid; gap:.2cqw; margin-top:.3cqw; font-variant-numeric:tabular-nums; }
.s11 .st .chk span::before{ content:"✓"; color:var(--sat); margin-right:.45cqw; font-weight:700; }
.s11 .limits{ display:grid; grid-template-columns:1fr 1fr; gap:2.4cqw; font-size:1.42cqw; line-height:1.36; padding:1.1cqw 1.4cqw; background:var(--sbox); border:1px solid var(--sr); }
.s11 .limits b{ display:block; font-weight:600; margin-bottom:.2cqw; }
.s11 .limits .w b{ color:var(--sw); }

@media (prefers-reduced-motion: reduce){ *{ animation:none!important; transition:none!important; } }
</style>
'''

S4 = r'''
    <div class="eyebrow"><span>P1 · 수온 이상 탐지</span><span>최종 답안 <span class="sha">57844ef2</span></span><span>test 169,011행</span></div>
    <h3>세 계열의 탐지기를 따로 학습하고, <em>고정 규칙</em>으로 판단을 합쳤다</h3>
    <div class="body">
      <svg viewBox="0 0 640 132" preserveAspectRatio="xMinYMin meet" aria-labelledby="{id}t">
        <title id="{id}t">P1 구성: XGBoost O와 LightGBM B를 셀 라우터로 결합하고 MS-TCN 합집합, GI spike 규칙을 거쳐 최종 답안</title>
        <defs><marker id="{id}ar" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L8 4 L0 8 z"/></marker></defs>
        <g font-size="12">
          <rect class="box" x="0" y="2" width="100" height="42"/>
          <text x="9" y="20" font-weight="600">O · XGBoost</text><text x="9" y="36" class="m" font-size="11">트리 1개</text>
          <rect class="box" x="0" y="56" width="100" height="42"/>
          <text x="9" y="74" font-weight="600">B · LightGBM</text><text x="9" y="90" class="m" font-size="11">3-seed 트리</text>
          <line class="ln" x1="100" y1="23" x2="128" y2="44" marker-end="url(#{id}ar)"/>
          <line class="ln" x1="100" y1="77" x2="128" y2="56" marker-end="url(#{id}ar)"/>
          <rect class="box hi" x="132" y="29" width="130" height="42"/>
          <text x="141" y="47" font-weight="600">셀 라우터</text><text x="141" y="63" class="m" font-size="11">6셀만 O 추가·B 제거</text>
          <line class="ln" x1="262" y1="50" x2="290" y2="50" marker-end="url(#{id}ar)"/>
          <rect class="box hi" x="294" y="29" width="130" height="42"/>
          <text x="303" y="47" font-weight="600">∪ MS-TCN 3-seed</text><text x="303" y="63" class="m" font-size="11">구간 탐지 합집합</text>
          <line class="ln" x1="424" y1="50" x2="452" y2="50" marker-end="url(#{id}ar)"/>
          <rect class="box hi" x="456" y="29" width="130" height="42"/>
          <text x="465" y="47" font-weight="600">+ GI spike 규칙</text><text x="465" y="63" class="m" font-size="11">일반 규칙, 2행 추가</text>
          <line class="ln" x1="586" y1="50" x2="612" y2="50" marker-end="url(#{id}ar)"/>
          <text x="616" y="46" font-weight="600">최종</text><text x="616" y="62" class="m" font-size="11">답안</text>
        </g>
        <text x="0" y="113" font-size="10" class="m">각 단계의 양성 판정 행 수 (성능 점수가 아님)</text>
        <g font-size="10.5" class="mono">
          <text x="0" y="128" class="m">O 6,504 · B 5,856</text>
          <text x="197" y="128" text-anchor="middle" class="m">6,061</text>
          <text x="359" y="128" text-anchor="middle" class="m">6,394</text>
          <text x="521" y="128" text-anchor="middle" class="m">6,396</text>
          <text x="640" y="128" text-anchor="end" class="m">/ 169,011행</text>
        </g>
      </svg>
      <div class="facts">
        <div class="fact"><div class="k">PUBLIC F1</div><div class="v">0.833548</div><div class="d">선택 답안 Public 점수 28.909341</div></div>
        <div class="fact"><div class="k">로컬 재생성</div><div class="v">7 fit<small>6,323초</small></div><div class="d">빈 모델 폴더 → 학습 → 답안 SHA 일치</div></div>
        <div class="fact"><div class="k">학습 OOF F1 · 회고 재계산</div><div class="v s">B 0.8647 → 라우터 0.8669</div><div class="d">독립 검증 아님 · MS-TCN 기여는 이 값에 없음</div></div>
      </div>
    </div>
    <div class="foot"><span>구성 요소별 독립 성능 기여량은 미확인</span><span>6셀 정책은 고정값, 원 선택 프로그램은 미복구</span><span>재생성 성공은 점수 향상이나 적격 판정이 아님</span></div>
'''

S7 = r'''
    <div class="eyebrow"><span>P2 · 중간층 수온 복원</span><span>최종 답안 <span class="sha">794268f1</span></span><span>동일 L120 3-seed 모델 · 새 학습 0</span></div>
    <h3>같은 모델에 시간 평균 한 단계만 더해 Public RMSE <em>0.4059 → 0.3953℃</em></h3>
    <div class="body">
      <div class="pipe">
        <div class="step"><b>L120 3-seed 예측</b><span>배포 데이터로 처음부터 학습, 3개 평균</span></div>
        <div class="arrow">→</div>
        <div class="step new"><b>±30분 중심 시간 평균</b><span>10분 간격 7칸, 층별 · 이번에 추가</span></div>
        <div class="arrow">→</div>
        <div class="step"><b>양끝층 범위 clip</b><span>같은 시각 공개 상·하층 사이로</span></div>
        <div class="arrow">→</div>
        <div class="step"><b>단조 투영(PAVA)</b><span>양끝층이 정하는 깊이 순서 유지</span></div>
      </div>
      <div class="charts">
        <div>
          <div class="ct">Public RMSE (℃) · Δ −0.010666℃<small>점수 28.240037 → 28.373869</small></div>
          <svg viewBox="0 0 300 118" preserveAspectRatio="xMinYMin meet" aria-labelledby="{id}t">
            <title id="{id}t">Public RMSE: 투영만 0.405920℃, 시간 평균 후 투영 0.395254℃</title>
            <g font-size="10" class="m">
              <line class="ax" x1="90" y1="8" x2="90" y2="92"/>
              <line class="ax" x1="90" y1="92" x2="290" y2="92"/>
              <text x="90" y="106" text-anchor="middle">0</text><text x="134.4" y="106" text-anchor="middle">0.1</text><text x="178.9" y="106" text-anchor="middle">0.2</text><text x="223.3" y="106" text-anchor="middle">0.3</text><text x="267.8" y="106" text-anchor="middle">0.4</text>
              <text x="290" y="117" text-anchor="end">℃</text>
            </g>
            <g font-size="11">
              <text x="84" y="32" text-anchor="end">투영만</text>
              <rect class="bb" x="90" y="18" width="180.4" height="20"><title>0.405920℃ · 점수 28.240037</title></rect>
              <text x="275" y="32" class="mono" font-size="10.5">0.405920</text>
              <text x="84" y="70" text-anchor="end">시간 평균 후 투영</text>
              <rect class="bc" x="90" y="56" width="175.7" height="20"><title>0.395254℃ · 점수 28.373869</title></rect>
              <text x="270" y="70" class="mono" font-size="10.5" font-weight="600">0.395254</text>
            </g>
          </svg>
        </div>
        <div>
          <div class="ct">내부 8블록 ΔRMSE (℃)<small>OOF 회고 · 7개 개선, B7 악화</small></div>
          <svg viewBox="0 0 340 152" preserveAspectRatio="xMinYMin meet" aria-labelledby="{id}b">
            <title id="{id}b">내부 8블록 RMSE 변화: B1 −0.084, B2 −0.045, B3 −0.022, B4 −0.002, B5 −0.006, B6 −0.021, B7 +0.015, B8 −0.014</title>
            <g font-size="10" class="m">
              <line class="zero" x1="250" y1="2" x2="250" y2="138"/>
              <text x="250" y="150" text-anchor="middle">0</text>
              <text x="90" y="150" text-anchor="middle">−0.08</text>
              <text x="170" y="150" text-anchor="middle">−0.04</text>
              <text x="290" y="150" text-anchor="middle">+0.02</text>
            </g>
            <g font-size="10.5">
              <text x="0" y="16">B1</text><rect class="bc" x="81.7" y="6" width="168.3" height="12"/><text x="77" y="16" text-anchor="end" class="mono">−0.084</text>
              <text x="0" y="33">B2</text><rect class="bc" x="160.2" y="23" width="89.8" height="12"/><text x="156" y="33" text-anchor="end" class="mono">−0.045</text>
              <text x="0" y="50">B3</text><rect class="bc" x="206.6" y="40" width="43.4" height="12"/><text x="202" y="50" text-anchor="end" class="mono">−0.022</text>
              <text x="0" y="67">B4</text><rect class="bc" x="246.5" y="57" width="3.5" height="12"/><text x="242" y="67" text-anchor="end" class="mono">−0.002</text>
              <text x="0" y="84">B5</text><rect class="bc" x="238" y="74" width="12" height="12"/><text x="234" y="84" text-anchor="end" class="mono">−0.006</text>
              <text x="0" y="101">B6</text><rect class="bc" x="207.3" y="91" width="42.7" height="12"/><text x="203" y="101" text-anchor="end" class="mono">−0.021</text>
              <text x="0" y="118">B7</text><rect class="bw" x="251" y="108" width="29.2" height="12"/><text x="284" y="118" class="mono" font-weight="600">+0.015</text>
              <text x="0" y="135">B8</text><rect class="bc" x="222.4" y="125" width="27.6" height="12"/><text x="218" y="135" text-anchor="end" class="mono">−0.014</text>
            </g>
          </svg>
        </div>
      </div>
    </div>
    <div class="foot"><span>평활 창은 노출된 OOF에서 선택 → 완전히 독립된 검증의 확증 아님</span><span>중심 ±30분 평균 → 무지연 실시간 복원 아님</span><span>Private 미확인</span></div>
'''

S11 = r'''
    <div class="eyebrow"><span>AI 활용 · P2 재현 패키지 검토</span><span>2026-09-07</span><span>최종 답안 <span class="sha">794268f1</span> 변경 없음</span></div>
    <h3>사람이 범위를 정하고, AI가 <em>재현을 막는 조건</em>을 찾고, 사람이 승인해 최소로 고쳤다</h3>
    <div class="body">
      <div class="steps">
        <div class="st" data-n="1"><span class="who">사람</span><b>목적과 경계</b>읽기 전용 독립 감사. 새 학습·답안 변경·업로드·최종 확인 금지. 실행 중인 P3 학습은 불변.</div>
        <div class="st" data-n="2"><span class="who">사람 → AI</span><b>입력과 산출물 요구</b>최종 ZIP·receipt·운영진 공지를 주고 <code>[근거|사실|결함|최소 수정|필수 여부]</code> 표로 답하게 함.</div>
        <div class="st hit" data-n="3"><span class="who">AI</span><b>발견한 차단 조건</b>평활 단계가 <code>assert sha == BASE</code>로 과거 답안과 비트까지 같아야 진행. 검증 PC의 GPU가 다르면 답안이 “다르게”가 아니라 “안” 나옴.</div>
        <div class="st" data-n="4"><span class="who">사람 승인 · 최소 수정</span><b>기록하고 계속</b>과거 SHA는 기대값·실제값·일치 여부로 기록만. 대신 현재 실행의 무결성 QA(스키마·키 순서·유한값·별도 PID 재생)를 통과해야 계속.</div>
        <div class="st" data-n="5"><span class="who">확인</span><b>같은 답안, 더 안전한 경로</b><div class="chk"><span>수리 테스트 12개 PASS</span><span>빈 모델 학습 3 fit 233.6초</span><span>저장 모델 경로 53.4초</span><span>두 경로 모두 794268f1</span></div></div>
      </div>
      <div class="limits">
        <div><b>여전히 별도 검증인 것</b>다른 GPU·CPU·새 가상환경·네트워크 차단 OS에서의 재현. 같은 PC의 두 프로세스 일치까지만 확인.</div>
        <div class="w"><b>AI 지적이 다 맞지는 않았다</b>“GitHub에 최종 평활 코드가 없다”는 지적은 ZIP 안 파일명으로만 검색한 오판. 재검토에서 정정.</div>
      </div>
    </div>
    <div class="foot"><span>점수는 그대로: 재현 위험 감소이지 성능 향상이 아님</span><span>당시 프롬프트·감사·정정 문서는 저장소에 원문 보존</span></div>
'''

def board(num, cls, variant, body, cap):
    label = "방향 A" if variant == "A" else "방향 B"
    return f'''
  <figure>
  <div class="slide {variant} {cls}" role="img" aria-label="{num}장 {label} 시안">{body.replace('{id}', f'{cls}{variant}')}
  </div>
  <figcaption><b>{label}</b><span>{cap}</span></figcaption>
  </figure>'''

sections = [
  ("4장 · P1 모델 구성과 구간 판단", "같은 내용, 방향 A(밝은 기술 편집형)와 방향 B(절제된 어두운 배경형)", 4, "s4", S4,
   "밝은 바탕 · 남색 글자 · 청록 강조 · 구조도 중심", "어두운 바탕 · 높은 대비 · 장식 없음 · 같은 구조도"),
  ("7장 · P2 동일 기반 Public 비교", "막대 축은 0℃에서 시작해 작은 개선을 과장하지 않음 · 내부 8블록은 부호와 값을 직접 표기", 7, "s7", S7,
   "처리 순서 한 줄 · Public 비교 · 내부 블록 위험", "같은 배치 · 어두운 바탕에서 막대와 값의 대비 유지"),
  ("11장 · AI의 실제 재현 결함 검토·수리 사례", "순서가 실제로 있는 내용이라 번호를 씀 · 성능 향상이 아니라 재현 경로의 위험 감소 사례", 11, "s11", S11,
   "5단계 타임라인 · 역할(사람/AI) 표기 · 한계와 오판을 같은 장에", "같은 타임라인 · 어두운 바탕 · 강조는 3단계 하나에만"),
]

out = [HEAD, '''
<div class="wrap">
<header>
  <div class="kicker">Ocean AI 본선 발표 · 대표 3장 · 디자인 A/B 시안</div>
  <h1>4장·7장·11장을 두 방향으로 실제 렌더링한 시안</h1>
  <p>아래 6개 보드는 말로 설명한 배치안이 아니라 실제로 렌더링한 시안입니다. 각 장의 발화, 작은 화면 문제, 선택 이유는 보고서 본문에 있습니다. 보드는 폭에 맞춰 글자와 그림이 함께 축소되므로, 온라인 발표에서 보일 크기를 가늠하려면 브라우저 폭을 넓혀 한 보드를 크게 보십시오.</p>
  <div class="rules">
    <span>모든 수치는 저장소 커밋 845e029의 receipt·result 파일에서 가져왔습니다</span>
    <span>없는 수치(P1 구성 요소별 점수 등)는 그리지 않았습니다</span>
    <span>막대 축은 0에서 시작합니다</span>
  </div>
</header>
''']
for title, sub, num, cls, body, capA, capB in sections:
    out.append(f'''
<section>
  <div class="sec-head"><h2>{title}</h2><span class="sub">{sub}</span></div>
  <div class="pair">{board(num, cls, "A", body, capA)}{board(num, cls, "B", body, capB)}
  </div>
</section>''')
out.append('''
<div class="notes">
  <h2>이 시안에서 지킨 것</h2>
  <ul>
    <li><b>실제 시안</b>: 위 6개 보드의 제목·문구·그림·배치는 그대로 슬라이드로 옮길 수 있습니다. 발화 대본은 슬라이드 본문에 넣지 않았습니다.</li>
    <li><b>없는 수치를 만들지 않음</b>: P1 구성 요소별 점수 막대는 없습니다. 4장의 행 수는 “양성 판정 개수”이지 성능이 아니라고 그림 안에 적었습니다.</li>
    <li><b>축과 단위</b>: 7장 Public 막대는 0℃에서 시작하고, 값·단위·비교 기준을 막대 옆에 직접 표기했습니다. 내부 8블록은 부호가 보이도록 0선 기준 양방향 막대로 그렸습니다.</li>
    <li><b>글자 크기</b>: 보드 폭 기준으로 제목 3.1%, 본문 1.4~1.65%, 각주 1.45%입니다. 1280px 폭에서는 제목 약 40px, 본문 약 18~21px, 각주 약 19px에 해당합니다. 발표 창이 작아지면 각주와 5단계 본문부터 읽기 어려워집니다.</li>
    <li><b>다른 것은 색과 바탕뿐</b>: A와 B는 같은 구조·같은 문구입니다. 두 방향의 차이를 색과 대비만으로 비교할 수 있게 일부러 같은 배치를 썼습니다.</li>
  </ul>
  <p class="src">출처: 저장소 choihyunjin1/-oceanaidata_track1, 브랜치 codex/p1-qc, 커밋 845e029 — reports/p1_original_source_package_20260906_v1, reports/p1_historical_path_audit_20260906_v1, reports/p2_l120_s3_smooth7_projection_20260907_v1, reports/p2_smooth7_portability_repair_20260907_v2, docs/ocean_v2_codex/P2_PORTABILITY_REPAIR_AND_AUDIT_RESPONSE_20260907.md, docs/ocean_v2_codex/FABLE_FINAL_SELECTION_RECHECK_20260907.md</p>
</div>
</div>
''')
open('slide_mockups_AB_20260912.html','w').write(''.join(out))
print("ok", sum(len(x) for x in out))
