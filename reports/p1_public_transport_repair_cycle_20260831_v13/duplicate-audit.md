# v13 duplicate audit

`CAUSAL_ASYNC_PEER_CONSENSUS_ADD_ONLY` literal rule은 미실행이다. 과거 synchronous RPCA는 window 0이었고, async latent GP는 learned score와 support 부족으로 no-output이었다. v13은 anchor bits만 사용하며 quorum 2와 inclusive 10-minute lookback을 고정한다. 같은 layer, year 경계, station discontinuity를 넘지 않으며 future/raw feature/label을 proposal에 쓰지 않는다. 실패 뒤 quorum이나 lookback을 완화하지 않는다.
