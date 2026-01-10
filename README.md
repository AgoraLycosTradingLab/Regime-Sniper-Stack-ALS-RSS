***********************
General Disclaimer
This material is provided for informational and educational purposes only and does not constitute investment advice, financial advice, trading advice, or a recommendation to buy or sell any securities or financial instruments.

Agora Lycos Trading Lab is a research-focused entity. All content reflects research opinions, models, and historical analysis, which may be incomplete, incorrect, or change without notice. Past performance is not indicative of future results.

No representation is made regarding the accuracy, completeness, or suitability of the information provided. Use of any information is at your own risk.
***********************

Regime Sniper Stack(ALS-RSS)
Run the files in this order:
1. Alpha R1 Macro Indicator
2. Entropy Regime Scanner
3. Stock Entropy Regime Scanner
4. PII Stock Scanner
5. RMVF Technical Scanner

# Regime-Sniper-Stack-ALS-RSS
ALS-RSS (Regime Sniper Stack) is a modular, Python-based macro-to-micro trading framework. It detects market regimes, gates risk exposure, activates factor sleeves, and outputs disciplined signals. Built for research, backtesting, and live simulation with transparency and control.

Regime Sniper Stack(ALS-RSS)
1. Alpha R1 Macro Indicator
   Alpha R1 Macro Indicator is the top-down regime engine of ALS-RSS. It ingests cross-asset market proxies (equities, volatility, credit, rates, USD, oil), computes trend and risk signals, and classifies the market into Risk-On, Risk-Off, or Transition regimes. It outputs regime confidence and dynamically gates factor families for downstream scanners.
   
2. Entropy Regime Scanner
   Entropy Regime Scanner (ERS) detects market volatility regimes using an entropy-based Markov transition model. It identifies low-entropy conditions associated with higher forward price movement, independent of direction. The scanner gates risk, quantifies regime “lift,” and classifies assets into actionable buckets (pullback, breakout, overlap) for disciplined signal generation.
   
3. Stock Entropy Regime Scanner
   Stock Entropy Regime Scanner (SERS) applies entropy-based volatility regime detection at the individual stock level. It filters for liquid equities, identifies low-entropy conditions linked to higher forward movement, and uses a recent-regime gate to improve timing. Stocks are classified into pullback, breakout, overlap, or watchlist buckets for actionable equity selection.
   
4. PII Stock Scanner
   PII Stock Scanner identifies high-quality pullback entries using the Pullback Integrity Index (PII). It blends trend alignment, pullback depth, volatility, momentum, and volume behavior into a single score, then applies strict risk filters and an EMA20 trigger to generate disciplined BUY signals and a ranked watchlist across filtered equity universes.
   
8. RMVF Technical Scanner
  RMVF Technical Scanner is a rule-based execution and confirmation engine. It evaluates stocks using a multi-signal checklist—RSI, MACD, EMA trend, VWAP slope, Fibonacci reactions, ATR expansion, and volume confirmation. Candidates must pass a minimum score threshold to qualify, ensuring momentum-backed, technically validated trade setups.
