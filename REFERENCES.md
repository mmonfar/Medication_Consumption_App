# References

Generated from `references/citations.db` by `python -m medication_app.cite export`.
Do not edit by hand -- edit the database and regenerate.

11 sources.

## barber-2023

Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. (2023) Conformal prediction beyond exchangeability. *The Annals of Statistics*, 51(2). 816–845.

Relevant if real dispensing data shows distribution drift, which would break the exchangeability assumption our intervals rest on.

**Retrieved**

- 2026-09-18 via bibliography of fpp (identified, not read directly) — cited in fpppy/05-05-toolbox.md §5.5

**Used for**

- Conformal prediction beyond exchangeability is needed under distribution drift [via fpp §5.5] → `not implemented — relevant once real data arrives`

## bates-granger-1969

Bates, J. M., & Granger, C. W. J. (1969) The combination of forecasts. *Operational Research Quarterly*, 20(4). 451–468.

Basis for forecast combination — a candidate improvement not yet implemented here.

**Retrieved**

- 2026-09-18 via bibliography of fpp (identified, not read directly) — cited in fpppy/13-13-practical.md §13.4

**Used for**

- Averaging forecasts from several models reliably improves accuracy [via fpp §13.4] → `not implemented — candidate next step`

## croston-1972

Croston, J. D. (1972) Forecasting and stock control for intermittent demands. *Operational Research Quarterly*, 23(3). 289–303.

**Retrieved**

- 2026-09-18 via bibliography of fpp (identified, not read directly) — cited in fpppy/13-13-practical.md §13.2

**Used for**

- Croston's method splits the series into non-zero demand and inter-arrival time, smoothing each by SES, forecasting the ratio [via fpp §13.2] → `statsforecast CrostonClassic`

## fpp-python

Hyndman, R. J., & Athanasopoulos, G. (n.d.) *Forecasting: Principles and Practice (the Pythonic Way)*. https://otexts.com/fpppy/

*Licence:* CC BY-NC-ND — paraphrased here; no substantive text reproduced

Primary methodological source for this project. Held in the vault as `forecasting-principles-practice`, 773 indexed chunks.

**Retrieved**

- 2026-09-18 via refs_books brain vault (MCP: brain_search / brain_read_source_page) — fpppy/05-05-toolbox.md — 5.5 Conformal prediction; Scaled errors (query: "prediction intervals; MASE scaled error")
- 2026-09-18 via refs_books brain vault (MCP: brain_search / brain_read_source_page) — fpppy/07-07-regression.md — Spurious regression; scenario-based forecasting (query: "dynamic regression scenario forecasting predictors")
- 2026-09-18 via refs_books brain vault (MCP: brain_search / brain_read_source_page) — fpppy/13-13-practical.md — 13.2 Time series of counts (query: "time series forecasting demand exponential smoothing")
- 2026-09-18 via refs_books brain vault (MCP: brain_search / brain_read_source_page) — fpppy/13-13-practical.md — 13.4 Forecast combinations (query: "combining forecasts")

**Used for**

- Split conformal intervals use empirical quantiles of past absolute h-step-ahead errors, assuming only exchangeability [§5.5] → `medication_app/intervals.py fit_conformal()` _(verified against this project's data)_
- High R-squared with autocorrelated residuals is the signature of spurious regression [§7.3, Spurious regression] → `medication_app/models.py Elasticity.is_suspect` _(verified against this project's data)_
- When a predictor's own future values are unknown, use scenario-based forecasting rather than a fitted forecast of the predictor [§7.6] → `medication_app/models.py Elasticity.adjust() (comorbidity slider)` _(verified against this project's data)_
- Outliers can be found via STL decomposition and a 3-IQR rule on the remainder [§13.6] → `not implemented — relevant once real data arrives`

## hyndman-koehler-2006

Hyndman, R. J., & Koehler, A. B. (2006) Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4). 679–688.

Origin of MASE and RMSSE.

**Retrieved**

- 2026-09-18 via bibliography of fpp (identified, not read directly) — cited in fpppy/05-05-toolbox.md, Scaled errors

**Used for**

- MASE and RMSSE scale errors by in-sample naive error, making them scale-free and defined when actuals are zero [via fpp §5.9, Scaled errors] → `medication_app/backtest.py mase(), rmsse()` _(verified against this project's data)_
- Absolute error is minimised by the conditional median, which is zero for a mostly-zero series, so MASE rewards forecasting zero; squared error targets the mean instead [via fpp §5.9] → `medication_app/backtest.py cross_validate() primary metric` _(verified against this project's data)_

## statsforecast

Nixtla (2025) statsforecast (version 2.1.1). https://nixtlaverse.nixtla.io/statsforecast/

Implementation of AutoETS, AutoARIMA, CrostonClassic and CrostonSBA used here.

**Retrieved**

- 2026-09-18 via pip install — version 2.1.1

## shenstone-hyndman-2005

Shenstone, L., & Hyndman, R. J. (2005) Stochastic models underlying Croston's method for intermittent demand forecasting. *Journal of Forecasting*, 24(6). 389–402.

Establishes that no stochastic model underlies Croston's method — the reason it admits no analytic prediction interval, and the reason this project uses conformal intervals instead.

**Retrieved**

- 2026-09-18 via bibliography of fpp (identified, not read directly) — cited in fpppy/13-13-practical.md §13.2

**Used for**

- No stochastic model underlies Croston's method, so it yields no analytic prediction intervals [via fpp §13.2] → `medication_app/models.py NO_INTERVAL_MODELS` _(verified against this project's data)_

## stankeviciute-2021

Stankeviciute, K., Alaa, A. M., & van der Schaar, M. (2021) Conformal time-series forecasting. *Advances in Neural Information Processing Systems*, 34. 6216–6228. Curran Associates, Inc..

**Retrieved**

- 2026-09-18 via bibliography of fpp (identified, not read directly) — cited in fpppy/05-05-toolbox.md §5.5

## syntetos-boylan-2001

Syntetos, A. A., & Boylan, J. E. (2001) On the bias of intermittent demand estimates. *International Journal of Production Economics*, 71. 457–466.

**Retrieved**

- 2026-09-18 via bibliography of fpp (identified, not read directly) — cited in fpppy/13-13-practical.md §13.2

## syntetos-boylan-2005

Syntetos, A. A., & Boylan, J. E. (2005) The accuracy of intermittent demand estimates. *International Journal of Forecasting*, 21(2). 303–314.

Source of the SBA deflation factor correcting Croston's bias. CrostonSBA is the model this project ships for Meropenem.

**Retrieved**

- 2026-09-18 via bibliography of fpp (identified, not read directly) — cited in fpppy/13-13-practical.md §13.2

**Used for**

- Croston's estimates are biased; SBA applies a deflating factor to correct it [via fpp §13.2] → `medication_app/models.py CrostonSBA candidate` _(verified against this project's data)_

## refs-books-guide

refs_books vault (curated) (2026) Application guide — which forecasting method, when. *refs_books/books/forecasting-principles-practice/APPLICATION-GUIDE.md*,

Vault-local curated selection guide, explicitly not the book's text. Its section 4 decision path governs model choice here.

**Retrieved**

- 2026-09-18 via refs_books brain vault (MCP: brain_search / brain_read_source_page) — sections 2 and 4 (query: "method selection by data characteristics")

**Used for**

- A naive/seasonal-naive/mean benchmark must be fitted first, and nothing ships unless it beats the right benchmark on held-out data [§4.1] → `medication_app/backtest.py select_model()` _(verified against this project's data)_
- ETS/ARIMA require ample history; very short series point to benchmarks only [§2, §4.3] → `medication_app/models.py candidate_models()` _(verified against this project's data)_
- Intermittent/count data with many zero periods points to Croston's method and variants [§2] → `medication_app/models.py candidate_models()` _(verified against this project's data)_
- Methods must be compared by rolling-origin cross-validation, not a single train/test split [§4.9] → `medication_app/backtest.py cross_validate()` _(verified against this project's data)_
