"""Seed and export the citation database: python -m medication_app.cite [export]

The seed below is the actual research trail for this project's forecasting
design. Sources reached through the refs_books brain vault record that route in
their retrieval row; primary papers cited *by* those sources are recorded as
sources in their own right, with a retrieval noting they were identified via
the bibliography rather than read directly. That distinction matters -- it is
the difference between "we read this" and "our source cites this".
"""

from __future__ import annotations

import sys

from .citations import (
    Source,
    add_claim,
    add_retrieval,
    add_source,
    connect,
    export_markdown,
    stats,
)

VAULT = "refs_books brain vault (MCP: brain_search / brain_read_source_page)"
BIBLIOGRAPHY = "bibliography of fpp (identified, not read directly)"

SOURCES = [
    Source(
        key="fpp-python",
        type="book",
        authors="Hyndman, R. J., & Athanasopoulos, G.",
        year="n.d.",
        title="Forecasting: Principles and Practice (the Pythonic Way)",
        url="https://otexts.com/fpppy/",
        licence="CC BY-NC-ND — paraphrased here; no substantive text reproduced",
        notes="Primary methodological source for this project. Held in the vault as `forecasting-principles-practice`, 773 indexed chunks.",
    ),
    Source(
        key="refs-books-guide",
        type="internal",
        authors="refs_books vault (curated)",
        year="2026",
        title="Application guide — which forecasting method, when",
        container="refs_books/books/forecasting-principles-practice/APPLICATION-GUIDE.md",
        notes="Vault-local curated selection guide, explicitly not the book's text. Its section 4 decision path governs model choice here.",
    ),
    Source(
        key="croston-1972",
        type="article",
        authors="Croston, J. D.",
        year="1972",
        title="Forecasting and stock control for intermittent demands",
        container="Operational Research Quarterly",
        volume="23(3)",
        pages="289–303",
    ),
    Source(
        key="syntetos-boylan-2005",
        type="article",
        authors="Syntetos, A. A., & Boylan, J. E.",
        year="2005",
        title="The accuracy of intermittent demand estimates",
        container="International Journal of Forecasting",
        volume="21(2)",
        pages="303–314",
        notes="Source of the SBA deflation factor correcting Croston's bias. CrostonSBA is the model this project ships for Meropenem.",
    ),
    Source(
        key="syntetos-boylan-2001",
        type="article",
        authors="Syntetos, A. A., & Boylan, J. E.",
        year="2001",
        title="On the bias of intermittent demand estimates",
        container="International Journal of Production Economics",
        volume="71",
        pages="457–466",
    ),
    Source(
        key="shenstone-hyndman-2005",
        type="article",
        authors="Shenstone, L., & Hyndman, R. J.",
        year="2005",
        title="Stochastic models underlying Croston's method for intermittent demand forecasting",
        container="Journal of Forecasting",
        volume="24(6)",
        pages="389–402",
        notes="Establishes that no stochastic model underlies Croston's method — the reason it admits no analytic prediction interval, and the reason this project uses conformal intervals instead.",
    ),
    Source(
        key="hyndman-koehler-2006",
        type="article",
        authors="Hyndman, R. J., & Koehler, A. B.",
        year="2006",
        title="Another look at measures of forecast accuracy",
        container="International Journal of Forecasting",
        volume="22(4)",
        pages="679–688",
        notes="Origin of MASE and RMSSE.",
    ),
    Source(
        key="stankeviciute-2021",
        type="inproceedings",
        authors="Stankeviciute, K., Alaa, A. M., & van der Schaar, M.",
        year="2021",
        title="Conformal time-series forecasting",
        container="Advances in Neural Information Processing Systems",
        volume="34",
        pages="6216–6228",
        publisher="Curran Associates, Inc.",
    ),
    Source(
        key="barber-2023",
        type="article",
        authors="Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J.",
        year="2023",
        title="Conformal prediction beyond exchangeability",
        container="The Annals of Statistics",
        volume="51(2)",
        pages="816–845",
        notes="Relevant if real dispensing data shows distribution drift, which would break the exchangeability assumption our intervals rest on.",
    ),
    Source(
        key="bates-granger-1969",
        type="article",
        authors="Bates, J. M., & Granger, C. W. J.",
        year="1969",
        title="The combination of forecasts",
        container="Operational Research Quarterly",
        volume="20(4)",
        pages="451–468",
        notes="Basis for forecast combination — a candidate improvement not yet implemented here.",
    ),
    Source(
        key="statsforecast",
        type="software",
        authors="Nixtla",
        year="2025",
        title="statsforecast (version 2.1.1)",
        url="https://nixtlaverse.nixtla.io/statsforecast/",
        notes="Implementation of AutoETS, AutoARIMA, CrostonClassic and CrostonSBA used here.",
    ),
]

RETRIEVALS = [
    ("fpp-python", VAULT, "fpppy/13-13-practical.md — 13.2 Time series of counts", "time series forecasting demand exponential smoothing"),
    ("fpp-python", VAULT, "fpppy/05-05-toolbox.md — 5.5 Conformal prediction; Scaled errors", "prediction intervals; MASE scaled error"),
    ("fpp-python", VAULT, "fpppy/07-07-regression.md — Spurious regression; scenario-based forecasting", "dynamic regression scenario forecasting predictors"),
    ("fpp-python", VAULT, "fpppy/13-13-practical.md — 13.4 Forecast combinations", "combining forecasts"),
    ("refs-books-guide", VAULT, "sections 2 and 4", "method selection by data characteristics"),
    ("croston-1972", BIBLIOGRAPHY, "cited in fpppy/13-13-practical.md §13.2", ""),
    ("syntetos-boylan-2005", BIBLIOGRAPHY, "cited in fpppy/13-13-practical.md §13.2", ""),
    ("syntetos-boylan-2001", BIBLIOGRAPHY, "cited in fpppy/13-13-practical.md §13.2", ""),
    ("shenstone-hyndman-2005", BIBLIOGRAPHY, "cited in fpppy/13-13-practical.md §13.2", ""),
    ("hyndman-koehler-2006", BIBLIOGRAPHY, "cited in fpppy/05-05-toolbox.md, Scaled errors", ""),
    ("stankeviciute-2021", BIBLIOGRAPHY, "cited in fpppy/05-05-toolbox.md §5.5", ""),
    ("barber-2023", BIBLIOGRAPHY, "cited in fpppy/05-05-toolbox.md §5.5", ""),
    ("bates-granger-1969", BIBLIOGRAPHY, "cited in fpppy/13-13-practical.md §13.4", ""),
    ("statsforecast", "pip install", "version 2.1.1", ""),
]

CLAIMS = [
    (
        "A naive/seasonal-naive/mean benchmark must be fitted first, and nothing ships unless it beats the right benchmark on held-out data",
        "refs-books-guide", "§4.1", "medication_app/backtest.py select_model()", True,
    ),
    (
        "ETS/ARIMA require ample history; very short series point to benchmarks only",
        "refs-books-guide", "§2, §4.3", "medication_app/models.py candidate_models()", True,
    ),
    (
        "Intermittent/count data with many zero periods points to Croston's method and variants",
        "refs-books-guide", "§2", "medication_app/models.py candidate_models()", True,
    ),
    (
        "Croston's method splits the series into non-zero demand and inter-arrival time, smoothing each by SES, forecasting the ratio",
        "croston-1972", "via fpp §13.2", "statsforecast CrostonClassic", False,
    ),
    (
        "Croston's estimates are biased; SBA applies a deflating factor to correct it",
        "syntetos-boylan-2005", "via fpp §13.2", "medication_app/models.py CrostonSBA candidate", True,
    ),
    (
        "No stochastic model underlies Croston's method, so it yields no analytic prediction intervals",
        "shenstone-hyndman-2005", "via fpp §13.2", "medication_app/models.py NO_INTERVAL_MODELS", True,
    ),
    (
        "MASE and RMSSE scale errors by in-sample naive error, making them scale-free and defined when actuals are zero",
        "hyndman-koehler-2006", "via fpp §5.9, Scaled errors", "medication_app/backtest.py mase(), rmsse()", True,
    ),
    (
        "Absolute error is minimised by the conditional median, which is zero for a mostly-zero series, so MASE rewards forecasting zero; squared error targets the mean instead",
        "hyndman-koehler-2006", "via fpp §5.9", "medication_app/backtest.py cross_validate() primary metric", True,
    ),
    (
        "Split conformal intervals use empirical quantiles of past absolute h-step-ahead errors, assuming only exchangeability",
        "fpp-python", "§5.5", "medication_app/intervals.py fit_conformal()", True,
    ),
    (
        "Conformal prediction beyond exchangeability is needed under distribution drift",
        "barber-2023", "via fpp §5.5", "not implemented — relevant once real data arrives", False,
    ),
    (
        "High R-squared with autocorrelated residuals is the signature of spurious regression",
        "fpp-python", "§7.3, Spurious regression", "medication_app/models.py Elasticity.is_suspect", True,
    ),
    (
        "When a predictor's own future values are unknown, use scenario-based forecasting rather than a fitted forecast of the predictor",
        "fpp-python", "§7.6", "medication_app/models.py Elasticity.adjust() (comorbidity slider)", True,
    ),
    (
        "Methods must be compared by rolling-origin cross-validation, not a single train/test split",
        "refs-books-guide", "§4.9", "medication_app/backtest.py cross_validate()", True,
    ),
    (
        "Averaging forecasts from several models reliably improves accuracy",
        "bates-granger-1969", "via fpp §13.4", "not implemented — candidate next step", False,
    ),
    (
        "Outliers can be found via STL decomposition and a 3-IQR rule on the remainder",
        "fpp-python", "§13.6", "not implemented — relevant once real data arrives", False,
    ),
]


def seed() -> None:
    with connect() as connection:
        for source in SOURCES:
            add_source(connection, source)
        for source_key, via, locator, query in RETRIEVALS:
            add_retrieval(connection, source_key, via, locator, query)
        for claim, source_key, section, implemented_in, verified in CLAIMS:
            add_claim(connection, claim, source_key, section, implemented_in, verified)
        counts = stats(connection)
    print(
        f"seeded: {counts['sources']} sources, {counts['retrievals']} retrievals, "
        f"{counts['claims']} claims ({counts['verified_claims']} verified against this project's data)"
    )


def main() -> None:
    seed()
    with connect() as connection:
        path = export_markdown(connection)
    print(f"exported {path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "export":
        with connect() as connection:
            print(f"exported {export_markdown(connection)}")
    else:
        main()
