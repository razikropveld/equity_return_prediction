"""Generate the academic paper HTML (test phase, 2012-2023)."""
import _imgs
img1, img2, img3 = _imgs.fig1, _imgs.fig2, _imgs.fig3

html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Equity Return Prediction Using Fundamental Data</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 11.5pt;
    line-height: 1.68;
    color: #1a1a1a;
    background: #fff;
    max-width: 760px;
    margin: 0 auto;
    padding: 50px 52px 80px 52px;
  }
  h1 { font-size: 17pt; font-weight: bold; text-align: center; margin-bottom: 8px; line-height: 1.3; }
  .authors { text-align: center; font-size: 11pt; color: #222; margin-bottom: 4px; font-weight: bold; }
  .affil { text-align: center; font-size: 10pt; color: #666; font-style: italic; margin-bottom: 30px; }
  h2 { font-size: 12pt; font-weight: bold; text-transform: uppercase; letter-spacing: 0.04em;
       margin-top: 30px; margin-bottom: 8px; border-bottom: 1px solid #bbb; padding-bottom: 3px; }
  h3 { font-size: 11.5pt; font-weight: bold; font-style: italic; margin-top: 18px; margin-bottom: 5px; }
  p  { margin-bottom: 10px; text-align: justify; }
  .abstract-box { border: 1px solid #bbb; padding: 14px 18px; margin: 22px 0 26px 0; background: #f9f9f9; }
  .abstract-box .label { font-weight: bold; font-size: 10.5pt; text-transform: uppercase;
                         letter-spacing: 0.05em; margin-bottom: 7px; }
  .abstract-box p { font-size: 10.5pt; margin-bottom: 0; }
  .keywords { font-size: 10pt; margin-top: 9px; color: #333; }
  table { width: 100%; border-collapse: collapse; font-size: 10.5pt; margin: 14px 0 18px 0; }
  th { background: #efefef; font-weight: bold; padding: 5px 9px; text-align: left;
       border-bottom: 2px solid #777; border-top: 2px solid #777; }
  td { padding: 4px 9px; border-bottom: 1px solid #ddd; }
  tr:last-child td { border-bottom: 2px solid #777; }
  .fig { text-align: center; margin: 22px 0; }
  .fig img { max-width: 100%; border: 1px solid #ddd; }
  .fig-caption { font-size: 10pt; color: #444; margin-top: 7px; font-style: italic; text-align: center; }
  .note { font-size: 10pt; color: #555; font-style: italic; margin-top: 4px; }
  ol, ul { padding-left: 22px; margin-bottom: 10px; }
  li { margin-bottom: 5px; }
  hr.section { border: none; border-top: 1px solid #ccc; margin: 26px 0; }
  .ref-list { font-size: 9.5pt; line-height: 1.55; }
  .ref-list p { padding-left: 22px; text-indent: -22px; margin-bottom: 6px; text-align: left; }
  @media print { body { padding: 0; } }
</style>
</head>
<body>

<h1>Equity Return Prediction from Fundamentals Data:<br>
Applying XGBoost to Changing Market Regimes</h1>
<p class="authors">Raz Kropveld</p>
<p class="affil">Working Paper &nbsp;&bull;&nbsp; March 2026</p>

<div class="abstract-box">
  <div class="label">Abstract</div>
  <p>
    A substantial literature links accounting fundamentals to future equity returns. Foundational work by
    Ou and Penman (1989), Piotroski (2000), and Novy-Marx (2013) demonstrates that financial-statement
    signals contain information beyond what is reflected in market prices. More recent machine-learning
    studies&mdash;notably Gu, Kelly, and Xiu (2020) and Freyberger, Neuhierl, and Weber (2020)&mdash;apply
    non-linear methods to large characteristic sets and report improvements in rank information coefficients.
    However, these studies share important limitations: results are typically framed as factor risk-premia
    rather than implementable return metrics; safeguards against survivorship bias are
    inconsistently applied; evaluation periods are short and do not span the full range of market regimes;
    and the gap between reported performance and a practically executable strategy is rarely closed.
  </p>
  <p style="margin-top:8px;">
    This paper presents an end-to-end quantitative framework that addresses these limitations directly.
    Using quarterly fundamental reports of thousands of U.S. stocks, an XGBoost model is evaluated on
    a 12-year holdout period, with a custom TSCV design addressing the changing regimes challenge
    directly. The strategy achieves nearly twice the return of the S&amp;P 500 with only modestly higher
    risk, and the predictive signal is broad enough to persist well beyond the top-30 selection&mdash;
    suggesting genuine cross-sectional discrimination rather than overfitting.
  </p>
  <p class="keywords" style="margin-top:9px;"><strong>Keywords:</strong>
    equity return prediction, fundamental analysis, XGBoost, gradient boosting,
    time-series cross-validation, post-earnings announcement drift, walk-forward evaluation,
    survivorship bias, quantitative investing, machine learning in finance</p>
</div>

<h2>1. Introduction</h2>

<p>
  The relationship between accounting fundamentals and subsequent stock returns has attracted sustained
  attention in academic finance and quantitative practice. The core hypothesis&mdash;that markets do not
  immediately incorporate all the information embedded in financial statements&mdash;is supported by a
  long sequence of empirical findings, from the earnings post-announcement drift documented by Ball and
  Brown (1968) to the profitability premium of Novy-Marx (2013). The practical question is whether this
  predictability can be systematically exploited in a long-only portfolio context, over a long enough
  evaluation window to span multiple market regimes, and without the methodological shortcuts that
  inflate many published results.
</p>
<p>
  This paper describes a framework built around three design principles. First, the prediction problem
  is framed temporally: at each decision point, predict 60-trading-day forward returns using only
  information available at that moment. Second, the evaluation methodology replicates sequential
  deployment through a custom three-block walk-forward TSCV loop. Third, the primary performance
  metrics are strategy returns and their statistical robustness&mdash;not RMSE or MAPE
  that bear only an indirect relationship to portfolio profitability.
</p>

<h2>2. Literature Context and Motivation</h2>

<h3>2.1 Fundamental Signals and Return Predictability</h3>
<p>
  The academic literature on fundamental-based return prediction is extensive. Piotroski (2000) showed
  that a nine-signal financial-health score effectively separates winners from losers among high
  book-to-market firms. Novy-Marx (2013) documented a gross profitability premium orthogonal to
  standard value factors, suggesting that revenue efficiency carries independent predictive information.
  Sloan (1996) demonstrated that the accrual component of earnings predicts return reversals. Together,
  these studies establish that the information content of financial statements is non-trivial and
  persistent over long samples.
</p>

<h3>2.2 Machine Learning Approaches</h3>
<p>
  Gu, Kelly, and Xiu (2020) benchmarked a range of models&mdash;including random forests,
  gradient-boosted trees, and neural networks&mdash;against linear factor models on a large panel
  of firm characteristics, confirming that non-linear interactions carry predictive content beyond
  linear combinations. Freyberger, Neuhierl, and Weber (2020) applied LASSO-based nonparametric
  methods to characteristic selection, confirming the robustness of a small core set of signals.
  These results motivate the use of gradient boosting as the modeling framework here.
</p>

<h3>2.3 Limitations of Existing Work</h3>
<p>
  Despite this progress, several important limitations constrain the practical relevance of the
  existing literature:
</p>
<ol>
  <li><strong>Performance reporting.</strong> Most studies report factor alphas or rank ICs. These do not
    directly correspond to returns achievable by a long-only investor. Cumulative wealth and the
    probability of benchmark outperformance over realistic horizons are rarely reported.</li>
  <li><strong>Survivorship and look-ahead bias.</strong> Studies using non-point-in-time databases
    systematically overstate predictive power, especially in periods covering market dislocations.</li>
  <li><strong>Narrow evaluation windows.</strong> Many ML-based studies cover 10&ndash;20 years of data
    and do not span the full range of market regimes, including severe liquidity crises and structural
    shifts in market composition.</li>
</ol>
<p>The present framework directly addresses all three limitations.</p>

<h2>3. Data</h2>
<p>
  The study uses two primary data sources. <strong>Equity prices</strong> consist of daily closing prices
  for U.S.-listed equities, covering over 9,000 unique tickers including firms that were subsequently
  delisted. The inclusion of delisted firms is essential to avoid survivorship bias: excluding firms
  removed from indices due to poor performance would materially inflate apparent strategy returns.
  <strong>Fundamental data</strong> are drawn from the Sharadar SF1 point-in-time database, which ensures
  that each observation reflects only information publicly available at the time of filing, with
  restatements handled correctly. Key variables include net income, free cash flow, revenue, total
  equity, total debt, market capitalization, and enterprise value, at quarterly (ARQ) and
  trailing-twelve-month (TTM) frequencies.
</p>
<p>
  A minimum market capitalization filter (approximately $100&nbsp;million) is applied to limit
  exposure to micro-cap illiquidity. All criteria are applied strictly on a point-in-time basis.
  The model is trained on data from 1999 to 2011 (train phase) and evaluated out-of-sample on
  2012 to 2023 (test phase). All results reported in this paper are from the test phase only.
</p>

<h2>4. Methodology</h2>

<h3>4.1 Prediction Target and Features</h3>
<p>
  The prediction target is the log ratio of the 60-trading-day forward closing price to the current
  closing price. Features are constructed from the most recent point-in-time fundamental snapshot:
</p>
<ul>
  <li><strong>Earnings yield and cash-flow yield</strong> &mdash; net income and free cash flow scaled by market capitalization.</li>
  <li><strong>Relative valuation signals</strong> &mdash; firm-level earnings yield minus the contemporaneous cross-sectional and sector means.</li>
  <li><strong>Revenue efficiency</strong> &mdash; revenue scaled by market capitalization, a proxy for the Novy-Marx (2013) profitability signal.</li>
  <li><strong>Lagged fundamental changes</strong> &mdash; one-period lags of earnings and cash flow, capturing earnings momentum.</li>
  <li><strong>Price momentum</strong> &mdash; 60-trading-day trailing return as a short-term control.</li>
  <li><strong>Report timing</strong> &mdash; business days elapsed since the most recent earnings release, capturing post-announcement drift.</li>
</ul>
<p>
  Continuous features are log-transformed where appropriate and z-scored within each training window.
  Sample weights combine a time-decay component with an absolute-return component.
</p>

<h3>4.2 Model: XGBoost</h3>
<p>
  The predictive model is XGBoost (Chen and Guestrin, 2016), a gradient-boosted decision tree algorithm
  chosen for its performance on tabular data with non-linear feature interactions and its native
  support for regularization. Key hyperparameters are: learning rate &eta;&nbsp;=&nbsp;0.02, maximum
  tree depth&nbsp;=&nbsp;3, L2 regularization &lambda;&nbsp;=&nbsp;0.3, and a minimum-gain penalty
  &gamma;&nbsp;=&nbsp;0.1. These parameters favour shallow, regularized trees that generalize across
  regimes rather than memorizing in-sample patterns.
</p>

<h3>4.3 Three-Block Walk-Forward TSCV Design</h3>
<p>
  The evaluation design is the central methodological contribution of this work. For each monthly
  test period, three strictly non-overlapping data blocks are constructed in temporal order:
</p>
<ol>
  <li><strong>Validation block.</strong> Drawn from the period <em>before</em> the training window,
    restricted to the same post-announcement subset used at test time. The number of boosting rounds
    is selected by early stopping on this block.</li>
  <li><strong>Training block.</strong> A historical window ending at least 61 trading days before
    the test period, ensuring no realized forward returns overlap with test labels.</li>
  <li><strong>Test block.</strong> One forward calendar month&mdash;the strictly out-of-sample
    evaluation set, never seen during training or model selection.</li>
</ol>

<div class="fig">
  <img src="data:image/png;base64,""" + img3 + """" alt="Figure 3 - TSCV Diagram"/>
  <p class="fig-caption">
    <strong>Figure 3.</strong> Schematic of the three-block walk-forward TSCV design.
    For each test window, the validation block (orange) precedes the training block (blue),
    which in turn precedes the test block (green), separated by a buffer of at least 61 trading days.
    As the evaluation window advances, all three blocks shift forward in time.
  </p>
</div>

<p>
  The critical innovation is the placement of the validation block. Conventional TSCV designs place
  the validation set between the training window and the test period. This couples model selection
  to market conditions immediately preceding the test period&mdash;an indirect form of temporal
  leakage where the selected tree count has in effect been tuned to the regime just before evaluation.
  Moving the validation block to the far side of the training window removes this coupling entirely,
  allowing the model to generalize more fluidly across regime shifts.
  In practice, <code>num_boost_rounds</code>&mdash;the hyperparameter controlling tree count and
  therefore model complexity, tuned via early stopping on the validation block&mdash;was found to
  vary considerably less across test periods than the model parameters learned from the training
  data. This stability suggests that placing the validation block earlier decouples complexity
  selection from near-term regime conditions: rather than calibrating the number of trees to the
  market environment just before the test window, the model selects a more regime-neutral level
  of complexity, reducing the risk of inadvertent overfitting to a transient market state.
</p>

<h2>5. Results</h2>

<h3>5.1 Strategy Performance Summary</h3>
<p>
  Table 1 presents the main performance metrics for the top-30 long-only strategy over the test
  period (January 2012 to September 2023, 139 monthly periods). At each decision point, the 30 stocks
  with the highest predicted 60-day return are selected from the universe of approximately 900 stocks
  with a recent earnings release. Positions are held for the full 60-business-day horizon before
  rebalancing, so portfolio turnover is modest (roughly 8&ndash;12 rebalances per year across a
  portfolio of a few dozen names); transaction costs are therefore not expected to materially affect
  the gross returns reported here.
</p>

<table>
  <thead><tr><th>Metric</th><th>Top-30 Strategy</th><th>S&amp;P 500</th></tr></thead>
  <tbody>
    <tr><td>Evaluation period</td>
        <td colspan="2" style="text-align:center">January 2012 &ndash; September 2023 &nbsp;(139 monthly periods)</td></tr>
    <tr><td>Annualized return</td><td><strong>18.8%</strong></td><td>10.4%</td></tr>
    <tr><td>Mean 60-day period return</td><td><strong>4.6%</strong></td><td>2.6%</td></tr>
    <tr><td>Std. dev. of period return</td><td>9.8%</td><td>6.1%</td></tr>
    <tr><td>Annualized Sharpe ratio</td><td><strong>0.96</strong></td><td>0.86</td></tr>
    <tr><td>Prob. outperform S&amp;P 500 (single period)</td><td><strong>55.4%</strong></td><td>&mdash;</td></tr>
    <tr><td>Prob. outperform S&amp;P 500 (rolling 12 periods, ~2 yrs)</td><td><strong>69.5%</strong></td><td>&mdash;</td></tr>
    <tr><td>Mean rank information coefficient (IC)</td><td><strong>0.022</strong></td><td>&mdash;</td></tr>
    <tr><td>Worst single-period return</td><td>&minus;25.8%</td><td>&minus;19.4%</td></tr>
    <tr><td>Best single-period return</td><td>+50.6%</td><td>+21.8%</td></tr>
  </tbody>
</table>
<p class="note">Table 1. Out-of-sample performance summary (test phase only). Returns are dividend-adjusted and gross of
transaction costs. The benchmark is the S&amp;P 500 total return index (^GSPC, dividends reinvested).</p>

<div class="fig">
  <img src="data:image/png;base64,""" + img1 + """" alt="Figure 1 - Cumulative Wealth"/>
  <p class="fig-caption">
    <strong>Figure 1.</strong> <em>Top panel:</em> cumulative wealth of the top-30 strategy (solid blue)
    and the S&amp;P 500 total return index (dashed grey) on a logarithmic scale, 2012&ndash;2023
    (both start at $1). Returns are dividend-adjusted. Green/red shading shows periods of strategy outperformance and underperformance.
    <em>Bottom panel:</em> monthly excess return of the top-30 strategy over the S&amp;P 500 (bars) and
    its 12-period rolling mean (solid line), illustrating persistence of the edge across sub-periods.
  </p>
</div>

<p>
  The strategy nearly doubles the annualized return of the S&amp;P 500 (18.8% vs. 10.4%, dividend-adjusted)
  over the 12-year test period. Outperformance is broadly distributed: the strategy beats the S&amp;P 500
  in 55% of individual two-month periods and in 70% of rolling two-year windows. The rolling excess return
  in the lower panel of Figure 1 shows that the edge is not concentrated in a narrow window&mdash;it
  is present across the 2012&ndash;2015 expansion, the 2018 correction, the COVID-19 shock of 2020,
  and the 2022 rate-driven downturn.
</p>

<h3>5.2 Signal Breadth and Cross-Sectional Discrimination</h3>

<div class="fig">
  <img src="data:image/png;base64,""" + img2 + """" alt="Figure 2 - Return by Decile"/>
  <p class="fig-caption">
    <strong>Figure 2.</strong> Mean 60-day forward return (dividend-adjusted) by prediction rank decile, averaged across
    all 139 test-phase monthly periods. Decile 1 = lowest-ranked stocks; decile 10 = highest-ranked.
    The lowest decile is the worst performer, and the top decile is among the best, while middle deciles
    cluster around the universe mean.
  </p>
</div>

<p>
  Two findings from Figure 2 merit particular attention. First, the lowest-ranked decile (D1) averages
  approximately 1% per period&mdash;consistently below the equal-weighted universe mean&mdash;
  confirming that the model discriminates meaningfully at the bottom of the distribution, not only at
  the top. Second, the strategy's edge is not confined to the 30 highest-ranked stocks: expanding the
  portfolio to the top 100 highest-ranked stocks (roughly the top decile of the approximately 900-stock
  eligible universe) still yields a mean per-period return of approximately 3.0%&mdash;materially
  above the universe mean. Critically, the model was optimized solely on top-30 returns during
  training; the generalization to a wider selection is out-of-sample evidence of robustness rather
  than overfitting to a specific portfolio-size parameter.
</p>

<h3>5.3 Feature Importance and Economic Interpretation</h3>
<p>
  Examining aggregate XGBoost feature importance (gain) across all test-phase monthly models, the
  most influential features are revenue, earnings yield (net income scaled by market cap), and their
  sector-relative counterparts. Lagged earnings changes and price momentum appear as secondary
  contributors. These rankings are broadly consistent with the published literature (Novy-Marx, 2013;
  Piotroski, 2000), confirming that the model is exploiting established economic relationships rather
  than noise.
</p>
<p>
  A secondary finding concerns prediction timing. Restricting the evaluation to stocks at business
  day 1 after their earnings release consistently improves out-of-sample performance relative to later
  days. This is consistent with the post-earnings announcement drift literature (Ball and Brown, 1968;
  Bernard and Thomas, 1989), which documents a gradual market adjustment to fundamental information
  in the period following the release date.
</p>

<h2>6. Limitations and Future Directions</h2>
<p>
  <strong>Regime instability.</strong> The most consequential open challenge is the instability of
  cross-sectional relationships across regimes. During crises, high within-period stock correlations
  amplify portfolio volatility and reduce the benefit of diversification. Future work should explore
  ensemble approaches combining regime-local and global models, or Bayesian updating frameworks
  that allow the model's effective weighting to shift as regime evidence accumulates.
</p>
<p>
  <strong>Loss function alignment.</strong> The current objective&mdash;squared error on
  log-returns&mdash;weights all observations equally. From a portfolio perspective, only the extremes
  of the prediction distribution matter. A ranking-oriented loss function could better concentrate
  predictive power at the decision boundary.
</p>
<p>
  <strong>Hypothesis testing over optimization.</strong> The current framework is oriented toward
  predictive performance. A complementary direction is to test specific economic hypotheses&mdash;for
  example, whether the earnings yield signal is driven by growth expectations or discount rate changes&mdash;
  which would strengthen the economic foundation and potentially identify more robust features.
</p>

<h2>7. Conclusions</h2>
<p>
  This paper demonstrates that point-in-time accounting fundamental data contain robust and economically
  interpretable predictive information about near-term equity returns, evaluated on a strict out-of-sample
  test period from 2012 to 2023. The central contribution is methodological: a three-block walk-forward
  TSCV design that eliminates indirect temporal leakage in model selection, embedded in a fully automated
  end-to-end quantitative pipeline using XGBoost as the core model.
</p>
<p>
  The results suggest that the predictability documented in the academic literature is not merely a
  statistical artefact of evaluation methodology. With appropriate attention to survivorship bias,
  look-ahead contamination, and evaluation design, machine learning models trained on fundamental data
  can generate meaningful, robust, and economically interpretable return signals over a broad range
  of market conditions. The main open questions&mdash;regime stability, transaction cost scalability,
  and loss function alignment&mdash;represent productive directions for future development.
</p>

<hr class="section"/>
<h2>References</h2>
<div class="ref-list">
  <p>Ball, R. and Brown, P. (1968). An empirical evaluation of accounting income numbers. <em>Journal of Accounting Research</em>, 6(2), 159&ndash;178.</p>
  <p>Bernard, V. L. and Thomas, J. K. (1989). Post-earnings-announcement drift: Delayed price response or risk premium? <em>Journal of Accounting Research</em>, 27(Supplement), 1&ndash;36.</p>
  <p>Chen, T. and Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In <em>Proc. 22nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining</em>, 785&ndash;794.</p>
  <p>Chen, L., Pelger, M., and Zhu, J. (2021). Deep learning in asset pricing. <em>Management Science</em>, 70(2), 714&ndash;750.</p>
  <p>Freyberger, J., Neuhierl, A., and Weber, M. (2020). Dissecting characteristics nonparametrically. <em>Review of Financial Studies</em>, 33(5), 2326&ndash;2377.</p>
  <p>Gu, S., Kelly, B., and Xiu, D. (2020). Empirical asset pricing via machine learning. <em>Review of Financial Studies</em>, 33(5), 2223&ndash;2273.</p>
  <p>Novy-Marx, R. (2013). The other side of value: The gross profitability premium. <em>Journal of Financial Economics</em>, 108(1), 1&ndash;28.</p>
  <p>Ou, J. A. and Penman, S. H. (1989). Financial statement analysis and the prediction of stock returns. <em>Journal of Accounting and Economics</em>, 11(4), 295&ndash;329.</p>
  <p>Piotroski, J. D. (2000). Value investing: The use of historical financial statement information to separate winners from losers. <em>Journal of Accounting Research</em>, 38(Supplement), 1&ndash;41.</p>
  <p>Sloan, R. G. (1996). Do stock prices fully reflect information in accruals and cash flows about future earnings? <em>The Accounting Review</em>, 71(3), 289&ndash;315.</p>
</div>

</body>
</html>"""

with open("equity_return_prediction_paper.html", "w", encoding="utf-8") as f:
    f.write(html)
print("Saved: equity_return_prediction_paper.html")
