# An agent-based macroeconomic model in Julia

A bottom-up model of an economy with about 13,700 households and 1,234 firms across 68 industries. The model is calibrated to US microdata (IPUMS ACS 2024, BLS Consumer Expenditure Survey 2024, the BEA input–output use table) and to company financials. It is built with [Agents.jl](https://juliadynamics.github.io/Agents.jl/stable/).

This was a research studentship project at the University of Cambridge in summer and autumn 2026. **It is a work in progress.** Initialisation, job matching, production with supply chains, and consumption are implemented. Firm hiring, firing and wage setting (Phase 5) are not built yet. The first test run is in `Output/`.

---

## Why an agent-based model?

Standard macroeconomic models (DSGE) describe the economy through a small number of representative agents who optimise and are in equilibrium every period. That makes them tractable. However, it rules out by assumption much of what drives real fluctuations: heterogeneity in income, wealth and skills, local interactions between particular buyers and sellers, and disequilibrium dynamics such as unsold inventories, unfilled vacancies and price adjustment that takes time.

An agent-based model (ABM) builds the economy from the bottom up instead. Each household and firm follows simple behavioural rules. Aggregates such as output, the price level, unemployment and the wealth distribution *emerge* from their interactions rather than being imposed. This project explores:

1. how far such a model can be disciplined by real microdata rather than arbitrary parameters, and
2. which macro dynamics emerge from simple rules for job search, pricing, supplier choice and consumption.

## Repository contents

```
├── Data/
│   ├── Calibration/        # parameters derived from microdata (written by the Python scripts)
│   ├── IO table.xlsx       # BEA Summary Use table, 2024
│   ├── BLS CES 2024.xlsx   # Consumer Expenditure Survey, Table 1203 (by income band)
│   ├── QCEW annual wage and employment.csv   # BLS QCEW, used as a wage sanity check
│   ├── BEA-Industry-and-Commodity-Codes-and-NAICS-Concordance.xlsx
│   ├── PCEBridge_Detail.xlsx, PCE_concordance.xlsx   # reference for mapping spending to industries (not read by code)
│   └── IPUMS_codebook.cbk  # codebook for the (not included) IPUMS extract
├── Model/
│   ├── ABM2.jl                   # the model: agents, initialisation, all phases, run loop
│   ├── Init_analysis.py          # IPUMS ACS → wages, productivity, worker mix by sector
│   └── Consumption_analysis.py   # BLS CES → consumption function, income elasticities
└── Output/
    ├── aggregates.csv            # Test Run 1: one row per period (25 years)
    └── ABM2 Test Run 1.html      # Test Run 1 report with charts (download and open in a browser)
```

**Data not included.** Two raw inputs are licensed and cannot be redistributed:

| File | Source | How to obtain |
|---|---|---|
| `Data/IPUMS.csv` | IPUMS USA, 2024 ACS 1-year sample | Create an extract at [usa.ipums.org](https://usa.ipums.org) with variables `PERWT, AGE, EDUCD, DEGFIELD, EMPSTAT, CLASSWKRD, INDNAICS, WKSWORK1, UHRSWORK, INCWAGE`, in CSV format (see `IPUMS_codebook.cbk`) |
| `Data/Firm financials.xlsx` | FAME (Bureau van Dijk), university licence | Sheet `Results`, one row per firm, with columns `NAICS 2022`, `total_revenue`, `num_employees`, `finished_goods`, `cash_balance`, `inventory_ratio` |

Everything derived from IPUMS is already in `Data/Calibration/`, so the IPUMS extract is needed only to re-run `Init_analysis.py`. The firm file *is* needed to initialise the model.

---

## Model overview

### Agents

| Agent | Key state |
|---|---|
| **Household** | qualification (6 levels), experience band (8 bands), productivity, employer, reservation utility, wealth, wage, income, expenditure budget, consumption basket (one firm per sector) |
| **Firm** | sector (68 NAICS-based industries), TFP, workers, inventory, suppliers (one per input sector), output, markup, price, liquidity, input cost |

Money is in thousands of USD. Quantities are in units at the initial price of 1.

### Initialisation (calibration from data)

- **Firms** come from company financials. Headcount is compressed 1,000 : 1. Revenue, inventory and cash are scaled by the same factor. All prices start at 1, so initial output equals scaled revenue.
- **Production** is `Q = TFP × (Σ worker productivity)^0.7`. TFP is backed out so that initial output matches revenue. The markup is whatever makes revenue cover wages plus input costs.
- **Supply chains.** Each firm's input requirement per unit of output comes from the BEA 2024 use table, split across sectors that share a BEA code by their revenue. Each firm draws one supplier per input sector, with probability proportional to supplier size.
- **Workers.** Each worker's (qualification, experience) pair is drawn from the joint distribution observed in that industry in the ACS. Wages come from a log-wage regression on qualification, experience and industry. Productivity is lognormal around the qualification mean. 5% of households start unemployed.
- **Consumption.** The expenditure budget is `E = C₀ + MPC_wage × wage + MPC_wealth × wealth`. C₀ and MPC_wage are estimated from the CES across income bands. The budget is split across sectors with a **Linear Expenditure System**. Its sector shares are set so that household demand absorbs whatever output is not used as intermediate inputs. Its income elasticities are estimated from CES spending categories mapped to industries.

### The annual cycle (`model_step!`)

| Phase | What happens | Status |
|---|---|---|
| **1. Job matching** | Unemployed workers see vacancies that match their qualification and experience. They rank vacancies by wage × idiosyncratic fit above their reservation utility, and choose up to 10 applications that maximise expected value given competition. Firms hire with probability proportional to productivity, and workers accept their best offer. The reservation utility of anyone still unemployed decays by 20%. | Implemented (no vacancies are created yet) |
| **2. Production** | Firms may switch to a cheaper supplier. They produce, buy inputs at last period's prices, and set price = (1 + markup) × unit cost, adjusted for the gap between opening inventory and the sector's target inventory ratio. | Implemented |
| **3. Consumption** | Households may switch to a cheaper firm in each sector. They recompute their budget and LES basket at current prices, then buy. | Implemented |
| **4. Income** | Firms pay wages. As a placeholder, all positive profit is paid out as dividends in proportion to household wealth. | Interim version |
| **5. Wages and vacancies** | Retained profit, wage adjustment, vacancy creation, firing and quits | Not yet built |

---

## Code guide: `Model/ABM2.jl`

The file is organised in numbered sections that match the phases above.

| Section | Contents |
|---|---|
| **0. Create agents** | `Vacancy` and `Consumption` structs; `Household` and `Firm` agents combined with `@multiagent EconomicAgent` |
| **0.2 Sectors** | Table of the 68 model sectors, NAICS → model sector (`sector_code`), model sector → BEA code (`SECTOR_BEA`) |
| **0.3 Qualifications and experience** | Definitions shared with `Init_analysis.py` |
| **0.5 Initialise** | Global parameters, then: **0.5.1** loading data and input coefficients (`read_io_table`, `load_input_coefficients`, `scale_input_coefficients!`); **0.5.2** firms (`add_firms!`, `assign_suppliers!`); **0.5.3** households (`add_household!`, `add_workers!`, `add_unemployed!`); **0.5.4** expenditure and LES baskets (`final_demand`, `exp_parameters`, `exp_basket!`, `add_consumption!`); **0.5.5** `initialise(; seed)` builds the model |
| **1. Job matching** | `job_matching!` → `index_vacancies`, `gather_applications!` (with `expected_value` / `choose_applications`), `screen_applicants`, `finalise_hires!`, `decay_unemployed!` |
| **2. Production** | `production!` → `switch_suppliers!`, `buy_inputs!`, `set_prices!`; shared `switch_firm` rule |
| **3. Consumption** | `consumption!` → `switch_shops!`, `update_quantities!`, `buy_goods!` |
| **4. Income** | `wage_transfer!`, `pay_dividends!` |
| **Build model** | `model_step!` runs the phases in order |
| **Iterate** | `period_stats` (one row of aggregates), `state_ok` (stops the run on NaN or non-positive prices), `run_model(; n_periods, seed)` writes `Output/aggregates.csv` |

Parameters are declared as `const` at the top of the section that uses them. For example, `PRICE_CUT_ELASTICITY`, `PRICE_RISE_ELASTICITY` and `SWITCH_SENSITIVITY` are in Section 2, and `MPC_WEALTH` and `LABOUR_ELASTICITY` are in 0.5.

### Calibration scripts

| Script | Reads | Writes to `Data/Calibration/` | Used in `ABM2.jl` for |
|---|---|---|---|
| `Init_analysis.py` | `IPUMS.csv`, `QCEW…csv` (check), sector list in `ABM2.jl` | `wage_regression`, `qual_productivity`, `exp_wage_premium`, `wage_residuals`, `wage_by_cell`, `qual_exp_mix_by_sector` | worker mix, wages and productivity by sector |
| `Consumption_analysis.py` | `BLS CES 2024.xlsx` | `consumption_by_band`, `consumption_function`, `income_elasticity` | `CONS_0`, `MPC_WAGE`, LES income elasticities |

`qual_mix_by_sector.csv` and `exp_mix_by_sector.csv` are marginal distributions from an earlier version of the script. The current model does not read them.

---

## Running the model

**Julia** (1.10 or later). Install the required packages once:

```julia
using Pkg
Pkg.add(["Agents", "StatsBase", "XLSX", "DataFrames", "CSV"])
```

`Random` and `Statistics` are part of the standard library. The code uses `@multiagent` and `variantof`, so it needs Agents.jl v6.

Then, with `Data/Firm financials.xlsx` in place, run it from the terminal:

```bash
julia Model/ABM2.jl          # 25 periods, seed 1, writes Output/aggregates.csv
```

or from the REPL:

```julia
include("Model/ABM2.jl")
model, data = run_model(n_periods = 25, seed = 1)
```

**Python** (optional, only to rebuild the calibration files). This needs `pandas`, `numpy` and `openpyxl`.

```bash
python3 Model/Consumption_analysis.py
python3 Model/Init_analysis.py        # needs Data/IPUMS.csv
```

---

## Test Run 1: results

`Output/ABM2 Test Run 1.html` contains the full report with charts. It covers 25 annual periods with seed 1 and Phases 1–4 as above.

- **The accounting holds.** Household wealth plus firm liquidity is constant in every period, so no money is created or destroyed.
- **The circular flow balances at first.** For roughly the first decade, household spending stays within a few percent of income and the median price stays close to 1.
- **Without Phase 5, quantities cannot adjust.** Output is fixed, demand is not capped by stock, and imbalances accumulate as inventory backlogs.
- **Backlogs turn into inflation that spreads.** Backlogged firms keep raising prices. Markup pricing passes those increases into their customers' unit costs through the supply chain. Upper-tail prices then explode from about period 6.

This instability is the expected result of running the model without the quantity-adjustment phase. It identifies what has to be built next.

## Next steps

1. **Phase 5:** vacancy creation, hiring and firing, so that surpluses and backlogs are cleared by quantities as well as prices.
2. Deliver only what is in stock, so that unfilled orders do not become permanent backlogs.
3. Bound the price adjustment (a multiplicative rule that cannot go negative, plus a cap on the inventory signal).
4. Add a credit limit or firm exit, so that liquidity cannot go negative without limit.
5. Retained earnings and wage bargaining, in place of the placeholder that pays out all profit as dividends.
6. Validation against stylised facts: firm-size and wealth distributions, the Beveridge and Phillips curves.

---

## Data sources

- Ruggles, S. et al. *IPUMS USA*, 2024 American Community Survey. Minneapolis, MN: IPUMS.
- US Bureau of Labor Statistics. *Consumer Expenditure Surveys*, 2024 (Table 1203); *Quarterly Census of Employment and Wages*.
- US Bureau of Economic Analysis. *Input–Output Accounts: Summary Use Table*, 2024; *PCE Bridge Tables*; *Industry and Commodity Codes and NAICS Concordance*.
- Bureau van Dijk. *FAME* company financials (licensed; not included).

## Acknowledgements

Research studentship supervised by Philip Kalikman, University of Cambridge, 2026.

## Author

Donny Feng, BA Economics, Emmanuel College, Cambridge
