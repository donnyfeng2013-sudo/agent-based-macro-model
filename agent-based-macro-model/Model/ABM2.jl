using Agents
using StatsBase: sample, Weights
using Statistics: median, mean, quantile
using Random
using XLSX, DataFrames, CSV                   

# ------------------------------------------------------------------
# 0. Create agents
# ------------------------------------------------------------------
mutable struct Vacancy
    firm::Int
    qualification::Int
    experience::Int
    wage::Float64
    applicants::Dict{Int, Float64}      # worker id; worker fit = wage * idiosyncratic fit
    pool::Float64                       # sun of applicants' productivity
end

struct Consumption
    sector::Int
    firm::Int
    quantity::Float64
    price::Float64
end

@agent struct Household(NoSpaceAgent)
    # fixed characteristics
    const qualification::Int            # index into QUALIFICATIONS (0.3)
    experience::Int                     # experience band, index into EXP_BANDS (0.3)
    const productivity::Float64         # skills, work ethic, intelligence etc.

    # employment variables
    employer::Int                       # id of employer, 0 = unemployed
    reservation_utility::Float64        # last drawn wage * fit, decays with no. of periods unemployed

    # income variables
    wealth::Float64
    wage::Float64
    income::Float64                     # wage + bonuses + dividends

    # consumption variables
    expenditure::Float64                # spending budget this period; last period's value is used for the income effect
    consumption::Vector{Consumption}    # one entry per sector bought from: sector, firm id, quantity, price
end

@agent struct Firm(NoSpaceAgent)
    # fixed characteristics
    sector::Int                         # index into SECTORS (0.2)
    tfp::Float64                        # firm productivity: Q = tfp * (Σ worker productivity)^LABOUR_ELASTICITY

    # quantity variables
    inventory::Float64                  # stock of finished goods, target = model.inventory_target[sector] × output, < 0 is a backlog
    workers::Vector{Int}                # list of worker ids employed
    inputs::Vector{Int}                 # in order of sector, lists ids of firms that this firm consumed from last period, 0 for no firm (no inputs required of that sector)
    output::Float64                     # units produced this period

    # price variables
    markup::Float64                     # margin over unit costs, saved from previous period
    price::Float64                      # set by taking markup over unit_cost, then adjusted by demand signals from inventory mismatch
    liquidity::Float64                  # firm's accumulated profit
    input_cost::Float64                 # expenditure on inputs (at last period's prices)

end

@multiagent EconomicAgent(Household, Firm)

# ------------------------------------------------------------------
# 0.2 Sectors: 3-digit NAICS 2022 codes present in Data/Firm financials.xlsx
# ------------------------------------------------------------------
# 68 sectors, 1,245 firms. `idx` is the 1-based sector index for Julia tuples/vectors.
# Sectors are 3-digit NAICS 2022, except 336 and 541, which are split along BEA Summary boundaries:
#   336 -> 3361MV (3361-3363) and 3364OT (3364-3369)
#   541 -> 5415 (5415) and 5412OP (5412-5419); 5411 (legal) has no firms in the data
# `BEA IO` is the matching code in Data/IO table.xlsx (BEA Summary Use table, 2024).
# Matching to NAICS done using BEA Data/'BEA-Industry-and-Commodity-Codes-and-NAICS-Concordance.xlsx'
#
# idx  Code    Title                                                    Firms  BEA IO
#   1  111     Crop production                                              6  111CA
#   2  112     Animal production and aquaculture                            1  111CA
#   3  115     Support activities for agriculture and forestry              5  113FF
#   4  211     Oil and gas extraction                                       4  211
#   5  212     Mining (except oil and gas)                                 21  212
#   6  213     Support activities for mining                                9  213
#   7  221     Utilities                                                    8  22
#   8  236     Construction of buildings                                    8  23
#   9  237     Heavy and civil engineering construction                     2  23
#  10  238     Specialty trade contractors                                  3  23
#  11  311     Food manufacturing                                          47  311FT
#  12  312     Beverage and tobacco product manufacturing                  29  311FT
#  13  313     Textile mills                                                3  313TT
#  14  314     Textile product mills                                        3  313TT
#  15  315     Apparel manufacturing                                        5  315AL
#  16  316     Leather and allied product manufacturing                     1  315AL
#  17  321     Wood product manufacturing                                  10  321
#  18  322     Paper manufacturing                                         14  322
#  19  323     Printing and related support activities                      3  323
#  20  324     Petroleum and coal products manufacturing                   12  324
#  21  325     Chemical manufacturing                                     199  325
#  22  326     Plastics and rubber products manufacturing                  11  326
#  23  327     Nonmetallic mineral product manufacturing                   13  327
#  24  331     Primary metal manufacturing                                 24  331
#  25  332     Fabricated metal product manufacturing                      43  332
#  26  333     Machinery manufacturing                                     92  333
#  27  334     Computer and electronic product manufacturing              228  334
#  28  335     Electrical equipment, appliance and component mfg           46  335
#  29  3361MV  Motor vehicles, bodies, trailers and parts (3361-3363)      43  3361MV
#  30  3364OT  Other transportation equipment (3364-3369)                  43  3364OT
#  31  337     Furniture and related product manufacturing                 17  337
#  32  339     Miscellaneous manufacturing                                 88  339
#  33  423     Merchant wholesalers, durable goods                         15  42
#  34  424     Merchant wholesalers, nondurable goods                      17  42
#  35  441     Motor vehicle and parts dealers                              8  441
#  36  444     Building material and garden supplies dealers                1  4A0
#  37  445     Food and beverage retailers                                  4  445
#  38  449     Furniture, electronics and appliance retailers               1  4A0
#  39  456     Health and personal care retailers                           2  4A0
#  40  457     Gasoline stations and fuel dealers                           2  4A0
#  41  458     Clothing, shoe and jewelry retailers                         3  4A0
#  42  459     Sporting goods, hobby, book and misc. retailers              5  4A0
#  43  481     Air transportation                                           2  481
#  44  484     Truck transportation                                         3  484
#  45  486     Pipeline transportation                                      5  486
#  46  488     Support activities for transportation                        2  487OS
#  47  492     Couriers and messengers                                      2  487OS
#  48  512     Motion picture and sound recording industries                1  512
#  49  513     Publishing industries (incl. software)                      23  511  (NAICS 2022 513 = BEA 511)
#  50  517     Telecommunications                                           7  513  (NAICS 2022 517 = BEA 513)
#  51  518     Computing infrastructure, data processing, hosting           9  514
#  52  519     Web search portals and other information services            5  514
#  53  522     Credit intermediation and related activities                 2  521CI
#  54  523     Securities, commodity contracts and investments              5  523
#  55  531     Real estate                                                  2  ORE
#  56  532     Rental and leasing services                                  2  532RL
#  57  533     Lessors of nonfinancial intangible assets                    1  532RL
#  58  5415    Computer systems design and related services (5415)         20  5415
#  59  5412OP  Other professional, scientific, technical (5412-5419)       28  5412OP
#  60  561     Administrative and support services                          4  561
#  61  562     Waste management and remediation services                    2  562
#  62  611     Educational services                                         4  61
#  63  621     Ambulatory health care services                              4  621
#  64  713     Amusement, gambling and recreation industries                1  713
#  65  721     Accommodation                                                1  721
#  66  722     Food services and drinking places                            5  722
#  67  811     Repair and maintenance                                       4  81*
#  68  812     Personal and laundry services                                3  81*
#
# Sectors sharing one BEA code are assumed to have the same input mix.
# Input shares from sectors with no firms will be dropped.

const SECTORS = [
    "111", "112", "115", "211", "212", "213", "221", "236", "237", "238",
    "311", "312", "313", "314", "315", "316", "321", "322", "323", "324",
    "325", "326", "327", "331", "332", "333", "334", "335", "3361MV", "3364OT",
    "337", "339", "423", "424", "441", "444", "445", "449", "456", "457",
    "458", "459", "481", "484", "486", "488", "492", "512", "513", "517",
    "518", "519", "522", "523", "531", "532", "533", "5415", "5412OP", "561",
    "562", "611", "621", "713", "721", "722", "811", "812",
]
const N_SECTORS    = length(SECTORS)                                  
const SECTOR_INDEX = Dict(code => i for (i, code) in enumerate(SECTORS)) 

# 6-digit NAICS 2022 code => model sector code (an entry of SECTORS)
function sector_code(naics::AbstractString)
    n3, n4 = naics[1:3], naics[1:4]
    if n3 == "336"                                                    
        return n4 in ("3361", "3362", "3363") ? "3361MV" : "3364OT"
    elseif n3 == "541"                                                
        n4 == "5411" && error("NAICS $naics is legal services (BEA 5411), which is not in SECTORS")
        return n4 == "5415" ? "5415" : "5412OP"
    else
        return n3
    end
end

# mapping NAICS => BEA
const SECTOR_BEA = [                                   
    "111CA", "111CA", "113FF", "211", "212", "213", "22", "23", "23", "23",
    "311FT", "311FT", "313TT", "313TT", "315AL", "315AL", "321", "322", "323", "324",
    "325", "326", "327", "331", "332", "333", "334", "335", "3361MV", "3364OT",
    "337", "339", "42", "42", "441", "4A0", "445", "4A0", "4A0", "4A0",
    "4A0", "4A0", "481", "484", "486", "487OS", "487OS", "512", "511", "513",
    "514", "514", "521CI", "523", "ORE", "532RL", "532RL", "5415", "5412OP", "561",
    "562", "61", "621", "713", "721", "722", "81", "81",
]

# ------------------------------------------------------------------
# 0.3 Qualifications and experience
# ------------------------------------------------------------------
# Same definitions set out in Model/Init_analysis.py (IPUMS ACS 2024), which writes the
# calibration files in Data/Calibration/. If you change one, change the other.
#
# idx  Qualification              IPUMS EDUCD             Starts work at age
#   1  High school                62-64                   18
#   2  Associate degree           81-83                   20
#   3  Bachelor's, STEM           101 + STEM DEGFIELD     22
#   4  Bachelor's, other          101 + other DEGFIELD    22
#   5  Master's / professional    114-115                 24
#   6  Doctorate                  116                     28
# STEM DEGFIELD codes: 13, 20, 21, 24, 25, 36, 37, 50, 51
# (environment, comm. technologies, computing, engineering + tech, biology, maths, physical sciences, nuclear/bio tech)

const QUALIFICATIONS = [
    "High school", "Associate degree", "Bachelor's, STEM",
    "Bachelor's, other", "Master's / professional", "Doctorate",
]
const N_QUALIFICATIONS = length(QUALIFICATIONS)

# Experience bands: potential experience = age - starting age (above).
const EXP_BANDS      = ["0-3", "3-6", "6-10", "10-15", "15-20", "20-25", "25-30", "30+"]
const EXP_BAND_EDGES = [0, 3, 6, 10, 15, 20, 25, 30]  # lower edge of each band (years)
const N_EXP_BANDS    = length(EXP_BANDS)

# ------------------------------------------------------------------
# 0.5 Initialise
# ------------------------------------------------------------------
lognormal_mean(rng, m, s) = m * exp(s * randn(rng) - s^2 / 2)

const DATA_DIR  = joinpath(@__DIR__, "..", "Data")
const FIRM_DATA = joinpath(DATA_DIR, "Firm financials.xlsx")
const IO_DATA   = joinpath(DATA_DIR, "IO table.xlsx")
const MIX_DATA  = joinpath(DATA_DIR, "Calibration", "qual_exp_mix_by_sector.csv")   # from Model/Init_analysis.py
const PROD_DATA = joinpath(DATA_DIR, "Calibration", "qual_productivity.csv")        # from Model/Init_analysis.py
const CELL_WAGE_DATA = joinpath(DATA_DIR, "Calibration", "wage_by_cell.csv")        # from Model/Init_analysis.py
const ELASTICITY_DATA = joinpath(DATA_DIR, "Calibration", "income_elasticity.csv")  # from Model/Consumption_analysis.py

const INIT_PRICE     = 1.0      # all prices start at 1, so quantity = revenue in the starting year
const PROD_LOGSD     = 0.5      # log-sd of worker productivity around its qualification mean (wage_residuals.csv)
const RU_SD          = 0.5      # reservation_utility = wage * Normal(1, RU_SD), floored at 0 to prevent decay from working backwards

const WEALTH_TO_WAGE = 13.0     # mean initial wealth to wage ratio
const WEALTH_LOGSD   = 1.8      # assuming a lognormal distribution around the predicted wealth from wage
const LABOUR_ELASTICITY = 0.7   # Q = tfp * (Σ productivity)^LABOUR_ELASTICITY
const UNEMPLOYMENT_RATE = 0.05  # share of all households unemployed at t = 0

# Expenditure E = CONS_0 + MPC_WAGE × wage + MPC_WEALTH × wealth  (th USD per year)
const CONS_0     = 32.374       # CONS_0 and MPC_WAGE from BLS CES 2024 (Model/Consumption_analysis.py)
const MPC_WAGE   = 0.326        
const SUBSISTENCE = CONS_0      # minimum expenditure (Γ) at t = 0 prices
const MPC_WEALTH = 0.27         # calibrated so that total consumer demand + input demand = output at t = 0
                                # literature range 0.02-0.06 per year, consumers in this model act in place of government, investment etc.

# --- 0.5.1 Load data ------------------------------------------------
load_firms(path = FIRM_DATA) = DataFrame(XLSX.readtable(path, "Results"))
firm_sector(row) = SECTOR_INDEX[sector_code(string(row["NAICS 2022"]))]

# Load data from BEA Summary Use table. Returns use(row, col), which reads one cell ("---" => 0).
# Rows: supplying commodities, plus T005 (total intermediate inputs) and T018 (total industry output).
# Columns: buying industries.
function read_io_table(path = IO_DATA)
    t      = XLSX.readxlsx(path)["Table"][:]                                                    # whole sheet as a matrix
    header = findfirst(r -> any(isequal("111CA"), t[r, 2:end]), 1:size(t, 1))                   # row holding industry codes
    col    = Dict(string(t[header, c]) => c for c in 1:size(t, 2) if !ismissing(t[header, c]))  # buying industries
    row    = Dict(string(t[r, 1]) => r for r in 1:size(t, 1) if !ismissing(t[r, 1]))            # supplying industries
    return (r, c) -> (x = t[row[r], col[c]]; x isa Real ? Float64(x) : 0.0)
end

# total sector revenue from firms in the sample
function sector_revenue(firms)
    revenue = zeros(N_SECTORS)
    for r in eachrow(firms)
        revenue[firm_sector(r)] += r["total_revenue"]
    end
    return revenue
end

# share of each sector in the revenue of all sectors with the same BEA code
bea_split(revenue) = [revenue[s] / sum(revenue[SECTOR_BEA .== SECTOR_BEA[s]]) for s in 1:N_SECTORS]

# Calculate input_coef[buyer, supplier]: units of supplier output needed for each unit of buyer output.
# total input_cost = BEA total intermediate / total output (T005 / T018) of the buyer's industry.
# input mix across sectors given by expenditure split across sectors in BEA table. Drop suppliers not in SECTORS.
# sectors that share BEA code have input shares distributed by revenue.
function load_input_coefficients(firms, use)
    split      = bea_split(sector_revenue(firms))                              # share of revenue among sectors with same BEA code
    kept_codes = unique(SECTOR_BEA)                                            # drop BEA codes without at least one sector
    coef = zeros(N_SECTORS, N_SECTORS)
    for b in 1:N_SECTORS
        buyer       = SECTOR_BEA[b]
        input_share = use("T005", buyer) / use("T018", buyer)                  # input cost as a share of revenue
        kept_inputs = sum(use(k, buyer) for k in kept_codes)                   # adds up buyer purchases from suppliers that exist
        for s in 1:N_SECTORS
            coef[b, s] = input_share * use(SECTOR_BEA[s], buyer) / kept_inputs * split[s]   # b: buyer, s: supplier i.e. how much buyer b needs from each supplier per unit of own output
        end
    end
    return coef
end

# --- 0.5.2 Firms ------------------------------------------------
# Compression for memory: 12.6m real employees rescaled at 1000 : 1.
# Each firm gets n = max(1, round(N / EMPLOYEES_PER_WORKER)) workers, 
# Revenue (and input costs), finished_goods and cash_balance are scaled by k = n / N.
# Productivity, markups, wages are unchanged.

#   total_revenue = price (1.0) * Q
#   Q             = tfp * (Σ productivity)^LABOUR_ELASTICITY
#   total_cost    = Σ wages + input_cost,  
#   input_cost    = BEA input share * Q * price
#   markup     = revenue / total_cost - 1 

const EMPLOYEES_PER_WORKER = 1000                             

scaled_headcount(N) = max(1, round(Int, N / EMPLOYEES_PER_WORKER))
firm_output(row) = scaled_headcount(row["num_employees"]) / row["num_employees"] * row["total_revenue"] / INIT_PRICE   # Q after compression

# Some sectors have higher input demand than output, leading to negative final demand from consumers. 
# E.g. Input demand / Output: 522 Credit 94x 112 Animal Production 38x 561 Admin Support 36x
# Possible cause: sector mix is different from actual economy, certain sectors overrepresented and create higher input demand for specific sectors
# Scale down input_coef by Q / input_demand for short sectors (these sectors will not have final demand from consumers).
                                   
function scale_input_coefficients!(coef, firms)
    Q = zeros(N_SECTORS)                                        # output of each sector (value at INIT_PRICE)
    for r in eachrow(firms)
        Q[firm_sector(r)] += firm_output(r)
    end
    demand = coef' * Q                                          # demand[s] = Σ_b coef[b, s] × Q[b]: input demand on s
    for s in 1:N_SECTORS
        demand[s] > Q[s] || continue                
        coef[:, s] .*= Q[s] / demand[s]
    end
    return coef
end   

# target inventory / output ratio for each sector in which demand signal is neutral
# set at median starting inventory_ratio in Data/Firm financials
function load_inventory_targets(firms)
    ratios = [Float64[] for _ in 1:N_SECTORS]
    for r in eachrow(firms)
        push!(ratios[firm_sector(r)], r["inventory_ratio"])
    end
    return [isempty(x) ? 0.0 : median(x) for x in ratios]
end

function add_firms!(model, firms::DataFrame)
    for row in eachrow(firms)
        s         = firm_sector(row)
        n_workers = scaled_headcount(row["num_employees"])
        k         = n_workers / row["num_employees"]                            # scale factor
        Q         = firm_output(row)                                            # = k * total_revenue / INIT_PRICE
        input_cost = sum(@view model.input_coef[s, :]) * Q * INIT_PRICE

        f = add_agent!(EconomicAgent ∘ Firm, model;
            sector    = s,
            tfp       = 0.0,                                                    # set below, once workers are known
            inventory = k * row["finished_goods"],
            workers   = Int[],                                                  # filled by add_workers!
            inputs    = zeros(Int, N_SECTORS),                                  # supplier ids set in assign_suppliers!
            output    = Q,
            markup    = 0.0,                                                    # set below, once wages are known
            price     = INIT_PRICE,
            liquidity = k * row["cash_balance"],
            input_cost = input_cost)
        add_workers!(model, f, n_workers)

        total_cost = wage_bill(f, model) + input_cost                           
        f.markup   = Q * INIT_PRICE / total_cost - 1                            # firms in this model have lower total costs and higher markup than FAME (no interest, capital expenses etc.)
        f.tfp      = Q / labour_input(f, model)                                 # tfp calibrated such that output matches revenue at t = 0
    end
    n_negative = count(f -> variantof(f) === Firm && f.markup < 0, allagents(model))
    @info "Firms added" n_firms = nrow(firms) negative_markup = n_negative
    return model
end

# Seeds firms with one supplier for every sector the firm buys inputs from; 0 = no input from that sector.
# Supplier drawn with probability ∝ output, excluding the firm itself
function assign_suppliers!(model)
    rng  = abmrng(model)
    size = Dict(f.id => f.output for f in allagents(model) if variantof(f) === Firm)
    for f in allagents(model)
        variantof(f) === Firm || continue
        for s in 1:N_SECTORS
            model.input_coef[f.sector, s] > 0 || continue
            ids = filter(!=(f.id), get(model.firms_in_sector, s, Int[]))    # firms do not buy from themselves
            isempty(ids) && continue
            f.inputs[s] = sample(rng, ids, Weights([size[j] for j in ids]))
        end
    end
    return model
end

# --- 0.5.3 Households ------------------------------------------------
# Each worker's (qualification, experience band) is drawn from the joint distribution of qual and exp 
# given each sector in IPUMS ACS 2024 (Model/Init_analysis.py).

# mix[s][q, e]: share of sector s workers with qualification q and experience band e
function load_qual_exp_mix(path = MIX_DATA)
    df  = CSV.read(path, DataFrame; types = Dict(:sector => String))
    mix = [zeros(N_QUALIFICATIONS, N_EXP_BANDS) for _ in 1:N_SECTORS]
    for r in eachrow(df)
        mix[SECTOR_INDEX[r.sector]][r.qualification, r.exp_band] = r.share
    end
    return mix
end

# cell_wage[s, q, e]: predicted wage (th USD) of a sector s worker with qualification q and experience band e
function load_wage_by_cell(path = CELL_WAGE_DATA)
    df    = CSV.read(path, DataFrame; types = Dict(:sector => String))
    wages = zeros(N_SECTORS, N_QUALIFICATIONS, N_EXP_BANDS)
    for r in eachrow(df)
        wages[SECTOR_INDEX[r.sector], r.qualification, r.exp_band] = r.wage / 1000   # USD -> th USD
    end
    return wages
end

# mean productivity of each qualification (th USD): wage at the sample-average industry and experience mix
load_qual_productivity(path = PROD_DATA) = CSV.read(path, DataFrame).mean_productivity ./ 1000   # USD -> th USD

const QUAL_EXP_CELLS = vec(CartesianIndices((N_QUALIFICATIONS, N_EXP_BANDS)))  # cell k => (q, e), same order as vec(mix)

# draw one (qualification, experience band) pair with probability equal to its share
function draw_qual_exp(rng, mix::Matrix{Float64})
    cell = sample(rng, QUAL_EXP_CELLS, Weights(vec(mix)))
    return cell[1], cell[2]
end

# add household with qual and exp given the distribution from their sector
# unemployed households: will draw some 'firm' proportional to the num. of employees at the firm,
# predict qual, exp, wealth, and reservation_utility based on that distribution,
# but employer and wage will be set to 0.
function add_household!(model, sector, employer)
    rng            = abmrng(model)
    qual, exp_band = draw_qual_exp(rng, model.qual_exp_mix[sector])
    pred_wage      = model.cell_wage[sector, qual, exp_band]                    # predicted wage (th USD)
    wage           = employer == 0 ? 0.0 : pred_wage
    return add_agent!(EconomicAgent ∘ Household, model;
        qualification       = qual,
        experience          = exp_band,
        productivity        = lognormal_mean(rng, model.qual_productivity[qual], PROD_LOGSD),
        employer            = employer,
        reservation_utility = pred_wage * max(0.0, 1 + RU_SD * randn(rng)),    
        wealth              = WEALTH_TO_WAGE * pred_wage * lognormal_mean(rng, 1, WEALTH_LOGSD),
        wage                = wage,
        income              = wage,                                             # t = 0: no bonuses or dividends yet
        expenditure         = 0.0,                                              # set in add_expenditure!
        consumption         = Consumption[])                                    # set in add_consumption!
end

# employed households, called in add_firms!
function add_workers!(model, f, n)
    for _ in 1:n
        hh = add_household!(model, f.sector, f.id)
        push!(f.workers, hh.id)                                                 # push worker id to f.workers
    end
    return f
end

# unemployed households, called in initialise!
function add_unemployed!(model)
    rng        = abmrng(model)
    firms      = [f for f in allagents(model) if variantof(f) === Firm]
    headcount  = Weights([length(f.workers) for f in firms])
    n          = round(Int, UNEMPLOYMENT_RATE / (1 - UNEMPLOYMENT_RATE) * sum(headcount))
    for _ in 1:n
        add_household!(model, sample(rng, firms, headcount).sector, 0)
    end
    @info "Unemployed added" n
    return model
end

# --- 0.5.4 Expenditures ----------------------------------------------------
expenditure_rule(hh) = max(0.0, CONS_0 + MPC_WAGE * hh.wage + MPC_WEALTH * hh.wealth)

# also reused in Phase 3
function add_expenditure!(model)
    for hh in allagents(model)
        variantof(hh) === Household || continue
        hh.expenditure = expenditure_rule(hh)
    end
    return model
end

# Fit household consumption baskets to firm excess output (BLS CES consumption shares do not clear the market).
# 1. Aggregate expenditure shares a_i come from residual demand in each sector after subtracting input demand from output.
#    Sectors which have higher input demand than output have had their input coefficients scaled down such that residual demand = 0. 
# 2. Each household's expenditure follows the Linear Expenditure System p_i q_i = p_i γ_i + β_i (E - Σ_j p_j γ_j). 
# 3. Chance of household consuming from a firm in a sector which household has positive demand in is proportional to the residual 
#    demand at the firm (probability for firms with negative final demand = 0).
function final_demand(model)
    F = Dict{Int, Float64}()                                                    # firm id: residual final demand
    for b in allagents(model)                                                   
        variantof(b) === Firm || continue
        sales_b = output(b, model) * b.price                                    # for every buyer firm, 
        F[b.id] = get(F, b.id, 0.0) + sales_b                                   # adds firm output to their own balance
        for s in 1:N_SECTORS                            
            j = b.inputs[s]                                                     # iterate across firm's suppliers
            j == 0 && continue
            F[j] = get(F, j, 0.0) - model.input_coef[b.sector, s] * sales_b     # subtract inputs from other firms' balances
        end
    end
    return F
end

# income elasticity of each sector calibrated from BLS CES 2024 data (Model/Consumption_analysis.py). 
# Sectors not covered by CES share one elasticity chosen such that Σβ = 1. 
function load_income_elasticities(path = ELASTICITY_DATA)
    df = CSV.read(path, DataFrame; types = Dict(:sector => String))
    η, mapped = ones(N_SECTORS), falses(N_SECTORS)
    for r in eachrow(df)
        η[SECTOR_INDEX[r.sector]]      = r.income_elasticity
        mapped[SECTOR_INDEX[r.sector]] = r.mapped
    end
    return η, mapped
end

# Derived from the LES demand equation:
# β_i = η_i × a_i                   (a_i: sector share of residual final demand, Σβ = 1)
# γ_i = a_i × Ē - β_i × (Ē - Γ)     (Ē: mean household budget, Γ = Σγ = SUBSISTENCE)
function exp_parameters(a, η, mapped, Ē, Γ)
    η = copy(η)
    η_unmapped = (1 - sum(η[mapped] .* a[mapped])) / sum(a[.!mapped])
    η_unmapped > 0 || error("mapped sectors already take all of Σβ = 1; rescale all elasticities instead")
    η[.!mapped] .= η_unmapped
    β = η .* a
    γ = a .* Ē .- β .* (Ē - Γ)
    return β, γ, η_unmapped
end

# Set household quantities from the LES given initial prices of firms in its basket:
# 1. LES quantity q_i = γ_i + β_i (E - Σ_j p_j γ_j) / p_i for every good
# 2. goods with q_i < 0 are set to 0; the extra spending this needs is taken from all other goods
#    by scaling their quantities down equally, so spending stays at E
function exp_basket!(hh, model, E)
    γ, β        = model.exp_gamma, model.exp_beta
    subsistence = sum(model[c.firm].price * γ[c.sector] for c in hh.consumption)
    positive    = 0.0                                               # spending on goods with q_i > 0
    for (k, c) in enumerate(hh.consumption)
        p = model[c.firm].price
        q = γ[c.sector] + β[c.sector] * (E - subsistence) / p
        hh.consumption[k] = Consumption(c.sector, c.firm, q, p)
        q > 0 && (positive += p * q)
    end
    scale = positive > 0 ? E / positive : 0.0                       # = 1 when no quantity is negative
    for (k, c) in enumerate(hh.consumption)
        q = c.quantity > 0 ? c.quantity * scale : 0.0
        hh.consumption[k] = Consumption(c.sector, c.firm, q, c.price)
    end
    return hh
end

function add_consumption!(model)
    rng     = abmrng(model)
    F       = final_demand(model)                                                   # due to random assignment of suppliers, select firms can get higher input demand than output
    sectors = model.firms_in_sector
    F_pos   = Dict(j => max(x, 0.0) for (j, x) in F)                                # firms where input demand > output have household demand set to 0 instead of negative
    sec_F   = Dict(s => max(sum(F[j] for j in ids), 0.0) for (s, ids) in sectors)   # residual demand for sector s
    total   = sum(values(sec_F))
    tol     = 1e-9 * total                                                          # round sectors with near-zero quantities to 0
    demand_sectors = sort([s for s in keys(sec_F) if sec_F[s] > tol])               # sectors with positive net demand from households
    weights = Dict(s => Weights([F_pos[j] for j in sectors[s]]) for s in demand_sectors)

    # LES parameters from aggregate shares a, mean budget and income elasticities
    households = [hh for hh in allagents(model) if variantof(hh) === Household]
    a = zeros(N_SECTORS)
    for s in demand_sectors
        a[s] = sec_F[s]
    end
    a ./= sum(a)
    Ē = sum(hh.expenditure for hh in households) / length(households)
    η, mapped = load_income_elasticities()
    β, γ, η_unmapped = exp_parameters(a, η, mapped, Ē, SUBSISTENCE)
    model.exp_beta  .= β
    model.exp_gamma .= γ

    for hh in households
        for s in demand_sectors
            j = sample(rng, sectors[s], weights[s])                                 # probability of firm ∝ its residual demand
            push!(hh.consumption, Consumption(s, j, 0.0, model[j].price))           # quantity set by exp_basket!
        end
        exp_basket!(hh, model, hh.expenditure)
    end

    # Checks: every household spends exactly its budget; household spending by sector vs residual demand
    # (they differ only because negative quantities were set to 0 and other goods scaled down)
    spending  = sum(hh.expenditure for hh in households)
    by_sector = zeros(N_SECTORS)
    for hh in households, c in hh.consumption
        by_sector[c.sector] += c.quantity * c.price
    end
    @assert isapprox(sum(by_sector), spending; rtol = 1e-9) "household spending does not equal total budget"
    zeroed   = count(hh -> any(c -> c.quantity == 0, hh.consumption), households)
    shifted  = sum(abs, by_sector .- a .* spending) / 2 / spending        # share of spending moved between sectors
    sec_gap  = maximum(abs(by_sector[s] / (a[s] * spending) - 1) for s in demand_sectors)

    @info "Consumption baskets" household_spending = round(spending) final_demand = round(total) spending_to_final_demand = round(spending / total, digits = 2) firms_with_negative_F = count(<(0), values(F)) sectors_without_household_demand = N_SECTORS - length(demand_sectors)
    @info "LES calibration" mean_budget = round(Ē, digits = 1) budget_to_subsistence = round(Ē / SUBSISTENCE, digits = 2) max_elasticity_with_γ_nonnegative = round(Ē / (Ē - SUBSISTENCE), digits = 3) negative_elasticities = count(<(0), η[mapped]) unmapped_elasticity = round(η_unmapped, digits = 3) sectors_with_negative_γ = count(<(0), γ) households_with_zeroed_goods = zeroed spending_shifted_between_sectors = round(shifted, digits = 3) largest_sector_demand_gap = round(sec_gap, digits = 3)
    return model
end

# --- 0.5.5 Combined ----------------------------------------------------
function initialise(; seed = 1)
    use   = read_io_table()
    firms = load_firms()
    model = StandardABM(
        EconomicAgent;
        properties  = Dict(
            :vacancies   => Vacancy[],
            :firms_in_sector  => Dict{Int, Vector{Int}}(),         # sector => firm ids, filled once firms are added
            :inventory_target => load_inventory_targets(firms),     # [sector], target inventory / output
            :exp_beta    => zeros(N_SECTORS),                   # [sector], LES share of budget above subsistence, set in add_consumption!
            :exp_gamma   => zeros(N_SECTORS),                   # [sector], LES subsistence quantity, set in add_consumption!
            :cell_wage   => load_wage_by_cell(),                # [sector, qualification, experience band], th USD
            :input_coef  => scale_input_coefficients!(load_input_coefficients(firms, use), firms),     # [buyer, supplier], per unit of output
            :qual_exp_mix => load_qual_exp_mix(),               # [sector][qualification, experience band] shares
            :qual_productivity => load_qual_productivity(),     # th USD, mean productivity by qualification
        ),
        model_step! = model_step!,
        rng         = Xoshiro(seed),
    )
    add_firms!(model, firms)
    merge!(model.firms_in_sector, firms_by_sector(model))      # firms do not enter or exit, so build once
    assign_suppliers!(model)
    add_unemployed!(model)
    add_expenditure!(model)
    add_consumption!(model)
    return model
end


# ------------------------------------------------------------------
# 1. Job-matching
# ------------------------------------------------------------------
const N_APPLICATIONS   = 10                     # maximum applications per seeker per period
const FIT_MEAN         = 1.0                    # idiosyncratic fit ~ N(1, 0.5)
const FIT_SD           = 0.5
const RU_DECAY         = 0.8                    # decay of reservation utility per unemployed period

function job_matching!(model)
    index  = index_vacancies(model)             # 1.1  job seekers find eligible vacancies
    gather_applications!(model, index)          # 1.2  seekers, in random order, apply to vacancies maximising expected net utility
    offers = screen_applicants(model)           # 1.3  each vacancy hires proportional to productivity
    finalise_hires!(model, offers)              # 1.4  workers accept best offer; update wage/employer/RU
    decay_unemployed!(model)                    # 1.5  RU *= 0.8 for those still unemployed
    return model
end

function index_vacancies(model)
    index = Dict{Tuple{Int,Int}, Vector{Int}}()
    for (v, vac) in enumerate(model.vacancies)
        push!(get!(index, (vac.qualification, vac.experience), Int[]), v)
    end
    return index
end

# Job-seekers balance higher wage and lower offer probability in more competitive positions
# V = u₁p₁ + u₂p₂(1-p₁) + u₃p₃(1-p₁)(1-p₂) + ...
function expected_value(u, p, set)
    V, p_no_better = 0.0, 1.0                                           # p_no_better: all better jobs rejected
    for k in sort(set, by = i -> u[i], rev = true)
        V           += p_no_better * p[k] * u[k]
        p_no_better *= 1 - p[k]
    end
    return V
end

# Keep adding the application that increases V the most, up to max_apps applications
function choose_applications(u, p, max_apps)
    chosen, V = Int[], 0.0
    remaining = collect(eachindex(u))
    while length(chosen) < max_apps && !isempty(remaining)
        gains = [expected_value(u, p, [chosen; k]) - V for k in remaining]
        best  = argmax(gains)
        V += gains[best]
        push!(chosen, remaining[best])
        deleteat!(remaining, best)
    end
    return chosen                                                       # vector of open vacancies in order of expected utility from applying
end

function gather_applications!(model, index)
    rng = abmrng(model)
    for vac in model.vacancies                                          # drop last period's applicants
        empty!(vac.applicants)
        vac.pool = 0.0
    end
    seekers = [a for a in allagents(model) if variantof(a) === Household && a.employer == 0]
    for a in shuffle(rng, seekers)                                      # put job-seekers in random order
        candidates = get(index, (a.qualification, a.experience), Int[]) # looks up index from index_vacancies to match positions to jobseeker
        isempty(candidates) && continue

        u = [model.vacancies[v].wage * (FIT_MEAN + FIT_SD * randn(rng)) for v in candidates]  # score = wage * fit, fit ~ N(1, 0.5)
        keep       = u .> a.reservation_utility                         # only keep vacancies above reservation
        candidates = candidates[keep]
        u          = u[keep]
        isempty(candidates) && continue

        p = [a.productivity / (model.vacancies[v].pool + a.productivity) for v in candidates]  # estimate offer probability
        for k in choose_applications(u, p, N_APPLICATIONS)
            vac = model.vacancies[candidates[k]]
            vac.applicants[a.id] = u[k]
            vac.pool += a.productivity                                  
        end
    end
    return model
end

function screen_applicants(model)
    rng = abmrng(model)
    offers = Dict{Int, Tuple{Int,Float64}}()                            # worker id => (vacancy index, score)
    for (v, vac) in enumerate(model.vacancies)
        isempty(vac.applicants) && continue
        ids     = collect(keys(vac.applicants))
        weights = Weights([model[id].productivity for id in ids])
        winner  = sample(rng, ids, weights)                             # hire proportional to productivity
        score   = vac.applicants[winner]
        if !haskey(offers, winner) || score > offers[winner][2]         # a worker keeps their best offer
            offers[winner] = (v, score)
        end
    end
    return offers
end

function finalise_hires!(model, offers)
    filled = Int[]
    for (worker, (v, score)) in offers
        hh = model[worker]
        vac = model.vacancies[v]
        hh.employer            = vac.firm               # update worker employer
        hh.wage                = vac.wage               # update worker wage
        hh.reservation_utility = score                  # accepted wage * fit is the new benchmark
        push!(model[vac.firm].workers, worker)
        push!(filled, v)
    end
    deleteat!(model.vacancies, sort!(filled))           # delete filled vacancies
    return model
end

function decay_unemployed!(model)
    for a in allagents(model)
        (variantof(a) === Household && a.employer == 0) || continue
        a.reservation_utility *= RU_DECAY               # still unemployed => lower the bar next period
    end
    return model
end

# ------------------------------------------------------------------
# 2. Production
# ------------------------------------------------------------------
const PRICE_CUT_ELASTICITY  = 0.1    # surplus: inventory 10% of output above target => price falls 1%
const PRICE_RISE_ELASTICITY = 0.05   # backlog: inventory 10% of output below target => price rises 0.5%
const SWITCH_SENSITIVITY    = 1.0    # probability of switching = SWITCH_SENSITIVITY × fraction by which rival is cheaper

labour_input(f, model) = isempty(f.workers) ? 0.0 : sum(model[w].productivity for w in f.workers)^LABOUR_ELASTICITY
output(f, model)    = f.tfp * labour_input(f, model)                                            # Q = tfp * (Σ productivity)^0.7
wage_bill(f, model) = isempty(f.workers) ? 0.0 : sum(model[w].wage         for w in f.workers)  # sum of wages in a firm

# price-driven switching, used by firms for inputs (Phase 2) and households for consumption (Phase 3):
# draw one rival in the same sector; if it is cheaper, switch with probability SWITCH_SENSITIVITY × (p_old - p_new) / p_old
function switch_firm(rng, model, current, rivals; exclude = 0)
    rival = rand(rng, rivals)
    rival == exclude && return current                              # firms never buy from themselves
    p_old, p_new = model[current].price, model[rival].price
    if p_new < p_old && rand(rng) < SWITCH_SENSITIVITY * (p_old - p_new) / p_old
        return rival
    end
    return current
end

function production!(model)
    switch_suppliers!(model)                 # 2.1  compare last period's prices, maybe switch supplier
    stock = opening_inventory(model)         # 2.2  end-of-last-period inventory = demand signal
    buy_inputs!(model)                       # 2.3  produce; buy inputs at last period's prices
    set_prices!(model, stock)                # 2.4  markup over unit cost, adjusted for inventory gap; output joins inventory
    return model
end

function switch_suppliers!(model)
    rng = abmrng(model)
    for f in allagents(model)
        variantof(f) === Firm || continue
        for (s, j) in enumerate(f.inputs)
            j == 0 && continue                                      # no inputs needed from sector s
            f.inputs[s] = switch_firm(rng, model, j, model.firms_in_sector[s]; exclude = f.id)
        end
    end
    return model
end

opening_inventory(model) = Dict(f.id => f.inventory for f in allagents(model) if variantof(f) === Firm)

# Firm requires input quantity per unit of own output as given by sector-level input_coef (scaled down)
# paid at the supplier's last-period price (prices this period only updated after unit costs known)
function buy_inputs!(model)
    for f in allagents(model)
        variantof(f) === Firm || continue
        f.output     = output(f, model)
        f.input_cost = 0.0
        for (s, j) in enumerate(f.inputs)
            j == 0 && continue
            supplier = model[j]
            q        = model.input_coef[f.sector, s] * f.output     # units of input from sector s
            cost     = q * supplier.price
            f.liquidity        -= cost                              # buyer pays
            supplier.liquidity += cost                              # supplier receives
            supplier.inventory -= q                                 # supplier delivers
            f.input_cost       += cost
        end
    end
    return model
end

# Set price  = (1 + markup) × unit_cost
# Adjust prices based on (opening_inventory - target) / output: if surplus, price falls, if backlog, price rises
function set_prices!(model, stock)
    for f in allagents(model)
        variantof(f) === Firm || continue
        f.output > 0 || continue                                    # no workers => no output => hold price & markup
        unit_cost  = (wage_bill(f, model) + f.input_cost) / f.output
        target     = model.inventory_target[f.sector] * f.output  
        signal     = (stock[f.id] - target) / f.output
        elasticity = signal > 0 ? PRICE_CUT_ELASTICITY : PRICE_RISE_ELASTICITY
        f.price    = (1 + f.markup) * unit_cost * (1 - elasticity * signal)
        f.markup   = f.price / unit_cost - 1                        # update price and markup
        f.inventory += f.output                                     # this period's output joins inventory
    end
    return model
end

# ------------------------------------------------------------------
# 3. Consumption
# ------------------------------------------------------------------
function consumption!(model)
    for hh in allagents(model)
        variantof(hh) === Household || continue
        switch_shops!(hh, model)                # 3.1  price-driven switching of firm in each sector
        update_quantities!(hh, model)           # 3.2-3.3  LES demand: price and budget effects on quantity demanded
        buy_goods!(hh, model)                   # 3.4  transact goods and money
    end
    return model
end

# same switching rule and sensitivity as firms' suppliers (Phase 2), except with this period prices
function switch_shops!(hh, model)
    rng = abmrng(model)
    for (k, c) in enumerate(hh.consumption)
        firm = switch_firm(rng, model, c.firm, model.firms_in_sector[c.sector])
        hh.consumption[k] = Consumption(c.sector, firm, c.quantity, c.price)
    end
    return hh
end

# LES demand at this period's prices and budget: price changes in any good move every quantity
# through the cost of subsistence (purchasing power), budget changes through β
function update_quantities!(hh, model)
    E = expenditure_rule(hh)                                    # expenditure budget this period
    exp_basket!(hh, model, E)
    hh.expenditure = E
    return hh
end

function buy_goods!(hh, model)
    for c in hh.consumption
        spend = c.quantity * c.price
        f = model[c.firm]
        hh.wealth   -= spend                                    # household pays
        f.liquidity += spend                                    # firm receives
        f.inventory -= c.quantity                               # firm delivers
    end
    return hh
end

function firms_by_sector(model)
    index = Dict{Int, Vector{Int}}()
    for f in allagents(model)
        variantof(f) === Firm || continue
        push!(get!(index, f.sector, Int[]), f.id)
    end
    return index
end

# ------------------------------------------------------------------
# 4. Income   (interim placeholder: wages, then ALL positive profit paid out as dividends)
# ------------------------------------------------------------------
# Profit this period = change in a firm's liquidity since the start of the period
#   = sales to households + sales of inputs - input costs - wages   (no other cash flows exist yet)
# Firms with positive profit pay all of it out; loss-making firms pay nothing and keep the loss.
# The pooled dividends go to households in proportion to their wealth (negative wealth counts as 0).

opening_liquidity(model) = Dict(f.id => f.liquidity for f in allagents(model) if variantof(f) === Firm)

function wage_transfer!(model)
    for f in allagents(model)
        variantof(f) === Firm || continue
        f.liquidity -= wage_bill(f, model)          # firm pays its wage bill
    end
    for hh in allagents(model)
        variantof(hh) === Household || continue
        hh.wealth += hh.wage                         # each worker receives their wage
    end
    return model
end

function pay_dividends!(model, cash)
    pool = 0.0                                                  # total dividends paid out this period
    for f in allagents(model)
        variantof(f) === Firm || continue
        profit = f.liquidity - cash[f.id]                       # cash = liquidity at the start of the period
        profit > 0 || continue
        f.liquidity -= profit
        pool        += profit
    end
    households = [h for h in allagents(model) if variantof(h) === Household]
    total      = sum(max(h.wealth, 0.0) for h in households)   # dividend weights
    for h in households
        dividend  = total > 0 ? pool * max(h.wealth, 0.0) / total : 0.0
        h.wealth += dividend
        h.income  = h.wage + dividend                           # income = wage + dividends (no bonuses yet)
    end
    return model
end

# 5. Wages and Vacancies   (Phase-5 TODO)

# ------------------------------------------------------------------
# Build model
# ------------------------------------------------------------------
function model_step!(model)
    cash = opening_liquidity(model)   # start-of-period liquidity, used to measure profit in Phase 4
    job_matching!(model)          # Phase 1
    production!(model)            # Phase 2
    consumption!(model)           # Phase 3
    wage_transfer!(model)         # Phase 4 (interim): wages
    pay_dividends!(model, cash)   # Phase 4 (interim): all positive profit paid out as dividends
    # wage_and_vacancy_setting!(model)  # Phase 5 (later)
    return model
end

# ------------------------------------------------------------------
# Iterate: run n periods and collect aggregate (macro) data
# ------------------------------------------------------------------
# One row per period (period 0 = state after initialisation). Columns:
#   wealth_total, liquidity_total, expenditure_total   money stocks and household spending (th USD)
#   wages_total, dividends_total                       household income this period (th USD)
#   output, units_sold, inventory_change               units; units_sold = household purchases + input sales,
#                                                      inventory_change = output - units_sold
#   wealth_p10, wealth_median, wealth_p90              household wealth distribution
#   price_p10, price_median, price_mean, price_p90     posted prices across firms (unweighted)
#   price_index                                        output-weighted mean price
#   inv_ratio_min, inv_ratio_max                       lowest / highest inventory ÷ output across firms (backlog / surplus)
# Note: at period 0, units_sold is the seeded (planned) baskets and inputs; no trade has happened yet.

# units each firm buys from its suppliers this period (same rule as buy_inputs!)
function input_units(f, model)
    units = 0.0
    for (s, j) in enumerate(f.inputs)
        j == 0 && continue
        units += model.input_coef[f.sector, s] * f.output
    end
    return units
end

# Sanity check after each period: every price finite and > 0, every wealth / liquidity finite.
# On failure, prints the first offending firm / household and returns false so the run stops early
# (quantile() cannot handle NaN, and LES quantities are meaningless at p <= 0).
function state_ok(model, t)
    bad_firms = [f.id for f in allagents(model) if variantof(f) === Firm &&
                 !(isfinite(f.price) && f.price > 0 && isfinite(f.liquidity))]
    bad_hh    = [h.id for h in allagents(model) if variantof(h) === Household && !isfinite(h.wealth)]
    isempty(bad_firms) && isempty(bad_hh) && return true
    @warn "Run stopped: non-finite or non-positive values" period = t n_bad_firms = length(bad_firms) n_bad_households = length(bad_hh)
    if !isempty(bad_firms)
        f = model[bad_firms[1]]
        @warn "First bad firm" id = f.id sector = SECTORS[f.sector] workers = length(f.workers) inventory_gap = f.inventory / f.output - model.inventory_target[f.sector] price = f.price liquidity = f.liquidity
    end
    if !isempty(bad_hh)
        h = model[bad_hh[1]]
        @warn "First bad household" id = h.id wealth = h.wealth wage = h.wage expenditure = h.expenditure
    end
    return false
end

# one pass over all agents; returns a NamedTuple = one row of the output table
function period_stats(model, t)
    wealth, prices, outputs, inv_ratio = Float64[], Float64[], Float64[], Float64[]
    expenditure, hh_units, liquidity, input_sold = 0.0, 0.0, 0.0, 0.0
    wages, dividends = 0.0, 0.0
    for a in allagents(model)
        if variantof(a) === Household
            push!(wealth, a.wealth)
            wages     += a.wage
            dividends += a.income - a.wage                      # income = wage + dividends
            for c in a.consumption
                expenditure += c.quantity * c.price
                hh_units    += c.quantity
            end
        else                                                    # Firm
            push!(prices, a.price)
            push!(outputs, a.output)
            a.output > 0 && push!(inv_ratio, a.inventory / a.output)
            liquidity  += a.liquidity
            input_sold += input_units(a, model)
        end
    end
    total_output = sum(outputs)
    units_sold   = hh_units + input_sold
    return (
        period            = t,
        wealth_total      = sum(wealth),
        liquidity_total   = liquidity,
        expenditure_total = expenditure,
        wages_total       = wages,
        dividends_total   = dividends,
        output            = total_output,
        units_sold        = units_sold,
        inventory_change  = total_output - units_sold,
        wealth_p10        = quantile(wealth, 0.1),
        wealth_median     = median(wealth),
        wealth_p90        = quantile(wealth, 0.9),
        price_p10         = quantile(prices, 0.1),
        price_median      = median(prices),
        price_mean        = mean(prices),
        price_p90         = quantile(prices, 0.9),
        price_index       = sum(prices .* outputs) / total_output,
        inv_ratio_min     = minimum(inv_ratio),
        inv_ratio_max     = maximum(inv_ratio),
    )
end

function run_model(; n_periods = 25, save = true,
                     path = joinpath(@__DIR__, "..", "Output", "aggregates.csv"), kwargs...)
    model = initialise(; kwargs...)
    rows  = [period_stats(model, 0)]                    # period 0 = initial state
    for t in 1:n_periods
        step!(model)                                    # runs model_step! once (all implemented phases)
        state_ok(model, t) || break                     # stop at the first period with NaN / Inf / price <= 0
        push!(rows, period_stats(model, t))
    end
    data = DataFrame(rows)                              # Vector of NamedTuples => one column per field
    if save
        mkpath(dirname(path))
        CSV.write(path, data)
        @info "Saved aggregates" path periods_run = nrow(data) - 1
    end
    return model, data
end

# Run as a script (`julia ABM2.jl`); does nothing when include()-ed into the REPL
if abspath(PROGRAM_FILE) == @__FILE__
    run_model()
end
