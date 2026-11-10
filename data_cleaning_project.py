"""
Data Cleaning & Visualization Project
Messy retail sales data -> cleaned dataset -> dashboard + findings.
Run: python data_cleaning_project.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import seaborn as sns

OUT = "/mnt/user-data/outputs"
rng = np.random.default_rng(42)

# ------------------------------------------------------------------
# 1. GENERATE A RAW, MESSY DATASET
# ------------------------------------------------------------------
def make_raw(n=1500):
    regions = ["North", "South", "East", "West"]
    cats = {"Electronics": (300, 120), "Clothing": (60, 25),
            "Home": (120, 45), "Sports": (90, 35)}
    dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    rows = []
    for i in range(n):
        cat = rng.choice(list(cats))
        mu, sd = cats[cat]
        qty = int(rng.integers(1, 6))
        price = max(5, rng.normal(mu, sd))
        d = rng.choice(dates)
        # seasonality: more sales in Nov/Dec
        if pd.Timestamp(d).month in (11, 12) and rng.random() < 0.5:
            d = rng.choice(dates[(dates.month >= 11)])
        rows.append({
            "order_id": 1000 + i,
            "order_date": pd.Timestamp(d),
            "region": rng.choice(regions),
            "category": cat,
            "quantity": qty,
            "unit_price": round(price, 2),
            "discount_pct": rng.choice([0, 5, 10, 15, 20], p=[.5, .2, .15, .1, .05]),
            "customer_age": int(np.clip(rng.normal(38, 12), 18, 80)),
            "rating": int(rng.choice([1, 2, 3, 4, 5], p=[.05, .08, .17, .35, .35])),
        })
    df = pd.DataFrame(rows)

    # --- inject problems ---
    df["order_date"] = df["order_date"].dt.strftime("%Y-%m-%d")
    bad = rng.choice(df.index, 120, replace=False)
    df.loc[bad, "order_date"] = pd.to_datetime(df.loc[bad, "order_date"]).dt.strftime("%d/%m/%Y")
    df["region"] = df["region"].astype(object)
    for idx in rng.choice(df.index, 200, replace=False):
        df.at[idx, "region"] = rng.choice([df.at[idx, "region"].lower(),
                                           df.at[idx, "region"].upper(),
                                           " " + df.at[idx, "region"] + " "])
    for idx in rng.choice(df.index, 100, replace=False):
        df.at[idx, "category"] = rng.choice(["electronics", "ELECTRONICS", "Clothes", "home "])
    df["unit_price"] = df["unit_price"].astype(object)
    for idx in rng.choice(df.index, 60, replace=False):
        df.at[idx, "unit_price"] = f"${df.at[idx, 'unit_price']}"
    for col, k in [("customer_age", 130), ("rating", 90), ("region", 40),
                   ("unit_price", 50), ("discount_pct", 30)]:
        df.loc[rng.choice(df.index, k, replace=False), col] = np.nan
    # outliers / impossible values
    df.loc[rng.choice(df.index, 8, replace=False), "unit_price"] = 9999.99
    df.loc[rng.choice(df.index, 5, replace=False), "customer_age"] = 250
    df.loc[rng.choice(df.index, 4, replace=False), "quantity"] = -3
    # exact duplicates
    df = pd.concat([df, df.sample(75, random_state=1)], ignore_index=True)
    return df.sample(frac=1, random_state=7).reset_index(drop=True)


raw = make_raw()
raw.to_csv(f"{OUT}/raw_sales.csv", index=False)

# ------------------------------------------------------------------
# 2. INSPECT
# ------------------------------------------------------------------
report = {"rows_raw": len(raw), "missing_raw": int(raw.isna().sum().sum())}
print("RAW SHAPE:", raw.shape)
print(raw.isna().sum(), "\n")

df = raw.copy()

# ------------------------------------------------------------------
# 3. CLEAN
# ------------------------------------------------------------------
# 3a. duplicates
report["duplicates_removed"] = int(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)

# 3b. types & text standardization
df["unit_price"] = pd.to_numeric(
    df["unit_price"].astype(str).str.replace("$", "", regex=False), errors="coerce")
_d = df["order_date"].astype(str)
_iso = pd.to_datetime(_d.where(~_d.str.contains("/")), format="%Y-%m-%d", errors="coerce")
_dmy = pd.to_datetime(_d.where(_d.str.contains("/")), format="%d/%m/%Y", errors="coerce")
df["order_date"] = _iso.fillna(_dmy)

for c in ["region", "category"]:
    df[c] = df[c].astype("string").str.strip().str.title()
df["category"] = df["category"].replace({"Clothes": "Clothing"})

# 3c. invalid values -> NaN (domain rules)
df.loc[df["quantity"] <= 0, "quantity"] = np.nan
df.loc[~df["customer_age"].between(16, 100), "customer_age"] = np.nan
df.loc[~df["rating"].between(1, 5), "rating"] = np.nan

# 3d. outliers in price: IQR rule within each category
def iqr_mask(s):
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    return (s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)

out_mask = df.groupby("category")["unit_price"].transform(iqr_mask).fillna(False)
report["price_outliers"] = int(out_mask.sum())
df.loc[out_mask, "unit_price"] = np.nan  # treat as bad entries, impute below

# 3e. missing values
report["missing_before_impute"] = int(df.isna().sum().sum())
df["price_was_missing"] = df["unit_price"].isna()          # keep flag
df["unit_price"] = df["unit_price"].fillna(df.groupby("category")["unit_price"].transform("median"))
df["customer_age"] = df["customer_age"].fillna(df["customer_age"].median())
df["quantity"] = df["quantity"].fillna(df["quantity"].median())
df["discount_pct"] = df["discount_pct"].fillna(0)
df["region"] = df["region"].fillna("Unknown")
df["rating"] = df["rating"]  # leave NaN: imputing ratings would bias satisfaction
df = df.dropna(subset=["order_date"])

# 3f. feature engineering
df["revenue"] = df["quantity"] * df["unit_price"] * (1 - df["discount_pct"] / 100)
df["month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
df["age_group"] = pd.cut(df["customer_age"], [0, 25, 35, 45, 55, 100],
                         labels=["<25", "25-34", "35-44", "45-54", "55+"])

report["rows_clean"] = len(df)
report["missing_clean"] = int(df.drop(columns="rating").isna().sum().sum())
df.to_csv(f"{OUT}/clean_sales.csv", index=False)

# ------------------------------------------------------------------
# 4. VISUALIZE (dashboard)
# ------------------------------------------------------------------
sns.set_theme(style="whitegrid", font_scale=0.95)
pal = sns.color_palette("deep")
fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.3)
fig.suptitle("Retail Sales Dashboard: 2024-2025", fontsize=22, fontweight="bold", y=0.97)

# KPIs
ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
kpis = [("Total revenue", f"${df.revenue.sum():,.0f}"),
        ("Orders", f"{len(df):,}"),
        ("Avg order value", f"${df.revenue.mean():,.0f}"),
        ("Avg rating", f"{df.rating.mean():.2f} / 5")]
for i, (k, v) in enumerate(kpis):
    ax.text(0.02, 0.9 - i * 0.25, v, fontsize=24, fontweight="bold", color=pal[0])
    ax.text(0.02, 0.9 - i * 0.25 - 0.09, k, fontsize=11, color="gray")

# Before/after cleaning
ax = fig.add_subplot(gs[0, 1])
labels = ["Rows", "Missing cells"]
before = [report["rows_raw"], report["missing_raw"]]
after = [report["rows_clean"], report["missing_clean"]]
x = np.arange(2)
ax.bar(x - .2, before, .4, label="Raw", color="#c44e52")
ax.bar(x + .2, after, .4, label="Clean", color="#55a868")
ax.set_xticks(x); ax.set_xticklabels(labels); ax.legend()
ax.set_title("Cleaning impact: rows and missing cells", fontweight="bold")
for i in range(2):
    ax.text(i - .2, before[i], before[i], ha="center", va="bottom", fontsize=9)
    ax.text(i + .2, after[i], after[i], ha="center", va="bottom", fontsize=9)

# Outliers before/after (price)
ax = fig.add_subplot(gs[0, 2])
tmp = pd.DataFrame({"Raw": pd.to_numeric(raw["unit_price"].astype(str).str.replace("$", "", regex=False), errors="coerce"),
                    "Clean": df["unit_price"]})
sns.boxplot(data=tmp, ax=ax, palette=["#c44e52", "#55a868"])
ax.set_title("Price outliers removed (box plot)", fontweight="bold")
ax.set_ylabel("Unit price ($)")

# Monthly revenue trend
ax = fig.add_subplot(gs[1, :2])
m = df.groupby("month")["revenue"].sum()
ax.plot(m.index, m.values, marker="o", color=pal[0], lw=2)
ax.fill_between(m.index, m.values, alpha=.15, color=pal[0])
peak = m.idxmax()
ax.annotate(f"Peak: {peak:%b %Y}", (peak, m.max()), xytext=(25, 12),
            textcoords="offset points", arrowprops=dict(arrowstyle="->"))
ax.set_title(f"Monthly revenue peaked in {peak:%B %Y}", fontweight="bold")
ax.set_ylabel("Revenue ($)")

# Revenue by category
ax = fig.add_subplot(gs[1, 2])
c = df.groupby("category")["revenue"].sum().sort_values()
ax.barh(c.index, c.values, color=pal[2])
ax.set_title("Electronics drives most revenue", fontweight="bold")
ax.set_xlabel("Revenue ($K)")
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}"))

# Region x category heatmap
ax = fig.add_subplot(gs[2, 0])
h = df.pivot_table(index="region", columns="category", values="revenue", aggfunc="sum")
sns.heatmap(h / 1000, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax, cbar=False)
ax.set_title("Revenue by region and category ($K)", fontweight="bold")
ax.tick_params(axis="y", rotation=0)

# Discount vs rating
ax = fig.add_subplot(gs[2, 1])
d = df.groupby("discount_pct")["rating"].mean().reset_index()
sns.barplot(data=d, x="discount_pct", y="rating", ax=ax, color=pal[4])
ax.set_ylim(1, 5)
ax.set_title("Does discounting change satisfaction?", fontweight="bold")
ax.set_xlabel("Discount (%)"); ax.set_ylabel("Avg rating")

# Correlation
ax = fig.add_subplot(gs[2, 2])
corr = df[["quantity", "unit_price", "discount_pct", "customer_age", "rating", "revenue"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
            cbar=False, annot_kws={"size": 8})
ax.set_title("Correlation matrix", fontweight="bold")

plt.savefig(f"{OUT}/dashboard.png", dpi=130, bbox_inches="tight")

# ------------------------------------------------------------------
# 5. FINDINGS
# ------------------------------------------------------------------
top_cat = df.groupby("category")["revenue"].sum().idxmax()
cat_share = df.groupby("category")["revenue"].sum().max() / df.revenue.sum()
top_reg = df[df.region != "Unknown"].groupby("region")["revenue"].sum().idxmax()
hol = df[df.order_date.dt.month.isin([11, 12])]["revenue"].sum() / df.revenue.sum()
summary = f"""# Key Findings

## Cleaning summary
- Rows: {report['rows_raw']:,} -> {report['rows_clean']:,} ({report['duplicates_removed']} exact duplicates removed)
- Missing cells (excluding rating): {report['missing_before_impute']:,} -> {report['missing_clean']}
- Price outliers (IQR x3 within category): {report['price_outliers']} treated as bad entries and imputed with category median
- Invalid values (negative quantity, age > 100) set to missing, then imputed
- Ratings left missing on purpose: imputing would bias satisfaction scores
- Text standardized (region, category), "$" stripped from prices, mixed date formats parsed

## Insights
1. **{top_cat}** generates {cat_share:.0%} of revenue, so inventory and marketing there matter most.
2. **{top_reg}** is the top region by revenue.
3. Nov-Dec account for {hol:.0%} of revenue (about 17% would be an even share), so {"the holiday season is a real peak worth planning for" if hol > 0.20 else "there is no strong holiday peak; demand is fairly steady across the year"}.
4. Discount level shows little relationship with ratings, so deeper discounts do not clearly buy satisfaction.

## Recommendations
- Concentrate promotions and inventory on the top category.
- Test discount levels before widening them, since the data shows no rating benefit.
- Add input validation at the source (date format, price type, age range) to stop recurring data issues.
"""
open(f"{OUT}/findings.md", "w").write(summary)
print(summary)
