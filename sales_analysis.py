import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

os.makedirs("figures", exist_ok=True)

# The CSV has no header row, so we supply the column names ourselves
COLUMNS = ["Transaction ID", "Date", "Product Category", "Product Name",
           "Units Sold", "Unit Price", "Sales Amount", "Region",
           "Payment Method", "Marketing Expenditure"]

df = pd.read_csv("data/Sales_data.csv", header=None, names=COLUMNS)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print(df.shape)
print(df.isnull().sum())
print(df["Date"].min(), df["Date"].max())


# 3a. Sales over time (daily and monthly)
plt.figure(figsize=(12, 5))
plt.plot(df["Date"], df["Sales Amount"])
plt.title("Daily Sales Amount"); plt.xlabel("Date"); plt.ylabel("Sales Amount")
plt.tight_layout(); plt.savefig("figures/sales_trend.png", dpi=150); plt.close()

monthly = df.resample("MS", on="Date")["Sales Amount"].sum()
print(monthly.round(0))
monthly.plot(kind="bar", figsize=(8, 4), title="Monthly Sales")
plt.tight_layout(); plt.savefig("figures/monthly_sales.png", dpi=150); plt.close()

# 3b. Sales by region and category
region = df.groupby("Region")["Sales Amount"].sum()
print(region)
region.plot(kind="bar", title="Total Sales by Region")
plt.tight_layout(); plt.savefig("figures/sales_by_region.png", dpi=150); plt.close()

category = df.groupby("Product Category")["Sales Amount"].sum().sort_values(ascending=False)
print(category)

# 3c. Correlation matrix
num = ["Units Sold", "Unit Price", "Sales Amount", "Marketing Expenditure"]
corr = df[num].corr()
print(corr.round(2))
plt.figure(figsize=(7, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout(); plt.savefig("figures/correlation.png", dpi=150); plt.close()



# 4a. Is Sales Amount just price x units?
gap = (df["Sales Amount"] - df["Unit Price"] * df["Units Sold"]).abs().max()
print("Largest gap between Sales and Price x Units:", gap)

# 4b. How large is marketing relative to sales?
ratio = df["Marketing Expenditure"] / df["Sales Amount"]
print(ratio.describe().round(3))



features = ["Marketing Expenditure", "Units Sold"]
X = df[features].to_numpy(dtype=float)
y = df["Sales Amount"].to_numpy(dtype=float)

# 5a. Design matrix A (column of ones + features) and the normal equation
A = np.column_stack([np.ones(len(X)), X])
beta, *_ = np.linalg.lstsq(A, y, rcond=None)
print("Normal-equation coefficients:", beta)

sk = LinearRegression().fit(X, y)
print("sklearn:", sk.intercept_, sk.coef_)   # should match

# 5b. SVD and condition number of the standardised features
Xs = (X - X.mean(axis=0)) / X.std(axis=0)
S = np.linalg.svd(Xs, compute_uv=False)
print("Singular values:", S)
print("Condition number:", S.max() / S.min())

# 5c. Eigenvalues of the correlation matrix
R = np.corrcoef(X, rowvar=False)
print("Eigenvalues:", np.linalg.eigvalsh(R))

# 5d. VIF = diagonal of the inverse correlation matrix
print("VIF:", np.diag(np.linalg.inv(R)))



split = int(len(df) * 0.8)            # first 80% of dates = train
train, test = df.iloc[:split], df.iloc[split:]

def fit_eval(cols, label, target="Sales Amount", tr=train, te=test):
    m = LinearRegression().fit(tr[cols], tr[target])
    pred = m.predict(te[cols])
    r2 = r2_score(te[target], pred)
    rmse = np.sqrt(mean_squared_error(te[target], pred))
    print(f"{label}: R2={r2:.4f}  RMSE={rmse:.2f}  coef={m.coef_}")
    return m, pred

# Model A: marketing only
fit_eval(["Marketing Expenditure"], "A  marketing only")

# Model B: marketing + units
m_b, pred_b = fit_eval(["Marketing Expenditure", "Units Sold"], "B  marketing + units")

# Model C: price + units (the identity: shows the limit of 'prediction' here)
fit_eval(["Unit Price", "Units Sold"], "C  price + units")

# Model D: log-log. log(S) = log(P) + log(U), exactly linear
d = df.copy()
for c in ["Unit Price", "Units Sold", "Sales Amount"]:
    d["log_" + c] = np.log(d[c])
tr, te = d.iloc[:split], d.iloc[split:]
fit_eval(["log_Unit Price", "log_Units Sold"], "D  log-log", "log_Sales Amount", tr, te)

# Model E: genuine forecast from information known in advance (time, region, category)
d["t"] = (d["Date"] - d["Date"].min()).dt.days
d = pd.concat([d,
               pd.get_dummies(d["Region"], prefix="R", drop_first=True).astype(float),
               pd.get_dummies(d["Product Category"], prefix="C", drop_first=True).astype(float)],
              axis=1)
tr, te = d.iloc[:split], d.iloc[split:]
cols_e = ["t"] + [c for c in d.columns if c.startswith(("R_", "C_"))]
m_e, pred_e = fit_eval(cols_e, "E  time + region + category", tr=tr, te=te)

naive = np.full(len(te), train["Sales Amount"].mean())
print("Naive mean baseline R2:", round(r2_score(te["Sales Amount"], naive), 4))



resid = te["Sales Amount"] - pred_e
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(te["Sales Amount"], pred_e, s=25)
lims = [te["Sales Amount"].min(), te["Sales Amount"].max()]
ax[0].plot(lims, lims, "r--")
ax[0].set(xlabel="Actual", ylabel="Predicted", title="Model E: Predicted vs Actual")
ax[1].scatter(pred_e, resid, s=25)
ax[1].axhline(0, color="r", ls="--")
ax[1].set(xlabel="Predicted", ylabel="Residual", title="Model E: Residuals")
plt.tight_layout(); plt.savefig("figures/diagnostics.png", dpi=150); plt.close()


