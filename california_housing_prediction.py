# ============================================================
#   CALIFORNIA HOUSING PRICE PREDICTION
#   Complete ML Training Code
#   Libraries: NumPy, Pandas, Scikit-Learn, Matplotlib, Seaborn
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================================
# STEP 1 — DATA GENERATION
# (California Housing Dataset ki same properties ke saath)
# ============================================================

print("=" * 60)
print("   CALIFORNIA HOUSING PRICE PREDICTION")
print("=" * 60)

np.random.seed(42)
n = 20640

# Geographic bounds — California
lat = np.random.uniform(32.5, 42.0, n)
lon = np.random.uniform(-124.5, -114.0, n)

# Features
med_inc    = np.random.lognormal(mean=1.5, sigma=0.6, size=n).clip(0.5, 15)
house_age  = np.random.uniform(1, 52, n)
ave_rooms  = np.random.lognormal(1.6, 0.4, n).clip(1, 20)
ave_bedrms = (ave_rooms * np.random.uniform(0.18, 0.3, n)).clip(1, 6)
population = np.random.lognormal(6.5, 0.8, n).clip(3, 35682)
ave_occup  = np.random.lognormal(1.1, 0.4, n).clip(0.5, 20)

# Target: House Price (income + location pe based)
coastal_bonus = np.exp(-0.5 * ((lon + 122)**2 + (lat - 37.5)**2) / 5)
price = (
    0.45 * med_inc
    + 0.30 * coastal_bonus
    + 0.03 * house_age / 10
    + 0.10 * ave_rooms / 5
    - 0.05 * ave_occup / 3
    + np.random.normal(0, 0.25, n)
).clip(0.15, 5.0)

# DataFrame banao
df = pd.DataFrame({
    'MedInc':      med_inc,
    'HouseAge':    house_age,
    'AveRooms':    ave_rooms,
    'AveBedrms':   ave_bedrms,
    'Population':  population,
    'AveOccup':    ave_occup,
    'Latitude':    lat,
    'Longitude':   lon,
    'MedHouseVal': price,
})

print(f"\n[STEP 1] Data Loaded")
print(f"  Shape  : {df.shape}")
print(f"  Columns: {list(df.columns)}")
print(f"\n{df.describe().round(2)}")

# ============================================================
# STEP 2 — DATA CLEANING
# ============================================================

print("\n" + "=" * 60)
print("   STEP 2 — DATA CLEANING")
print("=" * 60)

# Missing values check
missing = df.isnull().sum().sum()
print(f"\n  Missing Values : {missing}")

# Outlier removal — 1% to 99% range
before = len(df)
for col in ['AveRooms', 'AveBedrms', 'AveOccup', 'Population']:
    q_low  = df[col].quantile(0.01)
    q_high = df[col].quantile(0.99)
    df = df[(df[col] >= q_low) & (df[col] <= q_high)]

after = len(df)
print(f"  Rows Before    : {before}")
print(f"  Rows After     : {after}")
print(f"  Removed        : {before - after} outlier rows")

# ============================================================
# STEP 3 — FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 60)
print("   STEP 3 — FEATURE ENGINEERING")
print("=" * 60)

# 3 naye features banao
df['RoomsPerPerson']    = df['AveRooms']   / df['AveOccup']
df['BedroomRatio']      = df['AveBedrms']  / df['AveRooms']
df['PopulationDensity'] = df['Population'] / df['AveOccup']

print("\n  New Features Added:")
print("  ✔ RoomsPerPerson    = AveRooms / AveOccup")
print("  ✔ BedroomRatio      = AveBedrms / AveRooms")
print("  ✔ PopulationDensity = Population / AveOccup")

# Final feature list
features = [
    'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
    'Population', 'AveOccup', 'Latitude', 'Longitude',
    'RoomsPerPerson', 'BedroomRatio', 'PopulationDensity'
]

X = df[features]
y = df['MedHouseVal']

print(f"\n  Total Features : {len(features)}")
print(f"  X shape        : {X.shape}")
print(f"  y shape        : {y.shape}")

# ============================================================
# STEP 4 — TRAIN / TEST SPLIT & SCALING
# ============================================================

print("\n" + "=" * 60)
print("   STEP 4 — TRAIN/TEST SPLIT & SCALING")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# StandardScaler — mean=0, std=1
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"\n  Train Set : {X_train.shape[0]} rows (80%)")
print(f"  Test Set  : {X_test.shape[0]} rows (20%)")
print(f"  Scaling   : StandardScaler applied")

# ============================================================
# STEP 5 — MODEL TRAINING
# ============================================================

print("\n" + "=" * 60)
print("   STEP 5 — MODEL TRAINING")
print("=" * 60)

# ── Model 1: Linear Regression ──
print("\n  [1/4] Training Linear Regression...")
lr = LinearRegression()
lr.fit(X_train_s, y_train)
y_pred_lr = lr.predict(X_test_s)

# ── Model 2: Ridge Regression ──
print("  [2/4] Training Ridge Regression...")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_s, y_train)
y_pred_ridge = ridge.predict(X_test_s)

# ── Model 3: Random Forest ──
print("  [3/4] Training Random Forest (150 trees)...")
rf = RandomForestRegressor(
    n_estimators=150,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# ── Model 4: Gradient Boosting ──
print("  [4/4] Training Gradient Boosting (200 stages)...")
gb = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)

print("\n  All models trained successfully! ✔")

# ============================================================
# STEP 6 — MODEL EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("   STEP 6 — MODEL EVALUATION")
print("=" * 60)

def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {'name': name, 'R2': r2, 'RMSE': rmse, 'MAE': mae, 'pred': y_pred}

results = [
    evaluate("Linear Regression", y_test, y_pred_lr),
    evaluate("Ridge Regression",  y_test, y_pred_ridge),
    evaluate("Random Forest",     y_test, y_pred_rf),
    evaluate("Gradient Boosting", y_test, y_pred_gb),
]

print(f"\n  {'Model':<22} {'R²':>8} {'RMSE':>8} {'MAE':>8}")
print("  " + "-" * 50)
for r in results:
    print(f"  {r['name']:<22} {r['R2']:>8.4f} {r['RMSE']:>8.4f} {r['MAE']:>8.4f}")

# Best model
best  = max(results, key=lambda x: x['R2'])
bp    = best['pred']
bname = best['name']
bmodel = {'Linear Regression': lr, 'Ridge Regression': ridge,
          'Random Forest': rf, 'Gradient Boosting': gb}[bname]

print(f"\n  ★ Best Model : {bname}")
print(f"  ★ R² Score   : {best['R2']:.4f}  ({best['R2']*100:.1f}% accuracy)")
print(f"  ★ Avg Error  : ${best['MAE']*100000:,.0f} per house")

# ============================================================
# STEP 7 — VISUALIZATION DASHBOARD
# ============================================================

print("\n" + "=" * 60)
print("   STEP 7 — GENERATING DASHBOARD")
print("=" * 60)

plt.style.use('dark_background')

# Colors
ACCENT  = '#00C8FF'
ACCENT2 = '#FF6B6B'
GREEN   = '#4ECDC4'
YELLOW  = '#FFD93D'
GRID_C  = '#2a2a3e'
BG      = '#0d0d1a'

fig = plt.figure(figsize=(20, 22), facecolor=BG)
fig.suptitle('California Housing Price Prediction — ML Dashboard',
             fontsize=22, fontweight='bold', color='white', y=0.98)

gs = gridspec.GridSpec(4, 3, figure=fig,
                       hspace=0.45, wspace=0.35,
                       left=0.06, right=0.97,
                       top=0.94, bottom=0.04)

# ── Panel 1: Model Comparison Bar Chart ──
ax1 = fig.add_subplot(gs[0, :2])
names  = [r['name'] for r in results]
r2s    = [r['R2']   for r in results]
colors = [GREEN if n == bname else ACCENT for n in names]
bars   = ax1.barh(names, r2s, color=colors, height=0.5, edgecolor='none')
for bar, val in zip(bars, r2s):
    ax1.text(bar.get_width() - 0.02, bar.get_y() + bar.get_height() / 2,
             f'{val:.4f}', va='center', ha='right',
             fontsize=12, fontweight='bold', color='black')
ax1.set_xlim(0, 1.05)
ax1.set_xlabel('R² Score', color='white')
ax1.set_title('Model Comparison (R² Score)', color=ACCENT, fontsize=13, pad=10)
ax1.set_facecolor(GRID_C)
ax1.tick_params(colors='white')
ax1.spines[:].set_visible(False)

# ── Panel 2: Metrics Table ──
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis('off')
table_data = [[r['name'], f"{r['R2']:.4f}", f"{r['RMSE']:.4f}", f"{r['MAE']:.4f}"]
              for r in results]
tbl = ax2.table(cellText=table_data,
                colLabels=['Model', 'R²', 'RMSE', 'MAE'],
                loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor('#444')
    if row == 0:
        cell.set_facecolor(ACCENT)
        cell.set_text_props(color='black', fontweight='bold')
    elif results[row - 1]['name'] == bname:
        cell.set_facecolor('#1a3a1a')
        cell.set_text_props(color=GREEN)
    else:
        cell.set_facecolor(GRID_C)
        cell.set_text_props(color='white')
ax2.set_title('Metrics Summary', color=ACCENT, fontsize=13, pad=10)

# ── Panel 3: Actual vs Predicted ──
ax3 = fig.add_subplot(gs[1, :2])
ax3.scatter(y_test, bp, alpha=0.3, s=8, color=ACCENT)
lims = [min(y_test.min(), bp.min()), max(y_test.max(), bp.max())]
ax3.plot(lims, lims, color=ACCENT2, lw=2, linestyle='--', label='Perfect Fit')
ax3.set_xlabel('Actual Price ($100k)', color='white')
ax3.set_ylabel('Predicted Price ($100k)', color='white')
ax3.set_title(f'Actual vs Predicted — {bname}', color=ACCENT, fontsize=13)
ax3.set_facecolor(GRID_C)
ax3.legend(facecolor='#1a1a2e', labelcolor='white')
ax3.tick_params(colors='white')
ax3.spines[:].set_visible(False)
ax3.text(0.05, 0.92, f'R² = {best["R2"]:.4f}',
         transform=ax3.transAxes, color=GREEN,
         fontsize=12, fontweight='bold')

# ── Panel 4: Residuals Distribution ──
ax4 = fig.add_subplot(gs[1, 2])
residuals = y_test.values - bp
ax4.hist(residuals, bins=50, color=ACCENT2, alpha=0.8, edgecolor='none')
ax4.axvline(0, color='white', lw=1.5, linestyle='--')
ax4.set_xlabel('Residual', color='white')
ax4.set_ylabel('Frequency', color='white')
ax4.set_title('Residual Distribution', color=ACCENT, fontsize=13)
ax4.set_facecolor(GRID_C)
ax4.tick_params(colors='white')
ax4.spines[:].set_visible(False)

# ── Panel 5: Feature Importance ──
ax5 = fig.add_subplot(gs[2, :2])
if hasattr(bmodel, 'feature_importances_'):
    importances = bmodel.feature_importances_
else:
    importances = np.abs(bmodel.coef_)

fi_df = pd.DataFrame({'Feature': features, 'Importance': importances})
fi_df = fi_df.sort_values('Importance', ascending=True)
bar_colors = [YELLOW if v == fi_df['Importance'].max()
              else ACCENT for v in fi_df['Importance']]
ax5.barh(fi_df['Feature'], fi_df['Importance'],
         color=bar_colors, edgecolor='none')
ax5.set_title(f'Feature Importance — {bname}', color=ACCENT, fontsize=13)
ax5.set_facecolor(GRID_C)
ax5.tick_params(colors='white')
ax5.spines[:].set_visible(False)

# ── Panel 6: Correlation Heatmap ──
ax6 = fig.add_subplot(gs[2, 2])
corr = df[features + ['MedHouseVal']].corr()[['MedHouseVal']].drop('MedHouseVal')
sns.heatmap(corr, ax=ax6, cmap='coolwarm', annot=True, fmt='.2f',
            linewidths=0.5, linecolor='#1a1a2e',
            cbar_kws={'shrink': 0.7})
ax6.set_title('Feature Correlations\nwith House Price',
              color=ACCENT, fontsize=13)
ax6.tick_params(colors='white', labelsize=8)
ax6.set_ylabel('')

# ── Panel 7: Price Distribution ──
ax7 = fig.add_subplot(gs[3, 0])
ax7.hist(y, bins=50, color=GREEN, alpha=0.85, edgecolor='none')
ax7.axvline(y.median(), color=YELLOW, lw=2,
            linestyle='--', label=f'Median={y.median():.2f}')
ax7.set_xlabel('House Value ($100k)', color='white')
ax7.set_ylabel('Frequency', color='white')
ax7.set_title('Price Distribution', color=ACCENT, fontsize=13)
ax7.set_facecolor(GRID_C)
ax7.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
ax7.tick_params(colors='white')
ax7.spines[:].set_visible(False)

# ── Panel 8: Geographic Price Map ──
ax8 = fig.add_subplot(gs[3, 1:])
sc = ax8.scatter(df['Longitude'], df['Latitude'],
                 c=df['MedHouseVal'], cmap='plasma',
                 alpha=0.3, s=3)
cbar = plt.colorbar(sc, ax=ax8, fraction=0.03, pad=0.04)
cbar.set_label('House Value ($100k)', color='white', fontsize=9)
cbar.ax.tick_params(colors='white')
ax8.set_xlabel('Longitude', color='white')
ax8.set_ylabel('Latitude', color='white')
ax8.set_title('Geographic Price Distribution (California)',
              color=ACCENT, fontsize=13)
ax8.set_facecolor(GRID_C)
ax8.tick_params(colors='white')
ax8.spines[:].set_visible(False)

plt.savefig('housing_price_prediction.png', dpi=150,
            bbox_inches='tight', facecolor=BG)

print("\n  Dashboard saved → housing_price_prediction.png ✔")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("   FINAL RESULTS SUMMARY")
print("=" * 60)
print(f"\n  Dataset    : {len(df):,} rows, {len(features)} features")
print(f"  Train/Test : {len(X_train):,} / {len(X_test):,}")
print(f"\n  {'Model':<22} {'R²':>8} {'RMSE':>8} {'MAE':>8}")
print("  " + "-" * 50)
for r in sorted(results, key=lambda x: x['R2'], reverse=True):
    star = " ★" if r['name'] == bname else ""
    print(f"  {r['name']:<22} {r['R2']:>8.4f} {r['RMSE']:>8.4f} {r['MAE']:>8.4f}{star}")

print(f"\n  Best Model : {bname}")
print(f"  R² Score   : {best['R2']:.4f}")
print(f"  RMSE       : {best['RMSE']:.4f}")
print(f"  MAE        : ${best['MAE'] * 100000:,.0f} average error per house")
print("\n" + "=" * 60)
print("   TRAINING COMPLETE ✔")
print("=" * 60)
