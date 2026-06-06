# =============================================================================
# Titanic Dataset – Exploratory Data Analysis
# =============================================================================
# Author  : Python & AI Training
# Dataset : seaborn built-in Titanic (mirrors Kaggle Titanic dataset)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Style ──────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})
COLORS = {"survived": "#2ecc71", "died": "#e74c3c"}

df = pd.read_csv("titanic.csv")

print("=" * 60)
print("  TITANIC EDA – Summary Report")
print("=" * 60)

# ── 1. BASIC INFO ──────────────────────────────────────────────────────────
print("\n[1] Shape:", df.shape)
print("\n[2] Data Types:\n", df.dtypes)
print("\n[3] Missing Values:\n", df.isnull().sum()[df.isnull().sum() > 0])
print("\n[4] Descriptive Stats:\n", df.describe().round(2))

# ── 2. DATA CLEANING ───────────────────────────────────────────────────────
df['age'].fillna(df['age'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
df['deck'].fillna('Unknown', inplace=True)

# Derived features
df['age_group'] = pd.cut(df['age'],
                          bins=[0, 12, 18, 35, 60, 100],
                          labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])
df['family_size'] = df['sibsp'] + df['parch'] + 1
df['is_alone'] = (df['family_size'] == 1).astype(int)
df['fare_band'] = pd.qcut(df['fare'], q=4, labels=['Low', 'Mid', 'High', 'Very High'])

# ── 3. KEY INSIGHTS ────────────────────────────────────────────────────────
survival_rate = df['survived'].mean() * 100
print(f"\n[5] Overall Survival Rate: {survival_rate:.1f}%")

by_sex = df.groupby('sex')['survived'].mean().mul(100).round(1)
print("\n[6] Survival Rate by Sex:\n", by_sex)

by_class = df.groupby('pclass')['survived'].mean().mul(100).round(1)
print("\n[7] Survival Rate by Class:\n", by_class)

by_age = df.groupby('age_group')['survived'].mean().mul(100).round(1)
print("\n[8] Survival Rate by Age Group:\n", by_age)

# ── 4. PLOTS ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 24))
fig.suptitle("Titanic EDA – Full Dashboard", fontsize=18, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

# --- 4.1 Survival count ---
ax1 = fig.add_subplot(gs[0, 0])
counts = df['survived'].value_counts()
bars = ax1.bar(['Died', 'Survived'], counts.values,
               color=[COLORS['died'], COLORS['survived']], edgecolor='white', width=0.5)
for bar, val in zip(bars, counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f'{val}\n({val/len(df)*100:.1f}%)', ha='center', fontsize=10)
ax1.set(title='Survival Count', ylabel='Passengers')
ax1.set_ylim(0, max(counts.values) * 1.2)

# --- 4.2 Survival by Sex ---
ax2 = fig.add_subplot(gs[0, 1])
sex_surv = df.groupby(['sex', 'survived']).size().unstack()
sex_surv.plot(kind='bar', ax=ax2,
              color=[COLORS['died'], COLORS['survived']],
              edgecolor='white', legend=False)
ax2.set(title='Survival by Sex', xlabel='', ylabel='Count')
ax2.set_xticklabels(['Female', 'Male'], rotation=0)
ax2.legend(['Died', 'Survived'], loc='upper right', fontsize=9)

# --- 4.3 Survival by Pclass ---
ax3 = fig.add_subplot(gs[0, 2])
class_surv = df.groupby(['pclass', 'survived']).size().unstack()
class_surv.plot(kind='bar', ax=ax3,
                color=[COLORS['died'], COLORS['survived']],
                edgecolor='white', legend=False)
ax3.set(title='Survival by Passenger Class', xlabel='Class', ylabel='Count')
ax3.legend(['Died', 'Survived'], loc='upper right', fontsize=9)
ax3.set_xticklabels(['1st', '2nd', '3rd'], rotation=0)

# --- 4.4 Age distribution ---
ax4 = fig.add_subplot(gs[1, 0])
df[df['survived'] == 0]['age'].plot(kind='hist', bins=25, ax=ax4,
                                    alpha=0.6, color=COLORS['died'], label='Died')
df[df['survived'] == 1]['age'].plot(kind='hist', bins=25, ax=ax4,
                                    alpha=0.6, color=COLORS['survived'], label='Survived')
ax4.set(title='Age Distribution by Survival', xlabel='Age', ylabel='Count')
ax4.legend()
ax4.axvline(df['age'].median(), color='navy', linestyle='--', linewidth=1.2, label='Median Age')

# --- 4.5 Survival rate by age group ---
ax5 = fig.add_subplot(gs[1, 1])
age_rates = df.groupby('age_group', observed=True)['survived'].mean().mul(100)
bars = ax5.bar(age_rates.index, age_rates.values,
               color=sns.color_palette("Blues_d", len(age_rates)), edgecolor='white')
for bar, val in zip(bars, age_rates.values):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center', fontsize=9)
ax5.set(title='Survival Rate by Age Group', xlabel='Age Group', ylabel='Survival Rate (%)')
ax5.set_xticklabels(age_rates.index, rotation=15)

# --- 4.6 Fare distribution ---
ax6 = fig.add_subplot(gs[1, 2])
df[df['survived'] == 0]['fare'].clip(upper=300).plot(kind='hist', bins=30, ax=ax6,
                                                      alpha=0.6, color=COLORS['died'], label='Died')
df[df['survived'] == 1]['fare'].clip(upper=300).plot(kind='hist', bins=30, ax=ax6,
                                                      alpha=0.6, color=COLORS['survived'], label='Survived')
ax6.set(title='Fare Distribution by Survival (capped at 300)', xlabel='Fare (£)', ylabel='Count')
ax6.legend()

# --- 4.7 Heatmap: survival rate by class & sex ---
ax7 = fig.add_subplot(gs[2, 0])
pivot = df.pivot_table('survived', index='sex', columns='pclass', aggfunc='mean').mul(100)
sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax7,
            cbar_kws={'label': 'Survival %'}, linewidths=0.5)
ax7.set(title='Survival Rate (%) by Sex & Class', xlabel='Class', ylabel='Sex')

# --- 4.8 Family size vs survival ---
ax8 = fig.add_subplot(gs[2, 1])
fam_rate = df.groupby('family_size')['survived'].mean().mul(100)
ax8.plot(fam_rate.index, fam_rate.values, marker='o', color='steelblue', linewidth=2)
ax8.fill_between(fam_rate.index, fam_rate.values, alpha=0.15, color='steelblue')
ax8.set(title='Survival Rate by Family Size', xlabel='Family Size', ylabel='Survival Rate (%)')

# --- 4.9 Embarkation port ---
ax9 = fig.add_subplot(gs[2, 2])
emb_surv = df.groupby(['embarked', 'survived']).size().unstack().fillna(0)
emb_surv.plot(kind='bar', ax=ax9,
              color=[COLORS['died'], COLORS['survived']],
              edgecolor='white', legend=False)
ax9.set(title='Survival by Embarkation Port', xlabel='Port', ylabel='Count')
ax9.set_xticklabels(['Cherbourg (C)', 'Queenstown (Q)', 'Southampton (S)'], rotation=15)
ax9.legend(['Died', 'Survived'], loc='upper right', fontsize=9)

# --- 4.10 Correlation heatmap ---
ax10 = fig.add_subplot(gs[3, :2])
num_cols = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare', 'family_size', 'is_alone']
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax10, linewidths=0.5, cbar_kws={'shrink': 0.8})
ax10.set_title('Correlation Matrix – Numerical Features')

# --- 4.11 Survival by fare band & class ---
ax11 = fig.add_subplot(gs[3, 2])
fare_class = df.groupby(['fare_band', 'pclass'], observed=True)['survived'].mean().unstack().mul(100)
sns.heatmap(fare_class, annot=True, fmt='.0f', cmap='YlGn', ax=ax11,
            linewidths=0.5, cbar_kws={'label': 'Survival %'})
ax11.set(title='Survival Rate by Fare Band & Class', xlabel='Class', ylabel='Fare Band')

plt.savefig('titanic_eda_dashboard.png', bbox_inches='tight', dpi=130)
plt.close()
print("\n✅ Dashboard saved: titanic_eda_dashboard.png")

# ── 5. WRITTEN SUMMARY ─────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════╗
║              KEY FINDINGS & INSIGHTS                     ║
╠══════════════════════════════════════════════════════════╣
║ 1. SURVIVAL RATE: ~38% overall survived the disaster.    ║
║ 2. GENDER: Women had ~74% survival vs ~19% for men.      ║
║    ("Women and children first" policy was enforced.)     ║
║ 3. CLASS: 1st class 63%, 2nd class 47%, 3rd class 24%.   ║
║    Wealth strongly correlated with survival.             ║
║ 4. AGE: Children (≤12) had the highest survival rate.    ║
║    Seniors had the lowest.                               ║
║ 5. FARE: Higher fare payers (wealthier) survived more.   ║
║ 6. FAMILY SIZE: Small families (2–4) fared best.         ║
║    Solo travellers and large families did poorly.        ║
║ 7. PORT: Cherbourg (C) had highest survival – many       ║
║    1st-class passengers boarded there.                   ║
║ 8. CORRELATION: 'pclass' and 'fare' are strongest        ║
║    predictors of survival after sex.                     ║
╚══════════════════════════════════════════════════════════╝
""")
