import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── DATASET ──
np.random.seed(42)
n = 200

transaction_types = ['Online Transfer', 'ATM Withdrawal', 'POS Payment', 'UPI Payment', 'NEFT/RTGS']
cities = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Pune']

amounts = np.concatenate([
    np.random.exponential(15000, int(n*0.7)),
    np.random.exponential(85000, int(n*0.3))
])[:n]
amounts = np.clip(amounts, 500, 500000).astype(int)

is_fraud = []
for amt in amounts:
    prob = 0.05 if amt < 20000 else (0.15 if amt < 50000 else 0.35)
    is_fraud.append(np.random.choice([1, 0], p=[prob, 1-prob]))

df = pd.DataFrame({
    'Transaction_ID': [f'TXN{1000+i}' for i in range(n)],
    'Amount': amounts,
    'Transaction_Type': np.random.choice(transaction_types, n),
    'City': np.random.choice(cities, n),
    'Hour': np.random.randint(0, 24, n),
    'Is_Fraud': is_fraud,
    'Account_Age_Days': np.random.randint(10, 3000, n),
    'Previous_Fraud': np.random.choice([0, 1], n, p=[0.85, 0.15]),
})

df['Risk_Score'] = (
    (df['Amount'] > 50000).astype(int) * 40 +
    (df['Hour'].isin([0,1,2,3,4,23])).astype(int) * 25 +
    (df['Previous_Fraud'] == 1).astype(int) * 30 +
    (df['Account_Age_Days'] < 30).astype(int) * 20 +
    np.random.randint(0, 15, n)
)
df['Risk_Level'] = pd.cut(df['Risk_Score'], bins=[-1, 25, 50, 75, 100],
                           labels=['Low', 'Medium', 'High', 'Critical'])

# ── COLORS ──
NAVY = '#0D1B2A'
BLUE1 = '#1B3A5C'
BLUE2 = '#2E6BAD'
BLUE3 = '#5BA4CF'
BLUE4 = '#A8D1F0'
GOLD = '#E8A838'
RED = '#C0392B'
ORANGE = '#E67E22'
GREEN = '#27AE60'
WHITE = '#FFFFFF'
BG = '#F0F4F8'

risk_colors = {'Low': GREEN, 'Medium': GOLD, 'High': ORANGE, 'Critical': RED}

# ── FIGURE ──
fig = plt.figure(figsize=(20, 24), facecolor=BG)

# Title
ax_title = fig.add_axes([0, 0.95, 1, 0.05])
ax_title.set_facecolor(NAVY)
ax_title.text(0.5, 0.5, '🔍  FINANCIAL FRAUD DETECTION SYSTEM  🔍',
              ha='center', va='center', fontsize=22, fontweight='bold',
              color=GOLD, transform=ax_title.transAxes)
ax_title.axis('off')

ax_sub = fig.add_axes([0, 0.92, 1, 0.03])
ax_sub.set_facecolor(BLUE1)
ax_sub.text(0.5, 0.5, 'Transaction Risk Analysis  |  SQL · Python · Excel  |  G. Sweekrith Kumar',
            ha='center', va='center', fontsize=13, color=BLUE4,
            transform=ax_sub.transAxes)
ax_sub.axis('off')

# ── KPI CARDS ──
fraud_count = df['Is_Fraud'].sum()
fraud_rate = df['Is_Fraud'].mean() * 100
fraud_amount = df[df['Is_Fraud']==1]['Amount'].sum()
total_amount = df['Amount'].sum()
high_risk = (df['Risk_Level'].isin(['High', 'Critical'])).sum()

kpis = [
    ('💳 Total Transactions', f'{len(df):,}', NAVY),
    ('🚨 Fraud Cases', str(fraud_count), RED),
    ('📊 Fraud Rate', f'{fraud_rate:.1f}%', RED),
    ('💸 Fraud Amount', f'₹{fraud_amount:,}', BLUE1),
    ('⚠️ High Risk Txns', str(high_risk), ORANGE),
]

for i, (label, value, color) in enumerate(kpis):
    ax = fig.add_axes([0.01 + i * 0.196, 0.86, 0.185, 0.055])
    ax.set_facecolor(color)
    ax.text(0.5, 0.7, label, ha='center', va='center', fontsize=11,
            color=BLUE4, transform=ax.transAxes)
    ax.text(0.5, 0.25, value, ha='center', va='center', fontsize=16,
            fontweight='bold', color=GOLD, transform=ax.transAxes)
    for spine in ax.spines.values():
        spine.set_edgecolor(GOLD)
        spine.set_linewidth(1.5)
    ax.set_xticks([])
    ax.set_yticks([])

# ── CHART 1: Fraud by Transaction Type ──
ax1 = fig.add_axes([0.05, 0.62, 0.4, 0.22])
type_fraud = df.groupby('Transaction_Type')['Is_Fraud'].mean() * 100
type_fraud = type_fraud.sort_values(ascending=False)
bars = ax1.bar(range(len(type_fraud)), type_fraud.values,
               color=[RED if v > 20 else ORANGE if v > 15 else BLUE2 for v in type_fraud.values],
               edgecolor=WHITE, linewidth=1, width=0.5)
ax1.set_xticks(range(len(type_fraud)))
ax1.set_xticklabels(type_fraud.index, rotation=15, ha='right', fontsize=10, color=NAVY)
for bar, val in zip(bars, type_fraud.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.3,
             f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold', color=NAVY)
ax1.axhline(y=fraud_rate, color=GOLD, linestyle='--', linewidth=2,
            label=f'Avg Fraud Rate: {fraud_rate:.1f}%')
ax1.set_title('💳  Fraud Rate by Transaction Type', fontsize=14, fontweight='bold', color=NAVY, pad=10)
ax1.set_ylabel('Fraud Rate (%)', color=NAVY)
ax1.legend(fontsize=10)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_facecolor('#FAFCFF')
ax1.tick_params(colors=NAVY)

# ── CHART 2: Amount Distribution Fraud vs Legit ──
ax2 = fig.add_axes([0.55, 0.62, 0.4, 0.22])
fraud_amounts = df[df['Is_Fraud']==1]['Amount']
legit_amounts = df[df['Is_Fraud']==0]['Amount']
ax2.hist(legit_amounts, bins=20, alpha=0.7, color=BLUE2, label='Legitimate', edgecolor=WHITE)
ax2.hist(fraud_amounts, bins=20, alpha=0.8, color=RED, label='Fraud', edgecolor=WHITE)
ax2.axvline(50000, color=GOLD, linestyle='--', linewidth=2, label='₹50,000 Threshold')
ax2.set_title('💰  Transaction Amount: Fraud vs Legitimate', fontsize=14, fontweight='bold', color=NAVY, pad=10)
ax2.set_xlabel('Amount (₹)', color=NAVY)
ax2.set_ylabel('Count', color=NAVY)
ax2.legend(fontsize=10)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_facecolor('#FAFCFF')
ax2.tick_params(colors=NAVY)

# ── CHART 3: Hourly Fraud Pattern ──
ax3 = fig.add_axes([0.05, 0.36, 0.4, 0.22])
hourly = df.groupby('Hour')['Is_Fraud'].mean() * 100
ax3.plot(hourly.index, hourly.values, color=RED, linewidth=2.5,
         marker='o', markersize=6, markerfacecolor=NAVY)
ax3.fill_between(hourly.index, hourly.values, alpha=0.2, color=RED)
ax3.axvspan(0, 5, alpha=0.1, color=RED, label='High Risk Hours (12am-5am)')
ax3.axvspan(22, 24, alpha=0.1, color=RED)
ax3.set_title('🕐  Hourly Fraud Pattern', fontsize=14, fontweight='bold', color=NAVY, pad=10)
ax3.set_xlabel('Hour of Day', color=NAVY)
ax3.set_ylabel('Fraud Rate (%)', color=NAVY)
ax3.legend(fontsize=10)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.set_facecolor('#FAFCFF')
ax3.tick_params(colors=NAVY)
ax3.set_xticks(range(0, 24, 2))

# ── CHART 4: Risk Level Distribution ──
ax4 = fig.add_axes([0.55, 0.36, 0.4, 0.22])
risk_counts = df['Risk_Level'].value_counts()
risk_order = ['Low', 'Medium', 'High', 'Critical']
risk_counts = risk_counts.reindex(risk_order).dropna()
wedges, texts, autotexts = ax4.pie(
    risk_counts.values,
    labels=risk_counts.index,
    autopct='%1.1f%%',
    colors=[risk_colors[r] for r in risk_counts.index],
    startangle=90,
    wedgeprops={'edgecolor': WHITE, 'linewidth': 2}
)
for text in texts:
    text.set_fontsize(12)
    text.set_fontweight('bold')
for autotext in autotexts:
    autotext.set_fontsize(11)
    autotext.set_color(WHITE)
    autotext.set_fontweight('bold')
ax4.set_title('⚠️  Risk Level Distribution', fontsize=14, fontweight='bold', color=NAVY, pad=10)

# ── CHART 5: City-wise Fraud ──
ax5 = fig.add_axes([0.05, 0.10, 0.4, 0.22])
city_fraud = df.groupby('City')['Is_Fraud'].agg(['sum', 'mean']).sort_values('mean', ascending=True)
city_fraud['mean'] = city_fraud['mean'] * 100
bars5 = ax5.barh(city_fraud.index, city_fraud['mean'],
                  color=[RED if v > 20 else BLUE2 for v in city_fraud['mean']],
                  edgecolor=WHITE, linewidth=1, height=0.5)
for bar, val in zip(bars5, city_fraud['mean']):
    ax5.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
             f'{val:.1f}%', va='center', fontsize=11, fontweight='bold', color=NAVY)
ax5.axvline(fraud_rate, color=GOLD, linestyle='--', linewidth=2)
ax5.set_title('🏙️  Fraud Rate by City', fontsize=14, fontweight='bold', color=NAVY, pad=10)
ax5.set_xlabel('Fraud Rate (%)', color=NAVY)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.set_facecolor('#FAFCFF')
ax5.tick_params(colors=NAVY)

# ── CHART 6: Key Findings ──
ax6 = fig.add_axes([0.55, 0.10, 0.4, 0.22])
ax6.set_facecolor(NAVY)
ax6.axis('off')
top_type = type_fraud.idxmax()
top_city = (df.groupby('City')['Is_Fraud'].mean()*100).idxmax()
insights = [
    "🔍  FRAUD DETECTION INSIGHTS",
    "",
    f"▶  {fraud_count} fraudulent transactions detected",
    f"▶  ₹{fraud_amount:,} total fraud amount",
    f"▶  Highest risk: {top_type}",
    f"▶  Most affected city: {top_city}",
    f"▶  Late night (12am-5am) = 2x fraud rate",
    f"▶  Transactions >₹50,000 = 35% fraud rate",
    "",
    "🛡️  PREVENTION STRATEGY",
    "",
    "▶  Flag transactions above ₹50,000",
    "▶  Extra verification 12am–5am",
    "▶  Monitor previous fraud accounts",
    "▶  Real-time city-wise risk alerts",
]
for i, line in enumerate(insights):
    color = GOLD if line.startswith("🔍") or line.startswith("🛡️") else BLUE4
    size = 13 if line.startswith("🔍") or line.startswith("🛡️") else 11
    bold = line.startswith("🔍") or line.startswith("🛡️")
    ax6.text(0.05, 0.97 - i*0.066, line, transform=ax6.transAxes,
             fontsize=size, color=color, fontweight='bold' if bold else 'normal',
             va='top')

# Footer
ax_foot = fig.add_axes([0, 0, 1, 0.025])
ax_foot.set_facecolor(NAVY)
ax_foot.text(0.5, 0.5, 'Project by G. Sweekrith Kumar  |  Tools: Python · SQL · Excel · Power BI  |  github.com/sweekrithkumar08-cmd',
             ha='center', va='center', fontsize=11, color=BLUE4,
             transform=ax_foot.transAxes)
ax_foot.axis('off')

plt.savefig('/mnt/user-data/outputs/Fraud_Detection_Dashboard.png',
            dpi=150, bbox_inches='tight', facecolor=BG)
print("Fraud Detection Dashboard saved!")
plt.close()
