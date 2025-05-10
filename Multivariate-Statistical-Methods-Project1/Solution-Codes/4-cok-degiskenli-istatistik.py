from ISLP import load_data
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.regressionplots import influence_plot

# Renk paleti ve görselleştirme
colors = {
    'normal': 'blue',
    'outlier': 'red',
    'high_leverage': 'orange',
    'high_influence': 'purple',
    'threshold_line': 'red',
    'zero_line': 'grey'
}

# Görselleştirme ayarları
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Veri hazırlığı ve model oluşturma
Carseats = load_data('Carseats')

# Kategorik değişkenleri dummy değişkenlere dönüştür (1 for Yes, 0 for No)
Carseats['Urban'] = Carseats['Urban'].map({'Yes': 1, 'No': 0})
Carseats['US'] = Carseats['US'].map({'Yes': 1, 'No': 0})

# (a) İlk model için veri hazırlama
X_full = Carseats[['Price', 'Urban', 'US']]
# Bağımlı değişken (hedef değişken) tanımlanıyor
y = Carseats['Sales']
# OLS regresyonunun kesişme noktasını tahmin etmek için sabit terim ekleme
X_full = sm.add_constant(X_full)  # Sabit terim ekleme

# (a) Tam model kurma
model_full = sm.OLS(y, X_full).fit()

# (b) Katsayıların yorumlanması için fonksiyon
def interpret_coefficients(model):
    """Regresyon katsayılarını yorumla"""
    
    print("\n(b) Katsayıların Yorumlanması:")
    
    # Sabit terim
    const = model.params['const']
    print(f"Sabit Terim ({const:.4f}): Tüm bağımsız değişkenler sıfır olduğunda (Price=0, Urban=0, US=0),")
    print(f"  yani fiyat sıfır olduğunda, kırsal bölgede ve ABD dışında ortalama satış {const:.2f} birimdir.")
    
    # Price katsayısı
    price_coef = model.params['Price']
    price_p = model.pvalues['Price']
    sig_text = "istatistiksel olarak anlamlıdır (p < 0.05)" if price_p < 0.05 else "istatistiksel olarak anlamlı değildir (p >= 0.05)"
    print(f"Price Katsayısı ({price_coef:.4f}): Diğer değişkenler sabit tutulduğunda, fiyattaki 1 birimlik artış,")
    print(f"  satışlarda ortalama {abs(price_coef):.2f} birimlik bir {('artışa' if price_coef > 0 else 'azalışa')} neden olur.")
    print(f"  Bu etki {sig_text}.")
    
    # Urban katsayısı (eğer modelde varsa)
    if 'Urban' in model.params:
        urban_coef = model.params['Urban']
        urban_p = model.pvalues['Urban']
        sig_text = "istatistiksel olarak anlamlıdır (p < 0.05)" if urban_p < 0.05 else "istatistiksel olarak anlamlı değildir (p >= 0.05)"
        direction = "yüksek" if urban_coef > 0 else "düşük"
        print(f"Urban Katsayısı ({urban_coef:.4f}): Diğer değişkenler sabit tutulduğunda, kentsel bölgedeki satışlar,")
        print(f"  kırsal bölgeye göre ortalama {abs(urban_coef):.2f} birim daha {direction}tir.")
        print(f"  Bu etki {sig_text}.")
    
    # US katsayısı (eğer modelde varsa)
    if 'US' in model.params:
        us_coef = model.params['US']
        us_p = model.pvalues['US']
        sig_text = "istatistiksel olarak anlamlıdır (p < 0.05)" if us_p < 0.05 else "istatistiksel olarak anlamlı değildir (p >= 0.05)"
        direction = "yüksek" if us_coef > 0 else "düşük"
        print(f"US Katsayısı ({us_coef:.4f}): Diğer değişkenler sabit tutulduğunda, ABD içindeki satışlar,")
        print(f"  ABD dışına göre ortalama {abs(us_coef):.2f} birim daha {direction}tir.")
        print(f"  Bu etki {sig_text}.")

# Katsayıları yorumla
interpret_coefficients(model_full)

# (c) Model denklemi
print("\n(c) Model Denklemi:")
b0 = model_full.params['const']  # sabit terim
b1 = model_full.params['Price']  # Price katsayısı
b2 = model_full.params['Urban']  # Urban katsayısı
b3 = model_full.params['US']     # US katsayısı
print(f"Sales = {b0:.4f} + ({b1:.4f} × Price) + ({b2:.4f} × Urban) + ({b3:.4f} × US)")
print("Burada:")
print("Urban = 1 (Kentsel bölge), Urban = 0 (Kırsal bölge, referans kategori)")
print("US = 1 (ABD içinde), US = 0 (ABD dışında, referans kategori)")

# (d) Hipotez testleri ve anlamlı değişkenleri belirleme
print("\n(d) Hipotez Testleri:")
print("Sıfır Hipotezi (H0): βj = 0 (Değişkenin etkisi yoktur)")
print("Alternatif Hipotez (H1): βj ≠ 0 (Değişkenin etkisi vardır)")
print("\np-değerleri:")
for var, p_val in model_full.pvalues.items():
    if var != 'const':
        result = "REDDEDİLİR (Anlamlı etki vardır)" if p_val < 0.05 else "REDDEDİLEMEZ (Anlamlı etki yoktur)"
        print(f"{var}: {p_val:.4f} -> H0 {result}")

# Anlamlı değişkenleri belirleme
alpha = 0.05
anlamli_degiskenler = []
for var, p_val in model_full.pvalues.items():
    if p_val < alpha and var != 'const':
        anlamli_degiskenler.append(var)

print("\nAnlamlı değişkenler:", anlamli_degiskenler)

# (e) Sadeleştirilmiş model kurma
X = Carseats[anlamli_degiskenler]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

print("\n(e) Sadeleştirilmiş Model:")
print(f"Sadeleştirilmiş model denklemi:")
b0_simple = model.params['const']  # sabit terim
terms = []
for var in anlamli_degiskenler:
    coef = model.params[var]
    terms.append(f"({coef:.4f} × {var})")
equation = f"Sales = {b0_simple:.4f} + " + " + ".join(terms)
print(equation)

# (f) Model karşılaştırması
print("\n(f) Model Karşılaştırması:")
print(f"Tam Model R²: {model_full.rsquared:.4f}")
print(f"Tam Model Düzeltilmiş R²: {model_full.rsquared_adj:.4f}")
print(f"Sadeleştirilmiş Model R²: {model.rsquared:.4f}")
print(f"Sadeleştirilmiş Model Düzeltilmiş R²: {model.rsquared_adj:.4f}")

# Modeller arasındaki farkı yorumla
r2_diff = model_full.rsquared - model.rsquared
adj_r2_diff = model_full.rsquared_adj - model.rsquared_adj
print("\nYorum:")
if abs(adj_r2_diff) < 0.01:
    print("Sadeleştirilmiş model, tam model kadar iyi performans göstermektedir.")
    print("Anlamlı olmayan değişkenlerin çıkarılması modelin açıklama gücünü önemli ölçüde etkilememiştir.")
    if adj_r2_diff < 0:
        print("Sadeleştirilmiş modelin düzeltilmiş R² değeri daha yüksektir, bu da daha iyi bir model olduğunu gösterir.")
else:
    if adj_r2_diff > 0:
        print("Tam model daha yüksek açıklama gücüne sahiptir.")
    else:
        print("Sadeleştirilmiş model daha yüksek açıklama gücüne sahiptir.")

# (g) Güven aralıkları
conf_int = model.conf_int(alpha=0.05)
conf_int.columns = ['%95 Alt Sınır', '%95 Üst Sınır']

print("\n(g) %95 Güven Aralıkları:")
print(conf_int)

# Güven aralıklarını yorumla
print("\nYorum:")
for var in model.params.index:
    lower = conf_int.loc[var, '%95 Alt Sınır']
    upper = conf_int.loc[var, '%95 Üst Sınır']
    print(f"{var} katsayısının %95 güven aralığı: [{lower:.4f}, {upper:.4f}]")
    if lower * upper > 0:  # İşaretler aynı, yani sıfırı kapsamıyor
        direction = "pozitif" if lower > 0 else "negatif"
        print(f"  Bu aralık sıfırı içermediği için {var} değişkeninin satışlar üzerinde anlamlı bir {direction} etkisi vardır.")
    else:
        print(f"  Bu aralık sıfırı içerdiği için {var} değişkeninin satışlar üzerinde anlamlı bir etkisi olduğu söylenemez.")

# Diagnostik değerleri hesaplama
infl = model.get_influence()
student_resid = infl.resid_studentized_internal
cooks_d = infl.cooks_distance[0]
leverage = infl.hat_matrix_diag

# Eşik değerleri
n = len(y)  # gözlem sayısı
k = len(model.params)  # parametre sayısı
outlier_threshold = 3
leverage_threshold = 2 * (k + 1) / n
cooks_threshold = 4 / n

# Önemli gözlemleri belirleme
high_leverage = np.where(leverage > leverage_threshold)[0]
outliers = np.where(np.abs(student_resid) > outlier_threshold)[0]
high_influence = np.where(cooks_d > cooks_threshold)[0]
combined_indices = list(set(list(high_leverage) + list(outliers) + list(high_influence)))

# 1. Q-Q Plot
plt.figure(figsize=(10, 8))
sm.graphics.qqplot(student_resid, line='45', fit=True)
plt.title('Q-Q Plot (Standardize Edilmiş Artıklar)', fontweight='bold', fontsize=16)
plt.xlabel('Teorik Kantiller', fontsize=14)
plt.ylabel('Gözlenen Kantiller', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig('qq_plot.png', dpi=300, bbox_inches='tight')
plt.show() 

# 2. Residuals vs Fitted Values Plot
plt.figure(figsize=(10, 8))
plt.scatter(
    model.fittedvalues, student_resid, 
    edgecolor='k', alpha=0.7,
    color=colors['normal'],
    label='Normal Gözlemler'
)
plt.axhline(y=0, color=colors['zero_line'], linestyle='-')
plt.axhline(y=outlier_threshold, color=colors['threshold_line'], linestyle='--')
plt.axhline(y=-outlier_threshold, color=colors['threshold_line'], linestyle='--')
plt.xlabel('Tahmin Edilen Değerler', fontsize=14)
plt.ylabel('Standardize Edilmiş Artıklar', fontsize=14)
plt.title('Tahmin Değerlerine Karşı Artıklar', fontweight='bold', fontsize=16)
plt.grid(True)

# Aykırı değerleri etiketle ve farklı renkte göster
if len(outliers) > 0:
    plt.scatter(
        model.fittedvalues[outliers], student_resid[outliers],
        color=colors['outlier'], edgecolor='k', alpha=0.7, s=80,
        label='Aykırı Değerler'
    )
    
    for i in outliers:
        plt.annotate(
            i, 
            xy=(model.fittedvalues[i], student_resid[i]),
            xytext=(5, 5), 
            textcoords='offset points',
            arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
            fontweight='bold',
            fontsize=10
        )

# Her zaman bir lejant olmasını sağla
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('residuals_vs_fitted.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Leverage vs Studentized Residuals Plot
plt.figure(figsize=(10, 8))
plt.scatter(
    leverage, student_resid, 
    edgecolor='k', alpha=0.7,
    color=colors['normal'],
    label='Normal Gözlemler'
)
plt.axhline(y=0, color=colors['zero_line'], linestyle='-')
plt.axhline(y=outlier_threshold, color=colors['threshold_line'], linestyle='--')
plt.axhline(y=-outlier_threshold, color=colors['threshold_line'], linestyle='--')
plt.axvline(x=leverage_threshold, color=colors['threshold_line'], linestyle='--')
plt.xlabel('Kaldıraç (Leverage)', fontsize=14)
plt.ylabel('Standardize Edilmiş Artıklar', fontsize=14)
plt.title('Kaldıraç vs Standardize Edilmiş Artıklar', fontweight='bold', fontsize=16)
plt.grid(True)

# ÖNEMLİ: Önce verileri çiz, sonra eksen sınırlarını ayarla
x_min, x_max = min(leverage), max(leverage)
y_min, y_max = min(student_resid), max(student_resid)
# Biraz marj ekle
margin_x = (x_max - x_min) * 0.05
margin_y = (y_max - y_min) * 0.05
plt.xlim(x_min - margin_x, x_max + margin_x)
plt.ylim(y_min - margin_y, y_max + margin_y)

# Eşik değer açıklamaları 
plt.text(
    leverage_threshold + margin_x, 
    y_min + margin_y, 
    f'Kaldıraç Eşiği: {leverage_threshold:.3f}',
    color='red', 
    fontweight='bold'
)
plt.text(
    x_min + margin_x, 
    outlier_threshold - margin_y/2, 
    f'Artık Eşiği: ±{outlier_threshold}',
    color='red', 
    fontweight='bold'
)

# Farklı kategorilerdeki gözlemleri belirle ve göster
if len(outliers) > 0:
    plt.scatter(
        leverage[outliers], student_resid[outliers],
        color=colors['outlier'], edgecolor='k', alpha=0.7, s=80,
        label='Aykırı Değerler'
    )

if len(high_leverage) > 0:
    plt.scatter(
        leverage[high_leverage], student_resid[high_leverage],
        color=colors['high_leverage'], edgecolor='k', alpha=0.7, s=80,
        label='Yüksek Kaldıraç Noktaları'
    )

if len(high_influence) > 0:
    plt.scatter(
        leverage[high_influence], student_resid[high_influence],
        color=colors['high_influence'], edgecolor='k', alpha=0.7, s=100,
        label='Yüksek Etkili Gözlemler'
    )

# Önemli noktaları etiketle
for i in combined_indices:
    plt.annotate(
        i, 
        xy=(leverage[i], student_resid[i]),
        xytext=(5, 5), 
        textcoords='offset points',
        arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
        fontweight='bold',
        fontsize=10
    )

plt.legend(loc='best')
plt.tight_layout()
plt.savefig('leverage_vs_residuals.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Cook's Distance Plot
plt.figure(figsize=(10, 8))
plt.stem(
    np.arange(len(cooks_d)), cooks_d, 
    markerfmt="o", basefmt=" ",
    linefmt='-'
)
plt.axhline(y=cooks_threshold, color=colors['threshold_line'], linestyle='--', label=f'Eşik (4/n): {cooks_threshold:.4f}')
plt.xlabel('Gözlem Numarası', fontsize=14)
plt.ylabel("Cook's Distance", fontsize=14)
plt.title("Cook's Distance Plot", fontweight='bold', fontsize=16)
plt.grid(True)
plt.legend()

# Yüksek etkili gözlemleri etiketle
for i in high_influence:
    plt.annotate(
        i, 
        xy=(i, cooks_d[i]),
        xytext=(5, 5), 
        textcoords='offset points',
        arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
        fontweight='bold',
        fontsize=10
    )

plt.tight_layout()
plt.savefig('cooks_distance.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Influence Plot
plt.figure(figsize=(10, 8))
ax = plt.gca() 

# Influence plot çiz
influence_plot(
    model, 
    criterion="cooks",
    external=True,
    alpha=0.75,
    ax=ax,
    plot_alpha=0.75
)

# İşlemi basitleştir
ax.grid(True)
ax.axhline(y=0, color='grey', linestyle='-', alpha=0.3)
ax.axhline(y=outlier_threshold, color=colors['threshold_line'], linestyle='--', alpha=0.7)
ax.axhline(y=-outlier_threshold, color=colors['threshold_line'], linestyle='--', alpha=0.7)
ax.axvline(x=leverage_threshold, color=colors['threshold_line'], linestyle='--', alpha=0.7)

# Eksen sınırlarını belirle 
x_buffer = max(leverage) * 0.1
y_buffer = max(abs(min(student_resid)), abs(max(student_resid))) * 0.1
ax.set_xlim(0, max(leverage) + x_buffer)
ax.set_ylim(min(student_resid) - y_buffer, max(student_resid) + y_buffer)

# Eşik açıklamalarını sabit konumlara yerleştir
ax.text(
    leverage_threshold + 0.01, 
    min(student_resid) + y_buffer,
    f'Kaldıraç Eşiği: {leverage_threshold:.3f}',
    color='red', 
    fontweight='bold'
)
ax.text(
    0.01, 
    outlier_threshold - y_buffer/2,
    f'Artık Eşiği: ±{outlier_threshold}',
    color='red', 
    fontweight='bold'
)

# Önceki etiketleri temizle
for text in ax.texts:
    text.remove()

# Sadece önemli noktaları etiketle
for i in combined_indices:
    ax.annotate(
        i, 
        xy=(leverage[i], student_resid[i]),
        xytext=(5, 5), 
        textcoords='offset points',
        arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
        fontweight='bold',
        fontsize=10
    )

plt.title('Influence Plot (Etki Grafiği)', fontsize=16, fontweight='bold')
plt.xlabel('Kaldıraç (Leverage)', fontsize=14)
plt.ylabel('Studentized Artıklar', fontsize=14)

# Lejant ekle
plt.plot([], [], 'o', color='red', alpha=0.7, markersize=10, label='Yüksek Etkili Noktalar')
plt.plot([], [], 'o', color='blue', alpha=0.7, markersize=5, label='Normal Gözlemler')
plt.legend(loc='best', fontsize=12)

plt.tight_layout()
plt.savefig('influence_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Sonuçları raporla
print("\n(a) Tam Model Özeti:")
print(model_full.summary())

print("\n(h) Diagnostik Analiz Özeti:")
print(f"Toplam gözlem sayısı: {n}")
print(f"Kaldıraç eşik değeri (2(k+1)/n): {leverage_threshold:.4f}")
print(f"Cook's Distance eşik değeri (4/n): {cooks_threshold:.4f}")
print(f"Studentized artık eşik değeri: ±{outlier_threshold}")

print("\nDetaylı Analiz:")
if len(outliers) > 0:
    print(f"\nAykırı değerler (Studentized Artık > {outlier_threshold}):")
    for i in outliers:
        print(f"Gözlem {i}: Artık = {student_resid[i]:.4f}, Kaldıraç = {leverage[i]:.4f}, Cook's D = {cooks_d[i]:.4f}")
else:
    print("\nAykırı değer bulunamadı.")

if len(high_leverage) > 0:
    print(f"\nYüksek kaldıraç noktaları (Leverage > {leverage_threshold:.4f}):")
    for i in high_leverage:
        print(f"Gözlem {i}: Artık = {student_resid[i]:.4f}, Kaldıraç = {leverage[i]:.4f}, Cook's D = {cooks_d[i]:.4f}")
else:
    print("\nYüksek kaldıraç noktası bulunamadı.")

if len(high_influence) > 0:
    print(f"\nYüksek etkiye sahip gözlemler (Cook's D > {cooks_threshold:.4f}):")
    for i in high_influence:
        print(f"Gözlem {i}: Artık = {student_resid[i]:.4f}, Kaldıraç = {leverage[i]:.4f}, Cook's D = {cooks_d[i]:.4f}")
    
    print("\nYüksek etkili gözlemlerin değerleri:")
    relevant_cols = ['Sales'] + anlamli_degiskenler
    print(Carseats.iloc[high_influence][relevant_cols])
else:
    print("\nYüksek etkiye sahip gözlem bulunamadı.")