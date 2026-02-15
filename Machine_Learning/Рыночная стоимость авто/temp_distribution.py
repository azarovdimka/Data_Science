# Распределение числовых признаков
numerical_features = ['Price', 'RegistrationYear', 'Power', 'Kilometer', 'RegistrationMonth']

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, col in enumerate(numerical_features):
    df[col].hist(bins=50, ax=axes[i], edgecolor='black')
    axes[i].set_title(f'Распределение: {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Частота')

axes[-1].axis('off')  # Скрыть последний пустой график
plt.tight_layout()
plt.show()
