class OutlierRemover(BaseEstimator, TransformerMixin):
    """Класс для удаления выбросов из данных методом IQR только для train"""

    def __init__(self, columns: List[str] = None, factor=1.5, replace_with_nan=False):
        """Инициализация удалителя выбросов. Есть возможность указать столбцы, в которых убирать выбросы. По умолчанию размер услов 1.5 квартиля"""
        self.columns = columns
        self.factor = factor
        self.replace_with_nan = replace_with_nan
        self.bounds_dict = {}

    def fit_transform(self, X: pd.DataFrame, y: None = None, name='data', **fit_params) -> pd.DataFrame:
        """Обучает удалитель выбросов - описывает размер квантилей и границы усов. Удаляет выбросы"""

        if 'test' in name:
            print(f'- Удаление выбросов не проводится в тестовой выборке {name}, пропускаю шаг.')
            return X

        print('- Определяю границы выбросов методом IQR')
        
        df_transformed = X.copy()

        cols_to_process = self.columns if self.columns else df_transformed.select_dtypes(include=[np.number]).columns
        
        # Проверка на наличие выбросов
        outlier_cols = []
        for col in cols_to_process:
            if col in df_transformed.columns:
                Q1 = df_transformed[col].quantile(0.25)
                Q3 = df_transformed[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.factor * IQR
                upper_bound = Q3 + self.factor * IQR
                self.bounds_dict[col] = (lower_bound, upper_bound)
                
                outliers = (df_transformed[col] < lower_bound) | (df_transformed[col] > upper_bound)
                if outliers.any():
                    outlier_cols.append(col)
        
        if not outlier_cols:
            print('- Выбросы не обнаружены')
            return df_transformed
        
        print(f'- Обнаружены выбросы в столбцах: {outlier_cols}')
        
        # Показ боксплотов для столбцов с выбросами
        n_cols = len(outlier_cols)
        fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 4))
        if n_cols == 1:
            axes = [axes]
        
        for i, col in enumerate(outlier_cols):
            lower_bound, upper_bound = self.bounds_dict[col]
            print(f'Нормальные пределы для {col}: [{lower_bound:.2f} - {upper_bound:.2f}]')
            sns.boxplot(y=df_transformed[col], ax=axes[i])
            axes[i].set_title(f'Выбросы в {col}')
        
        plt.tight_layout()
        plt.show()
        
        # Обработка выбросов
        if self.replace_with_nan:
            print('- Заменяю выбросы на NaN\n')
            for col, (lower_bound, upper_bound) in self.bounds_dict.items():
                if col in df_transformed.columns:
                    outliers = (df_transformed[col] < lower_bound) | (df_transformed[col] > upper_bound)
                    df_transformed.loc[outliers, col] = np.nan
        else:
            print('- Выполняю удаление выбросов\n')
            for col, (lower_bound, upper_bound) in self.bounds_dict.items():
                if col in df_transformed.columns:
                    outliers = (df_transformed[col] < lower_bound) | (df_transformed[col] > upper_bound)
                    df_transformed = df_transformed[~outliers]
                
        return df_transformed