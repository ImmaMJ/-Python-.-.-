import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Загружаем файл HR.csv в pandas dataframe
print("1. ЗАГРУЗКА ДАННЫХ")
df = pd.read_csv('HR.csv')
print(f"Данные загружены. Размер: {df.shape}")
print(df.head())

# 2. Рассчитываем основные статистики
print("\n2. ОСНОВНЫЕ СТАТИСТИКИ")
numeric_columns = ['satisfaction_level', 'last_evaluation', 'number_project', 
                   'average_montly_hours', 'time_spend_company']

stats_results = {}
for col in numeric_columns:
    stats_results[col] = {
        'Среднее': df[col].mean(),
        'Медиана': df[col].median(),
        'Мода': df[col].mode()[0],
        'Минимум': df[col].min(),
        'Максимум': df[col].max(),
        # mad() недоступен в моей версии, рассчитаем вручную
        'Среднее отклонение': (df[col] - df[col].mean()).abs().mean()
    }

stats_df = pd.DataFrame(stats_results).T
print(stats_df)

# 3. Корреляционная матрица для количественных переменных
print("\n3. КОРРЕЛЯЦИОННАЯ МАТРИЦА")
correlation_matrix = df[numeric_columns].corr()

# Визуализация
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.3f')
plt.title('Корреляционная матрица количественных переменных')
plt.tight_layout()
plt.show()

# Находим самые скоррелированные и наименее скоррелированные переменные
corr_values = correlation_matrix.unstack()
# Убираем диагональные элементы (корреляция переменной с самой собой = 1)
corr_values = corr_values[corr_values.index.get_level_values(0) != corr_values.index.get_level_values(1)]

# Самые скоррелированные (по абсолютному значению)
most_correlated = corr_values.abs().sort_values(ascending=False).head(2)
print("Две самые скоррелированные переменные:")
for idx in most_correlated.index:
    var1, var2 = idx
    print(f"{var1} и {var2}: {correlation_matrix.loc[var1, var2]:.3f}")

# Наименее скоррелированные
least_correlated = corr_values.abs().sort_values(ascending=True).head(2)
print("\nДве наименее скоррелированные переменные:")
for idx in least_correlated.index:
    var1, var2 = idx
    print(f"{var1} и {var2}: {correlation_matrix.loc[var1, var2]:.3f}")

# 4. Количество сотрудников по департаментам
print("\n4. СОТРУДНИКИ ПО ДЕПАРТАМЕНТАМ")
department_counts = df['department'].value_counts()
print(department_counts)

# 5. Распределение сотрудников по зарплатам
print("\n5. РАСПРЕДЕЛЕНИЕ ПО ЗАРПЛАТАМ")
salary_distribution = df['salary'].value_counts()
print(salary_distribution)

plt.figure(figsize=(8, 6))
salary_distribution.plot(kind='bar', color=['skyblue', 'lightgreen', 'salmon'])
plt.title('Распределение сотрудников по зарплатам')
plt.xlabel('Уровень зарплаты')
plt.ylabel('Количество сотрудников')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 6. Распределение по зарплатам в каждом департаменте
print("\n6. ЗАРПЛАТЫ ПО ДЕПАРТАМЕНТАМ")
salary_by_department = pd.crosstab(df['department'], df['salary'])
print(salary_by_department)

plt.figure(figsize=(12, 8))
salary_by_department.plot(kind='bar', stacked=True)
plt.title('Распределение зарплат по департаментам')
plt.xlabel('Департамент')
plt.ylabel('Количество сотрудников')
plt.xticks(rotation=45)
plt.legend(title='Зарплата')
plt.tight_layout()
plt.show()

# 7. Проверка гипотезы: сотрудники с высоким окладом проводят на работе больше времени
print("\n7. ПРОВЕРКА ГИПОТЕЗЫ")
high_salary_hours = df[df['salary'] == 'high']['average_montly_hours']
low_salary_hours = df[df['salary'] == 'low']['average_montly_hours']

# T-тест для проверки статистической значимости
t_stat, p_value = stats.ttest_ind(high_salary_hours, low_salary_hours, equal_var=False)

print(f"Среднее время работы сотрудников с high зарплатой: {high_salary_hours.mean():.1f} часов")
print(f"Среднее время работы сотрудников с low зарплатой: {low_salary_hours.mean():.1f} часов")
print(f"t-статистика: {t_stat:.3f}")
print(f"p-value: {p_value:.3f}")

if p_value < 0.05 and high_salary_hours.mean() > low_salary_hours.mean():
    print("Гипотеза ПОДТВЕРЖДЕНА: сотрудники с высоким окладом проводят на работе больше времени")
else:
    print("Гипотеза НЕ ПОДТВЕРЖДЕНА")

# 8. Показатели среди уволившихся и не уволившихся
print("\n8. СРАВНЕНИЕ УВОЛИВШИХСЯ И НЕ УВОЛИВШИХСЯ")
left_employees = df[df['left'] == 1]
stayed_employees = df[df['left'] == 0]

comparison = pd.DataFrame({
    'Уволились': [
        left_employees['promotion_last_5years'].mean(),
        left_employees['satisfaction_level'].mean(),
        left_employees['number_project'].mean()
    ],
    'Не уволились': [
        stayed_employees['promotion_last_5years'].mean(),
        stayed_employees['satisfaction_level'].mean(),
        stayed_employees['number_project'].mean()
    ]
}, index=['Доля с повышением', 'Средняя удовлетворенность', 'Среднее количество проектов'])

print(comparison)

# 9. Разделение данных на тестовую и обучающую выборки
print("\n9. РАЗДЕЛЕНИЕ ДАННЫХ")
# Используем все факторы кроме department и salary
features = ['satisfaction_level', 'last_evaluation', 'number_project', 
            'average_montly_hours', 'time_spend_company', 'Work_accident', 
            'promotion_last_5years']

X = df[features]
y = df['left']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Обучающая выборка: {X_train.shape[0]} записей")
print(f"Тестовая выборка: {X_test.shape[0]} записей")

# 10. Построение LDA модели
print("\n10. ПОСТРОЕНИЕ LDA МОДЕЛИ")
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# 11. Оценка качества модели на тестовой выборке
print("\n11. ОЦЕНКА КАЧЕСТВА МОДЕЛИ")
y_pred = lda.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Точность (accuracy) модели: {accuracy:.3f}")

print("\nМатрица ошибок:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nОтчет по классификации:")
print(classification_report(y_test, y_pred))

print("\n" + "="*50)
print("ВСЕ ПУНКТЫ ЗАДАНИЯ ВЫПОЛНЕНЫ!")
print("="*50)