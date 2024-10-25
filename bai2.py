#add library
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from sklearn.metrics import confusion_matrix
from scipy.stats import norm, skew, probplot
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import optuna   
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score


data = pd.read_csv('C:/Users/PC/Downloads/ad_click_dataset.csv')
print(data)

data = data.drop(columns=['id','full_name'], axis=1)

print(data.shape)

print(data.info())

num_cols = data.select_dtypes(include=['float64', 'int64'])
cat_cols = data.select_dtypes(include=['object'])

print('Numeric Variables:')
print(num_cols.columns.tolist())

print("\nCategorical Variables:")
print(cat_cols.columns.tolist())

for col in cat_cols:
        print('We Have {} Unique values. Values in the {} Column: {}'.format(len(data[col].unique()),col,data[col].unique()))
        print('__'*30)

print(data.describe().T)

for feature in num_cols:
    zero_values = (data[feature] == 0).sum()
    null_values = data[feature].isnull().sum()
    unique_values = len(data[feature].unique())

    print(f"Feature: {feature}")
    print(f"Number of 0 Values: {zero_values}")
    print(f"Number of Null Values: {null_values}")
    print(f"Unique Values: {unique_values}")
    print("=" * 30)

print(data.isnull().sum())

#figure 1
plt.figure(figsize=(20,6))
plt.title('Heatmap for the null values in each column')
sns.heatmap(data.isnull(),cmap='viridis')
plt.show()
#Đoạn mã trên tạo ra một biểu đồ nhiệt lớn, trực quan hóa sự phân bố của các giá trị null trong các cột của một DataFrame.
#Các ô màu sáng sẽ chỉ ra rằng có giá trị null trong các cột tương ứng, trong khi các ô màu tối sẽ cho thấy không có giá trị null.

#Thay thế các giá trị null trong các cột bằng từ 'Unknown'
data['gender'] = data['gender'].fillna('Unknown')
data['device_type'] = data['device_type'].fillna('Unknown')
data['ad_position'] = data['ad_position'].fillna('Unknown')
data['browsing_history'] = data['browsing_history'].fillna('Unknown')
data['time_of_day'] = data['time_of_day'].fillna('Unknown')

#figure 2
plt.hist(data['age'], bins=20, edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


def knn_impute(data, n_neighbors=5):
    data_encoded = data.copy()

    category_mappings = {}
    for col in data_encoded.select_dtypes(include='object').columns:
        data_encoded[col] = data_encoded[col].astype('category').cat.codes
        category_mappings[col] = dict(enumerate(data[col].astype('category').cat.categories))

    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    data_imputed = pd.DataFrame(knn_imputer.fit_transform(data_encoded), columns=data_encoded.columns)

    for col in data.select_dtypes(include='object').columns:
        data_imputed[col] = data_imputed[col].round().astype(int).map(category_mappings[col])

    return data_imputed


data_imputed = knn_impute(data, n_neighbors=5)
data = data_imputed
#Đoạn mã trên định nghĩa một hàm để lấp đầy các giá trị thiếu trong một DataFrame bằng cách sử dụng phương pháp KNN.
#Hàm này mã hóa các cột phân loại, áp dụng KNNImputer để lấp đầy các giá trị thiếu.

print(data.isnull().sum())

#Câu lệnh trên cho bạn biết tổng số giá trị null trong từng cột của DataFrame

#figure 3
plt.hist(data['age'], bins=20, edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

#Việc tìm hiểu mối tương quan giữa các biến giúp bạn xác định những yếu tố nào có thể ảnh hưởng đến hành vi nhấp chuột (click).
corr = num_cols.corr()
top_corr = corr['click'].sort_values(ascending=False)[1:20].to_frame()
styled_corr = top_corr.style.background_gradient(axis=1, cmap=sns.light_palette('green', as_cmap=True))
print(top_corr)

#figure 4
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

#figure 5
data_encoded = pd.get_dummies(data, drop_first=True)
corr_matrix = data_encoded.corr()

# Heatmap
plt.figure(figsize=(20, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix including Categorical Features')
plt.show()

#Table_click
top_corr = corr_matrix['click'].sort_values(ascending=False)[1:20].to_frame()
styled_corr = top_corr.style.background_gradient(axis=1, cmap=sns.light_palette('green', as_cmap=True))
print(top_corr)

#figure 6
plt.figure(figsize=(8, 5))
sns.boxplot(data=data, x="age", color='skyblue', flierprops=dict(marker='o', markersize=8, markerfacecolor='red'))
plt.title('Age Distribution Box Plot')
plt.xlabel('age')
plt.ylabel('Value')
median_age = data['age'].median()
plt.axvline(median_age, color='green', linestyle='--', linewidth=2, label='Median')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Tim max-min age
min_age = data['age'].min()
max_age = data['age'].max()

print(f"Minimum age value: {min_age}")
print(f"Maximum age value: {max_age}")

data['age'] = data['age'].astype(int)
data['click'] = data['click'].astype(int)

#figure 7
data['click'].value_counts().plot(kind='bar', color='darkblue')
plt.xlabel('Click Values')
plt.ylabel('Frequency')
plt.title('Bar Chart of Click')
plt.show()

#Thong ke % click
distribution_click = data['click'].value_counts(normalize=True) * 100

print(distribution_click)

#figure 8
plt.hist(data['age'], bins=30, color='darkblue', edgecolor='yellow')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

#figure 9
# Defining age group boundaries
bins = [17, 24, 34, 44, 54, 64, 100]
labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']

grouped = (
    data.assign(age_group=pd.cut(data['age'], bins=bins, labels=labels))
    .groupby(['age_group', 'click'], observed=False)
    .size()
    .unstack(fill_value=0)
)

percentages = grouped.div(grouped.sum(axis=1), axis=0) * 100
plt.figure(figsize=(12, 6))
ax = grouped.plot(kind='bar', stacked=True)
plt.title('Click Status by Age Group')
plt.xlabel('Age Groups')
plt.ylabel('Number of Clicks')
plt.xticks(rotation=0)
plt.legend(['Did Not Click', 'Clicked'])

# Adding percentage labels on the bars
for i in range(len(grouped)):
    total = grouped.iloc[i].sum()
    for j in range(len(grouped.columns)):
        if total > 0:
            percentage = (grouped.iloc[i, j] / total) * 100
            ax.text(i, grouped.iloc[i, :j+1].sum() - grouped.iloc[i, j] / 2, f'{percentage:.1f}%',
                    ha='center', va='center', color='white', fontsize=8)

plt.show()


#figure 10
click_counts = data[data['click'] == 1]['gender'].value_counts(normalize=True) * 100
ax = click_counts.plot(kind='bar', color='darkblue')
plt.xlabel('Gender Values')
plt.ylabel('Frequency')
plt.title('Bar Chart of Gender')

for i in range(len(click_counts)):
    percentage = click_counts.iloc[i]
    ax.text(i, percentage / 2, f'{percentage:.1f}%',
            ha='center', va='center', color='white', fontsize=10)

plt.show()

#Đếm số lần xuất hiện của từng loại thiết bị trong cột device_type
print(data['device_type'].value_counts())

#figure 11 : Thong ke cot device_type
click_counts = data[data['click'] == 1]['device_type'].value_counts(normalize=True) * 100
ax = click_counts.plot(kind='bar', color='darkblue')
plt.xlabel('Device Types')
plt.ylabel('Click Percentage')
plt.title('Percentage of Clicks by Device Type')

for i in range(len(click_counts)):
    percentage = click_counts.iloc[i]
    ax.text(i, percentage / 2, f'{percentage:.1f}%',
            ha='center', va='center', color='white', fontsize=10)

plt.show()

# Đếm số lượng trong cột ad_position
print(data['ad_position'].value_counts())

#figure 12 : Thong ke cot ad_position
click_counts = data[data['click'] == 1]['ad_position'].value_counts(normalize=True) * 100
ax = click_counts.plot(kind='bar', color='darkblue')
plt.xlabel('Ad Positions')
plt.ylabel('Frequency')
plt.title('Bar Chart of Ad Positions')

for i in range(len(click_counts)):
    percentage = click_counts.iloc[i]
    ax.text(i, percentage / 2, f'{percentage:.1f}%',
            ha='center', va='center', color='white', fontsize=10)

plt.show()

# Đếm số lượng trong cột browsing_history
print(data['browsing_history'].value_counts())

#figure 13 : Thong ke cot browsing_history
click_counts = data[data['click'] == 1]['browsing_history'].value_counts(normalize=True) * 100
ax = click_counts.plot(kind='bar', color='darkblue')
plt.xlabel('Browsing History')
plt.ylabel('Frequency')
plt.title('Bar Chart of Browsing History')

for i in range(len(click_counts)):
    percentage = click_counts.iloc[i]
    ax.text(i, percentage / 2, f'{percentage:.1f}%',
            ha='center', va='center', color='white', fontsize=10)

plt.show()

#Dêm so luong cọt time_of_day
print(data['time_of_day'].value_counts())

#figure 14 : Thong ke cot time_of_day
click_counts = data[data['click'] == 1]['time_of_day'].value_counts(normalize=True) * 100
ax = click_counts.plot(kind='bar', color='darkblue')
plt.xlabel('Time of Day ')
plt.ylabel('Frequency')
plt.title('Bar Chart of Time of Day')

for i in range(len(click_counts)):
    percentage = click_counts.iloc[i]
    ax.text(i, percentage / 2, f'{percentage:.1f}%',
            ha='center', va='center', color='white', fontsize=10)

plt.show()

#figure 15 : Moi lien he giua ad_position va device_type
crosstab = pd.crosstab(data['device_type'], data['ad_position'])
ax = crosstab.plot(kind='bar', figsize=(10, 6), color=sns.color_palette("pastel"))

for p in ax.patches:
    ax.annotate(f'{p.get_height()}',
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom', fontsize=9)

plt.xlabel('Device Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Ad Position Count by Device Type', fontsize=14)
plt.legend(title='Ad Position', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

#figure 16 : device_type - age
sns.boxplot(x='device_type', y='age', hue='device_type', data=data, palette='Set3', legend=False)
plt.title('Age by Device Type')
plt.show()

#figure 17 : time_of_day và age
sns.boxplot(x='time_of_day', y='age', hue='time_of_day', data=data, palette='Set3')
plt.title('Age by Time of Day')
plt.xticks(rotation=45)
plt.show()

#Tinh do lech trong cot age
columns = ["age"]

skewed_feats = data[columns].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(15)

#Đếm các biến trong từng cột
for col in cat_cols:
        print('We Have {} Unique values. Values in the {} Column: {}'.format(len(data[col].unique()),col,data[col].unique()))
        print('__'*30)

#Chuyển giá trị thành dạng nhị phân để sử dụng machine_learning
data = pd.get_dummies(data, columns=cat_cols.columns, drop_first=True).astype(int)
#Check
print(data.info())

#Tao bản sao để dùng các models
KNN = data.copy()
XGBoost = data.copy()
LightGBM = data.copy()


#KNN
X = KNN.drop('click', axis=1)
y = KNN['click']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
smt=SMOTE()
X_train,y_train=smt.fit_resample(X_train,y_train)
#Finding the hyperparameter of KNN
# def objective(trial):
#     # Define hyperparameters
#     n_neighbors = trial.suggest_int('n_neighbors', 1, 20)  # Choose n_neighbors between 1 and 20
#     weights = trial.suggest_categorical('weights', ['uniform', 'distance'])  # Weight types
#     algorithm = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])  # Algorithm types

#     # Create and fit the KNN classifier
#     classifier_knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
#     classifier_knn.fit(X_train, y_train)

#     # Make predictions on the test data
#     y_pred = classifier_knn.predict(X_test)

#     # Calculate metrics
#     accuracy_knn = accuracy_score(y_test, y_pred)
#     precision_knn = precision_score(y_test, y_pred)
#     recall_knn = recall_score(y_test, y_pred)
#     f1_knn = f1_score(y_test, y_pred)

#     # Objective: Optimize the F1 score
#     return f1_knn  # We are trying to find the highest F1 score

# # Start the Optuna study
# study = optuna.create_study(direction='maximize')  # For maximum F1 score
# study.optimize(objective, n_trials=30)  # Run 100 trials

# # Print the best parameters and metrics
# print('Best parameters:', study.best_params)
# print('Best F1 Score:', study.best_value)

# # Evaluate with the best model
# best_params = study.best_params
# best_classifier_knn = KNeighborsClassifier(
#     n_neighbors=best_params['n_neighbors'],
#     weights=best_params['weights'],
#     algorithm=best_params['algorithm']
# )

# best_classifier_knn.fit(X_train, y_train)
# best_y_pred = best_classifier_knn.predict(X_test)

# # Print final metrics with the best model
# final_accuracy = accuracy_score(y_test, best_y_pred)
# final_precision = precision_score(y_test, best_y_pred)
# final_recall = recall_score(y_test, best_y_pred)
# final_f1 = f1_score(y_test, best_y_pred)

# print("Final Accuracy:", final_accuracy)
# print("Final Precision:", final_precision)
# print("Final Recall:", final_recall)
# print("Final F1 Score:", final_f1)

#Đoạn mã trên sử dụng Optuna để tìm kiếm các siêu tham số tốt nhất cho mô hình KNN nhằm tối đa hóa chỉ số F1
#Quá trình này bao gồm việc xác định các tham số, huấn luyện mô hình, đánh giá hiệu suất và in ra kết quả cuối cùng

classifier_knn = KNeighborsClassifier(n_neighbors = 8, weights='distance', algorithm = 'auto')

classifier_knn.fit(X_train, y_train)

cv_scores_knn = cross_val_score(classifier_knn, X, y, cv=5)

print("Cross-validation scores:", cv_scores_knn)
print("Mean CV accuracy:", cv_scores_knn.mean())

y_pred = classifier_knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

accuracy_knn = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy_knn)

precision_knn = precision_score(y_test, y_pred)
recall_knn = recall_score(y_test, y_pred)

print("Precision:", precision_knn)
print("Recall:", recall_knn)

f1_knn = f1_score(y_test, y_pred)
print("F1 Score:", f1_knn)

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred)
auc_score = auc(recall, precision)

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.plot(recall, precision, label=f'Precision-Recall Curve (AUC = {auc_score:.2f})', color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, label='ROC Curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()





