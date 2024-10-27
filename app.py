# Import thư viện
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.impute import KNNImputer

# Thiết lập tiêu đề ứng dụng
st.set_page_config(page_title="Phân Tích Dữ Liệu Click Quảng Cáo", layout="wide")

# CSS tùy chỉnh
st.markdown("""
    <style>
    .title {
        font-size: 50px; /* Tăng kích thước tiêu đề */
        color: #ff5733; /* Màu sắc nổi bật cho tiêu đề */
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
        transition: transform 0.3s; /* Thêm hiệu ứng chuyển động */
    }
    .title:hover {
        transform: scale(1.1); /* Hiệu ứng phóng to khi hover */
    }
    .subtitle {
        font-size: 30px; /* Tăng kích thước tiêu đề phụ */
        color: #333;
        font-weight: bold;
        margin: 15px 0;
    }
    .description {
        font-size: 20px;
        color: #555;
        margin: 10px 0;
    }
    .search-box {
        margin-bottom: 20px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Tiêu đề chính
st.markdown('<div class="title">Phân Tích Dữ Liệu Click Quảng Cáo</div>', unsafe_allow_html=True)



# Thanh tìm kiếm
search_term = st.text_input("Tìm kiếm thông tin trong dữ liệu:", "")


# Tải dữ liệu
data = pd.read_csv('./ad_click_dataset.csv')


# Tìm kiếm trong dữ liệu nếu có từ khóa
if search_term:
    st.subheader("Kết quả tìm kiếm")
    filtered_data = data[data.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)]
    st.write(filtered_data)
else:
    # Hiển thị dữ liệu
    st.subheader("Dữ liệu ban đầu")
    st.write(data)

# Loại bỏ các cột không cần thiết
data = data.drop(columns=['id', 'full_name'], axis=1)

# Thông tin dữ liệu
st.write(f"Kích thước dữ liệu: {data.shape}")
st.write(data.info())

# Phân loại các cột số và cột phân loại
num_cols = data.select_dtypes(include=['float64', 'int64'])
cat_cols = data.select_dtypes(include=['object'])

# Hiển thị biến số
st.subheader('Biến số:')
st.write(num_cols.columns.tolist())

# Hiển thị biến phân loại
st.subheader("Biến phân loại:")
st.write(cat_cols.columns.tolist())

# Mô tả dữ liệu
st.subheader("Mô tả dữ liệu")
st.write(data.describe().T)

# Thống kê giá trị 0, null và giá trị duy nhất
for feature in num_cols:
    zero_values = (data[feature] == 0).sum()
    null_values = data[feature].isnull().sum()
    unique_values = len(data[feature].unique())

    st.write(f"**Biến:** {feature}")
    st.write(f"Số lượng giá trị 0: {zero_values}")
    st.write(f"Số lượng giá trị null: {null_values}")
    st.write(f"Số lượng giá trị duy nhất: {unique_values}")
    st.write("=" * 30)

# Kiểm tra giá trị null
st.subheader("Kiểm tra giá trị null")
st.write(data.isnull().sum())

# Biểu đồ nhiệt cho giá trị null
st.subheader("Biểu đồ nhiệt cho các giá trị null")
fig, ax = plt.subplots(figsize=(20, 6))
ax.set_title('Biểu đồ nhiệt cho các giá trị null trong từng cột')
sns.heatmap(data.isnull(), ax=ax, cmap='viridis')
st.pyplot(fig)

# Thay thế các giá trị null
data['gender'] = data['gender'].fillna('Unknown')
data['device_type'] = data['device_type'].fillna('Unknown')
data['ad_position'] = data['ad_position'].fillna('Unknown')
data['browsing_history'] = data['browsing_history'].fillna('Unknown')
data['time_of_day'] = data['time_of_day'].fillna('Unknown')

# Biểu đồ phân phối tuổi
st.subheader("Phân phối tuổi")
fig, ax = plt.subplots()
ax.hist(data['age'], bins=20, edgecolor='black')
ax.set_title('Phân phối tuổi')
ax.set_xlabel('Tuổi')
ax.set_ylabel('Tần suất')
st.pyplot(fig)

# Hàm KNN Imputer
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

# Kiểm tra lại giá trị null
st.subheader("Giá trị null sau khi impute")
st.write(data.isnull().sum())

# Biểu đồ phân phối tuổi sau khi impute
st.subheader("Phân phối tuổi sau khi impute")
fig, ax = plt.subplots()
ax.hist(data['age'], bins=20, edgecolor='black')
ax.set_title('Phân phối tuổi')
ax.set_xlabel('Tuổi')
ax.set_ylabel('Tần suất')
st.pyplot(fig)

# Tính toán ma trận tương quan
corr = num_cols.corr()
top_corr = corr['click'].sort_values(ascending=False)[1:20].to_frame()
st.subheader("Ma trận tương quan")
st.write(top_corr)

# Biểu đồ nhiệt cho ma trận tương quan
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, ax=ax, cmap='coolwarm', fmt=".2f")
ax.set_title('Ma trận tương quan')
st.pyplot(fig)

# Mã hóa các biến phân loại
data_encoded = pd.get_dummies(data, drop_first=True)
corr_matrix = data_encoded.corr()

# Biểu đồ nhiệt cho ma trận tương quan bao gồm các biến phân loại
fig, ax = plt.subplots(figsize=(20, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', ax=ax, cmap='coolwarm', linewidths=0.5)
ax.set_title('Ma trận tương quan bao gồm các biến phân loại')
st.pyplot(fig)

# Tính toán phân phối click
distribution_click = data['click'].value_counts(normalize=True) * 100
st.subheader("Phân phối click")
st.write(distribution_click)

# Phân phối theo nhóm tuổi
bins = [17, 24, 34, 44, 54, 64, 100]
labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
grouped = (
    data.assign(age_group=pd.cut(data['age'], bins=bins, labels=labels))
    .groupby(['age_group', 'click'], observed=False)
    .size()
    .unstack(fill_value=0)
)

percentages = grouped.div(grouped.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(12, 6))
grouped.plot(kind='bar', stacked=True, ax=ax)
ax.set_title('Trạng thái Click theo nhóm tuổi')
ax.set_xlabel('Nhóm tuổi')
ax.set_ylabel('Số lượng Click')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(['Không Click', 'Click'])

# Thêm nhãn phần trăm lên cột
for i in range(len(grouped)):
    total = grouped.iloc[i].sum()
    for j in range(len(grouped.columns)):
        if total > 0:
            percentage = (grouped.iloc[i, j] / total) * 100
            ax.text(i, grouped.iloc[i, :j + 1].sum() - grouped.iloc[i, j] / 2, f'{percentage:.1f}%',
                    ha='center', va='center', color='white', fontsize=8)

st.pyplot(fig)

# Thống kê các loại thiết bị
st.subheader("Tỷ lệ click theo loại thiết bị")
click_counts = data[data['click'] == 1]['device_type'].value_counts(normalize=True) * 100
fig, ax = plt.subplots(figsize=(10, 6))
click_counts.plot(kind='bar', color='darkblue', ax=ax)
ax.set_xlabel('Loại thiết bị')
ax.set_ylabel('Tỷ lệ click')
ax.set_title('Tỷ lệ click theo loại thiết bị')

for i in range(len(click_counts)):
    percentage = click_counts.iloc[i]
    ax.text(i, percentage / 2, f'{percentage:.1f}%',
            ha='center', va='center', color='white', fontsize=10)

st.pyplot(fig)

# Đếm số lần xuất hiện của từng loại thiết bị
st.subheader("Số lần xuất hiện của từng loại thiết bị")
st.write(data['device_type'].value_counts())

# Thống kê về ad_position
st.subheader("Tỷ lệ click theo vị trí quảng cáo")
click_counts = data[data['click'] == 1]['ad_position'].value_counts(normalize=True) * 100
fig, ax = plt.subplots(figsize=(10, 6))
click_counts.plot(kind='bar', color='darkblue', ax=ax)
ax.set_xlabel('Vị trí quảng cáo')
ax.set_ylabel('Tỷ lệ click')
ax.set_title('Tỷ lệ click theo vị trí quảng cáo')

for i in range(len(click_counts)):
    percentage = click_counts.iloc[i]
    ax.text(i, percentage / 2, f'{percentage:.1f}%',
            ha='center', va='center', color='white', fontsize=10)

st.pyplot(fig)

# Biểu đồ tương quan giữa ad_position và device_type
st.subheader("Số lượng Vị trí quảng cáo theo Loại thiết bị")
crosstab = pd.crosstab(data['device_type'], data['ad_position'])
fig, ax = plt.subplots(figsize=(10, 6))
crosstab.plot(kind='bar', color=sns.color_palette("pastel"), ax=ax)

for p in ax.patches:
    ax.annotate(f'{p.get_height()}',
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Loại thiết bị', fontsize=12)
ax.set_ylabel('Số lượng', fontsize=12)
ax.set_title('Số lượng Vị trí quảng cáo theo Loại thiết bị', fontsize=14)
ax.legend(title='Vị trí quảng cáo', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
st.pyplot(fig)

# Biểu đồ độ tuổi theo loại thiết bị
st.subheader("Độ tuổi theo loại thiết bị")
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='device_type', y='age', hue='device_type', data=data, palette='Set3', ax=ax)
ax.set_title('Độ tuổi theo loại thiết bị')
st.pyplot(fig)

# Biểu đồ độ tuổi theo thời gian trong ngày
st.subheader("Độ tuổi theo thời gian trong ngày")
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='time_of_day', y='age', hue='time_of_day', data=data, palette='Set3', ax=ax)
ax.set_title('Độ tuổi theo thời gian trong ngày')
plt.xticks(rotation=45)
st.pyplot(fig)

# Thêm thông tin liên hệ
st.markdown("""
    <hr>
    <h3 style='text-align: center;'>Liên hệ</h3>
    <p style='text-align: center;'>Nếu bạn cần thêm thông tin hoặc có câu hỏi, vui lòng liên hệ với chúng tôi:</p>
    <p style='text-align: center;'>Email: <a href='mailto:hoang.nguyenhuy2024ll@hcmut.edu.vn'>hoang.nguyenhuy2024ll@hcmut.edu.vn</a></p>
    <p style='text-align: center;'>Số điện thoại: 0393 875 279</p>
    <p style='text-align: center;'>Địa chỉ: 83 Đường Tô Ký, Thành phố Hồ Chí Minh, Việt Nam</p>
    """, unsafe_allow_html=True)
