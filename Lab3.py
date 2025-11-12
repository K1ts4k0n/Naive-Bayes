import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from tabulate import tabulate

# โหลดข้อมูล
df = pd.read_csv('adult.csv')

# ลบแถวที่มี '?' ในข้อมูล
df = df.replace('?', pd.NA).dropna()

# เลือกเฉพาะคอลัมน์ที่ใช้
cols = ['age', 'education', 'occupation', 'hours.per.week', 'sex', 'income']
df = df[cols]

# กำหนด categorical columns ที่ต้อง encode
categorical_cols = ['education', 'occupation', 'sex', 'income']

# สร้าง LabelEncoder สำหรับแต่ละคอลัมน์ categorical
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# กำหนด feature กับ target
feature_cols = ['age', 'education', 'occupation', 'hours.per.week', 'sex']
X = df[feature_cols]
y = df['income']

# แบ่ง train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

# ----- เริ่มต้นส่วนที่เกี่ยวข้องกับทฤษฎี Naive Bayes -----
# GaussianNB คือการประยุกต์ Naive Bayes โดยสมมติว่า
# แต่ละฟีเจอร์เป็นตัวแปรที่มีการแจกแจงแบบ Gaussian (normal distribution)
# และสมมติว่าแต่ละฟีเจอร์เป็นอิสระ (Naive assumption)
# โมเดลจะเรียนรู้พารามิเตอร์ mean และ variance ของแต่ละฟีเจอร์ในแต่ละคลาส
model = GaussianNB()

# ฝึกโมเดลด้วยข้อมูล train
model.fit(X_train, y_train)

# ทำนายค่าด้วยชุดทดสอบ
y_pred = model.predict(X_test)

# ประเมินผลความแม่นยำของโมเดล
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ฟังก์ชันช่วยถอดรหัส label ให้เป็นข้อความเดิมเพื่อความเข้าใจง่าย
def decode_labels(df_subset):
    """ถอดรหัส categorical กลับเป็นข้อความ"""
    df_decoded = df_subset.copy()
    for col in categorical_cols:
        if col in df_decoded.columns:
            le = label_encoders[col]
            df_decoded[col] = le.inverse_transform(df_decoded[col].astype(int))
    return df_decoded

def predict_income():
    print("\n-- First 100 records (age, education, occupation, hours.per.week, sex, income) --")
    print(tabulate(
        decode_labels(df.head(100)).reset_index(drop=True),
        headers='keys', tablefmt='grid'
    ))

    print("\n-- Enter details to predict income --")
    input_dict = {}
    for col in feature_cols:
        if col in categorical_cols:
            le = label_encoders[col]
            options = list(le.classes_)
            print(f"Choose {col} from options:")
            print(", ".join(options))
            val = input(f"Enter {col}: ").strip()
            if val not in options:
                print(f"Invalid input for {col}. Please choose from the given options.")
                return
            input_dict[col] = le.transform([val])[0]
        else:
            val = input(f"Enter {col} (numeric): ").strip()
            try:
                val = float(val)
            except:
                print(f"Invalid numeric input for {col}.")
                return
            input_dict[col] = val

    input_df = pd.DataFrame([input_dict])

    # ----- ทำนายความน่าจะเป็นของแต่ละคลาส -----
    # predict_proba คืนค่าความน่าจะเป็น posterior P(class|features)
    proba = model.predict_proba(input_df)[0]

    income_le = label_encoders['income']
    idx_50k = list(income_le.classes_).index('>50K')

    print(f"\nProbability income >50K: {proba[idx_50k]*100:.2f}%")
    print(f"Probability income <=50K: {proba[1-idx_50k]*100:.2f}%")

    


predict_income()

