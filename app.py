import streamlit as st
import joblib

# --- MANUAL ENCODING LISTS: Must match your model training exactly! ---
workclass_list = [
    'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
    'Local-gov', 'State-gov', 'Without-pay', 'Never-worked', 'Others'
]
marital_status_list = [
    'Married-civ-spouse', 'Divorced', 'Never-married', 'Separated',
    'Widowed', 'Married-spouse-absent', 'Others'
]
occupation_list = [
    'Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial',
    'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical',
    'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv',
    'Armed-Forces', 'Others'
]
relationship_list = [
    'Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried', 'Others'
]
race_list = ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Others']
gender_list = ['Male', 'Female']
native_country_list = ['United-States', 'Others']

def encode(val, lst):
    try:
        return lst.index(val)
    except Exception:
        return len(lst) - 1  # 'Others'

@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

model = load_model()

st.title("Employee Salary Classification")
st.markdown("""
Predict if an employee's income is **>50K** or **<=50K** (annual, USD), using UCI Adult data attributes.  
_All preprocessing and feature handling as per your model training. You must keep the encoding lists and feature order EXACTLY as during model training._
""")

# --- User Inputs: Only 13 features as required by your model (no education str, no income) ---
age = st.sidebar.slider('Age', 17, 90, 30)
workclass = st.sidebar.selectbox('Workclass', workclass_list)
fnlwgt = st.sidebar.number_input('fnlwgt', 10000, 1000000, 150000)
educational_num = st.sidebar.slider('Educational-Num', 1, 16, 10)
marital_status = st.sidebar.selectbox('Marital Status', marital_status_list)
occupation = st.sidebar.selectbox('Occupation', occupation_list)
relationship = st.sidebar.selectbox('Relationship', relationship_list)
race = st.sidebar.selectbox('Race', race_list)
gender = st.sidebar.selectbox('Gender', gender_list)
capital_gain = st.sidebar.number_input('Capital Gain', 0, 100000, 0)
capital_loss = st.sidebar.number_input('Capital Loss', 0, 5000, 0)
hours_per_week = st.sidebar.slider('Hours per week', 1, 99, 40)
native_country = st.sidebar.selectbox('Native Country', native_country_list)

if st.button('Predict'):
    input_data = [[
        age,
        encode(workclass, workclass_list),
        fnlwgt,
        educational_num,
        encode(marital_status, marital_status_list),
        encode(occupation, occupation_list),
        encode(relationship, relationship_list),
        encode(race, race_list),
        encode(gender, gender_list),
        capital_gain,
        capital_loss,
        hours_per_week,
        encode(native_country, native_country_list)
    ]]
    st.write("▶️ **Model input**:", input_data)  # Show user what goes into the model
try:
    pred = model.predict(input_data)[0]  # pred will be 0/1 OR a string ('<=50K', '>50K')
    if str(pred) in ['1', '>50K']:
        pred_label = ">50K"
    else:
        pred_label = "<=50K"
    st.success(f"**Predicted Income:** {pred_label}")

    # Optional: probability output if classifier supports it
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_data)[0]
        st.write(f"Probability '<=50K': {probs[0]:.2%} &nbsp;&nbsp; | &nbsp;&nbsp; '>50K': {probs[1]:.2%}")

except Exception as e:
    st.error(
        f"⚠️ Prediction failed:\n{e}\n"
        "Check that ALL 13 features are present and encoding lists match model training."
    )


st.markdown("---")
st.caption("No label/target is sent to the model. If you get an error, verify you have 13 features, all int/float (no strings), and encoding lists match your training notebook.")
