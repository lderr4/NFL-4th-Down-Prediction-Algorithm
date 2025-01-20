import streamlit as st
import requests
from constants import SLIDER_DEFAULTS, API_URL, COLUMN_NAME_CLEANER, BAR_CHART_DEFAULTS
from pandas import Series, DataFrame, melt
import matplotlib.pyplot as plt
import altair as alt

if "sliders" not in st.session_state:
    st.session_state.sliders = {key: val[2] for key, val in SLIDER_DEFAULTS.items()}
    st.session_state.bar_chart = BAR_CHART_DEFAULTS
    st.session_state.wpa_dict = {}


def fetch_random_row():
    response = requests.get(f"{API_URL}/random_test_row")
    if response.status_code == 200:
        random_row_list = response.json()
        if isinstance(random_row_list, list) and len(random_row_list) > 0:
            row = random_row_list[0]
            row = {key: value for key, value in row.items() if key in st.session_state.sliders}
            st.session_state.sliders.update(row)
            st.success("Random play taken from Test Set!")
        else:
            st.error("Unexpected response format or empty list.")
    else:
        st.error(f"Failed to fetch random row: {response.status_code}")


def fetch_model_prediction():
    
    response = requests.post(f"{API_URL}/get_classifier_predict_proba", json={"play": st.session_state.sliders})
    if response.status_code == 200:
        st.success("Model Prediction Retrieved Successfully!")
        response = response.json()['response']
        st.session_state.bar_chart.update(response['predict_probabilities'])
        st.session_state.wpa_dict = response['wpa_predictions']

    else:
        st.error(f"Error: {response.status_code} - {response.text}")

st.title("NFL 4th Down Prediction Model")
st.subheader("Check out the source code: \nhttps://github.com/lderr4/NFL-4th-Down-Prediction-Algorithm/blob/main/README.md")

col1, col2 = st.columns([1, 1])  # Adjust proportions as needed (e.g., [1, 2] for unequal columns)
with col1:
    st.header("Play Input")
    st.button("Load Random Play from Test Set", on_click=fetch_random_row)
    with st.expander("Model Input Features", expanded=True):
        
        for key, value in st.session_state.sliders.items():

            min_val, max_val, _ = SLIDER_DEFAULTS[key]
            if isinstance(min_val, int):
                value = int(value)
                st.session_state.sliders[key] = st.slider(
                    COLUMN_NAME_CLEANER[key], min_value=min_val, max_value=max_val, value=value, step=1
                )
            elif isinstance(min_val, float):
                value = float(value)
                st.session_state.sliders[key] = st.slider(
                    COLUMN_NAME_CLEANER[key], min_value=min_val, max_value=max_val, value=value, step=0.01
                )
            else:
                st.session_state.sliders[key] = st.selectbox(
                    COLUMN_NAME_CLEANER[key], options=[0, 1], index=int(value)
                )
with col2:
    
    st.header("Model Output")
    st.button("Make Prediction", on_click=fetch_model_prediction)
    
    # predict probabilities
    df = Series(st.session_state.bar_chart).reset_index()
    df.columns = ['Play Type', 'Predicted Probability']
    
    df = df.sort_values(by='Predicted Probability', ascending=False)
    df['Play Type'] = df['Play Type'].map(COLUMN_NAME_CLEANER)
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Play Type', 
        axis=alt.Axis(labelAngle=0)),
        y='Predicted Probability'
    ).properties(
        width=400,
        height=300  
    )
    st.altair_chart(chart)


    st.subheader("Win Probability Analysis")

    current_win_probability = st.session_state.sliders["wp_avg"]
    st.metric("Current Win Probability", f"{current_win_probability * 100:.2f}%")

    wpa_preds = DataFrame(
        {
            "Play Type": list(st.session_state.wpa_dict.keys()),
            "WPA Change": list(st.session_state.wpa_dict.values()),
        }
    )
    wpa_preds["WPA Change (%)"] = wpa_preds["WPA Change"] * 100  # Convert to percentage

    for _, row in wpa_preds.iterrows():

        play_type = row["Play Type"]
        wpa_change = row["WPA Change (%)"]

        st.metric(
            label=f"Predicted Win Probability after: {COLUMN_NAME_CLEANER[play_type]}",
            value=f"{(current_win_probability + wpa_change / 100) * 100:.2f}%",  # Adjusted win probability
            delta=f"{wpa_change:+.2f}%",
    )
