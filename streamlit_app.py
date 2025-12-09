import streamlit as st
import pandas as pd
import joblib
import json
import shap
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Phishing Detection", layout="centered")
st.title("🔐 Phishing Website Detection (Logistic Regression Only)")

st.write("Use the form below to test whether a website is phishing or legitimate based on AI prediction.")

# ---------------------------------------------------------
# Load ONLY Logistic Regression model
# ---------------------------------------------------------
model_path = "models/phishing_model_lr.pkl"
metrics_path = "metrics/metrics_lr.json"

model = joblib.load(model_path)
features = list(model.feature_names_in_)

# ---------------------------------------------------------
# Load evaluation metrics
# ---------------------------------------------------------
try:
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    report_df = pd.DataFrame(metrics["classification_report"]).transpose()
    st.subheader("📊 Model Evaluation Metrics")
    st.dataframe(report_df[['precision', 'recall', 'f1-score', 'support']])
except Exception as e:
    st.warning(f"Could not load metrics: {e}")

# ---------------------------------------------------------
# Input Form
# ---------------------------------------------------------
label_map = {-1: "Suspicious / Malicious", 0: "Neutral / Uncertain", 1: "Legitimate / Safe"}

input_data = {}

with st.form("phishing_form"):
    st.subheader("🔢 Feature Inputs")

    for feature in features:
        input_data[feature] = st.radio(
            feature,
            options=[-1, 0, 1],
            format_func=lambda x: label_map[x],
            key=feature
        )

    submitted = st.form_submit_button("🔎 Predict")

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if submitted:
    input_df = pd.DataFrame([input_data])[features]
    prediction = model.predict(input_df)[0]

    # Correct label interpretation
    if prediction == -1:
        st.error("🚨 Prediction: **Phishing / Malicious**")
    elif prediction == 0:
        st.warning("⚠️ Prediction: **Uncertain / Suspicious**")
    else:
        st.success("✅ Prediction: **Legitimate / Safe**")

    # -----------------------------------------------------
    # SHAP Explanation
    # ------------------------------------------------------
    with st.expander("🧠 Why this prediction?"):
        bg_data = pd.read_csv("data/phishing.csv")
        bg_data = bg_data.rename(columns={"Result": "Label"})
        bg_data["Label"] = bg_data["Label"].map({-1: 1, 1: 0})
        X_bg = bg_data[features]

        # Logistic Regression uses linear SHAP
        explainer = shap.Explainer(model, X_bg, algorithm="linear")
        shap_values = explainer(input_df)

        if hasattr(shap_values, "values"):
            shap_val_row = shap_values.values[0]
        else:
            shap_val_row = shap_values[0].values[0]

        top_indices = np.argsort(np.abs(shap_val_row))[::-1][:3]
        top_features = [features[i] for i in top_indices]

        st.subheader("🔍 Feature Influence on Prediction")
        shap.plots.bar(shap.Explanation(values=shap_val_row, feature_names=features), show=False)
        st.pyplot(plt.gcf(), clear_figure=True)

        # Friendly explanation text
        feature_expl_dict = {
            "having_IP_Address": "the URL uses an IP address",
            "URL_Length": "the URL is unusually long",
            "Shortining_Service": "a URL shortening service is used",
            "having_At_Symbol": "the URL contains an '@' symbol",
            "double_slash_redirecting": "the URL contains extra double slashes",
            "Prefix_Suffix": "a hyphen is used in the domain",
            "having_Sub_Domain": "the URL contains many subdomains",
            "SSLfinal_State": "the SSL certificate is invalid",
            "Domain_registeration_length": "the domain is newly registered",
            "Favicon": "the favicon loads from a different domain",
            "port": "a non-standard port is used",
            "HTTPS_token": "the word HTTPS appears suspiciously",
            "Request_URL": "resources load from suspicious domains",
            "URL_of_Anchor": "anchor tags link to unusual sites",
            "Links_in_tags": "tags link to suspicious resources",
            "SFH": "the form handler is abnormal",
            "Submitting_to_email": "the form submits to an email",
            "Abnormal_URL": "the URL structure is abnormal",
            "Redirect": "the website performs multiple redirects",
            "on_mouseover": "content changes on mouseover",
            "RightClick": "right-click is disabled",
            "popUpWidnow": "pop-ups appear unexpectedly",
            "Iframe": "hidden iframes are used",
            "age_of_domain": "the domain is too new",
            "DNSRecord": "DNS record is missing",
            "web_traffic": "web traffic is very low",
            "Page_Rank": "the site has poor PageRank",
            "Google_Index": "the site is not indexed by Google",
            "Links_pointing_to_page": "very few backlinks",
            "Statistical_report": "listed in phishing databases"
        }

        readable = [feature_expl_dict.get(f, f) for f in top_features]

        explanation = (
            f"The model classified this website as "
            f"{'phishing' if prediction == -1 else 'legitimate'} primarily because "
            f"{', '.join(readable[:-1])}, and {readable[-1]}."
        )

        st.info(explanation)
