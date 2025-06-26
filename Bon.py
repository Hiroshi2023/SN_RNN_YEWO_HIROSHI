import streamlit as st
import yfinance as yf
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from neuralprophet import NeuralProphet
from sklearn.metrics import mean_squared_error
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import json
import os
from dotenv import load_dotenv


import smtplib
from email.message import EmailMessage
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

st.set_page_config(layout="wide")

SENDER_EMAIL = os.environ.get("EMAIL_USER")
SENDER_PASSWORD = os.environ.get("EMAIL_PASS")

load_dotenv()


def send_email(sender_email, sender_password, recipient_email, subject, body):
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg.set_content(body)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)

        return True
    except Exception as e:
        print(f"Erreur : {e}")
        return False
    


def load_lottie_file(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
lottie_animation = load_lottie_file("Animation.json")
lottie_animation2 = load_lottie_file("Animation2.json")
lottie_animation3 = load_lottie_file("Animation3.json")
lottie_animation1 = load_lottie_file("Animation1.json")
# ------------------------
# BANDES DE BOLLINGER
# ------------------------
def add_bollinger_bands(df, window=20):
    df['SMA'] = df['Close'].rolling(window=window).mean()
    df['STD'] = df['Close'].rolling(window=window).std()
    df['Upper Band'] = df['SMA'] + (2 * df['STD'])
    df['Lower Band'] = df['SMA'] - (2 * df['STD'])
    return df

def plot_bollinger(df):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df['Close'], label='Prix', color='blue')
    ax.plot(df.index, df['Upper Band'], label='Bande supérieure', color='green', linestyle='--')
    ax.plot(df.index, df['Lower Band'], label='Bande inférieure', color='red', linestyle='--')
    ax.fill_between(df.index, df['Lower Band'], df['Upper Band'], color='gray', alpha=0.2)
    ax.set_title("📊 Bandes de Bollinger sur 20 jours")
    ax.set_xlabel("Date")
    ax.set_ylabel("Prix (USD)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# ------------------------
# GRU MODEL
# ------------------------
class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.gru = nn.GRU(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        gru_out, _ = self.gru(input_seq)
        predictions = self.linear(gru_out[-1])
        return predictions

# ------------------------
# LOAD DATA
# ------------------------
@st.cache_data
def load_data():
    df = yf.download('AMD', period='4y')
    df = df[['Close']].dropna()
    return df

# ------------------------
# Cached GRU PREDICTION
# ------------------------
@st.cache_resource
def get_gru_predictions(df):
    return train_gru(df)

@st.cache_resource
def get_np_predictions(df):
    return predict_neural_prophet(df)


# ------------------------
# GRU PREDICTION
# ------------------------
def train_gru(df, n_steps=20, future_steps=21):
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)

    X, y = [], []
    for i in range(n_steps, len(data_normalized)):
        X.append(data_normalized[i - n_steps:i])
        y.append(data_normalized[i])
    X, y = np.array(X), np.array(y)

    X_train = torch.from_numpy(X).float()
    y_train = torch.from_numpy(y).float()

    model = GRUModel()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):
        model.zero_grad()
        output = model(X_train.permute(1, 0, 2))
        loss = loss_function(output, y_train)
        loss.backward()
        optimizer.step()

    # Prediction
    preds = []
    last_seq = X[-1]
    for _ in range(future_steps):
        with torch.no_grad():
            input_seq = torch.from_numpy(last_seq).float().unsqueeze(1)
            pred = model(input_seq)[0].item()
        preds.append(pred)
        last_seq = np.append(last_seq[1:], [[pred]], axis=0)

    return scaler.inverse_transform(np.array(preds).reshape(-1, 1))

def backtest_gru(df, n_steps=20, backtest_days=21):
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)

    X, y = [], []
    for i in range(n_steps, len(data_normalized)):
        X.append(data_normalized[i - n_steps:i])
        y.append(data_normalized[i])
    X, y = np.array(X), np.array(y)

    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float()

    model = GRUModel()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):
        model.zero_grad()
        output = model(X_tensor.permute(1, 0, 2))
        loss = loss_function(output, y_tensor)
        loss.backward()
        optimizer.step()

    # Backtest sur les X derniers jours
    preds_bt = []
    true_bt = []
    for i in range(-backtest_days, 0):
        seq = data_normalized[i - n_steps:i]
        input_seq = torch.from_numpy(seq).float().unsqueeze(1)
        with torch.no_grad():
            pred = model(input_seq)[0].item()
        preds_bt.append(pred)
        true_bt.append(data_normalized[i][0])  # valeur réelle normalisée

    # Convertir en prix réels
    preds_bt = scaler.inverse_transform(np.array(preds_bt).reshape(-1, 1))
    true_bt = scaler.inverse_transform(np.array(true_bt).reshape(-1, 1))

    dates_bt = df.index[-backtest_days:]
    return pd.DataFrame({
        'Date': dates_bt,
        'Réel': true_bt.flatten(),
        'Prévu': preds_bt.flatten()
    }).set_index('Date')


def plot_predictions(df, future_df, label_future, color_future, add_margin=True, margin_pct=0.05):
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(df.index, df['Close'], label='Réel', color='red')
    ax.plot(future_df.index, future_df[label_future], label=label_future, color=color_future)

    if add_margin:
        # Marge d'incertitude de ±5%
        upper_bound = future_df[label_future] * (1 + margin_pct)
        lower_bound = future_df[label_future] * (1 - margin_pct)

        ax.fill_between(future_df.index, lower_bound, upper_bound,
                        color=color_future, alpha=0.2, label='Marge de confiance ±5%')

    ax.set_title(f"Évolution des actions AMD - {label_future}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Prix (USD)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)



# ------------------------
# NEURAL PROPHET PREDICTION
# ------------------------
def predict_neural_prophet(df, future_days=21):
    # ⚠️ Ne conserve que 'Close' au cas où il y a d'autres colonnes (SMA, etc.)
    df_clean = df[['Close']].copy()
    df_full = df_clean.asfreq('D').fillna(method='ffill')
    df_reset = df_full.reset_index()
    df_reset.columns = ['ds', 'y']

    model = NeuralProphet()
    model.fit(df_reset, freq="B", epochs=100)

    future = model.make_future_dataframe(df_reset, periods=future_days)
    forecast = model.predict(future)

    forecast = forecast[['ds', 'yhat1']]
    forecast = forecast.set_index('ds')
    forecast.rename(columns={'yhat1': 'NeuralProphet'}, inplace=True)
    return forecast


# ------------------------
# STREAMLIT UI
# ------------------------

#st.set_page_config(page_title="Prédiction AMD", layout="wide")
# --- Menu ---


with st.sidebar:
    selection = option_menu(
        menu_title="Menu",
        options=["Accueil","Donnees Precedentes", "Modele GRU", "Modele Neural Prophet","MSE", "Simulateur","Back Testing GRU", "A Propos","Contact Us"],
        icons=["check","check", "check", "check","check","check", "check","check","check"],
        menu_icon="globe",
        default_index=0,
        orientation="vertical",
        styles={
            "container": {"padding": "5px", "background-color": "#f0f2f6"},
            "icon": {"color": "blue", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "5px",
                "--hover-color": "#e0e0e0"
            },
            "nav-link-selected": {
                "background-color": "#28a745",
                "color": "white",
                "font-weight": "bold"
            }
        }
    )


df = load_data()

gru_preds = get_gru_predictions(df)
future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=21)
gru_df = pd.DataFrame(gru_preds, index=future_dates, columns=['GRU'])


np_df = get_np_predictions(df)
# Supprimer les anciennes données pour ne garder que la prédiction future
np_df_future = np_df[np_df.index > df.index[-1]]



if selection == "Accueil":
    st.markdown("""
    <div style="background-color: #28a745; border-radius: 15px; padding: 20px; 
                box-shadow: 0 1px 1px rgba(0, 0, 0, 0.1); margin: 0px 0;">
        <h3 style="color:#e0e0e0; font-weight: bold;text-align:center;">📈 Prédiction des actions AMD - GRU & NeuralProphet</h3>    </div>  """, unsafe_allow_html=True)
    
    col1, col2= st.columns([2,1])
    with col2:
        st_lottie(lottie_animation1, speed=1, width=300, height=300, key="lottie2")
        st_lottie(lottie_animation, speed=1, width=300, height=200, key="lottie1")

    with col1:
        st.markdown("""
    <div style="background-color: #fd7e14; border-radius: 15px; padding: 20px; 
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1); margin: 20px 0;">
        <h3 style="color:#2c3e50; font-weight: bold;">
            🎯 Bienvenue dans l'application de prédiction boursière AMD
        </h3>
        <p style="color: #34495e; font-size: 16px;">
Cette plateforme interactive de prévision des cours boursiers de l'action AMD (Advanced Micro Devices) basée sur les données de Yahoo Finance.
Cette application utilise deux puissants modèles de Deep Learning et d’analyse de séries temporelles :
        <li>GRU (Gated Recurrent Unit) pour la modélisation séquentielle, </li>
        <li>NeuralProphet, inspiré de Facebook Prophet, optimisé pour la prévision multivariée.</li>
        </p>
    </div>
    """, unsafe_allow_html=True)
        col1, col2,col3,col4= st.columns(4)
        with col1:
            st.markdown("""
    <div style="background-color: #28a745; border-radius: 15px; padding: 20px; 
                box-shadow: 0 1px 1px rgba(0, 0, 0, 0.1); margin: 0px 0;">
        <h5 style="color:#e0e0e0; font-weight: bold;">Graphiques</h5>    </div>  """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
    <div style="background-color: #28a745; border-radius: 15px; padding: 20px; 
                box-shadow: 0 1px 1px rgba(0, 0, 0, 0.1); margin: 0px 0;">
        <h5 style="color:#e0e0e0; font-weight: bold;">Simulateur</h5>    </div>  """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
    <div style="background-color: #28a745; border-radius: 15px; padding: 20px; 
                box-shadow: 0 1px 1px rgba(0, 0, 0, 0.1); margin: 0px 0;">
        <h5 style="color:#e0e0e0; font-weight: bold;">Back Testing</h5>    </div>  """, unsafe_allow_html=True)
        with col4:
            st.markdown("""
    <div style="background-color: #28a745; border-radius: 15px; padding: 20px; 
                box-shadow: 0 1px 1px rgba(0, 0, 0, 0.1); margin: 0px 0;">
        <h5 style="color:#e0e0e0; font-weight: bold;">Predictions</h5>    </div>  """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #fd7e14; border-radius: 15px; padding: 20px; 
                box-shadow: 0 1px 1px rgba(0, 0, 0, 0.1); margin: 0px 0;">
        <h5 style="color:#2c3e50; font-weight: bold;">📈 Prenez une longueur d’avance sur le marché grâce à l’intelligence artificielle</h5>    </div>  """, unsafe_allow_html=True)

elif selection == "Donnees Precedentes":
    st.markdown("""
    <div style="background-color: #28a745; border-radius: 15px; padding: 20px; 
                box-shadow: 0 1px 1px rgba(0, 0, 0, 0.1); margin: 0px 0;">
        <h3 style="color:#e0e0e0; font-weight: bold;text-align:center;">Données AMD - 4 dernières années</h3>    </div>  """, unsafe_allow_html=True)
    st.markdown("""<div style='text-align: center;'><h1 style='text-decoration: underline; color: sky-blue;'>Voici les données précédentes pour AMD</h1></div>""", unsafe_allow_html=True)
    st.dataframe(df)
    st.markdown("""<div style='text-align: center;'><h1 style='text-decoration: underline; color: sky-blue;'>Graphique des prix de clôture d'AMD</h1></div>""", unsafe_allow_html=True)
    st.subheader("Voici l'évolution des prix de clôture d'AMD sur les 4 dernières années :")
    st.line_chart(df['Close'])
# Ajout des bandes de Bollinger
    st.markdown("""
    <div style="background-color: #28a745; border-radius: 15px; padding: 20px; 
                box-shadow: 0 1px 1px rgba(0, 0, 0, 0.1); margin: 0px 0;">
        <h3 style="color:#e0e0e0; font-weight: bold;text-align:center;">Bandes de Bollinger (20 jours)</h3>    </div>  """, unsafe_allow_html=True)
    st.markdown("""<div style='text-align: center;'><h1 style='text-decoration: underline; color: sky-blue;'>Voici les bandes de Bollinger pour AMD</h1></div>""", unsafe_allow_html=True)
    st.info("Les bandes de Bollinger sont des indicateurs techniques qui aident à évaluer la volatilité du marché et à identifier les conditions de surachat ou de survente. Elles sont composées d'une moyenne mobile simple (SMA) et de deux bandes situées à deux écarts-types au-dessus et en dessous de la SMA.")
    st.info("Les bandes de Bollinger sont calculées sur une période de 20 jours, ce qui permet de lisser les fluctuations des prix et de fournir une vue d'ensemble de la tendance du marché.")
    st.info("Les bandes supérieures et inférieures sont tracées autour de la SMA, et elles s'élargissent ou se rétrécissent en fonction de la volatilité des prix. Lorsque les bandes sont éloignées, cela indique une forte volatilité, tandis que des bandes rapprochées suggèrent une faible volatilité.")

    df = add_bollinger_bands(df)
    plot_bollinger(df)

elif selection == "Modele GRU":
    st.markdown("""
    <div style="background-color: #28a745; border-radius: 15px; padding: 20px; 
                box-shadow: 0 1px 1px rgba(0, 0, 0, 0.1); margin: 0px 0;">
        <h3 style="color:#e0e0e0; font-weight: bold;text-align:center;">Prédiction avec GRU (3 semaines)</h3>   
                 </div>  """, unsafe_allow_html=True)
    st.info("Le modèle GRU (Gated Recurrent Unit) est un type de réseau de neurones récurrent qui est particulièrement efficace pour traiter des séquences de données, comme les séries temporelles financières. Il est conçu pour capturer les dépendances à long terme dans les données tout en étant moins complexe que les LSTM (Long Short-Term Memory).")
    st.info("Le modèle GRU est entraîné sur les données historiques des prix de clôture d'AMD pour apprendre les motifs et les tendances. Une fois entraîné, il peut prédire les prix futurs en se basant sur les séquences passées.")
# GRU Prediction
    st.markdown("""<div style='text-align: center;'><h1 style='text-decoration: underline; color: sky-blue;'>Predictions Tabulaires</h1></div>""", unsafe_allow_html=True)
    st.dataframe(gru_df)
    st.markdown("""<div style='text-align: center;'><h1 style='text-decoration: underline; color: sky-blue;'>Representation Graphique</h1></div>""", unsafe_allow_html=True)

    plot_predictions(df, gru_df, label_future="GRU", color_future="blue")

elif selection == "Modele Neural Prophet":
    st.markdown("""
    <div style="background-color: #28a745; border-radius: 15px; padding: 20px; 
                box-shadow: 0 1px 1px rgba(0, 0, 0, 0.1); margin: 0px 0;">
        <h3 style="color:#e0e0e0; font-weight: bold;text-align:center;">Prédiction avec NeuralProphet (3 semaines)</h3>   
                 </div>  """, unsafe_allow_html=True)
    st.info("Le modèle NeuralProphet est utilisé pour prédire les prix futurs des actions AMD.")
    st.info("Le modèle NeuralProphet est un modèle de prévision de séries temporelles qui combine les avantages de Facebook Prophet et des réseaux de neurones. Il est conçu pour capturer les tendances, les saisons et les effets de vacances dans les données temporelles.")
# Neural Prophet Prediction
    st.markdown("""<div style='text-align: center;'><h1 style='text-decoration: underline; color: sky-blue;'>Predictions Tabulaires</h1></div>""", unsafe_allow_html=True)
    st.dataframe(np_df_future)
    st.info("NeuralProphet est particulièrement adapté aux données financières, car il peut gérer les irrégularités et les changements de tendance. Il est capable de s'adapter aux variations saisonnières et aux événements spéciaux qui peuvent influencer les prix des actions.")

    st.markdown("""<div style='text-align: center;'><h1 style='text-decoration: underline; color: sky-blue;'>Representation Graphique</h1></div>""", unsafe_allow_html=True)
    plot_predictions(df, np_df_future, label_future="NeuralProphet", color_future="green")
    st.info("Le modèle est entraîné sur les données historiques des prix de clôture d'AMD pour apprendre les motifs et les tendances. Une fois entraîné, il peut prédire les prix futurs en se basant sur les séquences passées.")

elif selection == "MSE":
    st.markdown("""
    <div style="background-color: #28a745; border-radius: 15px; padding: 20px; 
                box-shadow: 0 1px 1px rgba(0, 0, 0, 0.1); margin: 0px 0;">
        <h3 style="color:#e0e0e0; font-weight: bold;text-align:center;">MSE (Mean Squared Error)</h3>   
                 </div>  """, unsafe_allow_html=True)
    st.markdown("""<div style='text-align: center;'><h1 style='text-decoration: underline; color: sky-blue;'>Comparaison des modèles</h1></div>""", unsafe_allow_html=True)
    st.markdown("### Comparaison des modèles GRU et NeuralProphet sur les 21 derniers jours")
    st.info("Nous allons comparer les performances des modèles GRU et NeuralProphet en utilisant le MSE (Mean Squared Error) sur les 21 derniers jours de données réelles. Le MSE est une mesure de la qualité d'un modèle de prédiction, où un MSE plus bas indique une meilleure performance.")


    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
    <div style="background-color: #28a745; border-radius: 15px; padding: 20px; 
                box-shadow: 0 1px 1px rgba(0, 0, 0, 0.1); margin: 0px 0;">
        <h3 style="color:#e0e0e0; font-weight: bold;text-align:center;">MSE GRU</h3>   
                 </div>  """, unsafe_allow_html=True)
    # Comparer les 21 derniers jours réels avec les 21 jours prédits (gru)
        df_eval_gru = df[-21:]
        mse_gru = mean_squared_error(df_eval_gru['Close'], gru_df['GRU'][:len(df_eval_gru)])
        st.markdown(f"""<div style='background-color:#fff3cd;border-left:8px solid #fd7e14;
                padding:20px;
                margin:20px 0;
                border-radius:10px;
                font-size:75px;
                font-weight:bold;
                color:#7a4600;'>
                📉 {mse_gru:.2f}
            </div>
            """,
            unsafe_allow_html=True)

    with col2:
        st.markdown("""
    <div style="background-color: #28a745; border-radius: 15px; padding: 20px; 
                box-shadow: 0 1px 1px rgba(0, 0, 0, 0.1); margin: 0px 0;">
        <h3 style="color:#e0e0e0; font-weight: bold;text-align:center;">MSE NEURAL PROPHET 21jours</h3>   
                 </div>  """, unsafe_allow_html=True)
    # Comparer les 21 derniers jours réels avec les 21 jours prédits (neural prophet)
        df_eval_np = df[-21:]
        mse_np = mean_squared_error(df_eval_np['Close'], np_df_future['NeuralProphet'][:len(df_eval_np)])

        st.markdown(f"""<div style='background-color:#fff3cd;border-left:8px solid #fd7e14;
                padding:20px;
                margin:20px 0;
                border-radius:10px;
                font-size:75px;
                font-weight:bold;
                color:#7a4600;'>
                📉 {mse_np:.2f}
            </div>
            """,
            unsafe_allow_html=True)
        
    if mse_gru < mse_np:
        st.success(f"Le modèle GRU est plus précis avec un MSE de {mse_gru:.2f} contre {mse_np:.2f} pour NeuralProphet.")
    elif mse_gru > mse_np:
        st.success(f"Le modèle NeuralProphet est plus précis avec un MSE de {mse_np:.2f} contre {mse_gru:.2f} pour GRU.")
    else:
        st.success(f"Les deux modèles ont le même MSE de {mse_gru:.2f}.")
    st.info("Le MSE (Mean Squared Error) est une mesure de la qualité d'un modèle de prédiction. Il calcule la moyenne des carrés des erreurs entre les valeurs prédites et les valeurs réelles. Un MSE plus bas indique une meilleure performance du modèle.")
    st.info("Dans cette section, nous allons comparer les performances des modèles GRU et NeuralProphet en calculant le MSE sur les 21 derniers jours de données réelles. Cela nous permettra de voir quel modèle est le plus précis dans ses prédictions.")


elif selection == "Simulateur":
    # Simulateur d'Investissement
    st.markdown("""
    <div style="background-color: #28a745; border-radius: 15px; padding: 20px; 
                box-shadow: 0 1px 1px rgba(0, 0, 0, 0.1); margin: 0px 0;">
        <h3 style="color:#e0e0e0; font-weight: bold;text-align:center;">Simulateur d'investissement</h3>   
                 </div>  """, unsafe_allow_html=True)
    st.markdown("""<div style='text-align: center;'><h1 style='text-decoration: underline; color: sky-blue;'>Simuler un investissement</h1></div>""", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1,3])
    with col1:
        st_lottie(lottie_animation3, speed=1, width=300, height=200, key="lottie4")
    with col2:
        st.info("Ce simulateur vous permet de simuler un investissement dans les actions AMD en choisissant un montant à investir et une date d'achat. Il calcule ensuite le nombre d'actions que vous auriez pu acheter et la valeur estimée de votre investissement après 3 semaines, en utilisant les prévisions du modèle GRU.")

    with st.form("investment_form"):
        st.markdown("### Paramètres de simulation")
        capital = st.number_input("\U0001F4B5 Montant à investir (USD)", min_value=100.0, value=1000.0)
        date_achat = st.date_input("\U0001F4C5 Date d'achat", min_value=df.index[0].date(), max_value=df.index[-1].date())
        submitted = st.form_submit_button("Simuler")

    if submitted:
        try:
            date_achat_ts = pd.Timestamp(date_achat)

            # Vérifie si la date existe dans l'index
            if date_achat_ts in df.index:
                prix_achat = float(df.loc[date_achat_ts]['Close'])
            else:
                # Trouve la date boursière la plus proche avant la date choisie
                date_achat_ts = df.index[df.index.get_loc(date_achat_ts, method='pad')]
                prix_achat = float(df.loc[date_achat_ts]['Close'])
                st.info(f"La date choisie n'étant pas un jour de bourse, la date utilisée est le {date_achat_ts.date()}.")

            nb_actions = capital / prix_achat
            prix_future = gru_df['GRU'].iloc[-1]
            valeur_future = nb_actions * prix_future
            gain = valeur_future - capital

            st.markdown(
                    f"""
                    <div style="margin-top: 20px; margin-bottom: 20px;">
                        <table style="width: 100%; border-collapse: collapse; font-size: 22px; text-align: left;">
                            <tr style="background-color: #e8f5e9; color: #1b5e20;">
                                <th style="padding: 12px;">📅 Prix d'achat le {date_achat_ts.date()}</th>
                                <td style="padding: 12px; font-weight: bold;">{prix_achat:.2f} USD</td>
                            </tr>
                            <tr style="background-color: #e3f2fd; color: #0d47a1;">
                                <th style="padding: 12px;">📊 Nombre d'actions simulées</th>
                                <td style="padding: 12px; font-weight: bold;">{nb_actions:.2f}</td>
                            </tr>
                            <tr style="background-color: #fff3cd; color: #7a4600;">
                                <th style="padding: 12px;">🔮 Valeur estimée après 3 semaines</th>
                                <td style="padding: 12px; font-weight: bold;">{valeur_future:.2f} USD</td>
                            </tr>
                            <tr style="background-color: #f8d7da; color: #721c24;" if gain < 0 else "background-color: #d4edda; color: #155724;">
                                <th style="padding: 12px;">💹 Gain / Perte estimé(e)</th>
                                <td style="padding: 12px; font-weight: bold;">{'+' if gain >= 0 else ''}{gain:.2f} USD</td>
                            </tr>
                        </table>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


        except Exception as e:
            st.error(f"Erreur lors de la simulation : {e}")

    st.info("Le simulateur prend en compte le prix d'achat des actions à la date choisie, ainsi que la valeur future estimée des actions après 3 semaines. Il vous permet de visualiser le potentiel de gain ou de perte de votre investissement en fonction des prévisions du modèle GRU.")
    st.info("Pour utiliser le simulateur, entrez le montant que vous souhaitez investir et choisissez une date d'achat. Le simulateur calculera le nombre d'actions que vous auriez pu acheter à ce prix et la valeur estimée de votre investissement après 3 semaines, en utilisant les prévisions du modèle GRU.")


elif selection == "Back Testing GRU":
# Backtesting visuel GRU
    st.markdown("""
    <div style="background-color: #28a745; border-radius: 15px; padding: 20px; 
                box-shadow: 0 1px 1px rgba(0, 0, 0, 0.1); margin: 0px 0;">
        <h3 style="color:#e0e0e0; font-weight: bold;text-align:center;">Back Testing Visuel GRU</h3>   
                 </div>  """, unsafe_allow_html=True)
    st.info("Le backtesting d’un modèle GRU (Gated Recurrent Unit) en finance consiste à évaluer la performance du modèle en utilisant des données historiques, comme si on l’avait utilisé dans le passé pour prédire les cours d’une action (comme AMD) et prendre des décisions d’investissement.")
    backtest_df = backtest_gru(df)
    st.markdown("""<div style='text-align: center;'><h1 style='text-decoration: underline; color: sky-blue;'>Affichage Tabulaire</h1></div>""", unsafe_allow_html=True)
    st.dataframe(backtest_df)

    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(backtest_df.index, backtest_df['Réel'], label='Réel', color='red')
    ax.plot(backtest_df.index, backtest_df['Prévu'], label='Prévision GRU', color='blue', linestyle='--')

    mse_backtest = mean_squared_error(backtest_df['Réel'], backtest_df['Prévu'])
    st.markdown("""
    <div style="background-color: #28a745; border-radius: 15px; padding: 20px; 
                box-shadow: 0 1px 1px rgba(0, 0, 0, 0.1); margin: 0px 0;">
        <h3 style="color:#e0e0e0; font-weight: bold;text-align:center;">MSE Back Test</h3>   
                 </div>  """, unsafe_allow_html=True)
    st.markdown(f"""<div style='background-color:#fff3cd;border-left:8px solid #fd7e14;
                padding:20px;
                margin:20px 0;
                border-radius:10px;
                font-size:75px;
                font-weight:bold;
                color:#7a4600;'>
                📉 {mse_backtest:.2f}
            </div>
            """,
            unsafe_allow_html=True)
    
    st.markdown("""<div style='text-align: center;'><h1 style='text-decoration: underline; color: sky-blue;'>Affichage Graphique</h1></div>""", unsafe_allow_html=True)

    ax.set_title(f"Backtest GRU - MSE : {mse_backtest:.4f}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Prix (USD)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)


elif selection == "A Propos":
    st.markdown("""
    <div style="background-color: #28a745; border-radius: 15px; padding: 20px; 
                box-shadow: 0 1px 1px rgba(0, 0, 0, 0.1); margin: 5px 0;">
        <h3 style="color:#e0e0e0; font-weight: bold;text-align:center;">A Propos</h3>   
                 </div>  """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #cccccc; border-radius: 15px; padding: 20px; 
                box-shadow: 0 1px 1px rgba(0, 0, 0, 0.1); margin: 5px 0;" >
    <h3>🎯Objectif de l'application</h3>
        <ul style="color: #34495e; font-size: 16px;">
            <li>Fournir des prédictions fiables du cours de l'action AMD sur une période de 3 semaines.</li>
            <li>Permettre la visualisation des tendances historiques et futures.</li>
            <li>Offrir un simulateur d’investissement pour estimer les gains ou pertes potentiels.</li>
            <li>Évaluer les modèles à l’aide de métriques comme le MSE.</li>
            <li>Intégrer un backtesting pour tester les performances des modèles dans le passé.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background-color: #cccccc; border-radius: 15px; padding: 20px; 
                box-shadow: 0 1px 1px rgba(0, 0, 0, 0.1); margin: 5px 0;">
    <h3>🧠 Technologies utilisées</h3>
        <ul style="color: #ccccc; font-size: 16px;">
            <li>GRU (Gated Recurrent Unit) : un réseau neuronal récurrent adapté aux séries temporelles</li>
            <li>NeuralProphet : modèle basé sur Facebook Prophet avec des capacités avancées.</li>
            <li>Streamlit : pour l’interface utilisateur web.</li>
            <li>Pandas, NumPy, Plotly, Matplotlib : pour le traitement des données et la visualisation.</li>
            <li>Yahoo Finance API : pour l'extraction des données boursières.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)  

    st.markdown("""
    <div style="background-color: #cccccc; border-radius: 15px; padding: 20px; 
                box-shadow: 0 1px 1px rgba(0, 0, 0, 0.1); margin: 5px 0;">
    <h3>📈 Fonctionnalités clés</h3>
        <ul style="color: #34495e; font-size: 16px;">
            <li>📊 Prévisions GRU & NeuralProphet côte à côte.</li>
            <li>🧪 Backtesting sur les dernières semaines.</li>
            <li>💰 Simulateur d'achat d'actions avec calcul du gain ou de la perte estimée.</li>
            <li>🎯 Analyse comparative des performances des modèles.</li>
            <li>🌐 Interface responsive et professionnelle.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)  

    st.markdown("""
    <div style="background-color: #cccccc; border-radius: 15px; padding: 20px; 
                box-shadow: 0 1px 1px rgba(0, 0, 0, 0.1); margin: 5px 0;">
    <h3>👨‍💻 Auteur</h3>
        <ul style="color: #34495e; font-size: 16px;">
            <li>Développé par YEWO FEUTCHOU HIROSHI</li>
            <li>Etudiant en Master 2 Intelligence Artificielle</li>
            <li>Keyce Informatique & IA </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)  

elif selection == "Contact Us":
    st.markdown("""
    <div style="background-color: #28a745; border-radius: 15px; padding: 20px; 
                box-shadow: 0 1px 1px rgba(0, 0, 0, 0.1); margin: 5px 0;">
        <h3 style="color:#e0e0e0; font-weight: bold;text-align:center;">📧 Contacter le concepteur</h3>   
                 </div>  """, unsafe_allow_html=True)
    col1, col2 = st.columns([3,1])
    with col1:

        st.info("Si vous avez des questions ou suggestions, envoyez-nous un e-mail:")
    
    # Formulaire de contact
        email = st.text_input("Votre adresse e-mail")
        subject = st.text_input("Sujet")
        message = st.text_area("Message")
        if st.button("Envoyer"):
            if email and subject and message:
                success = send_email(SENDER_EMAIL, SENDER_PASSWORD, SENDER_EMAIL, f"De {email} : {subject}", message)
                if success:
                    st.success("Message envoyé avec succès !")
                else:
                    st.error("Échec de l'envoi.")
            else:
                st.warning("Tous les champs sont requis.")
    with col2:
        st_lottie(lottie_animation2, speed=1, width=200, height=200, key="lottie8")
