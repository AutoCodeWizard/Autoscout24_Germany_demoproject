# importieren der benötigten Bibliotheken
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LinearLocator
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from PIL import Image

# Farbwörterbuch für die Marken
full_color_dict = {
    'BMW': '#1A76D2', 'Volkswagen': '#1E253F', 'SEAT': '#2dcf12', 'Renault': '#FAB711', 'Peugeot': '#094FA3', 'Toyota': '#EB0A1E',
    'Opel': '#096ce1', 'Mazda': '#f4e23c', 'Ford': '#0C4DA1', 'Mercedes-Benz': '#1A2B3F', 'Chevrolet': '#c3f779', 'Audi': '#1B3E70', 
    'Fiat': '#77094b', 'Kia': '#cd3185', 'Dacia': '#bfc4a6', 'MINI': '#7586da', 'Hyundai': '#da9152', 'Skoda': '#da8424', 
    'Citroen': '#e664d3', 'Infiniti': '#823e3e', 'Suzuki': '#ef4737', 'SsangYong': '#15488c', 'smart': '#af63a8', 'Cupra': '#ffd17e', 
    'Volvo': '#c63291', 'Jaguar': '#a1fdfe', 'Porsche': '#B12D30', 'Nissan': '#983646', 'Honda': '#E40521', 'Lada': '#ffbcb0', 
    'Mitsubishi': '#6903cd', 'Others': '#bc2e11', 'Lexus': '#3f292e', 'Jeep': '#521e5e', 'Maserati': '#a64f2e', 'Bentley': '#67d1b5',
    'Land': '#2f792b', 'Alfa': '#96debe', 'Subaru': '#b68989', 'Dodge': '#724cfe', 'Microcar': '#5aba77', 'Lamborghini': '#37fc84', 
    'Baic': '#874cce', 'Tesla': '#357b2e', 'Chrysler': '#2ff981', '9ff': '#4a7f8b', 'McLaren': '#330237', 'Aston': '#dfe516',
    'Rolls-Royce': '#ec29d9', 'Alpine': '#9e3e9a', 'Lancia': '#b1baf7', 'Abarth': '#7b5765', 'DS': '#9d16e1', 'Daihatsu': '#71ce9b', 
    'Ligier': '#6b4381', 'Ferrari': '#6c0544', 'Caravans-Wohnm': '#46256c', 'Aixam': '#e628a7', 'Piaggio': '#d29b4c', 'Zhidou': '#857459', 
    'Morgan': '#cdc219', 'Maybach': '#ccbae2', 'Tazzari': '#1e3d8b', 'Trucks-Lkw': '#65bcf9', 'RAM': '#3f4f6c', 'Iveco': '#43e2b4', 
    'DAF': '#b14a71', 'Alpina': '#3d0a95', 'Polestar': '#357b7e', 'Brilliance': '#0c1955', 'FISKER': '#670397', 'Cadillac': '#340193', 
    'Trailer-Anhänger': '#b896d8', 'Isuzu': '#74a397', 'Corvette': '#9be3da', 'DFSK': '#42d4ee', 'Estrima': '#133aa0'
}

# Streamlit Cache
@st.cache_data
@st.cache_resource

# Daten laden
def load_data():
    """
    Lädt die Daten aus einer externen CSV-Datei von GitHub und gibt einen DataFrame zurück.
    
    Returns:
        pd.DataFrame: DataFrame mit den geladenen Daten
    """
    url = "https://raw.githubusercontent.com/AutoCodeWizard/Autoscout24_Germany_demoproject/3febc23de7b156ca8f7a328029cef644bb06c73a/autoscout24.csv"
    df = pd.read_csv(url)
    return df


def clean_data(df):
    """
    Reinigt den DataFrame und bereitet ihn für die Analyse vor.

    Args:
        df (pd.DataFrame): Ursprünglicher DataFrame

    Returns:
        pd.DataFrame: Bereinigter DataFrame
    """

    # Zeilen mit fehlenden oder null Werten entfernen
    df = df.dropna().reset_index(drop=True)

    # Ungültige Zeilen aus der Spalte 'hp' entfernen
    df = df[df['hp'] != 'null']

    # Den Datentyp der Spalte 'hp' in int ändern
    df['hp'] = df['hp'].astype(int)

    # Zeilen mit ungewöhnlich hohen oder niedrigen Werten entfernen
    df = df[(df['mileage'] > 0) & (df['mileage'] < 2000000)]
    df = df[(df['price'] > 100) & (df['price'] < 3000000)]
    df = df[(df['hp'] > 1) & (df['hp'] < 3000)]

    return df

def plot_avg_car_price(df):
    """
    Erstellt einen Plot, der die durchschnittlichen Verkaufspreise von Autos sowie deren Preissteigerung über die Jahre darstellt.

    Args:
        df (pandas.DataFrame): Dataframe, das die Autodaten enthält. Es muss eine Spalte 'year' und eine Spalte 'price' geben.

    Returns:
        None: Die Funktion gibt nichts zurück, aber sie zeigt ein Plot an.
    """
    sns.set_style("darkgrid")

    # Daten aufbereiten
    avg_price_per_year = df.groupby('year')['price'].mean()
    price_increase_percent = (avg_price_per_year / avg_price_per_year.loc[2011] - 1) * 100
    
    def plus_percent(x, pos):
        """
        Hilfsfunktion für die % Formatierung der y-Achse.
        """
        return f"+{int(x)}%"
    
    def hide_lowest_tick(ax):
        """
        Versteckt die Beschriftung des niedrigsten Ticks auf der y-Achse.
        """
        yticks = ax.yaxis.get_major_ticks()
        if yticks:
            yticks[0].label2.set_visible(False)

    def thousands(x, pos):
        """
        Hilfsfunktion für die Tausenderformatierung der y-Achse.
        """
        return '%1.0fk' % (x * 1e-3)
    
    # Plot erstellen
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.lineplot(x=avg_price_per_year.index,
                 y=avg_price_per_year.values, 
                 ax=ax, color='#2980b9', 
                 linewidth=2.5)
    ax.set_title('Jährlicher Durchschnittspreis der auf autoscout24.de verkauften Fahrzeuge im Zeitraum von 2011 - 2021', 
                color='#555867', fontsize=12, fontweight='bold')
    ax.set_xlabel('Jahr', fontsize=12)
    ax.set_ylabel('Durchschnittlicher Verkaufspreis €', fontsize=12)
    ax.set_xticks(range(min(avg_price_per_year.index), max(avg_price_per_year.index) + 1))
    formatter = FuncFormatter(thousands)
    ax.yaxis.set_major_formatter(formatter)
    axes2 = ax.twinx()
    sns.lineplot(x=price_increase_percent.index,
                 y=price_increase_percent.values, 
                 ax=axes2, color='#2980b9', 
                 linewidth=2.5)
    axes2.set_ylabel('Preissteigerung seit 2011 (%)', fontsize=12)
    axes2.yaxis.set_major_formatter(FuncFormatter(plus_percent))
    axes2.tick_params(axis='y', which='both', length=0)
    ax.yaxis.set_major_locator(LinearLocator(13))
    axes2.yaxis.set_major_locator(LinearLocator(13))
    hide_lowest_tick(axes2)
    initial_avg_price = avg_price_per_year.loc[2011]
    # Linien für die Preissteigerung hinzufügen
    #colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    #for i, factor in enumerate([2, 3, 4]):
    #    y_value = initial_avg_price * factor
    #    color = colors[i]
    #    ax.axhline(y=y_value, color=color, linestyle='--')
    #    ax.text(min(avg_price_per_year.index),
    #             y_value, 
    #             f"{factor}x des Preises von 2011", 
    #             verticalalignment='bottom', 
    #             horizontalalignment='left', 
    #             color=color, fontsize=12)
    #    x_values = np.array(avg_price_per_year.index)
    #    y_values = np.array(avg_price_per_year.values)
    #    intersections = find_intersections(x_values, y_values, y_value)
    #    for intersect in intersections:
    #        ax.axvline(x=intersect, 
    #                   ymin=0, 
    #                   ymax=(y_value - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0]), 
    #                   linestyle='--', color=color)
    plt.tight_layout()
    st.pyplot(fig)

def plot_car_data(df):
    # Erstellung eines DataFrame für Gebrauchtwagen
    used_cars_df = df[df['offerType'] == 'Used']
    num_used_cars_per_year = used_cars_df.groupby('year').size()

    # Erstellung eines DataFrame für restliche offerTypes
    filtered_offer_types = ['Demonstration', "Employee's car", 'Pre-registered', 'New']
    filtered_df = df[df['offerType'].isin(filtered_offer_types)]
    translation_dict = {
        'Demonstration': 'Vorführwagen',
        "Employee's car": 'Firmenfahrzeuge',
        'Pre-registered': 'Tageszulassungen',
        'New': 'Neuwagen'
    }
    filtered_df.loc[:, 'offerType'] = filtered_df['offerType'].map(translation_dict)
    num_cars_per_year_offerType_filtered = filtered_df.groupby(['year', 'offerType']).size().reset_index(name='count')

    # Erstellung der Subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))

    # Plot für die Anzahl der verkauften Gebrauchtwagen pro Jahr
    sns.barplot(x=num_used_cars_per_year.index, y=num_used_cars_per_year.values, ax=axs[0], palette='viridis')
    axs[0].set_title("Anzahl der auf autoscout24.de verkauften Gebrauchtwagen", fontsize=16, fontweight='bold')
    axs[0].set_xlabel("Jahr", fontsize=12)
    axs[0].set_ylabel("Anzahl der Autos", fontsize=12)

    # Plot für die Anzahl der verkauften Exklusiv Fahrzeugangebote pro Jahr
    custom_palette = ['#0B3D91', '#28A745', '#AAB7B8', '#D35400']
    sns.barplot(x='year', y='count', hue='offerType', data=num_cars_per_year_offerType_filtered, ax=axs[1], palette=custom_palette)
    axs[1].set_title("Anzahl der auf autoscout24.de verkauften exklusiv Fahrzeugangebote", fontsize=16, fontweight='bold')
    axs[1].set_xlabel("Jahr", fontsize=12)
    axs[1].set_ylabel("Anzahl der Autos", fontsize=12)
    axs[1].legend(title='Angebotstyp', title_fontsize='20', loc='upper left', fontsize='15')

    plt.tight_layout()
    st.pyplot(fig)  # Figur an st.pyplot übergeben

def plot_ecdf(data, feature, color='skyblue'): # brauche ich eigenlich hier gar nicht, aber lasse es erstmal drin
        """
        Plottet die empirische kumulative Verteilungsfunktion (ECDF) für ein gegebenes Feature.
        
        Args:
            data (pd.DataFrame): Der Datensatz, der das Feature enthält.
            feature (str): Der Name des Features, für das die ECDF geplottet werden soll.
            color (str): Die Farbe des ECDF-Plots.
        
        Returns:
            None
        """
        # Daten für das Feature sortieren
        x = np.sort(data[feature])
        # ECDF-Werte berechnen
        y = np.arange(1, len(x) + 1) / len(x)
        # ECDF plotten
        plt.plot(x, y, marker='.', linestyle='none', color=color)
        plt.title(f'ECDF of {feature}')
        plt.xlabel(feature)
        plt.ylabel('ECDF')

def predict_car_price(mileage, hp, year, make, model, fuel, gear, model_filtered, label_encoder_dict):
    """
    Sagt den Verkaufspreis eines Autos vorher, das vom Benutzer definiert wird.

    Args:
        mileage (int): Die Laufleistung des Autos.
        hp (int): Die Pferdestärken des Autos.
        year (int): Das Baujahr des Autos.
        make (str): Der Hersteller des Autos.
        model (str): Das Modell des Autos.
        fuel (str): Der Kraftstoff des Autos.
        gear (str): Das Getriebe des Autos.
    
    Returns:
        str: Die Vorhersage des Verkaufspreises des Autos.
    """
    # Überprüfen, ob die Eingaben im Trainingsdatensatz vorhanden sind
    for feature_name, encoder in label_encoder_dict.items():
        if feature_name == 'make':
            if make not in encoder.classes_:
                return f"Der Hersteller {make} wurde im Trainingsdatensatz nicht gefunden. Bitte einen anderen Hersteller wählen."
        if feature_name == 'model':
            if model not in encoder.classes_:
                return f"Das Modell {model} wurde im Trainingsdatensatz nicht gefunden. Bitte ein anderes Modell wählen."
        # weitere Überprüfungen für 'fuel' und 'gear' hinzufügen
        if feature_name == 'fuel':
            if fuel not in encoder.classes_:
                return f"Der Kraftstoff {fuel} wurde im Trainingsdatensatz nicht gefunden. Bitte einen anderen Kraftstoff wählen."
        if feature_name == 'gear':
            if gear not in encoder.classes_:
                return f"Das Getriebe {gear} wurde im Trainingsdatensatz nicht gefunden. Bitte ein anderes Getriebe wählen."  

    user_data = np.array([[mileage, hp, year,
                           label_encoder_dict['make'].transform([make])[0],
                           label_encoder_dict['model'].transform([model])[0],
                           label_encoder_dict['fuel'].transform([fuel])[0],
                           label_encoder_dict['gear'].transform([gear])[0]]])
    
    predicted_price = model_filtered.predict(user_data)
    if predicted_price[0] < 0:
        predicted_price[0] = 0
    
    return f"{predicted_price[0]:.2f} €"


# Initialisieren des LabelEncoders für jede kategorische Variable
label_encoder_dict = {
    'make': LabelEncoder(),
    'model': LabelEncoder(),
    'fuel': LabelEncoder(),
    'gear': LabelEncoder()
}

def display_image(image_url):
    """
    Lädt und zeigt ein Bild aus einer URL an.

    Args:
        image_url (str): URL des Bildes, das angezeigt werden soll.

    Returns:
        None
    """
    st.image(image_url, caption=None, width=None, use_column_width=True, clamp=None, channels="RGB", output_format="auto")

image_url_1 = "https://raw.githubusercontent.com/AutoCodeWizard/Autoscout24_Germany_demoproject/3febc23de7b156ca8f7a328029cef644bb06c73a/Bilder/bild_1.png"
image_url_2 = "https://raw.githubusercontent.com/AutoCodeWizard/Autoscout24_Germany_demoproject/3febc23de7b156ca8f7a328029cef644bb06c73a/Bilder/bild_2.jpg"
image_url_3 = "https://raw.githubusercontent.com/AutoCodeWizard/Autoscout24_Germany_demoproject/3febc23de7b156ca8f7a328029cef644bb06c73a/Bilder/bild_3.jpg"
image_url_4 = "https://raw.githubusercontent.com/AutoCodeWizard/Autoscout24_Germany_demoproject/3febc23de7b156ca8f7a328029cef644bb06c73a/Bilder/bild_4.jpg"
image_url_5 = "https://raw.githubusercontent.com/AutoCodeWizard/Autoscout24_Germany_demoproject/3febc23de7b156ca8f7a328029cef644bb06c73a/Bilder/bild_5.png"


# Hauptfunktion der Streamlit App
def main():
    # Streamlit Einstellungen
    st.set_page_config(layout='wide')

    left, middle, right = st.columns(3)

    with left:
        display_image(image_url_1)

    with middle:
        st.title(':grey[Datenanalyse]')

    with right:
        display_image(image_url_2)
  
# Daten laden
    df = load_data()

# Daten bereinigen
    df = clean_data(df)

    st.markdown('Datenbasis: https://www.kaggle.com/datasets/ander289386/cars-germany')
    st.markdown('Application Ersteller: https://www.linkedin.com/in/lukas-hamann-76b736295')

    # Zählen der Anzahl der Autos für jeden Hersteller
    most_common_makes = df['make'].value_counts()

    # Datensatz filtern, um alle vorkommenden Hersteller zu behalten
    df_filtered = df[df['make'].isin(most_common_makes.index)]

        
# ML Modell für Preisvorhersage / Es werden nur die fünf am häufigsten vorkommenden Hersteller verwendet
    st.header("Machine Learning für zukünftige Preisvorhersage:")

    # Kategorische Variablen in numerische umwandeln
    for col in ['make', 'model', 'fuel', 'gear']:
        df_filtered[col] = label_encoder_dict[col].fit_transform(df_filtered[col])

    # Features und Zielvariable
    features = df_filtered[['mileage', 'hp', 'year', 'make', 'model', 'fuel', 'gear']]
    target = df_filtered['price']

    # Daten aufteilen und Modell trainieren
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Dropdown für den Hersteller
    col1, col2, col3 = st.columns(3)

    with col1:
        make_input = st.selectbox("Hersteller:", most_common_makes.index)
        # Dynamische Aktualisierung der Modelle auf Grundlage des ausgewählten Herstellers
        unique_models_for_make = df[df['make'] == make_input]['model'].unique()
        model_input = st.selectbox("Modell:", unique_models_for_make)
        year_input = st.selectbox("Baujahr:", sorted(df['year'].unique(), reverse=True))

    with col2:
        fuel_input = st.selectbox("Kraftstoff:", df['fuel'].unique())
        gear_input = st.selectbox("Getriebe:", df['gear'].unique())

    with col3:
        mileage_input = st.number_input("Laufleistung (in km):", min_value=0, max_value=1_000_000, value=50_000)
        hp_input = st.number_input("Pferdestärken:", min_value=0, max_value=1000, value=100)
        

    # Vorhersage und Darstellung
    predicted_price = predict_car_price(mileage_input, hp_input, year_input, make_input, model_input, fuel_input, gear_input, model, label_encoder_dict)
   
    col1, col2, col3 = st.columns(3)
    with col2:
        st.markdown("#### Vorhergesagter Preis:")
        
    with col3:
        st.markdown(f"# {predicted_price}")
 

    with col1:
        # Modellbewertung
        y_pred = model.predict(X_test)
        y_pred[y_pred < 0] = 0  # Negative Preise auf 0 setzen
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        st.write(f"MSE: {mse:.2f}")
        st.write(f"RMSE: {rmse:.2f}")
        st.write(f"R2 Score: {r2:.2f}")

# teilung durch Bild
    display_image(image_url_3)

# Durchschnittspreis pro Jahr und Preissteigerung im Vergleich zu 2011
    plot_avg_car_price(df)

# Marken Fahrzeugverkäufe pro Jahr Vergleich 2011-2021
    st.subheader(':grey[Marken Fahrzeugverkäufe pro Jahr im Vergleich 2011-2021:]')

    # Erstellen der Spalten für die Dropdown-Menüs
    col1, col2, col3 = st.columns(3)

    # Sortieren der Marken alphabetisch
    unique_brands = sorted(df['make'].unique())

    # Erstellen der Dropdown-Auswahl für die Automarken mit default-Werten VW BMW und Mercedes-Benz
    selected_brand1 = col1.selectbox('', unique_brands, index=unique_brands.index('Chevrolet'))
    selected_brand2 = col2.selectbox('', unique_brands, index=unique_brands.index('Toyota')) 
    selected_brand3 = col3.selectbox('', unique_brands,  index=unique_brands.index('Mercedes-Benz'))
        
    # Filtern der Daten für die ausgewählten Marken
    df_filtered1 = df[df['make'] == selected_brand1]
    df_filtered2 = df[df['make'] == selected_brand2]
    df_filtered3 = df[df['make'] == selected_brand3]

    # Zählen der Einträge für jedes Jahr
    brand_year_count1 = df_filtered1.groupby('year').size().reset_index(name='count')
    brand_year_count1['brand'] = selected_brand1

    brand_year_count2 = df_filtered2.groupby('year').size().reset_index(name='count')
    brand_year_count2['brand'] = selected_brand2

    brand_year_count3 = df_filtered3.groupby('year').size().reset_index(name='count')
    brand_year_count3['brand'] = selected_brand3

    # Kombinieren der drei DataFrames
    combined_brand_year_count = pd.concat([brand_year_count1, brand_year_count2, brand_year_count3])

    # Erstellen des Barplots
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.barplot(x='year', y='count', hue='brand', data=combined_brand_year_count, ax=ax, palette=full_color_dict)
    plt.xlabel('Jahr')
    plt.ylabel('Anzahl der Verkäufe')

    # Ändern des Legendentitels und der Schriftgröße
    # Positionieren der Legende außerhalb des Plots
    leg = ax.legend(title='Hersteller', title_fontsize='10', labelspacing=0.5, fontsize='8', loc='upper left', bbox_to_anchor=(1, 1))

    # Anpassen des Layouts, um Platz für die Legende zu schaffen
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    st.pyplot(fig)

# teilung durch Bild
    display_image(image_url_4)

# Numerische Features
    st.header('numerische Features aus dem Datensatz:')

    image_5 = Image.open('C:/Users/lukas/Projekte/Autoscout24_Germany_demoproject/Bilder/bild_5.png')
    st.image(image_5, caption=None, width=None, use_column_width=True, clamp=None, channels="RGB", output_format="auto")

    st.markdown('''
    ### Analyse der Scatterplots

    - **mileage vs price**: Es gibt eine schwache negative Korrelation zwischen dem Kilometerstand (`mileage`) und dem Preis (`price`). Autos mit geringerem Kilometerstand neigen dazu, teurer zu sein.

    - **mileage vs hp**: Keine deutliche Korrelation.

    - **mileage vs year**: Es gibt eine negative Korrelation zwischen dem Kilometerstand (`mileage`) und dem Baujahr (`year`). Neuere Autos haben weniger Kilometer zurückgelegt.

    - **price vs hp**: Es gibt eine positive Korrelation zwischen dem Preis (`price`) und den Pferdestärken (`hp`). Autos mit mehr PS sind teurer.

    - **price vs year**: Es gibt eine leichte positive Korrelation zwischen dem Preis (`price`) und dem Baujahr (`year`). Neuere Autos sind teurer als ältere.

    - **hp vs year**: Keine deutliche Korrelation.

    Die Farben in den Scatterplots repräsentieren das Baujahr der Autos (`year`). Es ist ersichtlich, dass neuere Autos (hellere Punkte) in der Regel teurer sind und weniger Kilometer zurückgelegt haben.
    Das haben wir uns sicher schon gedacht, aber die Daten bestätigen es auch nochmal. 😄
    ''')


# teilung durch Bild
    display_image(image_url_3)

# Streamlit App ausführen
if __name__ == "__main__":
    main()
