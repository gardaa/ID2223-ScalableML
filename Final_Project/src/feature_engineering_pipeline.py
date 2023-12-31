import pandas as pd
import hopsworks
from IPython.display import display
import matplotlib.pyplot as plt
from datetime import datetime

# Login to Hopsworks and get the feature store handle
def hopsworks_login_and_upload(df):
    HOPSWORKS_API_KEY = "zB1HFw6waUQEsgoH.zTP79bPsYXZzR1hZN8L5lF7NJmgIJNR6ji7r4HenjkeeSel2MVi6Ca61AbrzGOy8"
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    try:
        icelandic_house_price_fg = fs.get_or_create_feature_group(
            name="icelandic_house_prices",
            version=1,
            primary_key=["postalcode", "price", "year", "area", "rooms", "type"],
            description="icelandic house price dataset"
        )

        icelandic_house_price_fg.insert(df, overwrite=True)

        print("Feature group created/loaded and data inserted successfully")

    except Exception as e:
        print("Error:", e)

def display_data(df):
    display(df)
    df.info()
    display(df.describe())

def feature_extraction(df):
    ihp_df_copy = df.copy()
    ihp_df_copy = ihp_df_copy.drop(columns=['FAERSLUNUMER','EMNR','SKJALANUMER','FASTNUM','HEIMILISFANG',
                                            'HEINUM','SVFN','SVEITARFELAG','THINGLYSTDAGS','FASTEIGNAMAT',
                                            'FASTEIGNAMAT_GILDANDI','BRUNABOTAMAT_GILDANDI','FEPILOG',
                                            'LOD_FLM','LOD_FLMEIN','ONOTHAEFUR_SAMNINGUR'], axis=1)
    # features that will be used for filtering but will be dropped before training: FULLBUID (complete)

    return ihp_df_copy

def translate_columns(df):
    column_mapping = {
        'POSTNR': 'postalcode',
        'UTGDAG': 'date',
        'KAUPVERD': 'price',
        'BYGGAR': 'year',
        'EINFLM': 'area',
        'FJHERB': 'rooms',
        'TEGUND': 'type',
        'FULLBUID': 'complete'
    }
    
    df.rename(columns=column_mapping, inplace=True)

    translate_type(df)


    return df

def translate_type(df):
    value_mapping = {
        'Fjölbýli': 0, # Apartment
        'Sérbýli': 1, # Semi-detached
        'Einbýli': 2 # House
    }

    df['type'] = df['type'].replace(value_mapping)

    return df

def data_cleaning(df):
# 1. Change DATE to date format and filter by date, try from 01/01/21
    df = filter_by_date(df)
# 2. Filter out all that have missing values in any of the columns
    df = df.dropna()
# 3. Filter by type, only apartments, semi-detached and house
    df = filter_by_type(df)
# 4. Filter out COMPLETE = 0
# 5. Filter out YEAR = 0
# 6. Filter out ROOMS = 0 and bigger than 15
# 7. Filter out area bigger than 940 (largest private home in Iceland)
# 8. Filter out price larger than 500 million or less than 7 million
    df = filter_outliers(df)
# Drop DATE and COMPLETE column
    df = df.drop(['complete','date'], axis=1)
# PRICE is in 1000s, at some point must multiply column by 1000
    df['price'] = df['price'] * 1000
# Change all columns to int to fix error
    df['rooms'] = df['rooms'].astype(int)
    df['type'] = df['type'].astype(str)
    df['postalcode'] = df['postalcode'].astype(str)
    df['year'] = df['year'].astype(int)
    df['area'] = df['area'].astype(int)
    df['price'] = df['price'].astype(int)
# Define categorical values
    df['postalcode'] = pd.Categorical(df['postalcode'])
    df['type'] = pd.Categorical(df['type'])

    return df

def filter_by_date(df):
    # Convert 'DATE' column to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Define the filtering condition
    filtering_condition = df['date'] >= '2021-01-01'

    # Apply the filter
    df_filtered = df[filtering_condition]

    return df_filtered


def filter_by_type(df):
    allowed_types = [0,1,2]
    df_filtered = df[df['type'].isin(allowed_types)]
    return df_filtered

def filter_outliers(df):
    # Define filtering conditions
    condition_complete = df['complete'] != 0
    condition_year = (df['year'] != 0) & (df['year'] != '    ')
    condition_rooms = (df['rooms'] > 0) & (df['rooms'] < 15)
    condition_area = df['area'] <= 940
    condition_price = (df['price'] <= 500000) & (df['price'] >= 7000)

    # Combine the conditions using bitwise AND (&)
    combined_condition = condition_complete & condition_year & condition_rooms & condition_area & condition_price

    # Apply the filter
    df_filtered = df[combined_condition]

    return df_filtered

# Sort the DataFrame by 'PRICE' column
def plot_price(df,type,number):
    if type=='top':
        top_prices =  df['price'].nlargest(number)
        df_sorted = top_prices.sort_values().reset_index(drop=True)
        plt.plot(df_sorted)

    if type=='bottom':
        bottom_prices =  df['price'].nsmallest(number)
        df_sorted = bottom_prices.sort_values().reset_index(drop=True)
        plt.plot(df_sorted)

    # Plot the 'PRICE' column
    # plt.plot(df_sorted)

    # Add labels and title
    plt.xlabel('Order')
    plt.ylabel('Price')
    plt.title('Line Plot of PRICE Column')

    # Show the plot
    plt.show()


def timestamp_update():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('upload_complete.txt', 'w') as complete_file:
        complete_file.write(timestamp)

def main():
    icelandic_house_prices_df = pd.read_csv('Final_Project/data/kaupskra.csv', sep=';')
    ihp_df1 = feature_extraction(icelandic_house_prices_df)
    ihp_df1 = translate_columns(ihp_df1)
    ihp_df1 = data_cleaning(ihp_df1)
    ihp_df1.to_csv('Final_Project/data/kaupskra_clean.csv')
    hopsworks_login_and_upload(ihp_df1)
    timestamp_update()
    # display_data(ihp_df1)
    # plot_price(ihp_df1,'top',29233)

main()