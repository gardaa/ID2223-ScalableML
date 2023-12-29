import os
import pandas as pd
import requests
from datetime import datetime
from io import StringIO 
import schedule
import time

def fetch_update_csv():
    # URL of the CSV file
    csv_url = "https://frs3o1zldvgn.objectstorage.eu-frankfurt-1.oci.customer-oci.com/n/frs3o1zldvgn/b/public_data_for_download/o/kaupskra.csv"

    # Fetch the CSV data
    response = requests.get(csv_url)
    
    if response.status_code == 200:
        # Load the CSV data into a DataFrame
        df = pd.read_csv(StringIO(response.text))

        # Get the current working directory
        current_dir = os.getcwd()

        # Save the DataFrame to a local CSV file in the GitHub repository
        local_file_path = os.path.join(current_dir, 'kaupskra.csv')
        df.to_csv(local_file_path, index=False)

        print(f"CSV file updated successfully at {datetime.now()}")

    else:
        print(f"Failed to fetch CSV data. Status code: {response.status_code}")

# Schedule the task to run every 23rd of the month
schedule.every().month.day.at("00:00").do(fetch_update_csv)

# Keep the script running to execute scheduled tasks
while True:
    schedule.run_pending()
    time.sleep(1)