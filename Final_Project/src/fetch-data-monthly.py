import os
import pandas as pd
import requests
from datetime import datetime
from io import StringIO 

def fetch_update_csv():
    # URL of the CSV file
    csv_url = "https://frs3o1zldvgn.objectstorage.eu-frankfurt-1.oci.customer-oci.com/n/frs3o1zldvgn/b/public_data_for_download/o/kaupskra.csv"

    # Fetch the CSV data
    response = requests.get(csv_url)
    
    if response.status_code == 200:
        # Load the CSV data into a DataFrame
        df = pd.read_csv(StringIO(response.text))

        # Specify the desired path in your GitHub repository
        relative_path = 'Final_Project/data'
        
        # Get the current working directory
        current_dir = os.getcwd()

        # Save the DataFrame to a local CSV file in the specified path
        path = os.path.join(current_dir, relative_path, 'kaupskra.csv')
        df.to_csv(path, index=False)

        print(f"CSV file updated successfully at {datetime.now()}")

    else:
        print(f"Failed to fetch CSV data. Status code: {response.status_code}")

# Execute the function once per run
fetch_update_csv()