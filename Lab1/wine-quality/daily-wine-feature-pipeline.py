import os
import random
import modal
import hopsworks

LOCAL = False

if LOCAL == False:
    stub = modal.Stub()
    image = modal.Image.debian_slim().pip_install(["hopsworks"]) 

    @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
    def f():
        g()

# Generate a synthetic wine using random values between a min and max range
def generate_wine(quality, type_min, type_max, fixed_acidity_min, fixed_acidity_max,
                  volatile_acidity_min, volatile_acidity_max, citric_acid_min, citric_acid_max, 
                  residual_sugar_min, residual_sugar_max, chlorides_min, chlorides_max, 
                  free_sulfur_dioxide_min, free_sulfur_dioxide_max, total_sulfur_dioxide_min, 
                  total_sulfur_dioxide_max, density_min, density_max, ph_min, ph_max, 
                  sulphates_min, sulphates_max, alcohol_min, alcohol_max):
    
    import pandas as pd
    import random

    wine_df = pd.DataFrame({
        "type": [random.randint(type_min, type_max)],
        "fixed_acidity": [random.uniform(fixed_acidity_min, fixed_acidity_max)],
        "volatile_acidity": [random.uniform(volatile_acidity_min, volatile_acidity_max)],
        "citric_acid": [random.uniform(citric_acid_min, citric_acid_max)],
        "residual_sugar": [random.uniform(residual_sugar_min, residual_sugar_max)],
        "chlorides": [random.uniform(chlorides_min, chlorides_max)],
        "free_sulfur_dioxide": [random.uniform(free_sulfur_dioxide_min, free_sulfur_dioxide_max)],
        "total_sulfur_dioxide": [random.uniform(total_sulfur_dioxide_min, total_sulfur_dioxide_max)],
        "density": [random.uniform(density_min, density_max)],
        "ph": [random.uniform(ph_min, ph_max)],
        "sulphates": [random.uniform(sulphates_min, sulphates_max)],
        "alcohol": [random.uniform(alcohol_min, alcohol_max)]
        #"quality": [random.randint(quality_min, quality_max)]
    })

    wine_df['quality'] = int(quality)
    return wine_df

# Generate random flower with the min values and max values for each column as the boundries
def get_random_wine():
    quality = random.uniform(3,8)
    wine = generate_wine(quality,0,1,3.8,15.9,0.08,1.58,0,1.66,0.6,65.8,0.009,0.611,1,289,6,440,0.987110,1.038980,2.72,4.01,0.22,2,8,14.9)
    print("Wine added")
    print(wine)
    return wine

# Log in to Hopsworks, get feature store and insert the new synthtic wine into the feature store
def g():
    project = hopsworks.login()
    fs = project.get_feature_store()
    wine_df = get_random_wine()
    wine_fg = fs.get_feature_group(name="wine",version=1)
    wine_fg.insert(wine_df)

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        #stub.deploy("wine_daily")
        with stub.run():
            f.remote()
