import xarray as xr
import pandas as pd
import glob
import numpy as np
from scipy.interpolate import griddata
import os
import shutil
from shapely.geometry import Point
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib import cm
import cartopy.crs as ccrs
import calendar
from scipy.stats import ttest_ind
from math import ceil
from math import pi
import matplotlib.colors as mcolors
import warnings

# ML imports
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
warnings.filterwarnings("ignore")

# Optional imports
try:
    from lightgbm import LGBMRegressor
    has_lgb = True
except ImportError:
    has_lgb = False

try:
    from catboost import CatBoostRegressor
    has_cat = True
except ImportError:
    has_cat = False

folder_dir = "D:/Climate/Paper_Worth"
"""
## TEMPERATURE
curr_variable = "Temperature (°C)"
curr_dataset = "ERA5"
curr_var = "tas"
curr_folder = "Temperature"
curr_unit="celsius"
curr_unit_sym = "(°C)"
curr_val_range = [-5, 45]
end_year=2022
vmin, vmax = -5, 5
cmap = cm.get_cmap('coolwarm')
## PRECIPITATION
"""
curr_variable = "Precipitation (mm/day)"
curr_dataset = "CHIRPS"
curr_var = "pr"
curr_folder = "Precipitation"
curr_unit="mm/day"
curr_unit_sym = "mm/day"
curr_val_range = [0,10]
end_year=2024
vmin, vmax = -3, 3
cmap = cm.get_cmap('coolwarm_r')

def lat_lon_clip_csv1():
    # Define study region boundaries
    lon_min, lon_max = 55.0, 105.0
    lat_min, lat_max = 0.0, 40.0
    target_resolution = 1.0  # 1° x 1° resolution
    
    # Generate new grid
    new_lons = np.arange(lon_min, lon_max + target_resolution, target_resolution)
    new_lats = np.arange(lat_min, lat_max + target_resolution, target_resolution)
    new_lons, new_lats = np.meshgrid(new_lons, new_lats)
    
    # Get all CMIP6 NetCDF files
    #######################################################################
    #The file path which contains raw nc files 
    nc_files = glob.glob(fr"{folder_dir}\{curr_folder}\1. Raw nc Files\*.nc")
    
    # Ensure the output directory exists
    output_path = fr"{folder_dir}\{curr_folder}\2. Spatial Clipped CSV Files/"
    os.makedirs(output_path, exist_ok=True)
    
    # Loop through all models
    for nc_file in nc_files:
        print(f"Processing: {nc_file}")
        
        # Open NetCDF file
        ds = xr.open_dataset(nc_file)
    
        # Clip to study region
        ds_clipped = ds.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
    
        # Convert time to "YYYY-MM" format
        ds_clipped["time"] = ds_clipped["time"].dt.strftime("%Y-%m")
    
        # Extract unique time steps
        time_steps = ds_clipped["time"].values
    
        # Initialize empty list to store interpolated data
        all_data = []
    
        # Loop through each time step
        for time_step in time_steps:
            print(f"  🔄 Processing time step: {time_step}")
    
            # Select data for the current time step
            ds_time = ds_clipped.sel(time=time_step)
    
            # Convert to DataFrame
            df = ds_time.to_dataframe().reset_index()
    
            # Ensure 'tas' exists in dataset
            if 'tas' not in df.columns:
                raise ValueError(f"Variable 'tas' not found in {nc_file}. Check variable names.")
    
            # Get original lat/lon points and corresponding values
            original_points = df[['lon', 'lat']].values
            original_values = df['tas'].values
    
            # Interpolate using griddata (linear interpolation)
            new_values = griddata(original_points, original_values, (new_lons.ravel(), new_lats.ravel()), method='linear')
    
            # Create DataFrame for the interpolated data
            df_new = pd.DataFrame({
                'time': time_step,
                'lat': new_lats.ravel(),
                'lon': new_lons.ravel(),
                'tas': new_values
            })
    
            # Remove NaN values
            df_new.dropna(inplace=True)
    
            # Append to list
            all_data.append(df_new)
    
        # Concatenate all time steps
        final_df = pd.concat(all_data, ignore_index=True)
    
        # Save as CSV
        csv_filename = os.path.join(output_path, os.path.basename(nc_file).replace('.nc', '.csv'))
        final_df.to_csv(csv_filename, index=False)
    
        print(f"✅ Saved: {csv_filename}")
    
    print("🎉 All files processed successfully!")
    print("################################# Done 1")

def rearrange_folders2():    
    
    # Input folder where all CSVs are present
    input_folder = fr"{folder_dir}\{curr_folder}\2. Spatial Clipped CSV Files/"
    
    # Output base folder where model-specific folders will be created
    output_base_folder = fr"{folder_dir}\{curr_folder}\3. Folder CSV Files/"
    os.makedirs(output_base_folder, exist_ok=True)
    
    # List all CSV files
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    
    for file in csv_files:
        # Extract model name between 'tas_Amon_' and '_historical'
        if file.startswith(f'{curr_var}_Amon_') and '_historical' in file:
            try:
                model_name = file.split(f'{curr_var}_Amon_')[1].split('_historical')[0]
            except IndexError:
                print(f"Skipping file (unexpected format): {file}")
                continue
        else:
            # Handle simple filenames: remove '.csv' extension
            model_name = file.replace('.csv', '')
        # Create model folder if it doesn't exist
        model_folder = os.path.join(output_base_folder, model_name)
        os.makedirs(model_folder, exist_ok=True)
    
        # Move (or copy) the file to the model folder
        src_path = os.path.join(input_folder, file)
        dst_path = os.path.join(model_folder, file)
        
        shutil.move(src_path, dst_path)  # Use shutil.copy() if you prefer copying instead of moving
    
        print(f"✅ Moved {file} to {model_folder}")
    
    print("🎉 All files sorted successfully!")
    print("################################# Done 2")

def csv_merge_concate3():
    # Path to the Big Folder
    big_folder_path = fr"{folder_dir}\{curr_folder}\3. Folder CSV Files"
    
    # List all folders inside Big Folder
    folders = [os.path.join(big_folder_path, name) 
               for name in os.listdir(big_folder_path) 
               if os.path.isdir(os.path.join(big_folder_path, name))]
    
    # Prepare the output folder
    output_folder = fr"{folder_dir}\{curr_folder}\4. Merge Concatenate CSV"
    os.makedirs(output_folder, exist_ok=True)
    
    # Prepare output file names list
    output_file_names = []
    
    for folder_path in folders:
        # Get list of CSV files inside the folder
        files = os.listdir(folder_path)
        csv_files = [f for f in files if f.endswith('.csv')]
    
        if not csv_files:
            print(f"⚠️ No CSV files found in {folder_path}, skipping.")
            continue
    
        # Pick the first file to extract model name
        first_file = csv_files[0]
    
        if first_file.startswith(f'{curr_var}_Amon_') and '_historical' in first_file:
            try:
                model_name = first_file.split(f'{curr_var}_Amon_')[1].split('_historical')[0]
            except IndexError:
                print(f"Skipping file (unexpected format): {first_file}")
                continue
        else:
            # Handle simple filenames: remove '.csv' extension
            model_name = first_file.replace('.csv', '')
    
        output_file_names.append(model_name + ".csv")
    
    # Now process and merge
    for folder_path, output_name in zip(folders, output_file_names):
        print(f"Processing folder: {folder_path} | Output Name: {output_name}")
        output_file = os.path.join(output_folder, output_name)
        print(f"Will save at: {output_file}")
    
        # List all CSV files in the folder
        csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]
    
        if not csv_files:
            print(f"⚠️ No CSV files in {folder_path}, skipping merge.")
            continue
    
        # Read and concatenate all CSV files
        dataframes = []
        for file in csv_files:
            df = pd.read_csv(file)
            dataframes.append(df)
    
        merged_df = pd.concat(dataframes, ignore_index=True)
    
        # Save the merged dataframe
        merged_df.to_csv(output_file, index=False)
        print(f"✅ Merged CSV saved as {output_file}")
    
    print("\n🎉 All folders processed successfully!")
    print("################################# Done 3")

def time_clip4():
    
    # Define the common timeframe
    start_date = "1981-01"
    end_date = "2014-12"

    # Folder containing processed CSV files
    input_folder = fr"{folder_dir}\{curr_folder}\4. Merge Concatenate CSV"
    output_folder = fr"{folder_dir}\{curr_folder}\5. Time Clip"

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Get all CSV files in the folder
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

    # Loop through all CSV files
    for csv_file in csv_files:
        # Load CSV into DataFrame
        df = pd.read_csv(csv_file)

        # Convert "time" column to datetime format for filtering
        df["time"] = pd.to_datetime(df["time"], format="%Y-%m")

        # Filter data within 1981-01 to 2014-12
        df_filtered = df[(df["time"] >= start_date) & (df["time"] <= end_date)]

        # Convert "time" back to string format
        df_filtered["time"] = df_filtered["time"].dt.strftime("%Y-%m")

        # Save the filtered CSV
        output_filename = os.path.join(output_folder, os.path.basename(csv_file))
        df_filtered.to_csv(output_filename, index=False)

        print(f"Saved: {output_filename}")
    print("################################# Done 4")

def conversion_cel5():
    # Set the directory where your original CSV files are stored
    input_dir = fr"{folder_dir}\{curr_folder}\\5. Time Clip"

    # Set the directory where you want to save the converted files
    output_dir = fr"{folder_dir}\{curr_folder}\6. Celsius Files"
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    # Use glob to list all CSV files in the input directory
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))

    # Loop through each CSV file
    for csv_file in csv_files:
        print(f"Processing file: {csv_file}")
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)
        
        # Check if the column "tas" exists
        if "tas" in df.columns:
            # Multiply the "tas" column by 86400 to convert units
            df["tas"] = df["tas"] -273.15            
            # Create a new filename by appending '{curr_unit}' before the file extension
            base_filename = os.path.basename(csv_file)  # Get only the filename
            base, ext = os.path.splitext(base_filename)
            new_filename = base + f"_{curr_unit}" + ext
            output_path = os.path.join(output_dir, new_filename)
            
            # Save the updated DataFrame to the output directory
            df.to_csv(output_path, index=False)
            print(f"✅ Converted and saved as: {output_path}")
        else:
            print(f"⚠️ Column 'tas' not found in {csv_file}. Skipping file.")
    print("################################# Done 5")

def regrid_csv6():
    #Save all the file in Folder  3. Conversion Cel
    # 📂 File Paths
    output_folder = fr"{folder_dir}\{curr_folder}/7. Regrid/" # Folder for regridded CMIP6 data
    cmip6_folder = fr"{folder_dir}\{curr_folder}/6. Celsius Files/"  # Folder containing 7 CMIP6 models
    ERA5_file = r"D:\Climate\Temperature\Temp_Data\4. Regridded/Regridded_ERA5_curr.csv"  # ERA5 dataset
    os.makedirs(output_folder, exist_ok=True)
    # 📂 List of CMIP6 models

    # Use glob to list all CSV files in that directory.
    csv_files = glob.glob(os.path.join(cmip6_folder, "*.csv"))
    print(csv_files)
    # 📌 Read ERA5 Data (This is already regridded)
    df_ERA5 = pd.read_csv(ERA5_file)

    # 📌 Define Common Grid from ERA5
    common_lats = np.sort(df_ERA5["lat"].unique())
    common_lons = np.sort(df_ERA5["lon"].unique())
    common_grid_lons, common_grid_lats = np.meshgrid(common_lons, common_lats)
    for csv_file in csv_files:
        print(f"Processing file: {csv_file}")
        # Read the CSV file into a DataFrame.
        df_model = pd.read_csv(csv_file)
        model_name = os.path.splitext(os.path.basename(csv_file))[0]
    # 🔹 Regrid Each CMIP6 Model to the ERA5 Grid
        # 🔹 Initialize a list to store regridded data
        regridded_data = []

        # Process Each Unique Timestamp
        for time in df_model["time"].unique():
            df_time_slice = df_model[df_model["time"] == time]  # Filter data for current time

            # Extract lat/lon/tas values for interpolation
            cmip6_points = df_time_slice[['lat', 'lon']].values
            cmip6_values = df_time_slice['tas'].values

            # Perform Interpolation to ERA5 Grid
            regridded_values = griddata(cmip6_points, cmip6_values,
                                        (common_grid_lats, common_grid_lons), method='linear')

            # Create DataFrame for Regridded Data
            df_regridded = pd.DataFrame({
                'time': time,  # Assign correct time for each row
                "lat": common_grid_lats.ravel(),
                "lon": common_grid_lons.ravel(),
                "tas": regridded_values.ravel()
            })

            # Drop NaN values caused by interpolation gaps
            df_regridded.dropna(inplace=True)

            # Append to list
            regridded_data.append(df_regridded)

        # 🔹 Combine all regridded data into a single DataFrame
        df_final = pd.concat(regridded_data, ignore_index=True)

        # Save the regridded CMIP6 model
        regridded_path = output_folder + f"{model_name}_Regridded.csv"
        df_final.to_csv(regridded_path, index=False)

        print(f"✅ {model_name} regridded and saved to: {regridded_path}")

    print("✅ All CMIP6 models successfully regridded to match ERA5!")
    print("################################# Done 6")
       
def to_polygonwise7():
    # ---------- Step 1: Load Shapefile ----------
    shapefile_path = r"D:\Climate\Analysis\Study Region ROI\ROI_CLIM_PROJ.shp"
    gdf = gpd.read_file(shapefile_path)

    # Ensure required columns exist
    if not {'label', 'geometry'}.issubset(gdf.columns):
        raise ValueError("Shapefile is missing required columns: 'label' or 'geometry'")

    # ---------- Step 2: Get List of Model CSV Files ----------
    input_folder = r"D:\Climate\Paper_Worth\Temperature/7. Regrid/" 
    output_folder = r"D:\Climate\Paper_Worth\Temperature/8. Polygonewise/"

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # List of models to process
    model_files = glob.glob(os.path.join(input_folder, f"*_{curr_unit}_Regridded.csv"))  # Assuming filenames follow a pattern

    # ---------- Step 3: Process Each Model ----------
    for model_file in model_files:
        model_name = os.path.basename(model_file).replace(f"_{curr_unit}_Regridded.csv", "")  # Extract model name

        print(f"🔄 Processing Model: {model_name}")

        # Load Temperature CSV
        df = pd.read_csv(model_file)

        # Convert CSV lat/lon to GeoDataFrame
        geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
        precip_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=gdf.crs)

        # Spatial Join to Assign Zones (Only Using `label`)
        joined = gpd.sjoin(precip_gdf, gdf[['label', 'geometry']], how="left", predicate="within")

        # Drop unnecessary columns
        joined = joined[['time', 'label', 'tas']].dropna()

        # Rename 'label' column to 'region_label' for clarity
        joined = joined.rename(columns={'label': 'region_label'})

        print(f"📅 Unique time values for {model_name}: {joined['time'].nunique()}")

        # Compute Spatial Mean for Each Zone
        grouped = joined.groupby(['time', 'region_label'])['tas'].mean().unstack()

        # Rename columns (optional)
        grouped = grouped.rename_axis(None, axis=1).reset_index()

        # Save the New CSV
        output_csv_path = os.path.join(output_folder, f"{model_name}.csv")
        grouped.to_csv(output_csv_path, index=False)

        print(f"✅ Saved: {output_csv_path}")

    print("🎉 All models processed successfully!")
    print("################################# Done 7")
    
def evaluation8(save_dir, eval_type):
    # Ensure output directory exists
    os.makedirs(save_dir, exist_ok=True)

    def monthly_climatology8_1():    
        ##############################################################################
        # 1. Reading & Preprocessing
        ##############################################################################

        def read_data(csv_path):
            
            df = pd.read_csv(csv_path)
            df['time'] = pd.to_datetime(df['time'])
            return df

        def compute_monthly_climatology(df, month):
            
            df_month = df[df['time'].dt.month == month]
            climatology = df_month.drop(columns='time').mean()
            return climatology

        def save_climatology_to_csv(climatologies, month, save_dir):
            
            df_clim = pd.DataFrame({name: series for name, series in climatologies.items()})
            csv_path = os.path.join(save_dir, f"Month_{month:02d}_Climatology_{curr_var}.csv")
            df_clim.to_csv(csv_path, index=True)
            print(f"Saved climatology CSV for Month {month} at {csv_path}")

        ##############################################################################
        # 2. Plotting Choropleth Maps with a Custom Gridspec Colorbar

        def plot_all_choropleths(gdf, climatologies, month, model_names, save_path=None):
            
            # Remove BMME and WMME
            #model_names = [name for name in model_names if name not in ['BMME', 'AMME', 'RMME']]
            
            # Determine global min and max across all datasets (for consistent color scaling)
            all_values = []
            for name in model_names:
                if name in climatologies:
                    all_values.extend(climatologies[name].dropna().values.tolist())
            global_min_specific = min(all_values)
            global_max_specific = max(all_values)
            global_min = curr_val_range[0]
            global_max = curr_val_range[1]
            
            n_plots = len(model_names)
            ncols = 6
            nrows = int(np.ceil(n_plots / ncols))
            
            # Create a gridspec: additional row at bottom for the colorbar.
            gs = gridspec.GridSpec(nrows + 1, ncols, 
                                   height_ratios=[1]*nrows + [0.05],
                                   hspace=0.1, wspace=0.1)
            fig = plt.figure(figsize=(4*ncols, 4*nrows))
            
            # Create axes for each subplot
            axes = []
            for i in range(n_plots):
                ax = fig.add_subplot(gs[i // ncols, i % ncols])
                axes.append(ax)
            
            # Plot each dataset's choropleth
            for i, name in enumerate(model_names):
                if name in climatologies:
                    clim_series = climatologies[name]
                    # Prepare a DataFrame for merging (CSV header must match shapefile's "label")
                    df_clim = pd.DataFrame({'label': clim_series.index, 'Value': clim_series.values})
                    merged = gdf.merge(df_clim, on='label', how='left')
                    merged.plot(ax=axes[i], column='Value', cmap='YlOrBr',
                                vmin=global_min, vmax=global_max, edgecolor='black', legend=False)
                    axes[i].set_title(name, fontsize=10)
                    axes[i].axis('off')
                else:
                    axes[i].set_visible(False)
                    
            # Hide any extra axes
            total_axes = nrows * ncols
            for j in range(n_plots, total_axes):
                ax = fig.add_subplot(gs[j // ncols, j % ncols])
                ax.set_visible(False)
            
            # Create horizontal colorbar in the bottom row of gridspec (across some columns)
            cbar_ax = fig.add_subplot(gs[nrows, 1:ncols-1])
            norm = Normalize(vmin=global_min, vmax=global_max)
            sm = ScalarMappable(cmap='YlOrBr', norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
            cbar.set_label(f"{curr_variable}", fontsize=12)
            cbar.ax.tick_params(labelsize=10)
            
            # Adjust suptitle to be closer to subplots
            fig.suptitle(f'Month {month} Climatology: {curr_variable}', fontsize=16, y=0.98)
            
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show()

        ##############################################################################
        # 3. Pattern Correlation & Taylor Diagram (Without External Library)
        ##############################################################################

        def compute_pattern_correlations(climatologies):
            
            ref = climatologies[curr_dataset]
            correlations = {}
            for model, series in climatologies.items():
                if model != curr_dataset:
                    # Align indices and drop NaNs
                    common = pd.concat([ref, series], axis=1, join='inner').dropna()
                    if len(common) > 1:
                        corr = np.corrcoef(common.iloc[:,0], common.iloc[:,1])[0, 1]
                    else:
                        corr = np.nan
                    correlations[model] = corr
            return correlations

        def compute_tss(std_model, std_obs, correlation):
            
            crmsd = np.sqrt(std_model**2 + std_obs**2 - 2*std_model*std_obs*correlation)
            tss = 1 - (crmsd / (np.sqrt(2)*std_obs))
            return tss

        def plot_taylor_diagram(climatologies, correlations, save_path):
            
            cmap = plt.cm.get_cmap('tab20')
            # Exclude ERA5 and the removed models
            models = [m for m in climatologies.keys() if m != curr_dataset and m not in ['BMME', 'AMME', 'RMME']]
            
            ref_std = np.std(climatologies[curr_dataset].dropna().values)
            std_devs = []
            corrs = []
            for model in models:
                std = np.std(climatologies[model].dropna().values)
                std_devs.append(std)
                corrs.append(correlations.get(model, np.nan))
            
            fig = plt.figure(figsize=(6,6))
            ax = fig.add_subplot(111, polar=True)
            
            # Plot reference point: correlation=1 => theta=0, radius = ref_std
            ax.plot(0, ref_std, 'ko', markersize=10, label=f'{curr_dataset} (Ref)')
            
            # Plot each model with distinct color and compute TSS
            for i, model in enumerate(models):
                if np.isnan(corrs[i]):
                    continue
                theta = np.arccos(corrs[i])  # convert correlation to angle in radians
                r = std_devs[i]
                tss = compute_tss(r, ref_std, corrs[i])
                ax.plot(theta, r, 'o', markersize=8, color=cmap(i), 
                        label=f'{model} (TSS={tss:.2f})')
            
            # Limit to the first quadrant
            ax.set_thetamin(0)
            ax.set_thetamax(90)
            
            # Set xticks corresponding to correlations
            corr_ticks = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
            theta_ticks = [np.arccos(c) for c in corr_ticks]
            ax.set_xticks(theta_ticks)
            ax.set_xticklabels([str(c) for c in corr_ticks])
            
            ax.set_ylabel('Standard Deviation')
            ax.set_title('Taylor Diagram', y=1.05, fontsize=14)
            
            # Place horizontal legend at the bottom
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=8)
            
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved Taylor Diagram at {save_path}")

        ##############################################################################
        # 4. Main Workflow (Looping Over 12 Months)
        ##############################################################################

        def main():
            # Base path for CSV files (adjust as needed)
            input_folder = fr"{folder_dir}\{curr_folder}/8. Polygonewise/"
            model_files = glob.glob(os.path.join(input_folder, "*.csv"))  # Assuming filenames follow a pattern
            model_files.append(fr"{folder_dir}\{curr_folder}\0. Observation - {curr_dataset}\Polygonewise Obs/{curr_dataset}.csv")
            if eval_type=="Ensemble":
                ensemble_folder = glob.glob(os.path.join(fr"{folder_dir}\{curr_folder}/10. Ensemble/", "*.csv"))
                print("Ensemble Folder", ensemble_folder)
                for file in ensemble_folder:
                    model_files.append(file)
            # Construct model names from file names
            model_names = [os.path.basename(f).replace('.csv','') for f in model_files]  
            
            # Read the shapefile (must have a "label" column matching CSV headers)
            shapefile_path = r"D:/Climate/Analysis/Study Region ROI/ROI_CLIM_PROJ.shp"
            gdf = gpd.read_file(shapefile_path)
            
            # Directory to save plots and CSVs
            #save_dir = fr"D:\Climate\Paper_Worth\Temperature\9. Evaluation\9.1 Climatology CSV Plots"
            os.makedirs(save_dir, exist_ok=True)
            
            # Loop over months 1 to 12
            for month in range(1, 13):
                climatologies = {}
                for full_path, name in zip(model_files, model_names):
                    try:
                        
                        df = read_data(full_path)
                        climatologies[name] = compute_monthly_climatology(df, month)
                    except Exception as e:
                        print(f"Error processing {name} for month {month}: {e}")
                
                # Save climatology CSV for the month
                save_climatology_to_csv(climatologies, month, save_dir)
                
                # Plot and save choropleth maps
                choro_save = os.path.join(save_dir, f"Month_{month:02d}_Climatology_{curr_var}.png")
                plot_all_choropleths(gdf, climatologies, month, model_names, save_path=choro_save)
                print(f"Saved choropleth maps for Month {month} to {choro_save}")
                
                # Compute pattern correlations relative to ERA5
                correlations = compute_pattern_correlations(climatologies)
                
                # Plot and save the Taylor diagram
                taylor_save = os.path.join(save_dir, f"Month_{month:02d}_Taylor_{curr_var}.png")
                plot_taylor_diagram(climatologies, correlations, taylor_save)
                
        if __name__ == "__main__":
            main()
    
    def monthwise_ranking8_2():
        ##############################################################################
        # 1. Reading Data & Computing Spatial Stats
        ##############################################################################

        def read_climatology(csv_file):
            
            df = pd.read_csv(csv_file, index_col=0)
            return df

        def compute_spatial_correlation(df, ref_col=curr_dataset):
            
            if ref_col not in df.columns:
                raise ValueError(f"Reference column '{ref_col}' not found in DataFrame.")

            # Reference data
            ref_data = df[ref_col].dropna()
            std_ref = np.std(ref_data.values)

            results = []
            # Add reference entry (correlation = 1.0, TSS = 1.0 by definition)
            results.append({
               "Model": ref_col,
               "Correlation": 1.0,
               "Std_Model": std_ref,
               "Std_Ref": std_ref,
               "TSS": 1.0
            })
            for col in df.columns:
                if col == ref_col:
                    continue
                model_data = df[col].dropna()

                # Align both series on the same index
                common = pd.concat([ref_data, model_data], axis=1, join='inner').dropna()
                if len(common) < 2:
                    corr = np.nan
                else:
                    corr = np.corrcoef(common.iloc[:,0], common.iloc[:,1])[0, 1]

                std_mod = np.std(common.iloc[:,1].values)
                tss = compute_taylor_skill_score(std_mod, std_ref, corr)

                results.append({
                    "Model": col,
                    "Correlation": corr,
                    "Std_Model": std_mod,
                    "Std_Ref": std_ref,
                    "TSS": tss
                })

            return pd.DataFrame(results)

        def compute_taylor_skill_score(std_mod, std_ref, correlation):
            
            crmsd = np.sqrt(std_mod**2 + std_ref**2 - 2.0 * std_mod * std_ref * correlation)
            tss = 1.0 - (crmsd / (np.sqrt(2)*std_ref))
            return tss

        ##############################################################################
        # 2. Plotting a Taylor Diagram
        ##############################################################################

        def plot_taylor_diagram(stats_df, ref_col=curr_dataset, save_path=None):
            
            # We'll use the default colormap or a discrete colormap for distinct colors
            models = stats_df["Model"].values
            corrs = stats_df["Correlation"].values
            std_mods = stats_df["Std_Model"].values
            std_ref = stats_df["Std_Ref"].iloc[0]  # same for all rows

            fig = plt.figure(figsize=(6,6))
            ax = fig.add_subplot(111, polar=True)

            # Plot reference
            ax.plot(0, std_ref, 'ko', markersize=10, label=f'{ref_col} (Ref)')

            # Plot each model
            for i, model in enumerate(models):
                corr = corrs[i]
                r_mod = std_mods[i]
                if np.isnan(corr):
                    continue
                theta = np.arccos(corr)  # correlation -> angle
                tss_val = stats_df["TSS"].iloc[i]
                ax.plot(theta, r_mod, 'o', label=f'{model} (TSS={tss_val:.2f})')

            # Limit to first quadrant (0-90 deg)
            ax.set_thetamin(0)
            ax.set_thetamax(90)

            # Correlation ticks
            corr_ticks = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
            theta_ticks = [np.arccos(t) for t in corr_ticks]
            ax.set_xticks(theta_ticks)
            ax.set_xticklabels([str(t) for t in corr_ticks])

            ax.set_title("Taylor Diagram", fontsize=14)
            ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            ax.set_ylim([0, max(std_mods.max(), std_ref)*1.2])

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Taylor diagram saved to {save_path}")
            else:
                plt.show()

        ##############################################################################
        # 3. Main: Loop Over All 12 CSVs
        ##############################################################################

        def main():
            # We assume the files are named like: Month_01_Climatology.csv, Month_02_Climatology.csv, etc.
            # If they're in a different folder, adjust path. 
            for month in range(1, 13):
                some_folder = fr"{save_dir}/"
                csv_file = os.path.join(some_folder, f"Month_{month:02d}_Climatology_{curr_var}.csv")
               
                if not os.path.exists(csv_file):
                    print(f"File not found: {csv_file}")
                    continue

                # Read the monthly climatology
                df = read_climatology(csv_file)

                # Compute pattern correlation & TSS
                stats_df = compute_spatial_correlation(df, ref_col=curr_dataset)

                # Save the stats to CSV
                stats_csv = fr"{save_dir}\Month_{month:02d}_TaylorStats_{curr_var}.csv"
                stats_df.to_csv(stats_csv, index=False)
                print(f"Taylor diagram data saved to {stats_csv}")
                
                # Rank models by TSS (descending, best=1)
                ranked_df = stats_df.sort_values(by="TSS", ascending=False).reset_index(drop=True)
                ranked_df["Rank"] = ranked_df.index + 1  # best=1
                print(f"\nRanking by TSS for Month {month:02d} (1=best):\n", ranked_df[["Model","TSS","Rank"]])

                # Optionally save the ranking as well
                ranked_csv = fr"{save_dir}/Month_{month:02d}_TaylorRanking_{curr_var}.csv"
                ranked_df.to_csv(ranked_csv, index=False)
                print(f"Ranking saved to {ranked_csv}\n{'-'*60}")

        if __name__ == "__main__":
            main()

    def overall_ranking8_3():
        # If your CSVs are in another folder, adjust or prepend the path
        ranking_files = [
            f"Month_{m:02d}_TaylorRanking_{curr_var}.csv" for m in range(1, 13)
        ]
        
        # Dictionary to accumulate TSS values for each model
        # Key = model name, Value = list of TSS from each month
        tss_data = {}
        
        # Loop through each monthly ranking CSV
        for csv in ranking_files:
            csv_file= fr"{save_dir}/{csv}"
            if not os.path.exists(csv_file):
                print(f"File not found: {csv_file} (skipping)")
                continue
            
            df = pd.read_csv(csv_file)  # columns: Model, TSS, Rank, ...
            
            # Accumulate TSS for each model
            for _, row in df.iterrows():
                
                model = row["Model"]            
                tss_val = row["TSS"]
                if pd.isna(tss_val):
                    continue
                if model not in tss_data:
                    tss_data[model] = []
                tss_data[model].append(tss_val)
        
        # Now compute the average TSS for each model across the months
        results = []
        for model, tss_list in tss_data.items():
            if len(tss_list) == 0:
                avg_tss = float('nan')
            else:
                avg_tss = sum(tss_list) / len(tss_list)
            results.append({"Model": model, "Avg_TSS": avg_tss, "Months_Count": len(tss_list)})
        
        # Convert to DataFrame and sort by average TSS descending
        results_df = pd.DataFrame(results)
        results_df.sort_values(by="Avg_TSS", ascending=False, inplace=True)
        results_df.reset_index(drop=True, inplace=True)
        
        print("Overall Model Performance (sorted by Avg_TSS descending):")
        print(results_df)
        results_df.to_csv(fr"{save_dir}/Overall_Ranking.csv", index=False)
        
        # Get the top 5
        top5_df = results_df.head(6)
        print("\nTop 5 Best Performing Models (historical):")
        print(top5_df)
        
        # Save top 5 to a CSV
        out_csv = fr"{save_dir}/Overall_Top5.csv"
        top5_df.to_csv(out_csv, index=False)
        print(f"\nTop 5 models saved to {out_csv}")
        top5_models = top5_df["Model"].iloc[:5].apply(lambda x: f"{x}.csv").tolist()
        print("TOP 5 Models Array: ", top5_models)
        return top5_models
    
    
    if __name__ == "__main__":
        monthly_climatology8_1() 
        monthwise_ranking8_2()
        overall_ranking8_3()
        top5_models=overall_ranking8_3()
        
        print("################################# Done 8")
        return top5_models
    
        
def ensemble_make9(top5_models):
        # BMME: files to exclude (only filenames)
   # List of BMME files (only file names, not full paths)
    bmme_files = top5_models

    # Input and output folders
    input_folder = r"D:\Climate\Paper_Worth\Temperature\8. Polygonewise"
    output_folder = r"D:\Climate\Paper_Worth\Temperature\10. Ensemble"
    os.makedirs(output_folder, exist_ok=True)

    # List all CSV files (AMME)
    amme_files = glob.glob(os.path.join(input_folder, "*.csv"))

    # BMME full paths
    bmme = [os.path.join(input_folder, f) for f in bmme_files]

    # RMME: Files in AMME excluding BMME
    rmme = [f for f in amme_files if os.path.basename(f) not in bmme_files]

    # Create a dictionary for convenience
    models = {
        "AMME": amme_files,
        "BMME": bmme,
        "RMME": rmme
    }

    # Function to load and prepare a dataframe
    def load_and_format_csv(file_path):
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, errors='coerce')  # Handle bad dates
        df.interpolate(method='linear', inplace=True)         # Fill missing values
        return df

    # Process each model group
    for model_name, file_list in models.items():
        print(f"\nProcessing {model_name.upper()} ensemble with {len(file_list)} files...")

        # Load all DataFrames
        model_dfs = [load_and_format_csv(f) for f in file_list]
        print(len(model_dfs))
        # Check shapes before stacking
        for i, df in enumerate(model_dfs):
            print(f"Model {i+1}: {os.path.basename(file_list[i])} --> shape = {df.shape}")

        # Stack into a 3D array
        model_array = np.stack([df.values for df in model_dfs], axis=-1)

        # Compute ensemble mean
        ensemble_mean = np.nanmean(model_array, axis=-1)

        # Create DataFrame
        ensemble_df = pd.DataFrame(ensemble_mean, index=model_dfs[0].index, columns=model_dfs[0].columns)

        # Format index as 'YYYY-MM'
        ensemble_df.index = ensemble_df.index.strftime('%Y-%m')

        # Save the ensemble
        output_file = os.path.join(output_folder, f"{model_name}.csv")
        try:
            ensemble_df.to_csv(output_file)
            print(f"✅ Saved {model_name}.csv successfully at {output_file}")
        except PermissionError as e:
            print(f"❌ PermissionError: {e}")

    print("\n🎉 All ensembles (AMME, BMME, RMME) created and saved successfully!")
    print("################################# Done 9")

def annual_Temperature_plot10():
    
    # Define markers and colors
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'X']
    colors = plt.cm.get_cmap("gist_ncar", 25)
    # Input and output folders
    input_folder = fr"{folder_dir}\{curr_folder}\8. Polygonewise"
    ensemble_folder = fr"{folder_dir}\{curr_folder}\10. Ensemble"
    obs_folder = fr"{folder_dir}\{curr_folder}\0. Observation - {curr_dataset}\Polygonewise Obs"

    model_files = glob.glob(os.path.join(input_folder, "*.csv"))
    ensemble_files = glob.glob(os.path.join(ensemble_folder, "*.csv"))

    era5_file = glob.glob(os.path.join(obs_folder, "*.csv"))
    model_files.extend(ensemble_files)
    model_files.extend(era5_file)
    print("Files:", model_files, len(model_files))

    model_names=["ACCESS-CM2", "ACCESS-ESM1-5", "AWI-ESM-1-1-LR", "BCC-ESM1", 
                 "CESM2-WACCM", "CMCC-CM2-HR4", "EC-Earth3-CC", "FGOALS-f3-L","GFDL-ESM4",
                 "GISS-E2-1-G","IITM-ESM", "INM-CM4-8", "IPSL-CM6A-LR", "KACE-1-0-G","KIOST-ESM",
                 "MIROC6", "MPI-ESM-1-2-HAM", "NESM3", "SAM0-UNICON","TaiESM1",
                 "AMME",  "BMME","RMME", f"{curr_dataset}"
                 ]
    # Read and process data
    all_data = []
    for i, file in enumerate(model_files):
        df = pd.read_csv(file)
        df['time'] = pd.to_datetime(df['time'])
        df['month'] = df['time'].dt.month
        df['mean_pr'] = df.drop(columns=['time'], errors='ignore').mean(axis=1)
        monthly_mean = df.groupby('month')['mean_pr'].mean()
        all_data.append(monthly_mean)

    # Convert to DataFrame
    combined_df = pd.DataFrame(all_data).T
    combined_df.columns = model_names

    # Define special cases for line plots
    line_models = ["AMME", "BMME", "RMME",]
    line_obs = [f"{curr_dataset}"]

    # Create the plot
    plt.figure(figsize=(12, 7))  # Bigger figure to avoid legend overlap

    # Scatter plot for individual models
    for i, col in enumerate(combined_df.columns[:-4]):  # Exclude last 4 models
        plt.scatter(combined_df.index, combined_df[col], marker=markers[i % len(markers)], 
                    color=colors(i), label=col, alpha=0.7, s=50, edgecolors='black',linewidths=0.8)

    # Line plot for ensemble means and ERA5
    for col in line_models:
        if col=="AMME" or col=="RMME":
           plt.plot(combined_df.index, combined_df[col],  linestyle='--', marker='o', linewidth=1, label=col) 
        else:
            plt.plot(combined_df.index, combined_df[col],  linestyle='-',marker='o', linewidth=2, label=col)
    # Line plot for ensemble means and ERA5
    for col in line_obs:
        plt.plot(combined_df.index, combined_df[col],  linestyle='-', color='black', marker='o', linewidth=2, label=col)

    # Labels and title
    plt.xlabel("Month")
    plt.ylabel(f"{curr_variable}")
    plt.xticks(ticks=np.arange(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.title(f"Monthly Mean {curr_folder} - Models vs Observations")

    # Grid and legend
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=8, ncol=4, frameon=False)
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.savefig(fr"{folder_dir}\{curr_folder}\10. Ensemble/Annual-{curr_folder}.png")
    plt.show()
    print("################################# Done 10")
    
    

    # Extract ensemble models and OBS
    obs = combined_df[f'{curr_dataset}']
    models_to_compare = ['AMME', 'BMME', 'RMME']

    # Initialize dictionary to store metrics
    metrics = {'Model': [], 'RMSE': [], 'MAE': [], 'R2': [], 'Composite_Score': []}

    # Weights for the composite score
    w_rmse, w_mae, w_r2 = 0.4, 0.3, 0.3

    for model in models_to_compare:
        pred = combined_df[model]

        rmse = mean_squared_error(obs, pred, squared=False)
        mae = mean_absolute_error(obs, pred)
        r2 = r2_score(obs, pred)

        # Normalize (inverted) for composite score (higher = better)
        rmse_score = 1 / (1 + rmse)
        mae_score = 1 / (1 + mae)
        composite_score = w_rmse * rmse_score + w_mae * mae_score + w_r2 * r2

        # Store values
        metrics['Model'].append(model)
        metrics['RMSE'].append(rmse)
        metrics['MAE'].append(mae)
        metrics['R2'].append(r2)
        metrics['Composite_Score'].append(composite_score)

    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics)
    # Normalize metrics to [0, 1] for fair comparison
    df_norm = metrics_df.copy()
    
    # Normalize so that higher = better for all metrics
    # For RMSE and MAE, lower is better → invert after normalization
    for col in ['RMSE', 'MAE']:
        df_norm[col] = 1 - (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
    
    # For R2 and Composite Score, higher is better → normalize normally
    for col in ['R2', 'Composite_Score']:
        df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
    
    # Set index to model name
    df_norm.set_index('Model', inplace=True)
    

    # Sort models by composite score (highest = best)
    metrics_df_sorted = metrics_df.sort_values(by='Composite_Score', ascending=True)
    
    # Select only the desired models
    selected_models = ['AMME', 'BMME', 'RMME']
    metrics_subset = metrics_df[metrics_df['Model'].isin(selected_models)]
    
    # Save to CSV
    output_csv = fr"{folder_dir}\{curr_folder}\11. Overall Taylor and Heatmaps/Ensemble_model_scores.csv"
    metrics_subset.to_csv(output_csv, index=False)
    
    print(f"Saved selected ensemble scores to: {output_csv}")
    


    
def taylor_overall11(save_dir):

    # Function to read and aggregate Taylor statistics from 12 files
    def read_and_average_taylor_stats():
        files = [f"{save_dir}\Month_{m:02d}_TaylorStats_{curr_var}.csv"for m in range(1, 13)]
        model_stats = {}
        for file in files:
            if not os.path.exists(file):
                print(f"File not found: {file} (skipping)")
                continue

            df = pd.read_csv(file)
            
            for _, row in df.iterrows():
                model = row["Model"]
                
                corr = row["Correlation"]
                std_model = row["Std_Model"]
                tss = row["TSS"]
                std_ref = row["Std_Ref"]  # Assumed same for all months

                if model not in model_stats:
                    model_stats[model] = {"Correlation": [], "Std_Model": [], "TSS": [], "Std_Ref": std_ref}
                
                model_stats[model]["Correlation"].append(corr)
                model_stats[model]["Std_Model"].append(std_model)
                model_stats[model]["TSS"].append(tss)

        # Compute average values
        avg_stats = []
        for model, stats in model_stats.items():
            avg_corr = round(np.mean(stats["Correlation"]),3)
            avg_std_model = round(np.mean(stats["Std_Model"]),3)
            avg_tss = round(np.mean(stats["TSS"]),3)
            std_ref = stats["Std_Ref"]
            
            avg_stats.append({
                "Model": model,
                "Correlation": avg_corr,
                "Std_Model": avg_std_model,
                "TSS": avg_tss,
                "Std_Ref": std_ref
            })

        df_avg = pd.DataFrame(avg_stats)
        # Save to CSV
        df_avg.to_csv(fr"{folder_dir}\{curr_folder}\11. Overall Taylor and Heatmaps/Average_model_stats_Ens.csv", index=False)
        # Print the table with values
        print("\n===== Average Taylor Statistics for Each Model =====\n")
        print(df_avg.to_string(index=False))  # Display table in a clean format

        return df_avg

    # Function to plot Taylor Diagram with correct axis labeling

    def plot_taylor_diagram(df, folder_dir, curr_folder):
        df["Theta"] = np.arccos(df["Correlation"])  # Convert correlation to angle (radians)
    
        # Define plot limits
        std_ref = df["Std_Ref"].iloc[0]  # Assuming same reference std for all
        max_std = max(df["Std_Model"].max(), std_ref) * 1.2
    
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    
        # Add standard deviation reference circles
        std_levels = np.arange(1, 5, 1)  # Standard deviations from 1 to 11

        for s in std_levels:
            ax.plot(np.linspace(0, np.pi/2, 100), [s]*100, color='gray', lw=0.5, linestyle='--')
    
        # Add constant correlation arcs
        corr_vals = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0]
        for c in corr_vals:
            theta = np.arccos(c)
            r = np.linspace(0, max_std, 300)
            ax.plot([theta]*len(r), r, color='gray', lw=0.5, linestyle='--')
    
        # Get the observation (usually ERA5) row
        # Draw red arc using reference std deviation across the first quadrant
        # Get the observation (e.g., ERA5) row and angle
        obs_row = df[df["Model"].str.lower() == "chirps"].iloc[0]
        std_obs = obs_row["Std_Model"]  # The radius where ERA5 lies
        
        # Draw red arc across entire angle (0 to π/2 or more if needed)
        theta_arc = np.linspace(0, np.pi/2, 300)  # 0 to 90 degrees
        r_arc = np.full_like(theta_arc, std_obs)  # Constant radius = observation's std
        ax.plot(theta_arc, r_arc, color="black", linewidth=0.8)

        # Use unique colors
        colors = plt.cm.get_cmap("gist_ncar", len(df))
    
        # Plot each model
        for i, row in df.iterrows():
            theta, r, model = row["Theta"], row["Std_Model"], row["Model"]
            if model.lower() == "chirps":
                ax.scatter(theta, r, label=model, marker='*', s=150, color='green', edgecolor='black', linewidth=0.8)
            else:
                ax.scatter(theta, r, label=model, s=100, color=colors(i), edgecolor='black', linewidth=0.8)
    
        # Axes formatting
        ax.set_thetamin(0)
        ax.set_thetamax(90)
        ax.set_ylim(0, max_std)
    
        # Set correlation labels
        ax.set_xticks(np.arccos([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]))
        ax.set_xticklabels(["1.0", "0.9","0.8", "0.7","0.6", "0.5","0.4", "0.3", "0.2", "0.1", "0.0"], fontsize=10)
        ax.set_xlabel("Standard Deviation", labelpad=20)

    
        # Set std deviation radial labels
        ax.set_yticks(std_levels)
        ax.set_yticklabels([f"{s:.1f}" for s in std_levels], fontsize=10)
    
        # Label for correlation arc
        ax.text(np.arccos(0.7), max_std * 1.05, "Correlation", fontsize=12,
                rotation=np.degrees(-np.arccos(0.7)), ha="center", va="bottom", color="black")
    
        plt.title("Taylor Diagram", fontsize=14, pad=30)
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05), fontsize=9)
    
        plt.tight_layout()
        plt.savefig(fr"{folder_dir}\{curr_folder}\11. Overall Taylor and Heatmaps\Ens_Over_all_Taylor_Diagram_Models.png",
                    dpi=300, bbox_inches='tight', pad_inches=0.2)
        plt.show()


    # Function to plot heatmap
    def plot_heatmap(df):
        df = df.set_index("Model").T  # Transpose so models are on x-axis
        df = df.drop("Std_Ref")  # Remove Std_Ref since it's not a varying metric

        # Define color maps for different metrics
        cmap_dict = {
            "Correlation": "coolwarm_r",  # Diverging for correlation (-1 to 1 usually)
            "Std_Model": "YlOrBr",  # Sequential for standard deviation
            "TSS": "Blues",  # Sequential for skill score
        }

        # Create subplots for separate color scales
        fig, axes = plt.subplots(nrows=3, figsize=(15, 6), sharex=True)

        for ax, metric in zip(axes, df.index):
            sns.heatmap(df.loc[[metric]], annot=True, fmt=".2f", cmap=cmap_dict[metric], linewidths=0.5, cbar=True, ax=ax)

        plt.xticks(rotation=90, ha="right")
        plt.suptitle("Heatmap of Average Taylor Statistics", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(fr"{folder_dir}\{curr_folder}\11. Overall Taylor and Heatmaps\Ens_TaylorStats_Heatmap_Models.png", dpi=300, bbox_inches='tight', pad_inches=0.2)
        plt.show()

    # Main execution
    # Main execution
    df_avg = read_and_average_taylor_stats()

    # Reorder and rename models for plotting
    priority_models = ["AMME", "BMME", "RMME"]
    non_priority = df_avg[~df_avg["Model"].isin(priority_models)]
    priority = df_avg[df_avg["Model"].isin(priority_models)]
    df_avg = pd.concat([non_priority, priority], ignore_index=True)

    # Rename for plotting
    model_name_map = {
        "AMME": "AMME",
        "BMME": "BMME",
        "RMME": "RMME"
    }
    df_avg["Model"] = df_avg["Model"].replace(model_name_map)

    plot_taylor_diagram(df_avg, folder_dir, curr_folder)
    #plot_heatmap(df_avg)

def individual_heatmap12(save_dir):
    
    # Path to the folder containing monthly Taylor stats CSV files
    base_path = fr"{save_dir}"

    # List of metrics to process
    metrics = ["Correlation", "Std_Model", "TSS"]
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Models to exclude
    exclude_models = ["CHIRPS","AMME", "RMME", "BMME", "ERA5"]

    # Initialize a dictionary to hold DataFrames for each metric
    metric_data = {metric: pd.DataFrame(index=month_labels) for metric in metrics}

    # Loop through months and collect data
    for m in range(1, 13):
        file = os.path.join(base_path, f"Month_{m:02d}_TaylorStats_{curr_var}.csv")
        if not os.path.exists(file):
            print(f"Missing file for month {m}: {file}")
            continue

        df = pd.read_csv(file)
        df = df[~df["Model"].isin(exclude_models)]  # Exclude the unwanted models

        for metric in metrics:
            values = df.set_index("Model")[metric]
            metric_data[metric].loc[month_labels[m-1], values.index] = values

    # Plotting heatmaps
    cmap_dict = {
        "Correlation": "coolwarm_r",
        "Std_Model": "YlGnBu",
        "TSS": "Blues"
    }
    # Path to save summary text
    summary_path = os.path.join(base_path, f"Summary_Stats_{curr_var}.txt")
    with open(summary_path, "w") as f:
        for metric in metrics:
            data = metric_data[metric]
            sentence = f"{metric} - {curr_folder}\n"
            f.write(sentence)
            for model in data.columns:
                series = data[model]
                high_val = series.max()
                low_val = series.min()
    
                high_months = series[series == high_val].index.tolist()
                low_months = series[series == low_val].index.tolist()
    
                high_str = ', '.join(high_months)
                low_str = ', '.join(low_months)
    
                sentence = f"{model} has highest value of ({high_val:.2f}) in {high_str} and lowest value of ({low_val:.2f}) in {low_str}\n"
                f.write(sentence)
            

    for metric in metrics:
        plt.figure(figsize=(12, 7))
        sns.heatmap(metric_data[metric], annot=True, fmt=".2f", cmap=cmap_dict[metric],
                    linewidths=0.5, cbar_kws={'label': metric})
        plt.title(f"{metric} Heatmap: {curr_folder}", fontsize=16, fontweight="bold")
        plt.xlabel("CMIP6 Models")
        plt.ylabel("Month")
        plt.tight_layout()
        plt.savefig(fr"{folder_dir}\{curr_folder}\11. Overall Taylor and Heatmaps/Heatmap_{metric}.png", dpi=300)
        plt.show()

def prepare_SSP_data13():
    #### TO DO TO AMEND
    ssp_input_folder= r""
    lat_lon_clip_csv1(ssp_input_folder)    
    rearrange_folders2()
    csv_merge_concate3()
    time_clip4()
    conversion_cel5()
    regrid_csv6()
    to_polygonwise7()
    
def machine_learning_part14():
    def training_testing_forescating_14_1():
        # -----------------------
        # CONFIGURATION
        # -----------------------
        if curr_folder=="Temperature":
            
            # Years for splitting
            TRAIN_YEARS = list(range(2015, 2022))  # 2015–2023
            TEST_YEAR   = 2022                     # test on 2022
            FUTURE_YEARS = list(range(2025, 2101)) # forecast 2025–2100
        elif curr_folder=="Precipitation":
            
            # Years for splitting
            TRAIN_YEARS = list(range(2015, 2024))  # 2015–2023
            TEST_YEAR   = 2024                     # test on 2022
            FUTURE_YEARS = list(range(2025, 2101)) # forecast 2025–2100
        ssp245_file = fr"D:\Climate\Paper_Worth\{curr_folder}\12. SSP Data\SSP245_Ensemble_CMIP6_{curr_var}.csv"
        ssp585_file = fr"D:\Climate\Paper_Worth\{curr_folder}\12. SSP Data\SSP585_Ensemble_CMIP6_{curr_var}.csv"
        obs_file    = fr"D:\Climate\Paper_Worth\{curr_folder}\12. SSP Data\{curr_dataset}_2015-{end_year}.csv"

        output_dir  = fr"D:\Climate\Paper_Worth\{curr_folder}\12. SSP Data\12.1 Forecast_Scenario_ML_Model_2025-2100"
        os.makedirs(output_dir, exist_ok=True)


        # -----------------------
        # LOAD DATA
        # -----------------------
        def load_data(path):
            df = pd.read_csv(path, parse_dates=['time'])
            df['year'] = df['time'].dt.year
            df['month'] = df['time'].dt.month
            return df

        obs_df   = load_data(obs_file)
        ssp245   = load_data(ssp245_file)
        ssp585   = load_data(ssp585_file)

        # Zone columns (all except time, year, month)
        zones = [c for c in obs_df.columns if c not in ('time','year','month')]

        # -----------------------
        # DEFINE MODELS
        # -----------------------
        models = {
            'LinearRegression': LinearRegression(),
            'RandomForest':     RandomForestRegressor(n_estimators=100, random_state=0),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=0),
            'SVR':              SVR(),
            'XGBoost':          XGBRegressor(n_estimators=100, verbosity=0, random_state=0),
            'MLP':              MLPRegressor(hidden_layer_sizes=(100,50), max_iter=1000, random_state=0),
            'BayesianRidge':    BayesianRidge()
        }
        if has_lgb:
            models['LightGBM'] = LGBMRegressor(n_estimators=100, verbosity=-1, random_state=0)
        if has_cat:
            models['CatBoost'] = CatBoostRegressor(verbose=False, random_state=0)

        # Stacking ensemble of the first three
        base = list(models.items())[:3]
        models['StackingEnsemble'] = StackingRegressor(estimators=base, final_estimator=LinearRegression())

        # -----------------------
        # TRAIN → TEST → FORECAST
        # -----------------------
        def run_pipeline(scenario_df, scenario_name):
            # Split into train/test/future
            train_mask  = scenario_df['year'].isin(TRAIN_YEARS)
            test_mask   = (scenario_df['year'] == TEST_YEAR)
            future_mask = scenario_df['year'].isin(FUTURE_YEARS)

            train_cmip = scenario_df[train_mask].reset_index(drop=True)
            test_cmip  = scenario_df[test_mask].reset_index(drop=True)
            future_cmip= scenario_df[future_mask].reset_index(drop=True)

            train_obs = obs_df[obs_df['year'].isin(TRAIN_YEARS)].reset_index(drop=True)
            test_obs  = obs_df[obs_df['year'] == TEST_YEAR].reset_index(drop=True)

            # Prepare results
            eval_rows = []
            forecasts = {m: pd.DataFrame({'time': future_cmip['time']}) for m in models}

            # For each model and zone
            for mname, model in models.items():
                print(f"Training & testing {mname} for {scenario_name}...")
                for zone in zones:
                    # Build features
                    X_tr = pd.DataFrame({
                        'cmip6': train_cmip[zone],
                        'month': train_cmip['month'],
                        'year':  train_cmip['year']
                    })
                    y_tr = train_obs[zone]

                    X_te = pd.DataFrame({
                        'cmip6': test_cmip[zone],
                        'month': test_cmip['month'],
                        'year':  test_cmip['year']
                    })
                    y_te = test_obs[zone]

                    X_fu = pd.DataFrame({
                        'cmip6': future_cmip[zone],
                        'month': future_cmip['month'],
                        'year':  future_cmip['year']
                    })

                    # Train & predict
                    model.fit(X_tr, y_tr)
                    y_pred = model.predict(X_te)

                    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
                    mae  = mean_absolute_error(y_te, y_pred)
                    r2   = r2_score(y_te, y_pred)
                    eval_rows.append([scenario_name, mname, zone, rmse, mae, r2])

                    # Forecast
                    forecasts[mname][f"{zone}_corrected"] = model.predict(X_fu)

            # Save evaluation
            eval_df = pd.DataFrame(eval_rows, columns=['Scenario','Model','Zone','RMSE','MAE','R2'])
            eval_df.to_csv(os.path.join(output_dir, f"Evaluation_{scenario_name}_{TEST_YEAR}.csv"), index=False)
            print("Saved Evaluation File ---->", output_dir)
            # Save forecasts
            for mname, fc_df in forecasts.items():
                fc_df.to_csv(os.path.join(output_dir, f"Forecast_{scenario_name}_{mname}_2025_2100.csv"), index=False)
                print("Saved Forecast File ---->", output_dir)
        # Run for both scenarios
        run_pipeline(ssp245, "SSP245")
        run_pipeline(ssp585, "SSP585")

        print("✅ All done. Check the 8. ML Final folder for evaluation scores and forecasts.")
        print("############################# Done 14.1")

    def finding_evaluation_metrics_14_2():
        # ------------------------
        # CONFIGURATION
        # ------------------------
        if curr_folder=="Temperature":
            
            # Years for splitting
            TRAIN_YEARS = list(range(2015, 2022))  # 2015–2023
            TEST_YEAR   = 2022                     # test on 2022
            
        elif curr_folder=="Precipitation":
           
            # Years for splitting
            TRAIN_YEARS = list(range(2015, 2024))  # 2015–2023
            TEST_YEAR   = 2024                     # test on 2022
        # ------------------------
        # LOOP OVER SSP SCENARIOS
        # ------------------------
        for ssp_no in [245, 585]:

        
            evaluation_csv = fr"D:\Climate\Paper_Worth\{curr_folder}\12. SSP Data\12.1 Forecast_Scenario_ML_Model_2025-2100\Evaluation_SSP{ssp_no}_{TEST_YEAR}.csv"
            ssp_csv        = fr"D:\Climate\Paper_Worth\{curr_folder}\12. SSP Data/SSP{ssp_no}_Ensemble_CMIP6_{curr_var}.csv"
            obs_csv        = fr"D:\Climate\Paper_Worth\{curr_folder}\12. SSP Data\{curr_dataset}_2015-{end_year}.csv"
            output_dir     = fr"D:\Climate\Paper_Worth\{curr_folder}\12. SSP Data/12.2 Finding Evaluation"
            plot_dir       = os.path.join(output_dir, f"Plots_{TEST_YEAR}_SSP_{ssp_no}")
            os.makedirs(plot_dir, exist_ok=True)
    
            
                
            # ------------------------
            # 1. Pick best model per zone from evaluation CSV
            # ------------------------
            eval_df = pd.read_csv(evaluation_csv)
            zones   = eval_df['Zone'].unique().tolist()
    
            best_models = {}
    
            best_models = {}

            for zone in zones:
                sub = eval_df[eval_df['Zone'] == zone].copy()
            
                # Skip if there is no data for this zone
                if sub.empty:
                    continue
            
                # Select the model with the lowest RMSE
                best_model = sub.loc[sub['RMSE'].idxmin(), 'Model']
                best_models[zone] = best_model
            
            print("Best models per zone based on lowest RMSE:")
            print(best_models)

    
            # ------------------------
            # 2. Load CMIP6 and CHIRPS
            # ------------------------
            def load_data(path):
                df = pd.read_csv(path, parse_dates=['time'])
                df['year']  = df['time'].dt.year
                df['month'] = df['time'].dt.month
                return df
    
            ssp = load_data(ssp_csv)
            obs    = load_data(obs_csv)
    
            cmip_test = ssp[ssp['year'] == TEST_YEAR].reset_index(drop=True)
            obs_test  = obs[obs['year'] == TEST_YEAR].reset_index(drop=True)
    
            # ------------------------
            # 3. Model constructors
            # ------------------------
            constructors = {
                'LinearRegression': LinearRegression,
                'RandomForest':     RandomForestRegressor,
                'GradientBoosting': GradientBoostingRegressor,
                'SVR':              SVR,
                'XGBoost':          XGBRegressor,
                'MLP':              MLPRegressor,
                'BayesianRidge':    BayesianRidge
            }
            if has_lgb:
                constructors['LightGBM'] = LGBMRegressor
            if has_cat:
                constructors['CatBoost'] = CatBoostRegressor
    
            # ------------------------
            # 4. Retrain best models, predict 2022, compute metrics & plot
            # ------------------------
            records = []
    
            for zone in zones:
                model_name = best_models[zone]
                print(f"Zone: {zone} | Best model: {model_name}")
                
                # instantiate
                if model_name == 'StackingEnsemble':
                    base = [(n, constructors[n]()) for n in list(constructors)[:3]]
                    model = StackingRegressor(estimators=base, final_estimator=LinearRegression())
                else:
                    model = constructors[model_name]()
                
                # build train split
                train_cmip = ssp[ssp['year'].isin(TRAIN_YEARS)].reset_index(drop=True)
                train_obs  = obs[obs['year'].isin(TRAIN_YEARS)].reset_index(drop=True)
                
                # features
                def make_X(df):
                    return pd.DataFrame({
                        'cmip6': df[zone],
                        'month': df['month'],
                        'year':  df['year']
                    })
                
                X_tr = make_X(train_cmip)
                y_tr = train_obs[zone]
                X_te = make_X(cmip_test)
                y_te = obs_test[zone]
                
                # train & predict
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)
                
                # metrics
                rmse = np.sqrt(mean_squared_error(y_te, y_pred))
                mae  = mean_absolute_error(y_te, y_pred)
                r2   = r2_score(y_te, y_pred)
                
                records.append({
                'Model': model_name,
                'Zone':  zone,
                'RMSE':  rmse,
                })
    
                # timeseries plotting
                times = cmip_test['time']
                y_raw = cmip_test[zone].values
                y_obs = y_te.values
                y_ml  = y_pred
                
                plt.figure(figsize=(8,3))
                plt.plot(times, y_raw, '-o', label='Raw CMIP6',   color='orange')
                plt.plot(times, y_obs, '-s', label='Observed',     color='blue')
                plt.plot(times, y_ml,  '-^', label='BiasCorr ML',  color='green')
                plt.title(f"{zone} – 2022 Forecast vs Observed")
                plt.xticks(times, [t.strftime("%b") for t in times], rotation=45)
                plt.ylabel("Precipitation")
                plt.legend(fontsize=8)
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f"{zone}_2022_comparison (SSP {ssp_no}.png"), dpi=300)
                plt.close()
    
            # ------------------------
            # 5. Save accuracy metrics CSV
            # ------------------------
            acc_df = pd.DataFrame.from_records(records)
    
            # Normalize RMSE, MAE, R² for computing composite score
            acc_df['RMSE_norm'] = (acc_df['RMSE'] - acc_df['RMSE'].min()) / (acc_df['RMSE'].max() - acc_df['RMSE'].min() + 1e-8)
            
            # Drop the intermediate normalized columns
            acc_df.drop(columns=['RMSE_norm'], inplace=True)
    
            acc_csv = os.path.join(output_dir, f"BestModel_Accuracy_{end_year}_SSP_{ssp_no}.csv")
            acc_df.to_csv(acc_csv, index=False)
            print(f"Saved accuracy metrics to: {acc_csv}")
            print(f"Plots saved under: {plot_dir}")

    def finding_skill_score_14_3():
        for ssp_no in [245,585]:
            # ---- USER CONFIG ----

            if curr_folder=="Temperature":
                
                # Years for splitting
                TRAIN_YEARS = list(range(2015, 2022))  # 2015–2023
                TEST_YEAR   = 2022                     # test on 2022
                FUTURE_YEARS = list(range(2025, 2101)) # forecast 2025–2100
            elif curr_folder=="Precipitation":
                
                # Years for splitting
                TRAIN_YEARS = list(range(2015, 2024))  # 2015–2023
                TEST_YEAR   = 2024                     # test on 2022
                FUTURE_YEARS = list(range(2025, 2101)) # forecast 2025–2100
            evaluation_csv = fr"D:\Climate\Paper_Worth\{curr_folder}\12. SSP Data\12.1 Forecast_Scenario_ML_Model_2025-2100\Evaluation_SSP{ssp_no}_{TEST_YEAR}.csv"
            ssp_csv        = fr"D:\Climate\Paper_Worth\{curr_folder}\12. SSP Data/SSP{ssp_no}_Ensemble_CMIP6_{curr_var}.csv"
            obs_csv        = fr"D:\Climate\Paper_Worth\{curr_folder}\12. SSP Data\{curr_dataset}_2015-{end_year}.csv"
            output_csv     = fr"D:\Climate\Paper_Worth\{curr_folder}\12. SSP Data\SkillScore_SSP{ssp_no}_{TEST_YEAR}.csv"

            # ---- 1) Load best‐model per zone ----
            # ---- 1) Load best‐model per zone ----
            eval_df = pd.read_csv(evaluation_csv)
            zones   = eval_df['Zone'].unique().tolist()
    
            best_models = {}

            for zone in zones:
                sub = eval_df[eval_df['Zone'] == zone].copy()
            
                # Select the best model based on lowest RMSE
                best = sub.loc[sub['RMSE'].idxmin(), 'Model']
                best_models[zone] = best

    
            # ---- 2) Load CMIP6 & CHIRPS for 2022 ----
            def load_df(path):
                df = pd.read_csv(path, parse_dates=['time'])
                df['year']  = df['time'].dt.year
                df['month'] = df['time'].dt.month
                return df
    
            ssp = load_df(ssp_csv)
            obs    = load_df(obs_csv)
    
            cmip_test = ssp[ssp['year']==TEST_YEAR].reset_index(drop=True)
            obs_test  = obs   [obs   ['year']==TEST_YEAR].reset_index(drop=True)
    
            # ---- 3) Model constructors ----
            constructors = {
                'LinearRegression': LinearRegression,
                'RandomForest':     RandomForestRegressor,
                'GradientBoosting': GradientBoostingRegressor,
                'SVR':              SVR,
                'XGBoost':          XGBRegressor,
                'MLP':              MLPRegressor,
                'BayesianRidge':    BayesianRidge
            }
            if has_lgb:
                constructors['LightGBM'] = LGBMRegressor
            if has_cat:
                constructors['CatBoost'] = CatBoostRegressor
    
            # ---- 4) Compute skill per zone ----
            records = []
            for zone in zones:
                # Prepare obs and raw cmip6
                y_raw = cmip_test[zone].values
                y_obs = obs_test[zone].values
                mask  = ~np.isnan(y_raw) & ~np.isnan(y_obs)
    
                # Raw metrics
                rmse_raw = np.sqrt(mean_squared_error(y_obs[mask], y_raw[mask]))
                mae_raw  = np.mean(np.abs(y_obs[mask] - y_raw[mask]))
                r2_raw   = np.corrcoef(y_obs[mask], y_raw[mask])[0,1]**2 if len(y_obs[mask]) > 1 else np.nan
    
                # Train ML model
                cmip_train = ssp[ssp['year'].isin(TRAIN_YEARS)].reset_index(drop=True)
                obs_train  = obs[obs['year'].isin(TRAIN_YEARS)].reset_index(drop=True)
    
                mname = best_models[zone]
                if mname == 'StackingEnsemble':
                    base = [(n, constructors[n]()) for n in list(constructors)[:3]]
                    model = StackingRegressor(estimators=base, final_estimator=LinearRegression())
                else:
                    model = constructors[mname]()
    
                def make_X(df):
                    return pd.DataFrame({
                        'cmip6': df[zone],
                        'month': df['month'],
                        'year':  df['year']
                    })
    
                X_tr = make_X(cmip_train)
                y_tr = obs_train[zone]
                X_te = make_X(cmip_test)
    
                model.fit(X_tr, y_tr)
                y_ml = model.predict(X_te)
    
                # ML RMSE
                rmse_ml = np.sqrt(mean_squared_error(y_obs[mask], y_ml[mask]))
                
                # Compute skill based on RMSE only
                skill = 1 - (rmse_ml / rmse_raw) if rmse_raw > 0 else np.nan
                
                records.append({
                    'Zone':      zone,
                    'BestModel': mname,
                    'Raw_RMSE':  rmse_raw,
                    'ML_RMSE':   rmse_ml,
                    'Skill':     skill
                })

    
    
            # ---- 5) Save to CSV ----
            skill_df = pd.DataFrame.from_records(records, columns=[
                'Zone', 'BestModel',
                'Raw_RMSE', 
                'ML_RMSE', 
                'Skill'
            ])
            skill_df.to_csv(output_csv, index=False)
            print("Skill scores saved to:", output_csv)
    
    def final_bias_corrected_14_4():

        # -------------------------
        # CONFIGURATION
        # -------------------------
        for scenario, ssp_no in zip(["SSP245", "SSP585"], [245, 585]):
            if curr_folder=="Temperature":
                
                TEST_YEAR=2022
            elif curr_folder=="Precipitation":
               
                TEST_YEAR=2024
            skill_file    = fr"D:\Climate\Paper_Worth\{curr_folder}\12. SSP Data\SkillScore_{scenario}_{TEST_YEAR}.csv"
            raw_file       = fr"D:\Climate\Paper_Worth\{curr_folder}\12. SSP Data/SSP{ssp_no}_Ensemble_CMIP6_{curr_var}.csv"
            ml_forecast_dir= fr"D:\Climate\Paper_Worth\{curr_folder}\12. SSP Data\12.1 Forecast_Scenario_ML_Model_2025-2100"
            output_file    = fr"D:\Climate\Paper_Worth\{curr_folder}\12. SSP Data\Final_BiasCorrected_{scenario}_2025_2100.csv"
    
            # -------------------------
            # 1) Read skill scores
            # -------------------------
            skill_df = pd.read_csv(skill_file)

            # Zones where Raw RMSE is better (or equal)
            bad_zones = skill_df.loc[skill_df["Raw_RMSE"] <= skill_df["ML_RMSE"], "Zone"].tolist()
            print("Zones where raw RMSE is better (will fallback to raw):", bad_zones)

            # Save bad zones with RMSEs
            bad_zones_df = skill_df.loc[skill_df["Zone"].isin(bad_zones), ["Zone", "Raw_RMSE", "ML_RMSE"]]
            bad_zones_file = fr"D:\Climate\Paper_Worth\{curr_folder}\12. SSP Data\Bad_Zones_{scenario}_{end_year}.txt"
            with open(bad_zones_file, "w") as f:
                f.write(f"Zones where raw RMSE is better - {scenario} Scenario ({end_year})\n")
                f.write("-" * 50 + "\n")
                for _, row in bad_zones_df.iterrows():
                    f.write(f"Zone: {row['Zone']}, ML_RMSE:{row['ML_RMSE']:.4f} CMIP6_RMSE: {row['Raw_RMSE']:.4f}\n")
            print(f"📝 Bad zones saved to: {bad_zones_file}")

            # For each zone, get best ML model (still needed for good zones)
            best_models = (
                skill_df.loc[skill_df.groupby("Zone")["ML_RMSE"].idxmin()]
                .set_index("Zone")["BestModel"]
                .to_dict()
            )
    
            # -------------------------
            # 2) Read raw ensemble
            # -------------------------
            raw_df = pd.read_csv(raw_file, parse_dates=['time'])
            raw_fut = raw_df[raw_df['time'].dt.year >= 2025].reset_index(drop=True)
    
            # -------------------------
            # 3) Build final DataFrame
            # -------------------------
            final = pd.DataFrame({'time': raw_fut['time']})
            zones = [c for c in raw_fut.columns if c not in ('time','month','year')]
    
            for zone in zones:
                if zone in bad_zones:
                    final[f"{zone}_corrected"] = raw_fut[zone].values
                else:
                    model = best_models[zone]
                    ml_file = os.path.join(
                        ml_forecast_dir,
                        f"Forecast_{scenario}_{model}_2025_2100.csv"
                    )
                    ml_df = pd.read_csv(ml_file, parse_dates=['time'])
                    final[f"{zone}_corrected"] = ml_df[f"{zone}_corrected"].values
    
            # -------------------------
            # 4) Save
            # -------------------------
            final.to_csv(output_file, index=False)
            print("✅ Final bias-corrected forecast saved to:", output_file)


    def full_difference_plot_14_5():
        
        # Configuration
        scenarios = [245, 585]
        periods = {
            "Mid-Century": (2046, 2054),
            "Post Mid-Century": (2066, 2074)
        }
        input_obs_csv   = fr"D:\Climate\Paper_Worth\{curr_folder}\12. SSP Data\{curr_dataset}_2015-{end_year}.csv"
        shapefile_path  = r"D:\Climate\Analysis\Study Region ROI\ROI_CLIM_PROJ.shp"
        output_dir      = fr"D:\Climate\Paper_Worth\{curr_folder}/13. ML Difference Plots"
        obs_df = pd.read_csv(input_obs_csv)
        obs_df['time'] = pd.to_datetime(obs_df['time'], format='%Y-%m')
        obs_df['year'] = obs_df['time'].dt.year
        obs_df['month'] = obs_df['time'].dt.month
        past = obs_df[(obs_df.year >= 2015) & (obs_df.year <= 2024)]
        exclude = {'time', 'year', 'month'}
        month_names = list(calendar.month_name)[1:]
        # T-Test Function
        def run_ttest(p1, p2, zones):
            pv = {m: {} for m in range(1, 13)}
            for m in range(1, 13):
                for z in zones:
                    a = p1[p1.month == m][z].dropna()
                    b = p2[p2.month == m][z].dropna()
                    pv[m][z] = ttest_ind(a, b, equal_var=False).pvalue if len(a) > 1 and len(b) > 1 else np.nan
            df = pd.DataFrame.from_dict(pv, orient='index')
            df.index = month_names
            df.index.name = 'Month'
            return df
        # Plot Function 
        def plot_maps(diff_dict, mode, period, scenario):
            
            norm = Normalize(vmin=vmin, vmax=vmax)
            fig, axs = plt.subplots(3, 4, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(16, 12))
            axs = axs.flatten()
            bounds = gdf.total_bounds
            for m in range(1, 13):
                ax = axs[m - 1]
                ax.set_extent([bounds[0], bounds[2], bounds[1], bounds[3]], crs=ccrs.PlateCarree())
                for _, row in gdf.iterrows():
                    lab = row['label']
                    if lab in diff_dict[m]:
                        color = cmap(norm(diff_dict[m][lab]))
                    else:
                        color = 'white'
                    ax.add_geometries([row.geometry], crs=ccrs.PlateCarree(),
                                      facecolor=color, edgecolor='black', linewidth=0.5)
                ax.set_title(calendar.month_name[m], fontsize=11)
            fig.subplots_adjust(top=0.92, hspace=0.3, wspace=0.2)
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axs, orientation='horizontal', fraction=0.02, pad=0.05)
            cbar.set_label(f'Δ {curr_variable}', fontsize=12)
            plt.suptitle(f"Significant Difference in {curr_folder} of SSP{scenario} : {period} ({years[0]}-{years[1]}) - {curr_dataset} (2015-{end_year})",
                         fontsize=14, fontweight='bold')
            plt.savefig(f"{output_dir}/{curr_folder}_{scenario}_{period}.png", dpi=300, bbox_inches='tight')
            plt.show()
        # Main Loop 
        gdf = gpd.read_file(shapefile_path)
        for scenario in scenarios:
            ml_csv = fr"D:\Climate\Paper_Worth\{curr_folder}\12. SSP Data\Final_BiasCorrected_SSP{scenario}_2025_2100.csv"
            ml_df = pd.read_csv(ml_csv)
        
            ml_df['time'] = pd.to_datetime(ml_df['time'], format='%Y-%m-%d')
            ml_df = ml_df.rename(columns={c: c.replace('_corrected', '') for c in ml_df.columns if c.endswith('_corrected')})
        
            ml_df['year'] = ml_df['time'].dt.year
            ml_df['month'] = ml_df['time'].dt.month
        
            zones = sorted(set(obs_df.columns) & set(ml_df.columns) - exclude)
        
            for period, years in periods.items():
                fut_ml = ml_df[(ml_df.year >= years[0]) & (ml_df.year <= years[1])]
                p_ml = run_ttest(past, fut_ml, zones)
        
                mean_past = past.groupby('month')[zones].mean()
                mean_ml = fut_ml.groupby('month')[zones].mean()
        
                diff_ml = {m: {} for m in range(1, 13)}
                print(f"\n--- Scenario SSP{scenario} | Period: {period} ---")
                for m in range(1, 13):
                    inc_zones, dec_zones = [], []
                    for z in zones:
                        pval = p_ml.at[month_names[m - 1], z]
                        if pd.notna(pval) and pval < 0.05:
                            diff = mean_ml.at[m, z] - mean_past.at[m, z]
                            diff_ml[m][z] = diff
                            if diff > 0:
                                inc_zones.append(z)
                            elif diff < 0:
                                dec_zones.append(z)
                    print(f"{calendar.month_name[m]} - Increase: {inc_zones} | Decrease: {dec_zones}")
                # Define summary file path
                # Define summary file path
                summary_file = os.path.join(output_dir, f"Summary_SSP{scenario}_{period}_{curr_folder}.txt")
                
                with open(summary_file, 'w') as f:
                    # Write heading
                    f.write(f"{'SSP'+str(scenario)} {period} {curr_folder}\n\n")
                
                    for m in range(1, 13):
                        inc_zones, dec_zones = [], []
                        for z in zones:
                            pval = p_ml.at[month_names[m - 1], z]
                            if pd.notna(pval) and pval < 0.05:
                                diff = mean_ml.at[m, z] - mean_past.at[m, z]
                                diff_ml[m][z] = diff
                                if curr_folder=="Temperature":
                                    if diff > 0:
                                        inc_zones.append(f"{z} (+{diff:.1f}°C)")
                                    elif diff < 0:
                                        dec_zones.append(f"{z} ({diff:.1f}°C)")
                                elif curr_folder=="Precipitation":
                                    if diff > 0:
                                        inc_zones.append(f"{z} (+{diff:.1f}mm/day)")
                                    elif diff < 0:
                                        dec_zones.append(f"{z} ({diff:.1f}mm/day)")
                
                        # Join zone names with differences
                        inc_str = ', '.join(inc_zones) if inc_zones else "None"
                        dec_str = ', '.join(dec_zones) if dec_zones else "None"
                        
                        # Write summary sentence
                        sentence = (
                            f"{calendar.month_name[m]} sees increase in {curr_folder} in zones [{inc_str}] "
                            f"and decrease in {curr_folder}  in zones [{dec_str}]\n"
                        )
                        f.write(sentence)


                plot_maps(diff_ml, mode="ml_only", period=period, scenario=scenario)



    if __name__ == "__main__":
        #training_testing_forescating_14_1()
        #finding_evaluation_metrics_14_2()
        #finding_skill_score_14_3()
        #final_bias_corrected_14_4()
        full_difference_plot_14_5()

def anomaly_trendlines15(bmme_path, ssp245_ens_path, ssp585_ens_path):
    # Define markers and fixed colors (so that none are too light)
    markers = ['s', 'o', 'o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'X']
    colors = ["blue", "orange", "red"]

    # Model file paths and names
    model_files = [
        
        fr"{bmme_path}",
        fr"{ssp245_ens_path}",
        fr"{ssp585_ens_path}",
    ]
    model_names = ["Historical", "SSP - 245", "SSP - 585", ]

    # We'll store the monthly means per model in a dictionary.
    # Each model's data is a DataFrame with index = month (1 to 12) and columns = zones.
    model_data = {}
    zone_list = None

    # Process each model file
    for i, file in enumerate(model_files):
        try:
            df = pd.read_csv(file)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue

        # Convert 'time' to datetime and extract month
        df['time'] = pd.to_datetime(df['time'])
        df['month'] = df['time'].dt.month

        # Determine zone columns (all columns except 'time' and 'month')
        zones = [col for col in df.columns if col not in ['time', 'month']]
        # For the first file, save the zone list (assume consistency across files)
        if zone_list is None:
            zone_list = zones
        # Group by month and compute the mean for each zone
        monthly_means = df.groupby('month')[zones].mean()  # Shape: (12, n_zones)
        model_data[model_names[i]] = monthly_means

    # Create subplots—one for each zone.
    n_zones = len(zone_list)
    cols = 5  # fixed grid: 5 columns
    rows = 5  # fixed grid: 5 rows, i.e. 25 subplot positions

    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    axes = axes.flatten()

    # For each zone, plot monthly mean curves for all models
    for j, zone in enumerate(zone_list):
        ax = axes[j]
        for i, model in enumerate(model_names):
            if model in model_data:
                monthly = model_data[model]  # DataFrame with index=month
                # For AMME and RMME, use dashed lines and no marker; otherwise, use solid with marker.
                linestyle = '--' if model in ['AMME', 'RMME'] else '-'
                marker = None if model in ['AMME', 'RMME'] else markers[i % len(markers)]
                markersize= 5 if model in ['Historical'] else 3
                ax.plot(monthly.index, monthly[zone],
                        linestyle=linestyle, marker=marker, markersize=markersize,
                        color=colors[i],
                        label=model, alpha=0.8)
        ax.set_title(zone, fontsize=12)
        ax.set_xticks(np.arange(1, 13))
        # Show month labels on every subplot
        ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], rotation=45, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)

    # Remove extra subplots:
    total_axes = len(axes)
    # We'll keep the very last subplot (axes[-1]) for the legend,
    # so remove all subplots from index n_zones to total_axes - 1.
    for k in range(n_zones, total_axes - 1):
        fig.delaxes(axes[k])

    # Use the last subplot as the legend panel
    legend_ax = axes[-1]
    legend_ax.axis('off')
    handles = []
    for i, model in enumerate(model_names):
        linestyle = '--' if model in ['AMME', 'RMME'] else '-'
        marker = None if model in ['AMME', 'RMME'] else markers[i % len(markers)]
        markersize= 5 if model in ['Historical'] else 3
        line = plt.Line2D([0], [0],
                          marker=marker, markersize=markersize,
                          color=colors[i],
                          label=model, linestyle=linestyle)
        handles.append(line)
    legend_ax.legend(handles=handles, loc='center', ncol=1, fontsize=10)

    # Adjust spacing to reduce white space
    plt.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.05, wspace=0.25, hspace=0.35)

    # Add common x and y labels using fig.text (adjust positions to avoid overlap)
    fig.text(0.5, 0.02, 'Month', ha='center', fontsize=16)
    fig.text(0.02, 0.5, f'{curr_variable}', va='center', rotation='vertical', fontsize=16)
    fig.suptitle(f"Monthly Mean {curr_folder} for Each Zone", fontsize=18, fontweight='bold', y=0.98)

    output_path = r"D:\Climate\Paper_Worth\Temperature\14. Anomaly Plot\Monthly_Mean_Precip_SSP_by_Zonewise.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

def ml_eval_plot16():
    # Load your data
    # Load your data
    df = pd.read_csv(r"D:\Climate\Paper_Worth\Temperature\12. SSP Data\12.1 Forecast_Scenario_ML_Model_2025-2100\Evaluation_SSP585_2022.csv")
    
    # Normalize (lower is better → inverted)
    df['RMSE_norm'] = 1 / (1 + df['RMSE'])
    df['MAE_norm']  = 1 / (1 + df['MAE'])
    
    # Composite Score (weights: RMSE=0.4, MAE=0.3, R2=0.3)
    df['Composite_Score'] = 0.4 * df['RMSE_norm'] + 0.3 * df['MAE_norm'] + 0.3 * df['R2']
    
    # Prepare zone positions
    zones = sorted(df['Zone'].unique())
    zone_positions = {zone: i for i, zone in enumerate(zones)}
    df['Zone_Pos'] = df['Zone'].map(zone_positions)
    
    # Set color palette (distinct and colorblind-friendly)
    models = df['Model'].unique()
    colors = cm.get_cmap('Paired', len(models))  # Alternatives: 'Set2', 'Dark2', 'tab20'
    
    # Plot
    plt.figure(figsize=(14, 6))
    for i, model in enumerate(models):
        subset = df[df['Model'] == model]
        plt.scatter(
            subset['Zone_Pos'], subset['Composite_Score'],
            label=model,
            s=80,
            color=colors(i),
            edgecolor='black',
            linewidth=0.8,
            alpha=0.9
        )
    
    # Format axes
    plt.xticks(ticks=range(len(zones)), labels=zones, rotation=45)
    plt.xlabel("Zone", fontsize=12)
    plt.ylabel("Composite Score", fontsize=12)
    plt.ylim(0.4, 1)
    plt.title("ML Models per Zone in SSP 585 for Temperature (Evaluation-Test Year: 2022)", fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Legend outside
    plt.legend(title='ML Model', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9, title_fontsize=10)
    
    plt.tight_layout()
    plt.savefig(fr"D:\Climate\Paper_Worth\Temperature\12. SSP Data/SSP585-Temperature.png", dpi=300)
    plt.show()
    # Find the best model per zone based on Composite Score
    best_models = df.loc[df.groupby('Zone')['Composite_Score'].idxmax()].copy()
    
    # Round composite score to 3 decimal places
    best_models['Composite_Score'] = best_models['Composite_Score'].round(3)
    
    # Keep only required columns
    best_models = best_models[['Zone', 'Model', 'Composite_Score']]
    
    # Save to CSV
    output_csv = fr"D:\Climate\Paper_Worth\Temperature\12. SSP Data/585_best_model_per_zone.csv"
    best_models.to_csv(output_csv, index=False)
    
    print(f"Saved best model per zone to: {output_csv}")
 
def supplemenatary17():
    def mean_plot_17_1():

        # Load the shapefile for showing country boundary lines over the map
        gdf = gpd.read_file(r"D:\Climate\Analysis\Study Region ROI\ROI_CLIM_PROJ.shp")
        folder_dir = "D:/Climate/Paper_Worth"

        csv_files = [
            fr"D:\Climate\Paper_Worth\{curr_folder}\10. Ensemble/BMME.csv",
            # ... Add up to 10 or more
        ]
        model_names = [
             f"BMME ({curr_folder})"
            # ... Corresponding model names
        ]
        # -----------------------------
        # Loop over each CSV & Model
        # -----------------------------
        for csv_file, model_name in zip(csv_files, model_names):
            print(f"\nProcessing model: {model_name}")
            
            # Create a unique output folder for each model
            output_path =fr"D:\Climate\Paper_Worth\{curr_folder}\14. Supplementary\{model_name}"
            os.makedirs(output_path, exist_ok=True)

            # -----------------------------
            # Load the CSV data
            # -----------------------------
            df = pd.read_csv(csv_file)
            # ✅ Convert 'time' column to datetime format
            df['time'] = pd.to_datetime(df['time'], format='%Y-%m')
            
            # ✅ Extract month and year
            df['month'] = df['time'].dt.month
            df['year'] = df['time'].dt.year
            
            # ✅ Drop 'time' column if not needed
            df = df.drop(columns=['time'])
            
            # ----------------------------
            # Grouped data calculations
            # ----------------------------
            long_term_mean = df.groupby('month').mean()
            long_term_variance = df.groupby('month').var()
            long_term_std = df.groupby('month').std()
            
            # Ensure months are labeled correctly for plotting
            long_term_mean['month'] = range(1, 13)
            long_term_variance['month'] = range(1, 13)
            long_term_std['month'] = range(1, 13)
            result_df = pd.DataFrame(long_term_mean)
            result_df.to_csv(f"{output_path}/Data-{model_name}-Mean.csv", index=False)
            result_df = pd.DataFrame(long_term_variance)
            result_df.to_csv(f"{output_path}/Data-{model_name}-Variance.csv", index=False)
            result_df = pd.DataFrame(long_term_std)
            result_df.to_csv(f"{output_path}/Data-{model_name}-Standard Deviation.csv", index=False)
            
            # ----------------------------
            # Function to plot either mean or variance
            # ----------------------------
            def plot_data(data, title, cbar_label):
                cmap = cm.get_cmap('coolwarm')
                norm = Normalize(vmin=vmin, vmax=vmax)
            
                fig, axes = plt.subplots(3, 4, figsize=(16, 12),
                                         subplot_kw={'projection': ccrs.PlateCarree()})
                fig.subplots_adjust(wspace=0.05, hspace=0.15)  # spacing between maps
            
                for i, ax in enumerate(axes.flat):
                    month = i + 1
                    month_data = data[data['month'] == month]
            
                    # Plot polygons with appropriate color
                    for idx, row in gdf.iterrows():
                        label = row['label']
                        if label in month_data.columns:
                            value = month_data[label].values[0]
                            color = cmap(norm(value))
                        else:
                            color = 'lightgrey'
                        ax.add_geometries([row['geometry']], crs=ccrs.PlateCarree(),
                                          facecolor=color, edgecolor='black', linewidth=1)
            
                    ax.set_title(calendar.month_name[month], fontsize=12, fontweight='bold')
                    ax.set_extent([gdf.total_bounds[0], gdf.total_bounds[2],
                                   gdf.total_bounds[1], gdf.total_bounds[3]])
                    ax.axis('off')
            
                    # Draw black border box
                    rect = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                         linewidth=1.5, edgecolor='black',
                                         facecolor='none', zorder=100)
                    ax.add_patch(rect)
            
                # Add colorbar
                cbar_ax = fig.add_axes([0.3, 0.08, 0.4, 0.02])  # [left, bottom, width, height]
                sm = cm.ScalarMappable(cmap=cmap, norm=norm)
                sm._A = []
                cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
                cbar.set_label(cbar_label, fontsize=12)
                cbar.ax.tick_params(labelsize=10)
            
                # Main title
                plt.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
            
                # Save
                plt.savefig(fr"{output_path}\{model_name}-{title}.png", dpi=300)
                plt.show()


                
            
            #######################################################
            #SECTION 1 : Plot long-term mean
            plot_data(long_term_mean, f'{model_name} - Long-term Mean {curr_folder} (1981-2014)', f'Mean {curr_variable}')
            
             
            ###########################################################################
            # SECTION 2: TRENDLINE
            pr_columns = [col for col in df.columns if col not in ['year', 'month']]
            avg_max_10 = {month: [] for month in range(1, 13)}
            avg_mean_10 = {month: [] for month in range(1, 13)}
            avg_min_10 = {month: [] for month in range(1, 13)}
            
            for month in range(1, 13):
                month_data = df[df['month'] == month]
                for year, group in month_data.groupby('year'):
                    avg_max_10[month].append(group[pr_columns].max(axis=1).mean())
                    avg_mean_10[month].append(group[pr_columns].mean(axis=1).mean())
                    avg_min_10[month].append(group[pr_columns].min(axis=1).mean())
            
            long_term_mean_10 = {m: np.mean(avg_mean_10[m]) for m in range(1, 13)}
            anomalies_10 = {m: [] for m in range(1, 13)}
            
            for m in range(1, 13):
                anomalies_10[m] = np.array(avg_mean_10[m]) - long_term_mean_10[m]
            
            years_10 = df['year'].unique()
            
            fig = plt.figure(figsize=(15, 15))
            gs = gridspec.GridSpec(4, 3, hspace=0.3, wspace=0.3)
            
            for i in range(12):
                ax = fig.add_subplot(gs[i // 3, i % 3])
                month = i + 1
                ax.plot(years_10, anomalies_10[month], label='Anomaly', color='blue', linestyle='-', marker='o')
            
                z = np.polyfit(years_10, anomalies_10[month], 1)
                p = np.poly1d(z)
                ax.plot(years_10, p(years_10), color='red', linestyle='--', label='Trend', zorder=2)
            
                ax.set_title(calendar.month_name[month], fontsize=12, fontweight='bold')
                ax.set_xlabel('Year')
                ax.set_ylabel(f'Anomaly {curr_variable}')
                ax.grid(True)
                ax.set_ylim(-2.0, 3.0)
            
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=12)
            # Main title
            plt.suptitle(f"Anomaly Trendlines for {curr_variable}", fontsize=16, fontweight='bold', y=0.95)

            plt.tight_layout(rect=[0, 0, 1, 1])
            plt.savefig(fr"{output_path}\{model_name}-Trendlines.png")
            plt.show()
            

    def anomalies_ssp_17_2():
        
        # CSVs and Model Names
        csv_files = [
            fr"D:\Climate\Paper_Worth\{curr_folder}\12. SSP Data\Final_BiasCorrected_SSP245_2025_2100.csv",
            fr"D:\Climate\Paper_Worth\{curr_folder}\12. SSP Data\Final_BiasCorrected_SSP585_2025_2100.csv"
        ]

        model_names = ["SSP 245", "SSP 585"]
        colors = {
            "SSP 245": "orange",
            "SSP 585": "red"
        }

        output_path = fr"D:\Climate\Paper_Worth\{curr_folder}\14. Supplementary"
        os.makedirs(output_path, exist_ok=True)

        # Define periods
        periods = {
            'Mid Century': (2046, 2054),
            'Post Mid Century': (2066, 2074)
        }

        # ---------------------------
        # Main Loop for Each Period
        # ---------------------------
        for period_name, (start_year, end_year) in periods.items():
            model_anomalies = {}

            for csv_file, model_name in zip(csv_files, model_names):
                print(f"Processing {model_name} for {period_name}")

                df = pd.read_csv(csv_file)
                df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d')

                df['month'] = df['time'].dt.month
                df['year'] = df['time'].dt.year
                df = df.drop(columns=['time'])

                # Filter to selected period
                df_period = df[(df['year'] >= start_year) & (df['year'] <= end_year)]

                # Extract zone columns
                pr_columns = [col for col in df.columns if col not in ['year', 'month']]

                # Monthly long-term mean for the period
                long_term_mean = {}
                anomalies = {}

                for month in range(1, 13):
                    df_month = df_period[df_period['month'] == month]
                    month_mean = df_month[pr_columns].mean().mean()  # spatial + temporal mean
                    long_term_mean[month] = month_mean
                    month_anomaly = df_month[pr_columns].mean(axis=1) - month_mean
                    anomalies[month] = month_anomaly.groupby(df_month['year']).mean()

                model_anomalies[model_name] = anomalies

            # ---------------------------
            # Plotting
            # ---------------------------
            fig = plt.figure(figsize=(15, 15))
            gs = gridspec.GridSpec(4, 3, hspace=0.3, wspace=0.3)

            for i in range(12):
                ax = fig.add_subplot(gs[i // 3, i % 3])
                month = i + 1

                for model_name in model_names:
                    years = list(model_anomalies[model_name][month].index)
                    anomalies = model_anomalies[model_name][month].values
                    color = colors[model_name]

                    ax.plot(years, anomalies, label=f'{model_name} Anomaly', color=color)

                    # Trendline
                    z = np.polyfit(years, anomalies, 1)
                    p = np.poly1d(z)
                    ax.plot(years, p(years), linestyle='--', color=color, label=f'{model_name} Trend')

                ax.set_title(calendar.month_name[month], fontsize=12, fontweight='bold')
                ax.set_xlabel('Year')
                ax.set_ylabel(f'Anomaly {curr_unit_sym}')
                ax.grid(True)
                ax.set_ylim(-1.0, 2.0)

            # Add legend at bottom center
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=12)
            plt.tight_layout(rect=[0, 0.05, 1, 1])
            plt.suptitle(f" {period_name} Anomaly Trends for {curr_folder} ({start_year} - {end_year})",
                         fontsize=14, fontweight='bold')
            
            # Save and show
            plt.savefig(os.path.join(output_path, f"{period_name}_SSP245_SSP585_Anomaly_Trendlines.png"))
            plt.show()
            plt.close()
    if __name__== "__main__":
        mean_plot_17_1()
        anomalies_ssp_17_2()




if __name__ == "__main__":
    #lat_lon_clip_csv1()
    #rearrange_folders2()
    #csv_merge_concate3()
    #time_clip4()
    #conversion_cel5()
    #regrid_csv6()
    #to_polygonwise7()
    #top5_models=evaluation8("D:\Climate\Paper_Worth\Temperature\9. Evaluation\9.1 Climatology CSV Plots", "All")
    #ensemble_make9(top5_models)
   # annual_Temperature_plot10()
    #evaluation8(fr"{folder_dir}\{curr_folder}/9. Evaluation with Ensemble\9.1 Climatology CSV Plots Ens", "Ensemble")
    #taylor_overall11(fr"{folder_dir}\{curr_folder}\9. Evaluation with Ensemble\9.1 Climatology CSV Plots Ens")
    #individual_heatmap12(fr"{folder_dir}\{curr_folder}/9. Evaluation with Ensemble\9.1 Climatology CSV Plots Ens")
    #machine_learning_part14()
    """anomaly_trendlines15(r"D:\Climate\Paper_Worth\Temperature\10. Ensemble/BMME.csv",
                         r"D:\Climate\Paper_Worth\Temperature\12. SSP Data/245_Ensemble.csv",
                         r"D:\Climate\Paper_Worth\Temperature\12. SSP Data/585_Ensemble.csv",)"""
    #ml_eval_plot16()
    #supplemenatary17()