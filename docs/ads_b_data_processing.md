## Overview of ADS-B Data Processing Tasks

This document outlines the main steps involved in preparing ADS-B flight data for use in contrail avoidance modeling.
The processing pipeline consists of four key functions, each responsible for a specific aspect of data cleaning and preparation.

### 1. `generate_flight_dataframe_from_adsb_data`

- **Purpose:** Reads raw ADS-B flight data into a pandas DataFrame.
- **Actions:** Removes unnecessary columns to reduce memory usage.

### 2. `select_subset_of_adsb_flight_data`

- **Purpose:** Filters the dataset to focus on relevant flights or flight segments.
- **Filtering Options:**
  - **By Airports:** Select flights based on departure and arrival airports. Options include:
    - All flights
    - Regional flights
    - Regional and international UK flights
  - **By Time:** Select flights based on time period. Options include:
    - Whole dataset
    - First month of data

### 3. `clean_adsb_flight_dataframe`

- **Purpose:** Cleans and enhances the flight data for compatibility and analysis.
- **Key Steps:**
  - Converts `altitude_baro` to `flight_level` for compatibility with the COSIP grid model.
  - Adds a `distance_flown_in_segment` column, representing the distance flown between consecutive timestamps.
  - Removes rows where `distance_flown_in_segment` is zero, eliminating stationary or duplicate points.
  - Fills missing altitude values with the previous altitude for the same flight, ensuring that each timestamp has a valid altitude.
  - Interpolates across large data gaps using the `generate_interpolated_rows_of_large_distance_flights` function to improve data completeness and allow us to sample from the environmental grid more effectively.

- **Output:** Cleaned flight data.

### 4. `process_adsb_flight_data_for_environment`

- **Purpose:** Prepares the cleaned data for environmental modeling by removing irrelevant data points.
- **Actions:** Removes data points with low flight levels, focusing on relevant flight segments for COSIP grid modeling.

- **Output:** Flight data suitable for running the COSIP grid model.
