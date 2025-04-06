# Optimal College Bus Route Planner

A web application that helps college administrators plan optimal bus routes for students based on their home locations and bus capacity constraints. The system uses transportation algorithms to minimize fuel costs or travel time.

## Features

- Upload student addresses in CSV format
- Automatically cluster student locations into optimal bus routes
- Visualize routes on an interactive map
- Optimize routes based on bus capacity constraints
- Generate detailed route reports

## Technology Stack

- Backend: Python, Flask
- Algorithms: Transportation algorithms (Vogel's Approximation Method, Least Cost Method, MODI)
- Geolocation: GeoPy
- Data Processing: NumPy, Pandas
- Visualization: Folium (interactive maps)

## Setup and Installation

1. Clone the repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   python app.py
   ```
4. Access the application at http://localhost:5000

## Usage

1. Upload a CSV file containing student addresses
2. Specify the number of buses and their capacities
3. Choose the optimization method
4. View the suggested routes on the map
5. Export the route details if needed
