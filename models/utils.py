import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time
import folium
from folium.plugins import MarkerCluster
import requests
import polyline
from geopy.distance import geodesic

def geocode_address(address, geolocator=None, retries=3, delay=1):
    """
    Convert an address string to latitude and longitude
    
    Args:
        address: String address to geocode
        geolocator: Optional Nominatim geolocator instance
        retries: Number of retries if geocoding fails
        delay: Delay between retries in seconds
        
    Returns:
        Tuple of (latitude, longitude) or None if geocoding fails
    """
    if geolocator is None:
        geolocator = Nominatim(user_agent="college_bus_route_planner")
    
    for attempt in range(retries):
        try:
            location = geolocator.geocode(address)
            if location:
                return (location.latitude, location.longitude)
            return None
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            if attempt == retries - 1:
                print(f"Geocoding failed for {address}: {e}")
                return None
            time.sleep(delay)
    
    return None

def process_student_data(file_path):
    """
    Process student data from a CSV file
    
    Args:
        file_path: Path to CSV file containing student data
        
    Returns:
        Tuple of (student_locations, student_data) where:
            - student_locations is a list of (lat, lng) tuples
            - student_data is a pandas DataFrame with student information
    """
    # Read student data
    student_data = pd.read_csv(file_path)
    
    # Check if the CSV already has latitude and longitude columns
    if 'latitude' in student_data.columns and 'longitude' in student_data.columns:
        student_locations = list(zip(student_data['latitude'], student_data['longitude']))
        return student_locations, student_data
    
    # If not, geocode the addresses
    geolocator = Nominatim(user_agent="college_bus_route_planner")
    
    # Combine address fields if they exist separately
    if 'address' not in student_data.columns:
        address_columns = [col for col in student_data.columns if 'address' in col.lower()]
        if address_columns:
            student_data['address'] = student_data[address_columns].apply(
                lambda x: ', '.join(str(val) for val in x if pd.notna(val)), axis=1
            )
        else:
            raise ValueError("CSV file must contain either 'address' column or latitude/longitude columns")
    
    # Geocode addresses
    locations = []
    for address in student_data['address']:
        location = geocode_address(address, geolocator)
        locations.append(location)
        time.sleep(1)  # Be nice to the geocoding service
    
    # Add latitude and longitude to the DataFrame
    student_data['latitude'] = [loc[0] if loc else None for loc in locations]
    student_data['longitude'] = [loc[1] if loc else None for loc in locations]
    
    # Remove rows with failed geocoding
    student_data = student_data.dropna(subset=['latitude', 'longitude'])
    
    # Extract locations as list of tuples
    student_locations = list(zip(student_data['latitude'], student_data['longitude']))
    
    return student_locations, student_data

def get_road_route(start_point, end_point):
    """
    Get a route that follows roads between two points using OSRM API
    
    Args:
        start_point: Tuple of (latitude, longitude) for start location
        end_point: Tuple of (latitude, longitude) for end location
        
    Returns:
        List of (latitude, longitude) points representing the route
    """
    try:
        # Format coordinates for OSRM API (note: OSRM expects lng,lat order)
        coords = f"{start_point[1]},{start_point[0]};{end_point[1]},{end_point[0]}"
        
        # Call OSRM API
        url = f"http://router.project-osrm.org/route/v1/driving/{coords}?overview=full&geometries=polyline"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            if data["code"] == "Ok" and len(data["routes"]) > 0:
                # Decode the polyline to get the route coordinates
                route_points = polyline.decode(data["routes"][0]["geometry"])
                # Convert to (lat, lng) format
                route_points = [(lat, lng) for lat, lng in route_points]
                return route_points
    except Exception as e:
        print(f"Error getting road route: {e}")
    
    # If there's any error, return a direct line (just start and end points)
    return [start_point, end_point]

def create_route_map(student_data, bus_assignments, college_location, route_distances=None):
    """
    Create an interactive map showing bus routes
    
    Args:
        student_data: DataFrame containing student information
        bus_assignments: Dictionary mapping bus IDs to lists of student indices
        college_location: (lat, lng) tuple for college location
        route_distances: Optional dictionary mapping bus IDs to route distances
        
    Returns:
        Folium map object
    """
    # Create a map centered at the college
    m = folium.Map(location=college_location, zoom_start=12)
    
    # Add college marker
    folium.Marker(
        location=college_location,
        popup="College",
        icon=folium.Icon(color="red", icon="university", prefix="fa"),
    ).add_to(m)
    
    # Define colors for different bus routes
    colors = ['blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 
              'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'pink', 'lightblue', 
              'lightgreen', 'gray', 'black', 'lightgray']
    
    # Create a marker cluster for each bus
    for bus_id, student_indices in bus_assignments.items():
        if not student_indices:
            continue
            
        color = colors[bus_id % len(colors)]
        
        # Create a marker cluster for this bus
        marker_cluster = MarkerCluster(name=f"Bus {bus_id + 1}").add_to(m)
        
        # Add student markers to the cluster
        for idx in student_indices:
            student_row = student_data.iloc[idx]
            location = (student_row['latitude'], student_row['longitude'])
            
            # Get student name if available, otherwise use index
            student_name = student_row.get('name', f"Student {idx}")
            
            folium.Marker(
                location=location,
                popup=f"Bus {bus_id + 1}: {student_name}",
                icon=folium.Icon(color=color, icon="user", prefix="fa"),
            ).add_to(marker_cluster)
        
        # Create a route that follows roads
        route_points = [college_location]  # Start at college
        
        # Get the optimal order of student visits (simple greedy approach)
        current_location = college_location
        remaining_indices = student_indices.copy()
        visit_order = []
        
        while remaining_indices:
            # Find closest student
            min_distance = float('inf')
            closest_idx = None
            
            for idx in remaining_indices:
                student_row = student_data.iloc[idx]
                location = (student_row['latitude'], student_row['longitude'])
                distance = geodesic(current_location, location).kilometers
                if distance < min_distance:
                    min_distance = distance
                    closest_idx = idx
            
            # Add to route
            visit_order.append(closest_idx)
            current_location = (student_data.iloc[closest_idx]['latitude'], 
                               student_data.iloc[closest_idx]['longitude'])
            remaining_indices.remove(closest_idx)
        
        # Generate road-based routes between each point
        all_route_points = []
        current_point = college_location
        
        # Add routes from college to each student in order
        for idx in visit_order:
            student_row = student_data.iloc[idx]
            next_point = (student_row['latitude'], student_row['longitude'])
            
            # Get road-based route between current and next point
            road_route = get_road_route(current_point, next_point)
            all_route_points.extend(road_route)
            
            current_point = next_point
        
        # Add route back to college
        road_route = get_road_route(current_point, college_location)
        all_route_points.extend(road_route)
        
        # Add the route line
        route_info = f"Bus {bus_id + 1}"
        if route_distances and bus_id in route_distances:
            route_info += f" - {route_distances[bus_id]:.2f} km"
            
        folium.PolyLine(
            all_route_points,
            color=color,
            weight=3,
            opacity=0.7,
            popup=route_info
        ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def generate_route_report(student_data, bus_assignments, route_distances=None):
    """
    Generate a detailed report of bus routes
    
    Args:
        student_data: DataFrame containing student information
        bus_assignments: Dictionary mapping bus IDs to lists of student indices
        route_distances: Optional dictionary mapping bus IDs to route distances
        
    Returns:
        HTML string containing the report
    """
    report = "<h2>Bus Route Report</h2>"
    
    for bus_id, student_indices in bus_assignments.items():
        if not student_indices:
            continue
            
        report += f"<h3>Bus {bus_id + 1}</h3>"
        
        if route_distances and bus_id in route_distances:
            report += f"<p>Total route distance: {route_distances[bus_id]:.2f} km</p>"
            
        report += f"<p>Number of students: {len(student_indices)}</p>"
        report += "<table border='1'><tr><th>Student ID</th><th>Name</th><th>Address</th></tr>"
        
        for idx in student_indices:
            student_row = student_data.iloc[idx]
            student_id = student_row.get('id', idx)
            student_name = student_row.get('name', f"Student {idx}")
            student_address = student_row.get('address', '')
            
            report += f"<tr><td>{student_id}</td><td>{student_name}</td><td>{student_address}</td></tr>"
            
        report += "</table>"
    
    return report
