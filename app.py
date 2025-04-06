import os
import tempfile
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
import folium
from models.transportation_algorithms import VogelsApproximationMethod, LeastCostMethod, MODI, calculate_route_distances
from models.utils import process_student_data, create_route_map, generate_route_report

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev_key_for_testing')
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Store data in memory for simplicity
# In a production app, you would use a database
app_data = {
    'student_data': None,
    'student_locations': None,
    'college_location': None,
    'bus_assignments': None,
    'route_distances': None,
    'map': None
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    # If user does not select file, browser also submits an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file:
        # Save the file temporarily
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(temp_path)
        
        try:
            # Process the student data
            student_locations, student_data = process_student_data(temp_path)
            
            # Store in app data
            app_data['student_locations'] = student_locations
            app_data['student_data'] = student_data
            
            # Get college location from form
            college_lat = float(request.form.get('college_lat', 0))
            college_lng = float(request.form.get('college_lng', 0))
            app_data['college_location'] = (college_lat, college_lng)
            
            # Clean up
            os.remove(temp_path)
            
            flash(f'Successfully processed {len(student_locations)} student locations')
            return redirect(url_for('configure_buses'))
            
        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            return redirect(url_for('index'))
    
    return redirect(url_for('index'))

@app.route('/configure_buses')
def configure_buses():
    if app_data['student_locations'] is None:
        flash('Please upload student data first')
        return redirect(url_for('index'))
    
    return render_template('configure_buses.html', 
                          student_count=len(app_data['student_locations']))

@app.route('/optimize', methods=['POST'])
def optimize_routes():
    if app_data['student_locations'] is None:
        flash('Please upload student data first')
        return redirect(url_for('index'))
    
    try:
        # Get bus configuration from form
        num_buses = int(request.form.get('num_buses', 1))
        bus_capacities = []
        
        for i in range(num_buses):
            capacity = int(request.form.get(f'bus_capacity_{i}', 0))
            bus_capacities.append(capacity)
        
        # Get optimization method
        method = request.form.get('method', 'vogels')
        
        # Run the selected optimization algorithm
        if method == 'vogels':
            solver = VogelsApproximationMethod(
                app_data['student_locations'], 
                bus_capacities, 
                app_data['college_location']
            )
        elif method == 'least_cost':
            solver = LeastCostMethod(
                app_data['student_locations'], 
                bus_capacities, 
                app_data['college_location']
            )
        elif method == 'modi':
            solver = MODI(
                app_data['student_locations'], 
                bus_capacities, 
                app_data['college_location']
            )
        else:
            flash('Invalid optimization method')
            return redirect(url_for('configure_buses'))
        
        # Solve the transportation problem
        bus_assignments = solver.solve()
        app_data['bus_assignments'] = bus_assignments
        
        # Calculate route distances
        route_distances = calculate_route_distances(
            bus_assignments,
            app_data['student_locations'],
            app_data['college_location']
        )
        app_data['route_distances'] = route_distances
        
        # Create the route map
        route_map = create_route_map(
            app_data['student_data'],
            bus_assignments,
            app_data['college_location'],
            route_distances
        )
        app_data['map'] = route_map
        
        # Save the map to a file in the static folder
        os.makedirs('static', exist_ok=True)
        map_path = os.path.join('static', 'route_map.html')
        route_map.save(map_path)
        
        flash('Routes optimized successfully')
        return redirect(url_for('view_routes'))
        
    except Exception as e:
        flash(f'Error optimizing routes: {str(e)}')
        return redirect(url_for('configure_buses'))

@app.route('/view_routes')
def view_routes():
    if app_data['bus_assignments'] is None:
        flash('Please optimize routes first')
        return redirect(url_for('configure_buses'))
    
    # Generate route statistics
    stats = []
    for bus_id, student_indices in app_data['bus_assignments'].items():
        if not student_indices:
            continue
            
        bus_stats = {
            'id': bus_id + 1,  # 1-indexed for display
            'student_count': len(student_indices),
            'distance': app_data['route_distances'].get(bus_id, 0)
        }
        stats.append(bus_stats)
    
    return render_template('view_routes.html', stats=stats)

@app.route('/report')
def generate_report():
    if app_data['bus_assignments'] is None:
        flash('Please optimize routes first')
        return redirect(url_for('configure_buses'))
    
    report_html = generate_route_report(
        app_data['student_data'],
        app_data['bus_assignments'],
        app_data['route_distances']
    )
    
    return render_template('report.html', report=report_html)

@app.route('/download_sample')
def download_sample():
    """Provide a sample CSV file for users to download"""
    # Create a sample DataFrame
    sample_data = {
        'id': [1, 2, 3, 4, 5],
        'name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Davis'],
        'address': [
            '123 Main St, New York, NY',
            '456 Oak Ave, Los Angeles, CA',
            '789 Pine Rd, Chicago, IL',
            '101 Maple Dr, Houston, TX',
            '202 Cedar Ln, Phoenix, AZ'
        ]
    }
    df = pd.DataFrame(sample_data)
    
    # Save to a temporary file
    temp_file = os.path.join(app.config['UPLOAD_FOLDER'], 'sample_students.csv')
    df.to_csv(temp_file, index=False)
    
    return send_file(temp_file, as_attachment=True, download_name='sample_students.csv')

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True)
