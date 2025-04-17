import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import datetime
import base64
from io import BytesIO

# Set page configuration
st.set_page_config(page_title="Interactive Map Explorer", layout="wide")

# Application title and description
st.title("Interactive Map Explorer")
st.markdown("""
This application allows you to explore geographic data on an interactive map.
You can filter by data category and time range, and download the filtered dataset.
""")

# Function to load data
@st.cache_data
def load_data():
    # In a real application, you might load data from a database or API
    # For this example, we'll create sample data
    
    # Sample data: crime incidents in various cities
    data = {
        'latitude': [40.7128, 34.0522, 41.8781, 37.7749, 39.9526, 
                    40.7128, 34.0522, 41.8781, 37.7749, 39.9526],
        'longitude': [-74.0060, -118.2437, -87.6298, -122.4194, -75.1652,
                    -74.0060, -118.2437, -87.6298, -122.4194, -75.1652],
        'location': ['New York', 'Los Angeles', 'Chicago', 'San Francisco', 'Philadelphia',
                    'New York', 'Los Angeles', 'Chicago', 'San Francisco', 'Philadelphia'],
        'category': ['Theft', 'Assault', 'Theft', 'Vandalism', 'Theft',
                   'Traffic Accident', 'Vandalism', 'Traffic Accident', 'Assault', 'Traffic Accident'],
        'datetime': [
            datetime.datetime(2023, 1, 15, 14, 30),
            datetime.datetime(2023, 2, 20, 23, 15),
            datetime.datetime(2023, 3, 5, 10, 45),
            datetime.datetime(2023, 1, 10, 18, 20),
            datetime.datetime(2023, 2, 8, 8, 15),
            datetime.datetime(2023, 4, 12, 13, 0),
            datetime.datetime(2023, 3, 25, 2, 30),
            datetime.datetime(2023, 4, 18, 16, 45),
            datetime.datetime(2023, 5, 7, 14, 10),
            datetime.datetime(2023, 5, 22, 19, 30)
        ],
        'description': [
            'Bicycle stolen from front yard',
            'Altercation outside restaurant',
            'Shoplifting at convenience store',
            'Graffiti on public building',
            'Package theft from porch',
            'Minor collision at intersection',
            'Window broken at business',
            'Vehicle collision with property damage',
            'Dispute escalated to physical altercation',
            'Hit and run incident'
        ],
        'severity': [2, 4, 2, 1, 2, 3, 2, 3, 3, 4]
    }
    
    return pd.DataFrame(data)

# Function to generate download link for CSV
def get_csv_download_link(df, filename="filtered_data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href

# Load the data
df = load_data()

# Sidebar filters
st.sidebar.header("Data Filters")

# Category filter
categories = ['All'] + list(df['category'].unique())
selected_category = st.sidebar.selectbox("Select Category", categories)

# Date range filter
date_min = df['datetime'].min().date()
date_max = df['datetime'].max().date()
selected_date_range = st.sidebar.date_input(
    "Select Date Range",
    [date_min, date_max],
    min_value=date_min,
    max_value=date_max
)

# Apply filters
filtered_df = df.copy()

# Filter by category
if selected_category != 'All':
    filtered_df = filtered_df[filtered_df['category'] == selected_category]

# Filter by date range
if len(selected_date_range) == 2:
    start_date, end_date = selected_date_range
    filtered_df = filtered_df[
        (filtered_df['datetime'].dt.date >= start_date) & 
        (filtered_df['datetime'].dt.date <= end_date)
    ]

# Display filtered data count
st.sidebar.write(f"Showing {len(filtered_df)} records")

# Download button for filtered data
st.sidebar.markdown(get_csv_download_link(filtered_df), unsafe_allow_html=True)

# Create two columns for the layout
col1, col2 = st.columns([2, 1])

with col1:
    # Create map
    st.subheader("Interactive Map")
    
    # Initialize map centered at the mean of the filtered data
    if not filtered_df.empty:
        map_center = [filtered_df['latitude'].mean(), filtered_df['longitude'].mean()]
    else:
        map_center = [0, 0]  # Default center if no data
    
    m = folium.Map(location=map_center, zoom_start=5)
    
    # Add marker cluster
    marker_cluster = MarkerCluster().add_to(m)
    
    # Add markers for each data point
    for idx, row in filtered_df.iterrows():
        # Color based on category
        category_colors = {
            'Theft': 'red',
            'Assault': 'darkred',
            'Vandalism': 'orange',
            'Traffic Accident': 'blue'
        }
        color = category_colors.get(row['category'], 'gray')
        
        # Create popup content
        popup_content = f"""
        <b>Location:</b> {row['location']}<br>
        <b>Category:</b> {row['category']}<br>
        <b>Date/Time:</b> {row['datetime'].strftime('%Y-%m-%d %H:%M')}<br>
        <b>Description:</b> {row['description']}<br>
        <b>Severity:</b> {row['severity']}
        """
        
        # Add marker
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup_content, max_width=300),
            tooltip=f"{row['category']} in {row['location']}",
            icon=folium.Icon(color=color, icon="info-sign")
        ).add_to(marker_cluster)
    
    # Display the map
    folium_static(m, width=800)

with col2:
    # Display data table
    st.subheader("Data Table")
    
    # Format the datetime for display
    display_df = filtered_df.copy()
    display_df['datetime'] = display_df['datetime'].dt.strftime('%Y-%m-%d %H:%M')
    
    # Show the table with selected columns
    st.dataframe(
        display_df[['location', 'category', 'datetime', 'severity', 'description']],
        height=600
    )

# Add some statistics
st.subheader("Data Statistics")

# Show category distribution in the filtered data
cat_counts = filtered_df['category'].value_counts().reset_index()
cat_counts.columns = ['Category', 'Count']

col3, col4 = st.columns(2)

with col3:
    st.write("Category Distribution:")
    st.table(cat_counts)

with col4:
    if not filtered_df.empty:
        st.write("Severity Statistics:")
        st.table(pd.DataFrame({
            'Statistic': ['Mean', 'Median', 'Min', 'Max'],
            'Value': [
                round(filtered_df['severity'].mean(), 2),
                filtered_df['severity'].median(),
                filtered_df['severity'].min(),
                filtered_df['severity'].max()
            ]
        }))

# Footer
st.markdown("---")
st.markdown("Interactive Map Explorer - Geographic Data Visualization Tool")