import pickle
import streamlit as st
from ortools.constraint_solver import pywrapcp
import pandas as pd
from geopy.distance import geodesic

st.set_page_config(page_title='Biskfarm Beat Plan Optimization Route Order Sequence', layout="wide")

# Load the model and data
model = pickle.load(open('rfr_model.pkl', 'rb'))
data = pd.read_excel("BF_Final_Data_for_Analysis_9226_Retailers_Modified.xlsx")
df = pd.DataFrame(data)

# HTML template for styling
html_temp = """
    <div style="background-color:#032863;padding:2px">
    <h5 style="color:white;text-align:center;">Biskfarm Beat Plan Optimization Model</h5>
    </div>
    <div style="background-color:white;padding:7px">
    <h4 style="color:black;text-align:center;font-size:15px; font-weight:bold">Vehicle Route Order Sequence</h4>
    </div>
    <style>
    [data-testid="stAppViewContainer"]{
        opacity: 0.8;
        }
    [data-testid="stHeader"]{
        background-color:rgba(255,255,255);
        }
    [data-testid="stToolbar"]{
        right=2rem;
        }
    </style>
    """

st.markdown(html_temp, unsafe_allow_html=True)

# Streamlit UI
def main_streamlit():
    st.sidebar.title("Biskfarm BPO")

    # Take Inputs in the sidebar
    Distributor = st.sidebar.selectbox("Distributor", df['Distributor_Code'].unique())
    farm_num_df = df.groupby(['Distributor_Code'])['route_code'].unique().reset_index()
    pivot_df = pd.pivot_table(farm_num_df, index='Distributor_Code', values='route_code', aggfunc=lambda x: x)
    pivot_df_2 = pivot_df.applymap(lambda z: z[:1] if isinstance(z, list) else z)
    exploded_df = pivot_df['route_code'].explode().reset_index()
    filtered_routes = exploded_df[exploded_df['Distributor_Code'] == Distributor]
    RouteCode = st.sidebar.selectbox("Route Code", filtered_routes['route_code'].unique())

    if st.sidebar.button("Retailer Order Sequence"):
        # Filter data based on selected distributor and route code
        filtered_data = df[(df['Distributor_Code'] == Distributor) & (df['route_code'] == RouteCode)].reset_index(drop=True)

        # Display basic information about the selected route
        # st.subheader("Retailer Sequence")
        st.write(f"Selected Distributor: {Distributor}")
        st.write(f"Selected Route Code: {RouteCode}")

        if filtered_data.empty:
            st.warning("No data found for the selected distributor and route code.")
            return

        # Create data model for the selected route
        data = create_data_model(filtered_data)
        manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            return data['distance_matrix'][manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.time_limit.seconds = 900

        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            st.success("Optimization successful!")
            print_solution(manager, routing, solution, data)
        else:
            st.error("Optimization failed.")


# Function to calculate distance between two coordinates
def calculate_distance(coord1, coord2):
    return geodesic(coord1, coord2).meters


# Function to create data model for the selected route
def create_data_model(filtered_data):
    data = {}
    data['num_vehicles'] = 1
    data['depot'] = 0  # Starting and ending point

    # Combine Distributor_Code and route_code to create unique points
    filtered_data['point'] = filtered_data['Distributor_Code'].astype(str) + '_' + filtered_data['route_code'].astype(str)

    coordinates = [(lat, lon) for lat, lon in zip(filtered_data['LAT'], filtered_data['LON'])]
    addresses = filtered_data['address'].tolist()
    points = filtered_data['point'].tolist()

    data['coordinates'] = coordinates
    data['addresses'] = addresses
    data['points'] = points

    # Calculate distance matrix based on coordinates
    data['distance_matrix'] = [[calculate_distance(coord1, coord2) for coord2 in data['coordinates']] for coord1 in data['coordinates']]

    return data


# Function to print the optimization solution
def print_solution(manager, routing, solution, data):
    index = routing.Start(0)
    plan_output = 'Retailer Sequence: '
    route_distance = 0

    while not routing.IsEnd(index):
        plan_output += ' {} ->'.format(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)

    plan_output += ' {}'.format(manager.IndexToNode(index))
    st.write(plan_output)
    st.write('Total Distance: {} meters'.format(route_distance))

    index = routing.Start(0)
    retailers_sequence = []

    while not routing.IsEnd(index):
        retailers_sequence.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))

    st.write('Retailer Sequence:', retailers_sequence)


# Run the Streamlit app
if __name__ == '__main__':
    main_streamlit()
