import streamlit as st
from ortools.constraint_solver import pywrapcp
import pandas as pd
from geopy.distance import geodesic

# Read the dataset
data = {
    'LAT': [22.47493172, 22.5166444, 22.5137715, 22.5170472, 22.5170472, 22.5178231, 22.5157835, 22.5147072, 22.5183428,
            22.5207436, 22.5187175, 22.5168743, 22.519161, 22.5156279, 22.5166406, 22.5178167, 22.5174215, 22.5176453,
            22.5158306, 22.5149775, 22.5170472, 22.517213, 22.5170617, 22.5207507, 22.5168232, 22.5198122, 22.5170472,
            22.5185925],
    'LON': [88.37574768, 88.3771268, 88.3810669, 88.3769777, 88.3769777, 88.3773113, 88.3791872, 88.3800399, 88.376082,
            88.3735501, 88.3752368, 88.377145, 88.375725, 88.3791552, 88.377963, 88.377695, 88.3739007, 88.3746922,
            88.3784794, 88.379499, 88.3769777, 88.3787935, 88.3743883, 88.3735467, 88.3792819, 88.37363, 88.3769777,
            88.3759468],
    'address': [
        '87/10A Raja S.C.Mallick Rd', '1 N K GHOSH RD', '190 KASBA RATHTALA KOL', 'KASBA 3', 'kasba 3',
        '254 B.B .CHATTERJEE Road', '104/1 KUMAR Para LANE Kol', '220 B B CHATTERJEE ROAD KOL', '88B.B chatterjee road',
        'kasba c I t Market Kol',
        '77 B B CHATTERJEE  ROAD', '2/1N,k, GHOSHAL ROAD KOL', 'B.B.chattergee Road kolkata',
        '201/c B.B CHATTERJEE ROAD', '55 R.K.GHOSAL RAOD', '117 B B CHATTERJEE ROAD', '29 KUMAR PARA LANE',
        '15 KUMOR PARA', '222 B.B.CHATTERJEE RD KOL - 42',
        '222 B.B.CHATTERJEEW RD  KOL - 42', 'kasba 3', '122/B B B CHATTERJEE ROAD', '68,KUMAR Para LANE Kol',
        '14 D D N Sen ROAD Kol', '240 B B CHATTERJEE ROAD KOL', 'kasba', 'Kasba 3', '68 B.B.CHATERJEE ROAD KOLKATA'
    ]
}
df = pd.DataFrame(data)


def calculate_distance(coord1, coord2):
    return geodesic(coord1, coord2).meters


def create_data_model():
    data = {}
    data['num_vehicles'] = 1
    data['depot'] = 0  # Starting and ending point
    coordinates = [(lat, lon) for lat, lon in zip(df['LAT'], df['LON'])]
    addresses = df['address'].tolist()
    # Exclude starting and ending point from coordinates and addresses
    coordinates_without_depot = coordinates[1:]
    addresses_without_depot = addresses[1:]
    data['coordinates'] = coordinates_without_depot
    data['addresses'] = addresses_without_depot
    # Calculate distance matrix excluding starting and ending point
    data['distance_matrix'] = [[calculate_distance(coord1, coord2) for coord2 in data['coordinates']] for coord1 in
                               data['coordinates']]
    return data


def main():
    data = create_data_model()
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
        print_solution(manager, routing, solution, data)


def print_solution(manager, routing, solution, data):
    # st.write('Objective: {}'.format(solution.ObjectiveValue()))
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


# Streamlit UI
def main_streamlit():
    st.title("Vehicle Routing Problem Solver")
    data = create_data_model()
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


# Run Streamlit app
if __name__ == '__main__':
    main_streamlit()
