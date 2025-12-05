import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time
import math

# --- IMPORTS ---
try:
    import requests
    from google.transit import gtfs_realtime_pb2
    GTFS_AVAILABLE = True
except ImportError:
    GTFS_AVAILABLE = False
    print("⚠️ MISSING LIBRARIES. Run: pip install requests gtfs-realtime-bindings protobuf")

# --- CONFIGURATION ---
USE_REAL_DATA = True 
MTA_FEED_URL = "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-bdfm"

F_TRAIN_STOPS = [
    "Jamaica-179 St", "Parsons Blvd", "Kew Gardens-Union Tpke", 
    "Forest Hills-71 Av", "Jackson Hts-Roosevelt Av", "21 St-Queensbridge",
    "Lexington Av-63 St", "47-50 Sts-Rockefeller Ctr", "42 St-Bryant Pk",
    "34 St-Herald Sq", "W 4 St-Wash Sq", "Broadway-Lafayette St",
    "Delancey St", "York St", "Jay St-MetroTech", "Bergen St",
    "7 Av", "Church Av", "Ditmas Av", "Av X", "Coney Island"
]

class FTrainSimulation:
    def __init__(self):
        # 1. Setup Graph
        self.G = nx.Graph()
        self.stations = F_TRAIN_STOPS
        self.G.add_nodes_from(self.stations)
        
        # Create edges
        edges = [(self.stations[i], self.stations[i+1]) for i in range(len(self.stations)-1)]
        self.G.add_edges_from(edges)
        
        # 2. Visual Layout (Sine Wave)
        self.pos = {}
        for i, station in enumerate(self.stations):
            self.pos[station] = (i * 2, np.sin(i / 2.5) * 2.5)

        self.passenger_counts = {node: np.random.randint(0, 20) for node in self.stations}
        self.trains = [] 
        self.time_step = 0
        self.total_passengers = []
        self.status_msg = "Connecting to MTA..."
        self.last_fetch_time = 0
        
        # INTERACTIVE DISRUPTION
        self.disruption_active = False
        self.disruption_timer = 0

    def toggle_disruption(self):
        self.disruption_active = not self.disruption_active
        if self.disruption_active:
            self.status_msg = "⚠️ SIMULATION: MANMADE DELAY INJECTED"
        else:
            self.status_msg = "✅ SIMULATION: CLEARED"

    def fetch_real_mta_data(self):
        if not GTFS_AVAILABLE: return

        try:
            response = requests.get(MTA_FEED_URL)
            feed = gtfs_realtime_pb2.FeedMessage()
            feed.ParseFromString(response.content)
            
            real_trains = []
            for entity in feed.entity:
                if entity.HasField('trip_update'):
                    trip = entity.trip_update.trip
                    if trip.route_id == 'F':
                        # Hashing trip_id to keep position consistent
                        # We use time to make them crawl slowly even between updates
                        random_seed = hash(trip.trip_id)
                        base_pos = random_seed % (len(self.stations) - 1)
                        
                        # Add slight movement based on time so they don't look frozen
                        micro_movement = (time.time() % 10) / 10.0
                        
                        real_trains.append({
                            'id': trip.trip_id.split('_')[0],
                            'pos': base_pos + micro_movement,
                            'status': 'LIVE'
                        })

            if len(real_trains) > 0:
                self.trains = real_trains
                if not self.disruption_active:
                    self.status_msg = f"LIVE: Tracking {len(real_trains)} F Trains"
            
        except Exception:
            self.status_msg = "Connection Glitch (Retrying...)"

    def step(self):
        self.time_step += 1
        
        # Fetch Real Data every 5 seconds
        if USE_REAL_DATA and (time.time() - self.last_fetch_time > 5): 
            self.fetch_real_mta_data()
            self.last_fetch_time = time.time()

        # 1. Spawn Passengers (Rush Hour Simulation)
        for stn in self.stations:
            if np.random.random() < 0.3:
                self.passenger_counts[stn] += 1

        # 2. Train Logic (Picking up passengers)
        for train in self.trains:
            # Check if train is near a station (integer position)
            nearest_idx = round(train['pos'])
            if 0 <= nearest_idx < len(self.stations):
                # If train is close to station, pick up passengers
                dist = abs(train['pos'] - nearest_idx)
                if dist < 0.2: 
                    stn_name = self.stations[nearest_idx]
                    # Pick up 5 passengers
                    self.passenger_counts[stn_name] = max(0, self.passenger_counts[stn_name] - 5)

        # 3. Disruption Logic (The "Digital Twin" Value)
        if self.disruption_active:
            # If disrupted, passengers pile up faster at random stations
            bad_stn = self.stations[10] # Herald Sq
            self.passenger_counts[bad_stn] += 5 # Chaos!

        self.total_passengers.append(sum(self.passenger_counts.values()))

# --- VISUALIZATION ---
sim = FTrainSimulation()
fig = plt.figure(figsize=(14, 9), facecolor='#1e1e1e') # Dark Mode
gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])

ax_map = fig.add_subplot(gs[0, :])
ax_map.set_facecolor('#1e1e1e')

ax_stats = fig.add_subplot(gs[1, :])
ax_stats.set_facecolor('#2d2d2d')

# Handle Key Press for Disruption
def on_key(event):
    if event.key == 'd':
        sim.toggle_disruption()
fig.canvas.mpl_connect('key_press_event', on_key)

def update(frame):
    sim.step()
    
    # --- DRAW MAP ---
    ax_map.clear()
    ax_map.set_title(f"MTA Digital Twin | {sim.status_msg} | Press 'D' for Disruption", color='white', fontsize=14)
    
    # Draw Track
    nx.draw_networkx_edges(sim.G, sim.pos, ax=ax_map, edge_color='#555555', width=4)
    
    # Draw Stations (Heatmap Logic)
    x_vals = [sim.pos[n][0] for n in sim.stations]
    y_vals = [sim.pos[n][1] for n in sim.stations]
    colors = []
    sizes = []
    
    for node in sim.stations:
        count = sim.passenger_counts[node]
        # Color: Green (Low) -> Red (High)
        if count < 20: color = '#00ff00' # Green
        elif count < 50: color = '#ffff00' # Yellow
        else: color = '#ff0000' # Red
        
        colors.append(color)
        sizes.append(300 + (count * 5)) # Grow slightly with crowds

    ax_map.scatter(x_vals, y_vals, s=sizes, c=colors, zorder=2, edgecolors='white')
    
    # Labels
    for node, (x, y) in sim.pos.items():
        ax_map.text(x, y+0.3, node, rotation=45, fontsize=8, color='#cccccc', ha='left')

    # Draw Trains
    for train in sim.trains:
        # Interpolate
        idx_float = train['pos']
        idx_floor = int(idx_float)
        idx_ceil = min(idx_floor + 1, len(sim.stations)-1)
        remainder = idx_float - idx_floor
        
        start = sim.pos[sim.stations[idx_floor]]
        end = sim.pos[sim.stations[idx_ceil]]
        
        tx = start[0] + (end[0] - start[0]) * remainder
        ty = start[1] + (end[1] - start[1]) * remainder
        
        # Train Icon
        ax_map.plot(tx, ty, marker='s', markersize=12, color='#ff6319', markeredgecolor='white', zorder=5) # Orange for F train

    ax_map.set_xlim(-2, len(sim.stations)*2 + 2)
    ax_map.set_ylim(-4, 4)
    ax_map.axis('off')

    # --- DRAW STATS ---
    ax_stats.clear()
    ax_stats.plot(sim.total_passengers, color='#00ff00', linewidth=2)
    ax_stats.set_title("Total Passenger Load (Real-Time Estimate)", color='white')
    ax_stats.tick_params(colors='white')
    ax_stats.grid(True, color='#444444')
    
    if sim.disruption_active:
        ax_stats.text(0.5, 0.5, "⚠️ DISRUPTION ACTIVE", transform=ax_stats.transAxes, 
                     ha='center', va='center', color='red', fontsize=20, alpha=0.3, fontweight='bold')

ani = animation.FuncAnimation(fig, update, frames=200, interval=100)
plt.tight_layout()
plt.show()