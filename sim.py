import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time
from datetime import datetime
from collections import deque

# --- IMPORTS ---
try:
    import requests
    from google.transit import gtfs_realtime_pb2
    GTFS_AVAILABLE = True
except ImportError:
    GTFS_AVAILABLE = False
    print("⚠️ MISSING LIBRARIES. Run: pip install requests gtfs-realtime-bindings protobuf")
    print("🔄 Running in SIMULATION MODE instead...")

# --- CONFIGURATION ---
USE_REAL_DATA = GTFS_AVAILABLE  # Auto-disable if libraries missing
MTA_FEED_URL = "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-bdfm"
MTA_API_KEY = None  # Set this if you have one for better rate limits

F_TRAIN_STOPS = [
    # Queens - 10 stops
    "Jamaica-179 St", "169 St", "Parsons Blvd", "Sutphin Blvd", "Briarwood",
    "Kew Gardens-Union Tpke", "75 Av", "Forest Hills-71 Av", 
    "Jackson Hts-Roosevelt Av", "21 St-Queensbridge",
    # Manhattan - 13 stops
    "Roosevelt Island", "Lexington Av/63 St", "57 St", 
    "47-50 Sts-Rockefeller Ctr", "42 St-Bryant Pk", "34 St-Herald Sq",
    "23 St", "14 St", "W 4 St-Washington Sq", "Broadway-Lafayette St",
    "2 Av", "Delancey St-Essex St", "East Broadway",
    # Brooklyn - 22 stops
    "York St", "Jay St-MetroTech", "Bergen St", "Carroll St", "Smith-9 Sts",
    "4 Av-9 St", "7 Av", "15 St-Prospect Park", "Fort Hamilton Pkwy",
    "Church Av", "Ditmas Av", "18 Av", "Avenue I", "Bay Pkwy", "Avenue N",
    "Avenue P", "Kings Hwy", "Avenue U", "Avenue X", "Neptune Av",
    "W 8 St-NY Aquarium", "Coney Island-Stillwell Av"
]

# Stop IDs for mapping to station indices
STOP_ID_MAP = {
    # Queens
    "F11": 0, "F12": 1, "F14": 2, "F15": 3, "F16": 4,
    "F18": 5, "F20": 6, "F21": 7, "F22": 8, "F23": 9,
    # Manhattan
    "F24": 10, "F25": 11, "F26": 12, "F27": 13, "D15": 14, "D14": 15,
    "D13": 16, "D12": 17, "A24": 18, "D11": 19, "F18": 20, "F20": 21, "F22": 22,
    # Brooklyn
    "A41": 23, "A42": 24, "F24": 25, "F25": 26, "F26": 27,
    "F27": 28, "G22": 29, "F29": 30, "F30": 31, "F31": 32,
    "F32": 33, "F33": 34, "F34": 35, "F35": 36, "F36": 37,
    "F38": 38, "F39": 39, "F40": 40, "F41": 41, "F42": 42,
    "F43": 43, "D43": 44
}

class Train:
    def __init__(self, train_id, position, direction, status='MOVING'):
        self.id = train_id
        self.position = position  # Float between 0 and len(stations)-1
        self.direction = direction  # 'S' (south to Coney) or 'N' (north to Jamaica)
        self.status = status  # 'MOVING', 'STOPPED', 'BOARDING'
        self.passengers = np.random.randint(50, 200)
        self.last_update = time.time()
        self.last_position = position
        self.last_position_time = time.time()
        
    def interpolate_position(self):
        """Smoothly interpolate between known positions based on time"""
        # Don't interpolate, just return actual position from MTA feed
        return self.position

class FTrainSimulation:
    def __init__(self):
        # 1. Setup Graph
        self.G = nx.Graph()
        self.stations = F_TRAIN_STOPS
        self.G.add_nodes_from(self.stations)
        
        edges = [(self.stations[i], self.stations[i+1]) for i in range(len(self.stations)-1)]
        self.G.add_edges_from(edges)
        
        # 2. Visual Layout (Sine Wave with better spacing)
        self.pos = {}
        for i, station in enumerate(self.stations):
            self.pos[station] = (i * 2.2, np.sin(i / 2.8) * 3)

        # 3. Passenger Management
        self.passenger_counts = {node: np.random.randint(2, 8) for node in self.stations}
        self.passenger_history = {node: deque(maxlen=50) for node in self.stations}
        
        # 4. Train Management
        self.trains = []
        self.train_history = deque(maxlen=100)
        self._init_simulation_trains()
        
        # 5. Time & Status
        self.time_step = 0
        self.last_fetch_time = 0
        self.fetch_interval = 10  # seconds
        self.status_msg = "🔄 SIMULATION MODE" if not USE_REAL_DATA else "Connecting to MTA..."
        self.alerts = []
        
        # 6. Rush Hour Simulation
        self.rush_hour_multiplier = 1.0
        
        # 7. Interactive Features
        self.disruption_active = False
        self.disruption_location = 15  # 34 St-Herald Sq (major hub)
        self.selected_station = None
        
        # 8. Statistics
        self.total_passengers_history = deque(maxlen=200)
        self.trains_in_service = deque(maxlen=200)

    def _init_simulation_trains(self):
        """Initialize trains for simulation mode"""
        if not USE_REAL_DATA:
            # Create 8 trains going both directions
            for i in range(4):
                # Southbound trains
                self.trains.append(Train(
                    f"F-S{i+1}", 
                    i * 5, 
                    'S'
                ))
                # Northbound trains
                self.trains.append(Train(
                    f"F-N{i+1}", 
                    len(self.stations) - 1 - i * 5, 
                    'N'
                ))

    def get_rush_hour_multiplier(self):
        """Calculate passenger spawn rate based on time of day"""
        hour = datetime.now().hour
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            return 2.5  # Rush hour
        elif 10 <= hour <= 16:
            return 1.2  # Mid-day
        elif 20 <= hour <= 23:
            return 1.5  # Evening
        else:
            return 0.6  # Late night/early morning

    def fetch_real_mta_data(self):
        """Fetch live train positions from MTA GTFS-RT feed"""
        if not GTFS_AVAILABLE:
            return

        try:
            headers = {'x-api-key': MTA_API_KEY} if MTA_API_KEY else {}
            response = requests.get(MTA_FEED_URL, headers=headers, timeout=5)
            response.raise_for_status()
            
            feed = gtfs_realtime_pb2.FeedMessage()
            feed.ParseFromString(response.content)
            
            new_trains = {}
            f_train_count = 0
            
            # Process both vehicle positions and trip updates
            for entity in feed.entity:
                route_id = None
                train_id = None
                stop_id = None
                direction = 'S'
                status = 'MOVING'
                current_stop_sequence = None
                
                # Try vehicle position first (most accurate for real-time location)
                if entity.HasField('vehicle'):
                    vehicle = entity.vehicle
                    route_id = vehicle.trip.route_id
                    
                    # FILTER: Only process F trains
                    if route_id == 'F':
                        f_train_count += 1
                        train_id = vehicle.trip.trip_id
                        stop_id = vehicle.stop_id if vehicle.stop_id else None
                        current_stop_sequence = vehicle.current_stop_sequence if vehicle.current_stop_sequence else None
                        
                        # Determine direction from trip_id
                        if '.N' in train_id or train_id.endswith('N'):
                            direction = 'N'
                        elif '.S' in train_id or train_id.endswith('S'):
                            direction = 'S'
                        else:
                            direction = 'S'
                        
                        # Vehicle current_status: 0=INCOMING, 1=STOPPED_AT, 2=IN_TRANSIT_TO
                        status = 'STOPPED' if vehicle.current_status == 1 else 'MOVING'
                
                # Fallback to trip_update if no vehicle position
                elif entity.HasField('trip_update'):
                    trip = entity.trip_update.trip
                    route_id = trip.route_id
                    
                    # FILTER: Only process F trains
                    if route_id == 'F':
                        f_train_count += 1
                        train_id = trip.trip_id
                        
                        # Get current or next stop from stop_time_updates
                        if entity.trip_update.stop_time_update:
                            stop_id = entity.trip_update.stop_time_update[0].stop_id
                            current_stop_sequence = entity.trip_update.stop_time_update[0].stop_sequence
                        
                        direction = 'N' if '.N' in train_id else 'S'
                        status = 'MOVING'
                
                # If we found an F train, add it
                if train_id and route_id == 'F':
                    position = None
                    
                    # PRIORITY 1: Use stop_sequence (most reliable in MTA feeds)
                    if current_stop_sequence is not None:
                        # Stop sequence typically goes 1-45 for F train
                        # Map it to our 0-44 index range
                        position = min(max(current_stop_sequence - 1, 0), len(self.stations) - 1)
                    
                    # PRIORITY 2: Try mapping stop_id if we have it
                    elif stop_id:
                        position = STOP_ID_MAP.get(stop_id, None)
                        
                        # Try removing N/S suffix
                        if position is None:
                            base_stop = stop_id.rstrip('NS')
                            position = STOP_ID_MAP.get(base_stop, None)
                    
                    # Only add trains with valid positions
                    if position is not None:
                        new_trains[train_id] = Train(train_id, position, direction, status)
            
            # Update trains with real data ONLY
            if new_trains:
                existing_ids = {t.id: t for t in self.trains}
                
                for train_id, new_train in new_trains.items():
                    if train_id in existing_ids:
                        # Update existing train with NEW REAL POSITION from MTA
                        existing_ids[train_id].position = new_train.position
                        existing_ids[train_id].direction = new_train.direction
                        existing_ids[train_id].status = new_train.status
                        existing_ids[train_id].last_update = time.time()
                    else:
                        # Add new train
                        self.trains.append(new_train)
                
                # Remove trains that are no longer in the feed (older than 2 minutes)
                current_time = time.time()
                self.trains = [t for t in self.trains 
                              if t.id in new_trains or (current_time - t.last_update < 120)]
                
                # Limit to reasonable number
                if len(self.trains) > 12:
                    self.trains.sort(key=lambda x: x.last_update, reverse=True)
                    self.trains = self.trains[:12]
                
                self.status_msg = f"🚇 LIVE: {len(self.trains)} F Trains Active | Refresh: 10s"
            else:
                self.status_msg = "⚠️ No F trains with valid positions in feed"
            
            # Fetch service alerts
            self.fetch_alerts()
            
        except requests.exceptions.RequestException as e:
            self.status_msg = f"⚠️ Connection Error: {str(e)[:30]}..."
        except Exception as e:
            self.status_msg = f"⚠️ Data Error: {str(e)[:30]}..."

    def fetch_alerts(self):
        """Fetch service alerts for F train"""
        # This would require the service alerts feed
        # Placeholder for now
        pass

    def toggle_disruption(self):
        """Toggle manual disruption for testing"""
        self.disruption_active = not self.disruption_active
        if self.disruption_active:
            self.status_msg = "⚠️ DISRUPTION SIMULATION ACTIVE"
            self.alerts.append("Signal problems at Herald Square")
        else:
            self.status_msg = "✅ Disruption Cleared"
            self.alerts = [a for a in self.alerts if "Signal problems" not in a]

    def step(self):
        """Main simulation step"""
        self.time_step += 1
        dt = 1  # timestep
        
        # Update rush hour multiplier
        self.rush_hour_multiplier = self.get_rush_hour_multiplier()
        
        # Fetch real data periodically
        if USE_REAL_DATA and (time.time() - self.last_fetch_time > self.fetch_interval):
            self.fetch_real_mta_data()
            self.last_fetch_time = time.time()
        
        # 1. Passenger Spawning (with rush hour and station-specific rates)
        # Major hubs get more passengers, outer stations get fewer
        major_hubs = {
            "34 St-Herald Sq": 0.4,
            "W 4 St-Washington Sq": 0.35,
            "Jackson Hts-Roosevelt Av": 0.35,
            "Jay St-MetroTech": 0.3,
            "42 St-Bryant Pk": 0.3,
            "Coney Island-Stillwell Av": 0.25,
            "Jamaica-179 St": 0.25,
            "Lexington Av/63 St": 0.25,
            "14 St": 0.2,
        }
        
        for stn in self.stations:
            # Get station-specific rate (default 0.08 for outer stations)
            base_rate = major_hubs.get(stn, 0.08)
            spawn_rate = base_rate * self.rush_hour_multiplier
            
            # Only spawn if station isn't already overcrowded
            if self.passenger_counts[stn] < 60:  # Cap at 60 passengers per station
                if np.random.random() < spawn_rate:
                    self.passenger_counts[stn] += 1
        
        # 2. Passenger pickup at stations (based on real train positions)
        for train in self.trains:
            # Check if train is at or very near a station
            nearest_station_idx = round(train.position)
            if 0 <= nearest_station_idx < len(self.stations):
                distance_to_station = abs(train.position - nearest_station_idx)
                
                # If train is at the station (regardless of status), pick up passengers
                # MTA feed doesn't always report STOPPED accurately
                if distance_to_station < 0.3:
                    station_name = self.stations[nearest_station_idx]
                    
                    # Passengers board (limited by capacity)
                    available_capacity = 250 - train.passengers
                    boarding = min(self.passenger_counts[station_name], 30, available_capacity)
                    
                    # Random alighting (10-30 people get off)
                    alighting = min(np.random.randint(10, 30), train.passengers)
                    
                    # Update counts
                    self.passenger_counts[station_name] = max(0, self.passenger_counts[station_name] - boarding)
                    train.passengers += boarding - alighting
                    train.passengers = max(0, min(train.passengers, 250))
        
        # 3. Disruption Effects
        if self.disruption_active:
            disrupted_station = self.stations[self.disruption_location]
            # Passengers pile up
            self.passenger_counts[disrupted_station] += 3
            
            # Slow down nearby trains
            for train in self.trains:
                if abs(train.position - self.disruption_location) < 2:
                    train.position += 0.001  # Barely move
        
        # 4. Record History
        total_passengers = sum(self.passenger_counts.values())
        self.total_passengers_history.append(total_passengers)
        self.trains_in_service.append(len(self.trains))
        
        for station in self.stations:
            self.passenger_history[station].append(self.passenger_counts[station])

# --- VISUALIZATION ---
sim = FTrainSimulation()
fig = plt.figure(figsize=(18, 11), facecolor='#0a0a0a')  # Increased from 16x10
gs = fig.add_gridspec(3, 2, height_ratios=[3, 1, 1], width_ratios=[2, 1])

ax_map = fig.add_subplot(gs[0, :])
ax_map.set_facecolor('#0a0a0a')

ax_passengers = fig.add_subplot(gs[1, 0])
ax_passengers.set_facecolor('#1a1a1a')

ax_trains = fig.add_subplot(gs[1, 1])
ax_trains.set_facecolor('#1a1a1a')

ax_detail = fig.add_subplot(gs[2, :])
ax_detail.set_facecolor('#1a1a1a')

# Event Handlers
def on_key(event):
    if event.key == 'd':
        sim.toggle_disruption()
    elif event.key == 'r':
        sim.rush_hour_multiplier = 3.0  # Force rush hour

def on_click(event):
    if event.inaxes == ax_map:
        # Find nearest station
        if event.xdata and event.ydata:
            distances = []
            for station in sim.stations:
                x, y = sim.pos[station]
                dist = np.sqrt((x - event.xdata)**2 + (y - event.ydata)**2)
                distances.append((dist, station))
            sim.selected_station = min(distances)[1]

fig.canvas.mpl_connect('key_press_event', on_key)
fig.canvas.mpl_connect('button_press_event', on_click)

def update(frame):
    sim.step()
    
    # --- MAP VIEW ---
    ax_map.clear()
    time_str = datetime.now().strftime("%H:%M:%S")
    title = f"MTA F-Train Digital Twin | {time_str} | {sim.status_msg}"
    if sim.alerts:
        title += f" | 🚨 {sim.alerts[0]}"
    ax_map.set_title(title, color='white', fontsize=13, pad=15, wrap=True)  # Increased padding
    
    # Draw track
    nx.draw_networkx_edges(sim.G, sim.pos, ax=ax_map, edge_color='#333333', width=6, alpha=0.5)
    
    # Draw stations with heatmap
    x_vals, y_vals, colors, sizes = [], [], [], []
    for node in sim.stations:
        count = sim.passenger_counts[node]
        x, y = sim.pos[node]
        x_vals.append(x)
        y_vals.append(y)
        
        # Color gradient
        if count < 15:
            color = '#00ff88'  # Green
        elif count < 30:
            color = '#ffff00'  # Yellow
        elif count < 50:
            color = '#ff8800'  # Orange
        else:
            color = '#ff0000'  # Red
        
        colors.append(color)
        sizes.append(400 + count * 8)
    
    ax_map.scatter(x_vals, y_vals, s=sizes, c=colors, zorder=3, 
                   edgecolors='white', linewidths=2, alpha=0.8)
    
    # Station labels
    for node, (x, y) in sim.pos.items():
        label = node.split('-')[0]  # Shortened name
        ax_map.text(x, y + 0.5, label, rotation=35, fontsize=7, 
                   color='#aaaaaa', ha='left', va='bottom')
        # Passenger count
        count = sim.passenger_counts[node]
        ax_map.text(x, y - 0.4, f"{count}", fontsize=8, 
                   color='white', ha='center', va='top', fontweight='bold')
    
    # Draw trains with direction indicators
    for train in sim.trains:
        idx_float = train.position
        idx_floor = int(idx_float)
        idx_ceil = min(idx_floor + 1, len(sim.stations) - 1)
        remainder = idx_float - idx_floor
        
        start = sim.pos[sim.stations[idx_floor]]
        end = sim.pos[sim.stations[idx_ceil]]
        
        tx = start[0] + (end[0] - start[0]) * remainder
        ty = start[1] + (end[1] - start[1]) * remainder
        
        # Train marker (use matplotlib built-in markers)
        marker = 'v' if train.direction == 'S' else '^'  # triangle down/up
        color = '#ff6319' if train.status == 'MOVING' else '#ffaa00'
        ax_map.plot(tx, ty, marker=marker, markersize=15, color=color, 
                   markeredgecolor='white', markeredgewidth=2, zorder=5)
        
        # Train ID label
        ax_map.text(tx, ty + 0.7, train.id[:4], fontsize=6, 
                   color='#ffcc00', ha='center', fontweight='bold')
    
    # Highlight selected station
    if sim.selected_station:
        x, y = sim.pos[sim.selected_station]
        circle = plt.Circle((x, y), 0.6, color='cyan', fill=False, linewidth=3, zorder=4)
        ax_map.add_patch(circle)
    
    ax_map.set_xlim(-2, len(sim.stations) * 2.2 + 2)
    ax_map.set_ylim(-5, 5)
    ax_map.axis('off')
    
    # Add legend
    legend_text = "Controls: [D] Toggle Disruption | [R] Rush Hour | Click station for details"
    ax_map.text(0.02, 0.02, legend_text, transform=ax_map.transAxes, 
               color='#888888', fontsize=9, va='bottom')
    
    # --- PASSENGER GRAPH ---
    ax_passengers.clear()
    if len(sim.total_passengers_history) > 0:
        ax_passengers.plot(list(sim.total_passengers_history), color='#00ff88', linewidth=2)
        ax_passengers.fill_between(range(len(sim.total_passengers_history)), 
                                   list(sim.total_passengers_history), 
                                   alpha=0.3, color='#00ff88')
    ax_passengers.set_title("System Passenger Load", color='white', fontsize=10)
    ax_passengers.set_ylabel("Total Waiting", color='#888888', fontsize=8)
    ax_passengers.tick_params(colors='#888888', labelsize=8)
    ax_passengers.grid(True, color='#333333', alpha=0.3)
    ax_passengers.set_facecolor('#1a1a1a')
    
    # --- TRAINS GRAPH ---
    ax_trains.clear()
    if len(sim.trains_in_service) > 0:
        ax_trains.plot(list(sim.trains_in_service), color='#ff6319', linewidth=2)
    ax_trains.set_title("Trains in Service", color='white', fontsize=10)
    ax_trains.set_ylabel("Active Trains", color='#888888', fontsize=8)
    ax_trains.tick_params(colors='#888888', labelsize=8)
    ax_trains.grid(True, color='#333333', alpha=0.3)
    ax_trains.set_facecolor('#1a1a1a')
    
    # --- DETAIL VIEW ---
    ax_detail.clear()
    if sim.selected_station:
        history = list(sim.passenger_history[sim.selected_station])
        ax_detail.plot(history, color='#ffff00', linewidth=2, marker='o', markersize=3)
        ax_detail.fill_between(range(len(history)), history, alpha=0.3, color='#ffff00')
        ax_detail.set_title(f"📍 {sim.selected_station} - Passenger Trend", 
                           color='white', fontsize=10)
        ax_detail.set_ylabel("Waiting", color='#888888', fontsize=8)
    else:
        ax_detail.text(0.5, 0.5, "Click a station to see details", 
                      transform=ax_detail.transAxes, ha='center', va='center',
                      color='#666666', fontsize=12)
    
    ax_detail.tick_params(colors='#888888', labelsize=8)
    ax_detail.grid(True, color='#333333', alpha=0.3)
    ax_detail.set_facecolor('#1a1a1a')
    
    # Disruption overlay
    if sim.disruption_active:
        ax_map.text(0.5, 0.95, "⚠️ DISRUPTION ACTIVE", transform=ax_map.transAxes,
                   ha='center', va='top', color='red', fontsize=16, 
                   fontweight='bold', alpha=0.7)

ani = animation.FuncAnimation(fig, update, interval=100, cache_frame_data=False)
plt.tight_layout()
plt.subplots_adjust(top=0.96)  # Add space at top for title
plt.show()