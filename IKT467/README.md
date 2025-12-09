# IKT467 Traffic Jam Model: Mixed Autonomous & Human Traffic

**Authors:** Matin Mohammadi, Robert Hanssen, Sander Gnanavel  
**Course:** IKT467 - Final Project (Fall 2025)

## 🚀 Live Demo (Run in Browser)
The easiest way to test the simulation is via the live web application. No installation is required.

 **[Click Here to Launch Simulation](https://ikt467-traffic-jam-model.streamlit.app/)** 

---

## About
This project simulates traffic flow dynamics using the **Intelligent Driver Model (IDM)**. It investigates the impact of Autonomous Vehicles (AVs) on traffic stability, specifically focusing on the dampening of "phantom traffic jams" (stop-and-go waves).

The model allows for real-time visualization of:
- **Time-Space Diagrams:** Visualizing vehicle trajectories and wave propagation.
- **Fundamental Diagrams:** Analyzing the flow-density-speed relationship.
- **Safety Metrics:** Tracking time-to-collision and near-miss events.

### Scenarios Included
1. **Free Flow:** Stable low-density traffic.
2. **Phantom Jam:** Emergent stop-and-go waves caused by minor perturbations.
3. **Driver Variation:** Mixed aggressive/calm human driver profiles.
4. **Bottleneck:** Simulation of speed reduction zones.
5. **NetLogo Validation:** Specific scenarios designed to verify correctness against established agent-based models.

---

## 🛠 Local Installation
If you prefer to run the code locally, follow these steps:

### Prerequisites
- Python 3.9+
- pip

### 1. Clone the repository
```bash
git clone [https://github.com/Roberh18/AI_FALL_2025.git](https://github.com/Roberh18/AI_FALL_2025.git)
cd AI_FALL_2025/IKT467
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run IKT467_traffic_jam_model.py
```

The application will automatically open in your default web browser at http://localhost:8501.
