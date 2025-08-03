import pandas as pd
import numpy as np
from pyproj import Transformer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_interactions import ioff, panhandler, zoom_factory

# === 1. Données : Valeurs centrales RPC ===
valeurs_centrales_data = {
    "ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
           11, 12, 13, 14, 15, 16, 18, 19, 20, 22, 23, 25, 26],
    "Latitude Offset": [
        30.2992, 30.2994, 30.3007, 30.2996, 30.2998, 30.3003, 30.2999, 30.2835, 30.3027, 30.3151,
        30.2996, 30.2998, 30.3009, 30.3013, 30.3004, 30.301, 30.2994, 30.3, 30.2995, 30.2995,
        30.2944, 30.2783, 30.2749
    ],
    "Longitude Offset": [
        -81.6397, -81.6397, -81.6396, -81.6403, -81.6408, -81.6408, -81.6402, -81.6402, -81.6401, -81.64,
        -81.6406, -81.6409, -81.6401, -81.6386, -81.6414, -81.64, -81.6398, -81.6409, -81.6396, -81.6412,
        -81.6414, -81.6306, -81.619
    ],
    "Height Offset": [
        -21, -21, -19, -20, -19, -21, -21, -21, -19, -19,
        -21, -21, -21, -19, -19, -19, -21, -21, -19, -21,
        -21, -19, -19
    ]
}
df_valeurs = pd.DataFrame(valeurs_centrales_data)

# === 2. Données : Azimuth et Élévation ===
angle_data = {
    "ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
           11, 12, 13, 14, 15, 16, 18, 19, 20, 21,
           22, 23, 25, 26],
    "Az": [
        0.3455751918948773, 0.4188790204786391, 2.410299697004169, 4.665265090580843,
        1.9495327744776663, 0.7435102613495844, 3.8117990863556157, 3.7140606482439336,
        3.5883969421003417, 3.5238197597765515, 2.5185101106278176, 0.06632251157578452,
        3.2759830059933566, 4.54309204294124, 1.4905111812031575, 3.8117990863556157,
        4.32143522793796, 1.9442967867216832, 4.516912104161325, 4.434881629317592,
        0.9320058205649719, 2.9286624848464853, 3.7088246604879505, 4.679227724596798
    ],
    "El": [
        1.2095131716320704, 1.3334315485236679, 1.2356931104119853, 1.429424657383356,
        1.260127719939906, 1.2967796342317868, 1.3439035240356338, 1.2880529879718152,
        1.1501719770642633, 1.0088003076527223, 1.4765485471872026, 1.2636183784438948,
        1.2287117934040082, 1.171115928088195, 1.1030480872604163, 1.2147491593880533,
        1.457349925415265, 1.356120828799594, 1.101302758008422, 1.0053096491487339,
        1.3543754995475996, 1.2287117934040082, 1.1798425743481666, 1.0524335389525807
    ]
}
df_angles = pd.DataFrame(angle_data)

# === 3. Fusion ===
df_angles = df_angles[df_angles["ID"].isin(df_valeurs["ID"])]
df_combined = df_valeurs.merge(df_angles, on="ID")

# === 4. Transformation ECEF et estimation satellite ===
transformer = Transformer.from_crs("epsg:4326", "epsg:4978", always_xy=True)
sat_alt = 617000  # Altitude estimée

satellite_ecef = []
origin_ecef = []
vectors = []

for _, row in df_combined.iterrows():
    lon, lat, alt = row["Longitude Offset"], row["Latitude Offset"], row["Height Offset"]
    az, el = row["Az"], row["El"]

    dx = -np.cos(el) * np.sin(az)
    dy = -np.cos(el) * np.cos(az)
    dz =  np.sin(el)
    direction = np.array([dx, dy, dz])

    x0, y0, z0 = transformer.transform(lon, lat, alt)
    ground_point = np.array([x0, y0, z0])
    sat_pos = ground_point + direction * sat_alt

    origin_ecef.append(ground_point)
    satellite_ecef.append(sat_pos)
    vectors.append(sat_pos - ground_point)

origin_ecef = np.array(origin_ecef)
satellite_ecef = np.array(satellite_ecef)
vectors = np.array(vectors)
# === 4b. Calcul du point moyen des points au sol (en ECEF) ===
center_point = np.mean(origin_ecef, axis=0)

# === 4c. Conversion des points au sol vers ENU ===
def ecef_to_enu(ecef_coords, ref_coords, lat_ref, lon_ref):
    lat = np.radians(lat_ref)
    lon = np.radians(lon_ref)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    # Matrice de rotation ECEF → ENU
    R = np.array([
        [-sin_lon,             cos_lon,              0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon,    cos_lat],
        [ cos_lat * cos_lon,  cos_lat * sin_lon,    sin_lat]
    ])

    delta = ecef_coords - ref_coords
    return delta @ R.T

# Référence locale (moyenne des points)
ref_lat = df_combined["Latitude Offset"].mean()
ref_lon = df_combined["Longitude Offset"].mean()
ref_alt = df_combined["Height Offset"].mean()

ref_x, ref_y, ref_z = transformer.transform(ref_lon, ref_lat, ref_alt)
ref_coords = np.array([ref_x, ref_y, ref_z])

# Conversion
origin_enu = np.array([ecef_to_enu(p, ref_coords, ref_lat, ref_lon) for p in origin_ecef])

# === 5. Visualisation ENU ===
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Points au sol en ENU
ax.scatter(
    origin_enu[:, 0]/1000,  # Est
    origin_enu[:, 1]/1000,  # Nord
    origin_enu[:, 2]/1000,  # Up
    color='green',
    s=30,
    label='Point au sol'
)

# ID des points
for i, pos in enumerate(origin_enu):
    ax.text(pos[0], pos[1], pos[2], str(df_combined["ID"].iloc[i]), color='black', fontsize=8)

# Axes centrés
x_min, x_max = origin_enu[:, 0].min(), origin_enu[:, 0].max()
y_min, y_max = origin_enu[:, 1].min(), origin_enu[:, 1].max()
z_min, z_max = origin_enu[:, 2].min(), origin_enu[:, 2].max()

margin = 10  # mètres pour un zoom fin
ax.set_xlim(x_min - margin, x_max + margin)
ax.set_ylim(y_min - margin, y_max + margin)
ax.set_zlim(z_min - margin, z_max + margin)

# Labels
ax.set_title("Distribution des points au sol (ENU)")
ax.set_xlabel("Est (km)")
ax.set_ylabel("Nord (km)")
ax.set_zlabel("Up (km)")
ax.legend(loc='upper right')

plt.tight_layout()
plt.show()
