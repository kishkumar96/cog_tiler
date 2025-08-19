import requests
from io import BytesIO
import matplotlib.pyplot as plt

# WMS GetMap parameters for 0-360 longitude
wms_url = "http://192.168.4.152/cgi-bin/mapserv"
params = {
    "map": "/srv/example/sst.map",
    "SERVICE": "WMS",
    "VERSION": "1.1.1",
    "REQUEST": "GetMap",
    "LAYERS": "anom",
    "STYLES": "",
    "FORMAT": "image/png",
    "TRANSPARENT": "true",
    "SRS": "EPSG:4326",
    "BBOX": "0,-90,360,90",   # longitudes 0 to 360, latitudes -90 to 90
    "WIDTH": "1024",          # adjust as needed
    "HEIGHT": "512"
}

# Make the WMS request
response = requests.get(wms_url, params=params)
response.raise_for_status()

# Plot the image
img = plt.imread(BytesIO(response.content))
plt.figure(figsize=(12,6))
plt.imshow(img, extent=[0, 360, -90, 90], aspect='auto')  # extent matches BBOX
plt.title("WMS Layer: anom (0° to 360°)")
plt.xlabel("Longitude (0–360°)")
plt.ylabel("Latitude")
plt.show()