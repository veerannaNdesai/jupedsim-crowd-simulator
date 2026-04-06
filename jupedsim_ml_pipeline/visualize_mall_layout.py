import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as mpl_Polygon
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from geometry.mall_geometry import build_mall_geometry

def visualize():
    # 1. Build geometry
    walkable_area, zones, exit_points = build_mall_geometry()

    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_aspect('equal')
    ax.set_title('Mall Geometry Layout', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Meters (X)', fontsize=12)
    ax.set_ylabel('Meters (Y)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Define color palette for zones
    zone_colors = {
        "Lobby": "#AED6F1",           # Blue
        "RetailArea": "#F9E79F",      # Yellow
        "PedestrianPathway": "#D5DBDB",# Grey
        "QueueArea": "#F5B7B1",       # Reddish
        "ServicePoint": "#ABEBC6",    # Greenish
        "WaitingArea": "#D2B4DE",     # Purple
        "EntryExitPoint": "#FAD7A0"   # Orange
    }

    # 3. Plot walkable area
    # Plotting the walkable area as a background with a light grey boundary
    x, y = walkable_area.exterior.xy
    ax.plot(x, y, color='black', linewidth=2, alpha=0.8, zorder=1)
    ax.fill(x, y, color='#F8F9F9', alpha=1.0, zorder=0)

    # 4. Plot each zone
    for name, zmeta in zones.items():
        color = zone_colors.get(zmeta.zone_type, "#D0D3D4")
        
        # Plot Polygon
        patch = mpl_Polygon(list(zmeta.polygon.exterior.coords), closed=True,
                           facecolor=color, edgecolor='#2C3E50', alpha=0.7, 
                           label=zmeta.zone_type, zorder=2)
        ax.add_patch(patch)
        
        # Add Label at Centroid
        centroid = zmeta.polygon.centroid
        ax.text(centroid.x, centroid.y, f"{name}\n({zmeta.area_sqm}m²)", 
                ha='center', va='center', fontsize=9, fontweight='bold', 
                color='#34495E', zorder=3)

    # 5. Plot exit points
    for exit_pt in exit_points:
        x, y = exit_pt.polygon.exterior.xy
        ax.plot(x, y, color='#C0392B', linewidth=3, linestyle='-', zorder=4)
        ax.fill(x, y, color='#E74C3C', alpha=0.6, zorder=4)
        
        # Label exit
        centroid = exit_pt.polygon.centroid
        ax.text(centroid.x, centroid.y, "EXIT", ha='center', va='center', 
                fontsize=8, color='white', fontweight='bold', zorder=5)

    # 6. Add legend (handling unique labels)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', 
              bbox_to_anchor=(1, 1), fontsize=10)

    # 7. Final Polish
    plt.tight_layout()
    
    # Ensure visualization directory exists
    output_path = PROJECT_ROOT / "visualizations" / "mall_layout.png"
    plt.savefig(output_path, dpi=120)
    print(f"Visualization saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    visualize()
