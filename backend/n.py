from pathlib import Path

root = Path("data/PlantVillage")
classes = [d.name for d in sorted(root.iterdir()) if d.is_dir()]

print(f"Found {len(classes)} classes:")
for c in classes:
    count = len(list((root / c).iterdir()))
    print(f"  {c}  →  {count} images")