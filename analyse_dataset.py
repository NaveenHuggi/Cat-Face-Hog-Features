import json
import statistics

path = r'c:\Users\Navee\Desktop\cat face\all_50_cat_breeds_master_biometrics.json'

print("Loading JSON (large file, please wait)...")
with open(path, 'r') as f:
    data = json.load(f)

print("\n=== DATASET OVERVIEW ===")
print("Dataset Name :", data.get('dataset'))
print("Total Cats   :", data.get('total_cats_extracted'))

breeds = data.get('breeds', [])
print("Breeds in file:", len(breeds))

print("\n=== PER-BREED STATISTICS ===")
print(f"{'Breed':<45} {'Cats':>6} {'FeatureDim':>10} {'Min':>8} {'Max':>8} {'Mean':>8} {'StdDev':>8}")
print("-" * 100)

grand_total_cats = 0
all_dims = set()
for b in breeds:
    name = b['breed_name']
    cats = b.get('cats', [])
    grand_total_cats += len(cats)
    if not cats:
        print(f"{name:<45} {'0':>6} {'N/A':>10}")
        continue

    feature_dim = len(cats[0].get('features', []))
    all_dims.add(feature_dim)
    all_features = []
    for cat in cats:
        all_features.extend(cat['features'])

    mn   = min(all_features)
    mx   = max(all_features)
    mean = statistics.mean(all_features)
    std  = statistics.stdev(all_features)
    print(f"{name:<45} {len(cats):>6} {feature_dim:>10} {mn:>8.4f} {mx:>8.4f} {mean:>8.4f} {std:>8.4f}")

print("-" * 100)
print(f"{'TOTAL':<45} {grand_total_cats:>6}")
print("\nAll feature dimensions seen:", all_dims)

# HOG breakdown: 1764 = 9 orientations x 7x7 = 441 cells x 4 (maybe)
# Common: 1764 = 9 * 196  => 14x14 blocks, or 9 * 4 * 49 = 1764 (7x7 cells, 2x2 blocks)
dim = 1764
print(f"\n=== HOG FEATURE VECTOR BREAKDOWN (dim={dim}) ===")
# 1764 / 9 = 196 cells
cells = dim // 9
print(f"  Orientations: 9")
print(f"  Total cells (dim/9): {cells}  => sqrt ~ {cells**0.5:.1f} x {cells**0.5:.1f}")
# 14x14 = 196 cells
print(f"  Likely grid: 14 x 14 = 196 cells (if 1 cell per histogram)")
print(f"  Alternative: 7 x 7 cells, 2 x 2 block stride => (6 blocks x 6 blocks) x 4 cells x 9 = 1296 (not matching)")

# Sample a few cats from the most-represented breed
print("\n=== SAMPLE CAT ENTRIES (first 3 cats of largest breed) ===")
largest = max(breeds, key=lambda b: len(b.get('cats', [])))
print(f"Breed: {largest['breed_name']}")
for cat in largest['cats'][:3]:
    feats = cat['features']
    print(f"  Cat ID: {cat['cat_id']}, File: {cat['original_file']}, Feature Vector Length: {len(feats)}")
    print(f"    First 5 values : {feats[:5]}")
    print(f"    Last  5 values : {feats[-5:]}")
    print()

print("=== DONE ===")
