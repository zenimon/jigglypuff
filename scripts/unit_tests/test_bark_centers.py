import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.subbands import get_bark_band_centers


centers = get_bark_band_centers()

print("=" * 70)
print("IMPULSE GUARD - BARK BAND CENTER TEST")
print("=" * 70)

print()
print("Number of centers:", len(centers))
print("Shape:", centers.shape)

print()
print("Bark band center frequencies:")

for i, frequency in enumerate(centers):
    print(
        f"Band {i:02d}: "
        f"{frequency:.2f} Hz"
    )

print()
print("Minimum frequency:", centers.min())
print("Maximum frequency:", centers.max())

print()
print("DONE")
print("=" * 70)