import matplotlib.pyplot as plt

# OAHEGA class names and image numbers
classes = ["Happy", "Angry", "Sad", "Neutral", "Surprise", "Ahegao"]
counts  = [3740, 1313, 3934, 4027, 1234, 1205]

plt.figure(figsize=(8, 4))
plt.bar(classes, counts)
plt.xlabel("Emotion Class")
plt.ylabel("Number of Images")
plt.title("OAHEGA Dataset Class Distribution")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
