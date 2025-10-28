import numpy as np
clean = np.loadtxt("wifi_db/clean_dataset.txt")
noisy = np.loadtxt("wifi_db/noisy_dataset.txt")

track = []
counter = 0
for row_clean in clean:
    for row_noisy in noisy:
        if np.array_equal(row_clean[:-1], row_noisy[:-1]):
            if row_clean[-1] != row_noisy[-1]:
                counter += 1
                track.append(row_clean[-1])  # On garde juste la classe vraie

print("Nombre total d'erreurs :", counter)

# Comptage des erreurs par classe
apparition = {}
for label in track:
    apparition[label] = apparition.get(label, 0) + 1

print("Errors per class :", apparition)


print("Number per class noisy: ", np.bincount(noisy.astype(int)[:, -1]))
print("Number per class clean: ", np.bincount(clean.astype(int)[:, -1]))

