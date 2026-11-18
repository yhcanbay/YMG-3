from pathlib import Path
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

cikti = Path(__file__).resolve().parent / "ciktilar"
cikti.mkdir(exist_ok=True)
tf.keras.utils.set_random_seed(42)

conv = tf.keras.layers.Conv2D(
    filters=16, kernel_size=(3, 3), activation="relu",
    kernel_initializer="he_normal", data_format="channels_last"
)
dummy_input = tf.random.normal((1, 128, 128, 1))
output = conv(dummy_input)
weights, biases = conv.get_weights()

metin = (
    f"Giriş boyutu: {dummy_input.shape}\n"
    f"Çıkış boyutu: {output.shape}\n"
    f"Filtre ağırlıklarının boyutu: {weights.shape}\n"
    f"Bias boyutu: {biases.shape}\n"
    f"İlk filtre:\n{weights[:, :, 0, 0]}\n"
)
print(metin)
(cikti / "konvolusyon_sonucu.txt").write_text(metin, encoding="utf-8")
np.save(cikti / "baslangic_filtreleri.npy", weights)

# Aynı renk ölçeği ile 16 başlangıç filtresini göster.
limit = abs(weights).max()
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    resim = ax.imshow(weights[:, :, 0, i], cmap="coolwarm",
                      vmin=-limit, vmax=limit)
    ax.set_title(f"Filtre {i + 1}")
    ax.axis("off")
fig.suptitle("Eğitim öncesindeki 16 filtre")
fig.colorbar(resim, ax=axes.ravel().tolist(), shrink=0.6)
fig.savefig(cikti / "baslangic_filtreleri.png", dpi=160)
plt.close(fig)
