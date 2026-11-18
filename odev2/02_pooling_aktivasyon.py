from pathlib import Path
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

cikti = Path(__file__).resolve().parent / "ciktilar"
cikti.mkdir(exist_ok=True)
input_data = tf.constant([
    [1., 3., 2., 4.],
    [5., 6., 1., 2.],
    [7., 2., 8., 3.],
    [4., 1., 5., 9.]
])
input_data = tf.reshape(input_data, (1, 4, 4, 1))
pool = tf.keras.layers.MaxPooling2D((2, 2), strides=2,
                                     data_format="channels_last")
output = pool(input_data)
giris = input_data.numpy()[0, :, :, 0]
sonuc = output.numpy()[0, :, :, 0]
elle = np.array([[6., 4.], [7., 9.]])

x = tf.constant([-5., -2., -1., 0., 2., 4., 6.])
y = tf.keras.layers.ReLU()(x)
logits = tf.constant([[2.0, 1.0, 0.1]])
probabilities = tf.keras.layers.Softmax()(logits)
metin = (
    f"Pooling girişi:\n{giris}\n"
    f"Pooling sonucu:\n{sonuc}\n"
    f"Elle hesaplanan:\n{elle}\n"
    f"Sonuçlar aynı mı? {np.array_equal(sonuc, elle)}\n"
    f"Giriş şekli: {input_data.shape}\nÇıkış şekli: {output.shape}\n"
    f"ReLU girişi: {x.numpy()}\nReLU çıkışı: {y.numpy()}\n"
    f"Softmax sonucu: {probabilities.numpy()}\n"
    f"Olasılık toplamı: {tf.reduce_sum(probabilities).numpy()}\n"
    f"En yüksek olasılıklı sınıf indeksi: {tf.argmax(probabilities, axis=1).numpy()}\n"
)
print(metin)
(cikti / "pooling_aktivasyon_sonuclari.txt").write_text(metin, encoding="utf-8")

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
for ax, matris, baslik in zip(axes, [giris, sonuc], ["Giriş (4 x 4)", "Max pooling (2 x 2)"]):
    ax.imshow(matris, cmap="Blues", vmin=0, vmax=9)
    for satir in range(matris.shape[0]):
        for sutun in range(matris.shape[1]):
            ax.text(sutun, satir, str(int(matris[satir, sutun])),
                    ha="center", va="center", color="black")
    ax.set_title(baslik)
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
fig.savefig(cikti / "pooling_giris_cikis.png", dpi=160)
plt.close(fig)
