from pathlib import Path
import csv
import time
import platform
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

cikti = Path(__file__).resolve().parent / "ciktilar"
cikti.mkdir(exist_ok=True)
SEED = 42
EPOCHS = 5
BATCH_SIZE = 64

# İlk çalıştırmada Fashion-MNIST internetten indirilir.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]
print("Eğitim verisi:", x_train.shape)
print("Test verisi:", x_test.shape)


def model_olustur(filtre_sayisi):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(filtre_sayisi, (3, 3), activation="relu",
                               data_format="channels_last"),
        tf.keras.layers.MaxPooling2D((2, 2), data_format="channels_last"),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu",
                               data_format="channels_last"),
        tf.keras.layers.MaxPooling2D((2, 2), data_format="channels_last"),
        tf.keras.layers.Flatten(data_format="channels_last"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])


sonuclar = []
for filtre in [8, 16]:
    # Her deney yeni ağırlıklarla başlar. Diğer ayarlar aynı tutulur.
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    model = model_olustur(filtre)
    model.summary()
    with open(cikti / f"model_ozeti_{filtre}.txt", "w", encoding="utf-8") as dosya:
        model.summary(print_fn=lambda satir: dosya.write(satir + "\n"))

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    baslangic = time.perf_counter()
    history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_split=0.1, shuffle=True)
    sure = time.perf_counter() - baslangic
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Filtre: {filtre}, Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")

    satir = {
        "filtre_sayisi": filtre,
        "egitim_dogrulugu": history.history["accuracy"][-1],
        "dogrulama_dogrulugu": history.history["val_accuracy"][-1],
        "test_dogrulugu": test_accuracy,
        "test_loss": test_loss,
        "egitim_suresi_saniye": sure,
        "parametre_sayisi": model.count_params()
    }
    sonuclar.append(satir)
    with open(cikti / f"egitim_gecmisi_{filtre}.csv", "w", newline="", encoding="utf-8") as dosya:
        yazici = csv.writer(dosya)
        yazici.writerow(["epoch"] + list(history.history))
        for i in range(EPOCHS):
            yazici.writerow([i + 1] + [history.history[k][i] for k in history.history])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metrik, baslik in zip(axes, ["accuracy", "loss"], ["Doğruluk", "Kayıp"]):
        ax.plot(range(1, EPOCHS + 1), history.history[metrik], label="Eğitim")
        ax.plot(range(1, EPOCHS + 1), history.history["val_" + metrik], label="Doğrulama")
        ax.set_title(baslik)
        ax.set_xlabel("Epoch")
        ax.set_xticks(range(1, EPOCHS + 1))
        ax.legend()
    fig.suptitle(f"İlk katmanda {filtre} filtre")
    fig.tight_layout()
    fig.savefig(cikti / f"egitim_grafikleri_{filtre}.png", dpi=160)
    plt.close(fig)

    olasiliklar = model.predict(x_test[:10], verbose=0)
    tahminler = np.argmax(olasiliklar, axis=1)
    np.savetxt(cikti / f"ornek_tahminler_{filtre}.csv",
               np.column_stack([y_test[:10], tahminler]), delimiter=",", fmt="%d",
               header="gercek_etiket,tahmin", comments="")

with open(cikti / "deney_karsilastirma.csv", "w", newline="", encoding="utf-8-sig") as dosya:
    yazici = csv.DictWriter(dosya, fieldnames=list(sonuclar[0]))
    yazici.writeheader()
    yazici.writerows(sonuclar)

satirlar = ["# Ölçülen deney sonuçları", "",
    "Eğitim doğruluğu: son epoch eğitim metriği. Süre: yalnızca fit çağrısı, saniye.",
    "Doğruluklar tabloda yüzde olarak verilmiştir.", "",
    "| Filtre | Eğitim (%) | Doğrulama (%) | Test (%) | Test kaybı | Süre (sn) |",
    "|---|---|---|---|---|---|"]
for s in sonuclar:
    satirlar.append(f"| {s['filtre_sayisi']} | {100*s['egitim_dogrulugu']:.2f} | "
                    f"{100*s['dogrulama_dogrulugu']:.2f} | {100*s['test_dogrulugu']:.2f} | "
                    f"{s['test_loss']:.4f} | {s['egitim_suresi_saniye']:.2f} |")
fark = 100 * (sonuclar[1]["test_dogrulugu"] - sonuclar[0]["test_dogrulugu"])
satirlar += ["", f"16 filtreli deneyin test doğruluğu farkı: {fark:+.2f} yüzde puan.",
    "Bu sonuç yalnızca bu çalıştırmayı gösterir. Daha fazla filtrenin her zaman daha iyi olduğu sonucunu çıkaramam.",
    "İki deney birer kez ve sırayla çalıştırıldı. Isınma, donanım yükü ve rastgelelik süreyi/sonuçları etkileyebilir.",
    "", "## Çalışma ortamı", f"Python: {platform.python_version()}",
    f"TensorFlow: {tf.__version__}", f"NumPy: {np.__version__}",
    f"Sistem: {platform.platform()}", f"Cihazlar: {tf.config.list_physical_devices()}",
    "", "## Kendi yorumum", "Grafiklere bakıp eğitim-doğrulama farkını ve süreyi burada yorumlayacağım."]
(cikti / "olculen_sonuclar.md").write_text("\n".join(satirlar), encoding="utf-8")
print("\n".join(satirlar))
