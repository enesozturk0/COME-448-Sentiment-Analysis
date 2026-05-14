"""
COME 448 - Doğal Dil İşleme (NLP) (Sentiment Analysis)

3 Farklı Sınıflandırıcı ile Karşılaştırma

Veri Seti: Turkish Product Reviews (fthbrmnby/turkish_product_reviews)
Modeller:
    1. dbmdz/bert-base-turkish-cased (BERTurk)
    2. dbmdz/distilbert-base-turkish-cased (DistilBERTurk)
    3. xlm-roberta-base (XLM-RoBERTa)
"""

import os
import sys
import time
import warnings

# Windows konsol encoding düzeltmesi (Türkçe karakter desteği)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Hugging Face bağlantı ayarları
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

warnings.filterwarnings("ignore")

# ============================================================
# Yapılandırma
# ============================================================
MODELS = {
    "BERTurk": "dbmdz/bert-base-turkish-cased",
    "DistilBERTurk": "dbmdz/distilbert-base-turkish-cased",
    "XLM-RoBERTa": "xlm-roberta-base",
}

# Eğitim hiperparametreleri
MAX_LENGTH = 128          # Maksimum token uzunluğu
BATCH_SIZE = 16           # Batch boyutu
LEARNING_RATE = 2e-5      # Öğrenme oranı
NUM_EPOCHS = 3            # Epoch sayısı
WARMUP_STEPS = 500        # Warmup adımları
WEIGHT_DECAY = 0.01       # Ağırlık azaltma
TRAIN_SIZE = 10000        # Eğitim verisi boyutu (hızlı eğitim için)
TEST_SIZE = 2000          # Test verisi boyutu

# Sonuç klasörü
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Cihaz ayarı
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Kullanılan cihaz: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Bellek: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")


# ============================================================
# 1. Veri Yükleme ve Ön İşleme
# ============================================================
def load_and_prepare_data():
    """Turkish Product Reviews veri setini yükler ve hazırlar."""
    print("\n" + "=" * 60)
    print("1. VERİ SETİ YÜKLEME VE ÖN İŞLEME")
    print("=" * 60)

    # Veri setini Hugging Face'den yükle
    print("Veri seti indiriliyor: fthbrmnby/turkish_product_reviews")
    dataset = load_dataset("fthbrmnby/turkish_product_reviews")

    # Veri setinin yapısını incele
    print(f"\nVeri seti yapısı: {dataset}")
    print(f"Toplam veri sayısı: {len(dataset['train'])}")

    # Pandas DataFrame'e çevir
    df = pd.DataFrame(dataset["train"])
    print(f"\nSütunlar: {df.columns.tolist()}")
    print(f"\nİlk 5 satır:")
    print(df.head())

    # Etiket dağılımı
    print(f"\nEtiket dağılımı:")
    print(df["sentiment"].value_counts())

    # Etiketleri sayısal değerlere dönüştür
    # sentiment sütunu: "positive" -> 1, "negative" -> 0
    label_map = {"positive": 1, "negative": 0}
    if df["sentiment"].dtype == object:
        df["label"] = df["sentiment"].map(label_map)
    else:
        df["label"] = df["sentiment"]

    # Metin sütununu belirle
    text_column = "sentence" if "sentence" in df.columns else df.columns[0]
    df["text"] = df[text_column].astype(str)

    # Boş metinleri temizle
    df = df[df["text"].str.strip().str.len() > 0].reset_index(drop=True)

    # Sınıfları dengele (undersampling)
    df_pos = df[df["label"] == 1]
    df_neg = df[df["label"] == 0]
    print(f"\nDengeleme öncesi - Pozitif: {len(df_pos)}, Negatif: {len(df_neg)}")

    train_per_class = TRAIN_SIZE // 2  # Her sınıftan 5000
    test_per_class = TEST_SIZE // 2    # Her sınıftan 1000

    df_pos = df_pos.sample(frac=1, random_state=42).reset_index(drop=True)
    df_neg = df_neg.sample(frac=1, random_state=42).reset_index(drop=True)

    # Eğitim ve test ayrımı (dengeli)
    train_df = pd.concat([
        df_pos.head(train_per_class),
        df_neg.head(train_per_class)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    test_df = pd.concat([
        df_pos.iloc[train_per_class:train_per_class + test_per_class],
        df_neg.iloc[train_per_class:train_per_class + test_per_class]
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\nEğitim seti boyutu: {len(train_df)}")
    print(f"Test seti boyutu: {len(test_df)}")
    print(f"\nEğitim seti etiket dağılımı:")
    print(train_df["label"].value_counts())
    print(f"\nTest seti etiket dağılımı:")
    print(test_df["label"].value_counts())

    # Veri seti istatistikleri grafiği
    plot_data_statistics(train_df, test_df)

    return train_df, test_df


def plot_data_statistics(train_df, test_df):
    """Veri seti istatistiklerinin görselleştirilmesi."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Etiket dağılımı
    labels = ["Negatif", "Pozitif"]
    train_counts = [
        len(train_df[train_df["label"] == 0]),
        len(train_df[train_df["label"] == 1]),
    ]
    test_counts = [
        len(test_df[test_df["label"] == 0]),
        len(test_df[test_df["label"] == 1]),
    ]

    x = np.arange(len(labels))
    width = 0.35
    axes[0].bar(x - width / 2, train_counts, width, label="Eğitim", color="#4C72B0")
    axes[0].bar(x + width / 2, test_counts, width, label="Test", color="#DD8452")
    axes[0].set_xlabel("Duygu")
    axes[0].set_ylabel("Sayı")
    axes[0].set_title("Etiket Dağılımı")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].legend()

    # 2. Metin uzunluğu dağılımı (kelime sayısı)
    train_lengths = train_df["text"].apply(lambda x: len(x.split()))
    test_lengths = test_df["text"].apply(lambda x: len(x.split()))
    axes[1].hist(train_lengths, bins=50, alpha=0.7, label="Eğitim", color="#4C72B0")
    axes[1].hist(test_lengths, bins=50, alpha=0.7, label="Test", color="#DD8452")
    axes[1].set_xlabel("Kelime Sayısı")
    axes[1].set_ylabel("Frekans")
    axes[1].set_title("Metin Uzunluğu Dağılımı")
    axes[1].legend()

    # 3. Metin uzunluğu kutu grafiği (etiketlere göre)
    train_df_copy = train_df.copy()
    train_df_copy["kelime_sayisi"] = train_lengths
    train_df_copy["duygu"] = train_df_copy["label"].map({0: "Negatif", 1: "Pozitif"})
    sns.boxplot(data=train_df_copy, x="duygu", y="kelime_sayisi", ax=axes[2],
                palette=["#DD8452", "#4C72B0"])
    axes[2].set_xlabel("Duygu")
    axes[2].set_ylabel("Kelime Sayısı")
    axes[2].set_title("Duyguya Göre Metin Uzunluğu")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "veri_seti_istatistikleri.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Veri seti istatistikleri grafiği kaydedildi.")


# ============================================================
# 2. Tokenization (Her model için ayrı tokenizer)
# ============================================================
class SentimentDataset(torch.utils.data.Dataset):
    """Duygu analizi veri seti sınıfı."""

    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ============================================================
# 3. Metrik Hesaplama
# ============================================================
def compute_metrics(eval_pred):
    """Değerlendirme metriklerini hesaplar."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    acc = accuracy_score(labels, predictions)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


# ============================================================
# 4. Model Eğitimi
# ============================================================
def train_model(model_name, model_path, train_df, test_df):
    """Tek bir modeli eğitir ve sonuçlarını döndürür."""
    print(f"\n{'=' * 60}")
    print(f"MODEL EĞİTİMİ: {model_name}")
    print(f"Model yolu: {model_path}")
    print(f"{'=' * 60}")

    # Tokenizer yükle
    print(f"Tokenizer yükleniyor: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Veri setlerini oluştur
    print("Veri setleri tokenize ediliyor...")
    train_dataset = SentimentDataset(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        tokenizer,
    )
    test_dataset = SentimentDataset(
        test_df["text"].tolist(),
        test_df["label"].tolist(),
        tokenizer,
    )

    # Model yükle
    print(f"Model yükleniyor: {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )

    # Eğitim ayarları
    output_dir = os.path.join(RESULTS_DIR, f"checkpoints_{model_name.lower().replace('-', '_')}")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        learning_rate=LEARNING_RATE,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),  # GPU varsa mixed precision
        report_to="none",  # wandb vb. kapalı
        seed=42,
    )

    # Trainer oluştur
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Eğitimi başlat
    print(f"\nEğitim başlıyor...")
    start_time = time.time()
    train_result = trainer.train()
    training_time = time.time() - start_time
    print(f"Eğitim süresi: {training_time:.1f} saniye ({training_time / 60:.1f} dakika)")

    # Değerlendirme
    print(f"\nDeğerlendirme yapılıyor...")
    eval_result = trainer.evaluate()
    print(f"Değerlendirme sonuçları: {eval_result}")

    # Tahminler
    predictions_output = trainer.predict(test_dataset)
    predictions = np.argmax(predictions_output.predictions, axis=-1)
    probabilities = torch.softmax(
        torch.tensor(predictions_output.predictions), dim=-1
    ).numpy()

    # Detaylı sınıflandırma raporu
    print(f"\n--- {model_name} Sınıflandırma Raporu ---")
    report = classification_report(
        test_df["label"].tolist(),
        predictions,
        target_names=["Negatif", "Pozitif"],
    )
    print(report)

    # Sonuçları topla
    results = {
        "model_name": model_name,
        "accuracy": eval_result["eval_accuracy"],
        "precision": eval_result["eval_precision"],
        "recall": eval_result["eval_recall"],
        "f1": eval_result["eval_f1"],
        "training_time": training_time,
        "predictions": predictions,
        "probabilities": probabilities,
        "true_labels": test_df["label"].tolist(),
        "train_history": train_result,
    }

    # Checkpoint'leri temizle (disk alanı için)
    import shutil
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    return results


# ============================================================
# 5. Görselleştirme
# ============================================================
def plot_confusion_matrices(all_results):
    """Her model için confusion matrix çizer."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    labels = ["Negatif", "Pozitif"]

    for idx, result in enumerate(all_results):
        cm = confusion_matrix(result["true_labels"], result["predictions"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=axes[idx],
        )
        axes[idx].set_title(f'{result["model_name"]}\n(Accuracy: {result["accuracy"]:.4f})')
        axes[idx].set_ylabel("Gerçek Değer")
        axes[idx].set_xlabel("Tahmin")

    plt.suptitle("Confusion Matrix Karşılaştırması", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrices.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Confusion matrix grafikleri kaydedildi.")


def plot_metrics_comparison(all_results):
    """Model metriklerini karşılaştırma grafiği."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    model_names = [r["model_name"] for r in all_results]
    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    # 1. Metrik karşılaştırma bar chart
    x = np.arange(len(model_names))
    width = 0.2

    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        values = [r[metric] for r in all_results]
        bars = axes[0].bar(x + i * width, values, width, label=label, color=color)
        # Değerleri barların üzerine yaz
        for bar, val in zip(bars, values):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    axes[0].set_xlabel("Model")
    axes[0].set_ylabel("Skor")
    axes[0].set_title("Model Performans Karşılaştırması")
    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels(model_names)
    axes[0].legend()
    axes[0].set_ylim(0, 1.1)

    # 2. Eğitim süresi karşılaştırma
    times = [r["training_time"] for r in all_results]
    bars = axes[1].bar(model_names, times, color=colors[:3])
    for bar, val in zip(bars, times):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 1,
            f"{val:.0f}s",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    axes[1].set_xlabel("Model")
    axes[1].set_ylabel("Süre (saniye)")
    axes[1].set_title("Eğitim Süresi Karşılaştırması")

    plt.suptitle("Model Karşılaştırma Sonuçları", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "metrik_karsilastirma.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Metrik karşılaştırma grafikleri kaydedildi.")


def plot_roc_curves(all_results):
    """Her model için ROC eğrilerini çizer."""
    plt.figure(figsize=(10, 8))
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for idx, result in enumerate(all_results):
        true_labels = result["true_labels"]
        probs = result["probabilities"][:, 1]  # Pozitif sınıf olasılıkları
        fpr, tpr, _ = roc_curve(true_labels, probs)
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            color=colors[idx],
            lw=2,
            label=f'{result["model_name"]} (AUC = {roc_auc:.4f})',
        )

    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Rastgele (AUC = 0.5)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (Yanlış Pozitif Oranı)")
    plt.ylabel("True Positive Rate (Doğru Pozitif Oranı)")
    plt.title("ROC Eğrisi Karşılaştırması", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("ROC eğrisi grafikleri kaydedildi.")


def create_summary_table(all_results):
    """Özet tablo oluşturur ve kaydeder."""
    summary_data = []
    for r in all_results:
        summary_data.append({
            "Model": r["model_name"],
            "Accuracy": f'{r["accuracy"]:.4f}',
            "Precision": f'{r["precision"]:.4f}',
            "Recall": f'{r["recall"]:.4f}',
            "F1-Score": f'{r["f1"]:.4f}',
            "Eğitim Süresi (s)": f'{r["training_time"]:.1f}',
        })

    summary_df = pd.DataFrame(summary_data)
    print("\n" + "=" * 60)
    print("ÖZET SONUÇ TABLOSU")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    # CSV olarak kaydet
    summary_df.to_csv(os.path.join(RESULTS_DIR, "sonuc_tablosu.csv"), index=False, encoding="utf-8-sig")
    print("\nSonuç tablosu kaydedildi: results/sonuc_tablosu.csv")

    return summary_df


def print_final_analysis(all_results):
    """Son analiz ve yorumları yazdırır."""
    print("\n" + "=" * 60)
    print("SONUÇ ANALİZİ VE YORUMLAR")
    print("=" * 60)

    # En iyi model
    best_f1 = max(all_results, key=lambda x: x["f1"])
    best_acc = max(all_results, key=lambda x: x["accuracy"])
    fastest = min(all_results, key=lambda x: x["training_time"])

    print(f"\n🏆 En yüksek F1-Score: {best_f1['model_name']} ({best_f1['f1']:.4f})")
    print(f"🎯 En yüksek Accuracy: {best_acc['model_name']} ({best_acc['accuracy']:.4f})")
    print(f"⚡ En hızlı eğitim: {fastest['model_name']} ({fastest['training_time']:.1f} saniye)")

    print("\n--- Model Bazında Yorumlar ---")
    for r in all_results:
        print(f"\n📊 {r['model_name']}:")
        print(f"   Accuracy: {r['accuracy']:.4f} | F1: {r['f1']:.4f}")
        print(f"   Precision: {r['precision']:.4f} | Recall: {r['recall']:.4f}")
        print(f"   Eğitim süresi: {r['training_time']:.1f} saniye")


# ============================================================
# 6. Ana Çalıştırma
# ============================================================
def main():
    """Ana fonksiyon - tüm süreci yönetir."""
    print("=" * 60)
    print("COME 448 - DUYGU ANALİZİ PROJESİ")
    print("3 Farklı Sınıflandırıcı Karşılaştırması")
    print("=" * 60)
    print(f"PyTorch sürümü: {torch.__version__}")
    print(f"CUDA kullanılabilir: {torch.cuda.is_available()}")

    total_start = time.time()

    # 1. Veri yükleme
    train_df, test_df = load_and_prepare_data()

    # 2. Tüm modelleri eğit
    all_results = []
    for model_name, model_path in MODELS.items():
        try:
            result = train_model(model_name, model_path, train_df, test_df)
            all_results.append(result)
            # GPU belleğini temizle
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            print(f"\n❌ HATA: {model_name} eğitimi başarısız oldu!")
            print(f"   Hata mesajı: {str(e)}")
            continue

    if len(all_results) == 0:
        print("\n❌ Hiçbir model başarıyla eğitilemedi!")
        return

    # 3. Görselleştirme
    print("\n" + "=" * 60)
    print("GÖRSELLEŞTİRME")
    print("=" * 60)
    plot_confusion_matrices(all_results)
    plot_metrics_comparison(all_results)
    plot_roc_curves(all_results)

    # 4. Özet tablo
    create_summary_table(all_results)

    # 5. Son analiz
    print_final_analysis(all_results)

    total_time = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"Toplam çalışma süresi: {total_time:.1f} saniye ({total_time / 60:.1f} dakika)")
    print(f"Sonuçlar kaydedildi: {RESULTS_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
