# Taşkın Tahmini için Zaman Serisi ve Görüntü Özelliklerini Kullanan Makine Öğrenimi Modeli

Bu proje, zaman serisi halindeki yağış verileri ile uydu görüntülerinden elde edilen özellikleri kullanarak taşkın olaylarını (veya benzeri bir etiketi) tahmin etmeyi amaçlamaktadır. Proje kapsamında veri ön işleme, özellik mühendisliği, hiperparametre optimizasyonu ve çeşitli makine öğrenimi modellerinin (XGBoost, LightGBM, CatBoost) eğitimi ve birleştirilmesi (ensembling) adımları gerçekleştirilmiştir.

## 🎯 Amaç

Temel amaç, verilen `event_id`'lere bağlı zaman serisi yağış kayıtları ve ilişkili uydu görüntüsü bantlarından elde edilen bilgilerle, her bir zaman adımı için bir hedef değişkeni (label) tahmin etmektir. Veri setindeki etiket dengesizliği dikkate alınarak modeller geliştirilmiştir.

## 💾 Veri Seti

Bu projede üç ana veri kaynağı kullanılmıştır:
* `Train.csv`: Eğitim için kullanılan, `event_id`, `precipitation` (yağış) ve `label` (hedef değişken) sütunlarını içerir.
* `Test.csv`: Tahmin yapılacak test verilerini içerir, `event_id` ve `precipitation` sütunlarını içerir.
* `composite_images.npz`: Her bir `event_id`'nin temel kısmına karşılık gelen çok bantlı (B2, B3, B4, B8, B11) uydu görüntülerini ve eğim (slope) verisini içerir.

## 🛠️ Metodoloji

Projenin ana adımları aşağıdaki gibidir:

1.  **Kütüphane Kurulumu ve Ortam Hazırlığı**:
    * Gerekli Python kütüphaneleri (pandas, numpy, scikit-learn, xgboost, lightgbm, catboost, optuna, shap, torch vb.) kurulur.
    * GPU kullanımı kontrol edilir ve ayarlanır.

2.  **Veri Yükleme ve İlk Keşif (EDA)**:
    * CSV ve NPZ formatındaki veriler yüklenir.
    * Veri boyutları, ilk birkaç satırı ve etiket dağılımı incelenir.
    * Eksik değer kontrolü yapılır. Etiket dağılımının oldukça dengesiz olduğu gözlemlenmiştir.

3.  **Görüntü Özelliklerinin Çıkarılması**:
    * `composite_images.npz` dosyasından yüklenen uydu görüntüleri kullanılarak her bir olay için aşağıdaki indeksler ve istatistikler hesaplanır:
        * NDVI (Normalized Difference Vegetation Index) Ortalama ve Standart Sapması
        * NDWI (Normalized Difference Water Index) Ortalama ve Standart Sapması
        * Eğim (Slope) Ortalama ve Standart Sapması
    * Bu özellikler ana eğitim ve test veri çerçevelerine `event_id` üzerinden birleştirilir.

4.  **Zaman Serisi Özellik Mühendisliği**:
    * Yağış verileri (`precipitation`) kullanılarak çeşitli zaman serisi özellikleri türetilir:
        * Gün indeksi (`day`)
        * Farklı pencere boyutları (3, 7, 14, 30 gün) için hareketli ortalamalar (`ma_X`)
        * Farklı pencere boyutları için kümülatif toplamlar (`cum_X`)
        * Hareketli ortalamalardaki değişim (trend, `trend_X`)
        * Şiddetli yağış (`heavy_rain`, eşik > 20mm)
        * Ardışık şiddetli yağış gün sayısı (`consecutive_rain`)
        * Yağışın 7 günlük hareketli standart sapması (`std_precip`)
        * Logaritmik dönüşüm uygulanmış yağış (`log_precip`)
    * Oluşan NaN değerler 0 ile doldurulur.

5.  **Model Eğitimi ve Optimizasyonu**:
    * **XGBoost**:
        * Optuna kütüphanesi kullanılarak `GroupKFold` (olay ID'lerine göre gruplanmış 3 katlı çapraz doğrulama) ile hiperparametre optimizasyonu (learning_rate, max_depth, subsample vb.) yapılır. Hedef metrik `logloss`'tur.
        * Dengesiz sınıflar için `scale_pos_weight` parametresi kullanılır.
        * En iyi bulunan parametrelerle 5 katlı `GroupKFold` ile XGBoost modeli eğitilir.
    * **LightGBM (Opsiyonel)**:
        * Önceden tanımlanmış bir dizi hiperparametre ile 5 katlı `GroupKFold` kullanılarak LightGBM modeli eğitilir. `scale_pos_weight` burada da kullanılır.
    * **CatBoost (Opsiyonel)**:
        * Önceden tanımlanmış bir dizi hiperparametre ile 5 katlı `GroupKFold` kullanılarak CatBoost modeli eğitilir. `scale_pos_weight` ve `early_stopping_rounds` kullanılır.

6.  **Ensemble (Model Birleştirme)**:
    * En iyi hiperparametrelerle eğitilmiş XGBoost, LightGBM ve CatBoost modelleri tüm eğitim verisi üzerinde yeniden eğitilir.
    * Test seti üzerindeki tahminleri, bu üç modelin tahminlerinin basit ortalaması alınarak birleştirilir.

7.  **Tahmin ve Submission**:
    * Elde edilen ensemble modelinin test seti üzerindeki tahminleri, yarışma formatına uygun bir `submission.csv` dosyası olarak kaydedilir.

## 📚 Kullanılan Kütüphaneler

* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `scikit-learn`
* `xgboost`
* `lightgbm`
* `catboost`
* `shap`
* `optuna`
* `torch` (görüntü verilerinin GPU'da işlenmesi için)
* `google.colab` (Google Colab ortamında çalıştırılıyorsa)

## 🚀 Nasıl Çalıştırılır?

1.  Notebook'u bir Google Colab ortamında veya yerel makinenizde Jupyter Notebook/Lab ile açın.
2.  Gerekli kütüphanelerin kurulu olduğundan emin olun. İlk hücredeki `!pip install ...` komutları bu işlemi gerçekleştirecektir.
3.  Veri dosyalarının (`Train.csv`, `Test.csv`, `composite_images.npz`) yollarını notebook içerisindeki ilgili hücrede (`VERI YOLLARI ve OKUMA` bölümü) kendi dosya sisteminize göre güncelleyin. Google Drive kullanılıyorsa, Drive bağlantısının doğru yapıldığından emin olun.
4.  Hücreleri sırayla çalıştırın.
5.  Sonuç olarak `submission_ensemble.csv` dosyası belirttiğiniz yola kaydedilecektir.

## 📈 Sonuçlar

Bu proje sonucunda, farklı özellik mühendisliği teknikleri ve makine öğrenimi modelleri denenerek bir ensemble model oluşturulmuştur. Modelin performansı çapraz doğrulama sırasında `logloss` metriği ile değerlendirilmiştir. XGBoost modeli için Optuna ile hiperparametre optimizasyonu yapılmış ve en iyi parametreler belirlenmiştir. Final tahminler, XGBoost, LightGBM ve CatBoost modellerinin ortalaması alınarak elde edilmiştir.

## 💡 Gelecek Çalışmalar

* Daha gelişmiş zaman serisi özellikleri (örneğin, otoregresif özellikler, Fourier dönüşümleri) eklenebilir.
* Uydu görüntülerinden özellik çıkarmak için derin öğrenme tabanlı modeller (CNN'ler) kullanılabilir.
* Farklı ensemble teknikleri (örneğin, stacking) denenebilir.
* Optuna ile LightGBM ve CatBoost için de kapsamlı hiperparametre optimizasyonu yapılabilir.# taskin-tahmin-modeli
Zaman serisi ve görüntü özellikleri kullanılarak geliştirilen taşkın tahmini modeli
