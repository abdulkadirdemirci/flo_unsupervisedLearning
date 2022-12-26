import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
from yellowbrick.cluster import KElbowVisualizer
import flo_config
import flo_util
from flo_config import data_path
from flo_config import df_quantiles
from flo_util import data_preprocess
from flo_util import model_preprocess
from flo_util import cluster_stats
from flo_util import hierarchical_model_process

pd.set_option("display.expand_frame_repr", False)
# iş problemi
"""
FLO müşterilerini segmentlere ayırıp bu segmentlere göre
pazarlama stratejileri belirlemek istiyor. Buna yönelik
olarak müşterilerin davranışları tanımlanacak ve bu
davranışlardaki öbeklenmelere göre gruplar oluşturulacak.
"""

# verseti hikayesi
"""
Veri seti Flo’dan son alışverişlerini 2020 - 2021 yıllarında 
OmniChannel (hem online hem offline alışveriş yapan)
olarak yapan müşterilerin geçmiş alışveriş davranışlarından 
elde edilen bilgilerden oluşmaktadır.

customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi
last_order_date_offline : Müşterinin offline platformda yaptığı son alışveriş tarihi
last_order_date_online : Müşterinin online platformda yaptığı son alışveriş tarihi
last_order_channel : En son alışverişin yapıldığı kanal
first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
last_order_date : Müşterinin yaptığı son alışveriş tarihi
order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile)
master_id : Eşsiz müşteri numarası
"""

# Görev 1: Veriyi Hazırlama
"""
Adım 1: flo_data_20K.csv verisini okutunuz.
Adım 2: Müşterileri segmentlerken kullanacağınız değişkenleri seçiniz.
Not: Tenure (Müşterinin yaşı), Recency (en son kaç gün önce alışveriş yaptığı) gibi yeni değişkenler oluşturabilirsiniz.
"""

# order_channel, last_order_channel,interested_in_categories_12
# değişkenlerini kullanamayacagım malesef tarih ve numerik degerler ifade eden
# değişkenlerle ilerleyecegim

df = data_preprocess(data_path)


# Görev 2: K-Means ile Müşteri Segmentasyonu
"""
Adım 1: Değişkenleri standartlaştırınız.
Adım 2: Optimum küme sayısını belirleyiniz.
Adım 3: Modelinizi oluşturunuz ve müşterilerinizi segmentleyiniz.
Adım 4: Herbir segmenti istatistiksel olarak inceleyeniz.
"""

df = knn_model_preprocess(df, df.columns, scale=True)

cluster_stats(df, "NEW_clusters_elbow")

# Görev 3: Hierarchical Clustering ile Müşteri Segmentasyonu
"""
Adım 1: Görev 2'de standırlaştırdığınız dataframe'i kullanarak optimum küme sayısını belirleyiniz.
Adım 2: Modelinizi oluşturunuz ve müşterileriniz segmentleyiniz.
Adım 3: Her bir segmenti istatistiksel olarak inceleyeniz.
"""

df = hierarchical_model_process(df.iloc[:, 1:], 6)

cluster_stats(df, "NEW_clusters_hiyerarsik")
