import pandas as pd
import datetime
import time
import flo_config
from flo_config import df_quantiles
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering


def data_preprocess(data_path):
    """

    :param data_path: verinin bulunduğu path
    :return:
    """
    # read data
    print("veri seti okunuyor ....")
    df_ = pd.read_csv(data_path, parse_dates=["first_order_date",
                                              "last_order_date",
                                              "last_order_date_online",
                                              "last_order_date_offline"])
    df = df_.copy()

    # feature creating
    print("tenure ve receny değişkenleri oluşturuluyor ....")
    today = max(df["last_order_date_online"]) + datetime.timedelta(days=1)
    time.sleep(2)
    print(f"analiz tarihi {today} olarak belirlendi ....")
    df["NEW_recency"] = df["last_order_date_online"].apply(lambda x: (today - x).days)
    df["NEW_tenure"] = df["first_order_date"].apply(lambda x: (today - x).days)
    print("tenure ve recency değişkenleri oluşturuldu ....")
    time.sleep(2)
    print("distinct category number değişkeni oluşturuluyor ....")
    df["NEW_distinct_cat_number"] = df["interested_in_categories_12"].apply(lambda x: len(x))
    print("distinct category number değişkeni oluşturuldu ....")

    # determining variablesfor further investigations
    time.sleep(1)
    print("kullanılcacak değişkenler ayrılıyor ....")
    df = df[["master_id",
             "first_order_date",
             "last_order_date",
             "last_order_date_online",
             "last_order_date_offline",
             "order_num_total_ever_online",
             "order_num_total_ever_offline",
             "customer_value_total_ever_offline",
             "customer_value_total_ever_online",
             "NEW_recency",
             "NEW_tenure",
             "NEW_distinct_cat_number"]]
    print("kullanılcacak değişkenler ayrıldı ....")
    print("******TÜM İŞLEMLER BAŞARILI BİR ŞEKİLDE TAMAMLANDI******")

    return df


def knn_model_preprocess(dataframe, columns, scale=False, optimum_kume_number=None):
    """

    :param dataframe:
    :param scale: scaling işlemi yapılsın mı, default : False
    :param optimum_kume_number: belirlenmiş optimum küme sayısı
    :return:
    """

    columns = [col for col in dataframe.columns if
               ("master_id" not in col and dataframe[col].dtypes == "datetime64[ns]")]
    dataframe = dataframe.drop(columns, axis=1)
    if scale:
        print("scaling işlemi yapılıyor ....")
        scaler = MinMaxScaler((0, 1))
        scaler.fit(dataframe.iloc[:, 1:])
        dataframe.iloc[:, 1:] = scaler.transform(dataframe.iloc[:, 1:])
        time.sleep(1)
        print("scaling işlemi TAMAMLANDI ....")

        if optimum_kume_number == None:
            time.sleep(1)
            print("optimum küme sayısı grafiği olşturuluyor ....")
            time.sleep(1)
            kmeans = KMeans()
            elbow = KElbowVisualizer(kmeans, k=(2, 20))
            elbow.fit(dataframe.iloc[:, 1:])
            elbow.show()
            print(f"optimumu küme sayısı {elbow.elbow_value_} olarak belirlendi")
            time.sleep(1)
            print("optimum degere göre clusterlar oluşturuluyor ....")
            time.sleep(1)
            kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(dataframe.iloc[:, 1:])
            dataframe["NEW_clusters_knn_elbow"] = kmeans.labels_
            print("******TÜM İŞLEMLER BAŞARILI BİR ŞEKİLDE TAMAMLANDI******")
            return dataframe


        else:
            print("programcı tarafından belirlenen sayıya göre clusterlar oluşturuluyor ....")
            time.sleep(1)
            kmeans = KMeans(n_clusters=optimum_kume_number).fit(dataframe.iloc[:, 1:])
            dataframe["NEW_clusters_knn_programmers"] = kmeans.labels_
            print("******TÜM İŞLEMLER BAŞARILI BİR ŞEKİLDE TAMAMLANDI******")
            return dataframe

    else:
        print("!!!! scaling işlemi yapılmadan clustering yapılamaz !!!!")




def cluster_stats(dataframe, column):
    """

    :param dataframe: inceleme yapılacak dataframe
    :param column: groupby 'a alınacak degişken
    :return:
    """
    print("clusterlar ve gözlem sayıları")
    print("#####################################")
    print(dataframe.groupby(column)[column].count(),end= "\n\n")

    print("dataframe genel istatistikler")
    print("#####################################")
    print(dataframe.describe(df_quantiles).T,end="\n\n")

    print(f'{dataframe["NEW_clusters_elbow"].nunique()} adet sınıf bulunmaktadır')
    length = dataframe["NEW_clusters_elbow"].nunique()
    for i in range(length):
        print(f"class = {i} ' ye ait özet istatistiki bilgiler...")
        print(dataframe.loc[dataframe["NEW_clusters_elbow"]==i].describe().T, end="\n\n")


def hierarchical_model_process(dataframe,cluster_num):
    hc_average = linkage(dataframe, "average")
    plt.figure(figsize=(7, 5))
    plt.title("Dendrograms")
    dend = dendrogram(hc_average,
                      truncate_mode="lastp",
                      p=10,
                      show_contracted=True,
                      leaf_font_size=10)
    plt.axhline(y=1.3, color='r', linestyle='--')
    plt.axhline(y=1.2, color='b', linestyle='--')
    plt.show()

    cluster = AgglomerativeClustering(n_clusters=cluster_num, linkage="average")

    clusters = cluster.fit_predict(dataframe)
    dataframe["NEW_clusters_hiyerarsik"] = clusters

    return dataframe