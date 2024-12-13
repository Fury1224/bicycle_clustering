import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# 데이터 불러오기 및 초기 처리
business_df = pd.read_csv('사업체1.csv')
usage_df = pd.read_csv('이용1.csv')
population_density_df = pd.read_csv('인구밀도1.csv')


# 대여 및 반납 건수
usage_agg = usage_df.groupby('자치구')[['대여건수', '반납건수']].sum().reset_index()

# 인구 밀도
population_density_business = business_df.groupby('자치구').mean().reset_index()
population_density_ind = population_density_df.groupby('자치구').mean().reset_index()

# 사업체 종사자 인구 밀도
merged_df = usage_agg.merge(population_density_ind, on='자치구', how='left', suffixes=('_usage', '_pop_density'))
merged_df = merged_df.merge(population_density_business, on='자치구', how='left', suffixes=('', '_business'))

# 통합
merged_df.columns = ['자치구', '대여건수', '반납건수', '인구밀도_ind', '인구밀도_business']

# 비어있는 데이터는 드롭
merged_df.dropna(inplace=True)

######################################################################################################################################################
# 데이터 스케일링

# StandardScaler와 MinMaxScaler를 통해 데이터 스케일링
scalers = {'StandardScaler': StandardScaler(), 'MinMaxScaler': MinMaxScaler()}
scaled_data = {}

for scaler_name, scaler in scalers.items():
    scaled_values = scaler.fit_transform(merged_df[['대여건수', '반납건수', '인구밀도_ind', '인구밀도_business']])
    scaled_data[scaler_name] = pd.DataFrame(scaled_values, columns=['대여건수', '반납건수', '인구밀도_ind', '인구밀도_business'])
    scaled_data[scaler_name]['자치구'] = merged_df['자치구']


# StandardScaler로 스케일링 한 결과 차트화
fig, axes = plt.subplots(3, 1, figsize=(12, 15))

axes[0].bar(scaled_data['StandardScaler']['자치구'], scaled_data['StandardScaler']['대여건수'], label='대여건수')
axes[0].bar(scaled_data['StandardScaler']['자치구'], scaled_data['StandardScaler']['반납건수'], bottom=scaled_data['StandardScaler']['대여건수'], label='반납건수')
axes[0].set_title('대여 및 반납 건수 Standard Scaler')
axes[0].set_xticklabels(scaled_data['StandardScaler']['자치구'], rotation=30)
axes[0].legend()

axes[1].bar(scaled_data['StandardScaler']['자치구'], scaled_data['StandardScaler']['인구밀도_ind'], color='orange')
axes[1].set_title('인구밀도 Standard Scaler')
axes[1].set_xticklabels(scaled_data['StandardScaler']['자치구'], rotation=30)

axes[2].bar(scaled_data['StandardScaler']['자치구'], scaled_data['StandardScaler']['인구밀도_business'], color='green')
axes[2].set_title('사업체 종사자 밀도 Standard Scaler')
axes[2].set_xticklabels(scaled_data['StandardScaler']['자치구'], rotation=30)

plt.tight_layout()
plt.show()

# MinMaxScaler로 스케일링 한 결과 차트화
fig, axes = plt.subplots(3, 1, figsize=(12, 15))

axes[0].bar(scaled_data['MinMaxScaler']['자치구'], scaled_data['MinMaxScaler']['대여건수'], label='대여건수')
axes[0].bar(scaled_data['MinMaxScaler']['자치구'], scaled_data['MinMaxScaler']['반납건수'], bottom=scaled_data['MinMaxScaler']['대여건수'], label='반납건수')
axes[0].set_title('대여 및 반납 건수 MinMax Scaler')
axes[0].set_xticklabels(scaled_data['MinMaxScaler']['자치구'], rotation=30)
axes[0].legend()

axes[1].bar(scaled_data['MinMaxScaler']['자치구'], scaled_data['MinMaxScaler']['인구밀도_ind'], color='orange')
axes[1].set_title('인구밀도 MinMax Scaler')
axes[1].set_xticklabels(scaled_data['MinMaxScaler']['자치구'], rotation=30)

axes[2].bar(scaled_data['MinMaxScaler']['자치구'], scaled_data['MinMaxScaler']['인구밀도_business'], color='green')
axes[2].set_title('사업체 종사자 밀도 MinMax Scaler')
axes[2].set_xticklabels(scaled_data['MinMaxScaler']['자치구'], rotation=30)

plt.tight_layout()
plt.show()

# StandardScaler로 스케일링 한 데이터 통합
standard_scaled_merged = pd.concat([scaled_data['StandardScaler']['대여건수'],
                                    scaled_data['StandardScaler']['반납건수'],
                                    scaled_data['StandardScaler']['인구밀도_ind'],
                                    scaled_data['StandardScaler']['인구밀도_business']], axis=1)

# MinMaxScaler로 스케일링 한 데이터 통합
minmax_scaled_merged = pd.concat([scaled_data['MinMaxScaler']['대여건수'],
                                  scaled_data['MinMaxScaler']['반납건수'],
                                  scaled_data['MinMaxScaler']['인구밀도_ind'],
                                  scaled_data['MinMaxScaler']['인구밀도_business']], axis=1)

######################################################################################################################################################
# Elbow

# Elbow 방법으로 K-Means를 위한 최적의 K 구하기
inertia_standard_merged = []
inertia_minmax_merged = []
K_range = range(1, 11)

for k in K_range:
    kmeans_standard = KMeans(n_clusters=k, random_state=42).fit(standard_scaled_merged)
    kmeans_minmax = KMeans(n_clusters=k, random_state=42).fit(minmax_scaled_merged)
    inertia_standard_merged.append(kmeans_standard.inertia_)
    inertia_minmax_merged.append(kmeans_minmax.inertia_)

# 차트화
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# StandardScaler
axes[0].plot(K_range, inertia_standard_merged, 'o-', color='blue')
axes[0].set_title('Standard Scaler')
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia')
axes[0].grid(True)

# MinMaxScaler
axes[1].plot(K_range, inertia_minmax_merged, 's-', color='orange')
axes[1].set_title('MinMax Scaler')
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Inertia')
axes[1].grid(True)

plt.tight_layout()
plt.show()

# 최적의 K 값 선정 = 3
optimal_k = 3

######################################################################################################################################################
# 군집화

# 두 스케일링 된 데이터를 K-Means로 군집화
kmeans_standard_merged = KMeans(n_clusters=optimal_k, random_state=42).fit(standard_scaled_merged)
kmeans_minmax_merged = KMeans(n_clusters=optimal_k, random_state=42).fit(minmax_scaled_merged)

standard_scaled_merged['KMeans_Cluster'] = kmeans_standard_merged.labels_
minmax_scaled_merged['KMeans_Cluster'] = kmeans_minmax_merged.labels_

# 두 스케일링 된 데이터를 DBSCAN으로 군집화
dbscan_standard_merged = DBSCAN(eps=1, min_samples=3).fit(standard_scaled_merged.iloc[:, :-1])  # Exclude KMeans label column
dbscan_minmax_merged = DBSCAN(eps=0.5, min_samples=5).fit(minmax_scaled_merged.iloc[:, :-1])      # Exclude KMeans label column

standard_scaled_merged['DBSCAN_Cluster'] = dbscan_standard_merged.labels_
minmax_scaled_merged['DBSCAN_Cluster'] = dbscan_minmax_merged.labels_


# StandardScaler 차트화
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# K-Means
axes[0].scatter(standard_scaled_merged['대여건수'], standard_scaled_merged['인구밀도_ind'], 
                c=standard_scaled_merged['KMeans_Cluster'], cmap='viridis')
axes[0].set_title('StandardScaler Data')
axes[0].set_xlabel('대여건수')
axes[0].set_ylabel('인구밀도')

# DBSCAN
axes[1].scatter(standard_scaled_merged['대여건수'], standard_scaled_merged['인구밀도_ind'], 
                c=standard_scaled_merged['DBSCAN_Cluster'], cmap='viridis')
axes[1].set_title('StandardScaler Data')
axes[1].set_xlabel('대여건수')
axes[1].set_ylabel('인구밀도')

plt.tight_layout()
plt.show()

# MinMaxScaler 차트화
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# K-Means
axes[0].scatter(minmax_scaled_merged['대여건수'], minmax_scaled_merged['인구밀도_ind'], 
                c=minmax_scaled_merged['KMeans_Cluster'], cmap='viridis')
axes[0].set_title('MinMaxScaler Data')
axes[0].set_xlabel('대여건수')
axes[0].set_ylabel('인구밀도')

# DBSCAN
axes[1].scatter(minmax_scaled_merged['대여건수'], minmax_scaled_merged['인구밀도_ind'], 
                c=minmax_scaled_merged['DBSCAN_Cluster'], cmap='viridis')
axes[1].set_title('MinMaxScaler Data')
axes[1].set_xlabel('대여건수')
axes[1].set_ylabel('인구밀도')

plt.tight_layout()
plt.show()

######################################################################################################################################################
# 군집화 결과를 silhouette score로 평가

silhouette_scores = {
    'KMeans_StandardScaler': silhouette_score(standard_scaled_merged[['대여건수', '반납건수', '인구밀도_ind', '인구밀도_business']], standard_scaled_merged['KMeans_Cluster']),
    'KMeans_MinMaxScaler': silhouette_score(minmax_scaled_merged[['대여건수', '반납건수', '인구밀도_ind', '인구밀도_business']], minmax_scaled_merged['KMeans_Cluster'])
}

# DBSCAN 군집화에서 노이즈를 제외하여 평가
if len(set(standard_scaled_merged['DBSCAN_Cluster'])) > 1:
    silhouette_scores['DBSCAN_StandardScaler'] = silhouette_score(
        standard_scaled_merged[['대여건수', '반납건수', '인구밀도_ind', '인구밀도_business']], standard_scaled_merged['DBSCAN_Cluster'])
else:
    silhouette_scores['DBSCAN_StandardScaler'] = 'Not applicable (single cluster or noise)'

if len(set(minmax_scaled_merged['DBSCAN_Cluster'])) > 1:
    silhouette_scores['DBSCAN_MinMaxScaler'] = silhouette_score(
        minmax_scaled_merged[['대여건수', '반납건수', '인구밀도_ind', '인구밀도_business']], minmax_scaled_merged['DBSCAN_Cluster'])
else:
    silhouette_scores['DBSCAN_MinMaxScaler'] = 'Not applicable (single cluster or noise)'

# 최적의 군집화 찾기
best_clustering = max((key for key in silhouette_scores if isinstance(silhouette_scores[key], (int, float))), 
                      key=lambda k: silhouette_scores[k])
best_score = silhouette_scores[best_clustering]

print(silhouette_scores)

# 차트화
algorithms = list(silhouette_scores.keys())
scores = list(silhouette_scores.values())

plt.figure(figsize=(10, 6))
plt.bar(algorithms, scores, color=['skyblue', 'orange', 'green', 'purple'])
plt.ylabel('Silhouette Score')
plt.ylim(0, 1)  # Silhouette scores는 -1과 1사이 값
plt.show()

######################################################################################################################################################
# 상위 3개 자치구 선택

# 자치구 열을 포함시켜 데이터 병합
standard_scaled_merged = pd.concat([scaled_data['StandardScaler'][['대여건수', '반납건수', '인구밀도_ind', '인구밀도_business', '자치구']],
                                    standard_scaled_merged[['KMeans_Cluster', 'DBSCAN_Cluster']]], axis=1)

minmax_scaled_merged = pd.concat([scaled_data['MinMaxScaler'][['대여건수', '반납건수', '인구밀도_ind', '인구밀도_business', '자치구']],
                                  minmax_scaled_merged[['KMeans_Cluster', 'DBSCAN_Cluster']]], axis=1)

# 최적의 알고리즘은 MinMaxScaler를 통한 K-Means
top_clusters = minmax_scaled_merged['KMeans_Cluster'].unique()
top_cluster_label = minmax_scaled_merged[minmax_scaled_merged['KMeans_Cluster'].isin(top_clusters)]

# 각 특성의 평균값 계산
cluster_means = top_cluster_label.groupby('KMeans_Cluster')[['대여건수', '반납건수', '인구밀도_ind', '인구밀도_business']].mean()

# 각 클러스터에 고유 특성을 할당하여 분류
selected_features = []
dominant_features_unique = {}

for cluster in cluster_means.index:

    sorted_features = cluster_means.loc[cluster].sort_values(ascending=False).index.tolist()

    for feature in sorted_features:
        if feature not in selected_features:
            dominant_features_unique[cluster] = feature
            selected_features.append(feature)
            break

# 주요 특성을 기준으로 상위 3개 자치구 선택
top_locations_by_cluster_unique = {}

for cluster, feature in dominant_features_unique.items():
    top_3 = top_cluster_label[top_cluster_label['KMeans_Cluster'] == cluster].sort_values(by=feature, ascending=False).head(3)
    top_locations_by_cluster_unique[cluster] = {
        'dominant_feature': feature,
        'top_locations': top_3['자치구'].values
    }
    
######################################################################################################################################################
# 결과를 표 형태로 출력

# 최적의 알고리즘을 통한 군집화 결과
final_dataset = minmax_scaled_merged.copy()

# 소수점 3자리로 제한
final_dataset = final_dataset.round(3)

# 시각화
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')  # Hide the axis

# 최종 데이터셋 출력
table_data = final_dataset.head(25)
table = ax.table(cellText=table_data.values, colLabels=table_data.columns, cellLoc='center', loc='center')

table.auto_set_font_size(False)
table.set_fontsize(14)
table.scale(0.8, 1.8)  # 표 크기

for col in range(len(table_data.columns)):
    cell = table[(0, col)]
    cell.set_facecolor('#A9A9A9')
    cell.set_text_props(fontname='Malgun Gothic', weight='bold')

for key, cell in table.get_celld().items():
    cell.set_text_props(fontname='Malgun Gothic')

plt.show()


# 각 클러스터의 주요 특성
dominant_features_unique = {0: '종사자 밀도', 1: '반납건수', 2: '인구밀도'}

# 주요 특성에 따라 데이터 정렬
sorted_by_dominant_feature = {}
for cluster, dominant_feature in dominant_features_unique.items():
    sorted_data = final_dataset[final_dataset['Cluster'] == cluster].sort_values(by=dominant_feature, ascending=False)
    sorted_by_dominant_feature[cluster] = sorted_data[['자치구', '대여건수', '반납건수', '인구밀도', '종사자 밀도', 'Cluster']]

# 시각화
for cluster, data in sorted_by_dominant_feature.items():
    
    data = data.round(3)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
 
    table = ax.table(cellText=data.values, colLabels=data.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(0.9, 1.8)
    for col in range(len(data.columns)):
        cell = table[(0, col)]
        cell.set_facecolor('#A9A9A9')
        cell.set_text_props(fontname='Malgun Gothic', weight='bold')

    for key, cell in table.get_celld().items():
        cell.set_text_props(fontname='Malgun Gothic')

    plt.show()

######################################################################################################################################################
# Folium을 통한 시각화 

final_dataset = pd.DataFrame({
    '자치구': final_dataset['자치구'],
    '대여건수': final_dataset['대여건수'],
    '반납건수': final_dataset['반납건수'],
    '인구밀도': final_dataset['인구밀도_ind'],
    '종사자 밀도': final_dataset['인구밀도_business'],
    'Cluster': final_dataset['KMeans_Cluster']
})

# 각 자치구의 좌표값
coordinate = {'강남구' : [37.49656682008492, 127.0629441780405],
              '강동구' : [37.554354388064816, 127.14550350616773],
              '강북구' : [37.64215650436188, 127.01546970895886],
              '강서구' : [37.561669187734466, 126.80802800793967],
              '관악구' : [37.4824364431134, 126.91438663627962],
              '광진구' : [37.54546775781386, 127.08520507120973],
              '구로구' : [37.49495927652307, 126.85596663810817],
              '금천구' : [37.46096350316589, 126.90073696989165],
              '노원구' : [37.63874951368715, 127.07551406355162],
              '도봉구' : [37.666008556379964, 127.03672565464433],
              '동대문구' : [37.5822229896885, 127.05487550887021],
              '동작구' : [37.499012818031865, 126.95158666326857],
              '마포구' : [37.558204615797905, 126.90839700534154],
              '서대문구' : [37.57621865839605, 126.93538886333876],
              '서초구' : [37.473201860592496, 127.0309009686166],
              '성동구' : [37.558075756229144, 127.03889779960014],
              '성북구' : [37.605619623611716, 127.01759721103198],
              '송파구' : [37.505173184175824, 127.11516151899829],
              '양천구' : [37.524764417262325, 126.85543890979527],
              '영등포구' : [37.52199863826605, 126.90978725088188],
              '용산구' : [37.53129445816194, 126.97989628707137],
              '은평구' : [37.61896251050307, 126.92714212010071],
              '종로구' : [37.594836146187035, 126.97737869470801],
              '중구' : [37.55991122931274, 126.99606593036329],
              '중랑구' : [37.59415267439279, 127.09309234858445]}


# 중심 좌표 설정
latitude = 37.49656682008492
longitude = 127.0629441780405

# 클러스터 순회
for i in range(3):
    # 각 클러스터의 자치구 목록
    top_locations = top_locations_by_cluster_unique[i]['top_locations']
    
    # 지도 생성 : 첫 번째 자치구를 기준으로 지도 중심 설정
    first_location = coordinate[top_locations[0]]
    map_ = folium.Map(location=first_location, zoom_start=11)
    
    # 자치구 마커 추가
    for location in top_locations:
        lat, lon = coordinate[location]
        
        # 툴팁 설정
        if location == '동작구':
            tooltip = "<b style='font-size: 16px;'>한강을 끼고 용산과 종로, 강남역, 여의도의 중간 지점에 위치한 자치구로</b><br><b style='font-size: 16px;'>여러 주요 업무 지구와 인접해 있어 다양한 이동 패턴 지원이 가능하다</b>"
        elif location == '관악구':
            tooltip = "<b style='font-size: 16px;'>직장을 찾아 비수도권에서 수도권으로 이동하는</b><br> <b style='font-size: 16px;'>25~29세가 가장 많은 자치구이다</b>"
        elif location == '강서구':
            tooltip = "<b style='font-size: 16px;'>대여와 반납 건수가 모두 높은 자치구로 이동의 수요가 많아</b><br><b style='font-size: 16px;'>자전거 배치의 효율성을 높이고 지속적인 회전율을 높일 수 있다</b><br><b style='font-size: 16px;'>또한 마곡지구의 개발로 오피스텔과 중대형 기업의 입주 등으로</b><br><b style='font-size: 16px;'>유입 인구의 수요를 충족할 수 있다</b>"
        elif location == '강남구':
            tooltip = "<b style='font-size: 16px;'>강남 8학군과 대치동 학원가가 위치한 지역으로</b><br><b style='font-size: 16px;'>학생들의 통학 수요를 충족하고 수서역을 통해</b><br><b style='font-size: 16px;'>출퇴근하는 직장인들에게도 적절한 교통수단이 될 수 있다</b>"
        else:
            tooltip = location  # 기본 툴팁
            
        
        folium.Circle(
            location=[lat, lon],
            radius=1500, # 원 크기
            color='red', # 원 선 색상
            fill_color='yellow', # 원 내부 색상
            popup=location,
            tooltip=tooltip
        ).add_to(map_)
        
        folium.Marker(
        location=[lat, lon],
        icon=folium.DivIcon(
            html=f'<div style="font-size: 12px; color: black; font-weight: bold; white-space: nowrap; position: absolute; left: -12px;">{location}</div>'
            )
        ).add_to(map_)
        
        
    # 지도 HTML 파일로 저장
    map_.save(f"map{i}.html")
    
