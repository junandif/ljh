#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


# In[2]:


import geopandas as gpd
import pandas as pd

merged_gdf = gpd.read_file('D:/analysis/청주읍면동/cheongju1.shp', encoding='euc-kr')
  
print(merged_gdf.head())


# In[3]:


cheongju_emd = merged_gdf
cheongju_emd.explore()


# In[4]:


import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

df = pd.read_excel('D:/analysis/2017~2019.xlsx')
df1 = df.dropna(subset=['좌표정보(X)', '좌표정보(Y)'])
point = gpd.GeoDataFrame(df1, geometry=gpd.points_from_xy(df1['좌표정보(X)'], df1['좌표정보(Y)'], crs='epsg:5174'))


point_utm =point.to_crs(5179)
cheongju_emd=cheongju_emd.to_crs(5179)

fig, ax = plt.subplots(figsize=(10, 10))
cheongju_emd.boundary.plot(ax=ax, color='black', linewidth=0.1)
point_utm.plot(ax = ax, color = 'blue', markersize = 5)
plt.show()


# In[5]:


result_gdf = gpd.sjoin(cheongju_emd, point_utm, how='left', op='contains')
count_per_region = result_gdf.groupby('EMD_CD').size().reset_index(name='점포개수')
result_gdf2017 = merged_gdf.merge(count_per_region, how='left', on='EMD_CD')


# In[6]:


result_gdf2017.explore(column='점포개수')


# In[8]:


import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

df = pd.read_excel('D:/analysis/2018-2020.xlsx')
df1 = df.dropna(subset=['좌표정보(X)', '좌표정보(Y)'])
point = gpd.GeoDataFrame(df1, geometry=gpd.points_from_xy(df1['좌표정보(X)'], df1['좌표정보(Y)'], crs='epsg:5174'))


point_utm =point.to_crs(5179)
cheongju_emd=cheongju_emd.to_crs(5179)

fig, ax = plt.subplots(figsize=(10, 10))
cheongju_emd.boundary.plot(ax=ax, color='black', linewidth=0.1)
point_utm.plot(ax = ax, color = 'blue', markersize = 5)
plt.show()


# In[9]:


result_gdf = gpd.sjoin(cheongju_emd, point_utm, how='left', op='contains')
count_per_region = result_gdf.groupby('EMD_CD').size().reset_index(name='점포개수')
result_gdf2018 = merged_gdf.merge(count_per_region, how='left', on='EMD_CD')


# In[10]:


result_gdf2018.explore(column='점포개수')


# In[11]:


df = pd.read_excel('D:/analysis/인허가데이터(1)/2019-2021.xlsx')
df1 = df.dropna(subset=['좌표정보(X)', '좌표정보(Y)'])
point = gpd.GeoDataFrame(df1, geometry=gpd.points_from_xy(df1['좌표정보(X)'], df1['좌표정보(Y)'], crs='epsg:5174'))

point_utm =point.to_crs(5179)
cheongju_emd=cheongju_emd.to_crs(5179)


# In[12]:


fig, ax = plt.subplots(figsize=(10, 10))
cheongju_emd.boundary.plot(ax=ax, color='black', linewidth=0.1)
point_utm.plot(ax = ax, color = 'blue', markersize = 5)
plt.show()


# In[13]:


result_gdf = gpd.sjoin(cheongju_emd, point_utm, how='left', op='contains')
count_per_region = result_gdf.groupby('EMD_CD').size().reset_index(name='점포개수')
result_gdf2019 = merged_gdf.merge(count_per_region, how='left', on='EMD_CD')


# In[14]:


result_gdf2019.explore(column='점포개수')


# In[15]:


import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

df = pd.read_excel('D:/analysis/인허가데이터(1)/2020-2022.xlsx')
df1 = df.dropna(subset=['좌표정보(X)', '좌표정보(Y)'])

point = gpd.GeoDataFrame(df1, geometry=gpd.points_from_xy(df1['좌표정보(X)'], df1['좌표정보(Y)'], crs='epsg:5174'))

point_utm =point.to_crs(5179)
cheongju_emd.to_crs(5179)


# In[16]:


fig, ax = plt.subplots(figsize=(10, 10))
cheongju_emd.boundary.plot(ax=ax, color='black', linewidth=0.1)
point_utm.plot(ax = ax, color = 'red', markersize = 5)
plt.show()


# In[17]:


result_gdf = gpd.sjoin(cheongju_emd, point_utm, how='left', op='contains')
count_per_region = result_gdf.groupby('EMD_CD').size().reset_index(name='점포개수')
result_gdf2020 = merged_gdf.merge(count_per_region, how='left', on='EMD_CD')
result_gdf2020.explore(column='점포개수')


# In[26]:


result_gdf2017['면적'] = result_gdf2017.geometry.area
result_gdf2017['면적당점포개수']=result_gdf2017['점포개수']/result_gdf2017['면적']
result_gdf2017['순위'] = result_gdf2017['면적당점포개수'].rank(ascending=False, method='min').astype(int)

rank_sorted2017 = result_gdf2017.sort_values(by='순위', ascending=True)

result_gdf2017.explore(column='순위', cmap='viridis_r')


# In[25]:


result_gdf2018['면적'] = result_gdf2018.geometry.area
result_gdf2018['면적당점포개수']=result_gdf2018['점포개수']/result_gdf2018['면적']
result_gdf2018['순위'] = result_gdf2018['면적당점포개수'].rank(ascending=False, method='min').astype(int)

rank_sorted2018 = result_gdf2018.sort_values(by='순위', ascending=True)

result_gdf2018.explore(column='순위', cmap='viridis_r')


# In[24]:


result_gdf2019['면적'] = result_gdf2019.geometry.area
result_gdf2019['면적당점포개수']=result_gdf2019['점포개수']/result_gdf2019['면적']
result_gdf2019['순위'] = result_gdf2019['면적당점포개수'].rank(ascending=False, method='min').astype(int)
rank_sorted2019 = result_gdf2019.sort_values(by='순위', ascending=True)

result_gdf2019.explore(column='순위', cmap='viridis_r')


# In[23]:


result_gdf2020['면적'] = result_gdf2020.geometry.area
result_gdf2020['면적당점포개수']=result_gdf2020['점포개수']/result_gdf2020['면적']
result_gdf2020['순위'] = result_gdf2020['면적당점포개수'].rank(ascending=False, method='min').astype(int)

rank_sorted2020 = result_gdf2020.sort_values(by='순위', ascending=True)

result_gdf2020.explore(column='순위', cmap='viridis_r')


# In[27]:


import plotly.graph_objects as go
import pandas as pd

# 데이터를 '순위' 기준으로 내림차순 정렬
sorted_data = result_gdf2020.sort_values(by='순위', ascending=True)

# 상위 5개 지역 선택
top5_data = sorted_data.head(5)

# 기간과 순위 데이터
periods = ['2017-2019', '2018-2020', '2019-2021', '2020-2022']

# 그래프 데이터 초기화
data = []

# 각 지역에 대한 그래프 생성
for region in top5_data['EMD_NM'].unique():
    target_rows_2017 = result_gdf2017[result_gdf2017['EMD_NM'] == region]
    target_rows_2018 = result_gdf2018[result_gdf2018['EMD_NM'] == region]
    target_rows_2019 = result_gdf2019[result_gdf2019['EMD_NM'] == region]
    target_rows_2020 = result_gdf2020[result_gdf2020['EMD_NM'] == region]

    store_counts = [
        target_rows_2017['면적당점포개수'].values[0],
        target_rows_2018['면적당점포개수'].values[0],
        target_rows_2019['면적당점포개수'].values[0],
        target_rows_2020['면적당점포개수'].values[0]
    ]

    # 순위 값 가져오기
    rank_column = [
        target_rows_2017['순위'].values[0],
        target_rows_2018['순위'].values[0],
        target_rows_2019['순위'].values[0],
        target_rows_2020['순위'].values[0]
    ]

    # 점포개수 값 가져오기
    count_column = [
        target_rows_2017['점포개수'].values[0],
        target_rows_2018['점포개수'].values[0],
        target_rows_2019['점포개수'].values[0],
        target_rows_2020['점포개수'].values[0]
    ]

    # 그래프 데이터 추가
    trace = go.Scatter(
        x=periods,
        y=store_counts,
        mode='lines+markers',
        marker=dict(size=10),
        name=region,
        # 호버텍스트에 순위와 점포개수 추가
        hovertext=[f'{region}<br>순위: {rank}<br>점포개수: {count}' for rank, count in zip(rank_column, count_column)]
    )
    data.append(trace)

# 레이아웃 설정
layout = go.Layout(
    title='상위 5개 지역의 기간별 상권 순위',
    xaxis=dict(title='기간'),
    yaxis=dict(title='면적당점포개수'),
)

# 그래프 생성
fig = go.Figure(data, layout)

# 그래프 표시
fig.show()


# In[28]:


import plotly.graph_objects as go
import pandas as pd

# 데이터를 '순위' 기준으로 오름차순 정렬
sorted_data = result_gdf2020.sort_values(by='순위', ascending=True)

# 하위 5개 지역 선택
bottom5_data = sorted_data.tail(5)

# 기간과 순위 데이터
periods = ['2017-2019', '2018-2020', '2019-2021', '2020-2022']

# 그래프 데이터 초기화
data = []

# 각 지역에 대한 그래프 생성
for region in bottom5_data['EMD_NM'].unique():
    target_rows_2017 = result_gdf2017[result_gdf2017['EMD_NM'] == region]
    target_rows_2018 = result_gdf2018[result_gdf2018['EMD_NM'] == region]
    target_rows_2019 = result_gdf2019[result_gdf2019['EMD_NM'] == region]
    target_rows_2020 = result_gdf2020[result_gdf2020['EMD_NM'] == region]

    store_counts = [
        target_rows_2017['면적당점포개수'].values[0],
        target_rows_2018['면적당점포개수'].values[0],
        target_rows_2019['면적당점포개수'].values[0],
        target_rows_2020['면적당점포개수'].values[0]
    ]
    
    # 순위 값 가져오기
    rank_column = [
        target_rows_2017['순위'].values[0],
        target_rows_2018['순위'].values[0],
        target_rows_2019['순위'].values[0],
        target_rows_2020['순위'].values[0]
    ]
    
    # 점포개수 값 가져오기
    count_column = [
        target_rows_2017['점포개수'].values[0],
        target_rows_2018['점포개수'].values[0],
        target_rows_2019['점포개수'].values[0],
        target_rows_2020['점포개수'].values[0]
    ]
    
    # 그래프 데이터 추가
    trace = go.Scatter(
        x=periods,
        y=store_counts,
        mode='lines+markers',
        marker=dict(size=10),
        name=region,
        # 호버텍스트에 순위와 점포개수 추가
        hovertext=[f'{region}<br>순위: {rank}<br>점포개수: {count}' for rank, count in zip(rank_column, count_column)]
    )
    data.append(trace)

# 레이아웃 설정
layout = go.Layout(
    title='하위 5개 지역의 기간별 상권 순위',
    xaxis=dict(title='기간'),
    yaxis=dict(title='면적당점포개수'),
)

# 그래프 생성
fig = go.Figure(data, layout)

# 그래프 표시
fig.show()


# In[29]:


import plotly.express as px
import pandas as pd
result_gdf2020['순위변화']=result_gdf2020['순위']-result_gdf2017['순위']
# 데이터를 '순위' 기준으로 내림차순 정렬
sorted_data = result_gdf2020.sort_values(by='순위', ascending=True)

# 상위 20개 지역 선택
top20_data = sorted_data.head(20)

# 기간과 순위 데이터
periods = ['2017-2019', '2020-2022']

# 그래프 데이터 초기화
data = []

# 각 지역에 대한 그래프 생성
for region in top20_data['EMD_NM'].unique():
    target_rows_2017 = result_gdf2017[result_gdf2017['EMD_NM'] == region]
    target_rows_2022 = result_gdf2020[result_gdf2020['EMD_NM'] == region]

    # 순위 값 가져오기
    rank_2017 = target_rows_2017['순위'].values[0]
    rank_2022 = target_rows_2022['순위'].values[0]

    # 막대그래프 데이터 추가
    data.append({
        'Region': region,
        '2017-2019': rank_2017,
        '2020-2022': rank_2022,
        'Change': rank_2022 - rank_2017
    })

# 데이터프레임 생성
df = pd.DataFrame(data)

# 막대그래프 생성
fig = px.bar(
    df,
    x='Region',
    y='Change',
    title='상위 20개 지역의 2017-2019와 2020-2022 상권 순위 변화',
    labels={'Change': '순위 변화'},
    color='Change',
    color_continuous_scale=['blue', 'white', 'red'],  # 음수는 파란색, 0은 하얀색, 양수는 빨간색
    color_continuous_midpoint=0  # 중간값의 색상을 조정 (기본값은 0)
)

# Y축 뒤집기
fig.update_yaxes(autorange="reversed")

# 그래프 표시
fig.show()


# In[30]:


import plotly.express as px
import pandas as pd

# 데이터를 '순위' 기준으로 내림차순 정렬
sorted_data = result_gdf2020.sort_values(by='순위', ascending=True)

# 중위 20개 지역 선택
middle20_data = sorted_data.iloc[len(sorted_data)//2-10:len(sorted_data)//2+10]

# 기간과 순위 데이터
periods = ['2017-2019', '2020-2022']

# 그래프 데이터 초기화
data = []

# 각 지역에 대한 그래프 생성
for region in middle20_data['EMD_NM'].unique():
    target_rows_2017 = result_gdf2017[result_gdf2017['EMD_NM'] == region]
    target_rows_2022 = result_gdf2020[result_gdf2020['EMD_NM'] == region]

    # 순위 값 가져오기
    rank_2017 = target_rows_2017['순위'].values[0]
    rank_2022 = target_rows_2022['순위'].values[0]

    # 막대그래프 데이터 추가
    data.append({
        'Region': region,
        '2017-2019': rank_2017,
        '2020-2022': rank_2022,
        'Change': rank_2022 - rank_2017
    })

# 데이터프레임 생성
df_middle = pd.DataFrame(data)

# 막대그래프 생성
fig_middle = px.bar(
    df_middle,
    x='Region',
    y='Change',
    title='중위 20개 지역의 2017-2019와 2020-2022 상권 순위 변화',
    labels={'Change': '순위 변화'},
    color='Change',
    color_continuous_scale=['blue', 'white', 'red'],  # 음수는 파란색, 0은 하얀색, 양수는 빨간색
    color_continuous_midpoint=0  # 중간값의 색상을 조정 (기본값은 0)
)

# Y축 뒤집기
fig_middle.update_yaxes(autorange="reversed")

# 그래프 표시
fig_middle.show()


# In[31]:


import plotly.express as px
import pandas as pd

# 데이터를 '순위' 기준으로 오름차순 정렬
sorted_data = result_gdf2020.sort_values(by='순위', ascending=True)

# 하위 20개 지역 선택
bottom20_data = sorted_data.tail(20)

# 기간과 순위 데이터
periods = ['2017-2019', '2020-2022']

# 그래프 데이터 초기화
data = []

# 각 지역에 대한 그래프 생성
for region in bottom20_data['EMD_NM'].unique():
    target_rows_2017 = result_gdf2017[result_gdf2017['EMD_NM'] == region]
    target_rows_2022 = result_gdf2020[result_gdf2020['EMD_NM'] == region]

    # 순위 값 가져오기
    rank_2017 = target_rows_2017['순위'].values[0]
    rank_2022 = target_rows_2022['순위'].values[0]

    # 막대그래프 데이터 추가
    data.append({
        'Region': region,
        '2017-2019': rank_2017,
        '2020-2022': rank_2022,
        'Change': rank_2022 - rank_2017
    })

# 데이터프레임 생성
df_bottom = pd.DataFrame(data)

# 막대그래프 생성
fig_bottom = px.bar(
    df_bottom,
    x='Region',
    y='Change',
    title='하위 20개 지역의 2017-2019와 2020-2022 상권 순위 변화',
    labels={'Change': '순위 변화'},
    color='Change',
    color_continuous_scale=['blue', 'white', 'red'],  # 음수는 파란색, 0은 하얀색, 양수는 빨간색
    color_continuous_midpoint=0  # 중간값의 색상을 조정 (기본값은 0)
)

# Y축 뒤집기
fig_bottom.update_yaxes(autorange="reversed")

# 그래프 표시
fig_bottom.show()


# In[ ]:


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# 데이터를 '순위' 기준으로 내림차순 정렬
sorted_data = result_gdf2020.sort_values(by='순위', ascending=True)

# 상위 5개 지역 선택
top5_data = sorted_data.head(5)
# 하위 5개 지역 선택
bottom5_data = sorted_data.tail(5)

# 기간과 순위 데이터
periods = ['2017-2019', '2018-2020', '2019-2021', '2020-2022']

# 그래프 데이터 초기화
data_top5 = []
data_bottom5 = []

# 각 지역에 대한 그래프 생성
for region in top5_data['EMD_NM'].unique():
    target_rows_2017 = result_gdf2017[result_gdf2017['EMD_NM'] == region]
    target_rows_2018 = result_gdf2018[result_gdf2018['EMD_NM'] == region]
    target_rows_2019 = result_gdf2019[result_gdf2019['EMD_NM'] == region]
    target_rows_2020 = result_gdf2020[result_gdf2020['EMD_NM'] == region]

    store_counts = [
        target_rows_2017['면적당점포개수'].values[0],
        target_rows_2018['면적당점포개수'].values[0],
        target_rows_2019['면적당점포개수'].values[0],
        target_rows_2020['면적당점포개수'].values[0]
    ]

    # 순위 값 가져오기
    rank_column = [
        target_rows_2017['순위'].values[0],
        target_rows_2018['순위'].values[0],
        target_rows_2019['순위'].values[0],
        target_rows_2020['순위'].values[0]
    ]

    # 점포개수 값 가져오기
    count_column = [
        target_rows_2017['점포개수'].values[0],
        target_rows_2018['점포개수'].values[0],
        target_rows_2019['점포개수'].values[0],
        target_rows_2020['점포개수'].values[0]
    ]

    # 그래프 데이터 추가
    trace = go.Scatter(
        x=periods,
        y=store_counts,
        mode='lines+markers',
        marker=dict(size=10),
        name=region,
        # 호버텍스트에 순위와 점포개수 추가
        hovertext=[f'{region}<br>순위: {rank}<br>점포개수: {count}' for rank, count in zip(rank_column, count_column)]
    )
    data_top5.append(trace)

    # 각 지역에 대한 그래프 생성 (하위 5개 지역)
for region in bottom5_data['EMD_NM'].unique():
    target_rows_2017 = result_gdf2017[result_gdf2017['EMD_NM'] == region]
    target_rows_2018 = result_gdf2018[result_gdf2018['EMD_NM'] == region]
    target_rows_2019 = result_gdf2019[result_gdf2019['EMD_NM'] == region]
    target_rows_2020 = result_gdf2020[result_gdf2020['EMD_NM'] == region]

    store_counts = [
        target_rows_2017['면적당점포개수'].values[0],
        target_rows_2018['면적당점포개수'].values[0],
        target_rows_2019['면적당점포개수'].values[0],
        target_rows_2020['면적당점포개수'].values[0]
    ]

    # 순위 값 가져오기
    rank_column = [
        target_rows_2017['순위'].values[0],
        target_rows_2018['순위'].values[0],
        target_rows_2019['순위'].values[0],
        target_rows_2020['순위'].values[0]
    ]

    # 점포개수 값 가져오기
    count_column = [
        target_rows_2017['점포개수'].values[0],
        target_rows_2018['점포개수'].values[0],
        target_rows_2019['점포개수'].values[0],
        target_rows_2020['점포개수'].values[0]
    ]

    # 그래프 데이터 추가
    trace = go.Scatter(
        x=periods,
        y=store_counts,
        mode='lines+markers',
        marker=dict(size=10),
        name=region,
        # 호버텍스트에 순위와 점포개수 추가
        hovertext=[f'{region}<br>순위: {rank}<br>점포개수: {count}' for rank, count in zip(rank_column, count_column)]
    )
    data_bottom5.append(trace)
    
# 레이아웃 설정 (상위 5개와 하위 5개 지역)
layout_combined = go.Layout(
    title='상위 5개 지역과 하위 5개 지역의 기간별 상권 밀도',
    xaxis=dict(title='기간'),
    yaxis=dict(title='면적당점포개수'),
)

# 상위 5개 지역 그래프 데이터 생성
# fig_combined = go.Figure(data=data_top5, layout=layout_combined)

# 하위 5개 지역 그래프 데이터 추가
# for trace in data_bottom5:
    # fig_combined.add_trace(trace)

# 대시보드 레이아웃 정의 #마지막 그래프는 상위와 하위 5개 지역 합친 그래프
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("상권 분석 대시보드"),
  
    dcc.Tabs(id='tabs', value='tab-top_bottom', children=[
        dcc.Tab(label='상위 5개와 하위 5개', value='tab-top_bottom', children=[
            dcc.Graph(figure=go.Figure(data=data_top5, layout=layout_combined)),
            dcc.Graph(figure=go.Figure(data=data_bottom5, layout=layout_combined))
        ]),
        dcc.Tab(label='상위 20개, 중위 20개, 하위 20개', value='tab-top_middle_bottom')
    ]),

    html.Div(id='tab-content')
])

# 콜백을 사용하여 그래프 업데이트
@app.callback(
    Output('tab-content', 'children'),
    [Input('tabs', 'value')]
)
def update_graph(tab):
    if tab == 'tab-top_bottom':
        # 상위 5개 지역 그래프
        return dcc.Graph(
            id='top5-graph',
            figure=fig_combined
        )
    elif tab == 'tab-top_middle_bottom':
        # 순위변화에 따라 내림차순으로 정렬
        top20_data_sorted = top20_data.sort_values(by='순위변화', ascending=False)
        middle20_data_sorted = middle20_data.sort_values(by='순위변화', ascending=False)
        bottom20_data_sorted = bottom20_data.sort_values(by='순위변화', ascending=False)

        # 상위 20개 지역 그래프
        updated_figure_top20 = px.bar(
            top20_data_sorted,
            x='EMD_NM',
            y=-top20_data_sorted['순위변화'],
            title='상위 20개 지역의 순위변화에 따른 그래프',
            labels={'EMD_NM':'읍면동','순위변화': '순위 변화'},
            color=[('-' if val >= 0 else '+') for val in top20_data_sorted['순위변화']],
            color_discrete_map={'+': 'red', '-': 'blue'}
        )
        updated_figure_top20.update_layout(yaxis_title='순위변화')

        # 중위 20개 지역 그래프
        updated_figure_middle20 = px.bar(
            middle20_data_sorted,
            x='EMD_NM',
            y=-middle20_data_sorted['순위변화'],
            title='중위 20개 지역의 순위변화에 따른 그래프',
            labels={'EMD_NM':'읍면동','순위변화': '순위 변화'},
            color=[('-' if val >= 0 else '+') for val in middle20_data_sorted['순위변화']],
            color_discrete_map={'+': 'red', '-': 'blue'}
        )
        updated_figure_middle20.update_layout(yaxis_title='순위변화')

        # 하위 20개 지역 그래프
        updated_figure_bottom20 = px.bar(
            bottom20_data_sorted,
            x='EMD_NM',
            y=-bottom20_data_sorted['순위변화'],
            title='하위 20개 지역의 순위변화에 따른 그래프',
            labels={'EMD_NM':'읍면동','순위변화': '순위 변화'},
            color=[('-' if val >= 0 else '+') for val in bottom20_data_sorted['순위변화']],
            color_discrete_map={'+': 'red', '-': 'blue'}
        )
        updated_figure_bottom20.update_layout(yaxis_title='순위변화')

        # 중간과 하위 20개 지역 그래프를 리스트로 반환
        return [
            dcc.Graph(id='top20-change-graph', figure=updated_figure_top20),
            dcc.Graph(id='middle20-change-graph', figure=updated_figure_middle20),
            dcc.Graph(id='bottom20-change-graph', figure=updated_figure_bottom20)
        ]

    # 그 외의 경우에는 아무것도 반환하지 않음
    return None


# In[ ]:


if __name__ == '__main__':
    app.run_server(port=8053, debug=True)


# In[ ]:




