import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.plotly as py
import dash_html_components as html
import dash_core_components as dcc
from sklearn.decomposition import PCA
import time
import datetime



def apply_dtparameters(dt, portfolio_selection, return_interval, time_cluster, timerange_value):
    ''' Build portfolio, compute return, cluster the time series and filter time range '''
    date_from = unixToDatetime(timerange_value[0])
    date_to = unixToDatetime(timerange_value[1])
    
    dt['PORTFOLIO'] = dt[portfolio_selection].mean(axis=1,skipna=False)
    
    data_timerange = dt[(dt.index>date_from) & (dt.index<date_to)]
    
    data_interval = (data_timerange.
         resample(return_interval).
         first())
    
    period_frequency = str(time_cluster)+return_interval
    period_breaks = pd.date_range(end=data_interval.index[-1],
                                  periods=1000,
                                  freq=period_frequency)[:-1]
    
    data_period = (data_interval.
                   reset_index().
                   merge(pd.DataFrame({'Period':period_breaks}),
                           how='left',
                           left_on='Date',
                           right_on='Period'))
    
    data_period['Period'] = data_period['Period'].fillna(method='ffill')
    
    data_period.set_index(['Date','Period'],inplace=True)

    return data_period


def apply_return(dt,type_return='standard',apply_fillna=False):
    
    if apply_fillna==True:
        dt = dt.fillna(method='ffill')
        
    dt_return = dt.apply(lambda x:np.log(x)-np.log(x.shift(1)))
    
    if type_return=='cs_returnquant':
        dt_return = dt_return.rank(axis=1,pct=True)
        
    return(dt_return)
    
def PCA_label(ncomp=5):
    return(['PC'+str(i+1) for i in range(ncomp)])

def apply_PCA(dt,ncomp=5,na_threshold=.4,apply_fillna=False):
    
    original_index = dt.columns
    
    
    dt_nonna = (apply_return(dt,
                            type_return='standard',
                            apply_fillna=True).
        dropna(axis=1,thresh=dt.shape[0]*(1-na_threshold)).
        dropna(axis=0))
        
    pca = PCA(n_components=ncomp, random_state=1)
    pca.fit_transform(dt_nonna)
    
    pca_eigenvectors = pca.components_
    pca_eigenvalues = pca.explained_variance_
    PCA_loadings = pca_eigenvectors.T * np.sqrt(pca_eigenvalues)
    
    components_df = (pd.concat([pd.DataFrame({'PCA_ev':[np.argmax(abs(v))+1 for v in pca_eigenvectors.T],
                                  'PCA_max_ev':[max(v,key=abs) for v in pca_eigenvectors.T],
                                  'PCA_max_absev':[max(abs(v)) for v in pca_eigenvectors.T],
                                  'point':dt_nonna.columns}),
                                pd.DataFrame(pca_eigenvectors.T,
                                           columns=PCA_label(ncomp))
                                ],axis=1))
                                
            
            
    
    components_df = (pd.DataFrame({'point':original_index}).
                     merge(components_df,how='left').
                     set_index('point'))
    
    if apply_fillna==True:
       components_df.fillna(0,inplace=True) 
       
    return(components_df)

    
    


def apply_summ(dt,summ_type,key):
    if key=='Period':
        group = dt.groupby(dt.index.get_level_values('Period'))
    if key=='All':
        group = dt
    
    if summ_type=='mean':
        dt_summperperiod = group.mean()
    if summ_type=='VAR':
        dt_summperperiod = group.quantile(.1,axis=0)
    
    return(dt_summperperiod)
    
def apply_riskreturn(dt,laststate_tlength):
    
    dt_riskreturn = pd.DataFrame({'return':apply_summ(dt,
                                                      summ_type='mean',
                                                      key='All'),
                                  'risk':apply_summ(dt,
                                                      summ_type='VAR',
                                                      key='All'),
                                  'return_laststate':apply_summ(dt,
                                                      summ_type='mean',
                                                      key='Period').tail(laststate_tlength).mean()
                                                      })
    
    return(dt_riskreturn)

    
def unixTimeMillis(dt):
    ''' Convert datetime to unix timestamp '''
    'details: https://stackoverflow.com/questions/51063191/date-slider-with-plotly-dash-does-not-work/51063377#51063377'
    return int(time.mktime(dt.timetuple()))

def unixToDatetime(unix):
    ''' Convert unix timestamp to datetime. '''
    'details: https://stackoverflow.com/questions/51063191/date-slider-with-plotly-dash-does-not-work/51063377#51063377'
    return pd.to_datetime(unix,unit='s')

def getMarks(start, end, Nth=100):
    ''' Returns the marks for labeling.
        Every Nth value will be used.
    '''
    'details: https://stackoverflow.com/questions/51063191/date-slider-with-plotly-dash-does-not-work/51063377#51063377'

    result = {}
    for i, date in enumerate(data.index):
        if(i%Nth == 1):
            # Append value to dict
            result[unixTimeMillis(date)] = str(date.strftime('%Y-%m-%d'))

    return result
    
        
    











def GeneratePortfolioVis(data):
    
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date',inplace=True)
    
    app = dash.Dash()
    
    app.layout = html.Div([
        
        
        html.Div([
            html.H4(children='Portfolio Visualization Dashboard')
            
            ],style={'size':40, 'color':'red', 'width': '48%', 'display': 'inline-block', 'horizontalAlign' : "middle"}),
            
        html.Div([
            html.Div([
                html.Label('Return frequency:'),
                dcc.Dropdown(
                id='return-frequency',
                options=[
                        {'label':'Daily', 'value': 'D'},
                        {'label':'Weekly', 'value': 'W'},
                        {'label':'Quarterly', 'value': 'M'},
                        {'label':'Monthly', 'value': 'Q'},
                        {'label':'Annual', 'value': 'A'}],
                value='W'
                )
            ], className="four columns"),
            html.Div([
                html.Label('Time cluster length:'),
                dcc.Input(
                    id='time-cluster',
                    type='number',
                    value=10)
            ], className="four columns"),
            html.Div([
                html.Label('Number of PCA components:'),
                dcc.Input(
                    id='PCA-ncomp',
                    type='number',
                    value=5)
            ], className="four columns"),
        ], className="row"),
    
        html.Label('EW portfolio selection:'),
        dcc.Dropdown(
            id='portfolio_selection',
            options=[{'label': i, 'value': i} for i in data.columns.values],
            value=data.columns.values[:3],
            multi=True
        ),
    
    
        html.Div([
            html.Div([
                dcc.Graph(id='3D_Scatter'),
            ], className="eight columns"),
            html.Div([
                    dcc.Graph(id='plot_return'),
                    dcc.Graph(id='plot_price')
                     ], className="four columns")
        ], className="row"),
    
        dcc.RangeSlider(
                    id='timerange--slider',
                    min = unixTimeMillis(data.index.min()),
                    max = unixTimeMillis(data.index.max()),
                    value = [unixTimeMillis(data.index.min()),
                             unixTimeMillis(data.index.max())],
                    marks=getMarks(data.index.min(),
                                data.index.max())
                ),
    
        html.Div([
            html.Div([
                dcc.Graph(id='plot_pcaeigen')
            ], className="four columns"),
    
            html.Div([
                dcc.Graph(id='plot_hist')
            ], className="four columns"),
            html.Div([
                dcc.Graph(id='plot_risk')
            ], className="four columns"),
        ], className="row")
    
    
    
    ])
                    
    @app.callback(
    dash.dependencies.Output('3D_Scatter', 'figure'),
    [dash.dependencies.Input('portfolio_selection', 'value'),
     dash.dependencies.Input('return-frequency', 'value'),
     dash.dependencies.Input('time-cluster', 'value'),
     dash.dependencies.Input('timerange--slider', 'value'),
     dash.dependencies.Input('PCA-ncomp', 'value')])


    def update_3D_Scatter(portfolio_selection,
                     return_interval,
                     time_cluster,
                     timerange_value,
                     PCA_ncomp):
       
        
        global data_parameterized, data_returns, data_pca, data_returns_period, data_risk_period, data_risk_return   
        
        data_parameterized = apply_dtparameters(data, 
                                                portfolio_selection, 
                                                return_interval, 
                                                time_cluster, 
                                                timerange_value)
        
        data_returns = apply_return(data_parameterized,
                                    type_return='cs_returnquant')
        
        data_pca = apply_PCA(data_parameterized,
                             ncomp=PCA_ncomp,
                             na_threshold=.3,
                             apply_fillna=True)
        
        data_returns_period = apply_summ(data_returns,
                                         summ_type='mean',
                                         key='Period')
        data_risk_period = apply_summ(data_returns,
                                      summ_type='VAR',
                                      key='Period')
        
        
        data_risk_return = apply_riskreturn(data_returns,
                                            laststate_tlength=4)
        
        
        portfolio_flag = list(data_risk_return.index.values).index('PORTFOLIO')
       
        
        
        size_index = [np.exp(x*5.5) for x in list(data_risk_return.return_laststate)]
        size_index[portfolio_flag]=30
        
        colors=['gray','blue','red','green','yellow','orange','pink','purple']
        color_index=[colors[int(x)] for x in data_pca.PCA_ev.values]
        color_index[portfolio_flag]='black'
        
    
    
        return {
            'data': [go.Scatter3d(
                x=[x for x in data_pca['PCA_max_ev']],
                y=data_risk_return['risk'],
                z=data_risk_return['return'],
                text=data_risk_return.index,
                mode='markers',
                marker={
                    'opacity': 0.5,
                    'size': size_index,
                    'color': color_index,
                    'line': {'width': 0.5, 'color': 'white'}
                }
            )],
    
            'layout': go.Layout(
                xaxis={
                    'title': 'PCA Max Eigenvector',
                    'type': 'log'
                },
                yaxis={
                    'title': 'Risk',
                    'type': 'log'
                },
                scene = dict(
                        xaxis = dict(
                            title='PCA Max Eigenvector'),
                        yaxis = dict(
                            title='Risk'),
                        zaxis = dict(
                            title='Return')),
                showlegend=False,
                margin={'l': 5, 'b': 5, 't': 5, 'r': 0},
                hovermode='closest',
                height=400
            )
        }
    
    @app.callback(
        dash.dependencies.Output('plot_return', 'figure'),
        [dash.dependencies.Input('3D_Scatter', 'hoverData'),
         dash.dependencies.Input('return-frequency', 'value'),
         dash.dependencies.Input('time-cluster', 'value'),
         dash.dependencies.Input('timerange--slider', 'value')])
    def update_plot_return(hoverData, return_interval, time_cluster, timerange_value):
        
    
    
        
        point_selection = hoverData['points'][0]['text']
        
        
        title = '<b>{}</b><br>{}'.format('Return quantiles', point_selection)
    
        return {
            'data': [go.Scatter(
                            x=data_returns_period.index,
                            y=data_returns_period[point_selection],
                            mode='lines+markers',
                            name=point_selection
                                ),
                    go.Scatter(
                            x=data_returns_period.index,
                            y=data_returns_period['PORTFOLIO'],
                            mode='lines+markers',
                            name='Portfolio'
                                ),
                    go.Scatter(
                            x=data_returns_period.index,
                            y=np.repeat([0.5],len(data_returns_period[point_selection])),
                            mode="lines",
                            line= {'color':'gray', 'width':2},
                            name='50%'
                            )
                    ],
            'layout': {           
                'margin': {'l': 20, 'b': 30, 'r': 10, 't': 10},
                'annotations': [{
                    'x': 0, 'y': 0.85, 'xanchor': 'left', 'yanchor': 'bottom',
                    'xref': 'paper', 'yref': 'paper', 'showarrow': False,
                    'align': 'left', 'bgcolor': 'rgba(255, 255, 255, 0.5)',
                    'text': title
                }],
                'yaxis': {'range': [0,1], 'dtick':0.1},
                'xaxis': {'showgrid': False},
                'height': 200,
                'legend': {'orientation': 'h'}
            }
        }
    
    
    
    @app.callback(
        dash.dependencies.Output('plot_price', 'figure'),
        [dash.dependencies.Input('3D_Scatter', 'hoverData'),
         dash.dependencies.Input('return-frequency', 'value'),
         dash.dependencies.Input('time-cluster', 'value'),
         dash.dependencies.Input('timerange--slider', 'value')])
    def update_plot_price(hoverData, return_interval, time_cluster, timerange_value):
    
        point_selection = hoverData['points'][0]['text']
        
        rebasing = data_parameterized.fillna(method='bfill')[point_selection][0]/data_parameterized.fillna(method='bfill')['PORTFOLIO'][0]
    
        title = '<b>{}</b><br>{}'.format('Historical prices', point_selection)
    
        return {
            'data': [go.Scatter(
                            x=data_parameterized.index.get_level_values('Date'),
                            y=data_parameterized[point_selection],
                            mode='lines',
                            name=point_selection
                                ),
                    go.Scatter(
                            x=data_parameterized.index.get_level_values('Date'),
                            y=[r*rebasing for r in data_parameterized['PORTFOLIO']],
                            mode='lines',
                            name='Portfolio'
                                )
                    ],
            'layout': {           
                'margin': {'l': 20, 'b': 30, 'r': 10, 't': 10},
                'annotations': [{
                    'x': 0, 'y': 0.85, 'xanchor': 'left', 'yanchor': 'bottom',
                    'xref': 'paper', 'yref': 'paper', 'showarrow': False,
                    'align': 'left', 'bgcolor': 'rgba(255, 255, 255, 0.5)',
                    'text': title
                }],
                'yaxis': {},
                'xaxis': {'showgrid': False},
                'height': 200,
                'legend': {'orientation': 'h'}
            }
        }
    
    
    @app.callback(
        dash.dependencies.Output('plot_hist', 'figure'),
        [
         dash.dependencies.Input('3D_Scatter', 'hoverData'),
         dash.dependencies.Input('return-frequency', 'value'),
         dash.dependencies.Input('time-cluster', 'value'),
         dash.dependencies.Input('timerange--slider', 'value')])
    def update_plot_hist(hoverData, return_interval, time_cluster, timerange_value):
    
        point_selection = hoverData['points'][0]['text']
    
        title = '<b>{}</b><br>{}'.format('Return quantile histogram', point_selection)
        
        return {
            'data': [go.Histogram(x=data_returns_period[point_selection],
                                  nbinsx=15,
                                  histnorm='probability',
                                  opacity=0.75,
                                  name=point_selection),
                    go.Histogram(x=data_returns_period['PORTFOLIO'],
                                 nbinsx=15,
                                  histnorm='probability',
                                  opacity=0.75,
                                  name='PORTFOLIO')],
            'layout': {
                'height': 200,
                'margin': {'l': 20, 'b': 30, 'r': 10, 't': 10},
                'annotations': [{
                    'x': 0, 'y': 0.85, 'xanchor': 'left', 'yanchor': 'bottom',
                    'xref': 'paper', 'yref': 'paper', 'showarrow': False,
                    'align': 'left', 'bgcolor': 'rgba(255, 255, 255, 0.5)',
                    'text': title
                }],
                'xaxis': {'showgrid': False, 'range': [0,1]},
                'barmode': 'overlay'
            }
        }
    
    
    @app.callback(
        dash.dependencies.Output('plot_pcaeigen', 'figure'),
        [dash.dependencies.Input('3D_Scatter', 'hoverData'),
         dash.dependencies.Input('return-frequency', 'value'),
         dash.dependencies.Input('time-cluster', 'value'),
         dash.dependencies.Input('timerange--slider', 'value'),
         dash.dependencies.Input('PCA-ncomp', 'value')])
    def update_plot_pcaeigen(hoverData, return_interval, time_cluster, timerange_value,ncomp):
    
        point_selection = hoverData['points'][0]['text']
    
        title = '<b>{}</b><br>{}'.format('PCA eigenvectors', point_selection)
        
        return {
            'data': [go.Bar(x=PCA_label(ncomp),
                            y=list(data_pca.loc[point_selection,PCA_label(ncomp)]),
                            name=point_selection,
                            marker={'color':'rgb(55, 83, 109)'}),
                    go.Bar(x=PCA_label(ncomp),
                            y=list(data_pca.loc['PORTFOLIO',PCA_label(ncomp)]),
                            name='PORTFOLIO',
                            marker={'color':'rgb(26, 118, 255)'})],
            'layout': {
                'height': 200,
                'margin': {'l': 20, 'b': 30, 'r': 10, 't': 10},
                'annotations': [{
                    'x': 0, 'y': 0.85, 'xanchor': 'left', 'yanchor': 'bottom',
                    'xref': 'paper', 'yref': 'paper', 'showarrow': False,
                    'align': 'left', 'bgcolor': 'rgba(255, 255, 255, 0.5)',
                    'text': title
                }],
                'barmode': 'group'
            }
        }
    
    @app.callback(
        dash.dependencies.Output('plot_risk', 'figure'),
        [dash.dependencies.Input('3D_Scatter', 'hoverData'),
         dash.dependencies.Input('return-frequency', 'value'),
         dash.dependencies.Input('time-cluster', 'value'),
         dash.dependencies.Input('timerange--slider', 'value')])
    def update_plot_risk(hoverData, return_interval, time_cluster, timerange_value):
    
        point_selection = hoverData['points'][0]['text']
        
        
        title = '<b>{}</b><br>{}'.format('10% Var', point_selection)
    
        return {
            'data': [go.Scatter(
                            x=data_risk_period.index,
                            y=data_risk_period[point_selection],
                            mode='lines+markers',
                            name=point_selection
                                ),
                    go.Scatter(
                            x=data_risk_period.index,
                            y=data_risk_period['PORTFOLIO'],
                            mode='lines+markers',
                            name='Portfolio'
                                ),
                    go.Scatter(
                            x=data_risk_period.index,
                            y=np.repeat([0.5],len(data_risk_period[point_selection])),
                            mode="lines",
                            line= {'color':'gray', 'width':2},
                            name='50%'
                            )
                    ],
            'layout': {           
                'margin': {'l': 20, 'b': 30, 'r': 10, 't': 10},
                'annotations': [{
                    'x': 0, 'y': 0.85, 'xanchor': 'left', 'yanchor': 'bottom',
                    'xref': 'paper', 'yref': 'paper', 'showarrow': False,
                    'align': 'left', 'bgcolor': 'rgba(255, 255, 255, 0.5)',
                    'text': title
                }],
                'yaxis': {'range': [0,1], 'dtick':0.1},
                'xaxis': {'showgrid': False},
                'height': 200,
                'legend': {'orientation': 'h'}
            }
        }
    
    
    app.css.append_css({
        'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
    })
    
    app.run_server()
    

