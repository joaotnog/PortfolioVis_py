# PortfolioVis_py 

**PortfolioVis_py** is a tool for multidimensional portfolio visualization built in Python and Dash. Robust metrics of risk, return, correlation structure and individual asset performance can be assessed simultaneously. 

## Prerequisites

The following modules are required:

- dash
- dash_core_components
- dash_html_components
- pandas
- plotly
- sklearn
- time
- numpy
- datetime

## Usage

The function requires a dataset with asset prices and 'Date' as single input and can be run as:

```python
GeneratePortfolioVis(data=data_sample)
``` 

## Result Description

The user can plug into the UI:

- Return frequency.
- Length of time period clusters.
- Number of principal components for the PCA.
- Assets composing an equal weight portfolio.
- Time slice under analysis.

The following metrics are computed in the server:

- Robust_Return: Quantile of the cross-section of log returns.
- Robust_Risk: 10% VAR of Robust_Return.
- PCA: Principal component analysis of the log returns. 

The output is a 3d plot where each point represents an asset with dimensions:

- z-axis: Robust_Return.
- y-axis: Robust_Risk.
- x-axis: Largest PCA eigenvector.
- Point color: Asset cluster, assigned as the PC with highest absolute eigenvector.
- Point size: Average Robust_Return for recent time clusters as a proxy of current asset regime.
- Mouse hover generated charts: Portfolio vs Asset line charts of clustered Robust_Return, Robust_Risk, historical prices, histogram of Robust_Return and PCA eigenvectors.   
 

## License

This project is licensed under the MIT License.

