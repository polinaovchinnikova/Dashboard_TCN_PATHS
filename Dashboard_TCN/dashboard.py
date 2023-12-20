#Libraries 
import pandas as pd
from dash import Dash, Input, Output, dcc, html
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.express as px
import plotly.graph_objects as go 
from datetime import datetime

#Upload Data
data = (
    pd.read_csv("PolinaExport07042023.csv") # Upload new data file (replace "PolinaExport07042023.csv" with the path to your data file)
    .assign(ScreeningDate=lambda data: pd.to_datetime(data["ScreeningDate"], format="%m/%d/%Y")) # Convert "ScreeningDate" column to datetime format
    .rename(columns={"PtDatabase::CommJailEnrollment": "EnrollmentType"}) # Rename column "PtDatabase::CommJailEnrollment" to "EnrollmentType"
    .sort_values(by="ScreeningDate") # Sort the data by "ScreeningDate"
)

# Created a new variable that is true or false for enrollemnt and that is being used as a filter for the enrolled cards and graphs 
data['Enrolled']=data["PtDatabase::EnrollmentDate"].notna() & (data['PtDatabase::PIDStatus'] != 'Not Released in 90 Days') & (data['PtDatabase::PIDStatus'] != 'Did Not Complete BL')

# Get options for dropdowns
sites = pd.Series(data["Site"].sort_values().unique()).dropna() # Get unique sites from the "Site" column and remove any missing values
options = [{"label": "All", "value": slice(None)}] + [  # Create options for the dropdown menu
    {"label": site, "value": site} # Each option has a label and value corresponding to a site
    for site in sites
]

enrollment_statuses = pd.Series(data["EnrollmentType"].sort_values().unique()).dropna() # Get unique enrollment types and remove any missing values

external_stylesheets = [ # List of external style sheets for the Dash app
    {
        "href": (
            "https://fonts.googleapis.com/css2?"
            "family=Lato:wght@400;700&display=swap" # Link to Google Fonts for Lato font styles
        ),
        "rel": "stylesheet",
    },
    "style.css", # Additional style sheet named "style.css"
]
app = Dash(__name__, external_stylesheets=external_stylesheets) # Create a Dash app with the specified external stylesheets
app.title = "TCN PATHS" # Set the title of the Dash app to "TCN PATHS"

# Layout of dashboard, organized into html divisions

# Header
header = html.Div(
    children=[
        html.Div(
            children=[
                html.Img(src="/assets/tcn_logo.png", className="header-logo"), # Display the TCN logo
            ],
            className="header-logo-container",
        ),
        html.H1(
            children="",
            className="header-title",
        ),
    ],
    className="header",
)

# Menu for filtering
menu = html.Div(
    children=[
        html.Div(
            children=[
                html.Div(children="Site", className="menu-title"),  # Title for the site filter
                dcc.Dropdown(
                    id="site-filter",
                    options=[{"label": "All", "value": "All"}] # Default option for selecting all sites
                    + [{"label": site, "value": site} for site in sites], # Dropdown options for each site
                    value="All", # Initial value for the site filter
                    clearable=False, # Disable clearing the site filter
                    className="dropdown",
                ),
            ]
        ),
        html.Div(
            children=[
                html.Div(children="Enrollment Status", className="menu-title"), # Title for the enrollment status filter
                dcc.Dropdown(
                    id="enrollment-status-filter",
                    options=[{"label": "All", "value": "All"}]  # Default option for selecting all enrollment statuses
                    + [
                        {"label": enrollment_status, "value": enrollment_status}
                        for enrollment_status in enrollment_statuses # Dropdown options for each enrollment status
                    ],
                    value="All", # Initial value for the enrollment status filter
                    clearable=False, # Disable clearing the enrollment status filter
                    className="dropdown",
                ),
            ],
        ),
    ],
    className="menu",
)


# Site Card style
site_card = html.Div(
    children=[
        html.Div("Sites", className="data-card-title"),  # Title for site card
        html.Div(id="site-card-value", className="site-card-value"),  # Placeholder for site card value
        html.Div("Enrollment Count by Site", style={"font-size": "12px"}),  # Description for site card
    ],
    className="data-card",
)

# Age Card style
age_card = html.Div(
    children=[
        html.Div("Age", className="data-card-title"),  # Title for age card
        html.Div(id="age-card-value", className="data-card-value"),  # Placeholder for age card value
        html.Div("Mean Age", style={"font-size": "12px"}),  # Description for age card
    ],
    className="data-card",
)

# Race Card style
race_card = html.Div(
    children=[
        html.Div("Race", className="data-card-title"),  # Title for race card
        html.Div(id="race-card-value", className="data-card-value"),  # Placeholder for race card value
        html.Div("Count by Race", style={"font-size": "12px"}),  # Description for race card
    ],
    className="data-card",
)

# Gender Card style
gender_card = html.Div(
    children=[
        html.Div("Gender", className="data-card-title"),  # Title for gender card
        html.Div(id="gender-card-value", className="data-card-value"),  # Placeholder for gender card value
        html.Div("Count by Gender", style={"font-size": "12px"}),  # Description for gender card
    ],
    className="data-card",
)

# Conversion Rate Card style
conversion_rate_card = html.Div(
    children=[
        html.Div("Conversion Rate", className="data-card-title"),  # Title for conversion rate card
        html.Div(id="conversion-rate-card-value", className="data-card-value"),  # Placeholder for conversion rate card value
        html.Div("Screened vs. Enrolled", style={"font-size": "12px"}),  # Description for conversion rate card
    ],
    className="data-card",
)

# ARIMA enrollment projections card
arima_enrollment_card = html.Div(
    children=[
        html.Div("ARIMA - Enrollment Projections", className="data-card-title"),  # Title for ARIMA enrollment projections card
        html.Div(id="arima-enrollment-card-value", className="data-card-value"),  # Placeholder for ARIMA enrollment projections card value
        html.Div("Forecast for the next 3 months", style={"font-size": "12px"}),  # Description for ARIMA enrollment projections card
    ],
    className="data-card",
)

# Left side of graphs 
graphs_left=html.Div(
            children=[
                #Screening date 
                html.Div(
                    children=dcc.Graph(
                        id="screening-date-chart",
                        config={"displayModeBar": False},
                    ),
                    className="card",
                ),
                #ARIMA
                html.Div(
                    children=dcc.Graph(
                         id="arima-enrollment-chart",
                         config={"displayModeBar": False},
                     ),
                     className="card",
                ),
                #PID 
                html.Div(
                    children=dcc.Graph(
                        id="pid-status-chart",
                        config={"displayModeBar": False},
                    ),
                    className="card",
                ),
                # MOUD Type for enrolled
                html.Div(
                    children=dcc.Graph(
                        id="moudtype-enrolled-graph",
                        config={"displayModeBar": False},
                    ),
                    className="card",
                ),
        
                # DaysIncarcerated 
                html.Div(
                    children=dcc.Graph(
                        id="days-incarcerated-graph",
                        config={"displayModeBar": False},
                    ),
                    className="card",
                ),
            ],
            className="wrapper",
        )

# Right side graphs
graphs_right=html.Div(
    children=[
        #Enrolled date
         html.Div(
            children=dcc.Graph(
                id="enrolled_date_chart_figure",
                config={"displayModeBar": False},
            ),
            className="card",
        ),
    
        #Enrollment chart
        html.Div(
            children=dcc.Graph(
                id="enrollment-chart", #enrollment type
                config={"displayModeBar": False},
            ),
             className="card",
        ),
        #Referral Source
        html.Div(
            children=dcc.Graph(
                id="referral-source-chart",
                config={"displayModeBar": False},
            ),
            className="card",
        ),
        #OUD Score - bar graph for each score value
        html.Div(
            children=dcc.Graph(
                id="oudscore-graph",
                config={"displayModeBar": False},
            ),
            className="card",
        ),
    ],
    className="wrapper",
)

# Combine left side of graph and right side of graphs into one graph area
graph_area = html.Div(
    children=[
        graphs_left,
        graphs_right,
    ],
    className="graph-area",
)

# Total layout for dashboard with components placed in order
app.layout = html.Div(
    children=[
        header, 
        menu,
        site_card,
        html.Div(
            children=[ # Cards
                age_card,
                race_card,
                gender_card,
                conversion_rate_card,  
                arima_enrollment_card,
            ],
            className="data-display",
        ),
        graph_area,
    ]
)



# Function that updates everything else on the dashboard when the filters are changed
# ARIMA still causing a few problems so not currently showing it
@app.callback(
    #Graphs 
    Output("screening-date-chart", "figure"), # Screening date chart 
    Output("enrollment-chart", "figure"), #Enrollment Count/Type chart
    Output("arima-enrollment-chart", "figure"), #ARIMA chart 
    Output("referral-source-chart", "figure"), # Referral source chart 
    Output("enrolled_date_chart_figure", "figure"), #Enrolled date chart
    Output("pid-status-chart", "figure"),  #PID status chart
    Output("moudtype-enrolled-graph", "figure"), #MOUD type for enrolled participants chart
    Output("days-incarcerated-graph", "figure"), #Days incarcerated for enrolled participants chart 
    Output("oudscore-graph", "figure"), #OUD score distribution chart

    #Cards
    Output("age-card-value", "children"), # Age card 
    Output("race-card-value", "children"), # Race card
    Output("gender-card-value", "children"), # Gender card 
    Output("site-card-value", "children"),# Site card 
    Output("conversion-rate-card-value", "children"),  # Conversion rate card
    Output("arima-enrollment-card-value", "children"),  # ARIMA enrollment projections card

    #Update outputs based on filters 
    Input("site-filter", "value"), # Graphs change based on site filter
    Input("enrollment-status-filter", "value"), # Graphs change based on enrollment filter
)
# Update graph based on the site and enrollment status filters 
def update_charts(site, enrollment_status):
    query_args = []
    if site != "All":
        query_args.append("Site == @site") # Add filter condition for site if it's not "All"
    if enrollment_status != "All":
        query_args.append("`EnrollmentType` == @enrollment_status") # Add filter condition for enrollment status if it's not "All"

    if query_args == []:
        filtered_data = data # If no filter conditions, use the original data
        data_available = True  # Set data availability flag to True
    else:
        filtered_data = data.query(" and ".join(query_args)) # Apply the filter conditions to the data
        data_available = len(filtered_data) > 0  # Check if data is available

    if not data_available: # If no matching records found
        return (
            blank_figure(), # Return blank figures for each graph
            blank_figure(),
            blank_figure(),
            blank_figure(),
            blank_figure(),
            blank_figure(),
            blank_figure(),
            blank_figure(),
            blank_figure(),
            "NA", # Return "NA" for age
            [], # Return empty lists for additional data/card
            [], # Return empty lists for additional data/card 
            "No matching records found", # Return message for site
            "NA", # Return "NA" for conversion rate
            "NA", # Return "NA" for ARIMA
        )

## Screening Date chart 
    # Group by month and count the occurrences
    data_2022=filtered_data[filtered_data["ScreeningDate"] >= "2022-01-01"]
    data_2021=filtered_data[filtered_data["ScreeningDate"] < "2022-01-01"]
    screening_date_counts = data_2022["ScreeningDate"].dt.to_period('M').value_counts().sort_index()
    screening_date_counts_2021=len(data_2021)
    # Generate a range of months from the minimum to maximum dates
    # min_date = data_2022["ScreeningDate"].min().to_period('M')
    # max_date = data_2022["ScreeningDate"].max().to_period('M')
    min_date="2022-01"
    max_date=pd.to_datetime(data["PtDatabase::EnrollmentDate"]).max().to_period('M')
    months = pd.period_range(min_date, max_date, freq='M')
    dates=["2021 (all)"]+list(months.astype(str))
    # Initialize an array to hold the count values
    values = np.zeros(len(months)+1, dtype=int)
    # Fill in the count values for existing months
    # Prepend with value for 2021
    values[0]=screening_date_counts_2021
    for i, month in enumerate(months):
        if month in screening_date_counts.index:
            values[i+1] = screening_date_counts[month]
# graph
    screening_date_chart_figure = {
    "data": [
        {
            "x": dates,
            "y": values,
            "type": "bar",
            "hovertemplate": "Screening Date: %{x}<br>Count: %{y}<extra></extra>",
            "text": values,
            "textposition": "auto",
            "marker": {"line": {"color": "#FFFFFF", "width": 1}},  # Add a colored outline to each bar
        },
    ],
    "layout": {
        "title": {"text": "Screening Date", "x": 0.05, "xanchor": "left"},
        "xaxis": {"type": "category", "tickangle": -45},
        "yaxis": {"title": "Count", "fixedrange": True},
        "colorway": ["#065771"], 
        "annotations": [
            {
                "text": "Screening Date indicates the date of participant screening.",
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": 1.06, # Adjust the y-coordinate to position the footnote below the title
                "showarrow": False,
                "font": {"size": 11},
            },
        ],
        "margin": {"t": 50, "r": 10, "b": 80, "l": 60},  # Adjust the margins as needed
    },
}


## Enrolled Date chart 
    # Convert "PtDatabase::EnrollmentDate" column to datetime format
    filtered_data["PtDatabase::EnrollmentDate"] = pd.to_datetime(filtered_data["PtDatabase::EnrollmentDate"])
    # Group by month and count the occurrences
    data_2022 = filtered_data.loc[(filtered_data["PtDatabase::EnrollmentDate"] >= pd.to_datetime("2022-01-01")) & (filtered_data['Enrolled'])]
    data_2021 = filtered_data.loc[(filtered_data["PtDatabase::EnrollmentDate"] < pd.to_datetime("2022-01-01")) & (filtered_data['Enrolled'])]
    enrolled_date_counts = data_2022["PtDatabase::EnrollmentDate"].dt.to_period('M').value_counts().sort_index()
    enrolled_date_counts_2021 = len(data_2021)
    # Generate a range of months from the minimum to maximum dates, starting in 2022
    max_date=pd.to_datetime(data["PtDatabase::EnrollmentDate"]).max().to_period('M')
    months = pd.period_range(min_date, max_date, freq='M')
    dates = ["2021 (all)"] + list(months.astype(str))
    # Initialize an array to hold the count values
    values = np.zeros(len(months) + 1, dtype=int)
    values[0] = enrolled_date_counts_2021
    # Fill in the count values for existing months
    for i, month in enumerate(months):
     if month in enrolled_date_counts.index:
        values[i + 1] = enrolled_date_counts[month]
# graph
    enrolled_date_chart_figure = {
    "data": [
        {
            "x": dates,
            "y": values,
            "type": "bar",
            "hovertemplate": "Enrolled Date: %{x}<br>Count: %{y}<extra></extra>",
            "text": values,
            "textposition": "auto",
            "marker": {"line": {"color": "#FFFFFF", "width": 1}},
        },
    ],
    "layout": {
        "title": {"text": "Enrolled Date", "x": 0.05, "xanchor": "left"},
        "xaxis": {"type": "category", "tickangle": -45},
        "yaxis": {"title": "Count", "fixedrange": True},
        "colorway": ["#c1e7ff"],
        "annotations": [
            {
                "text": "Enrolled Date indicates the date of participant enrollment.",
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": 1.06, # Adjust the y-coordinate to position the footnote below the title
                "showarrow": False,
                "font": {"size": 11},
            },
        ],
        "margin": {"t": 50, "r": 10, "b": 80, "l": 60}, # Adjust the margins as needed
    },
}



## ARIMA - enrollment projection
    # Get all enrollment counts for each month
    all_date_counts = data_2022["PtDatabase::EnrollmentDate"].dt.to_period('M').value_counts().sort_index()
    # Generate a range of months from the minimum to maximum dates
    months = pd.period_range(min_date, max_date, freq='M')
    # Fill in the count values for months with zero
    for i, month in enumerate(months):
        if month not in all_date_counts.index:
            all_date_counts[month]=0
    # Resort months
    all_date_counts=all_date_counts.sort_index()
    # Map individual counts to running sum of enrollment counts
    all_date_counts[0]+=enrolled_date_counts_2021
    for i in range(1,  len(all_date_counts)):
        all_date_counts[i]=all_date_counts[i]+all_date_counts[i-1]
    # Create arima model and forecast
    model = ARIMA(all_date_counts, order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=3)
# graph
    arima_enrollment_chart_figure = {
        "data": [
        {
        "x": all_date_counts.index.to_timestamp(),
        "y": all_date_counts.values,
        "mode": "lines",
        "name": "Actual Enrollment",
        "hovertemplate": "Date: %{x}<br>Enrollment: %{y}<extra></extra>",
        },
        {
        "x": forecast.index.to_timestamp(),
        "y": np.round(forecast.values),
        "mode": "lines",
        "name": "Enrollment Forecast",
        "hovertemplate": "Date: %{x}<br>Forecasted Enrollment: %{y}<extra></extra>",
        },
        ],
        "layout": {
        "title": {"text": "Enrollment Forecast (ARIMA)", "x": 0.05, "xanchor": "left"},
        "xaxis": {"fixedrange": True},
        "yaxis": {"title": "Enrollment", "fixedrange": True},
        "annotations": [
            {
                "text": "Enrollment Forecast (ARIMA) predicts the enrolled counts for the next three months.",
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": 1.15, # Adjust the y-coordinate to position the footnote below the title
                "showarrow": False,
                "font": {"size": 11},
            },
        ],
        "colorway": ["#00B2FF", "#FF6F91"],
        },
    }

 ## Enrollment Count/Type chart
    enrollment_counts = filtered_data[filtered_data['Enrolled']]["EnrollmentType"].value_counts()
# graph
    enrollment_chart_figure = {
        "data": [
            {
                "x": enrollment_counts.index,
                "y": enrollment_counts.values,
                "type": "bar",
                "hovertemplate": "Enrollment Status: %{x}<br>Count: %{y}<extra></extra>",
                "text": enrollment_counts.values,
                "textposition": "auto",
                
            },
        ],
        "layout": {
            "title": {"text": "Enrollment Type", "x": 0.05, "xanchor": "left"},
            "xaxis": {"fixedrange": True},
            "yaxis": {"title": "Count", "fixedrange": True},
            "colorway": ["896978"], 
            "annotations": [
                {
                    "text": "Enrollment Type indicates the type of enrollment for participants.",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0,
                    "y": 1.15, # Adjust the y-coordinate to position the footnote below the title
                    "showarrow": False,
                    "font": {"size": 11},
                },
            ],
        },
    }


    ## PID Status chart
    pid_status_counts = filtered_data[filtered_data['Enrolled']]["PtDatabase::PIDStatus"].value_counts()
# graph
    pid_status_chart_figure = {
        "data": [
            {
                "x": pid_status_counts.index,
                "y": pid_status_counts.values,
                "type": "bar",
                "hovertemplate": "PID Status: %{x}<br>Count: %{y}<extra></extra>",
                "text": pid_status_counts.values,
                "textposition": "auto",
            },
        ],
        "layout": {
            "title": {"text": "PID Status", "x": 0.05, "xanchor": "left"},
            "xaxis": {"fixedrange": True},
            "yaxis": {"title": "Count", "fixedrange": True},
            "colorway": ["#FF8E4F"], 
            "annotations": [
                {
                    "text": "PID Status indicates the status of the Proportional-Integral-Derivative controller.",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0,
                    "y": 1.06, # Adjust the y-coordinate to position the footnote below the title
                    "showarrow": False,
                    "font": {"size": 11},
                },
            ],
            "margin": {"t": 50, "r": 50, "b": 100, "l": 60},  # Adjust the margins as needed
        },
    }

## Referral Source chart
    referral_source_counts = filtered_data["ReferralSource"].value_counts()
# graph
    referral_source_chart_figure = {
    "data": [
        {
            "x": referral_source_counts.index,
            "y": referral_source_counts.values,
            "type": "bar",
            "hovertemplate": "Referral Source: %{x}<br>Count: %{y}<extra></extra>",
            "text": referral_source_counts.values,
            "textposition": "auto",
        },
    ],
    "layout": {
        "title": {"text": "Referral Source", "x": 0.05, "xanchor": "left"},
        "xaxis": {"fixedrange": True, "tickangle": -45},
        "yaxis": {"title": "Count", "fixedrange": True},
        "colorway": ["#90D4D3"],
        "annotations": [
            {
                "text": "Referral Source indicates the source of participant referrals.",
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": 1.06,  # Adjust the y-coordinate to position the footnote below the title
                "showarrow": False,
                "font": {"size": 11},
            },
        ],
        "margin": {"t": 50, "r": 10, "b": 165, "l": 100},  # Adjust the margins as needed
    },
}
    

    # MOUDType for enrolled -  PtDatabase::EnrollmentDate
    moudtype_counts = filtered_data[filtered_data['Enrolled']]["MOUDType"].value_counts()
# graph
    moudtype_enrolled_graph = {
    "data": [
        {
            "x": moudtype_counts.index,
            "y": moudtype_counts.values,
            "type": "bar",
            "hovertemplate": "MOUD Type: %{x}<br>Count: %{y}<extra></extra>",
            "text": moudtype_counts.values,
            "textposition": "auto",
        },
    ],
    "layout": {
        "title": {"text": "MOUD Type for Enrolled Participants", "x": 0.05, "xanchor": "left"},
        "xaxis": {"fixedrange": True},
        "yaxis": {"title": "Count", "fixedrange": True},
        "colorway": ["#F9C217"],
        "annotations": [
            {
                "text": "MOUD Type indicates the type of Medication for Opioid Use Disorder for enrolled participants.",
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": 1.06,
                "showarrow": False,
                "font": {"size": 11},
            },
        ],
        "margin": {"t": 50, "r": 10, "b": 80, "l": 60},  # Adjust the margins as needed
    },
}


# OUDScore - bar graph (for each of the scores (how many per score value))
    oudscore_counts = filtered_data["OUDScore"].value_counts()
# graph
    oudscore_graph = {
    "data": [
        {
            "x": oudscore_counts.index,
            "y": oudscore_counts.values,
            "type": "bar",
            "hovertemplate": "OUD Score: %{x}<br>Count: %{y}<extra></extra>",
            "text": oudscore_counts.values,
            "textposition": "auto",
        },
    ],
    "layout": {
        "title": {"text": "OUD Score Distribution", "x": 0.05, "xanchor": "left"},
        "xaxis": {
            "title": "OUD Score",
            "fixedrange": True,
            "tickmode": "array",  # Set tick mode to "array"
            "tickvals": list(range(12)),  # Explicitly provide the tick values from 0 to 11
        },
        "yaxis": {"title": "Count", "fixedrange": True},
        "colorway": ["#003F5B"],
        "margin": {"t": 50, "r": 10, "b": 80, "l": 60},
        "annotations": [
            {
                "text": "OUD Score Distribution represents the count of each OUD Score value.",
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": 1.06,
                "showarrow": False,
                "font": {"size": 11},
            }
        ],
    },
}


# DaysIncarcerated 
    # Filter the data absolute values (no negative)
    filtered_days_incarcerated = filtered_data["DaysIncarcerated"].copy()
    filtered_days_incarcerated = abs(filtered_days_incarcerated)
    #filtered_days_incarcerated[filtered_days_incarcerated > 1095] = 1095 # Outliers above 3 years are brought to the top of the box plot
# graph
    days_incarcerated_graph = {
    "data": [
        {
            "y": filtered_days_incarcerated,
            "type": "box",
            "hovertemplate": "Days Incarcerated: %{y}<extra></extra>",
            "box": {
                "fillcolor": "#00B2FF",
                "line": {"color": "#163F5A", "width": 2}
            },
            "marker": {"color": "#163F5A"},
            "boxpoints": "all"
        },
    ],
    "layout": {
        "title": {
            "text": "Days Incarcerated for All Participants",
            "x": 0.05,
            "xanchor": "left",
            "font": {"size": 16},
        },
        "xaxis": {"title": "Participants", "fixedrange": True},
        "yaxis": {"title": "Days Incarcerated (Log Scale)", "type": "log", "fixedrange": True},
        "colorway": ["#00B2FF"],
        "margin": {"t": 50, "r": 10, "b": 80, "l": 60},
        "annotations": [
            {
                "text": "Days Incarcerated indicates the number of days spent in jail for all participants.",
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": 1.06,
                "showarrow": False,
                "font": {"size": 11},
            }
        ],
    },
}



## Cards ##

    ## Update age mean card
    age_mean = round(filtered_data["Age"].mean())

    ## Update race card
    race_counts = filtered_data["Race"].value_counts()
    race_lines = [
        html.Div(f"{race}: {count}", style={"marginBottom": "5px"})
        for race, count in race_counts.items()
    ]

    ## Update gender card
    gender_counts = filtered_data["Gender"].value_counts()
    gender_lines = [
        html.Div(f"{gender}: {count}", style={"marginBottom": "5px"})
        for gender, count in gender_counts.items()
    ]
   
    ## Update conversion rate card
    # Calculate the total number of screened participants
    total_screened = len(filtered_data["ScreeningDate"].dropna())

    # Calculate the number of participants who converted from screening to enrollment
    converted_participants = len(filtered_data[(filtered_data["Enrolled"]) & (filtered_data["ScreeningDate"].notna())])

    # Calculate the conversion rate and round it to 0 decimal places
    conversion_rate = round((converted_participants / total_screened) * 100)

    # Update conversion rate card
    conversion_rate_card_value = f"{conversion_rate}%"


    # Update site card
    site_counts = filtered_data[filtered_data['Enrolled']]["Site"].value_counts()
    total_enrollment = site_counts.sum()  # Calculate total enrollment based on site counts

    site_count_text = [
    html.Div(f"{site}: {count}", style={"marginBottom": "5px"})
    for site, count in site_counts.items()
    ]

    # Add total enrollment to the site count card
    site_count_text.append(
     html.Div(f"Total: {total_enrollment}", style={"marginBottom": "5px"})
    )

## Create a card for the ARIMA enrollment projections
    arima_enrollment_card = [
        html.Div(f"{idx.to_timestamp().strftime('%Y-%m')}: {round(forecast[idx])}", style={"marginBottom": "5px"})
        for idx in forecast.index
    ]



# Return the graphs and cards 
    return (
        screening_date_chart_figure,
        enrollment_chart_figure,
        arima_enrollment_chart_figure,
        referral_source_chart_figure,
        enrolled_date_chart_figure,
        pid_status_chart_figure,
        moudtype_enrolled_graph,
        days_incarcerated_graph,
        oudscore_graph,
        f"{age_mean:}",
        race_lines,
        gender_lines,
        site_count_text,
        conversion_rate_card_value,
        arima_enrollment_card
    )

# Function for the blank graph  when there are no matching records
def blank_figure():
    return {"data": [], "layout": {}}

# Print the local URL
if __name__ == "__main__":
    app.run_server(debug=True, port=8052)
    print("Running on http://127.0.0.1:8052/")

