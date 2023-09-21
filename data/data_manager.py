import os, pandas as pd

x_files = ['data/X_port_angeles.csv', 'data/X_pittsburgh.csv', 'data/X_tucson.csv', 'data/X_new_york.csv']
y_files = ['data/Y_port_angeles.csv', 'data/Y_pittsburgh.csv', 'data/Y_tucson.csv', 'data/Y_new_york.csv']
cities = ['Port Angeles, WA', 'Pittsburgh, PA', 'Tucson, AZ', 'New York, NY']

def make_deep_ensemble_directories_with_pre(pre_, sample_size=None):
    os.mkdir(pre_+'deep_ensemble_models_{}'.format(sample_size)) if not os.path.exists(pre_+'deep_ensemble_models_{}'.format(sample_size)) else None
    os.mkdir(pre_+'deep_ensemble_models_{}/port_angeles'.format(sample_size)) if not os.path.exists(pre_+'deep_ensemble_models_{}/port_angeles'.format(sample_size)) else None
    os.mkdir(pre_+'deep_ensemble_models_{}/pittsburgh'.format(sample_size)) if not os.path.exists(pre_+'deep_ensemble_models_{}/pittsburgh'.format(sample_size)) else None
    os.mkdir(pre_+'deep_ensemble_models_{}/tucson'.format(sample_size)) if not os.path.exists(pre_+'deep_ensemble_models_{}/tucson'.format(sample_size)) else None
    os.mkdir(pre_+'deep_ensemble_models_{}/new_york'.format(sample_size)) if not os.path.exists(pre_+'deep_ensemble_models_{}/new_york'.format(sample_size)) else None
    os.mkdir(pre_+'deep_ensemble_models_{}/port_angeles/summer'.format(sample_size)) if not os.path.exists(pre_+'deep_ensemble_models_{}/port_angeles/summer'.format(sample_size)) else None
    os.mkdir(pre_+'deep_ensemble_models_{}/port_angeles/winter'.format(sample_size)) if not os.path.exists(pre_+'deep_ensemble_models_{}/port_angeles/winter'.format(sample_size)) else None
    os.mkdir(pre_+'deep_ensemble_models_{}/pittsburgh/summer'.format(sample_size)) if not os.path.exists(pre_+'deep_ensemble_models_{}/pittsburgh/summer'.format(sample_size)) else None
    os.mkdir(pre_+'deep_ensemble_models_{}/pittsburgh/winter'.format(sample_size)) if not os.path.exists(pre_+'deep_ensemble_models_{}/pittsburgh/winter'.format(sample_size)) else None
    os.mkdir(pre_+'deep_ensemble_models_{}/tucson/summer'.format(sample_size)) if not os.path.exists(pre_+'deep_ensemble_models_{}/tucson/summer'.format(sample_size)) else None
    os.mkdir(pre_+'deep_ensemble_models_{}/tucson/winter'.format(sample_size)) if not os.path.exists(pre_+'deep_ensemble_models_{}/tucson/winter'.format(sample_size)) else None
    os.mkdir(pre_+'deep_ensemble_models_{}/new_york/summer'.format(sample_size)) if not os.path.exists(pre_+'deep_ensemble_models_{}/new_york/summer'.format(sample_size)) else None
    os.mkdir(pre_+'deep_ensemble_models_{}/new_york/winter'.format(sample_size)) if not os.path.exists(pre_+'deep_ensemble_models_{}/new_york/winter'.format(sample_size)) else None

def make_deep_ensemble_directories(sample_size=None):
    if sample_size is None:
        os.mkdir('deep_ensemble_models') if not os.path.exists('deep_ensemble_models') else None
        os.mkdir('deep_ensemble_models/port_angeles') if not os.path.exists('deep_ensemble_models/port_angeles') else None
        os.mkdir('deep_ensemble_models/pittsburgh') if not os.path.exists('deep_ensemble_models/pittsburgh') else None
        os.mkdir('deep_ensemble_models/tucson') if not os.path.exists('deep_ensemble_models/tucson') else None
        os.mkdir('deep_ensemble_models/new_york') if not os.path.exists('deep_ensemble_models/new_york') else None
        os.mkdir('deep_ensemble_models/port_angeles/summer') if not os.path.exists('deep_ensemble_models/port_angeles/summer') else None
        os.mkdir('deep_ensemble_models/port_angeles/winter') if not os.path.exists('deep_ensemble_models/port_angeles/winter') else None
        os.mkdir('deep_ensemble_models/pittsburgh/summer') if not os.path.exists('deep_ensemble_models/pittsburgh/summer') else None
        os.mkdir('deep_ensemble_models/pittsburgh/winter') if not os.path.exists('deep_ensemble_models/pittsburgh/winter') else None
        os.mkdir('deep_ensemble_models/tucson/summer') if not os.path.exists('deep_ensemble_models/tucson/summer') else None
        os.mkdir('deep_ensemble_models/tucson/winter') if not os.path.exists('deep_ensemble_models/tucson/winter') else None
        os.mkdir('deep_ensemble_models/new_york/summer') if not os.path.exists('deep_ensemble_models/new_york/summer') else None
        os.mkdir('deep_ensemble_models/new_york/winter') if not os.path.exists('deep_ensemble_models/new_york/winter') else None
    else:
        os.mkdir('deep_ensemble_models_{}'.format(sample_size)) if not os.path.exists('deep_ensemble_models_{}'.format(sample_size)) else None
        os.mkdir('deep_ensemble_models_{}/port_angeles'.format(sample_size)) if not os.path.exists('deep_ensemble_models_{}/port_angeles'.format(sample_size)) else None
        os.mkdir('deep_ensemble_models_{}/pittsburgh'.format(sample_size)) if not os.path.exists('deep_ensemble_models_{}/pittsburgh'.format(sample_size)) else None
        os.mkdir('deep_ensemble_models_{}/tucson'.format(sample_size)) if not os.path.exists('deep_ensemble_models_{}/tucson'.format(sample_size)) else None
        os.mkdir('deep_ensemble_models_{}/new_york'.format(sample_size)) if not os.path.exists('deep_ensemble_models_{}/new_york'.format(sample_size)) else None
        os.mkdir('deep_ensemble_models_{}/port_angeles/summer'.format(sample_size)) if not os.path.exists('deep_ensemble_models_{}/port_angeles/summer'.format(sample_size)) else None
        os.mkdir('deep_ensemble_models_{}/port_angeles/winter'.format(sample_size)) if not os.path.exists('deep_ensemble_models_{}/port_angeles/winter'.format(sample_size)) else None
        os.mkdir('deep_ensemble_models_{}/pittsburgh/summer'.format(sample_size)) if not os.path.exists('deep_ensemble_models_{}/pittsburgh/summer'.format(sample_size)) else None
        os.mkdir('deep_ensemble_models_{}/pittsburgh/winter'.format(sample_size)) if not os.path.exists('deep_ensemble_models_{}/pittsburgh/winter'.format(sample_size)) else None
        os.mkdir('deep_ensemble_models_{}/tucson/summer'.format(sample_size)) if not os.path.exists('deep_ensemble_models_{}/tucson/summer'.format(sample_size)) else None
        os.mkdir('deep_ensemble_models_{}/tucson/winter'.format(sample_size)) if not os.path.exists('deep_ensemble_models_{}/tucson/winter'.format(sample_size)) else None
        os.mkdir('deep_ensemble_models_{}/new_york/summer'.format(sample_size)) if not os.path.exists('deep_ensemble_models_{}/new_york/summer'.format(sample_size)) else None
        os.mkdir('deep_ensemble_models_{}/new_york/winter'.format(sample_size)) if not os.path.exists('deep_ensemble_models_{}/new_york/winter'.format(sample_size)) else None

port_angeles_winter_x, port_angeles_winter_y = pd.read_csv(x_files[0]), pd.read_csv(y_files[0])
port_angeles_summer_x, port_angeles_summer_y = pd.read_csv(x_files[0])[20544:], pd.read_csv(y_files[0])[20544:]
pittsburgh_winter_x, pittsburgh_winter_y = pd.read_csv(x_files[1]), pd.read_csv(y_files[1])
pittsburgh_summer_x, pittsburgh_summer_y = pd.read_csv(x_files[1])[20544:], pd.read_csv(y_files[1])[20544:]
tucson_winter_x, tucson_winter_y = pd.read_csv(x_files[2]), pd.read_csv(y_files[2])
tucson_summer_x, tucson_summer_y = pd.read_csv(x_files[2])[20544:], pd.read_csv(y_files[2])[20544:]
new_york_winter_x, new_york_winter_y = pd.read_csv(x_files[3]), pd.read_csv(y_files[3])
new_york_summer_x, new_york_summer_y = pd.read_csv(x_files[3])[20544:], pd.read_csv(y_files[3])[20544:]

port_angeles_year_x = pd.read_csv(x_files[0])
port_angeles_year_y = pd.read_csv(y_files[0])
pittsburgh_year_x = pd.read_csv(x_files[1])
pittsburgh_year_y = pd.read_csv(y_files[1])
tucson_year_x = pd.read_csv(x_files[2])
tucson_year_y = pd.read_csv(y_files[2])
new_york_year_x = pd.read_csv(x_files[3])
new_york_year_y = pd.read_csv(y_files[3])

data_tree = {
            'port_angeles': {
                'summer': (port_angeles_summer_x, port_angeles_summer_y), 
                'winter': (port_angeles_winter_x, port_angeles_winter_y)
                },
            'pittsburgh': {
                'summer': (pittsburgh_summer_x, pittsburgh_summer_y),
                'winter': (pittsburgh_winter_x, pittsburgh_winter_y),
                'all_year': (pittsburgh_year_x, pittsburgh_year_y)
                },
            'tucson': {
                'summer': (tucson_summer_x, tucson_summer_y),
                'winter': (tucson_winter_x, tucson_winter_y),
                'all_year': (tucson_year_x, tucson_year_y)
                },
            'new_york': {
                'summer': (new_york_summer_x, new_york_summer_y),
                'winter': (new_york_winter_x, new_york_winter_y),
                'all_year': (new_york_year_x, new_york_year_y)
                }
}

environment_var = ['Site Outdoor Air Drybulb Temperature(Environment)', 'Site Outdoor Air Relative Humidity(Environment)', 'Site Wind Speed(Environment)', 'Site Direct Solar Radiation Rate per Area(Environment)', 'Zone People Occupant Count(SPACE1-1)']

def get_environment_forecast(city, season='all_year'):
    data = data_tree[city][season]
    data = data[0]
    data = data[environment_var]
    return data
