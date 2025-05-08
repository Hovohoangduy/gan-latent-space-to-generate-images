import yaml

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config