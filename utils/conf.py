import ConfigParser

def read_config(config_path):
    config = ConfigParser.ConfigParser()
    config.read(config_path)
    conf=config.items("Cluster")
    conf=dict([ list(pair_i) for pair_i in conf])
    return conf