import ConfigParser

def read_config(config_path):
    config = ConfigParser.ConfigParser()
    config.read(config_path)
    sections=config.sections()
    print(sections)
    items=[]
    for section_i in sections:
    	items+=config.items(section_i)
    if(len(items)==0):
        raise Exception('No items in ' + str(config_path))
    print(items)
    conf_dir=dict([ list(pair_i) for pair_i in items])	
    return conf_dir