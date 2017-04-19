def action_pairs(actions):
    pairs=[]
    for action_i in actions:
        pairs+=action_i.to_pairs()
    return pairs

def to_dataset(pairs):
    y=[ pair_i[0] 
        for pair_i in pairs]
    X=[ pair_i[1] 
        for pair_i in pairs]
    X=np.array(X)
    return X,y	