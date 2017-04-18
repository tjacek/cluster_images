def action_pairs(actions):
    pairs=[]
    for action_i in actions:
        pairs+=action_i.to_pairs()
    return pairs	