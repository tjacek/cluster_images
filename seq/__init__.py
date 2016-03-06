import utils.files as files
ALPH="ABCDEFGHIJKLMN"

def create_seqs(actions,out_path):
    seqs=[action_to_seq(action_i) for action_i in actions]
    seq_txt=files.array_to_txt(seqs,sep="\n")
    files.save_string(out_path,seq_txt)
        
def action_to_seq(action):
    seq=""
    for cat_i in action.seq:
        seq+=ALPH[cat_i]
    seq+="#"+str(action.cat)
    seq+="#"+str(action.name)
    return seq

def get_cat(symbol):
	return ALPH.index(symbol)