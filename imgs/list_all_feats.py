import os

def make_csv(filename):
    with open(filename, "w") as f:
        for folder in os.walk("./"):
            if "feat" in folder[0] and "features" not in folder[0]: # If we are looking at a folder.
                num_feats = max([int(string[:-4]) for string in folder[2]])+1
                f.write(folder[0][7:] + ", " + str(num_feats) + "\n")

def make_js_file_line(filename): # TODO this is a janky hack
    feats = {}
    for folder in os.walk("./"):
        if "feat" in folder[0] and "features" not in folder[0]: # If we are looking at a folder.
            num_feats = max([int(string[:-4]) for string in folder[2]])+1
            feats[str(folder[0][7:])] = num_feats
    
    out_str = "const features = {" + ", ".join([str(feature)[5:]+": "+str(feats[feature]) for feature in feats]) + "};"
    with open(filename, "w") as f:
        f.write(out_str)

if __name__ == "__main__":
    # make_csv("imgs/features.csv")
    make_js_file_line("search2.js")