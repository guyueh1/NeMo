import os, sys

infile = sys.argv[1]

errors = {}
with open(infile, 'r') as inf:
    s = inf.readline()
    while len(s):
        if 'torch._dynamo.exc.Unsupported:' in s:
            s = s[:-1]
            if s in errors:
                errors[s] += 1
            else:
                errors[s] = 1
        s = inf.readline()
    
outfile = os.path.join(
    os.path.dirname(infile),
    "error.log"
)

with open(outfile, 'w') as outf:
    errors = sorted(errors.items(), key=lambda x: x[1], reverse=True)
    for k,v in errors:
        line = k + "," + str(v) + '\n'
        outf.write(line)