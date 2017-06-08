import codecs
def write_to_result(x,y,out1,out2):
  x = x.replace("...","")
  x = x.replace("="," ")
  x = x.replace("\t"," ")
  y = y.replace("...","")
  y = y.replace("\t"," ")
  y = y.replace("="," ")

  out1.write(x+"\n")
  out2.write(y+"\n")

f = open("../data/new/movie_lines.txt","r",encoding="utf8")
line = f.readline()
mylines = {}
while line:
  t = line.strip().split(" +++$+++ ")
  if (len(t)>4):
    mylines[t[0]] = t[4]
  line = f.readline()
f.close()

f = open("../data/new/movie_conversations.txt","r")
out1 = open("../data/new/source.txt","w")
out2 = open("../data/new/target.txt","w")
line = f.readline()
while line:
  t = line.strip().split(" +++$+++ ")
  con = eval(t[3])
  p=True
  for i in range(0,len(con)):
    if (con[i] not in mylines):
      p=False
  if (p):
    last = mylines[con[0]]
    for i in range(1,len(con)):
      write_to_result(last,mylines[con[i]],out1,out2)
      last = mylines[con[i]]
  line = f.readline()
f.close()
out1.close()
out2.close()
