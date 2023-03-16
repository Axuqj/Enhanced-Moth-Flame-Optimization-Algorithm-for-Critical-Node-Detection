
import xlrd


Name=['MOFCNDP','GCNP','greedy3d','greedy4d','HDCN','KSBG','LRBG','BCB']	
network=['ER','BA','WS']	

Rresult={name:{'ER':[[[],0],[[],0],[[],0],[[],0]],'BA':[[[],0],[[],0],[[],0],[[],0]],'WS':[[[],0],[[],0],[[],0],[[],0]]}  for name in Name}

data = xlrd.open_workbook(r'C:\\Users\\xqjwo\\Desktop\\new artical\\1.xls')
table = data.sheets()[0]

line=range(2,10)
row=range(110,122)

for i in range(len(line)): #列，也是算法序号
    for j in range(len(row)):
        ntw= network[int(j/4)]
        no=j%4
        Rresult[Name[i]][ntw][no][0] = int(table.cell_value(row[j],line[i]))

print(Rresult)
