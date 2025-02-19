

lines = open('/home/garvit/Downloads/submission.csv', 'r').readlines()[1:]

correctfirst=0
correctsecond=0
correctthird=0
for line in lines:
	splitline = line.split(',')
	correct = splitline[2][0]
	first = splitline[1][0]
	second = splitline[1][2]
	third = splitline[1][4]
	
	if(first == correct): correctfirst+=1
	elif(second == correct): correctsecond+=1
	elif(third == correct): correctthird+=1
	
	
print(correctfirst)
print(correctsecond)
print(correctthird)
