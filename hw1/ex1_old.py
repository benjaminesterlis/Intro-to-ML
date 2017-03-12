from numpy import linalg as LA
def K_NNeighbor(ImageSet,labelsVector,QImg ,k):
	k_nearest = []
	k_index=[]
	i = 0
	label = 0
	for img in ImageSet:
		k_nearest.append(img)
		k_index.append(i)
		i+=1
		if(i > k-1):
			index = norm_max(k_nearest,QImg)
			k_index.pop(index-1)
	for x in k_index:
		label += labelsVector[x]
	return (int)(label/k+0.5)

def norm_max(lst, vec): #len(lst)=k
	i = 0
	index = 0
	max_vec = lst[0]
	for l in lst:
		i+=1
		if LA.norm(max_vec-vec) < LA.norm(l-vec):
			max_vec = l
			index = i
	#print(max_vec , len(max_vec) , index)
	lst.pop(index-1)
	return index
