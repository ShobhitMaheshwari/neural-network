class Imagetools:
	def __init__(self):
		pass

	def difference(self, image1, image2):
		if(image1.shape != image2.shape):
			raise ValueError("Dimensions not equal")
		count = 0
		for x in range(0, len(image1.shape[0])):
			for y in range(0, len(image1.shape[1])):
				count += abs(image1[x][y] - image2[x][y])
		return count