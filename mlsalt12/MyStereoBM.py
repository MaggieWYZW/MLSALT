import numpy as np

class myStereoBM( numDisparity, blockSize):
	def __init__(self, numDisparity, blockSize):
		self.numDisparity=numDisparity
		self.blockSize=blockSize
		self.leftImage=[]
		self.rightImage=[]

	def computeCost(leftWindow, rightWindow):
		row, col = np.shape(leftWindow)[0], np.shpae(leftImage)[1]
		MAD = np.sum(np.absolute[leftWindow-rightWindow], dtype=np.float32)/(row*col)  # Mean Absolute Difference
		return MAD


	def locateMinCost(row, col):
		min_cost = 1000
		windowSize = self.blockSize-1
		distance = np.floor(self.blockSize)
		
		if col-self.numDisparity < 0:
			left_limit = 0
			right_limit = col+self.numDisparity
		elif col+self.numDisparity > np.shape(self.leftImage)[1]:
			left_limit = col-self.numDisparity
			right_limit = np.shape(self.leftImage)[1]
		else:
			left_limit = col-self.numDisparity
			right_limit = col+self.numDisparity

		w = row
		for h in range(left_limit: right_limit):
			if w < windowSize/2 and h < windowSize/2 :
				leftWindow = self.leftImage[0:w+distance][0:h+distance]
				rightWindow = self.rightImage[0:w+distance][0:h+distance]
			elif w > width - windowSize/2 and h < windowSize/2 :
				leftWindow = self.leftImage[w-distance:row-1][0:h+distance]
				rightWindow = self.rightImage[w-distance:row-1][0:h+distance]
			elif w < windowSize/2 and h > height - windowSize/2 :
				leftWindow = self.leftImage[0:w+distance][h-distance:col-1]
				rightWindow = self.rightImage[0:w+distance][h-distance:col-1]
			elif w > width - windowSize/2 and h > height - windowSize/2:
				leftWindow = self.leftImage[w-distance:row-1][h-distance:col-1]
				rightWindow = self.rightImage[w-distance:row-1][h-distance:col-1]
			else:
				leftWindow = self.leftImage[w-distance:w+distance][h-distance:h+distance]
				rightWindow = self.rightImage[w-distance:w+distance][h-distance:h+distance]

			cost = self.computeCost(leftWindow, rightWindow)
			if cost < min_cost:
				min_cost = cost
				location = [w, h]

		return location[0], location[1], min_cost


	def computeDisparity(self, leftImage, rightImage):
		self.leftImage=leftImage
		self.rightImage=rightImage

		num_row, num_col = np.shape(self.leftImage)[0], np.shape(self.leftImage)[1]
		for left_y in range(num_row):
			for left_x in range(num_col):
				right_y, right_x = locateMinCost(row, col)
				disparity = left_x - right_x
				disparity_map[left_y, left_x] = 255*disparity/self.numDisparity

		return disparity_map

