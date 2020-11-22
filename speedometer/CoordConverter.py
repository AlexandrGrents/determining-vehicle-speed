class CoordConverter(object):
	def __init__(self, zones, borders):
		self.zones = zones
		self.borders = borders

	def get_zone_number(self, x, y):
		i = 0
		for borders in self.borders:
			b, k = borders['b'], borders['k']
			if y < k * x + b:
				return i
			i += 1
		return i

	def convert(self, x, y, de = 3):
		"""x, y - точка для перевода, de - требуемая точность"""
		a,b,c,d,e,f,g,h = self.zones[self.get_zone_number(x, y)]
		denominator = g*x+h*y+1
		u = round((a*x+b*y+c)/denominator, de)
		v = round((d*x+e*y+f)/denominator, de)
		return u, v

	def calc_converted_dist(self, pos1, pos2, de = 7):
		if pos1[0] == pos2[0] and pos1[1] == pos2[1]:
			return 0
		u1, v1 = self.convert(*pos1)
		u2, v2 = self.convert(*pos2)
		dist = ((u1-u2)**2+(v1-v2)**2)**(0.5)
		return round(dist,de)