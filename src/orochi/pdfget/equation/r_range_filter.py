

def apply_r_filter(self,q,i):
	def smartfilter(sq):	
		sq1 = sq.copy()
		r1 = np.where( q < self.par["fitmax"] )	
		r2 = np.where( q < self.par["fitmin"] )
		sq1[r1] = 0
		
		weight = np.ones_like(sq1)
		weight[r2] = 1000. 
		weight[-1] = 1000.
		
		n = int(q[-1]/4)
		highq = np.poly1d( np.polyfit(q, sq1, n, w=weight))(q)
		
		sq -= highq
		sqd = high_pass(q, sq)

		return highq + sqd	
		
	def high_pass(q, sq):
		pad = 3		
		fq = sq * q			
		rho0 = float(self["number_density"]) if "number_density" in self.par else float(self.par.get("ndensity", {}).get("value", [0.0])[0])
		baseline = -4.0 * pi * rho0 * r
		range = np.where(r < self.par["rfilter"])
		g[range] = baseline[range]
		
		N = q.size
		fqt = self.idst_transform(q, g, pad)[0:N]		

		sqd = (fq[1:] - fqt[1:]) / q[1:]
		sqd = np.append(0, sqd)
			
		return sqd
	
	def scalefit(q, r, ratio):
		scale = np.abs(np.average(ratio[r]))
		a0 = 1.*scale
		a_min = 0.1*scale
		a_max = 10.0*scale
		
		def func(x, a1):	
			return a1 * q[r]				
		p1, p2 = curve_fit(func, q[r], ratio[r] * q[r], p0=[a0], bounds=([a_min,],[a_max,]),maxfev=100)		
		return p1

	i = i - self.baseline - self.compton	
		
	if self.par["rfilter"] >0.:
		sq = (i - self.f_sq_ave) / self.f_ave_sq
		
		dd = smartfilter(sq) * self.f_ave_sq
		i = i - dd
		self.baseline += dd
		
		ratio = i/self.f_sq_ave
		scale = scalefit(q, self.fitrange, ratio)
		
		i = i / scale	
		
	return i