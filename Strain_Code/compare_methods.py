# A set of code that reads multiple grid files and produces mean and variance
# In grid form. 

import numpy as np
import netcdf_functions
# import netCDF4

# takes in netcdfs from each method of strain calculation, and produces netcdfs of the means and standard deviations

# inputs data from netcdfs. netcdfs must have uniform grid size, and generally cover northern california
# outputs an n-dim array of lons, and m-d array of lats, and an m by n array of values
# def input_netcdf(nc):
# 	file = netCDF4.Dataset(nc, 'r', Format='NETCDF4')
# 	lon = file.variables['x'][:]
# 	lat = file.variables['y'][:]
# 	values = file.variables['z']
# 	return lon, lat, values

def input_netcdf(nc):  # the older netcdf functions. 
	[lon, lat, values] = netcdf_functions.read_grd_xyz(nc, "x", "y", "z");
	return lon, lat, values;



# inputs data from txt files
def input_txt(file):
	ifile = open(file, 'r');
	x = []; y = []; val = [];
	for line in ifile:
		temp = line.split()
		x.append(float(temp[0]))
		y.append(float(temp[1]))
		val.append(float(temp[2]))
	ifile.close()

	return x, y, val

# uses uniform data points from net cdfs with different coord boxes, and confines it to a uniform coord box
# all points must be coregistered even if boxes are different sizes
# i.e. the starting values must be different by multiples of the incriment
# this is controlled by configure_functions and produce_gridded, depending on the method.
def confine_to_grid(x, y, values, xmin, xmax, ymin, ymax, inc):
	print("registering %s to grid [%.2f %.2f %.2f %.2f %.2f] " % ("component", xmin, xmax, ymin, ymax, inc) );

	new_lon = np.arange(xmin, xmax, inc)
	new_lat = np.arange(ymin, ymax, inc)
	new_vals = []

	x = np.around(x, 5)
	y = np.around(y, 5)
	tol = 0.002

	for i in range(len(x)):
		for j in range(len(y)):
			if (xmin < x[i] < xmax) or (abs((x[i] - xmin)) < tol) or (abs((x[i] - xmax)) < tol):
				if (ymin < y[j] < ymax) or  (abs((y[j] - ymin)) < tol) or (abs((y[j] - ymax)) < tol):
					new_vals.append(values[j][i])

	final_vals = []
	for i in np.arange(0, len(new_vals), len(new_lat)):
		final_vals.append(new_vals[i:i+len(new_lat)])
	final_vals = np.transpose(final_vals)

	return new_lon, new_lat, final_vals

# this function makes sure arrays are of the same dimensions before attempting to produce any statistics
def check_coregistration(v1, v2, v3, v4, v5):
	if np.shape(v1) != np.shape(v2):
		print("\n   Oops! The shape of method 1 vals is %d by %d \n" % (np.shape(v1)[0], np.shape(v1)[1] ) );
		print("   But shape of method 2 vals is %d by %d " % (np.shape(v2)[0], np.shape(v2)[1] ) );
		print("   so not all value arrays are coregistered! \n")
	elif np.shape(v1) != np.shape(v3):
		print("\n   Oops! The shape of method 1 vals is %d by %d \n" % (np.shape(v1)[0], np.shape(v1)[1] ) );
		print("   But shape of method 3 vals is %d by %d \n" % (np.shape(v3)[0], np.shape(v3)[1] ) );
		print("   so not all value arrays are coregistered! \n")
	elif np.shape(v1) != np.shape(v4):
		print("\n   Oops! The shape of method 1 vals is %d by %d \n" % (np.shape(v1)[0], np.shape(v1)[1] ) );
		print("   But shape of method 4 vals is %d by %d \n" % (np.shape(v4)[0], np.shape(v4)[1] ) );
		print("   so not all value arrays are coregistered! \n")
	elif np.shape(v1) != np.shape(v5):
		print("\n   Oops! The shape of method 1 vals is %d by %d \n" % (np.shape(v1)[0], np.shape(v1)[1] ) );
		print("   But shape of method 5 vals is %d by %d " % (np.shape(v5)[0], np.shape(v5)[1] ) );
		print("   so not all value arrays are coregistered! \n")
	else:
		print("All methods are coregistered!")
	return

# gridwise, calculates means and standard deviations, and returns them as arrays with dimension latitude by longitude
def grid_means(x, y, vals1, vals2, vals3, vals4, vals5):
	mean_vals = np.nan * np.ones([len(y), len(x)])
	for j in range(len(y)):
		for i in range(len(x)):
			val1 = vals1[j][i]
			val2 = vals2[j][i]
			val3 = vals3[j][i]
			val4 = vals4[j][i]
			val5 = vals5[j][i]
			mean_val = np.nanmean([val1, val2, val3, val4, val5])
			if mean_val != float("-inf"):
				mean_vals[j][i] = mean_val
	return mean_vals

def grid_sds(x, y, vals1, vals2, vals3, vals4, vals5):
	sd_vals = np.nan * np.ones([len(y), len(x)])
	for j in range(len(y)):
		for i in range(len(x)):
			val1 = vals1[j][i]
			val2 = vals2[j][i]
			val3 = vals3[j][i]
			val4 = vals4[j][i]
			val5 = vals5[j][i]
			sd_val = np.nanstd([val1, val2, val3, val4, val5])
			sd_vals[j][i] = sd_val
	return sd_vals

def grid_means_log(x, y, vals1, vals2, vals3, vals4, vals5):
	mean_vals = np.nan * np.ones([len(y), len(x)])
	for j in range(len(y)):
		for i in range(len(x)):
			val1 = 10**vals1[j][i]
			val2 = 10**vals2[j][i]
			val3 = 10**vals3[j][i]
			val4 = 10**vals4[j][i]
			val5 = 10**vals5[j][i]
			mean_val = np.nanmean([val1, val2, val3, val4, val5])
			if mean_val != float("-inf"):
				mean_vals[j][i] = np.log10(mean_val)
	return mean_vals

def grid_sds_log(x, y, vals1, vals2, vals3, vals4, vals5):
	sd_vals = np.nan * np.ones([len(y), len(x)])
	for j in range(len(y)):
		for i in range(len(x)):
			val1 = 10**vals1[j][i]
			val2 = 10**vals2[j][i]
			val3 = 10**vals3[j][i]
			val4 = 10**vals4[j][i]
			val5 = 10**vals5[j][i]
			sd_val = np.nanstd([val1, val2, val3, val4, val5])
			sd_vals[j][i] = np.log10(sd_val)
	return sd_vals

def angle_means(x, y, vals1, vals2, vals3, vals4, vals5):
	mean_vals = np.nan * np.ones([len(y), len(x)])
	for j in range(len(y)):
		for i in range(len(x)):
			val1 = 2*np.radians(90-vals1[j][i])
			val2 = 2*np.radians(90-vals2[j][i])
			val3 = 2*np.radians(90-vals3[j][i])
			val4 = 2*np.radians(90-vals4[j][i])
			val5 = 2*np.radians(90-vals5[j][i])
			s = (sum((np.sin(val1), np.sin(val2), np.sin(val3), np.sin(val4), np.sin(val5))))/5
			c = (sum((np.cos(val1), np.cos(val2), np.cos(val3), np.cos(val4), np.cos(val5))))/5
			# R = (c**2 + s**2)**.5
			# t = np.arctan2(s, c)
			# strike = R*math.e**(math.i*t)
			strike = np.arctan2(s, c)/2
			theta = 90 - np.degrees(strike)
			if theta < 0:
				theta = 180 + theta
			elif theta > 180:
				theta = theta - 180
			if theta != float("-inf"):
				mean_vals[j][i] = theta
	return mean_vals

def angle_sds(x, y, vals1, vals2, vals3, vals4, vals5):
	sd_vals = np.nan * np.ones([len(y), len(x)])
	for j in range(len(y)):
		for i in range(len(x)):
			val1 = 2*np.radians(90-vals1[j][i])
			val2 = 2*np.radians(90-vals2[j][i])
			val3 = 2*np.radians(90-vals3[j][i])
			val4 = 2*np.radians(90-vals4[j][i])
			val5 = 2*np.radians(90-vals5[j][i])
			s = (sum((np.sin(val1), np.sin(val2), np.sin(val3), np.sin(val4), np.sin(val5))))/5
			c = (sum((np.cos(val1), np.cos(val2), np.cos(val3), np.cos(val4), np.cos(val5))))/5
			R = ((s**2 + c**2)**.5)
			V = 1-R
			# sd = np.degrees((2*V)**.5)
			sd = np.degrees((-2*np.log(R))**.5)/2
			# if sd != float("inf"):
			# 	sd_vals[j][i] = sd
			sd_vals[j][i] = sd
	return sd_vals

def array_means(x, y, vals1, vals2, vals4):
	means = np.zeros(len(x))
	for i in range(len(x)):
		val1 = vals1[i]
		val2 = vals2[i]
		val4 = vals4[i]
		mean = np.nanmean([val1, val2, val4])
		if mean != float("-inf"):
			means[i] = mean
	return means

def mask_by_value(outdir, grid1, grid2, cutoff_value):
	# grid1 = usually azimuth deviations
	# grid2 = usually I2nd
	lon1, lat1, val1 = input_netcdf(outdir+"deviations_"+grid1+".nc");
	lon2, lat2, val2 = input_netcdf(outdir+"means_"+grid2+".nc");
	masked_vals=np.zeros(np.shape(val2));
	for i in range(len(lon1)):
		for j in range(len(lat1)):
			if abs(val2[j][i])>cutoff_value:
				masked_vals[j][i]=val1[j][i];
			else:
				masked_vals[j][i]=np.nan;
	output_nc(lon1, lat1, masked_vals, outdir, "deviations",grid1);
	return;

# writes the uniform latitude, longitude, and whichever statistical values are desired.
# outputs to result directory for means as a netcdf which can then be manipulated further with gmt.
# stat = "means" or "deviations"
# component = "I2nd", "max_shear", "dilatation", "azimuth"
def output_nc(lon, lat, vals, outdir, stat, component):
	netcdf_functions.produce_output_netcdf(lon, lat, vals, 'per yr', outdir+stat+"_"+component+".nc");
	return

def output_txt(lon, lat, vals, outdir, stat, component):
	upfile=open(outdir+stat+"_"+component_name+".txt", 'w');
	for i in range(len(lon)):
		upfile.write("%f %f %f \n" % (lon[i], lat[i], vals[i]));
	return;

