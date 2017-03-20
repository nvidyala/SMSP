import preprocessing
import matplotlib.pyplot as plt

def metric_evaluation():

	ticker_ds = ['WMT']
	N = len(ticker_ds)
	S = 3 #sample size
	rmse = [[] for i in range(S)]
	r2_values = [[] for i in range(S)]


	for ticker in ticker_ds:
		print("Ticker value: %s" %(ticker))
		
		#local array rmse_test stores average rmse values of models per ticker
		rmse_test = [0 for i in range(S)]
		r2_test = []
		residual = [0 for i in range(4)]


		#represents samples per ticker
		for n in range(S):
			sample,r2,residual = preprocessing.main(ticker)
			rmse_test[0] += sample['pred_lin']/S
			rmse_test[1] += sample['pred_ridge']/S
			rmse_test[2] += sample['pred_svr_lin']/S

			r2_test.append(r2[n]/S)


		#store rmse values per ticker in final rmse table
		for i in range(S):
			rmse[i].append(rmse_test[i])
			r2_values[i].append(r2_test[i])

		#graphs residual fit plot of final sample for each ticker
		residual_fit_plot(residual)

	r2_linear = sum(r2_values[0])/len(r2_values)
	r2_ridge = sum(r2_values[1])/len(r2_values)
	r2_svr = sum(r2_values[2])/len(r2_values)

		
	print(rmse)
	print(r2_linear,r2_ridge,r2_svr)
 

def residual_fit_plot(residual):

	plt.plot(residual[3],residual[0],'ro')
	plt.show()

if __name__=="__main__":
	metric_evaluation()
