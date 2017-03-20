import quandl 
import pandas
import math
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

quandl.ApiConfig.api_key = '9yy6u5iWsMa4qvS9x48Q'

def retrieve_data(ticker):

	start="2016-03-01"
	end="2017-03-17"
	db_code="WIKI"
	ticker = ticker
	ip_code = db_code+'/'+ticker
	data = quandl.get(ip_code, start_date=start, end_date=end, collapse="daily")

	return(data) 

def define_features(data):

	d = [[] for i in range(0,6)]
	for number in range(0,6):
		for i in range(0,len(data)):
			if i<5:
				d[number].append(-float('inf'))
			else:
				d[number].append(data['Adj. Close'][i-5+number])

	#adding feature values in data list
	data['D1'] = d[0]
	data['D2'] = d[1]
	data['D3'] = d[2]
	data['D4'] = d[3]
	data['D5'] = d[4]

	data = data[['D1','D2','D3','D4','D5','Adj. Close']]

	return(data)

def regression_models(data):
	X = data.drop('Adj. Close',axis=1)[5:]
	y = data['Adj. Close'][5:]
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)								

	linear_model = LinearRegression()
	linear_model.fit(X_train,y_train)
	pred_lin = linear_model.predict(X_test)						
	svr_linear = SVR(kernel='linear')
	pred_svr_lin = svr_linear.fit(X_train,y_train).predict(X_test)
	# svr_poly = SVR(kernel='poly', degree=2)
	# pred_svr_poly = svr_poly.fit(X_train,y_train).predict(X_test)	

	ridge = Ridge()
	pred_ridge = ridge.fit(X_train,y_train).predict(X_test)


	#predicted_df = pandas.DataFrame(zip(pred_lin,pred_svr_poly,pred_svr_lin), columns = ['pred_lin','pred_svr_poly','pred_svr_lin'])
	data_final = pandas.DataFrame(zip(y_test,pred_lin,pred_ridge,pred_svr_lin), columns = ['actual','pred_lin','pred_ridge','pred_svr_lin'])
	return(data_final,y_test)

def metrics(y_test,data):

	accuracy = []
	r2 = []
	pred_lin = data['pred_lin']
	pred_ridge = data['pred_ridge']
	pred_svr_lin = data['pred_svr_lin']

	accuracy.append((mean_squared_error(y_test,pred_lin))**0.5)
	accuracy.append((mean_squared_error(y_test,pred_ridge))**0.5)
	accuracy.append((mean_squared_error(y_test,pred_svr_lin))**0.5)

	rmse_scores = dict(zip(['pred_lin','pred_ridge','pred_svr_lin'],accuracy[::1]))

	r2.append(r2_score(y_test,pred_lin))
	r2.append(r2_score(y_test,pred_ridge))
	r2.append(r2_score(y_test,pred_svr_lin))


	return rmse_scores,r2

def residual_fit(y_test,data):

	residual = [[] for i in range(4)]

	for i in range(len(y_test)):
		residual[0].append(abs(y_test[i]-data['pred_lin'][i]))
		residual[1].append(abs(y_test[i]-data['pred_ridge'][i]))
		residual[2].append(abs(y_test[i]-data['pred_svr_lin'][i]))
		residual[3].append(y_test[i])

	return residual

def main(ticker):
	ticker = ticker
	data = retrieve_data(ticker)
	input_data = define_features(data)
	final_data,y_test = regression_models(input_data)
	scores,r2 = metrics(y_test,final_data)
	residual = residual_fit(y_test,final_data)

	return scores,r2,residual

if __name__ == "__main__":
	main()