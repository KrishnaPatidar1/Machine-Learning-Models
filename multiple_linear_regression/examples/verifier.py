from sklearn.linear_model import LinearRegression

class Verifier:
    ''' This class is made to calculate the actual weights and bias'''

    @staticmethod
    def calculate_solution(X, Y):
        '''This function takes in scaled training set(X) and then calculates the
           correct weights and bias using sklearn
        '''
        model = LinearRegression()
        model.fit(X, Y)

        return (model.coef_, model.intercept_)

# end of class Verifier