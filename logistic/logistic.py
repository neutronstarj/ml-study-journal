"""
find the manx value of function by iteration 
f(x) = -x^2 +4x
"""

def Gradient_Ascent_test():
    def f_prime(x_old):#f(x) 's derivative
        return -2*x_old+4
    x_old= -1
    #the starting value, given is smaller than x_new
    x_new=0
    alpha = 0.01  #incresing rate
    presision = 0.00000001
    while abs(x_new-x_old)>presision:
        x_old= x_new 
        x_new= x_old+alpha*f_prime(x_old)
    print(x_new)



if __name__ =="__main__":
    Gradient_Ascent_test()
