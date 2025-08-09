# GMM Estimation for the CKLS Model
##############################################################################################
# This module provides functions to simulate the CKLS model and 
# estimate its parameters using the Generalized Method of Moments (GMM).

# For more details, refer to the CKLS model paper:
# CHAN, K.C., KAROLYI, G.A., LONGSTAFF, F.A. and SANDERS, A.B. (1992),
# An Empirical Comparison of Alternative Models of the Short-Term Interest Rate. 
# The Journal of Finance, 47: 1209-1227. https://doi.org/10.1111/j.1540-6261.1992.tb04011.x
##############################################################################################

import numpy as np
from scipy.optimize import root
from scipy.optimize import minimize
# for residual analysis :
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from scipy.stats import probplot, jarque_bera
import scipy.stats as stats

from sklearn.metrics import mean_squared_error


# Simulation :
def generate_rt(theta, r0=0.03, T=500):
    '''Simulate the CKLS model using the given parameters.

    Args:
        theta (array): Parameters of the CKLS model.
        r0 (float, optional): Initial interest rate. Defaults to 0.03.
        T (int, optional): Number of periods to simulate. Defaults to 500.

    Returns:
        array: Simulated interest rates.
    '''
    
    alpha, beta, sigma2, gamma = theta
    sigma = np.sqrt(max(sigma2, 1e-8))
    r = np.zeros(T+1)
    r[0] = max(r0, 1e-6)  # enforce initial positivity

    for t in range(T):
        epsilon = np.random.normal(0, 1)
        rt_safe = max(r[t], 1e-6)  # enforce positivity
        r[t+1] = r[t] + alpha + beta * r[t] + sigma * rt_safe**gamma * epsilon
        r[t+1] = max(r[t+1], 1e-6)  # ensure positivity again after step

    return r


# moments conditions of the the original paper :
def moment_conditions_original(theta, r):
    '''Calculate the moment conditions for the CKLS model.
    Args:
        theta (array): Parameters of the CKLS model.
        r (array): Simulated interest rates.
    Returns:
        array: Moment conditions.
    '''
    alpha, beta, sigma2, gamma = theta

    rt = r[:-1]
    rt1 = r[1:]

    rt_safe = np.where(rt > 1e-6, rt, 1e-6)
    
    eps = rt1 - rt - alpha - beta * rt

    m1 = np.mean(eps)
    m2 = np.mean(eps * rt)
    m3 = np.mean(eps**2 - sigma2 * rt_safe**(2 * gamma))
    m4 = np.mean((eps**2 - sigma2 * rt_safe**(2 * gamma)) * rt)

    return np.array([m1, m2, m3, m4])

# GMM estimation function, baed on the moment conditions of the original paper :
def estimate_gmm(r, theta_init):
    '''Estimate the parameters of the CKLS model using GMM.
    Args:
        r (array): Simulated interest rates.
        theta_init (array): Initial guess for the parameters.
        Returns:
        array: Estimated parameters.'''
    def objective(theta):
        return moment_conditions_original(theta, r)

    sol = root(objective, theta_init, method='hybr')

    if sol.success:
        return sol.x
    else:
        raise RuntimeError("Root finding did not converge: " + sol.message)
#######################################################################################

# Gmm estimation function, based on the minimisation equivalence.

# def moment_conditions_overidentified(theta, r):
#     '''Calculate the extended moment conditions for the CKLS model.
#     Args:
#         theta (array): Parameters of the CKLS model.
#         r (array): Simulated interest rates.
#     Returns:
#         array: Extended moment conditions.
#     '''
#     alpha, beta, sigma2, gamma = theta
#     rt = r[:-1]
#     rt1 = r[1:]

#     rt_safe = np.where(rt > 1e-6, rt, 1e-6)
#     eps = rt1 - rt - alpha - beta * rt
#     # Extended moment conditions
#     m1 = np.mean(eps)
#     m2 = np.mean(eps * rt)
#     m3 = np.mean(eps * rt**2)
#     m4 = np.mean(eps**2 - sigma2 * rt_safe**(2 * gamma))
#     m5 = np.mean((eps**2 - sigma2 * rt_safe**(2 * gamma)) * rt)
#     m6 = np.mean((eps**2 - sigma2 * rt_safe**(2 * gamma)) * rt**2)

#     return np.array([m1, m2, m3, m4, m5, m6])


def gmm_objective(theta, r, W):
    '''Objective function for GMM estimation.
    Args:
        theta (array): Parameters of the CKLS model.
        r (array): Simulated interest rates.
        W (array): Weight matrix.
    Returns:
        float: Objective value.
    '''
    g = moment_conditions_original(theta, r)
    # print(g.shape)
    return g.T @ W @ g

def compute_optimal_weight_matrix(r, theta_hat):
    '''Compute the optimal weight matrix for GMM estimation.
    Args:
        r (array): Simulated interest rates.
        theta_hat (array): Estimated parameters.
    Returns:
        array: Optimal weight matrix.
    '''
    g_values = []
    T = len(r) - 1
    for t in range(T):
        rt = r[t]
        rt1 = r[t+1]
        rt_safe = max(rt, 1e-6)
        eps = rt1 - rt - theta_hat[0] - theta_hat[1] * rt
        sigma2 = theta_hat[2]
        gamma = theta_hat[3]
        moment = np.array([
            eps,
            eps * rt,
            eps**2 - sigma2 * rt_safe**(2 * gamma),
            (eps**2 - sigma2 * rt_safe**(2 * gamma)) * rt
        ])
        g_values.append(moment)
    g_matrix = np.vstack(g_values)
    return np.linalg.inv(np.cov(g_matrix, rowvar=False))

# Two-step GMM estimation
def estimate_two_step_gmm(r, theta_init):
    '''Estimate the parameters of the CKLS model using two-step GMM.
    Args:
        r (array): Simulated interest rates.
        theta_init (array): Initial guess for the parameters.
    Returns:
        array: Estimated parameters.
    '''
    W1 = np.eye(4)
    # print(W1.shape)
    # print(f'r shape: {r.shape}')
    result1 = minimize(gmm_objective, theta_init, args=(r, W1), method='BFGS')
    theta1 = result1.x
    W_opt = compute_optimal_weight_matrix(r, theta1)
    result2 = minimize(gmm_objective, theta1, args=(r, W_opt), method='BFGS')
    return result1, result2, W_opt


###############################################################################
# In this section, we study the residuals of the GMM estimation.

def compute_residuals(r, theta):
    '''Compute the residuals of the CKLS model.
    Args:
        r (array): Simulated interest rates.
        theta (array): Parameters of the CKLS model.
    Returns:
        array: Residuals.
    '''
    alpha, beta, sigma, gamma = theta
    rt = r[:-1]
    rt1 = r[1:]
    with np.errstate(divide='ignore', invalid='ignore'):
        eps = (rt1 - rt - alpha - beta * rt) / (sigma * np.power(rt, gamma))
    return eps


def plot_residual_diagnostics(residuals, estimator_names, r_sample=None):
    '''Plot residual diagnostics for one or more CKLS model residuals and perform 4 statistical tests.
    Args:
        residuals (array or list of arrays): Residuals.
        estimator_names (str or list of str): Names of the estimators.
        r_sample (array): Simulated interest rates (optional).
    '''
    # Normalize inputs: make sure residuals and names are lists
    if not isinstance(residuals, list):
        residuals = [residuals]
    if not isinstance(estimator_names, list):
        estimator_names = [estimator_names]

    n = len(residuals)
    fig, axs = plt.subplots(n, 4, figsize=(20, 5 * n))
    if n == 1:
        axs = [axs]  # Make axs 2D-compatible

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

    for i, res in enumerate(residuals):
        name = estimator_names[i]
        color = colors[i % len(colors)]

        # 1. Time Series
        axs[i][0].plot(res, color=color)
        axs[i][0].set_title(f'{name} Residuals Time Series')

        # 2. Histogram + KDE
        sns.histplot(res, color=color, kde=True, ax=axs[i][1])
        axs[i][1].set_title(f'{name} Histogram + KDE')

        # 3. Q-Q Plot
        probplot(res, dist="norm", plot=axs[i][2])
        axs[i][2].set_title(f'{name} Q-Q Plot')

        # 4. ACF
        sm.graphics.tsa.plot_acf(res, ax=axs[i][3], color=color)
        axs[i][3].set_title(f'{name} ACF')

    plt.tight_layout()
    plt.show()

    print('''
| Test               | Null Hypothesis                      | If p < 0.05                   |
|--------------------|--------------------------------------|-------------------------------|
| Mean Zero (t-test) | Residuals have mean zero             | Bias exists (bad fit)         |
| Ljung-Box          | No autocorrelation in residuals      | Residuals are autocorrelated  |
| Breusch-Pagan      | Homoscedasticity (constant variance) | Residuals are heteroscedastic |
| Jarque-Bera        | Residuals are normally distributed   | Non-normal residuals          |
''')

    # Statistical Tests
    for i, res in enumerate(residuals):
        name = estimator_names[i]
        print(f'\n{name} Residuals:')

        # Mean Zero Test
        t_stat, p_val = stats.ttest_1samp(res, popmean=0)
        print(f"\nMean Zero Test: t = {t_stat:.3f}, p = {p_val:.3f} → {'Mean zero' if p_val > 0.05 else 'Mean not zero'}")
        
        # Ljung-Box Test
        lb_test = acorr_ljungbox(res, lags=10, return_df=True)
        lb_stat = lb_test['lb_stat'].values[0]
        lb_p = lb_test['lb_pvalue'].values[0]
        print(f"[Ljung-Box] Q(10) = {lb_stat:.3f}, p = {lb_p:.3f} → {'No autocorr' if lb_p > 0.05 else 'Autocorr detected'}")

        # Breusch-Pagan Test
        print(f'\nBreusch-Pagan Test:')
        if r_sample is not None:
            X = sm.add_constant(r_sample[:len(res)])
            bp_test = het_breuschpagan(res, X)
            print(f"[Breusch–Pagan] p = {bp_test[1]:.3f} → {'Homoskedastic' if bp_test[1] > 0.05 else 'Heteroskedasticity detected'}")
        else:
            print("Skipped (r_sample is None)")

        # Jarque-Bera Test
        stat, pval = jarque_bera(res)
        print(f"\nJarque-Bera Normality Test: JB stat = {stat:.3f}, p = {pval:.3f} → {'Normal' if pval > 0.05 else 'Non-normal'}")
        print("\n" + "="*60)


#####################################################################
# for real data we use the following function :

def rate_prediction(rate_data,train_portion = 0.8, theta_init = np.array([0.01, 0.01, 0.01, 0.1])):
    import warnings
    warnings.filterwarnings("ignore")
    
    size = rate_data.shape[0]
    print("Size of the data: ", size)
        
    date_column_name = rate_data.columns[0]
    rate_column_name = rate_data.columns[1]
    
    rate_data[rate_column_name].fillna(method='ffill', inplace=True)
    
    rate = rate_data[rate_column_name].to_numpy()
    
    train_size = int(len(rate) * train_portion)
        
    r_train = rate[:train_size]
    r_test = rate[train_size:]
    
    plt.figure(figsize=(10, 5))
    plt.plot(rate, label='Treasury Rate', color='blue')
    plt.title('Treasury Rate Over Time')
    plt.xlabel('Time (Index)')
    plt.ylabel('Interest Rate (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
    _,theta_estimated_updated_matrix,_ = estimate_two_step_gmm(r_train, theta_init)
    theta_estimated_1 = theta_estimated_updated_matrix.x
    print('method: estimate_two_step_gmm')
    print("Estimated parameters:", theta_estimated_1)
    
    
    alpha_hat = theta_estimated_1[0]
    beta_hat = theta_estimated_1[1]
    sigma_hat = theta_estimated_1[2]
    gamma_hat = theta_estimated_1[3]
    
    residuals_test = compute_residuals(r_train[1:], theta_estimated_1)
    plot_residual_diagnostics(residuals_test, 'Estimate two step GMM', rate[1:])
    
    # Backtesting
    r_test_pred = np.zeros(len(r_test))
    r_test_pred = r_test[:-1] + alpha_hat + beta_hat * r_test[:-1]

    rmse = mean_squared_error(r_test[1:], r_test_pred)
    print(f"Backtesting Out-of-sample RMSE: {rmse:.4f}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(r_test[1:], label='Actual')
    plt.plot(r_test_pred, label='Predicted', linestyle='--')
    plt.legend()
    plt.title("Out-of-Sample Forecast vs Actual")
    plt.show()
    
    # # volatility analysis :
    # actual_vol = np.abs(np.diff(r_test))
    # actual_vol = actual_vol * np.sqrt(12)
    # # Forecasted volatility
    # forecasted_vol = sigma_hat * r_test[:-1]**gamma_hat
    # forecasted_vol= forecasted_vol * np.sqrt(12)
    
    # plt.plot(actual_vol, '--', label='Actual')
    # plt.plot(forecasted_vol, '-', label='Forecast')
    # plt.ylabel("Interest Rate Volatility (% per month)")
    # plt.xlabel("Time")
    # plt.legend()
    # plt.title("Forecast vs Actual Volatility (CKLS Model)")
    # plt.show()
    
    theta_estimated_2 = estimate_gmm(r_train, theta_init)
    print('method: estimate_gmm')
    print("Estimated parameters:", theta_estimated_2)
    
    alpha_hat = theta_estimated_2[0]
    beta_hat = theta_estimated_2[1]
    sigma_hat = theta_estimated_2[2]
    gamma_hat = theta_estimated_2[3]
    residuals_test = compute_residuals(r_train[1:], theta_estimated_2)
    plot_residual_diagnostics(residuals_test, 'Estimate GMM', rate[1:])
    
    # Backtesting
    r_test_pred = np.zeros(len(r_test))
    r_test_pred = r_test[:-1] + alpha_hat + beta_hat * r_test[:-1]

    rmse = mean_squared_error(r_test[1:], r_test_pred)
    print(f"Backtesting Out-of-sample RMSE: {rmse:.4f}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(r_test[1:], label='Actual')
    plt.plot(r_test_pred, label='Predicted', linestyle='--')
    plt.legend()
    plt.title("Out-of-Sample Forecast vs Actual")
    plt.show()
    
    # # volatility analysis :
    # # Actual volatility
    # actual_vol = np.abs(np.diff(r_test))
    # actual_vol = actual_vol * np.sqrt(12)
    # # Forecasted volatility
    # forecasted_vol = sigma_hat * r_test[:-1]**gamma_hat
    # forecasted_vol= forecasted_vol * np.sqrt(12)
    # plt.plot(actual_vol, '--', label='Actual')
    # plt.plot(forecasted_vol, '-', label='Forecast')
    # plt.ylabel("Interest Rate Volatility (% per month)")
    # plt.xlabel("Time")
    # plt.legend()
    # plt.title("Forecast vs Actual Volatility (CKLS Model)")
    # plt.show()