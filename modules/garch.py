from arch import arch_model

def apply_garch(prices):
    model = arch_model(prices, vol='Garch', p=1, q=1)
    model_fit = model.fit(disp="off")
    return model_fit.conditional_volatility
