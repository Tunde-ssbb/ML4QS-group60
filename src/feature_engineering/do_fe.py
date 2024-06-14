import src.feature_engineering.frequency as ffe
import src.feature_engineering.temporal as tfe
import src.feature_engineering.domain as dfe


"""
features: 
- freqs [0.2,0.3,0.4,0.6,0.8,0.9]
- freq diff
- freq std
- difference in acceleration diff
"""



def feature_engineer(data):
    features = list(data.columns)
    features.remove("time")

    data = ffe.fourier_per_session(data, 100)

    data = ffe.remove_frequencies(data, 2, except_freq=[0.2,0.3,0.4,0.6,0.8,0.9])


    data = tfe.calculate_window_difference(data, 3, features)
    data = tfe.calculate_window_std(data, 40, features)

    data = dfe.arm_v_leg_acc_derivative_diff(data)

    print(f"engineered columns: {data.columns}")

    return data

