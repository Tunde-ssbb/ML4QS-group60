def arm_v_leg_acc_derivative_diff(data):
    sensors = ['acc_x', 'acc_y', 'acc_z']
    for sensor in sensors:
        leg = "leg_" + sensor + "_diff"
        arm = "arm_" + sensor + "_diff"

        data[sensor + "_derivative_diff"] = abs(data[leg] - data[arm])

    return data

    