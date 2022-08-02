import pandas as pd
import ast
import math

def quaternion_to_euler_angle(x, y, z, w):
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = math.degrees(math.atan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = math.degrees(math.atan2(t3, t4))

    return X, Y, Z


df = pd.read_csv("../../data/pid_batt_d0.csv")
droplist = []
print(df)
for index, row in df.iterrows():
    a = ast.literal_eval(df.iloc[index, 12])
    b = ast.literal_eval(df.iloc[index, 13])
    df.at[index, "l_pressure"] = a[0:3]
    df.at[index, "r_pressure"] = b[1:4]
    if sum(a[0:3]) + sum(b[1:4]) < 15.0:
        print("Pressure is " +  str(sum(a[0:3]) + sum(b[1:4])) + "Dropping row.")
        droplist.append(index)
    imu = ast.literal_eval(df.iloc[index, 11])
    e = list(quaternion_to_euler_angle(imu[0], imu[1], imu[2], imu[3]))
    e.extend(imu[4:])
    df.at[index, "imu"] = e 
df = df.drop(droplist)      
df.to_csv("../../data/pid_batt_d0c.csv",index=False)
print(df)
