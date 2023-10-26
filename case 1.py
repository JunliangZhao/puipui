import numpy as np
import random
from docplex.mp.model import Model
import time
###################################################################################################################
# create scenario network
num_shop = 1
num_airport = 1
num_warehouse = 1
num_node = num_shop + num_airport + num_warehouse
set_airport = list(range(num_airport))
set_shop = list(range(num_airport, num_airport+num_shop))
set_warehouse = list(range(num_airport+num_shop, num_airport+num_shop+num_warehouse))
set_node = set_airport + set_shop + set_warehouse
print('Airport:', set_airport)
print('Engine shop:', set_shop)
print('Warehouse:', set_warehouse)
print('test:', set_airport+set_warehouse)
# create time
num_time = 5
maintenance_time = 1
trans_time = np.array([[0, 1, 2],
                       [1, 0, 2],
                       [2, 2, 0]])
max_trans_time = trans_time.max() + maintenance_time
adv_time = list(range(max_trans_time))
set_time = list(range(max_trans_time, max_trans_time+num_time))
total_time = list(range(max_trans_time+num_time))
print(adv_time)
print(set_time)
print(total_time)
# other parameters, 1 airport,1 engine shop,1 offsite warehouse
trans_cost = np.array([[0, 50, 300],
                       [50, 0, 300],
                       [300, 300, 0]])
print('transcost:', trans_cost[1, 0]+1)
hold_cost = [10000, 10000, 3]
penalty_rate = 5000
inven_capacity = [0, 0, 20]
# demand
d = np.zeros((max_trans_time+num_time, num_node))
for k in adv_time:
    for r in set_node:
        d[k, r] = 0
for k in set_time:
    for r in set_airport:
        d[k, r] = 1
    for r in set_shop+set_warehouse:
        d[k, r] = 0
print(d)
# maintenance capacity of engine shop
m_capacity = np.zeros((max_trans_time+num_time, num_node))
for k in adv_time:
    for r in set_node:
        m_capacity[k, r] = 0
for k in set_time:
    for r in set_shop:
        m_capacity[k, r] = 10
    for r in set_airport+set_warehouse:
        m_capacity[k, r] = 0
print(m_capacity)

#################################################################################################################
# CPLEX Model
time_start = time.time()
mdl = Model(name="total_cost")
X = [(t, i, j) for t in total_time for i in set_node for j in set_node]
Y = [(t, i, j) for t in total_time for i in set_node for j in set_node]
Q = [(t, i) for t in total_time for i in set_node]
M = [(t, i) for t in total_time for i in set_node]
B = [(t, i) for t in total_time for i in set_airport]
# decision variables
trans_m = mdl.integer_var_dict(X, name='trans_m')
trans_h = mdl.integer_var_dict(Y, name='trans_h')
q = mdl.integer_var_dict(Q, name='q')
m = mdl.integer_var_dict(M, name='m')
b = mdl.integer_var_dict(B, name='b')

# constraints
mdl.add_constraints(b[t, i] == 0 for i in set_airport for t in adv_time)
mdl.add_constraints(m[t, i] == 0 for i in set_node for t in adv_time)
mdl.add_constraints(q[t, i] == 5 for i in set_warehouse for t in adv_time)
mdl.add_constraints(trans_m[t, i, j] == 0 for i in set_node for j in set_node for t in adv_time)
mdl.add_constraints(trans_h[t, i, j] == 0 for i in set_node for j in set_node for t in adv_time)

# Equation(2)
mdl.add_constraints(d[t, i] == mdl.sum(trans_m[t, i, j] for j in (set_shop + set_warehouse))
                    for i in set_airport for t in set_time)
# Equation(3)
mdl.add_constraints(b[t-1, i] + d[t, i] == mdl.sum(trans_h[t-trans_time[j, i], j, i] for j in (set_shop + set_warehouse))
                    + b[t, i] for i in set_airport for t in set_time)
# Equation(4)
mdl.add_constraints(q[t-1, i] + mdl.sum(trans_h[t-trans_time[j, i], j, i] for j in set_shop) == q[t, i]
                    + mdl.sum(trans_h[t, i, j] for j in set_airport) for i in set_warehouse for t in set_time)
# Equation(5)
mdl.add_constraints(m[t-1, i] + mdl.sum(trans_m[t-trans_time[j, i], j, i] for j in set_airport) == m[t, i]
                    + mdl.sum(trans_m[t, i, j] for j in set_shop) for i in set_warehouse for t in set_time)
# Equation(6)
mdl.add_constraints(m[t-1, i] + mdl.sum(trans_m[t-trans_time[j, i], j, i] for j in (set_airport+set_warehouse)) ==
                    m[t, i] + mdl.sum(trans_h[t, i, j] for j in (set_airport+set_warehouse))
                    for i in set_shop for t in set_time)
# Equation(7)
mdl.add_constraints(mdl.sum(trans_m[t-trans_time[i, j]-maintenance_time, j, i] for j in (set_airport+set_warehouse)) ==
                    mdl.sum(trans_h[t, i, j] for j in (set_airport+set_warehouse))
                    for i in set_shop for t in set_time)
# Equation(8)
mdl.add_constraints(trans_h[t, i, j]*(num_time-t-trans_time[i, j]) >= 0 for i in set_node for j in set_airport for t in set_time)
# Equation(9)
mdl.add_constraints(mdl.sum(trans_m[t, j, i] for j in (set_airport+set_warehouse)) <= m_capacity[t, i]
                    for i in set_shop for t in set_time)
# Equation(10)
mdl.add_constraints(q[t, i] + m[t, i] <= inven_capacity[i] for i in set_node for t in set_time)
# Equation(11)
mdl.add_constraints(0 <= trans_m[t, i, j] for i in set_node for j in set_node for t in set_time)
mdl.add_constraints(0 <= trans_h[t, i, j] for i in set_node for j in set_node for t in set_time)
mdl.add_constraints(0 <= m[t, i] for i in set_node for t in set_time)
mdl.add_constraints(0 <= q[t, i] for i in set_node for t in set_time)
mdl.add_constraints(0 <= b[t, i] for i in set_airport for t in set_time)

# 计算所有时间的运输成本总和，并添加为kpi
total_trans_cost = mdl.sum(trans_m[t, i, j]*trans_cost[i, j] + trans_h[t, j, i]*trans_cost[j, i]
                           for i in set_node for j in set_node for t in set_time)
mdl.add_kpi(total_trans_cost, "Total transportation cost")
# 计算所有时间的库存成本总和，并添加为kpi
total_hold_cost = mdl.sum((m[t, i] + q[t, i])*hold_cost[i] for i in set_node for t in set_time)
mdl.add_kpi(total_hold_cost, "Total holding cost")
# 计算所有时间的penalty总和，并添加为kpi
total_penalty = mdl.sum(b[t, i]*penalty_rate for i in set_airport for t in set_time)
mdl.add_kpi(total_penalty, "Total penalty")
# minimize 设置最小化目标函数
mdl.minimize(total_trans_cost+total_hold_cost+total_penalty)
time_solve = time.time()
solution = mdl.solve(log_output=True)
time_end = time.time()
running_time = round(time_end - time_solve, 2)
elapsed_time = round(time_end - time_start, 2)
if solution != None:
    print(solution, elapsed_time, running_time)
else:
    print('NA', elapsed_time, running_time)