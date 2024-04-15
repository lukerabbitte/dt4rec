import matplotlib.pyplot as plt
import os

all_ndcgs = [0.8920290396517051, 0.9571688280932877, 0.9503943754490124, 0.8690637541450841, 0.8425203618460528, 0.8779873431832844, 0.8749491681135776, 0.9133377192988742, 0.9574235381644002, 0.9077469645992271, 0.8649237078633162, 0.9133138714802549, 0.9388876247937893, 0.9275366307252084, 0.8387443574347943]

os.makedirs('../figs/ndcg', exist_ok=True)

plt.figure(figsize=(10, 6))
plt.plot(all_ndcgs)
plt.title('NDCG over time')
plt.xlabel('Time')
plt.ylabel('NDCG')
plt.grid(True)

plt.ylim(0)

plt.savefig('../figs/ndcg/ndcg_over_time.svg')
plt.close()