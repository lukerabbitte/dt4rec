import torch
import matplotlib.pyplot as plt

att = torch.load('/home/luke/code/minGPTrecommendations/attention_dumps/125/att_287000.pt')

avg_att = att.mean(dim=1)

print(avg_att.shape)

plt.figure(figsize=(12, 8))

plt.imshow(avg_att[0].cpu().detach().numpy(), cmap='viridis') 
plt.colorbar()

plt.title('Mean Attention Weights Across 8 Heads (After Training)', fontdict={'family': 'monospace'}, fontweight='bold', fontsize=16)

plt.ylabel('Context Length: 30 tokens\nBlock Size: 90 tokens', fontsize=12)

plt.savefig('/home/luke/code/minGPTrecommendations/figs/attention/125/287000_avg.svg')
