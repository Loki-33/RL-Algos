import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time


class AC(nn.Module):
    def __init__(self, n_obs, n_actions):
        super(AC, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, 128),
            nn.ReLU(),
        )
        self.actor = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.net(x)
        return self.actor(x), self.critic(x).squeeze(-1)

def compute_returns(rewards, gamma):
    returns = torch.zeros(len(rewards))
    R = 0
    for r in reversed(range(len(rewards))):
        R = rewards[r] + gamma * R
        returns[r] = R
    return returns

def worker(global_model, env_name, optimizer, gamma, worker_id):
    env = gym.make(env_name) 
    local_model = AC(env.observation_space.shape[0], env.action_space.n)
    local_model.load_state_dict(global_model.state_dict())
    
    for ep in range(1000):
        state,_ = env.reset()
        
        done = False
        rewards = []
        values = []
        log_probs = []
        entropies = []
        while not done:
            state_tensor = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0)
            logits, value = local_model(state_tensor)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            next_state, reward, done, truncated, _= env.step(action.item())
            
            done = done or truncated  

            entropies.append(entropy)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            state = next_state
        
        
        returns = compute_returns(rewards, gamma)
        values = torch.stack(values)
        log_probs = torch.stack(log_probs)
        
        advantage = returns - values.detach()
        
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        actor_loss = -(log_probs * advantage).mean()
        critic_loss = advantage.pow(2).mean()
        entropy =torch.stack(entropies)
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()
        
        
        optimizer.zero_grad()
        loss.backward()
        
        
        for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
            if local_param.grad is not None:
                if global_param.grad is None:
                    global_param.grad = local_param.grad.clone()
                else:
                    global_param.grad += local_param.grad.clone()

        
        optimizer.step()
        local_model.load_state_dict(global_model.state_dict())
        
        if ep % 50 == 0:
            print(f"Worker {worker_id}, Episode {ep}, Total Reward: {sum(rewards)}")

def train():
    env_name = 'CartPole-v1'
    n_workers = 4
    gamma = 0.99
    
    env = gym.make(env_name)
    global_model = AC(env.observation_space.shape[0], env.action_space.n)
    global_model.share_memory()
    optimizer = optim.Adam(global_model.parameters(), lr=1e-3)
    
    processes = []
    for worker_id in range(n_workers):
        p = mp.Process(target=worker, args=(global_model, env_name, optimizer, gamma, worker_id))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    torch.save(global_model.state_dict(), "a3c_model.pth")

if __name__ == '__main__':
    mp.set_start_method('spawn')
    
    # Train the model
    train()
    
