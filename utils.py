

import csv
import matplotlib.pyplot as plt

def log_result(episode, agent_id, total_reward, status, epsilon):
   
    with open("training_log_multi.csv", "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([episode, agent_id, total_reward, status, epsilon])

def plot_rewards():
   
    episode_data = {} 

    try:
        with open("training_log_multi.csv") as f:
            r = csv.reader(f)
            
          
            next(r, None) 
            
            for row in r:
                try:
                    
                    episode = int(row[0])
                    reward = float(row[2])

                    if episode not in episode_data:
                        
                        episode_data[episode] = [0.0, 0] 

                    
                    episode_data[episode][0] += reward
                    episode_data[episode][1] += 1
                    
                except ValueError:
                   
                    continue 

    except FileNotFoundError:
        print("Hata: training_log_multi.csv dosyası bulunamadı.")
        return
        
    
    eps = sorted(episode_data.keys())
    

    rewards_avg = [episode_data[e][0] / episode_data[e][1] for e in eps] 

    plt.figure(figsize=(10, 6))
    plt.plot(eps, rewards_avg)
    plt.title("Average Rewards / Episodes (All Agents)")
    plt.xlabel("Episode")
    plt.ylabel("Average Total Award")
    plt.grid(True)
    
   
    plt.savefig("rewards_plot.png") 
    plt.close()