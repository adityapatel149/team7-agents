# Team 7 Agents

This repository contains **reinforcement learning agents** developed by **Team 7** for the CS271: Reinforcement Learning course (San José State University).  
Agents here are trained and evaluated primarily on our custom Highway-Env–based environment, `Team7-v0`, and compared against agents running in another team’s (Team 8’s) custom environment.

---

## 🔗 Related Repositories (Custom Environments)

- **Team 7 Custom Environment (`Team7-v0`) – ours**  
  👉 https://github.com/adityapatel149/team7-custom-env  

- **Team 8 Custom Environment – other team**  
  👉 https://github.com/chiragr15/CS272_CustomNarrow_Env  

---

## 📂 Repository Structure

```text
team7-agents/
├── models/      # Saved models, checkpoints, and/or hyperparameter configs
├── plots/       # Training curves, evaluation plots, and other visualizations
├── scripts/     # Training, evaluation, and utility scripts for agents
├── tb_logs/     # TensorBoard logs from training runs
├── .gitignore   # Git ignore rules
└── README.md    # This file
```

### `models/`
Contains **trained agent weights**.

### `plots/`
Holds **visual outputs** such as learning curves, and violin plots.

### `scripts/`
Contains **training scripts**.

### `tb_logs/`
Contains **TensorBoard logs**.  
Run:
```bash
tensorboard --logdir tb_logs
```
---

## 👥 Authors

**Team 7 — CS271: Reinforcement Learning (San José State University)**  
- Aditya Patel  
- Karan Jain  
- Shareen Rodrigues  

Instructor: Genya Ishigaki

---

## 📄 License

This repository is intended for **academic and research use** as part of the CS271 course.
