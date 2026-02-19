# Code Explanation Documentation Index

## Complete Documentation for MARL Jammer Project

This folder contains comprehensive documentation explaining every aspect of the codebase.

---

## 📚 Documentation Files

| #   | File                                                         | Description                               |
| --- | ------------------------------------------------------------ | ----------------------------------------- |
| 1   | [01_THEORY_TO_CODE_MAPPING.md](01_THEORY_TO_CODE_MAPPING.md) | Line-by-line mapping of equations to code |
| 2   | [02_STEP_BY_STEP_DEMO.md](02_STEP_BY_STEP_DEMO.md)           | What to run in order for demo             |
| 3   | [03_CONCEPTS_GLOSSARY.md](03_CONCEPTS_GLOSSARY.md)           | RL terms, variables, definitions          |
| 4   | [04_DEPLOYMENT_GUIDE.md](04_DEPLOYMENT_GUIDE.md)             | Where weights are, deployment process     |
| 5   | [05_SCALABILITY.md](05_SCALABILITY.md)                       | Scaling to 100 enemies, 40 jammers        |
| 6   | [06_PROFESSOR_PRESENTATION.md](06_PROFESSOR_PRESENTATION.md) | Q&A prep, future work, talking points     |

---

## 📖 Reading Order

**For Understanding the Code:**

```
1. 03_CONCEPTS_GLOSSARY.md    → Learn the terminology
2. 01_THEORY_TO_CODE_MAPPING.md → See how theory becomes code
3. 04_DEPLOYMENT_GUIDE.md     → Understand output structure
```

**For Preparing the Demo:**

```
1. 02_STEP_BY_STEP_DEMO.md    → What commands to run
2. 06_PROFESSOR_PRESENTATION.md → What to say
```

**For Future Work Planning:**

```
1. 05_SCALABILITY.md          → Technical scaling considerations
2. 06_PROFESSOR_PRESENTATION.md → Research directions
```

---

## 🔗 Quick Reference

### Key Equations → Code

| Equation                  | Code Location                                |
| ------------------------- | -------------------------------------------- |
| FSPL: P_R = P_tx(c/4πfd)² | `src/physics/fspl.py:165-200`                |
| Laplacian: L = D - A      | `src/physics/communication_graph.py:177-195` |
| Lambda-2: 2nd eigenvalue  | `src/physics/communication_graph.py:220-280` |
| 5-term reward             | `src/environment/reward.py:70-110`           |
| PPO clipping              | `src/agents/ppo_agent.py:270-340`            |

### Key Files

| File                                        | Purpose              |
| ------------------------------------------- | -------------------- |
| `src/config.py`                             | All hyperparameters  |
| `src/train.py`                              | Training entry point |
| `src/evaluate.py`                           | Evaluation script    |
| `outputs/experiment_v1/actor_state_dict.pt` | Trained weights      |

### Current Best Results

| Metric         | Value   |
| -------------- | ------- |
| Mean Reward    | 82.4    |
| λ₂ Reduction   | 36.3%   |
| Training Steps | 200K    |
| Training Time  | ~20 min |

---

## 📁 Project Structure

```
MARL JAMMER/
├── docs/
│   ├── PROJECT_MASTER_GUIDE_v2.md  # Original specification
│   └── code_explanation/           # THIS FOLDER
│       ├── README.md               # This index
│       ├── 01_THEORY_TO_CODE_MAPPING.md
│       ├── 02_STEP_BY_STEP_DEMO.md
│       ├── 03_CONCEPTS_GLOSSARY.md
│       ├── 04_DEPLOYMENT_GUIDE.md
│       ├── 05_SCALABILITY.md
│       └── 06_PROFESSOR_PRESENTATION.md
├── src/
│   ├── physics/     # FSPL, graph, jamming
│   ├── environment/ # Gym env, reward
│   ├── agents/      # Actor, Critic, PPO
│   ├── config.py
│   ├── train.py
│   └── evaluate.py
└── outputs/
    └── experiment_v1/
        ├── actor_state_dict.pt
        ├── training_log.csv
        └── graphs/
```

---

## 🎓 For the Professor

This project demonstrates:

1. **Physics-grounded RL** - Real FSPL model, not toy rewards
2. **Graph-theoretic objective** - λ₂ is mathematically meaningful
3. **MARL with parameter sharing** - Scalable architecture
4. **Industry-standard training** - PPO with all modern tricks
5. **Clean code** - Modular, documented, reproducible

**Result:** 36% connectivity reduction achieved autonomously.

---

_Generated with ❤️ for academic clarity._
