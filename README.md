# Installation and usage

```bash
conda env create -f environment.yml

conda activate env_data_preparation

conda env export > environment.yml
```

🧩 OPTION 1 — Using Anaconda Navigator (no command line)

1.	Open “Anaconda Navigator”
2.	Go to the “Environments” tab on the left.
3.	Click “Create” (bottom left corner).
•	Name it: env_data_preparation
•	Python version: 3.12 (recommended)
4.	After the environment is created, click it -> select “Open Terminal” or go to “Channels -> conda-forge” (if available).
5.	In the “Search Packages” field, type pandas, tick it, and click Apply.
Do the same for matplotlib.
6.	When done, go back to the Home tab → pick your new env env_data_preparation in the dropdown → launch Spyder or VS Code from there.

