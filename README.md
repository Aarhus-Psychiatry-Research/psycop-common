Current priorities are on the 
[Board 🎬](https://github.com/orgs/Aarhus-Psychiatry-Research/projects/4/views/1).

psycop-t2d
==============================
![python versions](https://img.shields.io/badge/Python-%3E=3.7-blue)
[![Code style: black](https://img.shields.io/badge/Code%20Style-Black-black)](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html)

Prediction of type 2 diabetes among patients with visits to psychiatric hospital departments.

## Installing pre-commit hooks
`pre-commit install`

--------
## Testing configs
To run XGBoost with defaults but a synth dataset:

```
python src/psycopt2d/train_model.py --config-name test_config.yaml +model=xgboost
```

To test new integrations with WandB:
```python
python src/psycopt2d/train_model.py +model=xgboost project.wandb_model="run" --config-name integration_testing.yaml
```



## Train models:
To run XGBoost with defaults:

```
python src/psycopt2d/train_model.py +model=xgboost
```

if you want to change a hyperparameter simply run:

```
python src/psycopt2d/train_model.py  +model=xgboost ++model.args.n_estimators=20
```

to run a sweep with xgboost you will have to add the `--multirun` flag and specify the sweep config.
```
python src/psycopt2d/train_model.py --multirun +model=xgboost
```

If training with WandB in offline mode (e.g. if doing hyperparameter search), make sure to call the `sync_best_runs_to_wandb.py` script. 
This will upload the `top_n` best models to WandB, move the other runs to an archive folder, and delete the `.aucs/auc.csv` file which is used to track the best performing models.
```
python src/sync_best_runs_to_wandb.py --top_n 10 --project psycopt2d
```

## Logging Altair to WandB and saving as png
We use Selenium and chromedriver to save Altair charts as png. 

This works out-of-the-box on Overtaci.

Locally it requires you to:

### Install chromedriver
1. Download [chromedriver](https://chromedriver.chromium.org) 
2. Place it on PATH (e.g. `/usr/local/bin` on OSX) 
3. Remove from quarantine: `cd /usr/local/bin && xattr -d com.apple.quarantine chromedriver`

For more description, [see this guide](https://www.swtestacademy.com/install-chrome-driver-on-mac/).

### Install vega for png output
1. Install npm, `brew install npm`
2. Install the required packages, `npm install vega-lite vega-cli canvas`

More on the altair_saver [readme](https://github.com/altair-viz/altair_saver#selenium) 



Minimal example of logging Altair chart to WandB
```py
run = wandb.init()

source = pd.DataFrame(
    {
        "a": ["A", "B", "C", "D", "E", "F", "G", "H", "I"],
        "b": [28, 55, 43, 91, 81, 53, 19, 87, 52],
    }
)

chart = alt.Chart(source).mark_bar().encode(x="a", y="b")

tmp_filename = "chart.png"
chart.save(tmp_filename)

run.log({"dummy_chart" : wandb.Image(tmp_filename)})
```




<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
